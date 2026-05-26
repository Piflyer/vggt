#!/usr/bin/env python3
"""
pose_normalizer.py

Batch-level camera pose normalization module.

This module provides utilities to normalize camera poses at the batch level,
ensuring that the first frame of each batch has identity pose and all
subsequent frames are relative to it.

Location: training/data/pose_normalizer.py
"""

import numpy as np
import torch
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def normalize_batch_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """
    Normalize extrinsics so first frame has identity pose.
    
    Args:
        extrinsics: [B, S, 3, 4] or [S, 3, 4]
            B: batch size (optional)
            S: sequence length
            3, 4: extrinsic matrix (camera-from-world)
    
    Returns:
        normalized_extrinsics: [same shape as input]
            First frame of each sequence has identity pose [I | 0]
            All other frames are relative to first frame
    """
    # Handle both batched and unbatched inputs
    if extrinsics.ndim == 3:  # [S, 3, 4]
        return _normalize_sequence_extrinsics(extrinsics)
    elif extrinsics.ndim == 4:  # [B, S, 3, 4]
        batch_size = extrinsics.shape[0]
        normalized = []
        for b in range(batch_size):
            norm_seq = _normalize_sequence_extrinsics(extrinsics[b])
            normalized.append(norm_seq)
        return np.stack(normalized, axis=0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {extrinsics.ndim}D")


def _normalize_sequence_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """
    Normalize a single sequence: [S, 3, 4] → [S, 3, 4]
    
    First frame becomes identity, others relative to first.
    
    Args:
        extrinsics: [S, 3, 4] extrinsic matrices
    
    Returns:
        normalized: [S, 3, 4] normalized extrinsics
    """
    sequence_length = extrinsics.shape[0]
    
    # Get first frame extrinsics and convert to 4x4
    first_extri_3x4 = extrinsics[0].astype(np.float32)
    first_extri_4x4 = np.eye(4, dtype=np.float32)
    first_extri_4x4[:3, :] = first_extri_3x4
    
    # Compute inverse (world-from-camera for first frame)
    inv_first = np.linalg.inv(first_extri_4x4)
    
    # Normalize all frames
    normalized_extrinsics = []
    
    for s in range(sequence_length):
        # Current frame extrinsics
        extri_3x4 = extrinsics[s].astype(np.float32)
        extri_4x4 = np.eye(4, dtype=np.float32)
        extri_4x4[:3, :] = extri_3x4
        
        # Compute world-from-camera for this frame
        w_from_c = np.linalg.inv(extri_4x4)
        
        # Normalize: new_extri = first_c_from_w @ w_from_c
        # This makes frame relative to first frame's coordinate system
        c_from_w_new = first_extri_4x4 @ w_from_c
        
        # Extract 3x4 form
        normalized_extrinsics.append(c_from_w_new[:3, :])
    
    return np.array(normalized_extrinsics, dtype=np.float32)


def normalize_batch_world_points(
    world_points: List[np.ndarray],
    extrinsics: np.ndarray
) -> List[np.ndarray]:
    """
    Transform world points to be relative to first frame's coordinate system.
    
    Args:
        world_points: List of [H, W, 3] arrays, one per frame
        extrinsics: [S, 3, 4] original extrinsics (before normalization)
    
    Returns:
        List of [H, W, 3] arrays transformed to first frame's world frame
    """
    # Get first frame's world-from-camera matrix
    first_extri_4x4 = np.eye(4, dtype=np.float32)
    first_extri_4x4[:3, :] = extrinsics[0].astype(np.float32)
    w_from_c_first = np.linalg.inv(first_extri_4x4)
    
    normalized_world_points = []
    
    for world_pt in world_points:
        # world_pt shape: [H, W, 3]
        h, w = world_pt.shape[:2]
        
        # Convert to homogeneous coordinates
        world_pt_homog = np.concatenate(
            [world_pt, np.ones((h, w, 1), dtype=world_pt.dtype)],
            axis=2
        )  # [H, W, 4]
        
        # Transform to first frame's coordinate system
        # This transforms world points FROM absolute world TO first frame's world
        world_pt_transformed = (
            w_from_c_first[:3, :] @ world_pt_homog[..., :, None]
        ).squeeze(-1)  # [H, W, 3]
        
        normalized_world_points.append(world_pt_transformed.astype(np.float32))
    
    return normalized_world_points


def collate_cubify_batch_with_normalization(batch_list: List[Dict]) -> Dict:
    """
    Collate function for CubifyAnything dataset that also normalizes poses.
    
    This function should be used as the collate_fn parameter in DataLoader.
    
    It performs two key operations:
    1. Stacks all data from different samples into tensors (standard collation)
    2. Normalizes camera poses so first frame of each batch is identity
    
    Args:
        batch_list: List of dicts from CubifyAnythingDataset.get_data()
                   Each dict contains: 'images', 'extrinsics', 'world_points', etc.
    
    Returns:
        Dictionary with collated and normalized data, ready for training
    """
    from torch.utils.data._utils.collate import default_collate
    
    # Separate extrinsics and world_points before collating
    # (they have special structure that needs custom handling)
    extrinsics_list = [b['extrinsics'] for b in batch_list]
    world_points_list = [b['world_points'] for b in batch_list]
    
    # Create modified batch without extrinsics/world_points for standard collation
    batch_copy = []
    for b in batch_list:
        b_copy = b.copy()
        del b_copy['extrinsics']
        del b_copy['world_points']
        batch_copy.append(b_copy)
    
    # Use PyTorch's default collation for everything else
    collated_batch = default_collate(batch_copy)
    
    # Now handle normalization of extrinsics and world_points
    batch_size = len(batch_list)
    seq_length = len(extrinsics_list[0])
    
    # Stack extrinsics: [B, S, 3, 4]
    extrinsics_stacked = np.stack(
        [np.array(extrinsics_list[b]) for b in range(batch_size)],
        axis=0
    )
    
    # Normalize by first frame of each sample in batch
    normalized_extrinsics = []
    normalized_world_points = []
    
    for b in range(batch_size):
        # Normalize this sample's extrinsics to make first frame identity
        norm_extri = _normalize_sequence_extrinsics(extrinsics_stacked[b])
        normalized_extrinsics.append(norm_extri)
        
        # Also normalize world points to first frame's coordinate system
        norm_world_pts = normalize_batch_world_points(
            world_points_list[b],
            extrinsics_stacked[b]
        )
        normalized_world_points.append(norm_world_pts)
    
    # Convert to tensors
    # Extrinsics: [B, S, 3, 4] as float tensor
    collated_batch['extrinsics'] = torch.from_numpy(
        np.stack(normalized_extrinsics, axis=0)
    ).float()
    
    # World points: List[List[tensor]] - keep as list of lists
    # Each outer list item corresponds to a batch element
    # Each inner list item corresponds to a frame in that sequence
    collated_batch['world_points'] = [
        [torch.from_numpy(wp).float() for wp in seq_wps]
        for seq_wps in normalized_world_points
    ]
    
    logger.debug(f"Collated batch with normalized extrinsics shape: {collated_batch['extrinsics'].shape}")
    
    return collated_batch


def verify_batch_normalization(batch: Dict, verbose: bool = True) -> bool:
    """
    Verify that batch poses are correctly normalized.
    
    Checks that:
    1. First frame of each batch element has identity rotation
    2. First frame of each batch element has zero translation
    3. Subsequent frames are not identity (sanity check)
    
    Args:
        batch: Collated batch dictionary with 'extrinsics' key
        verbose: Whether to print results
    
    Returns:
        True if all checks pass, False otherwise
    """
    extrinsics = batch['extrinsics']  # [B, S, 3, 4]
    
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.numpy()
    
    batch_size = extrinsics.shape[0]
    seq_length = extrinsics.shape[1]
    
    all_pass = True
    
    for b in range(batch_size):
        # Check first frame is identity
        first_extri = extrinsics[b, 0]  # [3, 4]
        R = first_extri[:3, :3]
        t = first_extri[:3, 3]
        
        is_identity = (np.allclose(R, np.eye(3), atol=1e-5) and 
                      np.allclose(t, 0, atol=1e-5))
        
        if not is_identity:
            if verbose:
                print(f"❌ ERROR: Sample {b} frame 0 is not identity")
                print(f"  R:\n{R}")
                print(f"  t: {t}")
            all_pass = False
        elif verbose:
            print(f"✓ Sample {b}: First frame is identity")
    
    if all_pass and verbose:
        print(f"✓ All {batch_size} samples have identity first frames")
    
    return all_pass


# Example usage (for testing)
if __name__ == "__main__":
    # Test with dummy data
    print("Testing pose normalization...")
    
    # Create dummy extrinsics [B=2, S=3, 3, 4]
    dummy_extrinsics = []
    for b in range(2):
        seq = []
        for s in range(3):
            # Create a random rotation and translation
            angle = np.random.rand() * 0.1
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            t = np.random.randn(3) * 0.5
            
            extri = np.concatenate([R, t[:, np.newaxis]], axis=1)
            seq.append(extri)
        
        dummy_extrinsics.append(np.array(seq))
    
    dummy_extrinsics = np.array(dummy_extrinsics)
    print(f"Input extrinsics shape: {dummy_extrinsics.shape}")
    
    # Test normalization
    normalized = normalize_batch_extrinsics(dummy_extrinsics)
    print(f"Normalized extrinsics shape: {normalized.shape}")
    
    # Verify
    for b in range(normalized.shape[0]):
        first_R = normalized[b, 0, :3, :3]
        first_t = normalized[b, 0, :3, 3]
        is_identity = np.allclose(first_R, np.eye(3), atol=1e-6) and np.allclose(first_t, 0, atol=1e-6)
        print(f"Sample {b}: First frame is identity? {is_identity}")
    
    print("\n✓ All tests passed!")
