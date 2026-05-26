#!/usr/bin/env python3
"""
Visualization tool comparing ground truth vs. predicted camera poses.

This tool:
1. Loads a trained VGGT model
2. Feeds CubifyAnything dataset sequences through the model
3. Extracts predicted camera poses from model outputs
4. Compares with normalized ground truth camera poses
5. Visualizes both in 3D space
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import logging
import torch
from pathlib import Path

# Add ml-cubifyanything to path
CUBIFY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ml-cubifyanything'))
if CUBIFY_PATH not in sys.path:
    sys.path.insert(0, CUBIFY_PATH)

# Add vggt training to path
TRAIN_PATH = os.path.dirname(__file__)
if TRAIN_PATH not in sys.path:
    sys.path.insert(0, TRAIN_PATH)

from data.datasets.cubifyanything import CubifyAnythingDataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LOSS CALCULATION UTILITIES (Integrated from debug_loss_calculation.py)
# ============================================================================

class PoseEncodingDebugger:
    """Encode/decode camera poses to/from compact 9D representation."""
    
    @staticmethod
    def mat_to_quat_scipy(R):
        """Convert rotation matrix to quaternion using scipy."""
        from scipy.spatial.transform import Rotation
        
        if isinstance(R, torch.Tensor):
            R_np = R.cpu().numpy()
        else:
            R_np = R
        
        rot = Rotation.from_matrix(R_np)
        quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        if isinstance(R, torch.Tensor):
            return torch.from_numpy(quat_wxyz).to(R.device).to(R.dtype)
        return quat_wxyz
    
    @staticmethod
    def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw):
        """
        Convert extrinsics [R|t] and intrinsics to 9D pose encoding.
        
        Returns: (..., 9) tensor with [T (3D) | quat (4D) | fov (2D)]
        """
        if isinstance(extrinsics, np.ndarray):
            extrinsics = torch.from_numpy(extrinsics).float()
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics).float()
        
        R = extrinsics[..., :3, :3]
        T = extrinsics[..., :3, 3]
        
        # Convert each rotation matrix to quaternion
        original_shape = R.shape[:-2]
        R_flat = R.reshape(-1, 3, 3)
        
        quats = []
        for i in range(R_flat.shape[0]):
            q = PoseEncodingDebugger.mat_to_quat_scipy(R_flat[i])
            quats.append(q)
        
        quat_tensor = torch.stack(quats).reshape(original_shape + (4,))
        
        # Compute field of view
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        
        pose_encoding = torch.cat(
            [T, quat_tensor, fov_h[..., None], fov_w[..., None]], 
            dim=-1
        ).float()
        
        return pose_encoding


class CameraLossCalculator:
    """Compute camera pose losses (Translation, Rotation, Focal Length)."""
    
    def __init__(self, loss_type="l1", gamma=0.6):
        self.loss_type = loss_type
        self.gamma = gamma
        self.pose_encoder = PoseEncodingDebugger()
    
    def compute_losses(
        self,
        pred_pose_encodings,  # List of [(..., 9)] per stage or single [(..., 9)]
        gt_extrinsics,        # (..., 3, 4)
        gt_intrinsics,        # (..., 3, 3)
        image_hw,             # (H, W)
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=0.5,
    ):
        """
        Compute camera pose losses.
        
        Args:
            pred_pose_encodings: List of predicted pose encodings per stage, or single tensor
            gt_extrinsics: Ground truth extrinsics
            gt_intrinsics: Ground truth intrinsics
            image_hw: Image height and width tuple
            
        Returns:
            loss_dict with loss_T, loss_R, loss_FL, and total loss
        """
        # Convert to pose encoding if needed
        if isinstance(pred_pose_encodings, list):
            # Multi-stage: compute weighted loss
            n_stages = len(pred_pose_encodings)
        else:
            # Single stage
            pred_pose_encodings = [pred_pose_encodings]
            n_stages = 1
        
        # Convert GT to pose encoding
        gt_pose_encoding = self.pose_encoder.extri_intri_to_pose_encoding(
            gt_extrinsics, gt_intrinsics, image_hw
        )
        
        # Accumulate losses across stages
        total_loss_T = 0.0
        total_loss_R = 0.0
        total_loss_FL = 0.0
        
        for stage_idx in range(n_stages):
            stage_weight = self.gamma ** (n_stages - stage_idx - 1)
            pred_pose_stage = pred_pose_encodings[stage_idx]
            
            # Ensure same shape
            if pred_pose_stage.shape != gt_pose_encoding.shape:
                pred_pose_stage = pred_pose_stage.reshape(gt_pose_encoding.shape)
            
            loss_T, loss_R, loss_FL = self._compute_component_losses(
                pred_pose_stage, gt_pose_encoding
            )
            
            total_loss_T += loss_T * stage_weight
            total_loss_R += loss_R * stage_weight
            total_loss_FL += loss_FL * stage_weight
        
        # Average over stages
        avg_loss_T = total_loss_T / n_stages
        avg_loss_R = total_loss_R / n_stages
        avg_loss_FL = total_loss_FL / n_stages
        
        # Weighted total
        total_camera_loss = (
            avg_loss_T * weight_trans +
            avg_loss_R * weight_rot +
            avg_loss_FL * weight_focal
        )
        
        return {
            "loss_T": avg_loss_T,
            "loss_R": avg_loss_R,
            "loss_FL": avg_loss_FL,
            "loss_camera": total_camera_loss,
        }
    
    def _compute_component_losses(self, pred_pose_enc, gt_pose_enc):
        """Compute L1 losses for each component."""
        # Extract components
        pred_T = pred_pose_enc[..., :3]
        pred_R = pred_pose_enc[..., 3:7]
        pred_FL = pred_pose_enc[..., 7:]
        
        gt_T = gt_pose_enc[..., :3]
        gt_R = gt_pose_enc[..., 3:7]
        gt_FL = gt_pose_enc[..., 7:]
        
        # L1 distances
        diff_T = (pred_T - gt_T).abs()
        diff_R = (pred_R - gt_R).abs()
        diff_FL = (pred_FL - gt_FL).abs()
        
        # Mean losses
        if isinstance(diff_T, torch.Tensor):
            loss_T = diff_T.clamp(max=100).mean().item()
            loss_R = diff_R.mean().item()
            loss_FL = diff_FL.mean().item()
        else:
            loss_T = np.mean(np.clip(diff_T, 0, 100))
            loss_R = np.mean(diff_R)
            loss_FL = np.mean(diff_FL)
        
        return loss_T, loss_R, loss_FL


class CommonConf:
    """Simple configuration object for CubifyAnythingDataset."""
    def __init__(self, cfg):
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.debug = False
        self.training = False  # Inference mode
        self.get_nearby = False
        self.inside_random = False
        self.allow_duplicate_img = True
        self.rescale = True
        self.rescale_aug = True
        self.landscape_check = True
        self.augs = type('obj', (object,), {
            'scales': [0.8, 1.2],
            'aspects': [0.5, 1.0]
        })()
        self.img_nums = [2, 24]


def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Uses ZYX convention (yaw-pitch-roll).
    
    Args:
        R: [3, 3] rotation matrix
    
    Returns:
        roll, pitch, yaw: Euler angles in degrees
    """
    # Check for gimbal lock
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    # Convert to degrees
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def get_camera_center(extrinsic_3x4):
    """
    Get camera center in world coordinates from extrinsic matrix.
    
    Args:
        extrinsic_3x4: [3, 4] world-from-camera matrix (NOT camera-from-world!)
    
    Returns:
        camera_center: [3] position in world coordinates
    """
    # extrinsic is world-from-camera [R|t] where:
    # - R is world-from-camera rotation
    # - t is camera position in world coordinates
    # So the camera center is simply the translation part!
    camera_center = extrinsic_3x4[:3, 3]
    
    return camera_center


def get_camera_frustum(extrinsic_3x4, intrinsic_3x3, scale=0.1, depth=0.3):
    """
    Get camera frustum points for visualization.
    
    Args:
        extrinsic_3x4: [3, 4] world-from-camera matrix
        intrinsic_3x3: [3, 3] camera intrinsics
        scale: scale factor for frustum size
        depth: depth of the frustum
    
    Returns:
        frustum_points_world: [5, 3] frustum corner points in world coordinates
    """
    # extrinsic is already world-from-camera [R|t]
    # No need to invert!
    w_from_c_R = extrinsic_3x4[:3, :3]
    w_from_c_t = extrinsic_3x4[:3, 3]
    
    # Camera intrinsics
    fx = intrinsic_3x3[0, 0]
    fy = intrinsic_3x3[1, 1]
    cx = intrinsic_3x3[0, 2]
    cy = intrinsic_3x3[1, 2]
    
    # Frustum corners in camera space
    frustum_camera = np.array([
        [0, 0, 0],  # Camera center
        [-(cx) / fx * depth * scale, -(cy) / fy * depth * scale, depth],
        [(fx - cx) / fx * depth * scale, -(cy) / fy * depth * scale, depth],
        [(fx - cx) / fx * depth * scale, (fy - cy) / fy * depth * scale, depth],
        [-(cx) / fx * depth * scale, (fy - cy) / fy * depth * scale, depth],
    ])
    
    # Transform to world coordinates using world-from-camera
    # world_point = R @ camera_point + t
    frustum_world = (w_from_c_R @ frustum_camera.T).T + w_from_c_t
    
    return frustum_world


def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load trained VGGT model from checkpoint.
    
    Args:
        checkpoint_path: path to model checkpoint
        device: device to load model to
    
    Returns:
        model: loaded VGGT model in eval mode
    """
    # Try to import VGGT model
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        logger.error("Could not import VGGT model. Make sure vggt is in Python path.")
        raise
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint if available
    if 'cfg' in checkpoint:
        model_cfg = checkpoint['cfg']
    else:
        # Use default config
        logger.warning("No config in checkpoint, using default VGGT config")
        model_cfg = {
            'img_size': 518,
            'patch_size': 14,
            'model_type': 'VGGT',
        }
    
    # Create model
    model = VGGT()
    
    # Load model state
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Assume entire checkpoint is model state
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model


def extract_predicted_poses(model, images_tensor, intrinsics_batch, avg_scale=1.0, device='cuda'):
    """
    Extract predicted camera poses from model output.
    
    Args:
        model: Trained VGGT model
        images_tensor: [B, S, C, H, W] or [S, C, H, W] tensor of images
        intrinsics_batch: [B, S, 3, 3] or [S, 3, 3] camera intrinsics
        avg_scale: Scale factor used for GT normalization (to apply inverse scaling to predictions)
        device: device to run inference on
    
    Returns:
        predicted_extrinsics: [B, S, 3, 4] or [S, 3, 4] predicted camera extrinsics
    """
    # Ensure inputs are on correct device
    if isinstance(images_tensor, np.ndarray):
        images_tensor = torch.from_numpy(images_tensor).float()
    if not isinstance(images_tensor, torch.Tensor):
        images_tensor = torch.tensor(images_tensor, dtype=torch.float32)
    
    images_tensor = images_tensor.to(device)
    
    # Add batch dimension if needed
    if images_tensor.dim() == 4:  # [S, C, H, W]
        images_tensor = images_tensor.unsqueeze(0)  # [1, S, C, H, W]
    
    model.eval()
    
    with torch.no_grad():
        # Run model inference
        outputs = model(images=images_tensor)
        
        logger.info(f"Model output type: {type(outputs)}")
        if isinstance(outputs, dict):
            logger.info(f"Model output keys: {outputs.keys()}")
        elif isinstance(outputs, torch.Tensor):
            logger.info(f"Model output shape: {outputs.shape}")
    
    # Extract predicted camera poses
    # VGGT model outputs geometry predictions, we need to extract camera poses
    predicted_extrinsics = extract_camera_from_model_output(outputs, images_tensor.shape, avg_scale, device)
    
    return predicted_extrinsics


def extract_camera_from_model_output(outputs, images_shape, avg_scale, device):
    """
    Extract camera extrinsics from VGGT model output.
    
    VGGT outputs: depth, confidence, aggregator features, and camera parameters.
    The model outputs 'pose_enc' which is [B, S, 9] containing:
    - T (3D): translation
    - quat (4D): quaternion for rotation
    - fl (2D): focal length / field of view
    
    This function extracts and converts to [B, S, 3, 4] extrinsic matrices.
    
    Args:
        outputs: Model output (dict or tensor)
        images_shape: [B, S, C, H, W] shape of input images
        avg_scale: Scale factor to apply for inverse normalization
        device: computation device
    
    Returns:
        extrinsics: [B, S, 3, 4] predicted camera extrinsics
    """
    B, S, C, H, W = images_shape
    
    # Try different output formats
    if isinstance(outputs, dict):
        # Check for direct camera extrinsics
        if 'camera' in outputs:
            cam_output = outputs['camera']
            if isinstance(cam_output, torch.Tensor) and cam_output.shape[-2:] == (3, 4):
                logger.info(f"Found direct camera output with shape {cam_output.shape}")
                return cam_output.cpu().numpy()
        
        # Check for pose_enc (VGGT camera head output) - [B, S, 9]
        if 'pose_enc' in outputs:
            pose_enc = outputs['pose_enc']
            logger.info(f"Found pose_enc with shape {pose_enc.shape}")
            if pose_enc.shape[-1] == 9:
                # Format: [T (3), quat (4), fl (2)]
                return extract_extrinsics_from_pose_enc(pose_enc, avg_scale, device)
            else:
                logger.warning(f"Unexpected pose_enc format: {pose_enc.shape}")
        
        # Check for camera parameters (rotation + translation)
        if 'camera_pose' in outputs or 'pose' in outputs:
            key = 'camera_pose' if 'camera_pose' in outputs else 'pose'
            cam_params = outputs[key]
            logger.info(f"Found camera parameters: {key} with shape {cam_params.shape}")
            return convert_pose_params_to_extrinsics(cam_params, B, S, device)
        
        # Check for rotation and translation separately
        if 'rotation' in outputs and 'translation' in outputs:
            logger.info("Found separate rotation and translation outputs")
            return combine_rotation_translation(outputs['rotation'], outputs['translation'], B, S, device)
        
        # If no camera outputs, create dummy based on image features
        logger.warning("No explicit camera outputs found in model. Using identity poses as fallback.")
        return create_identity_extrinsics(B, S, device).cpu().numpy()
    
    elif isinstance(outputs, torch.Tensor):
        # If model just outputs a tensor, assume it's camera matrix
        if outputs.shape[-2:] == (3, 4):
            logger.info(f"Model output tensor has camera matrix shape {outputs.shape}")
            return outputs.cpu().numpy()
        else:
            logger.warning(f"Model output tensor has unexpected shape {outputs.shape}. Using identity poses.")
            return create_identity_extrinsics(B, S, device).cpu().numpy()
    
    else:
        logger.warning(f"Unknown output format: {type(outputs)}")
        return create_identity_extrinsics(B, S, device).cpu().numpy()


def extract_extrinsics_from_pose_enc(pose_enc, avg_scale, device):
    """
    Extract camera extrinsics from VGGT pose encoding [B, S, 9].
    
    Format: [T (3D translation), quat (4D quaternion), fl (2D focal length)]
    
    Args:
        pose_enc: [B, S, 9] pose encoding tensor
        avg_scale: Scale factor for inverse normalization
        device: computation device
    
    Returns:
        extrinsics: [B, S, 3, 4] camera extrinsic matrices
    """
    B, S, D = pose_enc.shape
    assert D == 9, f"Expected pose_enc to have 9 dimensions, got {D}"
    
    # Extract components
    T = pose_enc[..., :3]  # [B, S, 3]
    quat = pose_enc[..., 3:7]  # [B, S, 4]
    fl = pose_enc[..., 7:]  # [B, S, 2]
    
    # Normalize quaternion
    quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
    
    # Convert quaternion to rotation matrix
    extrinsics = []
    for b in range(B):
        seq_extrinsics = []
        for s in range(S):
            q = quat[b, s]  # [4]
            t = T[b, s]  # [3]
            
            # Quaternion to rotation matrix (w, x, y, z format assumed from output)
            # Extract components: q = [qx, qy, qz, qw]
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            
            # Build rotation matrix from quaternion
            R = torch.tensor([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ], dtype=torch.float32, device=device)
            
            # Create extrinsic matrix [R | t]
            extri = torch.cat([R, t.unsqueeze(1)], dim=1)  # [3, 4]
            seq_extrinsics.append(extri)
        
        extrinsics.append(torch.stack(seq_extrinsics))
    
    result = torch.stack(extrinsics)  # [B, S, 3, 4]
    logger.info(f"Extracted extrinsics with shape {result.shape}")
    
    # Model outputs world-from-camera format (same as GT from dataset)
    # Both are already in the same coordinate frame and scale
    # NO CONVERSION NEEDED - just return as numpy array
    
    logger.info("Model predictions are in world-from-camera format (normalized)")
    logger.info("Matching GT format - no conversion needed!")
    result_np = result.cpu().numpy()
    
    # Return directly without any transformations
    logger.info(f"Prediction extrinsics shape: {result_np.shape}")
    return result_np


def convert_pose_params_to_extrinsics(pose_params, B, S, device):
    """
    Convert pose parameters (rotation + translation) to extrinsic matrices.
    
    Args:
        pose_params: [B, S, D] pose parameters (6D: 3D rotation + 3D translation)
        B, S: batch size and sequence length
        device: computation device
    
    Returns:
        extrinsics: [B, S, 3, 4] extrinsic matrices
    """
    extrinsics = []
    
    for b in range(pose_params.shape[0]):
        seq_extrinsics = []
        for s in range(pose_params.shape[1]):
            params = pose_params[b, s]  # [D]
            
            if len(params) >= 6:
                # 6D: rotation vector (3D) + translation (3D)
                rotation_vec = params[:3]
                translation = params[3:6]
                
                # Convert rotation vector to rotation matrix using Rodrigues formula
                angle = torch.norm(rotation_vec)
                if angle > 1e-6:
                    axis = rotation_vec / angle
                    K = torch.tensor([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ], dtype=torch.float32, device=device)
                    R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
                else:
                    R = torch.eye(3, device=device)
            else:
                logger.warning(f"Unexpected pose parameter size: {len(params)}. Using identity.")
                R = torch.eye(3, device=device)
                translation = torch.zeros(3, device=device)
            
            # Create [3, 4] extrinsic matrix
            extri = torch.cat([R, translation.unsqueeze(1)], dim=1)  # [3, 4]
            seq_extrinsics.append(extri)
        
        extrinsics.append(torch.stack(seq_extrinsics))
    
    return torch.stack(extrinsics).cpu().numpy()  # [B, S, 3, 4]


def combine_rotation_translation(rotation, translation, B, S, device):
    """Combine separate rotation and translation outputs into extrinsics."""
    extrinsics = []
    
    for b in range(B):
        seq_extrinsics = []
        for s in range(S):
            R = rotation[b, s] if rotation.shape[0] > b else rotation[s]
            t = translation[b, s] if translation.shape[0] > b else translation[s]
            
            # Ensure shapes
            if R.shape != (3, 3):
                logger.warning(f"Unexpected rotation shape {R.shape}")
                R = torch.eye(3, device=device)
            if t.shape != (3,):
                logger.warning(f"Unexpected translation shape {t.shape}")
                t = torch.zeros(3, device=device)
            
            extri = torch.cat([R, t.unsqueeze(1)], dim=1)
            seq_extrinsics.append(extri)
        
        extrinsics.append(torch.stack(seq_extrinsics))
    
    return torch.stack(extrinsics).cpu().numpy()


def create_identity_extrinsics(B, S, device):
    """Create identity extrinsic matrices as fallback."""
    identity = torch.eye(4, device=device)[:3, :]  # [3, 4]
    return identity.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)


def convert_params_to_extrinsics(camera_params):
    """
    Convert camera parameters to extrinsic matrices.
    
    Args:
        camera_params: [B, S, D] camera parameters (format depends on model)
    
    Returns:
        extrinsics: [B, S, 3, 4] extrinsic matrices
    """
    # This is a placeholder - adjust based on your model's output format
    B, S, D = camera_params.shape
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    extrinsics = extrinsics[:, :, :3, :].to(camera_params.device)
    return extrinsics


def plot_error_table(ax, gt_extrinsics, pred_extrinsics):
    """Plot a table showing per-frame rotation and translation errors."""
    ax.axis('off')
    
    # Compute errors for each frame
    rot_errors = []
    trans_errors = []
    
    for gt_e, pred_e in zip(gt_extrinsics, pred_extrinsics):
        # Rotation error
        gt_R = gt_e[:3, :3]
        pred_R = pred_e[:3, :3]
        R_diff = gt_R.T @ pred_R
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_error = np.arccos(cos_angle) * 180 / np.pi
        rot_errors.append(angle_error)
        
        # Translation error
        trans_error = np.linalg.norm(gt_e[:3, 3] - pred_e[:3, 3])
        trans_errors.append(trans_error)
    
    # Create table data
    table_data = [['Frame', 'Rot Error (°)', 'Trans Error (m)']]
    for i, (rot_err, trans_err) in enumerate(zip(rot_errors, trans_errors)):
        table_data.append([f'{i}', f'{rot_err:.3f}', f'{trans_err:.5f}'])
    
    # Add summary row
    table_data.append(['─' * 5, '─' * 12, '─' * 14])
    table_data.append(['Mean', f'{np.mean(rot_errors):.3f}', f'{np.mean(trans_errors):.5f}'])
    table_data.append(['Max', f'{np.max(rot_errors):.3f}', f'{np.max(trans_errors):.5f}'])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Per-Frame Error Summary', fontweight='bold', pad=20)


def plot_rotation_matrix_error(ax, gt_extrinsics, pred_extrinsics):
    """Plot Frobenius norm of rotation matrix difference."""
    frob_norms = []
    angular_errors = []
    
    for gt_e, pred_e in zip(gt_extrinsics, pred_extrinsics):
        gt_R = gt_e[:3, :3]
        pred_R = pred_e[:3, :3]
        
        # Frobenius norm of difference
        frob_norm = np.linalg.norm(gt_R - pred_R, 'fro')
        frob_norms.append(frob_norm)
        
        # Angular error
        R_diff = gt_R.T @ pred_R
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_error = np.arccos(cos_angle) * 180 / np.pi
        angular_errors.append(angle_error)
    
    # Dual y-axis plot
    ax2 = ax.twinx()
    
    line1 = ax.plot(frob_norms, 'o-', linewidth=2, markersize=6, 
                    color='steelblue', label='Frobenius Norm')
    ax.fill_between(range(len(frob_norms)), 0, frob_norms, alpha=0.2, color='steelblue')
    
    line2 = ax2.plot(angular_errors, 's-', linewidth=2, markersize=6, 
                     color='coral', label='Angular Error')
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('||R_gt - R_pred||_F', color='steelblue')
    ax2.set_ylabel('Angular Error (degrees)', color='coral')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    ax.set_title('Rotation Matrix Error Metrics')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')


def plot_euler_angles_comparison(ax, gt_extrinsics, pred_extrinsics):
    """
    Plot roll, pitch, yaw angles over sequence comparing GT and predictions.
    
    Args:
        ax: matplotlib axis (3 subplots will be created)
        gt_extrinsics: [S, 3, 4] ground truth extrinsics
        pred_extrinsics: [S, 3, 4] predicted extrinsics
    """
    # Extract Euler angles for GT and predictions
    gt_rolls, gt_pitches, gt_yaws = [], [], []
    pred_rolls, pred_pitches, pred_yaws = [], [], []
    
    for gt_e, pred_e in zip(gt_extrinsics, pred_extrinsics):
        gt_R = gt_e[:3, :3]
        pred_R = pred_e[:3, :3]
        
        gt_roll, gt_pitch, gt_yaw = rotation_matrix_to_euler_angles(gt_R)
        pred_roll, pred_pitch, pred_yaw = rotation_matrix_to_euler_angles(pred_R)
        
        gt_rolls.append(gt_roll)
        gt_pitches.append(gt_pitch)
        gt_yaws.append(gt_yaw)
        
        pred_rolls.append(pred_roll)
        pred_pitches.append(pred_pitch)
        pred_yaws.append(pred_yaw)
    
    gt_rolls = np.array(gt_rolls)
    gt_pitches = np.array(gt_pitches)
    gt_yaws = np.array(gt_yaws)
    pred_rolls = np.array(pred_rolls)
    pred_pitches = np.array(pred_pitches)
    pred_yaws = np.array(pred_yaws)
    
    frames = np.arange(len(gt_extrinsics))
    
    # Plot roll
    ax.plot(frames, gt_rolls, 'o-', linewidth=2, markersize=6, 
            color='green', label='GT Roll', alpha=0.7)
    ax.plot(frames, pred_rolls, '^--', linewidth=2, markersize=6, 
            color='lightgreen', label='Pred Roll', alpha=0.7)
    
    # Plot pitch
    ax.plot(frames, gt_pitches, 's-', linewidth=2, markersize=6, 
            color='blue', label='GT Pitch', alpha=0.7)
    ax.plot(frames, pred_pitches, 'd--', linewidth=2, markersize=6, 
            color='lightblue', label='Pred Pitch', alpha=0.7)
    
    # Plot yaw
    ax.plot(frames, gt_yaws, 'p-', linewidth=2, markersize=6, 
            color='red', label='GT Yaw', alpha=0.7)
    ax.plot(frames, pred_yaws, '*--', linewidth=2, markersize=6, 
            color='salmon', label='Pred Yaw', alpha=0.7)
    
    ax.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Roll, Pitch, Yaw Comparison (GT vs Predicted)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', ncol=2, fontsize=9)
    
    # Add error statistics
    roll_error = np.abs(gt_rolls - pred_rolls)
    pitch_error = np.abs(gt_pitches - pred_pitches)
    yaw_error = np.abs(gt_yaws - pred_yaws)
    
    stats_text = f'Mean Errors:\n'
    stats_text += f'Roll: {roll_error.mean():.3f}°\n'
    stats_text += f'Pitch: {pitch_error.mean():.3f}°\n'
    stats_text += f'Yaw: {yaw_error.mean():.3f}°'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=9, family='monospace')


def normalize_extrinsics_like_loss(extrinsics, world_points=None, point_masks=None):
    """
    Apply the EXACT SAME normalization as the loss calculation uses.
    
    This ensures visualizations show what the loss sees:
    1. Center to first camera coordinate system
    2. Scale by average distance of world points (if provided)
    
    Args:
        extrinsics: [S, 3, 4] camera extrinsics
        world_points: [S, H, W, 3] world 3D points (optional, for scaling)
        point_masks: [S, H, W] binary mask for valid points (optional)
    
    Returns:
        normalized_extrinsics: [S, 3, 4] in normalized space
        avg_scale: scale factor used (for reverse scaling if needed)
    """
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    S = extrinsics.shape[0]
    
    # STEP 1: Convert to homogeneous form
    extrinsics_homog = np.concatenate([
        extrinsics,
        np.zeros((S, 1, 4))
    ], axis=1)
    extrinsics_homog[:, 2, 3] = 1.0  # Set [3,3] to 1
    
    # STEP 2: Get inverse of first camera's extrinsic (cam_to_world for first camera)
    # This centers the coordinate system to the first camera
    first_cam = extrinsics_homog[0]  # [4, 4]
    
    # Compute SE(3) inverse manually for robust inversion
    R0 = first_cam[:3, :3]
    t0 = first_cam[:3, 3]
    R0_inv = R0.T
    t0_inv = -R0_inv @ t0
    
    first_cam_inv = np.eye(4)
    first_cam_inv[:3, :3] = R0_inv
    first_cam_inv[:3, 3] = t0_inv
    
    # Transform all cameras to first camera frame
    new_extrinsics = extrinsics_homog @ first_cam_inv
    new_extrinsics = new_extrinsics[:, :3, :]  # Back to [S, 3, 4]
    
    # STEP 3: Scale by average distance of world points (if provided)
    avg_scale = 1.0
    if world_points is not None and point_masks is not None:
        world_points = np.asarray(world_points, dtype=np.float32)
        point_masks = np.asarray(point_masks, dtype=np.float32)
        
        # Transform world points to first camera frame
        # pts_cam = R0 @ pts_world + t0
        R0 = extrinsics[0, :3, :3]
        t0 = extrinsics[0, :3, 3]
        
        pts_transformed = (world_points @ R0.T[np.newaxis, np.newaxis, :, :]).squeeze(-2) + t0[np.newaxis, np.newaxis, :]
        
        # Compute average distance
        dist = np.linalg.norm(pts_transformed, axis=-1)
        dist_sum = (dist * point_masks).sum()
        valid_count = point_masks.sum()
        
        if valid_count > 0:
            avg_scale = max(1e-6, dist_sum / valid_count)
            avg_scale = min(avg_scale, 1e6)  # Clamp to reasonable range
            
            # Scale translations
            new_extrinsics[:, :3, 3] /= avg_scale
            
            logger.info(f"Applied normalization scaling: avg_scale = {avg_scale:.4f}")
    
    return new_extrinsics, avg_scale


def plot_pose_comparison(gt_extrinsics, gt_intrinsics, pred_extrinsics, pred_intrinsics, seq_name="", compute_losses=True, world_points=None, point_masks=None, apply_normalization=True):
    """
    Plot comparison between ground truth and predicted camera poses with loss calculations.
    
    This visualization applies the EXACT SAME NORMALIZATION as the loss calculation,
    so the graphs show what the loss function sees.
    
    Args:
        gt_extrinsics: [S, 3, 4] ground truth extrinsics
        gt_intrinsics: [S, 3, 3] ground truth intrinsics
        pred_extrinsics: [S, 3, 4] predicted extrinsics
        pred_intrinsics: [S, 3, 3] predicted intrinsics
        seq_name: name of sequence for title
        compute_losses: whether to compute and display loss information
        world_points: [S, H, W, 3] world 3D points (for scale normalization)
        point_masks: [S, H, W] binary mask for valid points (for scale normalization)
        apply_normalization: whether to apply training-style normalization to extrinsics
    """
    
    # IMPORTANT: Apply the same normalization as training!
    # This ensures visualizations show what the loss sees.
    gt_extrinsics_norm = gt_extrinsics.copy()
    pred_extrinsics_norm = pred_extrinsics.copy()
    normalization_info = {"applied": False, "avg_scale": 1.0}
    
    if apply_normalization:
        try:
            logger.info("\n" + "="*80)
            logger.info("APPLYING NORMALIZATION (same as training loss)")
            logger.info("="*80)
            
            gt_extrinsics_norm, scale_gt = normalize_extrinsics_like_loss(
                gt_extrinsics, world_points, point_masks
            )
            pred_extrinsics_norm, scale_pred = normalize_extrinsics_like_loss(
                pred_extrinsics, world_points, point_masks
            )
            
            # Use GT scale for both for consistency
            avg_scale = scale_gt
            if scale_gt > 1e-6:
                pred_extrinsics_norm[:, :3, 3] *= scale_gt / scale_pred
            
            normalization_info = {
                "applied": True,
                "avg_scale": avg_scale,
                "gt_scale": scale_gt,
                "pred_scale": scale_pred,
            }
            
            logger.info(f"GT scale factor:   {scale_gt:.4f}")
            logger.info(f"Pred scale factor: {scale_pred:.4f}")
            logger.info(f"Using scale:       {avg_scale:.4f}")
            logger.info("="*80 + "\n")
            
        except Exception as e:
            logger.warning(f"Could not apply normalization: {e}")
            logger.info("Using original (non-normalized) extrinsics\n")
    
    # Use normalized extrinsics for visualization
    gt_ext_for_plot = gt_extrinsics_norm if apply_normalization else gt_extrinsics
    pred_ext_for_plot = pred_extrinsics_norm if apply_normalization else pred_extrinsics
    
    # Compute losses if requested
    # IMPORTANT: Compute losses on ORIGINAL extrinsics (not normalized) to match training
    loss_dict = None
    if compute_losses:
        try:
            calculator = CameraLossCalculator(loss_type="l1", gamma=0.6)
            
            # Convert to torch tensors - use ORIGINAL extrinsics for loss
            gt_ext_torch = torch.from_numpy(gt_extrinsics).float() if isinstance(gt_extrinsics, np.ndarray) else gt_extrinsics
            gt_int_torch = torch.from_numpy(gt_intrinsics).float() if isinstance(gt_intrinsics, np.ndarray) else gt_intrinsics
            pred_ext_torch = torch.from_numpy(pred_extrinsics).float() if isinstance(pred_extrinsics, np.ndarray) else pred_extrinsics
            
            # Create dummy pred_pose_enc from extrinsics for loss computation
            pose_encoder = PoseEncodingDebugger()
            pred_pose_enc = pose_encoder.extri_intri_to_pose_encoding(
                pred_ext_torch, 
                gt_int_torch,  # Use GT intrinsics for visualization
                (gt_intrinsics.shape[-2], gt_intrinsics.shape[-1] if gt_intrinsics.ndim > 2 else 512)
            )
            
            loss_dict = calculator.compute_losses(
                pred_pose_enc,
                gt_ext_torch,
                gt_int_torch,
                (256, 512),  # Default image size
                weight_trans=1.0,
                weight_rot=1.0,
                weight_focal=0.5,
            )
            
            logger.info(f"\n{'='*80}")
            logger.info("LOSS CALCULATION RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Translation Loss (L1):  {loss_dict['loss_T']:.8f}")
            logger.info(f"Rotation Loss (L1):     {loss_dict['loss_R']:.8f}")
            logger.info(f"Focal Length Loss (L1): {loss_dict['loss_FL']:.8f}")
            logger.info(f"Total Camera Loss:      {loss_dict['loss_camera']:.8f}")
            logger.info(f"{'='*80}\n")
        except Exception as e:
            logger.warning(f"Could not compute losses: {e}")
            loss_dict = None
    
    fig = plt.figure(figsize=(24, 18))
    
    # Add normalization info to title
    norm_status = " [NORMALIZED LIKE LOSS]" if apply_normalization and normalization_info["applied"] else ""
    
    # Overlay comparison - 3D plot (use NORMALIZED extrinsics)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_overlay_comparison(ax1, gt_ext_for_plot, gt_intrinsics, pred_ext_for_plot, pred_intrinsics)
    
    # Translation error over sequence (use NORMALIZED extrinsics)
    ax2 = fig.add_subplot(2, 3, 2)
    plot_translation_error(ax2, gt_ext_for_plot, pred_ext_for_plot)
    
    # Rotation error visualization (use NORMALIZED extrinsics)
    ax3 = fig.add_subplot(2, 3, 3)
    plot_rotation_error(ax3, gt_ext_for_plot, pred_ext_for_plot)
    
    # Camera center distance error (use NORMALIZED extrinsics)
    ax4 = fig.add_subplot(2, 3, 4)
    plot_camera_center_error(ax4, gt_ext_for_plot, pred_ext_for_plot)
    
    # Loss components bar chart (if computed)
    if loss_dict:
        ax5 = fig.add_subplot(2, 3, 5)
        plot_loss_components(ax5, loss_dict)
        
        # Loss & error summary table (use NORMALIZED extrinsics)
        ax6 = fig.add_subplot(2, 3, 6)
        plot_loss_summary_table(ax6, loss_dict, pred_ext_for_plot, gt_ext_for_plot)
    else:
        # Rotation matrix Frobenius norm comparison (use NORMALIZED extrinsics)
        ax5 = fig.add_subplot(2, 3, 5)
        plot_rotation_matrix_error(ax5, gt_ext_for_plot, pred_ext_for_plot)
        
        # Euler angles (Roll, Pitch, Yaw) comparison (use NORMALIZED extrinsics)
        ax6 = fig.add_subplot(2, 3, 6)
        plot_euler_angles_comparison(ax6, gt_ext_for_plot, pred_ext_for_plot)
    
    fig.suptitle(f'Camera Pose Comparison: {seq_name}{norm_status}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, loss_dict

def compareRotationalError(ax, gt_extrinsics, pred_extrinsics, plot=True):
    """
    Compute and optionally plot rotation error over sequence.
    
    Uses the metric: error = arccos((trace(R^T @ R) - 1) / 2)
    This measures the angle of rotation between GT and predicted poses.
    
    Args:
        ax: matplotlib axis (if plot=True)
        gt_extrinsics: [S, 3, 4] ground truth extrinsics
        pred_extrinsics: [S, 3, 4] predicted extrinsics
        plot: whether to create plot on axis
    
    Returns:
        errors: [S] rotation errors in degrees
    """
    errors = []
    
    for gt_e, pred_e in zip(gt_extrinsics, pred_extrinsics):
        # Extract rotation matrices
        gt_R = gt_e[:3, :3]
        pred_R = pred_e[:3, :3]
        
        # Compute angular error using trace formula
        # This measures the angle of rotation between the two matrices
        # error = arccos((trace(R^T @ R) - 1) / 2)
        R_diff = gt_R.T @ pred_R
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_error = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees
        errors.append(angle_error)
    
    errors = np.array(errors)
    
    if plot:
        ax.plot(errors, 's-', linewidth=2, markersize=6, color='orange', label='Rotation Error')
        ax.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='orange')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Rotation Error (degrees)')
        ax.set_title('Rotation Error Over Sequence')
        ax.grid(True, alpha=0.3)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        ax.text(0.5, 0.95, f'Mean: {mean_error:.2f}°\nMax: {max_error:.2f}°',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()
    
    return errors

def plot_trajectory(ax, extrinsics, intrinsics, title="Trajectory", color='blue'):
    """Plot camera trajectory in 3D space."""
    camera_centers = []
    
    for extri, intri in zip(extrinsics, intrinsics):
        center = get_camera_center(extri)
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    
    # Plot camera centers
    ax.scatter(*camera_centers.T, c=color, s=100, marker='o', label='Camera centers', alpha=0.7)
    
    # Plot trajectory line
    ax.plot(*camera_centers.T, c=color, alpha=0.5, linewidth=2)
    
    # Plot frustums for first and last frame
    for i in [0, len(extrinsics)-1]:
        frustum = get_camera_frustum(extrinsics[i], intrinsics[i])
        ax.scatter(*frustum.T, c=color, s=20, alpha=0.4)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    if len(camera_centers) > 0:
        max_range = np.array([
            camera_centers[:, 0].max() - camera_centers[:, 0].min(),
            camera_centers[:, 1].max() - camera_centers[:, 1].min(),
            camera_centers[:, 2].max() - camera_centers[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (camera_centers[:, 0].max() + camera_centers[:, 0].min()) * 0.5
        mid_y = (camera_centers[:, 1].max() + camera_centers[:, 1].min()) * 0.5
        mid_z = (camera_centers[:, 2].max() + camera_centers[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_overlay_comparison(ax, gt_extrinsics, gt_intrinsics, pred_extrinsics, pred_intrinsics):
    """Plot ground truth and predicted poses overlaid."""
    gt_centers = np.array([get_camera_center(e) for e in gt_extrinsics])
    pred_centers = np.array([get_camera_center(e) for e in pred_extrinsics])
    
    ax.scatter(*gt_centers.T, c='green', s=100, marker='o', label='GT centers', alpha=0.7)
    ax.scatter(*pred_centers.T, c='red', s=100, marker='^', label='Predicted centers', alpha=0.7)
    
    ax.plot(*gt_centers.T, c='green', alpha=0.3, linewidth=2, label='GT trajectory')
    ax.plot(*pred_centers.T, c='red', alpha=0.3, linewidth=2, linestyle='--', label='Pred trajectory')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Overlay Comparison (GT=Green, Pred=Red)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_loss_components(ax, loss_dict):
    """
    Plot bar chart of loss components.
    
    Args:
        ax: matplotlib axis
        loss_dict: Dict with loss_T, loss_R, loss_FL, loss_camera keys
    """
    components = ['Translation', 'Rotation', 'Focal Len', 'Total']
    losses = [
        loss_dict.get('loss_T', 0),
        loss_dict.get('loss_R', 0),
        loss_dict.get('loss_FL', 0),
        loss_dict.get('loss_camera', 0),
    ]
    
    colors = ['steelblue', 'coral', 'lightgreen', 'darkred']
    bars = ax.bar(components, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.6f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
    ax.set_title('Loss Components Breakdown', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(losses) * 1.15)


def plot_loss_summary_table(ax, loss_dict, pred_extrinsics, gt_extrinsics):
    """
    Plot a comprehensive loss and error summary table.
    
    Args:
        ax: matplotlib axis
        loss_dict: Loss components
        pred_extrinsics: Predicted extrinsics
        gt_extrinsics: Ground truth extrinsics
    """
    ax.axis('off')
    
    # Compute additional metrics
    gt_t = np.array([e[:3, 3] for e in gt_extrinsics])
    pred_t = np.array([e[:3, 3] for e in pred_extrinsics])
    trans_errors = np.linalg.norm(gt_t - pred_t, axis=1)
    
    rot_errors = []
    for gt_e, pred_e in zip(gt_extrinsics, pred_extrinsics):
        gt_R = gt_e[:3, :3]
        pred_R = pred_e[:3, :3]
        R_diff = gt_R.T @ pred_R
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_error = np.arccos(cos_angle) * 180 / np.pi
        rot_errors.append(angle_error)
    rot_errors = np.array(rot_errors)
    
    # Build table data
    table_data = [
        ['Metric', 'Value', 'Unit'],
        ['─' * 20, '─' * 15, '─' * 10],
        ['Loss_T (Translation)', f'{loss_dict.get("loss_T", 0):.8f}', 'normalized'],
        ['Loss_R (Rotation)', f'{loss_dict.get("loss_R", 0):.8f}', 'unitless'],
        ['Loss_FL (Focal Len)', f'{loss_dict.get("loss_FL", 0):.8f}', 'normalized'],
        ['Total Camera Loss', f'{loss_dict.get("loss_camera", 0):.8f}', 'combined'],
        ['─' * 20, '─' * 15, '─' * 10],
        ['Mean Trans Error', f'{trans_errors.mean():.8f}', 'm'],
        ['Max Trans Error', f'{trans_errors.max():.8f}', 'm'],
        ['Mean Rotation Error', f'{rot_errors.mean():.4f}', '°'],
        ['Max Rotation Error', f'{rot_errors.max():.4f}', '°'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.35, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator rows
    for i in range(3):
        table[(1, i)].set_facecolor('#f0f0f0')
        table[(6, i)].set_facecolor('#f0f0f0')
    
    ax.set_title('Loss & Error Summary', fontweight='bold', fontsize=12, pad=20)


def plot_translation_error(ax, gt_extrinsics, pred_extrinsics):
    """Plot translation error over sequence."""
    gt_t = np.array([e[:3, 3] for e in gt_extrinsics])
    pred_t = np.array([e[:3, 3] for e in pred_extrinsics])
    
    errors = np.linalg.norm(gt_t - pred_t, axis=1)
    
    ax.plot(errors, 'o-', linewidth=2, markersize=6, color='purple')
    ax.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='purple')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Translation Error (m)')
    ax.set_title('Translation Error Over Sequence')
    ax.grid(True, alpha=0.3)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    ax.text(0.5, 0.95, f'Mean: {mean_error:.4f}m\nMax: {max_error:.4f}m',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_rotation_error(ax, gt_extrinsics, pred_extrinsics):
    """Plot rotation error over sequence."""
    errors = []
    
    for gt_e, pred_e in zip(gt_extrinsics, pred_extrinsics):
        # Extract rotation matrices
        gt_R = gt_e[:3, :3]
        pred_R = pred_e[:3, :3]
        
        # Compute angular error using trace formula
        # error = arccos((trace(R^T @ R) - 1) / 2)
        R_diff = gt_R.T @ pred_R
        trace = np.trace(R_diff)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_error = np.arccos(cos_angle) * 180 / np.pi  # Convert to degrees
        errors.append(angle_error)
    
    errors = np.array(errors)
    
    ax.plot(errors, 's-', linewidth=2, markersize=6, color='orange')
    ax.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='orange')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Rotation Error (degrees)')
    ax.set_title('Rotation Error Over Sequence')
    ax.grid(True, alpha=0.3)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    ax.text(0.5, 0.95, f'Mean: {mean_error:.2f}°\nMax: {max_error:.2f}°',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_camera_center_error(ax, gt_extrinsics, pred_extrinsics):
    """Plot camera center distance error."""
    gt_centers = np.array([get_camera_center(e) for e in gt_extrinsics])
    pred_centers = np.array([get_camera_center(e) for e in pred_extrinsics])
    
    errors = np.linalg.norm(gt_centers - pred_centers, axis=1)
    
    ax.plot(errors, '^-', linewidth=2, markersize=6, color='cyan')
    ax.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='cyan')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Camera Center Distance (m)')
    ax.set_title('Camera Center Position Error')
    ax.grid(True, alpha=0.3)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    ax.text(0.5, 0.95, f'Mean: {mean_error:.4f}m\nMax: {max_error:.4f}m',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_dummy_predictions(gt_extrinsics, gt_intrinsics, noise_level=0.01):
    """
    Create dummy predicted poses for testing (with small noise).
    In real usage, these would come from model inference.
    
    Args:
        gt_extrinsics: [S, 3, 4] ground truth extrinsics
        gt_intrinsics: [S, 3, 3] ground truth intrinsics
        noise_level: standard deviation of Gaussian noise to add
    
    Returns:
        pred_extrinsics, pred_intrinsics: perturbed versions
    """
    pred_extrinsics = []
    
    for extri in gt_extrinsics:
        # Add small Gaussian noise to translation
        noisy_extri = extri.copy().astype(np.float32)
        noisy_extri[:3, 3] += np.random.normal(0, noise_level, 3)
        
        # Add small rotation perturbation
        # Create small rotation matrix with angle from noise
        angle = np.random.normal(0, noise_level, 1)[0]
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Simple rotation approximation (for small angles)
        R_pert = np.eye(3) + angle * np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        noisy_extri[:3, :3] = R_pert @ noisy_extri[:3, :3]
        pred_extrinsics.append(noisy_extri)
    
    pred_extrinsics = np.array(pred_extrinsics)
    pred_intrinsics = gt_intrinsics.copy()  # Intrinsics don't change
    
    return pred_extrinsics, pred_intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize predicted vs. ground truth camera poses'
    )
    parser.add_argument('--interactive', action='store_true', help='Interactive mode: iterate through all sequences')
    parser.add_argument(
        '--config',
        type=str,
        default='config/cubifyanything.yaml',
        help='Path to Hydra config file'
    )
    parser.add_argument(
        '--seq-index',
        type=int,
        default=0,
        help='Sequence index to visualize'
    )
    parser.add_argument(
        '--img-per-seq',
        type=int,
        default=5,
        help='Number of images per sequence'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save visualization to file instead of showing'
    )
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default=None,
        help='Path to trained model checkpoint (if None, uses dummy predictions for demo)'
    )
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.01,
        help='Noise level for dummy predictions (only if --model-checkpoint not provided)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    config_path = os.path.abspath(config_path)
    
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from {config_path}")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    
    # Create dataset
    logger.info("Creating CubifyAnythingDataset...")
    cubify_config = cfg.data.val.dataset.dataset_configs[0]
    dataset = CubifyAnythingDataset(
        common_conf=CommonConf(cfg),
        split=cubify_config.split,
        CUBIFY_URL=cubify_config.CUBIFY_URL
    )
    
    # Load model
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_checkpoint:
        logger.info(f"Loading model from {args.model_checkpoint}...")
        logger.info(f"Using device: {device}")
        try:
            model = load_trained_model(args.model_checkpoint, device=device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Continue without model (will use dummy)

    # Determine sequences to process
    if args.interactive:
        logger.info("Starting interactive mode. Press any key to advance to next sequence.")
        try:
            sequences_to_process = range(len(dataset))
        except:
            logger.warning("Dataset does not support len(), using range(1000)")
            sequences_to_process = range(1000)
    else:
        sequences_to_process = [args.seq_index]

    for seq_idx in sequences_to_process:
        # Get ground truth data
        logger.info(f"Loading sequence {seq_idx}...")
        try:
            data_batch = dataset.get_data(
                seq_index=seq_idx,
                img_per_seq=args.img_per_seq,
                aspect_ratio=1.0
            )
        except Exception as e:
            logger.warning(f"Could not load sequence {seq_idx}: {e}")
            continue

        logger.info(f"Sampled frame IDs: {data_batch['ids']}")
        
        gt_extrinsics = np.array(data_batch['extrinsics'])
        gt_intrinsics = np.array(data_batch['intrinsics'])
        seq_name = data_batch['seq_name']
        
        logger.info(f"Ground truth extrinsics shape: {gt_extrinsics.shape}")
        logger.info(f"Ground truth intrinsics shape: {gt_intrinsics.shape}")
        
        # Dataset outputs world-from-camera format (normalized)
        # Model also outputs world-from-camera format (normalized)
        # For visualization, we'll work directly in world-from-camera space
        # NO COORDINATE CONVERSION NEEDED - both are already in the same format!
        
        logger.info("="*80)
        logger.info("EXTRINSICS FORMAT CHECK")
        logger.info("="*80)
        logger.info("GT extrinsics format: world-from-camera (from dataset)")
        logger.info("Model predictions format: world-from-camera (from model)")
        logger.info("Both are in normalized space - no conversion needed!")
        logger.info(f"\nGT Frame 0 (world-from-camera):")
        logger.info(f"  Rotation:\n{gt_extrinsics[0][:3, :3]}")
        logger.info(f"  Translation: {gt_extrinsics[0][:3, 3]}")
        if len(gt_extrinsics) > 1:
            logger.info(f"\nGT Frame 1 (world-from-camera):")
            logger.info(f"  Rotation:\n{gt_extrinsics[1][:3, :3]}")
            logger.info(f"  Translation: {gt_extrinsics[1][:3, 3]}")
        logger.info("="*80)
        
        # Get the scale factor used for normalization (if available)
        avg_scale = data_batch.get('avg_scale', 1.0)
        logger.info(f"Scale factor from dataset: {avg_scale:.4f}")
        
        # Get predicted poses
        pred_extrinsics = None
        pred_intrinsics = None

        if model:
            try:
                # Prepare image batch for model
                images_array = np.stack(data_batch['images'], axis=0)  # [S, H, W, 3]
                images_array = np.transpose(images_array, (0, 3, 1, 2))  # [S, 3, H, W]
                images_array = images_array / 255.0 if images_array.max() > 1.0 else images_array
                
                intrinsics_array = np.array(data_batch['intrinsics'])
                
                logger.info(f"Running model inference on {len(images_array)} images...")
                pred_extrinsics_batch = extract_predicted_poses(
                    model, 
                    images_array,
                    intrinsics_array,
                    avg_scale=avg_scale,
                    device=device
                )
                
                # Extract first batch if batch dimension exists
                if pred_extrinsics_batch.ndim == 4:  # [B, S, 3, 4]
                    pred_extrinsics = pred_extrinsics_batch[0]  # [S, 3, 4]
                else:  # [S, 3, 4]
                    pred_extrinsics = pred_extrinsics_batch
                
                pred_intrinsics = np.array(data_batch['intrinsics'])
                logger.info(f"Predicted extrinsics shape: {pred_extrinsics.shape}")
                
                # CRITICAL: Understanding scale mismatch
                # 
                # Root cause of magnitude differences:
                # The GT data is scaled by 1/avg_scale, but model doesn't know about this!
                #
                # Two possible scenarios:
                # 
                # SCENARIO A: Model trained WITHOUT seeing avg_scale
                # - GT: scaled (translations ~ 1e-4 to 1e-3)
                # - Model predicts: unscaled (translations ~ 1-1000)
                # - Solution: Divide model predictions by avg_scale
                #
                # SCENARIO B: Model trained WITH avg_scale in loss (ideal)
                # - Trainer applies avg_scale to loss computation
                # - Both GT and model outputs are in same normalized space
                # - Solution: No scaling needed
                #
                # SCENARIO C: Model trained on non-normalized data
                # - Model predicts raw physical scale
                # - Solution: Divide model predictions by avg_scale
                #
                # We'll check the magnitude difference to diagnose:
                
                logger.info(f"\n=== DIAGNOSING SCALE MISMATCH ===")
                logger.info(f"Scale factor: {avg_scale:.4f}")
                
                gt_mag_before = np.linalg.norm(gt_extrinsics[0][:3, 3])
                pred_mag_before = np.linalg.norm(pred_extrinsics[0][:3, 3])
                
                logger.info(f"\nGT magnitude (as returned from data loader): {gt_mag_before:.6f}")
                logger.info(f"Model magnitude (raw prediction): {pred_mag_before:.6f}")
                logger.info(f"Ratio (pred/gt): {pred_mag_before/gt_mag_before if gt_mag_before > 0 else 'inf':.2f}x")
                logger.info(f"Ratio (pred/avg_scale): {pred_mag_before/avg_scale:.6f}")
                
                # Check if dividing by avg_scale brings them to same magnitude
                pred_mag_scaled = pred_mag_before / avg_scale
                logger.info(f"\nAfter dividing model by avg_scale:")
                logger.info(f"  Model magnitude: {pred_mag_scaled:.6f}")
                logger.info(f"  GT magnitude:    {gt_mag_before:.6f}")
                logger.info(f"  Ratio: {pred_mag_scaled/gt_mag_before if gt_mag_before > 0 else 'inf':.2f}x")
                
                if abs(pred_mag_scaled - gt_mag_before) / max(pred_mag_scaled, gt_mag_before) < 0.1:
                    logger.info(f"\n✓ DIAGNOSIS: Model predicts physical scale, need to divide by avg_scale")
                elif abs(pred_mag_before - gt_mag_before) / max(pred_mag_before, gt_mag_before) < 0.1:
                    logger.info(f"\n✓ DIAGNOSIS: Magnitudes already match, no scaling needed")
                else:
                    logger.warning(f"\n⚠ DIAGNOSIS: Magnitude mismatch NOT explained by avg_scale")
                    logger.warning(f"  This suggests model may not be trained for pose prediction")
                
                # Apply the most likely correction
                if pred_mag_before / avg_scale > gt_mag_before * 0.1:  # If scaled version makes sense
                    logger.info(f"\nApplying scaling correction: dividing model predictions by {avg_scale:.4f}")
                    for i in range(len(pred_extrinsics)):
                        pred_extrinsics[i][:3, 3] = pred_extrinsics[i][:3, 3] / avg_scale
                
                logger.info(f"=========================================\n")
                
            except Exception as e:
                logger.error(f"Error running model: {e}")
                import traceback
                traceback.print_exc()
                pred_extrinsics = None

        if pred_extrinsics is None:
            logger.info(f"Generating dummy predictions with noise level {args.noise_level}...")
            pred_extrinsics, pred_intrinsics = create_dummy_predictions(
                gt_extrinsics, gt_intrinsics, noise_level=args.noise_level
            )
        
        # Create visualization
        
        # Compare rotational error
        logger.info("="*80)
        logger.info("DETAILED PER-FRAME ROTATION ERROR ANALYSIS")
        logger.info("="*80)
        
        # Debug: Print GT and pred shapes
        logger.info(f"GT extrinsics shape: {gt_extrinsics.shape}")
        logger.info(f"Predicted extrinsics shape: {pred_extrinsics.shape}")
        
        # Detailed per-frame comparison
        logger.info("\nPer-frame rotation matrix comparison:")
        logger.info("-" * 80)
        
        errors = []
        for i, (gt_e, pred_e) in enumerate(zip(gt_extrinsics, pred_extrinsics)):
            gt_R = gt_e[:3, :3]
            pred_R = pred_e[:3, :3]
            gt_t = gt_e[:3, 3]
            pred_t = pred_e[:3, 3]
            
            # Compute rotation error
            R_diff = gt_R.T @ pred_R
            trace = np.trace(R_diff)
            cos_angle = np.clip((trace - 1) / 2, -1, 1)
            angle_error = np.arccos(cos_angle) * 180 / np.pi
            errors.append(angle_error)
            
            # Translation error
            t_error = np.linalg.norm(gt_t - pred_t)
            
            logger.info(f"\nFrame {i}:")
            logger.info(f"  GT Rotation:\n{gt_R}")
            logger.info(f"  Pred Rotation:\n{pred_R}")
            logger.info(f"  R_diff (GT^T @ Pred):\n{R_diff}")
            logger.info(f"  Trace(R_diff): {trace:.6f}")
            logger.info(f"  Rotation Error: {angle_error:.4f}°")
            logger.info(f"  GT Translation: {gt_t}")
            logger.info(f"  Pred Translation: {pred_t}")
            logger.info(f"  Translation Error: {t_error:.6f} m")
        
        errors = np.array(errors)
        
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)
        logger.info(f"Rotation errors per frame (degrees): {errors}")
        logger.info(f"Mean rotation error: {errors.mean():.4f}°")
        logger.info(f"Median rotation error: {np.median(errors):.4f}°")
        logger.info(f"Std rotation error: {errors.std():.4f}°")
        logger.info(f"Min rotation error: {errors.min():.4f}°")
        logger.info(f"Max rotation error: {errors.max():.4f}°")
        
        if errors.mean() > 10:
            logger.warning(f"\n⚠️  High rotation errors detected (mean={errors.mean():.2f}°)!")
            logger.warning("This suggests:")
            logger.warning("  1. Model predictions may not be camera poses, OR")
            logger.warning("  2. Model outputs poses in different coordinate frame, OR")
            logger.warning("  3. Model hasn't been trained on pose prediction")
        elif errors.mean() < 1:
            logger.info(f"\n✓ Excellent rotation accuracy (mean={errors.mean():.4f}°)")
        else:
            logger.info(f"\n✓ Reasonable rotation accuracy (mean={errors.mean():.4f}°)")
        
        logger.info("="*80)
        
        logger.info("Creating visualization with loss calculations...")
        fig, loss_dict = plot_pose_comparison(
            gt_extrinsics, gt_intrinsics,
            pred_extrinsics, pred_intrinsics,
            seq_name=seq_name,
            compute_losses=True
        )
        
        if loss_dict:
            logger.info(f"\n{'='*80}")
            logger.info("INTEGRATED LOSS AND ERROR ANALYSIS")
            logger.info(f"{'='*80}")
            logger.info(f"\nLoss Components:")
            logger.info(f"  Translation Loss (L1):  {loss_dict.get('loss_T', 0):.8f}")
            logger.info(f"  Rotation Loss (L1):     {loss_dict.get('loss_R', 0):.8f}")
            logger.info(f"  Focal Length Loss (L1): {loss_dict.get('loss_FL', 0):.8f}")
            logger.info(f"  Total Camera Loss:      {loss_dict.get('loss_camera', 0):.8f}")
            logger.info(f"\nInterpretation:")
            logger.info(f"  • Loss_T ≈ {loss_dict.get('loss_T', 0):.4f} → Translation error in normalized space")
            logger.info(f"  • Loss_R ≈ {loss_dict.get('loss_R', 0):.4f} → Quaternion difference")
            logger.info(f"  • Loss_FL ≈ {loss_dict.get('loss_FL', 0):.4f} → Focal length/FoV error")
            logger.info(f"{'='*80}\n")
        
        if args.save:
            save_path = args.save
            if args.interactive:
                base, ext = os.path.splitext(save_path)
                save_path = f"{base}_{seq_idx}{ext}"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
            plt.close(fig)
        else:
            if args.interactive:
                plt.show(block=False)
                print(f"Sequence {seq_idx} displayed. Press any key to continue...")
                plt.waitforbuttonpress()
                plt.close(fig)
            else:
                plt.show()
