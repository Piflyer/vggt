
import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf

# Add training directory to path to import datasets
sys.path.append(os.path.abspath("vggt/training"))

from data.datasets.cubifyanything import CubifyAnythingDataset
from data.datasets.cubifyanything_extracted import CubifyAnythingExtractedDataset

def compare_batches(batch1, batch2, path=""):
    """Recursively compare two batches."""
    if isinstance(batch1, dict):
        if not isinstance(batch2, dict):
            print(f"Type mismatch at {path}: {type(batch1)} vs {type(batch2)}")
            return False
        
        keys1 = set(batch1.keys())
        keys2 = set(batch2.keys())
        
        if keys1 != keys2:
            print(f"Key mismatch at {path}: {keys1} vs {keys2}")
            return False
            
        match = True
        for k in keys1:
            if not compare_batches(batch1[k], batch2[k], path=f"{path}.{k}" if path else k):
                match = False
        return match
        
    elif isinstance(batch1, (list, tuple)):
        if not isinstance(batch2, (list, tuple)):
            print(f"Type mismatch at {path}: {type(batch1)} vs {type(batch2)}")
            return False
            
        if len(batch1) != len(batch2):
            print(f"Length mismatch at {path}: {len(batch1)} vs {len(batch2)}")
            return False
            
        match = True
        for i, (item1, item2) in enumerate(zip(batch1, batch2)):
            if not compare_batches(item1, item2, path=f"{path}[{i}]"):
                match = False
        return match
        
    elif isinstance(batch1, (torch.Tensor, np.ndarray)):
        if isinstance(batch1, torch.Tensor):
            val1 = batch1.cpu().numpy()
        else:
            val1 = batch1
            
        if isinstance(batch2, torch.Tensor):
            val2 = batch2.cpu().numpy()
        else:
            val2 = batch2
            
        if val1.shape != val2.shape:
            print(f"Shape mismatch at {path}: {val1.shape} vs {val2.shape}")
            return False
            
        # Allow small floating point differences
        if np.issubdtype(val1.dtype, np.floating):
            if not np.allclose(val1, val2, rtol=1e-4, atol=1e-4):
                diff = np.abs(val1 - val2)
                print(f"Value mismatch at {path}: max diff {diff.max()}")
                return False
        else:
            if not np.array_equal(val1, val2):
                # For images, sometimes decoding differences might occur if libraries differ, 
                # but here we use PIL/tifffile in both (hopefully).
                # CubifyAnythingDataset uses PIL/tifffile via read_image_bytes.
                # Extracted uses PIL/tifffile via read_image_file.
                # Should be identical if same bytes.
                diff = np.abs(val1.astype(float) - val2.astype(float))
                if diff.max() > 0:
                     print(f"Value mismatch at {path}: max diff {diff.max()}")
                     return False
        return True
        
    else:
        if batch1 != batch2:
            print(f"Value mismatch at {path}: {batch1} vs {batch2}")
            return False
        return True

def main():
    # Mock common config
    common_conf = OmegaConf.create({
        "debug": False,
        "training": False,
        "get_nearby": False,
        "inside_random": False,
        "allow_duplicate_img": True,
        "img_size": 392,
        "patch_size": 14,
        "rescale": True,
        "rescale_aug": False,
        "landscape_check": True,
        "augs": {
            "aspects": [1.0, 1.0],
            "scales": [1.0, 1.0]
        }
    })

    print("Initializing Original Dataset...")
    # Point to the specific tar file we extracted
    tar_path = os.path.abspath("ml-cubifyanything/data/val/ca1m-val-47331311.tar")
    ds_orig = CubifyAnythingDataset(
        common_conf=common_conf,
        split="test",
        CUBIFY_URL=f"file://{tar_path}",
        min_num_images=2,
        len_train=100,
        len_test=100,
        lazy_load=True,
        load_arkit_depth=True
    )

    print("Initializing Extracted Dataset...")
    extracted_path = os.path.abspath("ml-cubifyanything/data/extracted/val")
    ds_extracted = CubifyAnythingExtractedDataset(
        common_conf=common_conf,
        split="test",
        DATA_ROOT=extracted_path,
        min_num_images=2,
        len_train=100,
        len_test=100,
        load_arkit_depth=True
    )

    video_id = 47331311
    print(f"Comparing data for video_id: {video_id}")

    # We need to ensure we pick the same frames.
    # Let's ask for specific indices.
    # Note: The datasets might sort frames differently if timestamps are not handled identically,
    # but both sort by timestamp.
    
    # Let's get data
    # We pass ids=[0, 1] to get the first two frames (sorted by timestamp)
    ids = [0, 1]
    img_per_seq = 2
    
    print("Getting data from Original Dataset...")
    batch_orig = ds_orig.get_data(
        seq_name=str(video_id),
        img_per_seq=img_per_seq,
        ids=ids,
        aspect_ratio=1.0
    )

    print("Getting data from Extracted Dataset...")
    batch_extracted = ds_extracted.get_data(
        seq_name=str(video_id),
        img_per_seq=img_per_seq,
        ids=ids,
        aspect_ratio=1.0
    )

    print("\n--- Comparison Results ---")
    if compare_batches(batch_orig, batch_extracted):
        print("SUCCESS: Batches are identical!")
    else:
        print("FAILURE: Batches differ.")

if __name__ == "__main__":
    main()
