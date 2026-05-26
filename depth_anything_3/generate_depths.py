import glob
import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3
import onnxruntime

# Add vggt to path to import visual_util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from vggt.visual_util import segment_sky, download_file_from_url
except ImportError:
    # Fallback if the path structure is different
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../vggt")))
    from vggt.visual_util import segment_sky, download_file_from_url

def save_k_matrix(path, k):
    """Saves the 3x3 K matrix as a flattened list in a text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k_flat = k.flatten().tolist()
    with open(path, 'w') as f:
        f.write(str(k_flat))

def process_dataset(root_dir, model_name="depth-anything/DA3NESTED-GIANT-LARGE", batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Sky Segmentation
    skyseg_path = "skyseg.onnx"
    if not os.path.exists(skyseg_path):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", skyseg_path
        )
    
    print("Loading sky segmentation model...")
    skyseg_session = onnxruntime.InferenceSession(skyseg_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Load Depth model
    print(f"Loading model {model_name}...")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.eval()

    # Find all .wide directories
    print(f"Searching for .wide folders in {root_dir}...")
    wide_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for d in dirnames:
            if d.endswith(".wide"):
                wide_dirs.append(os.path.join(dirpath, d))
    
    print(f"Found {len(wide_dirs)} .wide folders.")
    
    # Filter for folders that have image.png
    valid_tasks = []
    for wide_dir in wide_dirs:
        img_path = os.path.join(wide_dir, "image.png")
        if os.path.exists(img_path):
            # Derive the corresponding .gt directory
            gt_dir = wide_dir.replace(".wide", ".gt")
            valid_tasks.append({
                "wide_dir": wide_dir,
                "gt_dir": gt_dir,
                "img_path": img_path
            })
    
    # Process in batches
    print(f"Processing {len(valid_tasks)} images with Sky Masking...")
    
    pbar = tqdm(total=len(valid_tasks), desc="Generating Depths", dynamic_ncols=True)
    for i in range(0, len(valid_tasks), batch_size):
        batch = valid_tasks[i : i + batch_size]
        img_paths = [t["img_path"] for t in batch]
        
        # Inference
        try:
            prediction = model.inference(img_paths)
        except Exception as e:
            print(f"\nError during inference for batch starting at {i}: {e}")
            pbar.update(len(batch))
            continue

        # Target resolution
        target_w, target_h = 640, 480

        # Save results
        for j, task in enumerate(batch):
            depth = prediction.depth[j]  # [H, W] float32 (meters)
            intrinsics = prediction.intrinsics[j].copy()  # [3, 3] float32
            
            # 1. Apply Sky Masking to clean up horizon artifacts
            # Generate sky mask (255 = sky, 0 = non-sky based on visual_util logic)
            # We save the mask file in the .wide folder for future use/debugging
            mask_path = os.path.join(task["wide_dir"], "sky_mask.png")
            sky_mask = segment_sky(task["img_path"], skyseg_session, mask_filename=mask_path)
            
            # Ensure sky_mask matches the depth shape before masking
            if sky_mask.shape[:2] != depth.shape[:2]:
                sky_mask = cv2.resize(sky_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # If mask is < 128 (sky), set depth to background distance (50m)
            # This prevents the model from trying to learn metric depth for the infinite sky
            depth[sky_mask < 128] = 50.0

            # 2. Resize depth to target resolution
            if (depth.shape[1], depth.shape[0]) != (target_w, target_h):
                # Scale intrinsics to match target resolution
                scale_w = target_w / depth.shape[1]
                scale_h = target_h / depth.shape[0]
                intrinsics[0, 0] *= scale_w  # fx
                intrinsics[1, 1] *= scale_h  # fy
                intrinsics[0, 2] *= scale_w  # cx
                intrinsics[1, 2] *= scale_h  # cy
                
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Convert depth to uint16 (mm) for saving
            depth_mm = (depth * 1000).astype(np.uint16)
            
            # Paths for .wide
            wide_depth_png = os.path.join(task["wide_dir"], "depth.png")
            wide_k_path = os.path.join(task["wide_dir"], "depth", "k")
            
            # Paths for .gt
            gt_depth_png = os.path.join(task["gt_dir"], "depth.png")
            gt_k_path = os.path.join(task["gt_dir"], "depth", "k")
            
            # Save depth images
            cv2.imwrite(wide_depth_png, depth_mm)
            if os.path.exists(task["gt_dir"]):
                cv2.imwrite(gt_depth_png, depth_mm)
            else:
                os.makedirs(task["gt_dir"], exist_ok=True)
                cv2.imwrite(gt_depth_png, depth_mm)
            
            # Save K matrices
            save_k_matrix(wide_k_path, intrinsics)
            save_k_matrix(gt_k_path, intrinsics)
        
        pbar.update(len(batch))
    
    pbar.close()

if __name__ == "__main__":
    dataset_path = "/home/tim/vggt-cubify/ml-cubifyanything/data/extracted/rosbag_subset_fixed_poses"
    process_dataset(dataset_path)
