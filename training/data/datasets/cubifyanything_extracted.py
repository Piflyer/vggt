# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import logging
import random
import numpy as np
import torch
import tqdm
import gc
import json
import glob
import array
from pathlib import Path
from PIL import Image
import tifffile
import io
from collections import OrderedDict
from typing import Optional, Dict
from dataclasses import dataclass

from data.dataset_util import *
from data.base_dataset import BaseDataset

# Add the ml-cubifyanything directory to path to import CubifyAnythingDataset
CUBIFY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../ml-cubifyanything'))
if CUBIFY_PATH not in sys.path:
    sys.path.insert(0, CUBIFY_PATH)

# Import sensor info classes from cubifyanything
from cubifyanything.sensor import SensorArrayInfo, SensorInfo, PosedSensorInfo
from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from cubifyanything.instances import Instances3D
from cubifyanything.boxes import GeneralInstance3DBoxes, BoxDOF

# Helper functions for parsing (adapted from ml-cubifyanything/cubifyanything/dataset.py)
def parse_size(data):
    # data is bytes or string
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    # Strip whitespace and brackets, then parse
    data = data.strip().strip("[]").strip()
    return tuple(int(x.strip()) for x in data.split(","))

def parse_transform_3x3(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return torch.tensor(np.array(json.loads(data)).reshape(3, 3).astype(np.float32))

def parse_transform_4x4(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return torch.tensor(np.array(json.loads(data)).reshape(4, 4).astype(np.float32))

def read_image_file(file_path, expected_size=None, channels_first=True):
    # Read image from file path
    if str(file_path).endswith('.png'):
        image = np.array(Image.open(file_path))
    elif str(file_path).endswith('.tiff') or str(file_path).endswith('.tif'):
        image = tifffile.imread(file_path)
    else:
        # Fallback for other formats
        image = np.array(Image.open(file_path))

    if expected_size is not None:
        assert (image.shape[1], image.shape[0]) == expected_size, f"Expected {expected_size}, got {(image.shape[1], image.shape[0])}"

    if channels_first and (image.ndim > 2):
        image = np.moveaxis(image, -1, 0)

    return torch.tensor(image)

def read_instances(data):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    instances_data = json.loads(data)    
    instances = Instances3D()
    
    if len(instances_data) == 0:
        # Empty.
        instances.set("gt_ids", [])
        instances.set("gt_names", [])        
        # Note: empty_box is not imported, assuming empty list is fine or handling downstream
        # For now, just return empty instances
        return instances

    instances.set("gt_ids", [bi["id"] for bi in instances_data])
    instances.set("gt_names", [bi["category"] for bi in instances_data])    
    instances.set("gt_boxes_3d", GeneralInstance3DBoxes(
            np.concatenate((
                np.array([bi["position"] for bi in instances_data]),
                np.array([bi["scale"] for bi in instances_data])), axis=-1),
            np.array([bi["R"] for bi in instances_data])))
    return instances

MM_TO_M = 1000.0

@dataclass
class MemoryConfig:
    """Configuration for memory optimization"""
    max_cache_memory_gb: float = 2.0  # Maximum cache size in GB
    enable_gc_after_batch: bool = True  # Enable garbage collection after each batch
    gc_interval: int = 10  # Run GC every N batches
    prefetch_enabled: bool = False  # Enable prefetching (future optimization)
    use_incremental_scaling: bool = True  # Use incremental world points processing
    log_memory_usage: bool = False  # Log memory stats (for debugging)


class CubifyAnythingExtractedDataset(BaseDataset):
    """
    Adapter for CubifyAnything extracted dataset format to VGGT training format.
    
    This dataset loads extracted CubifyAnything files from a directory structure
    and converts them to the format expected by VGGT training pipeline.
    
    Directory structure expected:
    DATA_ROOT/
      video_id/
        timestamp.wide/
          image.png
          depth.tiff (optional)
          k.json
          ...
        timestamp.gt/
          depth.tiff
          rt.json
          ...
    """
    
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_ROOT: str = None,
        min_num_images: int = 2,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 1,
        load_arkit_depth: bool = True,
        apply_dust3r_normalization: bool = False,
        rescale_translation_mm_to_m: bool = True,
        max_depth: float = 10.0,
        background_depth: float = -1.0,  # If > 0, fill zero/invalid depth with this value
        memory_config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize the CubifyAnythingExtractedDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DATA_ROOT (str): Path to the directory containing extracted data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_ratio (int): Range for expanding nearby image selection.
            load_arkit_depth (bool): Whether to load ARKit depth alongside GT depth.
            apply_dust3r_normalization (bool): Apply DUSt3R-style scale normalization.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.min_num_images = min_num_images
        self.load_arkit_depth = load_arkit_depth
        self.apply_dust3r_normalization = apply_dust3r_normalization
        self.rescale_translation_mm_to_m = rescale_translation_mm_to_m
        self.max_depth = max_depth
        self.background_depth = background_depth
        
        # Memory optimization
        self.memory_config = memory_config or MemoryConfig()
        self.batch_counter = 0

        if DATA_ROOT is None:
            raise ValueError("DATA_ROOT must be specified.")

        self.DATA_ROOT = Path(DATA_ROOT)
        if not self.DATA_ROOT.exists():
            raise FileNotFoundError(f"DATA_ROOT {self.DATA_ROOT} does not exist.")

        self.split = split
        if split == "train":
            self.len_train = len_train
        elif split == "val" or split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"DATA_ROOT is {self.DATA_ROOT}")

        # Scan the directory to find sequences
        self._sequence_list = []
        self._sequence_metadata = {}
        self._load_sequences()

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: CubifyAnything Extracted Data size: {len(self._video_ids)}")
        logging.info(f"{status}: CubifyAnything Extracted Data dataset length: {len(self)}")

    def _load_sequences(self):
        """Scan the directory structure to identify sequences."""
        logging.info("Scanning extracted directory for sequences...")
        
        # Assuming directory structure: DATA_ROOT/video_id/...
        # We look for directories that look like video IDs (integers)
        video_dirs = [d for d in self.DATA_ROOT.iterdir() if d.is_dir()]
        
        temp_sequences = []
        
        for video_dir in tqdm.tqdm(video_dirs, desc="Scanning sequences"):
            try:
                video_id = int(video_dir.name)
            except ValueError:
                continue
                
            # Find all wide images to count frames
            wide_dirs = list(video_dir.glob("*.wide"))
            
            if len(wide_dirs) < self.min_num_images:
                continue
                
            # Collect timestamps
            timestamps = []
            for wd in wide_dirs:
                # name is "timestamp.wide"
                try:
                    ts_str = wd.name.split('.')[0]
                    timestamps.append(int(ts_str))
                except ValueError:
                    continue
            
            timestamps.sort()
            
            if len(timestamps) >= self.min_num_images:
                temp_sequences.append((video_id, timestamps))
        
        # Sort by video_id to ensure deterministic order
        temp_sequences.sort(key=lambda x: x[0])
        
        # Use compact arrays to store metadata to save memory
        # video_ids: array of video IDs
        # timestamp_offsets: start index in timestamps_flat for each video
        # timestamps_flat: all timestamps concatenated
        self._video_ids = array.array('Q')
        self._timestamp_offsets = array.array('Q')
        self._timestamps_flat = array.array('Q')
        self._video_id_to_index = {}
        
        offset = 0
        for i, (vid, tss) in enumerate(temp_sequences):
            self._video_ids.append(vid)
            self._video_id_to_index[vid] = i
            self._timestamp_offsets.append(offset)
            self._timestamps_flat.extend(tss)
            offset += len(tss)
        self._timestamp_offsets.append(offset) # Sentinel
        
        # Keep _sequence_list as a property or just use _video_ids
        # We'll use _video_ids in get_data
        
        logging.info(f"Found {len(self._video_ids)} valid sequences.")
        
        # Capping len_train is not necessary here as BaseDataset.get_data 
        # handles sequence sampling with modulo if seq_index >= num_videos.
        # This allows multiple samplings from the same long video per epoch.
        pass

    def _get_interpolated_pose(self, video_id, timestamp_idx, timestamps):
        """
        Finds unique poses around the current timestamp and interpolates.
        This handles low-frequency SLAM data by creating a smooth trajectory.
        """
        curr_pose = self._load_raw_rt(video_id, timestamps[timestamp_idx])
        
        # Find previous unique pose
        prev_idx = timestamp_idx - 1
        prev_pose = curr_pose
        while prev_idx >= 0:
            p = self._load_raw_rt(video_id, timestamps[prev_idx])
            if not np.allclose(p, curr_pose, atol=1e-6):
                prev_pose = p
                break
            prev_idx -= 1
            
        # Find next unique pose
        next_idx = timestamp_idx + 1
        next_pose = curr_pose
        while next_idx < len(timestamps):
            p = self._load_raw_rt(video_id, timestamps[next_idx])
            if not np.allclose(p, curr_pose, atol=1e-6):
                next_pose = p
                break
            next_idx += 1
            
        if prev_idx == -1 or next_idx == len(timestamps):
            return curr_pose
            
        # Interpolate between prev_idx and next_idx
        # alpha = 0 at prev_idx, alpha = 1 at next_idx
        alpha = (timestamp_idx - prev_idx) / (next_idx - prev_idx)
        
        # Linear interpolation of translation
        t_interp = (1 - alpha) * prev_pose[:3, 3] + alpha * next_pose[:3, 3]
        
        # Spherical Linear Interpolation (SLERP) for rotation
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
        
        rots = R.from_matrix([prev_pose[:3, :3], next_pose[:3, :3]])
        slerp = Slerp([0, 1], rots)
        r_interp = slerp([alpha])[0].as_matrix()
        
        interp_pose = np.eye(4, dtype=np.float32)
        interp_pose[:3, :3] = r_interp
        interp_pose[:3, 3] = t_interp
        return interp_pose

    def _load_raw_rt(self, video_id, timestamp):
        """Loads raw 4x4 matrix from rt file."""
        video_path = Path(self.DATA_ROOT) / str(video_id)
        gt_dir = video_path / f"{timestamp}.gt"
        with open(gt_dir / "rt", 'r') as f:
            data = f.read().strip().strip('[]').split(',')
            return np.array([float(x) for x in data], dtype=np.float32).reshape(4, 4)

    def _load_sample(self, video_id, timestamp, interpolated_pose=None):
        """Load a single sample from disk."""
        # Construct paths
        # video_id/timestamp.wide/image
        # video_id/timestamp.gt/depth
        # etc.
        
        video_path = self.DATA_ROOT / str(video_id)
        ts_str = str(timestamp)
        
        # Helper to read file content (bytes or string)
        def read_file(path):
            with open(path, 'rb') as f:
                return f.read()
        
        # Helper to read text file
        def read_text(path):
            with open(path, 'r') as f:
                return f.read()

        # Paths
        wide_dir = video_path / f"{ts_str}.wide"
        gt_dir = video_path / f"{ts_str}.gt"
        
        # Check if directories exist
        if not wide_dir.exists() or not gt_dir.exists():
            raise FileNotFoundError(f"Missing data for {video_id} {timestamp}")
            
        # Load metadata
        # Note: filenames in tar might be just "image", "depth", "k", etc.
        # Or they might have extensions.
        # We need to be robust.
        
        def find_file(directory, name):
            # Try exact match
            p = directory / name
            if p.exists() and p.is_file(): return p
            # Try with extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.json', '.txt']:
                p = directory / (name + ext)
                if p.exists() and p.is_file(): return p
            
            # Case insensitive search
            if directory.exists():
                for f in directory.iterdir():
                    if f.stem.lower() == name.lower() and f.is_file():
                        return f
                    if f.name.lower() == name.lower():
                        return f
            
            # Try glob as last resort (case sensitive usually)
            matches = list(directory.glob(f"{name}*"))
            if matches: return matches[0]
            raise FileNotFoundError(f"File {name} not found in {directory}")

        # Parse metadata
        # We need to reconstruct the 'sample' dict structure expected by _map_sample logic
        # But wait, we can just construct the result dict directly!
        # The original _map_sample takes raw tar data and produces the result dict.
        # We can skip the raw tar data step and produce the result dict directly from files.
        
        # Replicating _map_sample logic:
        
        # Wide Sensor
        wide_image_path = find_file(wide_dir, "image")
        wide_image_size_path = find_file(video_path / f"{ts_str}._wide" / "image", "size") # Note: _wide is separate dir in tar structure?
        # Wait, the tar structure in dataset.py keys: "_wide/image/size"
        # This implies a directory "_wide" at the same level as "wide".
        # Let's assume extraction preserved this.
        
        _wide_dir = video_path / f"{ts_str}._wide"
        _gt_dir = video_path / f"{ts_str}._gt"
        
        # Load sizes
        wide_image_size = parse_size(read_text(find_file(_wide_dir / "image", "size")))
        
        # Load Intrinsics
        wide_k = parse_transform_3x3(read_text(find_file(wide_dir / "image", "k")))
        
        wide = PosedSensorInfo()
        wide.RT = torch.eye(4)[None]
        wide.image = ImageMeasurementInfo(size=wide_image_size, K=wide_k[None])
        
        if self.load_arkit_depth:
            wide_depth_size = parse_size(read_text(find_file(_wide_dir / "depth", "size")))
            wide_depth_k = parse_transform_3x3(read_text(find_file(wide_dir / "depth", "k")))
            wide.depth = DepthMeasurementInfo(size=wide_depth_size, K=wide_depth_k[None])
            
        wide.T_gravity = parse_transform_3x3(read_text(find_file(wide_dir, "t_gravity")))[None]

        # GT Sensor - Use interpolated pose if provided
        # GT Sensor - Use interpolated pose if provided
        if interpolated_pose is not None:
            gt_rt = torch.from_numpy(interpolated_pose).float()[None]
        else:
            gt_rt = parse_transform_4x4(read_text(find_file(gt_dir, "rt")))[None]
            
        if self.rescale_translation_mm_to_m:
            gt_rt[:, :3, 3] = gt_rt[:, :3, 3] / MM_TO_M
        gt_depth_size = parse_size(read_text(find_file(_gt_dir / "depth", "size")))
        gt_depth_k = parse_transform_3x3(read_text(find_file(gt_dir / "depth", "k")))
        
        gt = PosedSensorInfo()
        gt.RT = gt_rt
        gt.depth = DepthMeasurementInfo(size=gt_depth_size, K=gt_depth_k[None])

        sensor_info = SensorArrayInfo()
        sensor_info.wide = wide
        sensor_info.gt = gt

        # Load Images and Depths
        wide_image = read_image_file(wide_image_path, expected_size=wide.image.size)[None]
        
        # Instances (optional, usually empty for training)
        # wide_instances = read_instances(read_text(find_file(wide_dir, "instances")))
        
        gt_depth_path = find_file(gt_dir, "depth")
        gt_depth = read_image_file(gt_depth_path, expected_size=gt.depth.size, channels_first=True)[None].float() / MM_TO_M
        
        result = dict(
            sensor_info=sensor_info,
            wide=dict(
                image=wide_image,
                # instances=wide_instances
            ),
            gt=dict(
                depth=gt_depth
            ),
            meta=dict(video_id=video_id, timestamp=float(timestamp) / 1e9)
        )

        if self.load_arkit_depth:
            wide_depth_path = find_file(wide_dir, "depth")
            result["wide"]["depth"] = read_image_file(wide_depth_path, expected_size=wide.depth.size, channels_first=True)[None].float() / MM_TO_M

        return result

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, len(self._video_ids) - 1)

        if seq_name is None:
            if len(self._video_ids) > 0:
                seq_index = seq_index % len(self._video_ids)
            video_id = self._video_ids[seq_index]
        else:
            video_id = int(seq_name) if isinstance(seq_name, str) else seq_name

        # Get timestamps for this video using compact arrays
        idx = self._video_id_to_index.get(video_id)
        if idx is None:
             raise ValueError(f"No timestamps found for video {video_id}")
        
        start_offset = self._timestamp_offsets[idx]
        end_offset = self._timestamp_offsets[idx+1]
        timestamps = self._timestamps_flat[start_offset:end_offset].tolist()

        if not timestamps:
             raise ValueError(f"No timestamps found for video {video_id}")

        num_images = len(timestamps)

        if ids is None:
            # Sample frames logic (same as original)
            max_distance = 30
            max_sequence_span = img_per_seq * max_distance
            max_start = max(0, num_images - max_sequence_span)
            start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            ids = [start_frame]
            current_pos = start_frame
            
            while len(ids) < img_per_seq and current_pos < num_images - 1:
                min_next = current_pos + 1
                max_next = min(current_pos + max_distance, num_images - 1)
                
                if min_next <= max_next:
                    next_frame = np.random.randint(min_next, max_next + 1)
                    ids.append(next_frame)
                    current_pos = next_frame
                else:
                    break
            
            if len(ids) < img_per_seq:
                available_indices = list(range(0, num_images))
                remaining_needed = img_per_seq - len(ids)
                step_size = max(1, (num_images - 1) // remaining_needed) if remaining_needed > 0 else 1
                step_size = min(step_size, max_distance)
                candidate_frames = list(range(1, num_images, step_size))
                
                if len(candidate_frames) >= remaining_needed:
                    additional_ids = np.random.choice(candidate_frames, remaining_needed, replace=False)
                    ids.extend(additional_ids.tolist())
                else:
                    additional_ids = np.random.choice(candidate_frames if candidate_frames else [1], remaining_needed, replace=True)
                    ids.extend(additional_ids.tolist())
            
            ids = list(ids[:img_per_seq])
            ids = np.array(ids, dtype=int)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        
        ids_list = list(ids)
        max_attempts = max(img_per_seq * 2, len(ids_list) * 3)
        attempt_count = 0
        
        # Load first frame for normalization reference
        # Note: ids contains indices into the timestamps list
        num_timestamps = len(timestamps)
        first_frame_idx = ids_list[0]
        
        if first_frame_idx >= num_timestamps:
            logging.warning(f"Requested frame index {first_frame_idx} is out of range for sequence {video_id} (max {num_timestamps-1}). Wrapping around.")
            first_frame_idx = first_frame_idx % num_timestamps
            
        first_timestamp = timestamps[first_frame_idx]
        first_sample = self._load_sample(video_id, first_timestamp)
        
        first_extrinsic = first_sample['sensor_info'].gt.RT[0].numpy()
        inv_first_extrinsic = np.linalg.inv(first_extrinsic)
        
        # Clean up first sample if not needed immediately (will be reloaded in loop if idx is 0)
        # Actually, we can cache it or just reload. Reloading is safer for loop logic simplicity.
        del first_sample

        while len(images) < img_per_seq and attempt_count < max_attempts:
            raw_idx = ids_list[attempt_count % len(ids_list)]
            frame_idx = raw_idx % num_timestamps
            attempt_count += 1
            
            timestamp = timestamps[frame_idx]
            try:
                # Calculate interpolated pose to handle low-frequency SLAM data
                interp_pose = self._get_interpolated_pose(video_id, frame_idx, timestamps)
                sample = self._load_sample(video_id, timestamp, interpolated_pose=interp_pose)
            except Exception as e:
                logging.warning(f"Failed to load sample {video_id} {timestamp}: {e}")
                continue
            
            # Extract image
            image_tensor = sample['wide']['image'][0]
            if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3, 4]:
                image = image_tensor.permute(1, 2, 0).numpy()
            else:
                image = image_tensor.numpy()
            
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            depth_tensor = sample['gt']['depth'][0]
            depth_map = depth_tensor.numpy().astype(np.float32)
            
            del image_tensor
            del depth_tensor
            
            if self.background_depth > 0:
                # Fill zeros (invalid depth) with background depth before thresholding
                depth_map[depth_map <= 1e-8] = self.background_depth

            depth_map = threshold_depth_map(
                depth_map, 
                max_percentile=98, 
                min_percentile=-1,
                max_depth=self.max_depth
            )

            original_size = np.array(image.shape[:2])

            sensor_info = sample['sensor_info']
            extri_4x4 = sensor_info.gt.RT[0].numpy()
            extri_opencv = extri_4x4[:3, :]
            intri_opencv = sensor_info.wide.image.K[0].numpy()

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=f"video_{video_id}_frame_{frame_idx}",
            )

            if (image.shape[:2] != target_image_shape).any():
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        if len(images) == 0:
            # Fallback logic
            logging.error(f"No valid images for {video_id}")
            # Simple recursion fallback
            return self.get_data(seq_index=(seq_index + 1) % len(self._sequence_list), img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)
        
        if len(images) < img_per_seq:
            original_count = len(images)
            while len(images) < img_per_seq:
                idx_to_dup = (len(images) - original_count) % original_count
                images.append(images[idx_to_dup])
                depths.append(depths[idx_to_dup])
                extrinsics.append(extrinsics[idx_to_dup])
                intrinsics.append(intrinsics[idx_to_dup])
                cam_points.append(cam_points[idx_to_dup])
                world_points.append(world_points[idx_to_dup])
                point_masks.append(point_masks[idx_to_dup])
                original_sizes.append(original_sizes[idx_to_dup])

        # === Coordinate Frame Normalization (Relative to First Frame) ===
        if len(extrinsics) > 0:
            # extrinsics currently contains raw C_from_W (W2C) matrices from the rt files
            raw_extrinsics_4x4 = []
            for extri_3x4 in extrinsics:
                m = np.eye(4, dtype=np.float32)
                m[:3, :] = extri_3x4
                raw_extrinsics_4x4.append(m)
            
            # To make the first frame Identity, we multiply all W2C matrices by the inverse of the first W2C
            # Result: Camera_i_from_Camera_0 = Camera_i_from_World @ World_from_Camera_0
            first_extri_raw = raw_extrinsics_4x4[0]
            first_extri_raw_inv = np.linalg.inv(first_extri_raw)
            
            final_extrinsics = []
            for raw_extri in raw_extrinsics_4x4:
                norm_extri = raw_extri @ first_extri_raw_inv
                final_extrinsics.append(norm_extri[:3, :])
            
            extrinsics = final_extrinsics
            
            # Normalize world points to the first frame's coordinate system
            # p_0 = Camera_0_from_World @ p_world
            normalized_world_points = []
            for wp in world_points:
                h, w = wp.shape[:2]
                wp_homog = np.ones((h, w, 4), dtype=np.float32)
                wp_homog[..., :3] = wp
                # Apply first_extri_raw: [4, 4] @ [H, W, 4] -> [H, W, 4]
                wp_norm = np.einsum('ij,hwj->hwi', first_extri_raw, wp_homog)
                normalized_world_points.append(wp_norm[..., :3])
            
            world_points = normalized_world_points
        
        avg_scale = 1.0
        if self.apply_dust3r_normalization and len(world_points) > 0 and len(point_masks) > 0:
            world_points_array = np.stack(world_points, axis=0)
            point_masks_array = np.stack(point_masks, axis=0)
            distances = np.linalg.norm(world_points_array, axis=-1)
            valid_distances = distances * point_masks_array
            distance_sum = valid_distances.sum()
            valid_count = point_masks_array.sum()
            
            if valid_count > 0:
                avg_scale = distance_sum / valid_count
                avg_scale = np.clip(avg_scale, 1e-6, 1e6)
                
                world_points = [wp / avg_scale for wp in world_points]
                cam_points = [cp / avg_scale for cp in cam_points]
                depths = [d / avg_scale for d in depths]
                extrinsics = [
                    np.concatenate([e[:3, :3], e[:3, 3:4] / avg_scale], axis=1)
                    for e in extrinsics
                ]
                logging.info(f"Applied DUSt3R-style scale normalization: avg_scale={avg_scale:.4f}")
            else:
                avg_scale = 1.0

        set_name = "cubifyanything"
        batch = {
            "seq_name": f"{set_name}_video_{video_id}",
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "avg_scale": avg_scale,
        }
        
        if self.memory_config.enable_gc_after_batch:
            self.batch_counter += 1
            if self.batch_counter % self.memory_config.gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return batch
