#!/usr/bin/env python3
"""
Visualization tool for camera poses from CubifyAnything dataset.

This tool loads sequences one at a time and visualizes:
1. Original camera poses (before normalization)
2. Normalized camera poses (after normalization)
3. Camera frustums in 3D space
4. Comparison of the two coordinate frames
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
import argparse
import logging

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


class CommonConf:
    """Simple configuration object for CubifyAnythingDataset."""
    def __init__(self, cfg):
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.debug = False
        self.training = True
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


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib visualization."""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        return self.set_positions((x1, y1), (x1 + dx, y1 + dy))

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        xs = np.array([x1, x1 + dx])
        ys = np.array([y1, y1 + dy])
        zs = np.array([z1, z1 + dz])
        
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def get_camera_center(extrinsic_3x4):
    """
    Get camera center in world coordinates from extrinsic matrix.
    
    Args:
        extrinsic_3x4: [3, 4] camera-from-world matrix
    
    Returns:
        camera_center: [3] position in world coordinates
    """
    # Convert to 4x4
    extri_4x4 = np.eye(4, dtype=np.float32)
    extri_4x4[:3, :] = extrinsic_3x4
    
    # world_from_camera = inverse(camera_from_world)
    w_from_c = np.linalg.inv(extri_4x4)
    
    # Camera center is the translation part of world_from_camera
    camera_center = w_from_c[:3, 3]
    
    return camera_center


def get_camera_frustum(extrinsic_3x4, intrinsic_3x3, scale=0.1, depth=0.3):
    """
    Get camera frustum points for visualization.
    
    Args:
        extrinsic_3x4: [3, 4] camera-from-world matrix
        intrinsic_3x3: [3, 3] camera intrinsics
        scale: scale factor for frustum size
        depth: depth of the frustum
    
    Returns:
        frustum_points_world: [8, 3] frustum corner points in world coordinates
    """
    # Get inverse matrices
    extri_4x4 = np.eye(4, dtype=np.float32)
    extri_4x4[:3, :] = extrinsic_3x4
    w_from_c = np.linalg.inv(extri_4x4)
    
    # Camera intrinsics
    fx = intrinsic_3x3[0, 0]
    fy = intrinsic_3x3[1, 1]
    cx = intrinsic_3x3[0, 2]
    cy = intrinsic_3x3[1, 2]
    
    # Frustum corners in camera space
    # Near plane (z=0.1)
    frustum_camera = np.array([
        [0, 0, 0],  # Camera center
        [-(cx) / fx * depth * scale, -(cy) / fy * depth * scale, depth],  # top-left
        [(fx - cx) / fx * depth * scale, -(cy) / fy * depth * scale, depth],  # top-right
        [(fx - cx) / fx * depth * scale, (fy - cy) / fy * depth * scale, depth],  # bottom-right
        [-(cx) / fx * depth * scale, (fy - cy) / fy * depth * scale, depth],  # bottom-left
    ])
    
    # Transform to world coordinates
    frustum_homog = np.concatenate(
        [frustum_camera, np.ones((frustum_camera.shape[0], 1))],
        axis=1
    )  # [5, 4]
    
    frustum_world = (w_from_c @ frustum_homog.T).T[:, :3]  # [5, 3]
    
    return frustum_world


def plot_camera_poses(dataset, seq_index=0, img_per_seq=5):
    """
    Plot and compare original vs normalized camera poses.
    
    Args:
        dataset: CubifyAnythingDataset instance
        seq_index: which sequence to visualize
        img_per_seq: number of images per sequence
    """
    # Get the sequence video_id
    video_id = dataset._sequence_list[seq_index]
    logger.info(f"Visualizing sequence index {seq_index}: video_id={video_id}")
    
    # Get raw data without normalization by directly accessing the frames
    frames = dataset._video_sequences[video_id]
    num_images = len(frames)
    logger.info(f"Sequence has {num_images} frames")
    
    # Sample frames
    step_size = max(1, num_images // img_per_seq)
    frame_indices = list(range(0, num_images, step_size))[:img_per_seq]
    
    # Extract original extrinsics and intrinsics
    original_extrinsics = []
    original_intrinsics = []
    
    for frame_idx in frame_indices:
        sample = frames[frame_idx]
        sensor_info = sample['sensor_info']
        
        extri_4x4 = sensor_info.gt.RT[0].numpy()
        extri_3x4 = extri_4x4[:3, :]
        intri_3x3 = sensor_info.gt.depth.K[0].numpy()
        
        original_extrinsics.append(extri_3x4)
        original_intrinsics.append(intri_3x3)
    
    # Get data through the dataset loader (which applies normalization)
    data_batch = dataset.get_data(
        seq_index=seq_index,
        img_per_seq=img_per_seq,
        aspect_ratio=1.0
    )
    
    normalized_extrinsics = data_batch['extrinsics'][:len(frame_indices)]
    normalized_intrinsics = data_batch['intrinsics'][:len(frame_indices)]
    
    logger.info(f"Loaded {len(normalized_extrinsics)} frames through dataset")
    
    # Create figure with 3D subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Original camera poses
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_camera_trajectory(
        ax1, 
        original_extrinsics, 
        original_intrinsics,
        title=f"Original Camera Poses (video_id={video_id})",
        color='blue'
    )
    
    # Subplot 2: Normalized camera poses
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_camera_trajectory(
        ax2,
        normalized_extrinsics,
        normalized_intrinsics,
        title="Normalized Camera Poses (First Frame = Identity)",
        color='red'
    )
    
    # Subplot 3: Extrinsics comparison - translation
    ax3 = fig.add_subplot(2, 2, 3)
    plot_translation_comparison(
        ax3,
        original_extrinsics,
        normalized_extrinsics,
        frame_indices
    )
    
    # Subplot 4: First frame comparison
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot original first frame
    frustum_orig = get_camera_frustum(
        original_extrinsics[0],
        original_intrinsics[0]
    )
    ax4.scatter(*frustum_orig.T, c='blue', s=50, label='Original Frame 0', alpha=0.7)
    
    # Plot normalized first frame
    frustum_norm = get_camera_frustum(
        normalized_extrinsics[0],
        normalized_intrinsics[0]
    )
    ax4.scatter(*frustum_norm.T, c='red', s=50, label='Normalized Frame 0', alpha=0.7)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('First Frame Frustum Comparison')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig


def plot_camera_trajectory(ax, extrinsics, intrinsics, title="Camera Trajectory", color='blue'):
    """Plot camera trajectory in 3D space."""
    camera_centers = []
    
    for extri, intri in zip(extrinsics, intrinsics):
        center = get_camera_center(extri)
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    
    # Plot camera centers
    ax.scatter(*camera_centers.T, c=color, s=100, marker='o', label='Camera centers')
    
    # Plot trajectory line
    ax.plot(*camera_centers.T, c=color, alpha=0.5, linewidth=2)
    
    # Plot frustums for first and last frame
    for i, (extri, intri) in enumerate(zip(extrinsics, intrinsics)):
        if i == 0 or i == len(extrinsics) - 1:
            frustum = get_camera_frustum(extri, intri)
            ax.scatter(*frustum.T, c=color, s=30, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.grid(True)
    
    # Set equal aspect ratio
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


def plot_translation_comparison(ax, original_extrinsics, normalized_extrinsics, frame_indices):
    """Compare translation vectors between original and normalized poses."""
    original_translations = []
    normalized_translations = []
    
    for orig_extri, norm_extri in zip(original_extrinsics, normalized_extrinsics):
        orig_t = orig_extri[:3, 3]
        norm_t = norm_extri[:3, 3]
        
        original_translations.append(np.linalg.norm(orig_t))
        normalized_translations.append(np.linalg.norm(norm_t))
    
    frame_range = range(len(frame_indices))
    
    ax.plot(frame_range, original_translations, 'o-', label='Original', color='blue', linewidth=2)
    ax.plot(frame_range, normalized_translations, 's-', label='Normalized', color='red', linewidth=2)
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Translation Magnitude (m)')
    ax.set_title('Camera Translation Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)


def interactive_viewer(dataset_config_path):
    """
    Interactive viewer for cycling through sequences.
    
    Args:
        dataset_config_path: path to Hydra config
    """
    # Load Hydra config
    config_dir = os.path.dirname(os.path.abspath(dataset_config_path))
    config_name = os.path.basename(dataset_config_path).replace('.yaml', '')
    
    logger.info(f"Loading config from {config_dir}/{config_name}")
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    
    # Create a simple common_conf object
    class CommonConf:
        def __init__(self):
            self.debug = False
            self.training = True
            self.get_nearby = False
            self.inside_random = False
            self.allow_duplicate_img = True
    
    # Create dataset
    logger.info("Creating CubifyAnythingDataset...")
    cubify_config = cfg.data.train.dataset.dataset_configs[0]
    dataset = CubifyAnythingDataset(
        common_conf=CommonConf(cfg),
        split=cubify_config.split,
        CUBIFY_URL=cubify_config.CUBIFY_URL
    )
    
    logger.info(f"Dataset created with {len(dataset._sequence_list)} sequences")
    
    # Interactive loop
    seq_idx = 0
    img_per_seq = 5
    
    while True:
        print("\n" + "="*60)
        print(f"Sequence {seq_idx} / {len(dataset._sequence_list)}")
        print("="*60)
        
        try:
            fig = plot_camera_poses(dataset, seq_index=seq_idx, img_per_seq=img_per_seq)
            plt.show()
        except Exception as e:
            logger.error(f"Error visualizing sequence {seq_idx}: {e}")
            import traceback
            traceback.print_exc()
        
        # Get user input
        user_input = input(
            "\nOptions:\n"
            "  'n' - Next sequence\n"
            "  'p' - Previous sequence\n"
            "  'j <num>' - Jump to sequence <num>\n"
            "  'i <num>' - Set images per sequence to <num>\n"
            "  'q' - Quit\n"
            "> "
        ).strip()
        
        if user_input.lower() == 'n':
            seq_idx = (seq_idx + 1) % len(dataset._sequence_list)
        elif user_input.lower() == 'p':
            seq_idx = (seq_idx - 1) % len(dataset._sequence_list)
        elif user_input.lower().startswith('j'):
            try:
                seq_idx = int(user_input.split()[1])
                seq_idx = max(0, min(seq_idx, len(dataset._sequence_list) - 1))
            except (ValueError, IndexError):
                logger.warning("Invalid input for 'j' command")
        elif user_input.lower().startswith('i'):
            try:
                img_per_seq = int(user_input.split()[1])
            except (ValueError, IndexError):
                logger.warning("Invalid input for 'i' command")
        elif user_input.lower() == 'q':
            break
        else:
            logger.warning(f"Unknown command: {user_input}")
        
        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize CubifyAnything camera poses'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/cubifyanything.yaml',
        help='Path to Hydra config file'
    )
    parser.add_argument(
        '--seq-index',
        type=int,
        default=None,
        help='Specific sequence index to visualize (if not set, opens interactive viewer)'
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
    
    args = parser.parse_args()
    
    if args.seq_index is not None:
        # Single sequence visualization
        config_dir = os.path.dirname(os.path.abspath(args.config))
        config_name = os.path.basename(args.config).replace('.yaml', '')
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name)
        
        cubify_config = cfg.data.train.dataset.dataset_configs[0]
        dataset = CubifyAnythingDataset(
            common_conf=CommonConf(cfg),
            split=cubify_config.split,
            CUBIFY_URL=cubify_config.CUBIFY_URL
        )
        
        fig = plot_camera_poses(dataset, seq_index=args.seq_index, img_per_seq=args.img_per_seq)
        
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {args.save}")
        else:
            plt.show()
    else:
        # Interactive viewer
        interactive_viewer(args.config)
