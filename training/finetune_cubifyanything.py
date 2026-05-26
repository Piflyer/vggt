#!/usr/bin/env python3
"""
Fine-tune VGGT on CubifyAnything Dataset

This script provides a simple interface to fine-tune a pre-trained VGGT model
on the CubifyAnything dataset.

Usage:
    # Basic usage with default settings
    python finetune_cubifyanything.py --checkpoint /path/to/vggt_checkpoint.pth

    # Specify custom data paths
    python finetune_cubifyanything.py \
        --checkpoint /path/to/vggt_checkpoint.pth \
        --train_tars /path/to/train/*.tar \
        --val_tars /path/to/val/*.tar

    # Multi-GPU training
    torchrun --nproc_per_node=4 finetune_cubifyanything.py \
        --checkpoint /path/to/vggt_checkpoint.pth

    # Adjust batch size and learning rate
    python finetune_cubifyanything.py \
        --checkpoint /path/to/vggt_checkpoint.pth \
        --batch_size 32 \
        --lr 1e-5 \
        --epochs 20
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hydra import initialize, compose
from omegaconf import OmegaConf
from trainer import Trainer


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def find_tar_files(pattern):
    """Find tar files matching the pattern"""
    if isinstance(pattern, list):
        files = []
        for p in pattern:
            files.extend(glob.glob(p))
        return sorted(set(files))
    return sorted(glob.glob(pattern))


def create_config(args):
    """Create Hydra config from arguments"""
    
    # Find tar files
    if args.train_tars:
        train_tars = find_tar_files(args.train_tars)
        if not train_tars:
            raise ValueError(f"No training tar files found matching: {args.train_tars}")
        logging.info(f"Found {len(train_tars)} training tar files")
    else:
        # Default: look in ml-cubifyanything/data/train
        default_train_path = "/home/tim/vggt-cubify/ml-cubifyanything/data/train/*.tar"
        train_tars = find_tar_files(default_train_path)
        if not train_tars:
            raise ValueError(f"No training tar files found. Please specify --train_tars")
        logging.info(f"Using default training data: {len(train_tars)} tar files")
    
    if args.val_tars:
        val_tars = find_tar_files(args.val_tars)
        if not val_tars:
            raise ValueError(f"No validation tar files found matching: {args.val_tars}")
        logging.info(f"Found {len(val_tars)} validation tar files")
    else:
        # Default: look in ml-cubifyanything/data/val
        default_val_path = "/home/tim/vggt-cubify/ml-cubifyanything/data/val/*.tar"
        val_tars = find_tar_files(default_val_path)
        if not val_tars:
            logging.warning("No validation tar files found. Validation will be skipped.")
            val_tars = []
        else:
            logging.info(f"Using default validation data: {len(val_tars)} tar files")
    
    # Convert tar file lists to URL strings
    train_urls = "{" + ",".join(train_tars) + "}"
    val_urls = "{" + ",".join(val_tars) + "}" if val_tars else ""
    
    # Load base config
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="cubifyanything")
    
    # Override with command-line arguments
    cfg.exp_name = args.exp_name
    cfg.max_epochs = args.epochs
    cfg.max_img_per_gpu = args.batch_size
    cfg.num_workers = args.num_workers
    
    # Update checkpoint path
    cfg.checkpoint.resume_checkpoint_path = args.checkpoint
    cfg.checkpoint.save_dir = f"logs/{args.exp_name}/ckpts"
    cfg.logging.log_dir = f"logs/{args.exp_name}"
    
    # Update learning rate
    cfg.optim.optimizer.lr = args.lr
    
    # Update scheduler with new learning rate
    if hasattr(cfg.optim.options, 'lr') and cfg.optim.options.lr:
        for scheduler_config in cfg.optim.options.lr:
            if hasattr(scheduler_config, 'scheduler'):
                schedulers = scheduler_config.scheduler.schedulers
                if len(schedulers) >= 2:
                    # Update warmup end value
                    schedulers[0].end_value = args.lr
                    # Update cosine start value
                    schedulers[1].start_value = args.lr
    
    # Update data paths
    cfg.data.train.dataset.dataset_configs[0].CUBIFY_URL = train_urls
    if val_urls:
        cfg.data.val.dataset.dataset_configs[0].CUBIFY_URL = val_urls
    
    # Update batch size in data config
    cfg.data.train.max_img_per_gpu = args.batch_size
    if val_urls:
        cfg.data.val.max_img_per_gpu = args.batch_size
    
    # Freeze modules if specified
    if args.freeze_aggregator:
        if not hasattr(cfg.optim, 'frozen_module_names'):
            cfg.optim.frozen_module_names = []
        if "*aggregator*" not in cfg.optim.frozen_module_names:
            cfg.optim.frozen_module_names.append("*aggregator*")
        logging.info("Freezing aggregator module")
    
    # Enable/disable heads
    cfg.model.enable_camera = args.enable_camera
    cfg.model.enable_depth = args.enable_depth
    cfg.model.enable_point = args.enable_point
    cfg.model.enable_track = args.enable_track
    
    # Update loss weights
    if not args.enable_camera:
        cfg.loss.camera = None
    if not args.enable_depth:
        cfg.loss.depth = None
    if not args.enable_point:
        cfg.loss.point = None
    if not args.enable_track:
        cfg.loss.track = None
    
    return cfg


def validate_checkpoint(checkpoint_path):
    """Validate that checkpoint exists"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logging.info(f"Using checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune VGGT on CubifyAnything dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained VGGT checkpoint (.pth file)"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_tars",
        type=str,
        nargs='+',
        default=None,
        help="Glob pattern(s) for training tar files (e.g., '/path/to/train/*.tar'). "
             "Default: /home/tim/vggt-cubify/ml-cubifyanything/data/train/*.tar"
    )
    parser.add_argument(
        "--val_tars",
        type=str,
        nargs='+',
        default=None,
        help="Glob pattern(s) for validation tar files (e.g., '/path/to/val/*.tar'). "
             "Default: /home/tim/vggt-cubify/ml-cubifyanything/data/val/*.tar"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=48,
        help="Maximum images per GPU (default: 48)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)"
    )
    
    # Experiment configuration
    parser.add_argument(
        "--exp_name",
        type=str,
        default="cubifyanything_finetune",
        help="Experiment name for logging and checkpoints (default: cubifyanything_finetune)"
    )
    
    # Model configuration
    parser.add_argument(
        "--freeze_aggregator",
        action="store_true",
        help="Freeze the aggregator module during training"
    )
    parser.add_argument(
        "--enable_camera",
        action="store_true",
        default=True,
        help="Enable camera head (default: True)"
    )
    parser.add_argument(
        "--enable_depth",
        action="store_true",
        default=True,
        help="Enable depth head (default: True)"
    )
    parser.add_argument(
        "--enable_point",
        action="store_true",
        default=False,
        help="Enable point head (default: False)"
    )
    parser.add_argument(
        "--enable_track",
        action="store_true",
        default=False,
        help="Enable tracking head (default: False)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("VGGT Fine-tuning on CubifyAnything Dataset")
    logging.info("=" * 80)
    
    # Validate checkpoint
    validate_checkpoint(args.checkpoint)
    
    # Create configuration
    logging.info("\nCreating training configuration...")
    cfg = create_config(args)
    
    # Print configuration summary
    logging.info("\n" + "=" * 80)
    logging.info("Training Configuration Summary")
    logging.info("=" * 80)
    logging.info(f"Experiment name: {cfg.exp_name}")
    logging.info(f"Checkpoint: {cfg.checkpoint.resume_checkpoint_path}")
    logging.info(f"Max epochs: {cfg.max_epochs}")
    logging.info(f"Batch size (max_img_per_gpu): {cfg.max_img_per_gpu}")
    logging.info(f"Learning rate: {cfg.optim.optimizer.lr}")
    logging.info(f"Num workers: {cfg.num_workers}")
    logging.info(f"\nEnabled heads:")
    logging.info(f"  Camera: {cfg.model.enable_camera}")
    logging.info(f"  Depth: {cfg.model.enable_depth}")
    logging.info(f"  Point: {cfg.model.enable_point}")
    logging.info(f"  Track: {cfg.model.enable_track}")
    logging.info(f"\nLogs will be saved to: {cfg.logging.log_dir}")
    logging.info(f"Checkpoints will be saved to: {cfg.checkpoint.save_dir}")
    logging.info("=" * 80 + "\n")
    
    # Create trainer and start training
    logging.info("Initializing trainer...")
    trainer = Trainer(**cfg)
    
    logging.info("\nStarting training...")
    logging.info("=" * 80 + "\n")
    
    trainer.run()
    
    logging.info("\n" + "=" * 80)
    logging.info("Training completed!")
    logging.info(f"Final checkpoints saved to: {cfg.checkpoint.save_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
