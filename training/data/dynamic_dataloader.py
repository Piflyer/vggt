# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from hydra.utils import instantiate
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
import torch.distributed as dist
from abc import ABC, abstractmethod

from .worker_fn import get_worker_init_fn
from .collate import collate_with_cubify_instances

class DynamicTorchDataset(ABC):
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn or collate_with_cubify_instances
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        # Instantiate the dataset
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]
        self.image_num_range = common_config.img_nums    # e.g., [2, 24]
        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers
        if dist.is_available() and dist.is_initialized():
            self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        else:
            self.sampler = DynamicDistributedSampler(
                self.dataset,
                num_replicas=1,
                rank=0,
                seed=seed,
                shuffle=shuffle,
            )
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu
        )

    def get_loader(self, epoch):
        print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        has_workers = self.num_workers and self.num_workers > 0
        prefetch_factor = 2 if has_workers else None
        persistent_workers = self.persistent_workers if has_workers else False

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=prefetch_factor,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )
        

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.
        
        Each batch is constrained to have a maximum of 32 total images across all
        sequences in the batch. If adding a sequence would exceed this limit, the
        current batch is yielded and a new batch is started.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)
        batches_yielded = 0
        max_batches = self.__len__() // self.max_img_per_gpu  # Rough estimate

        while batches_yielded < max_batches:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_image_num = min(random_image_num, self.max_img_per_gpu)  # Cap it!
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )
                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch with max 32 images limit
                current_batch = []
                total_images_in_batch = 0
                # max_images_per_batch = 32  # Hard limit on total images per batch
                
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, image_num, aspect_ratio)
                        seq_idx, img_num, aspect_ratio = item
                        
                        # # Check if adding this sequence would exceed 32 image limit
                        # if total_images_in_batch + img_num > max_images_per_batch:
                        #     # Yield current batch if it has samples, then start a new one
                        #     if len(current_batch) > 0:
                        #         yield current_batch
                        #         batches_yielded += 1
                        #         current_batch = []
                        #         total_images_in_batch = 0
                        
                        current_batch.append(item)
                        total_images_in_batch += img_num
                        
                    except StopIteration:
                        # Sampler exhausted - recreate iterator to cycle through dataset again
                        sampler_iterator = iter(self.sampler)
                        try:
                            item = next(sampler_iterator)
                            seq_idx, img_num, aspect_ratio = item
                            
                            # # Check limit for recycled iterator too
                            # if total_images_in_batch + img_num > max_images_per_batch:
                            #     if len(current_batch) > 0:
                            #         yield current_batch
                            #         batches_yielded += 1
                            #         current_batch = []
                            #         total_images_in_batch = 0
                            
                            current_batch.append(item)
                            total_images_in_batch += img_num
                        except StopIteration:
                            # Empty dataset or other issue
                            break

                # Only yield non-empty batches
                if len(current_batch) == 0:
                    # Could not collect any samples
                    break
                
                # Yield the final batch (guaranteed to have at least 1 sample)
                yield current_batch
                batches_yielded += 1

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    
    This sampler cycles infinitely through the dataset to support unlimited training iterations.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.aspect_ratio = None
        self.image_num = None
        self._indices_cache = None

    def _get_indices(self):
        """Get indices for this epoch from parent class and cache them."""
        indices_iter = super().__iter__()
        return list(indices_iter)

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio), cycling infinitely.
        Caches indices from parent class and cycles through them.
        """
        # Get indices once per epoch
        if self._indices_cache is None:
            self._indices_cache = self._get_indices()
        
        # Cycle infinitely through the cached indices
        idx_position = 0
        while True:
            if len(self._indices_cache) == 0:
                # Edge case: empty dataset
                break
            
            idx = self._indices_cache[idx_position % len(self._indices_cache)]
            yield (idx, self.image_num, self.aspect_ratio,)
            idx_position += 1

    def set_epoch(self, epoch: int):
        """Set epoch and invalidate cache to get new shuffled indices."""
        super().set_epoch(epoch)
        self._indices_cache = None  # Force re-generation of indices

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num
