# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import traceback
from collections.abc import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from cosmos_predict2.datasets.watchdog import OperationWatchdog


def concatenate_batches(n: int, data_batches: list[dict]) -> list[dict]:
    """
    Smartly concatenate n input data batches into m output data batches.
    Each data batch is a dictionary with values that can be torch tensor, string, or list.

    Args:
        n (int): Number of input batches to process per output batch
        data_batches (list): List of dictionary data batches

    Returns:
        list: List of concatenated data batches
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Calculate m based on the input
    total_batches = len(data_batches)
    if total_batches % n != 0:
        raise ValueError(f"Length of data_batches ({total_batches}) must be divisible by n ({n})")

    m = total_batches // n

    # Initialize output batches
    output_batches = []

    # Process in groups of n
    for i in range(m):
        # Get the corresponding batch from each group
        batches_to_concat = []
        for j in range(n):
            batch_idx = j * m + i
            batches_to_concat.append(data_batches[batch_idx])

        # Create a new merged dictionary
        merged_batch = {}

        # Get all unique keys from the dictionaries
        all_keys = set()
        for batch in batches_to_concat:
            all_keys.update(batch.keys())

        # Process each key
        for key in all_keys:
            # Collect values for this key from all batches
            values = []
            for batch in batches_to_concat:
                if key in batch:
                    values.append(batch[key])

            if not values:
                continue

            # Determine the type of the first non-None value
            first_value = next((v for v in values if v is not None), None)
            if first_value is None:
                merged_batch[key] = None
                continue

            # Handle different types
            if isinstance(first_value, torch.Tensor):
                # Assuming this is a tensor-like object with cat method (e.g., torch.Tensor)
                merged_batch[key] = torch.cat(values, dim=0)
            elif isinstance(first_value, str):
                merged_batch[key] = values[0]
            elif isinstance(first_value, list):
                # Extend lists
                merged_list = []
                for v in values:
                    merged_list.extend(v)
                merged_batch[key] = merged_list
            else:
                # For other types, just use the list of values
                merged_batch[key] = values

        output_batches.append(merged_batch)

    return output_batches


class CachedReplayDataLoader:
    """A DataLoader wrapper that asynchronously caches and replays data batches to
    mitigate slow loading issues. Assumes the underlying DataLoader is infinite.

    This class delegates all augmentation logic to an external augmentation function,
    which takes a batch from the data loader and returns multiple augmented versions.
    The class handles caching these augmented batches and optionally concatenating
    them when yielded.

    Attributes:
        data_loader (DataLoader): The underlying infinite DataLoader.
        cache_size (int): Maximum number of augmented batches to store in the cache.
        cache_augmentation_fn (Callable): Function to create multiple augmented versions of each batch.
        concat_size (int): Number of batches to concatenate when yielding from the iterator.
        rng (numpy.random.Generator): Controlled random number generator for deterministic behavior.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        cache_size: int,
        cache_augmentation_fn: Callable[[dict], list[dict]],
        concat_size: int = 1,
        name: str = "cached_replay_dataloader",
    ) -> None:
        """Initialize the CachedReplayDataLoader.

        Args:
            data_loader (DataLoader): The infinite DataLoader to fetch data batches from.
            cache_size (int): Maximum number of augmented data batches to store in the cache.
            cache_augmentation_fn (Callable[[Dict], List[Dict]]): Function that takes a batch and returns
                a list of augmented batches.
            concat_size (int, optional): Number of batches to concatenate when yielding. Defaults to 1.
        """
        self.data_loader = data_loader
        self.cache_size = cache_size
        self.cache_augmentation_fn = cache_augmentation_fn
        self.concat_size = concat_size

        # Create controlled random number generator for deterministic behavior
        self.rng = np.random.default_rng(123)

        # Create an iterator over the infinite DataLoader.
        self._data_iter: Iterator = iter(self.data_loader)
        # Internal cache to store augmented batches.
        self._cache: list[dict] = []
        # Condition variable to manage cache access.
        self._cache_cond = threading.Condition()
        # Event to signal the background thread to stop.
        self._stop_event = threading.Event()
        # Store exceptions from the background thread
        self._prefetch_exception = None

        self._watchdog = OperationWatchdog(warning_threshold=100, verbose_interval=600, name=name)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name=f"{name}_prefetch_thread"
        )
        self._prefetch_thread.start()

    def _prefetch_loop(self) -> None:
        """Continuously fetch batches from the DataLoader, augment them, and store in the cache.

        If the cache is full (reaches `cache_size`), this loop waits until space is available.
        Catches exceptions and stores them for later propagation to the main thread.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    with self._watchdog.watch("fetch raw batch", verbose_first_n=5):
                        batch = next(self._data_iter)
                except Exception as e:
                    # Capture DataLoader errors
                    self._set_exception(e, "Error fetching batch from DataLoader")
                    break

                try:
                    # Apply augmentation function to generate multiple augmented batches
                    with self._watchdog.watch("augmentation", verbose_first_n=5):
                        augmented_batches = self.cache_augmentation_fn(batch)
                except Exception as e:
                    # Capture augmentation function errors
                    self._set_exception(e, "Error in augmentation function")
                    break

                try:
                    # Use controlled random generator for shuffling
                    permutation = self.rng.permutation(len(augmented_batches))
                    augmented_batches = [augmented_batches[i] for i in permutation]

                    for aug_batch in augmented_batches:
                        with self._cache_cond:
                            while len(self._cache) >= self.cache_size and not self._stop_event.is_set():
                                self._cache_cond.wait(timeout=1.0)
                            if self._stop_event.is_set():
                                break
                            self._cache.append(aug_batch)
                            self._cache_cond.notify_all()
                except Exception as e:
                    # Capture other errors during caching
                    self._set_exception(e, "Error adding batch to cache")
                    break
        except Exception as e:
            # Catch any other unforeseen errors
            self._set_exception(e, "Unexpected error in prefetch thread")

    def _set_exception(self, exception: Exception, context: str = "") -> None:
        """Store an exception from the background thread with context information.

        Args:
            exception (Exception): The exception that was raised
            context (str, optional): Additional context about where the error occurred
        """
        error_info = f"{context}: {exception!s}\n{traceback.format_exc()}"
        with self._cache_cond:
            self._prefetch_exception = RuntimeError(error_info)
            self._cache_cond.notify_all()  # Wake up any waiting threads

    def _check_for_errors(self) -> None:
        """Check if the background thread has encountered an error and raise it if so."""
        if self._prefetch_exception is not None:
            raise self._prefetch_exception

    def __iter__(self) -> Iterator[dict]:
        """Yield augmented data batches from the cache, optionally concatenated based on concat_size.

        This method starts the background prefetch thread if it hasn't been started yet.
        If concat_size > 1, it collects multiple batches and concatenates them.

        Raises:
            RuntimeError: If the background thread encountered an error
        """
        while not self._stop_event.is_set():
            if self.concat_size <= 1:
                # Simple case: yield single batches
                with self._watchdog.watch("main thread fetch single batch", verbose_first_n=5):
                    with self._cache_cond:
                        while not self._cache and not self._stop_event.is_set() and self._prefetch_exception is None:
                            self._cache_cond.wait(timeout=1.0)  # Add timeout to periodically check for errors

                        # Check for errors before proceeding
                        self._check_for_errors()

                        if self._stop_event.is_set():
                            break

                        if not self._cache:  # If cache is still empty after timeout
                            continue

                        # Use controlled random generator to select batch index
                        idx = self.rng.integers(0, len(self._cache))
                        batch = self._cache.pop(idx)
                        self._cache_cond.notify_all()
                yield batch
            else:
                # Collect concat_size batches and concatenate them
                with self._watchdog.watch("main thread fetch smaples", verbose_first_n=5):
                    collected_batches = []
                    for _ in range(self.concat_size):
                        with self._cache_cond:
                            while (
                                not self._cache and not self._stop_event.is_set() and self._prefetch_exception is None
                            ):
                                self._cache_cond.wait(timeout=1.0)  # Add timeout to periodically check for errors

                            # Check for errors before proceeding
                            self._check_for_errors()

                            if self._stop_event.is_set():
                                break

                            if not self._cache:  # If cache is still empty after timeout
                                continue

                            # Use controlled random generator to select batch index
                            idx = self.rng.integers(0, len(self._cache))
                            batch = self._cache.pop(idx)
                            self._cache_cond.notify_all()
                        collected_batches.append(batch)

                if self._stop_event.is_set():
                    break

                if not collected_batches:
                    continue

                if len(collected_batches) < self.concat_size:
                    # Not enough batches collected, just concatenate the ones we have
                    concat_batches = concatenate_batches(len(collected_batches), collected_batches)
                    yield concat_batches[0]
                else:
                    # Concatenate the collected batches
                    try:
                        concat_batches = concatenate_batches(self.concat_size, collected_batches)
                        yield concat_batches[0]
                    except Exception as e:
                        # Handle errors in batch concatenation
                        raise RuntimeError(f"Error concatenating batches: {e!s}") from e

    def __len__(self) -> int:
        """Return the length of the underlying DataLoader."""
        return len(self.data_loader)

    def close(self) -> None:
        """Stop the prefetch thread and clear the cache.
        Also checks for any errors in the background thread and raises them.
        """
        self._stop_event.set()
        with self._cache_cond:
            self._cache_cond.notify_all()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)  # Add timeout to avoid hanging on thread join
        with self._cache_cond:
            self._cache.clear()

        # Check and propagate any errors from the background thread
        self._check_for_errors()


def get_cached_replay_dataloader(
    use_cache: bool = False,
    cache_size: int = 32,
    concat_size: int = 1,
    cache_augment_fn: Callable | None = None,
    cache_replay_name: str = "cached_replay_dataloader",
    webdataset: bool = True,
    **kwargs,
):
    if webdataset:
        from imaginaire.datasets.webdataset.dataloader import DataLoader as _DataLoader
    else:
        from torch.utils.data import DataLoader as _DataLoader

    if not use_cache:
        return _DataLoader(**kwargs)

    expected_batch_size = kwargs["batch_size"]
    assert expected_batch_size % concat_size == 0, (
        f"Batch size {expected_batch_size} must be divisible by concat_size {concat_size}"
    )
    kwargs["batch_size"] = expected_batch_size // concat_size

    dataloader = _DataLoader(**kwargs)

    # wrapper it with cached replay dataloader
    return CachedReplayDataLoader(
        data_loader=dataloader,
        cache_size=cache_size,
        concat_size=concat_size,
        cache_augmentation_fn=cache_augment_fn,
        name=cache_replay_name,
    )
