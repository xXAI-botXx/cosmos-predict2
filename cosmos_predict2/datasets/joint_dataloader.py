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


import numpy as np
import torch
import webdataset

from imaginaire.lazy_config import instantiate


class IterativeJointDataLoader(webdataset.WebLoader):
    r"""
    A joint dataloader that supports loading both images and videos.
    """

    def __init__(self, dataloaders: dict[str, dict[str, torch.utils.data.DataLoader | webdataset.WebLoader | int]]):
        """
        Initialize the JointDataLoader with multiple datasets.

        Args:
            dataloaders: key - dataset_name; value - {"dataloader": dataloader, "ratio": data_ratio}

        Example:
            joint_loader = IterativeJointDataLoader(
                dataloaders{
                    "image_data": {
                        "dataloader": webdataset.WebLoader(...),
                        "ratio": 4,
                    },
                    "video_data": {
                        "dataloader": torch.utils.data.DataLoader(...),
                        "ratio": 1,
                    },
                }
            )
        """
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []

        for dataset_name, dataloader_data in dataloaders.items():
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"]))
            self.data_ratios.append(dataloader_data["ratio"])

        self.global_id = 0
        self.ratio_sum = sum(self.data_ratios)

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        for data in self.dataloader_list:
            self.data_len += len(data)

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        while True:
            data_id = self.global_id % self.ratio_sum
            index_id = self._get_dataloader_index(data_id)
            curr_dataloader = self.dataloaders[index_id]
            output = next(curr_dataloader)
            output["dataset_name"] = self.dataset_name_list[index_id]
            self.global_id += 1
            del curr_dataloader
            yield output

    def _get_dataloader_index(self, data_id):
        """Maps global id to the corresponding dataloader index based on ratio."""
        for i, r in enumerate(self.data_ratios):
            if data_id < r:
                return i
            data_id -= r
        raise ValueError("Invalid data_id")


class RandomJointDataLoader(webdataset.WebLoader):
    r"""
    A joint dataloader that supports randomly samples batches from multiple datasets.
    """

    # def __init__(self, **kwargs):
    def __init__(self, dataloaders: dict[str, dict[str, torch.utils.data.DataLoader | webdataset.WebLoader | int]]):
        """
        Initialize the JointDataLoader with multiple datasets.

        Args:
            **kwargs: Arbitrary keyword arguments where each key is a string
                      representing the dataset name, and each value is either
                      a `webdataset.WebLoader` or `torch.utils.data.DataLoader`
                      instance.

        Raises:
            AssertionError: If any value in kwargs is not an instance of
                            `webdataset.WebLoader` or `torch.utils.data.DataLoader`.
            AssertionError: If any key in kwargs is not a string.

        Example:
            joint_loader = JointDataLoader(
                images=webdataset.WebLoader(...),
                videos=torch.utils.data.DataLoader(...)
            )
        """
        self.dataloader_list, self.dataset_name_list, self.data_ratios = [], [], []

        for dataset_name, dataloader_data in dataloaders.items():
            assert set(dataloader_data.keys()) == {"dataloader", "ratio"}, f"Invalid config: {dataloader_data}"
            self.dataset_name_list.append(dataset_name)
            self.dataloader_list.append(instantiate(dataloader_data["dataloader"]))
            self.data_ratios.append(dataloader_data["ratio"])

        assert sum(self.data_ratios) == 1.0, "Sum of sample probabilities should be equal to 1."

        self.data_len = 0
        self.dataloaders = [iter(dataloader) for dataloader in self.dataloader_list]
        for data in self.dataloader_list:
            self.data_len += len(data)

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        while True:
            # Sample a random dataset
            data_id = int(np.random.choice(len(self.dataloader_list), 1, p=self.data_ratios)[0])
            curr_dataloader = self.dataloaders[data_id]
            output = next(curr_dataloader)
            output["dataset_name"] = self.dataset_name_list[data_id]
            del curr_dataloader
            yield output
