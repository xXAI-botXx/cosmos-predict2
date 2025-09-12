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


import omegaconf
import torch
import torchvision.transforms.functional as transforms_F

from imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from imaginaire.datasets.webdataset.augmentors.image.misc import obtain_augmentation_size, obtain_image_size


class ReflectionPadding(Augmentor):
    def __init__(self, input_keys: list, output_keys: list | None = None, args: dict | None = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs reflection padding. This function also returns a padding mask.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """

        assert self.args is not None, "Please specify args in augmentation"
        if self.output_keys is None:
            self.output_keys = self.input_keys

        # Obtain image and augmentation sizes
        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        target_size = obtain_augmentation_size(data_dict, self.args)

        assert isinstance(target_size, (tuple, omegaconf.listconfig.ListConfig)), "Please specify target size as tuple"
        target_w, target_h = target_size

        target_w = int(target_w)
        target_h = int(target_h)

        # Calculate padding vals
        padding_left = int((target_w - orig_w) / 2)
        padding_right = target_w - orig_w - padding_left
        padding_top = int((target_h - orig_h) / 2)
        padding_bottom = target_h - orig_h - padding_top
        padding_vals = [padding_left, padding_top, padding_right, padding_bottom]

        for inp_key, out_key in zip(self.input_keys, self.output_keys, strict=False):
            if max(padding_vals[0], padding_vals[2]) >= orig_w or max(padding_vals[1], padding_vals[3]) >= orig_h:
                # In this case, we can't perform reflection padding. This is because padding values
                # are larger than the image size. So, perform edge padding instead.
                data_dict[out_key] = transforms_F.pad(data_dict[inp_key], padding_vals, padding_mode="edge")
            else:
                # Perform reflection padding
                data_dict[out_key] = transforms_F.pad(data_dict[inp_key], padding_vals, padding_mode="reflect")

            if out_key != inp_key:
                del data_dict[inp_key]

        # Return padding_mask when padding is performed.
        # Padding mask denotes which pixels are padded.
        padding_mask = torch.ones((1, target_h, target_w))
        padding_mask[:, padding_top : (padding_top + orig_h), padding_left : (padding_left + orig_w)] = 0
        data_dict["padding_mask"] = padding_mask
        data_dict["image_size"] = torch.tensor([target_h, target_w, orig_h, orig_w], dtype=torch.float)

        return data_dict
