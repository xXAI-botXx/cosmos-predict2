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

import random

import numpy as np
import torch

from imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from imaginaire.utils import log


def pad_and_resize(
    arr_np: np.ndarray, ntokens: int, is_mask_all_ones: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Function for padding and resizing a numpy array.
    Args:
        arr (np.ndarray): Input array
        ntokens (int): Number of output tokens after padding
        is_mask_all_ones (bool): if true, set mask to ones
    Returns:
        arr_padded (torch.Tensor): Padded output tensor
        mask (torch.Tensor): Padding mask
    """

    if isinstance(arr_np, np.ndarray):
        arr = torch.from_numpy(arr_np)
    elif isinstance(arr_np, torch.Tensor):
        arr = arr_np.clone().detach()
    else:
        raise TypeError("`arr_np` should be a numpy array or torch tensor.")
    embed_dim = arr.shape[1]

    arr_padded = torch.zeros(ntokens, embed_dim, device=arr.device, dtype=torch.float32)

    # If the input text is larger than num_text_tokens, clip it.
    if arr.shape[0] > ntokens:
        arr = arr[0:ntokens]

    mask = torch.LongTensor(ntokens).zero_()
    if len(arr.shape) > 1:
        mask[0 : arr.shape[0]] = 1

    if len(arr.shape) > 1:
        arr_padded[0 : arr.shape[0]] = arr

    if is_mask_all_ones:
        mask.fill_(1)

    return arr_padded, mask


class TextTransformForVideo(Augmentor):
    def __init__(self, input_keys: dict, output_keys: list | None = None, args: dict | None = None) -> None:
        super().__init__(input_keys, output_keys, args)

        # our caption is saved in json with format: {"<key>": "xxx", "<caption_windows_key1>": [{"start_frame": x, "end_frame": x, "<caption_type>": xxx}, ...], "<caption_windows_key2>": [{"start_frame":...]}
        # our t5 embedding is saved in pickle with format: [{"<embedding_caption_type1>": array1, "<embedding_caption_type2>": array2}, ...]
        self.captions_key: str = args[
            "captions_key"
        ]  # Saves the captions; this get mapped to the key in data_dict to fetch the caption field
        self.embeddings_key: str = args[
            "embeddings_key"
        ]  # Saves the embeddings; this get mapped to the key in data_dict to fetch the embedding field
        self.caption_windows_key: str = args[
            "caption_windows_key"
        ]  # key to get the caption windows from the caption field
        self.caption_type: str = args["caption_type"]  # key of caption type to fetch the caption from caption windows
        self.embedding_caption_type: str = args[
            "embedding_caption_type"
        ]  # key to get the embedding of a particular caption type from the embedding field
        self.t5_tokens_num = args["t5_tokens"]["num"]  # number of tokens we cap after padding
        self.is_mask_all_ones = args["is_mask_all_ones"]  # if true, set mask for t5 to all ones

        self.caption_probs: dict[str, float] = args[
            "caption_probs"
        ]  # probabilities for user/short/medium/long captions
        self.caption_style_mapping = {
            "long": self.caption_type,
            "short": f"{self.caption_type}_short",
            "medium": f"{self.caption_type}_medium",
            "user": f"{self.caption_type}_user",
        }
        self.embedding_style_mapping = {
            "long": self.embedding_caption_type,
            "short": f"{self.embedding_caption_type}_short",
            "medium": f"{self.embedding_caption_type}_medium",
            "user": f"{self.embedding_caption_type}_user",
        }
        assert self.caption_probs.keys() == self.caption_style_mapping.keys() == self.embedding_style_mapping.keys(), (
            "The keys for caption_probs, caption_style_mapping, and embedding_style_mapping should match"
        )

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """

        try:
            windows = data_dict[self.captions_key][self.caption_windows_key]
            n_windows = len(windows)
            chunk_index = data_dict["chunk_index"]

            if chunk_index == n_windows:
                # This will only happen when the number of captions does not match number of chunks due to re-transcoding the videos.
                log.warning(
                    f"TextTransform dataloader error: Found {data_dict['n_orig_video_frames']} in video but captioning is done with videos of {windows[-1]['end_frame']} frames. This mismatch is due to video re-transcoding.",
                    rank0_only=False,
                )
                chunk_index -= 1

            selected_caption_window = windows[chunk_index]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, chunk_index: {data_dict['chunk_index']}\n error {e}",
                rank0_only=False,
            )
            return None

        sampled_caption_style = None
        try:
            available_caption_styles = []
            for k in selected_caption_window.keys():
                caption_style = k.replace(self.caption_type, "").replace("_", "")
                if caption_style == "":  # it is long caption by default
                    available_caption_styles.append("long")
                elif caption_style in self.caption_style_mapping:
                    available_caption_styles.append(caption_style)
                else:
                    assert caption_style in ["startframe", "endframe"], f"Unsupported caption_type {caption_style}"

            probabilities_for_available_caption_styles = {
                k: v for k, v in self.caption_probs.items() if k in available_caption_styles
            }
            sampled_caption_style = random.choices(
                list(probabilities_for_available_caption_styles),
                weights=probabilities_for_available_caption_styles.values(),
            )[0]
            data_dict["ai_caption"] = selected_caption_window[self.caption_style_mapping[sampled_caption_style]]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, selected_caption_window: {selected_caption_window}\n error {e}",
                rank0_only=False,
            )
            return None
        if data_dict["ai_caption"] == "":
            log.warning(
                f"TextTransform dataloader error -- empty caption! url: {data_dict['__url__']}, key: {data_dict['__key__']}, selected_caption_window: {selected_caption_window}",
                rank0_only=False,
            )
            return None

        assert data_dict["ai_caption"] is not None and sampled_caption_style is not None
        data_dict["sampled_caption_style"] = sampled_caption_style

        del data_dict[self.captions_key]  # delete the field as we have extracted ai_caption from it

        ai_caption_embedding_data = data_dict[self.embeddings_key]
        try:
            t5_embedding = ai_caption_embedding_data[chunk_index][self.embedding_style_mapping[sampled_caption_style]]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, chunk_index: {data_dict['chunk_index']}, n embeddings: {len(ai_caption_embedding_data)}, n captions: {n_windows} \n error {e}",
                rank0_only=False,
            )
            return None
        out_t5, out_t5_mask = pad_and_resize(
            t5_embedding,
            self.t5_tokens_num,
            is_mask_all_ones=self.is_mask_all_ones,
        )
        data_dict["t5_text_embeddings"] = out_t5
        data_dict["t5_text_mask"] = out_t5_mask
        del data_dict[self.embeddings_key]  # delete the field as we have extracted t5 embedding from it

        return data_dict
