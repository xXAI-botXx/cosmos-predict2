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

from typing import IO, Any

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from torch import Tensor

from imaginaire.utils import log
from imaginaire.utils.easy_io import easy_io

try:
    import ffmpegcv
except Exception as e:  # ImportError cannot catch all problems
    log.info(e)
    ffmpegcv = None


def save_video(grid, video_name, fps=30):
    grid = (grid * 255).astype(np.uint8)
    grid = np.transpose(grid, (1, 2, 3, 0))
    with ffmpegcv.VideoWriter(video_name, "h264", fps) as writer:
        for frame in grid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)


def save_img_or_video(
    sample_C_T_H_W_in01: Tensor, save_fp_wo_ext: str | IO[Any], fps: int = 24, quality=None, ffmpeg_params=None
) -> None:
    """
    Save a tensor as an image or video file based on shape

        Args:
        sample_C_T_H_W_in01 (Tensor): Input tensor with shape (C, T, H, W) in [0, 1] range.
        save_fp_wo_ext (Union[str, IO[Any]]): File path without extension or file-like object.
        fps (int): Frames per second for video. Default is 24.
    """
    assert sample_C_T_H_W_in01.ndim == 4, "Only support 4D tensor"
    assert isinstance(save_fp_wo_ext, str) or hasattr(save_fp_wo_ext, "write"), (
        "save_fp_wo_ext must be a string or file-like object"
    )

    if torch.is_floating_point(sample_C_T_H_W_in01):
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.clamp(0, 1)
    else:
        assert sample_C_T_H_W_in01.dtype == torch.uint8, "Only support uint8 tensor"
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.float().div(255)

    kwargs = {}
    if quality is not None:
        kwargs["quality"] = quality
    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    if sample_C_T_H_W_in01.shape[1] == 1:
        save_obj = PILImage.fromarray(
            rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c 1 h w -> h w c").astype(np.uint8),
            mode="RGB",
        )
        ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="jpg",
            format="JPEG",
            quality=85,
            **kwargs,
        )
    else:
        save_obj = rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c t h w -> t h w c").astype(np.uint8)
        ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="mp4",
            format="mp4",
            fps=fps,
            **kwargs,
        )
