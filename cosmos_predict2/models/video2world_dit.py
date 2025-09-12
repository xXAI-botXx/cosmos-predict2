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


import torch

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import MiniTrainDIT


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            use_cuda_graphs=use_cuda_graphs,
        )
