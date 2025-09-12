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

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from imaginaire.utils.misc import get_local_tensor_if_DTensor


class DTensorFastEmaModelUpdater:
    """
    Similar as FastEmaModelUpdater
    """

    def __init__(self):
        # Flag to indicate whether the cache is taken or not. Useful to avoid cache overwrite
        self.is_cached = False

    @torch.no_grad()
    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters(), strict=False):
            if isinstance(tgt_params, DTensor) and isinstance(src_params, DTensor):
                tgt_params.to_local().data.copy_(src_params.to_local().data)
            else:
                tgt_params.to_local().data.copy_(src_params.to_local().data)

    @torch.no_grad()
    def update_average(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module, beta: float = 0.9999) -> None:
        target_list = []
        source_list = []
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters(), strict=False):
            assert tgt_params.dtype == torch.float32, (
                f"EMA model only works in FP32 dtype, got {tgt_params.dtype} instead."
            )
            if isinstance(tgt_params, DTensor) and isinstance(src_params, DTensor):
                target_list.append(tgt_params.to_local())
                source_list.append(src_params.to_local().data)
            else:
                target_list.append(tgt_params)
                source_list.append(src_params.data)
        torch._foreach_mul_(target_list, beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - beta)

    @torch.no_grad()
    def cache(self, parameters: Any, is_cpu: bool = False) -> None:
        assert self.is_cached is False, "EMA cache is already taken. Did you forget to restore it?"
        device = "cpu" if is_cpu else "cuda"
        self.collected_params = [param.to_local().clone().to(device) for param in parameters]
        self.is_cached = True

    @torch.no_grad()
    def restore(self, parameters: Any) -> None:
        assert self.is_cached, "EMA cache is not taken yet."
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.to_local().copy_(c_param.data.type_as(param.data))
        self.collected_params = []
        # Release the cache after we call restore
        self.is_cached = False


def broadcast_dtensor_model_states(model: torch.nn.Module, mesh: DeviceMesh):
    """Broadcast model states from replicate mesh's rank 0."""
    replicate_group = mesh.get_group("replicate")
    all_ranks = dist.get_process_group_ranks(replicate_group)
    if len(all_ranks) == 1:
        return

    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        # Get src rank which is the first rank in each replication group
        src_rank = all_ranks[0]
        # Broadcast the local tensor
        local_tensor = get_local_tensor_if_DTensor(tensor)
        dist.broadcast(
            local_tensor,
            src=src_rank,
            group=replicate_group,
        )
