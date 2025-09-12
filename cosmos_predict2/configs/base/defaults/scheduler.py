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

from hydra.core.config_store import ConfigStore

from cosmos_predict2.functional.lr_scheduler import LambdaLinearScheduler
from imaginaire.lazy_config import LazyCall as L

LambdaLinearSchedulerConfig = L(LambdaLinearScheduler)(
    warm_up_steps=[1000],
    cycle_lengths=[10000000000000],
    f_start=[1.0e-6],
    f_max=[1.0],
    f_min=[1.0],
)


class ConstantScheduler:
    """
    Linear instead of cosine decay for the main part of the cycle.
    """

    def __call__(self, n, **kwargs):
        return 1.0

    def schedule(self, n, **kwargs):
        return 1.0


def register_scheduler():
    cs = ConfigStore.instance()
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)
    cs.store(group="scheduler", package="scheduler", name="constant", node=L(ConstantScheduler)())
