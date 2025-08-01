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

from cosmos_predict2.configs.base.config_text2image import (
    PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    PREDICT2_TEXT2IMAGE_PIPELINE_14B,
)
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B  # 720p, 16fps
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_14B  # 720p, 16fps
from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_16FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_10FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_16FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_10FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_16FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_10FPS,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_16FPS,
)
from cosmos_predict2.configs.base.config_multiview import PREDICT2_MULTIVIEW_PIPELINE_2B_720P_10FPS_7VIEWS_29FRAMES

from cosmos_predict2.models.text2image_model import (
    Predict2Text2ImageModel,
    Predict2Text2ImageModelConfig,
)
from cosmos_predict2.models.video2world_model import (
    Predict2ModelManagerConfig,
    Predict2Video2WorldModel,
    Predict2Video2WorldModelConfig,
)
from cosmos_predict2.models.multiview_model import (
    Predict2MultiviewModel,
    Predict2MultiviewModelConfig,
)

from imaginaire.lazy_config import LazyCall as L

# 2b model config for predict2 text2image
PREDICT2_TEXT2IMAGE_FSDP_2B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Text2ImageModel)(
        config=Predict2Text2ImageModelConfig(
            pipe_config=PREDICT2_TEXT2IMAGE_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=1,
        ),
        _recursive_=False,
    ),
)

# 14b model config for predict2 text2image
PREDICT2_TEXT2IMAGE_FSDP_14B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Text2ImageModel)(
        config=Predict2Text2ImageModelConfig(
            pipe_config=PREDICT2_TEXT2IMAGE_PIPELINE_14B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

# default 2b model config for predict2 video2world (720p, 16fps)
PREDICT2_VIDEO2WORLD_FSDP_2B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

# default 14b model config for predict2 video2world (720p, 16fps)
PREDICT2_VIDEO2WORLD_FSDP_14B = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_14B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=32,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

# 2b model configs for predict2 video2world with different resolutions and fps
PREDICT2_VIDEO2WORLD_FSDP_2B_480P_10FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_10FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

PREDICT2_VIDEO2WORLD_FSDP_2B_480P_16FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_480P_16FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

PREDICT2_VIDEO2WORLD_FSDP_2B_720P_10FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_10FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-10fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

PREDICT2_VIDEO2WORLD_FSDP_2B_720P_16FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_720P_16FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

# 14b model configs for predict2 video2world with different resolutions and fps
PREDICT2_VIDEO2WORLD_FSDP_14B_480P_10FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_10FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-480p-10fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

PREDICT2_VIDEO2WORLD_FSDP_14B_480P_16FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_14B_480P_16FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-480p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

PREDICT2_VIDEO2WORLD_FSDP_14B_720P_10FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_10FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-720p-10fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

PREDICT2_VIDEO2WORLD_FSDP_14B_720P_16FPS = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_14B_720P_16FPS,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)

# 2b model configs for predict2 multiview
PREDICT2_MULTIVIEW_FSDP_2B_720P_10FPS_7VIEWS_29FRAMES = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2MultiviewModel)(
        config=Predict2MultiviewModelConfig(
            pipe_config=PREDICT2_MULTIVIEW_PIPELINE_2B_720P_10FPS_7VIEWS_29FRAMES,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Multiview/model-720p-10fps-7views-29frames.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=8,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
)


def register_model() -> None:
    cs = ConfigStore.instance()
    # predict2 t2i 2b model
    cs.store(group="model", package="_global_", name="predict2_text2image_fsdp_2b", node=PREDICT2_TEXT2IMAGE_FSDP_2B)
    # predict2 t2i 14b model
    cs.store(group="model", package="_global_", name="predict2_text2image_fsdp_14b", node=PREDICT2_TEXT2IMAGE_FSDP_14B)
    # predict2 v2w 2b model (default 720p, 16fps)
    cs.store(group="model", package="_global_", name="predict2_video2world_fsdp_2b", node=PREDICT2_VIDEO2WORLD_FSDP_2B)
    # predict2 v2w 14b model (default 720p, 16fps)
    cs.store(
        group="model", package="_global_", name="predict2_video2world_fsdp_14b", node=PREDICT2_VIDEO2WORLD_FSDP_14B
    )
    # predict2 v2w 2b model by resolution and fps
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_2b_480p_10fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_2B_480P_10FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_2b_480p_16fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_2B_480P_16FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_2b_720p_10fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_2B_720P_10FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_2b_720p_16fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_2B_720P_16FPS,
    )
    # predict2 v2w 14b model by resolution and fps
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_14b_480p_10fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_14B_480P_10FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_14b_480p_16fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_14B_480P_16FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_14b_720p_10fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_14B_720P_10FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_video2world_fsdp_14b_720p_16fps",
        node=PREDICT2_VIDEO2WORLD_FSDP_14B_720P_16FPS,
    )
    cs.store(
        group="model",
        package="_global_",
        name="predict2_multiview_fsdp_2b_720p_10fps_7views_29frames",
        node=PREDICT2_MULTIVIEW_FSDP_2B_720P_10FPS_7VIEWS_29FRAMES,
    )