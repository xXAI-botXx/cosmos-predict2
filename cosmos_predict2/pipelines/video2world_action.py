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

from typing import Any

import numpy as np
import torch
from megatron.core import parallel_state

from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from cosmos_predict2.models.utils import load_state_dict
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig, get_cosmos_text_encoder
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log, misc
from imaginaire.utils.ema import FastEmaModelUpdater

IS_PREPROCESSED_KEY = "is_preprocessed"
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class Video2WorldActionConditionedPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

    @staticmethod
    def from_config(
        config: Video2WorldPipelineConfig,
        dit_path: str = "",
        use_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = Video2WorldActionConditionedPipeline(device=device, torch_dtype=torch_dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition)
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )

        pipe.scaling = RectifiedFlowScaling(
            pipe.sigma_data, config.rectified_flow_t_scaling_factor, config.rectified_flow_loss_weight_uniform
        )

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert pipe.tokenizer.latent_ch == pipe.config.state_ch, (
            f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"
        )

        # 4. Load text encoder
        if use_text_encoder:
            pipe.text_encoder = get_cosmos_text_encoder(config=config.text_encoder, device=device)
        else:
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0, (
            "conditioner should not have learnable parameters"
        )

        if load_prompt_refiner:
            pipe.prompt_refiner = CosmosReason1(
                checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
                offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
                enabled=config.prompt_refiner_config.enabled,
            )

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
            pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None
            pipe.video_guardrail_runner = None

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        # with init_weights_on_device():
        # NOTE: we don't load checkpoint on meta device since we have additional action encoder
        dit_config = config.net
        pipe.dit = instantiate(dit_config).eval()  # inference

        if dit_path:
            state_dict = load_state_dict(dit_path)
            prefix_to_load = "net_ema." if load_ema_to_reg else "net."

            log.info(f"Loading {'[ema]/regular' if load_ema_to_reg else 'ema/[regular]'} weights from {dit_path}")
            # drop net./net_ema. prefix if it exists, depending on the load_ema_to_reg flag
            state_dict_dit_compatible = dict()
            for k, v in state_dict.items():
                if k.startswith(prefix_to_load):
                    state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
                else:
                    state_dict_dit_compatible[k] = v
            pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            del state_dict, state_dict_dit_compatible
            log.success(f"Successfully loaded DiT from {dit_path}")

        # 6-2. Handle EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            pipe.dit_ema_worker = FastEmaModelUpdater()  # default when not using FSDP

            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            # copying is only necessary when starting the training at iteration 0.
            # Actual state_dict should be loaded after the pipe is created.
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        actions: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            negative_prompt (str): Negative prompt.
            num_latent_conditional_frames (int, optional): The number of latent conditional frames. Defaults to 1.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape

        self.batch_size = 1
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            # NOTE: we don't use text embeddings for action conditional video2world
            "t5_text_embeddings": torch.zeros(
                self.batch_size,
                CosmosTextEncoderConfig.NUM_TOKENS,
                CosmosTextEncoderConfig.EMBED_DIM,
                dtype=torch.bfloat16,
            ).cuda(),
            "fps": torch.randint(16, 32, (self.batch_size,)),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_latent_conditional_frames,  # Specify number of conditional frames
            "action": actions,
        }

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            data_batch["neg_t5_text_embeddings"] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    @torch.no_grad()
    def __call__(
        self,
        first_frame: np.ndarray,
        actions: np.ndarray,
        prompt: str = "",
        negative_prompt: str = "",
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        seed: int = 0,
        solver_option: str = "2ab",
    ) -> torch.Tensor | None:
        # Parameter check
        # width, height = VIDEO_RES_SIZE_INFO[self.config.resolution]["16:9"]  # type: ignore
        # height, width = self.check_resize_height_width(height, width)
        assert num_conditional_frames in [1, 5], "num_conditional_frames must be 1 or 5"
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)

        # num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t)

        # transform first frame and actions to tensor
        vid_input = torch.from_numpy(first_frame).permute(2, 0, 1)[None, :, None, ...]
        actions_tensor = torch.from_numpy(actions).to(dtype=torch.bfloat16)[None, ...]

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            actions_tensor,
            prompt,
            negative_prompt,
            num_latent_conditional_frames=num_latent_conditional_frames,
        )

        # preprocess
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_video_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=True)

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),  # noqa: RUF005
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(scheduler.timesteps):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred = x0_fn(sample, sigma_in)

            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]

        # Run video guardrail on the generated video and apply postprocessing
        if self.video_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            # Clamp to safe range before normalization
            video = video.clamp(-1.0, 1.0)
            video_normalized = (video + 1) / 2  # [0, 1]

            # Convert tensor to NumPy frames for guardrail processing
            video_squeezed = video_normalized.squeeze(0)  # (C, T, H, W)
            frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

            # Run guardrail
            processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            if processed_frames is None:
                return None
            else:
                log.success("Passed guardrail on generated video")

            # Convert processed frames back to tensor format
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1  # back to [-1, 1]
            processed_video = processed_video.unsqueeze(0)

            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("Video generation completed successfully")
        return video
