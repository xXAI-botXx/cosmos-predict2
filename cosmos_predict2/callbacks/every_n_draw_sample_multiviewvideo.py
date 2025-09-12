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

from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from hydra import compose, initialize
from hydra.utils import instantiate

from cosmos_predict2.callbacks.every_n_draw_sample import (
    EveryNDrawSample,
    convert_to_primitive,
    is_primitive,
    resize_image,
)

# TODO: Remove callback dependency on model imports. Can pass keys as callback args.
from cosmos_predict2.pipelines.multiview import NUM_CONDITIONAL_FRAMES_KEY
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from imaginaire.visualize.video import save_img_or_video

TRAIN_SAMPLE_N_VIEWS_KEY = "train_sample_n_views"
CONTROL_WEIGHT_KEY = "control_weight"
# from projects.cosmos.transfer2_multiview.models.multiview_vid2vid_model_control_vace import CONTROL_WEIGHT_KEY

try:
    import ffmpegcv
except Exception as e:  # ImportError cannot catch all problems
    log.info(e)
    ffmpegcv = None
import cv2
import numpy as np


class EveryNDrawSampleMultiviewVideo(EveryNDrawSample):
    """
    This class is a modified version of EveryNDrawSample that saves 12 frames instead of 3.
    """

    def __init__(
        self,
        *args,
        sample_n_views,
        n_view_embed=None,
        dataset_name=None,
        ctrl_hint_keys=None,
        control_weights=[1.0],  # noqa: B006
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_n_views = sample_n_views
        self.n_view_embed = n_view_embed
        self.dataset_name = dataset_name
        self.ctrl_hint_keys = ctrl_hint_keys
        self.control_weights = control_weights

    def get_dataloader_iter(self, dataset_name: str):
        # Point Hydra at the root config package that owns make_config()
        with initialize(version_base=None, config_path="../configs/vid2vid"):
            # Compose the project's default config but override data_train
            cfg = compose(config_name="config", overrides=[f"data_train={dataset_name}"])

        # Hydra's instantiate turns the DictConfig node into a real DataLoader
        return iter(instantiate(cfg.dataloader_train))

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        if self.dataset_name is not None:
            self.dataloader_iter = self.get_dataloader_iter(self.dataset_name)
        return super().on_train_start(model, iteration)

    def _ensure_even_dimensions(self, frame: np.ndarray) -> np.ndarray:
        """
        ffmpeg (H.264) requires both H and W to be even.  If either is odd we pad
        by 1 pixel on the bottom/right using edge-replication.
        """
        h, w = frame.shape[:2]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        return frame

    def save_video(self, grid, video_name, fps: int = 30):
        log.info(f"Saving video to {video_name}")
        grid = (grid * 255).astype(np.uint8)
        grid = np.transpose(grid, (1, 2, 3, 0))  # (T, H, W, C)

        with ffmpegcv.VideoWriter(video_name, "h264", fps) as writer:
            for frame in grid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = self._ensure_even_dimensions(frame)
                writer.write(frame)

    def run_save(self, to_show, batch_size, n_views, base_fp_wo_ext) -> str | None:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        file_base_fp_12frames = f"{base_fp_wo_ext}_12frames.jpg"
        local_path_12frames = f"{self.local_dir}/{file_base_fp_12frames}"

        if self.rank == 0:
            to_show = to_show[:, :n_viz_sample]  # [n, b, c, t, h, w]
            # Select 12 frames for the grid
            _T = to_show.shape[3]
            n = 12
            twelve_frames_list = [round(ix * (_T - 1) / (n - 1)) for ix in range(n)]
            to_show_12frames = to_show[:, :, :, twelve_frames_list]
            to_show_12frames = rearrange(to_show_12frames, "n b c t h w -> 1 c (n h) (b t w)")
            image_grid_12frames = torchvision.utils.make_grid(to_show_12frames, nrow=1, padding=0, normalize=False)
            torchvision.utils.save_image(
                resize_image(image_grid_12frames, 1024), local_path_12frames, nrow=1, scale_each=True
            )
            # Create a single stacked video
            video_tensor = rearrange(to_show, "n b c (v t) h w -> t (n h) (b v w) c", v=n_views)

            # Resize width to 1024 while preserving aspect ratio (keep float to avoid quantization before resize)
            max_w = 2048
            T, H, W, C = video_tensor.shape
            if W > max_w:
                scale = max_w / W
                new_w = max_w
                new_h = int(H * scale)
                # video_tensor is currently float in 0-1 range -> convert [T, H, W, C] to [T, C, H, W]
                video_tensor_f = video_tensor.permute(0, 3, 1, 2)
                video_tensor_f = F.interpolate(
                    video_tensor_f, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                video_tensor = video_tensor_f.permute(0, 2, 3, 1)  # [T, H, W, C]

            video_tensor = rearrange(video_tensor, "T H W C -> C T H W")
            # Write the video
            video_fp = f"{self.local_dir}/{base_fp_wo_ext}.mp4"
            self.save_video(video_tensor.cpu().numpy(), video_fp, fps=self.fps)

            return local_path, local_path_12frames, video_fp
        return None

    def sample_first_n_views_from_data_batch(self, data_batch, n_views):
        new_data_batch = {}
        num_video_frames_per_view = data_batch["num_video_frames_per_view"]
        new_total_frames = num_video_frames_per_view * n_views
        new_total_t5_dim = CosmosTextEncoderConfig.NUM_TOKENS * n_views
        new_data_batch["video"] = data_batch["video"][:, :, 0:new_total_frames]
        new_data_batch["view_indices"] = data_batch["view_indices"][:, 0:new_total_frames]
        new_data_batch["sample_n_views"] = 0 * data_batch["sample_n_views"] + n_views
        new_data_batch["fps"] = data_batch["fps"]
        new_data_batch["t5_text_embeddings"] = data_batch["t5_text_embeddings"][:, 0:new_total_t5_dim]
        new_data_batch["t5_text_mask"] = data_batch["t5_text_mask"][:, 0:new_total_t5_dim]
        split_captions = data_batch["ai_caption"][0].split(" -- ")
        assert len(split_captions) == 6, f"Expected 6 view captions, got {len(split_captions)}"
        new_data_batch["ai_caption"] = [" -- ".join(split_captions[0:n_views])]
        new_data_batch["n_orig_video_frames_per_view"] = data_batch["n_orig_video_frames_per_view"]
        assert data_batch["ref_cam_view_idx_sample_position"].item() == -1, (
            f"ref_cam_view_idx_sample_position is not supported by batch sampling, got {data_batch['ref_cam_view_idx_sample_position']}"
        )
        new_data_batch["ref_cam_view_idx_sample_position"] = data_batch["ref_cam_view_idx_sample_position"]
        new_data_batch["camera_keys_selection"] = data_batch["camera_keys_selection"][0:n_views]
        new_data_batch["view_indices_selection"] = data_batch["view_indices_selection"]
        for key in [
            "__url__",
            "__key__",
            "__t5_url__",
            "image_size",
            "num_video_frames_per_view",
            "aspect_ratio",
            "padding_mask",
        ]:
            new_data_batch[key] = data_batch[key]
        if TRAIN_SAMPLE_N_VIEWS_KEY in data_batch:
            new_data_batch[TRAIN_SAMPLE_N_VIEWS_KEY] = 0  # Model will not apply additional sampling
        old_keys = set(list(data_batch.keys()))
        new_keys = set(list(new_data_batch.keys()))
        diff = old_keys.difference(new_keys)
        assert old_keys == new_keys, f"Expected old keys to equal new keys. Difference {diff}"
        return new_data_batch

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.dataset_name is None:
            return self.every_n_impl_multiview(
                trainer, model, None, data_batch, output_batch=output_batch, loss=loss, iteration=iteration
            )
        data_batch_sample_all = next(self.dataloader_iter)
        data_batch_sample_all[TRAIN_SAMPLE_N_VIEWS_KEY] = 0  # Model will not apply additional sampling
        if self.sample_n_views == data_batch_sample_all["sample_n_views"]:
            data_batch_sample_all = misc.to(data_batch_sample_all, **model.tensor_kwargs)
            data_batch_sample_all[model.pipe.input_video_key] = data_batch_sample_all[model.pipe.input_video_key].to(
                torch.uint8
            )
            return self.every_n_impl_multiview(
                trainer, model, None, data_batch_sample_all, output_batch=None, loss=None, iteration=iteration
            )
        data_batch_sample_n = self.sample_first_n_views_from_data_batch(data_batch_sample_all, self.sample_n_views)
        data_batch_sample_all = misc.to(data_batch_sample_all, **model.tensor_kwargs)
        data_batch_sample_all[model.pipe.input_video_key] = data_batch_sample_all[model.pipe.input_video_key].to(
            torch.uint8
        )
        data_batch_sample_n = misc.to(data_batch_sample_n, **model.tensor_kwargs)
        data_batch_sample_n[model.pipe.input_video_key] = data_batch_sample_n[model.pipe.input_video_key].to(
            torch.uint8
        )
        return self.every_n_impl_multiview(
            trainer,
            model,
            data_batch_sample_all,
            data_batch_sample_n,
            output_batch=None,
            loss=None,
            iteration=iteration,
        )

    @torch.no_grad()
    def every_n_impl_multiview(
        self, trainer, model, data_batch_sample_all, data_batch_sample_n, output_batch, loss, iteration
    ):
        if self.is_ema:
            if not model.config.pipe_config.ema.enabled:
                return
            context = partial(model.pipe.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        data_batch_for_info = data_batch_sample_all if data_batch_sample_all is not None else data_batch_sample_n
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch_for_info.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
            "sample_n_views": self.sample_n_views,
            "n_view_embed": self.n_view_embed,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        samples_img_fp = []
        with context():
            if self.is_x0:
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch_for_info,
                    output_batch,
                    loss,
                    iteration,
                )
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )
            if self.is_sample:
                for data_batch in [data_batch_sample_all, data_batch_sample_n]:
                    if data_batch is None:
                        samples_img_fp.append(None)
                        continue
                    sample_img_fp = self.sample(
                        trainer,
                        model,
                        data_batch,
                        output_batch,
                        loss,
                        iteration,
                    )
                    samples_img_fp.append(sample_img_fp)
            if self.fix_batch is not None:
                misc.to(self.fix_batch, "cpu")

            dist.barrier()
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        n_views = int(data_batch["sample_n_views"].cpu()[0])
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)
        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.pipe.get_data_and_condition(data_batch)
        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            data_batch["neg_t5_text_embeddings"] = misc.to(
                repeat(
                    self.negative_prompt_data["t5_text_embeddings"],
                    "l ... -> b (v l) ...",
                    b=batch_size,
                    v=n_views,
                ),
                **model.tensor_kwargs,
            )
            assert data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape, (
                f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            )
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]
        to_show = []
        for num_cond_frames in [0, 1]:
            for control_weight in self.control_weights:
                data_batch[NUM_CONDITIONAL_FRAMES_KEY] = num_cond_frames
                data_batch[CONTROL_WEIGHT_KEY] = control_weight
                for guidance in self.guidance:
                    sample = model.pipe.generate_samples_from_batch(
                        data_batch,
                        guidance=guidance,
                        # make sure no mismatch and also works for cp
                        state_shape=x0.shape[1:],
                        n_sample=x0.shape[0],
                        num_steps=self.num_sampling_step,
                        is_negative_prompt=True if self.use_negative_prompt else False,
                    )
                    to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())

        # Transfer2-multiview: visualize control input
        if self.ctrl_hint_keys:
            # visualize input video
            if "hint_key" in data_batch:
                hint = data_batch[data_batch["hint_key"]]
                for idx in range(0, hint.size(1), 3):
                    x_rgb = hint[:, idx : idx + 3]
                    to_show.append(x_rgb.float().cpu())
            else:
                for key in self.ctrl_hint_keys:
                    if key in data_batch and data_batch[key] is not None:
                        hint = data_batch[key]
                        log.info(f"hint: {hint.shape}")
                        to_show.append(hint.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}_{n_views}views"

        batch_size = x0.shape[0]

        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, n_views, base_fp_wo_ext)
            return local_path
        return None
