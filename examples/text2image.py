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

import argparse
import json
import os

from imaginaire.auxiliary.text_encoder import CosmosTextEncoder
from imaginaire.lazy_config.lazy import LazyConfig

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.base.config_text2image import (
    get_cosmos_predict2_text2image_pipeline,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from imaginaire.constants import (
    CosmosPredict2Text2ImageModelSize,
    CosmosPredict2Video2WorldAspectRatio,
    get_cosmos_predict2_text2image_checkpoint,
    print_environment_info,
)
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts

_DEFAULT_POSITIVE_PROMPT = "A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to Image Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=CosmosPredict2Text2ImageModelSize.__args__,
        default="2B",
        help="Size of the model to use for text-to-image generation",
    )
    parser.add_argument(
        "--distill_steps",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help="1~4 for timestep-distilled inference; 0 for the original non-distilled model",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )
    parser.add_argument("--prompt", type=str, default=_DEFAULT_POSITIVE_PROMPT, help="Text prompt for image generation")
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'prompt' and 'output_image' fields.",
    )
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt for image generation")
    parser.add_argument(
        "--aspect_ratio",
        choices=CosmosPredict2Video2WorldAspectRatio.__args__,
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_image.jpg",
        help="Path to save the generated image (include file extension)",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the inference.")
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Use fast tokenizer for generation.",
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace, text_encoder: CosmosTextEncoder | None = None) -> Text2ImagePipeline:
    print_environment_info(args)

    config = get_cosmos_predict2_text2image_pipeline(model_size=args.model_size, fast_tokenizer=args.use_fast_tokenizer)
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = get_cosmos_predict2_text2image_checkpoint(
            model_size=args.model_size, fast_tokenizer=args.use_fast_tokenizer, distilled=args.distill_steps > 0
        )
    log.info(f"Using dit_path: {dit_path}")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    config.guardrail_config.offload_model_to_cpu = args.offload_guardrail

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Save config
    output_path = os.path.splitext(args.save_path)[0]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    LazyConfig.save_yaml(config, f"{output_path}.yaml")

    # Check if we're in a distributed environment (called from text2world)
    is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()

    if is_distributed:
        # We're in a multi-GPU text2world context - only initialize on rank 0
        from imaginaire.utils.distributed import get_rank

        rank = get_rank()

        if rank == 0:
            log.info("Rank 0: Initializing Text2ImagePipeline for text2world")
            # Load models only on rank 0
            log.info(f"Initializing Text2ImagePipeline with model size: {args.model_size}")
            pipe = Text2ImagePipeline.from_config(
                config=config,
                dit_path=dit_path,
                device="cuda",
                torch_dtype=torch.bfloat16,
                load_ema_to_reg=args.load_ema,
                distill_steps=args.distill_steps,
            )
            return pipe
        else:
            log.info(f"Rank {rank}: Skipping Text2ImagePipeline initialization - will wait for rank 0")
            return None  # Return None for non-rank-0 processes
    else:
        # We're running as standalone text2image script
        # Only initialize distributed if num_gpus > 1 AND we're running standalone
        if hasattr(args, "num_gpus") and args.num_gpus > 1:
            log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")

            # Check if distributed environment is already initialized
            if not parallel_state.is_initialized():
                distributed.init()
                parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
                log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
            else:
                log.info("Distributed environment already initialized, skipping initialization")
                # Check if we need to reinitialize with different context parallel size
                current_cp_size = parallel_state.get_context_parallel_world_size()
                if current_cp_size != args.num_gpus:
                    log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                    log.warning("Using existing context parallel configuration")
                else:
                    log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

        # Load models for standalone execution
        log.info(f"Initializing Text2ImagePipeline with model size: {args.model_size}")
        pipe = Text2ImagePipeline.from_config(
            config=config,
            dit_path=dit_path,
            use_text_encoder=text_encoder is None,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=args.load_ema,
            distill_steps=args.distill_steps,
        )

        # Set the provided text encoder if one was passed
        if text_encoder is not None:
            pipe.text_encoder = text_encoder

        return pipe


def process_single_generation(
    pipe: Text2ImagePipeline,
    prompt: str,
    output_path: str,
    negative_prompt: str,
    aspect_ratio: str,
    seed: int,
    use_cuda_graphs: bool,
    benchmark: bool,
) -> bool:
    log.info(f"Running Text2ImagePipeline\nprompt: {prompt}")

    # When benchmarking, run inference 4 times, exclude the 1st due to warmup and average time.
    num_repeats = 4 if benchmark else 1
    time_sum = 0
    for i in range(num_repeats):
        # Generate image
        if benchmark and i > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        if benchmark and i > 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            time_sum += elapsed
            log.info(f"[iter {i} / {num_repeats - 1}] Generation time: {elapsed:.1f} seconds.")
    if benchmark:
        time_avg = time_sum / (num_repeats - 1)
        log.critical(f"The benchmarked generation time for Text2ImagePipeline is {time_avg:.1f} seconds.")

    if image is not None:
        # save the generated image
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated image to: {output_path}")
        save_image_or_video(image, output_path)
        log.success(f"Successfully saved image to: {output_path}")
        # save the prompts used to generate the video
        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")
        return True
    return False


def generate_image(args: argparse.Namespace, pipe: Text2ImagePipeline) -> None:
    if args.benchmark:
        log.warning(
            "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
        )
    # Text-to-image
    if args.batch_input_json is not None:
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json) as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(batch_inputs):
            log.info(f"Processing batch item {idx + 1}/{len(batch_inputs)}")
            prompt = item.get("prompt", "")
            output_image = item.get("output_image", f"output_{idx}.jpg")

            if not prompt:
                log.warning(f"Skipping item {idx}: Missing prompt")
                continue

            process_single_generation(
                pipe=pipe,
                prompt=prompt,
                output_path=output_image,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                seed=args.seed,
                use_cuda_graphs=args.use_cuda_graphs,
                benchmark=args.benchmark,
            )
    else:
        if args.use_cuda_graphs:
            log.warning(
                "Using CUDA Graphs for a single inference call may not be beneficial because of overhead of Graphs creation."
            )
        process_single_generation(
            pipe=pipe,
            prompt=args.prompt,
            output_path=args.save_path,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            seed=args.seed,
            use_cuda_graphs=args.use_cuda_graphs,
            benchmark=args.benchmark,
        )

    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe = setup_pipeline(args)
        generate_image(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
