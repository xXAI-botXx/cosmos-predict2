# Cosmos 2 for Sound Propagation

This repo tries out the Cosmos-2 Model for the Phygen Dataset/Benchmark.

### Current State

PhysGen Dataset successfull transformation into the right form for Cosmos-Predict 2.

Guide (below) which helps for building your system and python environment as well as build the dataset and start training and inferencing.

Training should work now, maybe there occur errors after one epoch which have to get fixed which is not a big problem and maybe all of those error are already fixed (many of these errors got fixed, but maybe there still errors after one epoch). 

The training is started with:
```bash
docker run --gpus '"device=1,2,3"' -d \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-train-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
EXP=predict2_video2world_training_1a_physgen && \
nohup torchrun --nproc_per_node=3 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=\$EXP \
    > train.log 2>&1 & tail -f train.log"
```

Currently inference is not possible due to not enough memory (but training did actually work which makes no sense at all). There is a evaluation notebook but it is not completly finished, which is marked as `FIXME` in the [evaluation.ipynb](evaluation.ipynb).

Inference command:
```bash
docker run --rm \
--shm-size=8g \
--gpus all \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-inference-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
torchrun --nproc_per_node=4 --master_port=12341\
 -m examples.video2world \
  --model_size 2B \
  --dit_path "/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt" \
  --prompt "Make a sound_reflection propagation with the sound source in the center of the image in one step." \
  --input_path "/workspace/datasets/physgen_test_raw/osm/input_physgen_0.png" \
  --save_path /workspace/output/cache_physgen/prediction/generated_video_from_post-training.mp4 \
  --disable_guardrail \
  --disable_prompt_refiner \
  --resolution 256 \
  --fps 1 \
  --aspect_ratio "1:1" \
  --num_conditional_frames 1 \
  --num_gpus 4"
```

Output:
```bash
(base) tippolit@lecun01:~/src/cosmos-predict2$ docker run --rm \
--shm-size=8g \
--gpus all \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-inference-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
torchrun --nproc_per_node=4 --master_port=12341\
  --num_gpus 4"onal_frames 1 \t/cache_physgen/prediction/generated_video_from_post-training.mp4 \in one step." \
Resolved 162 packages in 2ms
Bytecode compiled 19462 files in 812ms
W1130 13:18:39.406000 1 torch/distributed/run.py:792] 
W1130 13:18:39.406000 1 torch/distributed/run.py:792] *****************************************
W1130 13:18:39.406000 1 torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1130 13:18:39.406000 1 torch/distributed/run.py:792] *****************************************
fatal: detected dubious ownership in repository at '/workspace'
To add an exception for this directory, call:

        git config --global --add safe.directory /workspace
[11-30 13:18:47|INFO|imaginaire/constants.py:39:print_environment_info] imaginaire.constants: Namespace(checkpoints='checkpoints', text_encoder=<TextEncoderClass.T5: 't5'>)
[11-30 13:18:47|INFO|imaginaire/constants.py:40:print_environment_info] sys.argv: ['/workspace/examples/video2world.py', '--model_size', '2B', '--dit_path', '/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', '--prompt', 'Make']
[11-30 13:18:47|INFO|imaginaire/constants.py:41:print_environment_info] args: Namespace(model_size='2B', resolution='720', fps=16, dit_path='/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', load_ema=False, prompt='Make', input_path='assets/video2world/input0.jpg', negative_prompt='The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.', aspect_ratio='16:9', num_conditional_frames=1, batch_input_json=None, guidance=7, seed=0, save_path='output/generated_video.mp4', num_gpus=1, disable_guardrail=False, offload_guardrail=False, disable_prompt_refiner=False, offload_prompt_refiner=False, offload_text_encoder=False, downcast_text_encoder=False, benchmark=False, use_cuda_graphs=False, natten=False)
[11-30 13:18:47|INFO|examples/video2world.py:210:setup_pipeline] Using dit_path: /workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt
[11-30 13:18:47|INFO|imaginaire/utils/misc.py:139:set_random_seed] Using random seed 0.
[11-30 13:18:47|WARNING|imaginaire/lazy_config/lazy.py:441:save_yaml] Config is saved using omegaconf at output/generated_video.yaml.
[11-30 13:18:47|INFO|examples/video2world.py:259:setup_pipeline] Initializing Video2WorldPipeline with model size: 2B
[11-30 13:18:47|WARNING|cosmos_predict2/pipelines/video2world.py:320:from_config] precision torch.bfloat16
fatal: detected dubious ownership in repository at '/workspace'
To add an exception for this directory, call:

        git config --global --add safe.directory /workspace
[11-30 13:18:47|INFO|imaginaire/constants.py:39:print_environment_info] imaginaire.constants: Namespace(checkpoints='checkpoints', text_encoder=<TextEncoderClass.T5: 't5'>)
[11-30 13:18:47|INFO|imaginaire/constants.py:40:print_environment_info] sys.argv: ['/workspace/examples/video2world.py', '--model_size', '2B', '--dit_path', '/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', '--prompt', 'Make']
[11-30 13:18:47|INFO|imaginaire/constants.py:41:print_environment_info] args: Namespace(model_size='2B', resolution='720', fps=16, dit_path='/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', load_ema=False, prompt='Make', input_path='assets/video2world/input0.jpg', negative_prompt='The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.', aspect_ratio='16:9', num_conditional_frames=1, batch_input_json=None, guidance=7, seed=0, save_path='output/generated_video.mp4', num_gpus=1, disable_guardrail=False, offload_guardrail=False, disable_prompt_refiner=False, offload_prompt_refiner=False, offload_text_encoder=False, downcast_text_encoder=False, benchmark=False, use_cuda_graphs=False, natten=False)
[11-30 13:18:47|INFO|examples/video2world.py:210:setup_pipeline] Using dit_path: /workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt
[11-30 13:18:47|INFO|imaginaire/utils/misc.py:139:set_random_seed] Using random seed 0.
[11-30 13:18:47|WARNING|imaginaire/lazy_config/lazy.py:441:save_yaml] Config is saved using omegaconf at output/generated_video.yaml.
[11-30 13:18:47|INFO|examples/video2world.py:259:setup_pipeline] Initializing Video2WorldPipeline with model size: 2B
[11-30 13:18:47|WARNING|cosmos_predict2/pipelines/video2world.py:320:from_config] precision torch.bfloat16
fatal: detected dubious ownership in repository at '/workspace'
To add an exception for this directory, call:

        git config --global --add safe.directory /workspace
[11-30 13:18:47|INFO|imaginaire/constants.py:39:print_environment_info] imaginaire.constants: Namespace(checkpoints='checkpoints', text_encoder=<TextEncoderClass.T5: 't5'>)
[11-30 13:18:47|INFO|imaginaire/constants.py:40:print_environment_info] sys.argv: ['/workspace/examples/video2world.py', '--model_size', '2B', '--dit_path', '/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', '--prompt', 'Make']
[11-30 13:18:47|INFO|imaginaire/constants.py:41:print_environment_info] args: Namespace(model_size='2B', resolution='720', fps=16, dit_path='/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', load_ema=False, prompt='Make', input_path='assets/video2world/input0.jpg', negative_prompt='The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.', aspect_ratio='16:9', num_conditional_frames=1, batch_input_json=None, guidance=7, seed=0, save_path='output/generated_video.mp4', num_gpus=1, disable_guardrail=False, offload_guardrail=False, disable_prompt_refiner=False, offload_prompt_refiner=False, offload_text_encoder=False, downcast_text_encoder=False, benchmark=False, use_cuda_graphs=False, natten=False)
[11-30 13:18:47|INFO|examples/video2world.py:210:setup_pipeline] Using dit_path: /workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt
[11-30 13:18:47|INFO|imaginaire/utils/misc.py:139:set_random_seed] Using random seed 0.
fatal: detected dubious ownership in repository at '/workspace'
To add an exception for this directory, call:

        git config --global --add safe.directory /workspace
[11-30 13:18:47|INFO|imaginaire/constants.py:39:print_environment_info] imaginaire.constants: Namespace(checkpoints='checkpoints', text_encoder=<TextEncoderClass.T5: 't5'>)
[11-30 13:18:47|INFO|imaginaire/constants.py:40:print_environment_info] sys.argv: ['/workspace/examples/video2world.py', '--model_size', '2B', '--dit_path', '/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', '--prompt', 'Make']
[11-30 13:18:47|INFO|imaginaire/constants.py:41:print_environment_info] args: Namespace(model_size='2B', resolution='720', fps=16, dit_path='/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt', load_ema=False, prompt='Make', input_path='assets/video2world/input0.jpg', negative_prompt='The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.', aspect_ratio='16:9', num_conditional_frames=1, batch_input_json=None, guidance=7, seed=0, save_path='output/generated_video.mp4', num_gpus=1, disable_guardrail=False, offload_guardrail=False, disable_prompt_refiner=False, offload_prompt_refiner=False, offload_text_encoder=False, downcast_text_encoder=False, benchmark=False, use_cuda_graphs=False, natten=False)
[11-30 13:18:47|INFO|examples/video2world.py:210:setup_pipeline] Using dit_path: /workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt
[11-30 13:18:47|INFO|imaginaire/utils/misc.py:139:set_random_seed] Using random seed 0.
[11-30 13:18:47|WARNING|imaginaire/lazy_config/lazy.py:441:save_yaml] Config is saved using omegaconf at output/generated_video.yaml.
[11-30 13:18:47|INFO|examples/video2world.py:259:setup_pipeline] Initializing Video2WorldPipeline with model size: 2B
[11-30 13:18:47|WARNING|cosmos_predict2/pipelines/video2world.py:320:from_config] precision torch.bfloat16
[11-30 13:18:47|WARNING|imaginaire/lazy_config/lazy.py:441:save_yaml] Config is saved using omegaconf at output/generated_video.yaml.
[11-30 13:18:47|INFO|examples/video2world.py:259:setup_pipeline] Initializing Video2WorldPipeline with model size: 2B
[11-30 13:18:47|WARNING|cosmos_predict2/pipelines/video2world.py:320:from_config] precision torch.bfloat16
[DEBUG] Temporal Window: 16
[DEBUG] Temporal Window: 16
[DEBUG] Temporal Window: 16
[DEBUG] Temporal Window: 16
[11-30 13:18:49|INFO|cosmos_predict2/tokenizers/tokenizer.py:602:_video_vae] Loading checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|SUCCESS|cosmos_predict2/tokenizers/tokenizer.py:604:_video_vae] Successfully loaded checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|INFO|cosmos_predict2/tokenizers/tokenizer.py:602:_video_vae] Loading checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|SUCCESS|cosmos_predict2/tokenizers/tokenizer.py:604:_video_vae] Successfully loaded checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|INFO|cosmos_predict2/tokenizers/tokenizer.py:602:_video_vae] Loading checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|INFO|cosmos_predict2/tokenizers/tokenizer.py:602:_video_vae] Loading checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|SUCCESS|cosmos_predict2/tokenizers/tokenizer.py:604:_video_vae] Successfully loaded checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:18:49|SUCCESS|cosmos_predict2/tokenizers/tokenizer.py:604:_video_vae] Successfully loaded checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
[11-30 13:21:16|INFO|imaginaire/auxiliary/text_encoder.py:345:__init__] T5 Text encoder model instantiated
Traceback (most recent call last):
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/examples/video2world.py", line 417, in <module>
    pipe = setup_pipeline(args)
  File "/workspace/examples/video2world.py", line 260, in setup_pipeline
    pipe = Video2WorldPipeline.from_config(
  File "/workspace/cosmos_predict2/pipelines/video2world.py", line 346, in from_config
    pipe.text_encoder = get_cosmos_text_encoder(
  File "/workspace/imaginaire/auxiliary/text_encoder.py", line 429, in get_cosmos_text_encoder
    return CosmosT5TextEncoder(config=config.t5, device=device, torch_dtype=torch_dtype)
  File "/workspace/imaginaire/auxiliary/text_encoder.py", line 342, in __init__
    self.text_encoder = T5EncoderModel.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype).to(device)
  File "/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 3698, in to
    return super().to(*args, **kwargs)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  [Previous line repeated 4 more times]
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 57.00 MiB is free. Process 2213429 has 5.49 GiB memory in use. Process 2213427 has 4.63 GiB memory in use. Process 2213428 has 18.89 GiB memory in use. Process 2213430 has 10.39 GiB memory in use. Of the allocated memory 4.98 GiB is allocated by PyTorch, and 28.38 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/examples/video2world.py", line 417, in <module>
    pipe = setup_pipeline(args)
  File "/workspace/examples/video2world.py", line 260, in setup_pipeline
    pipe = Video2WorldPipeline.from_config(
  File "/workspace/cosmos_predict2/pipelines/video2world.py", line 346, in from_config
    pipe.text_encoder = get_cosmos_text_encoder(
  File "/workspace/imaginaire/auxiliary/text_encoder.py", line 429, in get_cosmos_text_encoder
    return CosmosT5TextEncoder(config=config.t5, device=device, torch_dtype=torch_dtype)
  File "/workspace/imaginaire/auxiliary/text_encoder.py", line 342, in __init__
    self.text_encoder = T5EncoderModel.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype).to(device)
  File "/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 3698, in to
    return super().to(*args, **kwargs)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  [Previous line repeated 4 more times]
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 77.00 MiB is free. Process 2213429 has 5.49 GiB memory in use. Process 2213427 has 4.63 GiB memory in use. Process 2213428 has 18.89 GiB memory in use. Process 2213430 has 10.38 GiB memory in use. Of the allocated memory 9.86 GiB is allocated by PyTorch, and 36.33 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Traceback (most recent call last):
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/examples/video2world.py", line 417, in <module>
    pipe = setup_pipeline(args)
  File "/workspace/examples/video2world.py", line 260, in setup_pipeline
    pipe = Video2WorldPipeline.from_config(
  File "/workspace/cosmos_predict2/pipelines/video2world.py", line 346, in from_config
    pipe.text_encoder = get_cosmos_text_encoder(
  File "/workspace/imaginaire/auxiliary/text_encoder.py", line 429, in get_cosmos_text_encoder
    return CosmosT5TextEncoder(config=config.t5, device=device, torch_dtype=torch_dtype)
  File "/workspace/imaginaire/auxiliary/text_encoder.py", line 342, in __init__
    self.text_encoder = T5EncoderModel.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype).to(device)
  File "/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 3698, in to
    return super().to(*args, **kwargs)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  [Previous line repeated 4 more times]
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 37.00 MiB is free. Process 2213429 has 5.49 GiB memory in use. Process 2213427 has 4.67 GiB memory in use. Process 2213428 has 18.89 GiB memory in use. Process 2213430 has 10.38 GiB memory in use. Of the allocated memory 4.17 GiB is allocated by PyTorch, and 20.38 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 21.88it/s]
Traceback (most recent call last):
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/root/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/workspace/examples/video2world.py", line 417, in <module>
    pipe = setup_pipeline(args)
  File "/workspace/examples/video2world.py", line 260, in setup_pipeline
    pipe = Video2WorldPipeline.from_config(
  File "/workspace/cosmos_predict2/pipelines/video2world.py", line 361, in from_config
    pipe.prompt_refiner = CosmosReason1(
  File "/workspace/cosmos_predict2/auxiliary/cosmos_reason1.py", line 132, in __init__
    self.model = self.model.to("cuda")
  File "/workspace/.venv/lib/python3.10/site-packages/transformers/modeling_utils.py", line 3698, in to
    return super().to(*args, **kwargs)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 17.00 MiB is free. Process 2213429 has 5.49 GiB memory in use. Process 2213427 has 4.67 GiB memory in use. Process 2213428 has 18.91 GiB memory in use. Process 2213430 has 10.38 GiB memory in use. Of the allocated memory 18.42 GiB is allocated by PyTorch, and 10.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
W1130 13:21:20.183000 1 torch/distributed/elastic/multiprocessing/api.py:897] Sending process 363 closing signal SIGTERM
W1130 13:21:20.184000 1 torch/distributed/elastic/multiprocessing/api.py:897] Sending process 364 closing signal SIGTERM
W1130 13:21:20.187000 1 torch/distributed/elastic/multiprocessing/api.py:897] Sending process 365 closing signal SIGTERM
E1130 13:21:21.592000 1 torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 3 (pid: 366) of binary: /workspace/.venv/bin/python3
Traceback (most recent call last):
  File "/workspace/.venv/bin/torchrun", line 10, in <module>
    sys.exit(main())
  File "/workspace/.venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/workspace/.venv/lib/python3.10/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/workspace/.venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/workspace/.venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
examples.video2world FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-11-30_13:21:20
  host      : 9d68ffecd72f
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 366)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```


### Downloading Dataset

1. Install Anaconda
    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    bash Anaconda3-2024.10-1-Linux-x86_64.sh

    # open new bash!
    export PATH="$HOME/anaconda3/bin:$PATH"
    conda init
    ```
2. Install Dependencies
    ```bash
    conda create -n physgen-dataset python=3.10 pip -y
    conda activate physgen-dataset
    conda install -c conda-forge cmake -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install sympy timm tqdm scikit-learn pyyaml pydantic pip install pillow wandb ipython ipykernel scikit-image pytorch-msssim pandas prime_printer shapely opencv-python datasets==3.6.0
    ```
3. Start Downloading Script
    ```bash
    conda activate physgen-dataset
    cd ~/src/cosmos-predict2
    python physgen_dataset.py \
        --output_real_path ./datasets/physgen_train_raw/real \
        --output_osm_path ./datasets/physgen_train_raw/osm \
        --variation sound_reflection \
        --input_type osm \
        --output_type standard \
        --data_mode train
    python physgen_dataset.py \
        --output_real_path ./datasets/physgen_test_raw/real \
        --output_osm_path ./datasets/physgen_test_raw/osm \
        --variation sound_reflection \
        --input_type osm \
        --output_type standard \
        --data_mode test
    python physgen_dataset.py \
        --output_real_path ./datasets/physgen_val_raw/real \
        --output_osm_path ./datasets/physgen_val_raw/osm \
        --variation sound_reflection \
        --input_type osm \
        --output_type standard \
        --data_mode validation
    ```
4. Convert the dataset(s)
    ```bash
    python physgen_cosmos_converter.py \
        --input_folder ./datasets/physgen_train_raw/osm \
        --target_folder ./datasets/physgen_train_raw/real \
        --output_folder ./datasets/physgen_train \
        --variation sound_reflection 
    python physgen_cosmos_converter.py \
        --input_folder ./datasets/physgen_test_raw/osm \
        --target_folder ./datasets/physgen_test_raw/real \
        --output_folder ./datasets/physgen_test \
        --variation sound_reflection
    python physgen_cosmos_converter.py \
        --input_folder ./datasets/physgen_val_raw/osm \
        --target_folder ./datasets/physgen_val_raw/real \
        --output_folder ./datasets/physgen_val \
        --variation sound_reflection  
    ```


### Installation

1. Clone Repo
    ```bash
    git clone https://github.com/xXAI-botXx/cosmos-predict2.git
    cd cosmos-predict2
    ```

2. Docker Installation
    ```bash
    # --- Docker ---
    # Make Sure Docker is installed
    docker --version
    which docker

    # If not run:
    sudo apt update
    sudo apt install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker’s official GPG key:
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up the Docker repository:
    echo \
    "deb [arch=$(dpkg --print-architecture) \
    signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine:
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # --- Nvidia Container Toolkit ---
    # Make sure nvidia container toolkit is installed
    dpkg -l | grep nvidia-container-toolkit

    # Else install it -> see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

    # --- Cosmos Container ---
    cd ~/src/cosmos-predict2
    docker build --no-cache -t cosmos-predict2-local -f Dockerfile .
    ```
<!--
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
    sudo apt-get install -y \
        nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
        nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
        libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
        libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
-->
<!--
2. Setup Docker Container (by installing nvidia container toolkit)
    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
    ```
-->
<!--
2. Install Anaconda
    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    bash Anaconda3-2024.10-1-Linux-x86_64.sh

    # open new bash!
    export PATH="$HOME/anaconda3/bin:$PATH"
    conda init
    ```

3. Setup Env
    ```bash
    # install uv + just
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv tool install rust-just

    # install dependencies/conda env
    uv lock

    wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
    sudo sh cuda_12.6.0_560.28.03_linux.run
    export PATH=/usr/local/cuda-12.6/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
    source ~/.bashrc




    just install-conda
    conda activate cosmos-predict2
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    just install-conda
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126
    just install-conda
    pip install \
        nvidia-cusparselt-cu12==0.6.3 \
        nvidia-cublas-cu12==12.6.4.1 \
        nvidia-cuda-runtime-cu12==12.6.77 \
        nvidia-cudnn-cu12==9.5.1.17 \
        nvidia-cusolver-cu12==11.7.1.2 \
        nvidia-cufft-cu12==11.3.0.4 \
        nvidia-cusparse-cu12==12.5.4.2 \
        nvidia-curand-cu12==10.3.7.77 \
        nvidia-nvjitlink-cu12==12.6.85 \
        nvidia-cuda-nvrtc-cu12==12.6.77 \
        nvidia-cuda-cupti-cu12==12.6.80

    # test
    python -c "import torch; print(torch.__version__)"
    pip check

    # export of your env
    pip freeze > requirements.txt

    # install train dependencies
    just install-training
    ```

3. Install project
    ```bash
    pip install -e .
    ```
-->
<!--
`nohup huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World > download_model.log 2>&1 &` -> check progress/finish with: `cat download_model.log` or with `ps aux | grep huggingface-cli` **OR**
-->
3. Download Pretrained model
    1. Make Huggingface Account + create a Access Token (somewhere in the settings) 
    2. Run `pip install huggingface_hub[cli]`
    3. Go to https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World and accept their terms (you have to click on "Expand to review access" in the "You need to agree to share your contact information to access this model" area) -> also here: https://huggingface.co/meta-llama/Llama-Guard-3-8B & https://huggingface.co/nvidia/Cosmos-Guardrail1
    4. Then go back to your bash/console and login with: `hf auth login` (use `hf auth --help`) -> as password use the generated token (`Add token as git credential? (Y/n)` => n)
    5. Start the downloading process of the prediction model: run `export CUR_LOCATION=$(pwd) && mkdir /ssd0/tippolit/cosmos-predict2 && cd /ssd0/tippolit/cosmos-predict2 && export HF_HOME="/ssd0/tippolit/hf_cache" && /home/tippolit/src/cosmos-predict2/scripts/download_checkpoints.py --model_types video2world --model_sizes 2B && cd "$CUR_LOCATION"`  -> default uses: `~/.cache/huggingface/` & if folder is existing: `rm -r /ssd0/tippolit/cosmos-predict2/`
    6. Check: `ls /ssd0/tippolit/cosmos-predict2/checkpoints`
<!--
    6. (Optional) Copy the model to your checkpoints -> use the last line in the download_model.txt
        ```bash
        mkdir /home/tippolit/src/cosmos-predict2/checkpoints/nvidia && cp -rL /home/tippolit/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
        /home/tippolit/src/cosmos-predict2/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World && \
        chmod -R a+rwx /home/tippolit/src/cosmos-predict2/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World
        ```
4. Donwload Tokenizer (https://huggingface.co/google-t5/t5-11b)
    1. Make Huggingface Account + create a Access Token (somewhere in the settings)
    2. Then go back to your bash/console and login with: huggingface-cli login -> as password use the generated token (`Add token as git credential? (Y/n)` => n)
    3. Start the downloading proces of the prediction model: `nohup huggingface-cli download google-t5/t5-11b > download_tokenizer.log 2>&1 &` -> check progress/finish with: `cat download_tokenizer.log` or with `ps aux | grep huggingface-cli`
    4. (After a while) Copy the model to your checkpoints -> use the last line in the download_tokenizer.txt
        ```bash
        mkdir /home/tippolit/src/cosmos-predict2/checkpoints/google-t5 && cp -r /home/tippolit/.cache/huggingface/hub/models--google-t5--t5-11b/snapshots/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3 \
        /home/tippolit/src/cosmos-predict2/checkpoints/google-t5/t5-11b/
        ```
-->
5. Testing:
    - Build Cuda Test Image
        ```bash
        docker build --no-cache -t cuda-check -f pytorch_cuda_check.Dockerfile .
        ```
    - Run Cuda Test Image/Container
        ```bash
        docker run --runtime=nvidia --rm --gpus all cuda-check
        ```


### Running

<!--
docker run --rm nvcr.io/nvidia/pytorch:25.04-py3 nvcc --version
-->

Old versions need: `--runtime=nvidia`

Get Device Numbers: `nvidia-smi -L`

Find the right (previously installed) image: `docker image ls`

Check versions (multiline commands):
```bash
echo -e "\n> CUDA TEST start <\n\n---------\nHOST GPU Versions\n---------\n" && \
echo "NVIDIA Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" && echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | sed 's/,//')" && \
echo -e "\n---------\nDOCKER GPU Versions\n---------\n" && \
docker run --gpus all --runtime=nvidia --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
cosmos-predict2-local \
/bin/bash -c 'echo "CUDA Version:" $(nvcc --version | grep release | awk "{print \$6}" | sed "s/,//")' && \
echo -e "\n> CUDA TEST end <\n" 
```

Testing:<br>
`docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi`

Then continue testing:
```
# Start docker (testwise)
docker run --gpus '"device=0,1"' -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
cosmos-predict2-local
# Or
docker run --gpus all -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
cosmos-predict2-local
# Or in my case
docker run --gpus all -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
cosmos-predict2-local

# in bash in docker run following commands:

# Verify Installation/Env
python /workspace/scripts/test_environment.py
python /workspace/test_cuda.py

# Another tests
nvidia-smi
python3 -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

exit
```

Create embeddings of your data (start again a docker as for the testing):
```bash
# Create Embeddings (you have to already downloaded and converted the physgen dataset as described on top)
python -m scripts.get_t5_embeddings --dataset_path datasets/physgen_train
python -m scripts.get_t5_embeddings --dataset_path datasets/physgen_val
python -m scripts.get_t5_embeddings --dataset_path datasets/physgen_test

# Dataset test
docker run --gpus '"device=0"' -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
cosmos-predict2-local bash -c "python physgen_data_test.py"

exit
```


Start your training:
```bash
# Start training in background
# --gpus '"device=2,3"' or --gpus all
# -> also adjust the amount of GPUs then: --nproc_per_node=2
docker run --gpus '"device=1,2,3"' -d \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-train-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
EXP=predict2_video2world_training_1a_physgen && \
nohup torchrun --nproc_per_node=3 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=\$EXP \
    > train.log 2>&1 & tail -f train.log"

# See the logs after training
docker logs -f cosmos-train-run
#      or
cat ~/src/cosmos-predict2/train.log

# Check if it is still alive
docker ps

# Stop Container
docker stop cosmos-train-run && docker rm /cosmos-train-run
```

Test inference:
```bash
docker run --gpus all --runtime=nvidia \
-it \
--rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-inference-run \
cosmos-predict2-local \
bash -c "cd /workspace && nvidia-smi && python -i"

docker run --gpus '"device=0"' --runtime=nvidia \
-it \
--rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-inference-run \
cosmos-predict2-local \
bash -c "cd /workspace && nvidia-smi && python -i"

*not working:
Then run:
    - import torch; print(torch.cuda.is_available())
    - print(torch.cuda.get_device_name(0))
    - exit()
```

```bash
docker run --gpus all -it --rm \
  -v ~/src/cosmos-predict2:/workspace \
  -v ~/src/cosmos-predict2/datasets:/workspace/datasets \
  -v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
  cosmos-predict2-local

then:
env | grep CUDA
ls /dev/nvidia*
nvidia-smi
which python
```

Test mounting:
```bash
docker run --gpus all -d --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/checkpoints \
--name cosmos-inference-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
( \
echo '--- 1. Listing /workspace ---' && ls -l /workspace ; \
echo -e '\n--- 2. Listing /checkpoints ---\n' && ls -l /checkpoints ; \
) > docker_info.log 2>&1
"
```

**Inference:**
<!-- runtime=nvidia 

EXP=predict2_video2world_training_1a_physgen && \
nohup torchrun --nproc_per_node=4 --master_port=12341 \
    -m inference --config=cosmos_predict2/configs/base/config.py --experiment=\$EXP \
    > inference.log 2>&1 & tail -f inference.log

--gpus "device=1,2,3"
or
--gpus '"device=1,2,3"'

uv pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
-->

```bash
docker run --rm \
--shm-size=8g \
--gpus all \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-inference-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
torchrun --nproc_per_node=4 --master_port=12341\
 -m examples.video2world \
  --model_size 2B \
  --dit_path "/workspace/checkpoints/posttraining/video2world/1a_physgen/checkpoints/model/iter_000020000.pt" \
  --prompt "Make a sound_reflection propagation with the sound source in the center of the image in one step." \
  --input_path "/workspace/datasets/physgen_test_raw/osm/input_physgen_0.png" \
  --save_path /workspace/output/cache_physgen/prediction/generated_video_from_post-training.mp4 \
  --disable_guardrail \
  --disable_prompt_refiner \
  --resolution 256 \
  --fps 1 \
  --aspect_ratio "1:1" \
  --num_conditional_frames 1 \
  --num_gpus 4"

# See the logs after training
docker logs -f cosmos-inference-run
#      or
cat ~/src/cosmos-predict2/inference.log

# Check if it is still alive
docker ps

# Stop Container
docker stop cosmos-inference-run && docker rm /cosmos-inference-run
```

For whole evaluation see: [evaluation.ipynb](./evaluation.ipynb)

<!--
Putting the checkpoints to another disk:
    1. Looking the disks
df -h
df -h ~/src/cosmos-predict2
ls /ssd0
ls /ssd0/tippolit

    2. Making the folder and transfer data
mkdir /ssd0/tippolit/cosmos-predict2
rsync -aP ~/src/cosmos-predict2/checkpoints /ssd0/tippolit/checkpoints

docker run --gpus '"device=0,1,2,3"' --runtime=nvidia -d \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/checkpoints:/workspace/checkpoints \
--name cosmos-train-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
EXP=predict2_video2world_training_1a_physgen && \
nohup torchrun --nproc_per_node=4 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=\$EXP \
    > train.log 2>&1 & tail -f train.log"

# Testing
docker run --gpus '"device=0,1,2,3"' --runtime=nvidia -it --shm-size=8g --rm --name test-gpu cosmos-predict2-local bash
    -> then: 
        - nvidia-smi
        - python -c "import torch; print(torch.cuda.is_available())"
        - python3 -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
-->


If you have problems with your tokenizer try this:
```bash
docker run --gpus all --runtime=nvidia -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v /ssd0/tippolit/checkpoints/checkpoints:/workspace/checkpoints \
cosmos-predict2-local

huggingface-cli download google-t5/t5-11b --local-dir checkpoints/google-t5/t5-11b --local-dir-use-symlinks False
```


<br><br><br><br>

---
# Original README Content:

---

<br><br>

<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

<h1 align="center">

> 🚨 **Update Notice**  
>
> The latest version of our Cosmos-Predict is now live!
>
> 👉 [**Cosmos-Predict2.5**](https://github.com/nvidia-cosmos/cosmos-predict2.5)

We recommend all users migrate to the new version for improved performance, features, and continued support.
</h1>

### Paper (coming soon!) | [Website](https://research.nvidia.com/labs/dir/cosmos-predict2/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959) | [PyPI](https://pypi.org/project/cosmos-predict2/)

Cosmos-Predict2 is a key branch of the [Cosmos World Foundation Models](https://www.nvidia.com/en-us/ai/cosmos) (WFMs) ecosystem for Physical AI, specializing in future state prediction through advanced world modeling. It offers two powerful capabilities: text-to-image generation for creating high-quality images from text descriptions, and video-to-world generation for producing visual simulations from video inputs.

We visualize the architecture of Cosmos-Predict2 in the following figure.

<p align="center">
    <img src="assets/cosmos-predict-diagram.png" alt="Cosmos-Predict Architecture Diagram" width=80%>
</p>

## News
* 2025-08-22: Cosmos-Predict2 is now available on [PyPI](https://pypi.org/project/cosmos-predict2/)! See [Getting Started](#getting-started) for usage.
* 2025-08-21: Cosmos-Predict2 now has pre-built dependencies! See [Setup Guide](documentations/setup.md).
* 2025-08-15: We released the [0.6B Text2Image](documentations/inference_text2image.md) model with fast tokenizer support!
* 2025-07-10: We released [Predict2 + NATTEN](documentations/performance.md#sparse-attention-powered-by-natten), bringing up to 2.6X end-to-end inference speedup with sparse attention ([Video](https://www.youtube.com/watch?v=o396JZsz4V4)).
* 2025-06-11: We released post-training and inference code, along with model weights. For a code walkthrough, please see this [video](https://www.youtube.com/watch?v=ibnVm6hPtxA).

## Models

* [Cosmos-Predict2-0.6B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-14B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1): Video + Text based future visual world generation, post-trained on GR00T Dreams GR1 dataset
* [Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID): Video + Text based future visual world generation, post-trained on GR00T Dreams DROID dataset
* [Cosmos-Predict2-2B-Sample-Action-Conditioned](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned): Video + Action based future visual world generation, post-trained on Bridge dataset
---

## Getting Started

System Requirements:

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver compatible with CUDA 12.6
* Linux x86-64
* glibc>=2.31 (e.g Ubuntu >=22.04)
* Python 3.10

We **HIGHLY** recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Cosmos-Predict2 can be installed with `pip` (requires `python==3.10`):

```shell
uv venv --python 3.10 --allow-existing
uv pip install -U "cosmos-predict2[cu126]" --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple
```

[Example Project](examples/project/README.md)

To run the repository examples below, please follow the [Setup Guide](documentations/setup.md)

## Diffusers

Cosmos-Predict2 is included in [`diffusers>=0.34.0`](https://huggingface.co/docs/transformers/en/index).

Run example inference scripts:

* [Text2Image](scripts/hf_text2image.py)

  ```shell
  ./scripts/hf_text2image.py output/hf_text2image --prompt "assets/text2image/example_prompt.txt" -v
  ```

* [Video2World](scripts/hf_video2world.py)

  ```shell
  ./scripts/hf_video2world.py output/hf_video2world --prompt "assets/video2world/example_prompt.txt" --image "assets/video2world/example_input.jpg" -v
  ```

## Quick Start

Here is a quick example demonstrating how to use Cosmos-Predict2-2B-Video2World for video generation:

```python
import torch
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.
pipe = Video2WorldPipeline.from_config(
    config=get_cosmos_predict2_video2world_pipeline(model_size="2B"),
    dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
)

# Specify the input image path and text prompt.
image_path = "assets/video2world/example_input.jpg"
prompt = "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."

# Run the video generation pipeline.
video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.
save_image_or_video(video, "output/test.mp4", fps=16)
```

**Input prompt:**
> A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation.

| Input image | Output video |
|-------------|--------------|
| ![Input Image](assets/video2world/example_input.jpg) | <video width="512" src="https://github-production-user-asset-6210df.s3.amazonaws.com/8789158/454153937-f015a579-1a8c-4c7f-8683-de2913e1c2f4.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250611%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250611T235940Z&X-Amz-Expires=300&X-Amz-Signature=97df8ab45e8de22d94d10b01eb6591fcdc234bf3faed4f3fe50f50d92722af21&X-Amz-SignedHeaders=host"></video> |

---

## User Guide
Our [setup guide](documentations/setup.md) provides complete information on
* [System requirements](documentations/setup.md#system-requirements): Detailed hardware and software prerequisites
* [Installation](documentations/setup.md#installation): Step-by-step setup with both Conda and Docker options
* [Downloading checkpoints](documentations/setup.md#downloading-checkpoints): Instructions for obtaining model weights
* [Troubleshooting](documentations/setup.md#troubleshooting): Solutions for common installation and CUDA compatibility issues

For inference examples and usage
* **[Text2Image Inference](documentations/inference_text2image.md)**: Guide for generating high-quality images from text prompts
* **[Video2World Inference](documentations/inference_video2world.md)**: Guide for generating videos from images/videos with text prompts, including:
  * Single and batch processing
  * Multi-frame conditioning
  * Multi-GPU inference for faster generation
  * Using the prompt refiner
  * Rejection sampling for quality improvement
* **[Text2World Inference](documentations/inference_text2world.md)**: Guide for generating videos directly from text prompts, including:
  * Single and batch processing
  * Multi-GPU inference for faster generation

For post-training customization
* **[Video2World Post-training guide](documentations/post-training_video2world.md)**: General guide to the video2world training system in the codebase.
* **[Video2World Post-training on Cosmos-NeMo-Assets](documentations/post-training_video2world_cosmos_nemo_assets.md)**: Case study for post-training on Cosmos-NeMo-Assets data
* **[Video2World Post-training on fisheye-view AgiBotWorld-Alpha dataset](documentations/post-training_video2world_agibot_fisheye.md)**: Case study for post-training on fisheye-view robot videos from AgiBotWorld-Alpha dataset.
* **[Video2World Post-training on GR00T Dreams GR1 and DROID datasets](documentations/post-training_video2world_gr00t.md)**: Case study for post-training on GR00T Dreams GR1 and DROID datasets.
* **[Video2World Action-conditioned Post-training on Bridge dataset](documentations/post-training_video2world_action.md)**: Case study for action-conditioned post-training on Bridge dataset.
* **[Text2Image Post-training guide](documentations/post-training_text2image.md)**: General guide to the text2image training system in the codebase.
* **[Text2Image Post-training on Cosmos-NeMo-Assets](documentations/post-training_text2image_cosmos_nemo_assets.md)**: Case study for post-training on Cosmos-NeMo-Assets image data.

Our [performance guide](documentations/performance.md) includes
* [Hardware requirements](documentations/performance.md#hardware-requirements): Recommended GPU configurations and memory requirements
* [Performance benchmarks](documentations/performance.md#performance-benchmarks): Detailed speed and quality comparisons across different GPU architectures
* [Model selection guide](documentations/performance.md#model-selection-guide): Practical advice for choosing between 2B and 14B variants based on your needs

---

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

---

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

This model includes safety and content moderation features powered by Llama Guard 3. Llama Guard 3 is used solely as a content input filter and is subject to its own license.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
