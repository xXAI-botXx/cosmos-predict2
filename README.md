# Cosmos 2 for Sound Propagation

This repo tries out the Cosmos-2 Model for the Phygen Dataset/Benchmark.


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
    docker build -t cosmos-predict2-local -f Dockerfile .
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

3. Download Pretrained model
    1. Make Huggingface Account + create a Access Token (somewhere in the settings)
    2. Go to https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World and accept their terms (you have to click on "Expand to review access" in the "You need to agree to share your contact information to access this model" area)
    3. Then go back to your bash/console and login with: huggingface-cli login -> as password use the generated token (`Add token as git credential? (Y/n)` => n)
    4. Start the downloading proces of the prediction model: `nohup huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World > download_model.log 2>&1 &` -> check progress/finish with: `cat download_model.log` or with `ps aux | grep huggingface-cli`
    5. (After a while) Copy the model to your checkpoints -> use the last line in the download_model.txt
        ```bash
        mkdir /home/tippolit/src/cosmos-predict2/checkpoints/nvidia && cp -rL /home/tippolit/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
        /home/tippolit/src/cosmos-predict2/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World && \
        chmod -R a+rwx /home/tippolit/src/cosmos-predict2/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World
        ```

<!--
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


### Running

```
# Get Device Number
nvidia-smi -L

# Find the right (previously installed) image
docker image ls

# Testing
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi
docker run --rm --gpus all --runtime=nvidia nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi

# Start docker
docker run --gpus '"device=0"' --runtime=nvidia -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v ~/src/cosmos-predict2/checkpoints:/workspace/checkpoints \
cosmos-predict2-local
# Or
docker run --gpus all --runtime=nvidia -it --rm \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v ~/src/cosmos-predict2/checkpoints:/workspace/checkpoints \
cosmos-predict2-local

# Verify Installation/Env
python /workspace/scripts/test_environment.py

# Create Embeddings (you have to already downloaded and converted the physgen dataset as described on top)
python -m scripts.get_t5_embeddings --dataset_path datasets/physgen_train
python -m scripts.get_t5_embeddings --dataset_path datasets/physgen_val
python -m scripts.get_t5_embeddings --dataset_path datasets/physgen_test

# Dataset test
python physgen_data_test.py

# Start Training -> adjust the nproc_per_node with the used gpus 
EXP=predict2_video2world_training_1a_physgen && \
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
# or
EXP=predict2_video2world_training_1a_physgen && \
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

exit
```

Or if you want to start in background:
```bash
# Start training in background
docker run --gpus '"device=0"' --runtime=nvidia -d \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v ~/src/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-train-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
EXP=predict2_video2world_training_1a_physgen && \
nohup torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=\$EXP \
    > train.log 2>&1 & tail -f train.log"
# or
docker run --gpus all --runtime=nvidia -d \
--shm-size=8g \
-v ~/src/cosmos-predict2:/workspace \
-v ~/src/cosmos-predict2/datasets:/workspace/datasets \
-v ~/src/cosmos-predict2/checkpoints:/workspace/checkpoints \
--name cosmos-train-run \
cosmos-predict2-local \
bash -c "cd /workspace && \
EXP=predict2_video2world_training_1a_physgen && \
nohup torchrun --nproc_per_node=4 --master_port=12341 \
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



<br><br><br><br>

---
# Original README Content:

---

<br><br>

<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### Paper (coming soon!) | [Website](https://research.nvidia.com/labs/dir/cosmos-predict2/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959)

Cosmos-Predict2 is a key branch of the [Cosmos World Foundation Models](https://www.nvidia.com/en-us/ai/cosmos) (WFMs) ecosystem for Physical AI, specializing in future state prediction through advanced world modeling. It offers two powerful capabilities: text-to-image generation for creating high-quality images from text descriptions, and video-to-world generation for producing visual simulations from video inputs.

We visualize the architecture of Cosmos-Predict2 in the following figure.

<p align="center">
    <img src="assets/cosmos-predict-diagram.png" alt="Cosmos-Predict Architecture Diagram" width=80%>
</p>

## News
* 2025-07-10: We released [Predict2 + NATTEN](documentations/performance.md#sparse-attention-powered-by-natten), bringing up to 2.6X end-to-end inference speedup with sparse attention ([Video](https://www.youtube.com/watch?v=o396JZsz4V4)).
* 2025-06-11: We released post-training and inference code, along with model weights. For a code walkthrough, please see this [video](https://www.youtube.com/watch?v=ibnVm6hPtxA).

## Models

* [Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-14B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1): Video + Text based future visual world generation, post-trained on GR00T Dreams GR1 dataset
* [Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID): Video + Text based future visual world generation, post-trained on GR00T Dreams DROID dataset
* [Cosmos-Predict2-2B-Sample-Action-Conditioned](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned): Video + Action based future visual world generation, post-trained on Bridge dataset
---

## Quick Start

Here is a quick example demonstrating how to use Cosmos-Predict2-2B-Video2World for video generation:

```python
import torch
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.
pipe = Video2WorldPipeline.from_config(
    config=PREDICT2_VIDEO2WORLD_PIPELINE_2B,
    dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
    text_encoder_path="checkpoints/google-t5/t5-11b",
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
