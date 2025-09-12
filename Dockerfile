# # Use NVIDIA PyTorch container as base image
# FROM nvcr.io/nvidia/pytorch:25.04-py3
# #          for CUDA 12.2 with Nvidia Driver 535
# # FROM nvcr.io/nvidia/pytorch:23.09-py3  
# # FROM nvcr.io/nvidia/pytorch:24.02-py3

# # CUDA reinstallment
# # Remove existing CUDA toolkit
# RUN rm -rf /usr/local/cuda
# RUN rm -rf /usr/local/cuda-12.9

# # Add Nvidia CUDA Repo
# RUN apt-get update && apt-get install -y software-properties-common wget gnupg2

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
# RUN apt-get update

# # Install CUDA Toolkit 12.0 (deb-Pakete from NVIDIA-Repository)
# # RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# # RUN dpkg -i cuda-keyring_1.0-1_all.deb
# # RUN apt-get update && apt-get install -y wget
# # RUN wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
# # RUN apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb
# # RUN apt-get update
# # RUN apt-get install -y cuda-toolkit-12-0
# # RUN apt-get install -y cuda-runtime-12-0

# RUN apt install -y cuda-12-5
# RUN apt-get install -y cuda-toolkit-12-5
# RUN apt-get install -y cuda-runtime-12-5
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-libraries-12-5

# # RUN apt-get update && apt-cache search libcusparse
# # RUN apt-cache search libcusparselt

# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     libcublas-12-5 \
# #     libcusparse-12-5 \
# #     libcusparseLt-12-5 \
# #     && rm -rf /var/lib/apt/lists/*

# # Check
# RUN ls /usr/local/cuda-12.5
# RUN ls /usr/local/cuda-12.5/lib64
# # RUN ls /usr/local/cuda-12.0/lib64/libcusparseLt.so.0

# # Alternative: Use CUDA 12.0 Base Installer:
# # RUN wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.60.13_linux.run
# # RUN sh cuda_12.0.1_525.60.13_linux.run --silent --toolkit

# # RUN apt-get update && apt-get install -y cuda-toolkit-12-0

# # Update CUDA path
# RUN ls -l /usr/local/cuda*
# ENV CUDA_HOME=/usr/local/cuda-12.5
# ENV CUDA_PATH=$CUDA_HOME/bin
# ENV PATH=$CUDA_HOME/bin:$PATH
# ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64

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

# Dockerfile using uv environment.

ARG TARGETPLATFORM
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# # Install basic tools
# RUN apt-get update && apt-get install -y git tree ffmpeg wget
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# RUN if [[ ${TARGETPLATFORM} == 'linux/amd64' ]]; then ln -s /lib64/libcuda.so.1 /lib64/libcuda.so; fi
# RUN apt-get install -y libglib2.0-0
# # RUN apt-get update && apt-get install -y cuda-toolkit-12-1
# # RUN apt-get update && apt-get install -y cuda-nvcc-12-1
# RUN [ -f /etc/pip/constraint.txt ] && sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt || echo "File not found, skipping sed"
# # RUN sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt

# # RUN if [[ ${TARGETPLATFORM} == 'linux/amd64' ]]; then ln -s /lib64/libcuda.so.1 /lib64/libcuda.so; fi

# # Install Flash Attention 3
# RUN MAX_JOBS=$(( $(nproc) / 4 )) pip install git+https://github.com/Dao-AILab/flash-attention.git@27f501d#subdirectory=hopper
# # RUN MAX_JOBS=$(( $(nproc) / 4 )) pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.6#subdirectory=hopper

# # RUN pip install transformer_engine
# # COPY cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.10/dist-packages/flash_attn_3/flash_attn_interface.py
# # COPY cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff
# # RUN ls -ld /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
# # RUN pip show transformer_engine
# # RUN patch /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff
# COPY cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/flash_attn_interface.py
# COPY cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff
# RUN patch /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff

FROM ${BASE_IMAGE}

# Set the DEBIAN_FRONTEND environment variable to avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        tree \
        wget

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

WORKDIR /workspace

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project --extra cu126

# Place executables in the environment at the front of the path
ENV PATH="/workspace/.venv/bin:$PATH"

ENTRYPOINT ["/workspace/bin/entrypoint.sh"]
CMD ["/bin/bash"]
