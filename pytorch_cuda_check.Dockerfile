# Use follwing command for building
#     docker build -t pytorch-cuda-test -f pytorch_cuda_check.Dockerfile .
# And running with:
#     docker run --rm --gpus all pytorch-cuda-test
#     docker run --runtime nvidia --rm --gpus all pytorch-cuda-test
#     docker run --runtime=nvidia --rm --gpus all pytorch-cuda-test
# 

# Minimal PyTorch + CUDA Test Image
FROM nvcr.io/nvidia/pytorch:25.04-py3
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.0.1-devel-ubuntu20.04
# FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM nvcr.io/nvidia/pytorch:23.09-py3 

# Verhindert interaktive Prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python + Pip installieren
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# Check CUDA installations
RUN ls -l /usr/local/cuda*

# PyTorch mit passender CUDA-Version installieren
# RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 \
#     --index-url https://download.pytorch.org/whl/cu121
# RUN pip install torch==2.3.1+cu120 torchvision==0.18.1+cu120 torchaudio==2.3.1 \
#     --index-url https://download.pytorch.org/whl/cu120
# RUN pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1 \
#     --index-url https://download.pytorch.org/whl/cu118

# Testscript hinzufügen
COPY test_cuda.py /workspace/test_cuda.py

WORKDIR /workspace

CMD ["python3", "test_cuda.py"]


