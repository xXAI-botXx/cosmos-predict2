# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:25.04-py3
#          for CUDA 12.2 with Nvidia Driver 535
# FROM nvcr.io/nvidia/pytorch:23.09-py3  
# FROM nvcr.io/nvidia/pytorch:24.02-py3

# CUDA reinstallment
# Remove existing CUDA toolkit
RUN rm -rf /usr/local/cuda
RUN rm -rf /usr/local/cuda-12.9

# Add Nvidia CUDA Repo
RUN apt-get update && apt-get install -y software-properties-common wget gnupg2

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
RUN apt-get update

# Install CUDA Toolkit 12.0 (deb-Pakete from NVIDIA-Repository)
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-get update && apt-get install -y wget
# RUN wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
# RUN apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb
# RUN apt-get update
# RUN apt-get install -y cuda-toolkit-12-0
# RUN apt-get install -y cuda-runtime-12-0

RUN apt install -y cuda-12-5
RUN apt-get install -y cuda-toolkit-12-5
RUN apt-get install -y cuda-runtime-12-5
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-5

# RUN apt-get update && apt-cache search libcusparse
# RUN apt-cache search libcusparselt

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libcublas-12-5 \
#     libcusparse-12-5 \
#     libcusparseLt-12-5 \
#     && rm -rf /var/lib/apt/lists/*

# Check
RUN ls /usr/local/cuda-12.5
RUN ls /usr/local/cuda-12.5/lib64
# RUN ls /usr/local/cuda-12.0/lib64/libcusparseLt.so.0

# Alternative: Use CUDA 12.0 Base Installer:
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.60.13_linux.run
# RUN sh cuda_12.0.1_525.60.13_linux.run --silent --toolkit

# RUN apt-get update && apt-get install -y cuda-toolkit-12-0

# Update CUDA path
RUN ls -l /usr/local/cuda*
ENV CUDA_HOME=/usr/local/cuda-12.5
ENV CUDA_PATH=$CUDA_HOME/bin
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64

ARG TARGETPLATFORM

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN if [[ ${TARGETPLATFORM} == 'linux/amd64' ]]; then ln -s /lib64/libcuda.so.1 /lib64/libcuda.so; fi
RUN apt-get install -y libglib2.0-0
# RUN apt-get update && apt-get install -y cuda-toolkit-12-1
# RUN apt-get update && apt-get install -y cuda-nvcc-12-1
RUN [ -f /etc/pip/constraint.txt ] && sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt || echo "File not found, skipping sed"
# RUN sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt

# RUN if [[ ${TARGETPLATFORM} == 'linux/amd64' ]]; then ln -s /lib64/libcuda.so.1 /lib64/libcuda.so; fi

# Install Flash Attention 3
RUN MAX_JOBS=$(( $(nproc) / 4 )) pip install git+https://github.com/Dao-AILab/flash-attention.git@27f501d#subdirectory=hopper
# RUN MAX_JOBS=$(( $(nproc) / 4 )) pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.6#subdirectory=hopper

# RUN pip install transformer_engine
# COPY cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.10/dist-packages/flash_attn_3/flash_attn_interface.py
# COPY cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff
# RUN ls -ld /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
# RUN pip show transformer_engine
# RUN patch /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff
COPY cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/flash_attn_interface.py
COPY cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff
RUN patch /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff

# Installing decord from source on ARM
COPY Video_Codec_SDK_13.0.19.zip* /workspace/Video_Codec_SDK_13.0.19.zip
RUN if [[ ${TARGETPLATFORM} == 'linux/arm64' ]]; then export DEBIAN_FRONTEND=noninteractive && \
apt-get update && \
apt-get install -y build-essential python3-dev python3-setuptools make cmake \
                   ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev git ssh unzip nano python3-pip && \
git clone --recursive https://github.com/dmlc/decord && \
cd decord && \
find . -type f -exec sed -i "s/AVInputFormat \*/const AVInputFormat \*/g" {} \; && \
sed -i "s/[[:space:]]AVCodec \*dec/const AVCodec \*dec/" src/video/video_reader.cc && \
sed -i "s/avcodec\.h>/avcodec\.h>\n#include <libavcodec\/bsf\.h>/" src/video/ffmpeg/ffmpeg_common.h && \
mkdir build && cd build && \
scp /workspace/Video_Codec_SDK_13.0.19.zip . && \
unzip Video_Codec_SDK_13.0.19.zip && \
cp Video_Codec_SDK_13.0.19/Lib/linux/stubs/aarch64/* /usr/local/cuda/lib64/ && \
cp Video_Codec_SDK_13.0.19/Interface/* /usr/local/cuda/include && \
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
make -j 4 && \
cd ../python && python3 setup.py install; fi
RUN if [[ ${TARGETPLATFORM} == 'linux/arm64' ]]; then apt remove -y python3-blinker; fi

# Install the dependencies from requirements-docker.txt
COPY ./requirements-docker.txt /requirements.txt
ARG NATTEN_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;10.3;12.0"
RUN pip install --no-cache-dir -r /requirements.txt
RUN mkdir -p /workspace
WORKDIR /workspace

CMD ["/bin/bash"]
