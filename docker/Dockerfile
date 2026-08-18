FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3-pip git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3.9 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/SWS-TER
COPY . .
RUN python3.9 -m pip install torch==2.3.0 torchvision==0.18.0 \
      --index-url https://download.pytorch.org/whl/cu118 \
    && python3.9 -m pip install -r requirements/runtime.txt \
    && python3.9 -m pip install -v mmcv-full==1.7.2 --no-build-isolation \
    && python3.9 -m pip install -e .

CMD ["/bin/bash"]

