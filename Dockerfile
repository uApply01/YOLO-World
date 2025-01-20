FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG MODEL="yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
ARG WEIGHT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    python3-dev     \
    python3-wheel

RUN pip3 install --upgrade pip \
    && pip3 install   \
        gradio        \
        opencv-python \
        supervision   \
        mmengine      \
        setuptools    \
        openmim       \
    && mim install mmcv==2.0.0 \
    && pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        wheel         \
        torch         \
        torchvision   \
        torchaudio    \
        fastapi

RUN git clone https://github.com/uApply01/YOLO-World.git

WORKDIR /YOLO-World
RUN git submodule update --init --recursive
RUN pip3 install -e .
RUN mkdir -p weights
RUN curl -o weights/$WEIGHT -L https://huggingface.co/wondervictor/YOLO-World/resolve/main/$WEIGHT

ENTRYPOINT [ "python3", "api_service.py" ]
CMD ["configs/pretrain/$MODEL", "weights/$WEIGHT"]