# =========================
# Stage 1: Build
# =========================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder
LABEL maintainer="Ming-Yang Ho"

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y \
    cmake \
    python3-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /app
COPY environment.yaml .

RUN conda env create -f environment.yaml -n fulltilt && \
    conda clean --all --yes

RUN conda init bash && \
    echo "conda activate fulltilt" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=fulltilt
ENV PATH=$CONDA_DIR/envs/fulltilt/bin:$PATH

RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install torchvision==0.20.1 timm==1.0.20

COPY . .
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
RUN pip install -e . --no-build-isolation

# =========================
# Stage 2: Runtime
# =========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y \
    python3 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

COPY --from=builder $CONDA_DIR $CONDA_DIR

RUN conda init bash && \
    echo "conda activate fulltilt" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=fulltilt
ENV PATH=$CONDA_DIR/envs/fulltilt/bin:$PATH

WORKDIR /app

COPY --from=builder /app /app

CMD ["python", "example.py"]