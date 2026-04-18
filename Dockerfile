# syntax=docker/dockerfile:1.7

# projects-base: shared GPU + ML + media toolchain for child projects.
# Child projects should start with: FROM ghcr.io/samjeg/projects-base:latest
# This image intentionally contains NO end-user application code.
#
# Multi-stage: builder pulls a cudnn-devel base with compiler toolchain, builds
# /opt/venv with torch + ML libs. Runtime stage copies only /opt/venv on top of
# cudnn-runtime — this drops ~10GB of compiler toolchain and CUDA headers.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.13

# ===========================================================================
# Stage 1 — builder: compiler toolchain + venv with all pip packages.
# ===========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PATH=/opt/venv/bin:${PATH}

# apt-get hardening: retry flaky mirrors and disable pipelining, which is
# the root cause of the intermittent "Hash Sum mismatch" failures we've seen
# from archive.ubuntu.com CDN edges serving stale package indexes.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\n' \
        > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

# PyTorch stack (CUDA 12.4 wheels). Flip PYTHON_VERSION build arg up when
# upstream publishes 3.14 wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch \
        torchvision \
        torchaudio

# Shared ML / media libs. Children layer app-specific deps on top.
RUN pip install \
        numpy \
        scipy \
        pillow \
        opencv-python-headless \
        imageio \
        imageio-ffmpeg \
        av \
        soundfile \
        librosa \
        einops \
        safetensors \
        huggingface_hub \
        transformers \
        accelerate \
        diffusers \
        sentencepiece \
        protobuf \
        tqdm \
        rich \
        pydantic \
        httpx \
        python-dotenv \
        typer

# Video/audio tooling used across child projects:
#   - openai-whisper: CLI + lib for speech-to-text (reaction, story, news styles)
#   - rembg[gpu]: CLI + lib for background removal (reaction, cut-out overlays)
# Pulled in their own RUN so adding/removing them doesn't invalidate the
# large shared-libs layer above.
RUN pip install \
        openai-whisper \
        rembg[gpu] \
        filetype

# Strip bytecode caches and tests from the venv — saves hundreds of MB.
RUN find /opt/venv -depth \
        \( -type d \( -name __pycache__ -o -name tests -o -name test \) \
           -o -name '*.pyc' -o -name '*.pyo' \) \
        -exec rm -rf {} + 2>/dev/null || true

# ===========================================================================
# Stage 2 — runtime: minimal base + copied venv. This is what gets pushed.
# ===========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache

# Runtime-only system deps. deadsnakes gives us Python 3.13; everything else
# is the minimal set needed by opencv/soundfile/librosa/ffmpeg at runtime.
# software-properties-common is purged after the PPA is added to save ~80MB.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\n' \
        > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        libopenblas0 \
        fonts-liberation \
        fonts-dejavu-core \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# openssh-server kept in its own thin layer so adding/removing it doesn't
# invalidate the ~589MB runtime apt layer above. This keeps incremental
# pushes small for base-image updates that only touch SSH behaviour.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Bring the fully-installed venv from builder. The venv's python symlink
# points at /usr/bin/python${PYTHON_VERSION}, which we just installed above.
COPY --from=builder /opt/venv /opt/venv

# Persistent model/weights cache — mount a Runpod volume here so downloads
# survive pod restarts.
RUN mkdir -p /cache/huggingface /cache/torch /workspace

# SSH sessions don't inherit Dockerfile ENV — PAM resets the environment
# based on /etc/environment. Persist the critical vars so `python`, torch,
# and the HF cache paths resolve correctly when users SSH into the pod.
RUN printf 'PATH="/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nVIRTUAL_ENV="/opt/venv"\nHF_HOME="/cache/huggingface"\nTORCH_HOME="/cache/torch"\nXDG_CACHE_HOME="/cache"\n' \
        > /etc/environment

# Runpod-compatible entrypoint: seed authorized_keys from the PUBLIC_KEY env
# var Runpod injects, generate sshd host keys on first boot, start sshd in
# the background, then exec the container's command. This is what makes
# `ssh root@pod-ip` work from outside.
COPY <<'EOF' /usr/local/bin/runpod-entrypoint.sh
#!/bin/bash
set -e
mkdir -p /run/sshd /root/.ssh
chmod 0755 /run/sshd
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
[ -e /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -A >/dev/null
/usr/sbin/sshd
exec "$@"
EOF
RUN chmod +x /usr/local/bin/runpod-entrypoint.sh

WORKDIR /workspace

# Fail the build early if the venv copy broke, CUDA is missing from torch,
# or any of the CLI tools have broken import chains. This catches broken
# pip extras (e.g. rembg's CLI deps) on GHA instead of on a Runpod pod.
RUN python -c "import torch, sys; print('python', sys.version); print('torch', torch.__version__, 'cuda_built', torch.version.cuda)" \
    && python -c "from rembg.cli import main; print('rembg cli ok')" \
    && python -c "import whisper; print('whisper ok')" \
    && whisper --help >/dev/null && echo 'whisper cli ok' \
    && rembg --help >/dev/null && echo 'rembg cli ok'

ENTRYPOINT ["/usr/local/bin/runpod-entrypoint.sh"]
CMD ["python"]
