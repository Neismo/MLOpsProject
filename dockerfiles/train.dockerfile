FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.12 and build deps
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
        ca-certificates \
        build-essential \
        gcc && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON=/usr/bin/python3.12

# Copy dependency files first
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

# Copy application code
COPY src/ src/

# Copy configs (needed for Hydra)
COPY configs/ configs/

# Entrypoint
CMD ["uv", "run", "python", "src/mlops_project/train.py"]