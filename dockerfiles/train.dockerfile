FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.12
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.12 \
        python3.12-dev \
        python3-pip \
        build-essential \
        gcc \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ src/

# Copy configs (needed for Hydra)
COPY configs/ configs/

# Entrypoint
ENTRYPOINT [".venv/bin/python", "-m", "mlops_project.train"]
