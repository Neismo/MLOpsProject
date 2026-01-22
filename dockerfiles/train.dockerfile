# Use an official Python base image
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

# Python
RUN apt-get update && apt-get install -y \
    python3.12 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Layer 1: Install dependencies only (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

# Layer 2: Copy application code
COPY src/ src/

# Layer 3: Copy configs (needed for Hydra)
COPY configs/ configs/

# Entrypoint
CMD ["uv", "run", "src/mlops_project/train.py"]
