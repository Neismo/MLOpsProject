FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Avoid interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python -> python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Python / uv settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies (NO dev deps)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

# Copy application code + configs
COPY src/ src/
COPY configs/ configs/

# Entrypoint
CMD ["uv", "run", "python", "src/mlops_project/train.py"]
