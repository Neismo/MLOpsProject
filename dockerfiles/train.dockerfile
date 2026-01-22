# Use an official Python base image
FROM python:3.12-slim

# Install uv via binary copy
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
CMD ["uv", "run", "python", "src/mlops_project/train.py"]
