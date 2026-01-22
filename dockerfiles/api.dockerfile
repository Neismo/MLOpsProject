FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies - CPU only for torch
COPY uv.lock pyproject.toml README.md ./
COPY src src/

# Override torch source to use CPU index on Linux
RUN sed -i "s/sys_platform == 'linux' or sys_platform == 'win32'/sys_platform == 'win32'/g" pyproject.toml && \
    sed -i "s/sys_platform == 'darwin'/sys_platform == 'darwin' or sys_platform == 'linux'/g" pyproject.toml && \
    uv pip install --system -e .

# Bake model and index into image
COPY faiss-endpoint/model /data/model
COPY faiss-endpoint/index /data/index

# Environment variables
ENV MODEL_PATH=/data/model
ENV INDEX_PATH=/data/index
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the API server
ENTRYPOINT ["python", "-m", "uvicorn", "mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
