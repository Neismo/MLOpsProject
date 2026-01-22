FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies - CPU only for torch
COPY uv.lock pyproject.toml README.md ./
COPY src src/

# Override torch source to use CPU index on Linux
RUN sed -i "s/sys_platform == 'linux' or sys_platform == 'win32'/sys_platform == 'win32'/g" pyproject.toml && \
    sed -i "s/sys_platform == 'darwin'/sys_platform == 'darwin' or sys_platform == 'linux'/g" pyproject.toml && \
    uv pip install --system -e .

# Environment variables for model and index paths
ENV MODEL_PATH=/local/model
ENV INDEX_PATH=/local/index
ENV PORT=8000

# Expose port
EXPOSE 8000

# Startup script that copies from GCS mount to local disk, then starts server
RUN echo '#!/bin/bash\n\
mkdir -p /local/model /local/index\n\
echo "Copying model to local disk..."\n\
cp -r /data/model/* /local/model/\n\
echo "Copying index to local disk..."\n\
cp -r /data/index/* /local/index/\n\
echo "Starting server..."\n\
exec python -m uvicorn mlops_project.api:app --host 0.0.0.0 --port 8000\n\
' > /start.sh && chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
