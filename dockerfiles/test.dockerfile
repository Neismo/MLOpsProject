FROM python:3.10-slim

WORKDIR /app

# Copy the script into the container
COPY test_gcs_mount.py /app/test_gcs_mount.py

# Default env (matches your Vertex logs)
ENV GCS_BUCKET=mlops-proj

CMD ["python", "/app/test_gcs_mount.py"]
