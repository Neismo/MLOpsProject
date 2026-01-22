import argparse
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# ---------- Parse CLI arguments ----------

parser = argparse.ArgumentParser(description="MLOps Abstract Embedding API")
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to the SentenceTransformer model directory",
)
args, _ = parser.parse_known_args()

app = FastAPI(
    title="MLOps Abstract Embedding API",
    version="1.0.0",
)

# ---------- Load model once at startup ----------

MODEL_PATH = Path(args.model_path or os.environ.get("MODEL_PATH", "models/contrastive-minilm/"))

device = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH.exists():
    logger.error(f"Model path does not exist: {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

try:
    model = SentenceTransformer(str(MODEL_PATH), device=device)
    logger.info(f"Model loaded from {MODEL_PATH} on {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# ---------- Request / Response schemas ----------


class AbstractRequest(BaseModel):
    abstract: str


class EmbeddingResponse(BaseModel):
    embedding_dim: int
    embedding: list[float]


class HealthResponse(BaseModel):
    status: str
    device: str


# ---------- Health check ----------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", device=device)


# ---------- Inference endpoint ----------


@app.post("/embed", response_model=EmbeddingResponse)
def embed_abstract(request: AbstractRequest) -> EmbeddingResponse:
    if not request.abstract.strip():
        logger.warning("Empty abstract received")
        raise HTTPException(status_code=400, detail="Abstract must not be empty")

    embedding = model.encode(
        request.abstract,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return EmbeddingResponse(
        embedding_dim=len(embedding),
        embedding=embedding.tolist(),
    )
