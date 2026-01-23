import argparse
import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger
from prometheus_client import Counter, make_asgi_app

from mlops_project.faiss_index import FAISSIndex

logger.remove()
logger.add(sys.stdout, level="INFO")

# ---------- Prometheus Metrics  ----------

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
)

# ---------- Parse CLI arguments ----------

parser = argparse.ArgumentParser(description="MLOps Abstract Embedding API")
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to the SentenceTransformer model directory",
)
parser.add_argument(
    "--index-path",
    type=str,
    default=None,
    help="Path to the FAISS index directory (optional)",
)
args, _ = parser.parse_known_args()

app = FastAPI(
    title="MLOps Abstract Embedding API",
    version="1.0.0",
)

# ---------- Mount static files ----------

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.mount("/metrics", make_asgi_app())


@app.get("/")
def root():
    REQUEST_COUNT.labels(endpoint="/", method="GET", status_code=302).inc()
    return RedirectResponse(url="/static/index.html")


# ---------- Load model once at startup ----------

MODEL_PATH = Path(args.model_path or os.environ.get("MODEL_PATH", "models/contrastive-minilm/"))

device = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH.exists():
    logger.error(f"Model path does not exist: {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

try:
    model = SentenceTransformer(str(MODEL_PATH), device=device)
    logger.info(f"Model loaded from {MODEL_PATH} on {device}")
    # Warmup the model
    _ = model.encode("warmup", convert_to_numpy=True, normalize_embeddings=True)
    logger.info("Model warmup complete")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# ---------- Load FAISS index (optional) ----------

INDEX_PATH = args.index_path or os.environ.get("INDEX_PATH")
PRELOAD_METADATA = os.environ.get("PRELOAD_METADATA", "false").lower() in ("true", "1", "yes")
faiss_index: FAISSIndex | None = None

if INDEX_PATH:
    index_path = Path(INDEX_PATH)
    if index_path.exists():
        try:
            faiss_index = FAISSIndex.load(index_path, preload_metadata=PRELOAD_METADATA)
            logger.info(f"FAISS index loaded from {index_path} (preload_metadata={PRELOAD_METADATA})")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
    else:
        logger.warning(f"FAISS index path does not exist: {index_path}")


# ---------- Request / Response schemas ----------


class AbstractRequest(BaseModel):
    abstract: str


class EmbeddingResponse(BaseModel):
    embedding_dim: int
    embedding: list[float]


class HealthResponse(BaseModel):
    status: str
    device: str


class SearchRequest(BaseModel):
    abstract: str
    k: int = Field(default=10, ge=1, le=100)
    nprobe: int | None = Field(default=None, ge=1, le=1024)


class SearchResultItem(BaseModel):
    score: float
    title: str
    abstract: str
    primary_subject: str
    subjects: list[str] = []
    arxiv_id: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


# ---------- Health check ----------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status_code=200).inc()
    return HealthResponse(status="ok", device=device)


# ---------- Inference endpoint ----------


@app.post("/embed", response_model=EmbeddingResponse)
def embed_abstract(request: AbstractRequest) -> EmbeddingResponse:
    if not request.abstract.strip():
        logger.warning("Empty abstract received")
        REQUEST_COUNT.labels(endpoint="/embed", method="POST", status_code=400).inc()
        raise HTTPException(status_code=400, detail="Abstract must not be empty")

    embedding = model.encode(
        request.abstract,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    REQUEST_COUNT.labels(endpoint="/embed", method="POST", status_code=200).inc()
    return EmbeddingResponse(
        embedding_dim=len(embedding),
        embedding=embedding.tolist(),
    )


# ---------- Search endpoint ----------


@app.post("/search", response_model=SearchResponse)
def search_similar(request: SearchRequest) -> SearchResponse:
    if faiss_index is None:
        REQUEST_COUNT.labels(endpoint="/search", method="POST", status_code=503).inc()
        raise HTTPException(
            status_code=503,
            detail="FAISS index not loaded. Set INDEX_PATH environment variable or --index-path argument.",
        )

    if not request.abstract.strip():
        REQUEST_COUNT.labels(endpoint="/search", method="POST", status_code=400).inc()
        logger.warning("Empty abstract received for search")
        raise HTTPException(status_code=400, detail="Abstract must not be empty")

    # Override nprobe if specified
    if request.nprobe is not None:
        faiss_index.nprobe = request.nprobe

    # Encode the query abstract
    t0 = time.perf_counter()
    query_embedding = model.encode(
        request.abstract,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    t1 = time.perf_counter()
    logger.info(f"Encoding took {(t1 - t0) * 1000:.2f}ms")

    # Search the index
    results = faiss_index.search(query_embedding, k=request.k)
    t2 = time.perf_counter()
    logger.info(f"Total search time {(t2 - t0) * 1000:.2f}ms")
    REQUEST_COUNT.labels(endpoint="/search", method="POST", status_code=200).inc()
    return SearchResponse(
        results=[
            SearchResultItem(
                score=r.score,
                title=r.title,
                abstract=r.abstract,
                primary_subject=r.primary_subject,
                subjects=r.subjects if isinstance(r.subjects, list) else [r.subjects] if r.subjects else [],
                arxiv_id=r.arxiv_id,
            )
            for r in results
        ]
    )
