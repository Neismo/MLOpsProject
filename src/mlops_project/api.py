from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(
    title="MLOps Abstract Embedding API",
    version="1.0.0",
)

# ---------- Load model once at startup ----------

MODEL_PATH = Path("models/contrastive-minilm/")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(str(MODEL_PATH), device=device)


# ---------- Request / Response schemas ----------


class AbstractRequest(BaseModel):
    abstract: str


class EmbeddingResponse(BaseModel):
    embedding_dim: int
    embedding: list[float]


# ---------- Health check ----------


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


# ---------- Inference endpoint ----------


@app.post("/embed", response_model=EmbeddingResponse)
def embed_abstract(request: AbstractRequest):
    if not request.abstract.strip():
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
