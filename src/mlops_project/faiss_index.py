"""FAISS vector index for similarity search on paper abstracts."""

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    """A single search result from the FAISS index."""

    score: float
    title: str
    abstract: str
    primary_subject: str
    subjects: list[str]
    arxiv_id: str | None = None


class FAISSIndex:
    """Wrapper for FAISS IndexIVFFlat with metadata storage."""

    DEFAULT_NLIST = 4096
    DEFAULT_NPROBE = 64
    EMBEDDING_DIM = 384

    def __init__(
        self,
        index: faiss.Index | None = None,
        metadata: list[dict[str, Any]] | None = None,
        nprobe: int = DEFAULT_NPROBE,
        db_path: Path | None = None,
    ):
        self.index = index
        self.metadata = metadata or []
        self.nprobe = nprobe
        self.db_path = db_path

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        nlist: int = DEFAULT_NLIST,
        nprobe: int = DEFAULT_NPROBE,
    ) -> "FAISSIndex":
        """Build a new FAISS index from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim), normalized vectors.
            metadata: List of dicts with keys: title, abstract, primary_subject, subjects.
            nlist: Number of clusters for IVF index.
            nprobe: Number of clusters to search at query time.

        Returns:
            Configured FAISSIndex instance.
        """
        n_samples, dim = embeddings.shape
        logger.info(f"Building FAISS index with {n_samples:,} vectors, dim={dim}, nlist={nlist}")

        if n_samples != len(metadata):
            raise ValueError(f"Embeddings ({n_samples}) and metadata ({len(metadata)}) size mismatch")

        # Use IndexFlatIP (inner product) as quantizer for cosine similarity
        # (assumes embeddings are L2-normalized)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train the index on the embeddings
        logger.info("Training IVF index...")
        index.train(embeddings.astype(np.float32))

        # Add vectors to the index
        logger.info("Adding vectors to index...")
        index.add(embeddings.astype(np.float32))

        index.nprobe = nprobe
        logger.info(f"Index built: {index.ntotal:,} vectors, nprobe={nprobe}")

        return cls(index=index, metadata=metadata, nprobe=nprobe)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Search for the k most similar papers.

        Args:
            query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim).
            k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by descending similarity score.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Set nprobe for this search
        self.index.nprobe = self.nprobe

        # Search
        t0 = time.perf_counter()
        scores, indices = self.index.search(query_embedding, k)
        t1 = time.perf_counter()
        logger.info(f"FAISS search took {(t1 - t0) * 1000:.2f}ms")

        results = []

        t2 = time.perf_counter()
        if self.metadata:
            # Use in-memory metadata list
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                meta = self.metadata[idx]
                results.append(
                    SearchResult(
                        score=float(score),
                        title=meta["title"],
                        abstract=meta["abstract"],
                        primary_subject=meta["primary_subject"],
                        subjects=meta.get("subjects", []),
                        arxiv_id=meta.get("arxiv_id"),
                    )
                )
        elif self.db_path:
            # Query SQLite for metadata
            conn = sqlite3.connect(self.db_path)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                cur = conn.execute(
                    "SELECT title, abstract, primary_subject, subjects, arxiv_id FROM papers WHERE id = ?",
                    (int(idx),),
                )
                row = cur.fetchone()
                if row:
                    title, abstract, primary_subject, subjects, arxiv_id = row
                    results.append(
                        SearchResult(
                            score=float(score),
                            title=title,
                            abstract=abstract,
                            primary_subject=primary_subject,
                            subjects=subjects.split("; ") if subjects else [],
                            arxiv_id=arxiv_id,
                        )
                    )
            conn.close()
        else:
            raise RuntimeError("No metadata available. Load with preload_metadata=True or ensure db_path exists.")

        t3 = time.perf_counter()
        logger.info(f"Metadata fetch took {(t3 - t2) * 1000:.2f}ms (mode: {'memory' if self.metadata else 'sqlite'})")

        return results

    def save(self, output_dir: str | Path) -> None:
        """Save the index and metadata to disk.

        Args:
            output_dir: Directory to save index.faiss and metadata.json.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / "index.faiss"
        metadata_path = output_dir / "metadata.json"

        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, str(index_path))

        logger.info(f"Saving metadata ({len(self.metadata):,} entries) to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump({"metadata": self.metadata, "nprobe": self.nprobe}, f)

        logger.info("Save complete")

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        nprobe: int | None = None,
        metadata_source: str | None = None,
    ) -> "FAISSIndex":
        """Load a FAISS index from disk.

        Args:
            index_dir: Directory containing index.faiss and (metadata.db or metadata.json).
            nprobe: Override nprobe value (uses saved value if None).
            metadata_source: "json" to load from metadata.json, "sqlite" to query metadata.db
                             at search time, or None to auto-detect (prefers sqlite if exists).

        Returns:
            Loaded FAISSIndex instance.
        """
        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        db_path_candidate = index_dir / "metadata.db"
        metadata_path = index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))

        metadata: list[dict[str, Any]] = []
        saved_nprobe = cls.DEFAULT_NPROBE
        db_path: Path | None = None

        # Determine metadata source
        if metadata_source is None:
            # Auto-detect: prefer sqlite if exists
            if db_path_candidate.exists():
                metadata_source = "sqlite"
            elif metadata_path.exists():
                metadata_source = "json"
            else:
                raise FileNotFoundError(f"Neither {db_path_candidate} nor {metadata_path} found")

        if metadata_source == "json":
            if not metadata_path.exists():
                raise FileNotFoundError(f"JSON metadata not found: {metadata_path}")
            logger.info(f"Loading metadata from {metadata_path}")
            t0 = time.perf_counter()
            with open(metadata_path) as f:
                data = json.load(f)
            metadata = data["metadata"]
            saved_nprobe = data.get("nprobe", cls.DEFAULT_NPROBE)
            elapsed = time.perf_counter() - t0
            logger.info(f"Loaded {len(metadata):,} entries from JSON in {elapsed:.1f}s")
        elif metadata_source == "sqlite":
            if not db_path_candidate.exists():
                raise FileNotFoundError(f"SQLite database not found: {db_path_candidate}")
            # Use SQLite queries at search time
            logger.info(f"Using SQLite database: {db_path_candidate}")
            db_path = db_path_candidate
            # Get nprobe from JSON if it exists
            if metadata_path.exists():
                with open(metadata_path) as f:
                    data = json.load(f)
                saved_nprobe = data.get("nprobe", cls.DEFAULT_NPROBE)
        else:
            raise ValueError(f"Invalid metadata_source: {metadata_source}. Use 'json' or 'sqlite'.")

        effective_nprobe = nprobe if nprobe is not None else saved_nprobe
        logger.info(f"Index loaded: {index.ntotal:,} vectors, nprobe={effective_nprobe}")

        return cls(index=index, metadata=metadata, nprobe=effective_nprobe, db_path=db_path)


def build_index_from_dataset(
    model: SentenceTransformer,
    dataset,
    batch_size: int = 256,
    nlist: int = FAISSIndex.DEFAULT_NLIST,
    nprobe: int = FAISSIndex.DEFAULT_NPROBE,
    show_progress: bool = True,
) -> FAISSIndex:
    """Build a FAISS index from a HuggingFace dataset.

    Args:
        model: SentenceTransformer model for encoding abstracts.
        dataset: HuggingFace dataset with columns: title, abstract, primary_subject, subjects.
        batch_size: Batch size for encoding.
        nlist: Number of IVF clusters.
        nprobe: Number of clusters to probe at search time.
        show_progress: Show progress bar during encoding.

    Returns:
        Built FAISSIndex.
    """
    from tqdm import tqdm

    n_samples = len(dataset)
    logger.info(f"Encoding {n_samples:,} abstracts with batch_size={batch_size}")

    # Pre-allocate embeddings array
    embedding_dim = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
    metadata = []

    # Process in batches using dataset iteration
    n_batches = (n_samples + batch_size - 1) // batch_size
    progress = tqdm(total=n_samples, disable=not show_progress, desc="Encoding")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        # Get batch slice from dataset
        batch = dataset[start_idx:end_idx]

        # Encode batch
        batch_embeddings = model.encode(
            batch["abstract"],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings[start_idx:end_idx] = batch_embeddings

        # Collect metadata
        for i in range(len(batch["abstract"])):
            metadata.append(
                {
                    "title": batch["title"][i],
                    "abstract": batch["abstract"][i],
                    "primary_subject": batch["primary_subject"][i],
                    "subjects": batch.get("subjects", [[]])[i] if "subjects" in batch else [],
                }
            )

        progress.update(end_idx - start_idx)

    progress.close()

    return FAISSIndex.build(embeddings, metadata, nlist=nlist, nprobe=nprobe)
