"""CLI script for building the FAISS index from paper abstracts."""

import sys
from pathlib import Path

import torch
import typer
from datasets import concatenate_datasets, load_from_disk
from loguru import logger
from sentence_transformers import SentenceTransformer

from mlops_project.faiss_index import FAISSIndex, build_index_from_dataset

logger.remove()
logger.add(sys.stdout, level="INFO")

app = typer.Typer(help="Build FAISS index from paper abstracts.")


@app.command()
def build(
    model_path: str = typer.Option(
        "models/all-MiniLM-L6-v2-mnrl-1M-balanced/checkpoint-3907",
        "--model-path",
        "-m",
        help="Path to the SentenceTransformer model.",
    ),
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d",
        help="Directory containing dataset splits.",
    ),
    output_dir: str = typer.Option(
        "data/faiss",
        "--output-dir",
        "-o",
        help="Output directory for FAISS index.",
    ),
    batch_size: int = typer.Option(
        256,
        "--batch-size",
        "-b",
        help="Batch size for encoding.",
    ),
    splits: list[str] = typer.Option(
        ["train", "eval", "test"],
        "--splits",
        "-s",
        help="Dataset splits to include.",
    ),
    nlist: int = typer.Option(
        FAISSIndex.DEFAULT_NLIST,
        "--nlist",
        help="Number of IVF clusters.",
    ),
    nprobe: int = typer.Option(
        FAISSIndex.DEFAULT_NPROBE,
        "--nprobe",
        help="Number of clusters to probe at search time.",
    ),
    compile_model: bool = typer.Option(
        False,
        "--compile",
        help="Use torch.compile() for faster inference.",
    ),
    tf32: bool = typer.Option(
        True,
        "--tf32/--no-tf32",
        help="Enable TF32 for faster matmul on Ampere+ GPUs.",
    ),
) -> None:
    """Build a FAISS index from paper abstract embeddings."""
    model_dir = Path(model_path)
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    if not model_dir.exists():
        logger.error(f"Model path does not exist: {model_dir}")
        raise typer.Exit(1)

    # Enable TF32 for faster matmul on Ampere+ GPUs
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul and cuDNN")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {model_dir} on {device}")
    model = SentenceTransformer(str(model_dir), device=device)

    # Compile model for faster inference
    if compile_model and device == "cuda":
        logger.info("Compiling model with torch.compile()...")
        model[0].auto_model = torch.compile(model[0].auto_model, mode="reduce-overhead")
        logger.info("Model compiled")

    # Load and concatenate datasets
    datasets_to_combine = []
    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            logger.warning(f"Split not found, skipping: {split_path}")
            continue
        logger.info(f"Loading split: {split}")
        ds = load_from_disk(str(split_path))
        datasets_to_combine.append(ds)

    if not datasets_to_combine:
        logger.error("No valid splits found")
        raise typer.Exit(1)

    combined_dataset = concatenate_datasets(datasets_to_combine)
    logger.info(f"Combined dataset size: {len(combined_dataset):,} papers")

    # Build index
    faiss_index = build_index_from_dataset(
        model=model,
        dataset=combined_dataset,
        batch_size=batch_size,
        nlist=nlist,
        nprobe=nprobe,
        show_progress=True,
    )

    # Save index
    faiss_index.save(output_path)
    logger.info(f"Index saved to {output_path}")


if __name__ == "__main__":
    app()
