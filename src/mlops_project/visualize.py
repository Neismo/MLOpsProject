"""CLI tool for visualizing paper embeddings as a 2D scatter plot."""

import sys
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

app = typer.Typer(help="Visualize paper embeddings as a 2D scatter plot.")


def load_metadata_subjects(index_dir: Path) -> list[str]:
    """Load primary_subject for each paper from metadata.

    Args:
        index_dir: Directory containing metadata.json or metadata.db.

    Returns:
        List of primary_subject strings, one per paper.
    """
    import json
    import sqlite3

    metadata_json = index_dir / "metadata.json"
    metadata_db = index_dir / "metadata.db"

    if metadata_db.exists():
        logger.info(f"Loading subjects from SQLite: {metadata_db}")
        conn = sqlite3.connect(metadata_db)
        cur = conn.execute("SELECT primary_subject FROM papers ORDER BY id")
        subjects = [row[0] for row in cur.fetchall()]
        conn.close()
        return subjects
    elif metadata_json.exists():
        logger.info(f"Loading subjects from JSON: {metadata_json}")
        with open(metadata_json) as f:
            data = json.load(f)
        return [m["primary_subject"] for m in data["metadata"]]
    else:
        raise FileNotFoundError(f"No metadata found in {index_dir}")


def stratified_sample_indices(
    subjects: list[str],
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample indices with stratification by subject.

    Args:
        subjects: List of subject labels.
        sample_size: Total number of samples to draw.
        rng: NumPy random generator.

    Returns:
        Array of sampled indices.
    """
    subject_to_indices: dict[str, list[int]] = {}
    for idx, subj in enumerate(subjects):
        subject_to_indices.setdefault(subj, []).append(idx)

    n_subjects = len(subject_to_indices)
    n_total = len(subjects)

    # Calculate samples per subject proportionally
    sampled_list: list[int] = []
    for subj, indices in subject_to_indices.items():
        proportion = len(indices) / n_total
        n_samples = max(1, int(sample_size * proportion))
        n_samples = min(n_samples, len(indices))
        sampled = rng.choice(indices, size=n_samples, replace=False)
        sampled_list.extend(sampled)

    # If we have too many, trim randomly
    sampled_indices = np.array(sampled_list)
    if len(sampled_indices) > sample_size:
        sampled_indices = rng.choice(sampled_indices, size=sample_size, replace=False)

    logger.info(f"Sampled {len(sampled_indices):,} points across {n_subjects} subjects")
    return sampled_indices


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP or t-SNE.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        method: "umap" or "tsne".
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, 2).
    """
    if method == "umap":
        import umap

        logger.info(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
            verbose=True,
        )
        return reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        logger.info("Running t-SNE...")
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            metric="cosine",
            random_state=random_state,
            verbose=1,
        )
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'.")


def get_clean_label(subject: str) -> str:
    """Extract clean label from subject string like 'Machine Learning (cs.LG)' -> 'Machine Learning'."""
    import re

    # Remove the parenthetical code at the end
    clean = re.sub(r"\s*\([^)]+\)\s*$", "", subject)
    return clean.strip()


def create_scatter_plot(
    coords: np.ndarray,
    subjects: list[str],
    output_path: Path,
    figsize: int = 16,
    dpi: int = 300,
    point_size: float = 1.5,
    alpha: float = 0.8,
    min_cluster_size_for_label: int = 150,
) -> None:
    """Create and save a scatter plot of embeddings colored by subject.

    Args:
        coords: 2D coordinates, shape (n_samples, 2).
        subjects: Subject labels for each point.
        output_path: Path to save the output image.
        figsize: Figure size (square).
        dpi: Output image DPI.
        point_size: Size of scatter points.
        alpha: Transparency of points.
        min_cluster_size_for_label: Only label clusters with at least this many points.
    """
    unique_subjects = sorted(set(subjects))
    n_subjects = len(unique_subjects)
    logger.info(f"Plotting {len(coords):,} points with {n_subjects} unique subjects")

    # Count points per subject
    subject_counts = {subj: sum(1 for s in subjects if s == subj) for subj in unique_subjects}

    # Generate bright, saturated colors that pop on dark background
    np.random.seed(42)
    hues = np.linspace(0, 1, n_subjects, endpoint=False)
    np.random.shuffle(hues)
    colors = []
    for h in hues:
        # Convert HSV to RGB with high saturation and value for bright colors
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((r, g, b, alpha))
    subject_to_color = {subj: colors[i] for i, subj in enumerate(unique_subjects)}

    # Dark theme colors
    bg_color = "#0d1117"  # GitHub dark background
    label_bg = "#161b22"  # Slightly lighter for labels
    label_text = "#e6edf3"  # Light text

    # Create figure with dark background
    _, ax = plt.subplots(figsize=(figsize, figsize), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Plot all points, sorted by count (smaller clusters on top)
    sorted_subjects = sorted(unique_subjects, key=lambda s: subject_counts[s], reverse=True)

    for subj in sorted_subjects:
        mask = np.array([s == subj for s in subjects])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[subject_to_color[subj]],
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )

    # Collect label positions for larger clusters only
    label_positions = []
    for subj in unique_subjects:
        count = subject_counts[subj]
        if count >= min_cluster_size_for_label:
            mask = np.array([s == subj for s in subjects])
            centroid_x = coords[mask, 0].mean()
            centroid_y = coords[mask, 1].mean()
            label_positions.append((centroid_x, centroid_y, subj, count))

    # Sort by count descending
    label_positions.sort(key=lambda x: x[3], reverse=True)

    # Add labels with readable names
    for x, y, subj, *_ in label_positions:
        clean_label = get_clean_label(subj)
        ax.annotate(
            clean_label,
            (x, y),
            fontsize=6,
            fontweight="medium",
            ha="center",
            va="center",
            color=label_text,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=label_bg,
                alpha=0.9,
                edgecolor="#30363d",
                linewidth=0.5,
            ),
        )

    # Remove all axes and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add subtle padding
    x_margin = (coords[:, 0].max() - coords[:, 0].min()) * 0.02
    y_margin = (coords[:, 1].max() - coords[:, 1].min()) * 0.02
    ax.set_xlim(coords[:, 0].min() - x_margin, coords[:, 0].max() + x_margin)
    ax.set_ylim(coords[:, 1].min() - y_margin, coords[:, 1].max() + y_margin)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


@app.command()
def visualize(
    index_dir: str = typer.Option(
        "data/faiss",
        "--index-dir",
        "-i",
        help="Path to FAISS index directory.",
    ),
    output: str = typer.Option(
        "embedding_landscape.png",
        "--output",
        "-o",
        help="Output image path.",
    ),
    sample_size: int = typer.Option(
        50000,
        "--sample-size",
        "-n",
        help="Number of points to visualize.",
    ),
    method: str = typer.Option(
        "umap",
        "--method",
        "-m",
        help="Dimensionality reduction method: 'umap' or 'tsne'.",
    ),
    n_neighbors: int = typer.Option(
        15,
        "--n-neighbors",
        help="UMAP n_neighbors parameter.",
    ),
    min_dist: float = typer.Option(
        0.1,
        "--min-dist",
        help="UMAP min_dist parameter.",
    ),
    dpi: int = typer.Option(
        300,
        "--dpi",
        help="Output image DPI.",
    ),
    figsize: int = typer.Option(
        16,
        "--figsize",
        help="Figure size (square).",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility.",
    ),
) -> None:
    """Generate a 2D scatter plot visualization of paper embeddings."""
    index_dir_path = Path(index_dir)
    output_path = Path(output)
    index_path = index_dir_path / "index.faiss"

    if not index_path.exists():
        logger.error(f"FAISS index not found: {index_path}")
        raise typer.Exit(1)

    # Load FAISS index
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    n_vectors = index.ntotal
    logger.info(f"Index contains {n_vectors:,} vectors")

    # Enable reconstruction for IVF indices
    if hasattr(index, "make_direct_map"):
        logger.info("Building direct map for vector reconstruction...")
        index.make_direct_map()
        logger.info("Direct map built")

    # Load metadata subjects
    all_subjects = load_metadata_subjects(index_dir_path)

    if len(all_subjects) != n_vectors:
        logger.error(f"Metadata ({len(all_subjects)}) and index ({n_vectors}) size mismatch")
        raise typer.Exit(1)

    # Sample indices
    rng = np.random.default_rng(seed)
    actual_sample_size = min(sample_size, n_vectors)
    sampled_indices = stratified_sample_indices(all_subjects, actual_sample_size, rng)
    sampled_indices = np.sort(sampled_indices)  # Sort for efficient reconstruction

    # Reconstruct embeddings for sampled indices
    logger.info(f"Reconstructing {len(sampled_indices):,} embeddings...")
    embeddings = np.zeros((len(sampled_indices), index.d), dtype=np.float32)
    for i, idx in enumerate(sampled_indices):
        embeddings[i] = index.reconstruct(int(idx))
        if (i + 1) % 10000 == 0:
            logger.info(f"  Reconstructed {i + 1:,}/{len(sampled_indices):,} embeddings")

    # Get subjects for sampled indices
    sampled_subjects = [all_subjects[idx] for idx in sampled_indices]

    # Reduce dimensions
    coords = reduce_dimensions(
        embeddings,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )

    # Create visualization
    create_scatter_plot(
        coords,
        sampled_subjects,
        output_path,
        figsize=figsize,
        dpi=dpi,
    )

    logger.info("Visualization complete!")


if __name__ == "__main__":
    app()
