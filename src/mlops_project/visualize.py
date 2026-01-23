"""CLI tool for visualizing paper embeddings as a 2D scatter plot."""

import sys
from math import cos, sin
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
    sampled_indices = []
    for subj, indices in subject_to_indices.items():
        proportion = len(indices) / n_total
        n_samples = max(1, int(sample_size * proportion))
        n_samples = min(n_samples, len(indices))
        sampled = rng.choice(indices, size=n_samples, replace=False)
        sampled_indices.extend(sampled)

    # If we have too many, trim randomly
    sampled_indices = np.array(sampled_indices)
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


def build_subject_palette(
    subjects: list[str],
    alpha: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], dict[str, int], dict[str, tuple[float, float, float, float]]]:
    """Create a stable, bright color palette per subject.

    Args:
        subjects: Subject labels for each point.
        alpha: Alpha channel for the generated colors.
        seed: Seed for deterministic color shuffling.

    Returns:
        (unique_subjects, subject_counts, subject_to_color)
    """
    unique_subjects = sorted(set(subjects))
    subject_counts: dict[str, int] = {}
    for subj in subjects:
        subject_counts[subj] = subject_counts.get(subj, 0) + 1

    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, len(unique_subjects), endpoint=False)
    rng.shuffle(hues)

    colors = []
    for h in hues:
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((r, g, b, alpha))

    subject_to_color = {subj: colors[i] for i, subj in enumerate(unique_subjects)}
    return unique_subjects, subject_counts, subject_to_color


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
    unique_subjects, subject_counts, subject_to_color = build_subject_palette(
        subjects,
        alpha=1.0,
    )
    logger.info(f"Plotting {len(coords):,} points with {len(unique_subjects)} unique subjects")

    # Dark theme colors
    bg_color = "#0d1117"  # GitHub dark background
    label_bg = "#161b22"  # Slightly lighter for labels
    label_text = "#e6edf3"  # Light text

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(figsize, figsize), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

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

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_margin = (x_max - x_min) * 0.04
    y_margin = (y_max - y_min) * 0.04
    full_x_min, full_x_max = x_min - x_margin, x_max + x_margin
    full_y_min, full_y_max = y_min - y_margin, y_max + y_margin

    x_range = full_x_max - full_x_min
    y_range = full_y_max - full_y_min
    max_range = max(x_range, y_range)
    if x_range < max_range:
        pad = (max_range - x_range) / 2
        full_x_min -= pad
        full_x_max += pad
    if y_range < max_range:
        pad = (max_range - y_range) / 2
        full_y_min -= pad
        full_y_max += pad

    ax.set_xlim(full_x_min, full_x_max)
    ax.set_ylim(full_y_min, full_y_max)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig(output_path, dpi=dpi, facecolor=bg_color, edgecolor="none")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


def create_scatter_animation(
    coords: np.ndarray,
    subjects: list[str],
    output_path: Path,
    figsize: float,
    dpi: int,
    point_size: float = 1.2,
    alpha: float = 0.8,
    min_cluster_size_for_label: int = 150,
    frames: int = 36,
    fps: int = 18,
    max_points: int | None = None,
    seed: int = 42,
    start_zoom: float = 0.3,
    max_rotation_deg: float = 10.0,
) -> None:
    """Create an animated scatter plot and save it as a GIF.

    Args:
        coords: 2D coordinates, shape (n_samples, 2).
        subjects: Subject labels for each point.
        output_path: Path to save the GIF.
        figsize: Figure size (square).
        dpi: Output GIF DPI.
        point_size: Size of scatter points.
        alpha: Target transparency of points.
        min_cluster_size_for_label: Only label clusters with at least this many points.
        frames: Number of animation frames.
        fps: Frames per second.
        max_points: Maximum number of points to render in the GIF.
        seed: Random seed for reproducibility.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        logger.warning(f"Skipping animation: Pillow not available ({exc})")
        return

    if len(coords) == 0:
        logger.warning("Skipping animation: no points available")
        return

    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(coords), size=max_points, replace=False)
        coords = coords[sample_idx]
        subjects = [subjects[i] for i in sample_idx]
        logger.info(f"Downsampled animation to {len(coords):,} points for GIF")

    unique_subjects, subject_counts, subject_to_color = build_subject_palette(
        subjects,
        alpha=1.0,
        seed=seed,
    )
    sorted_subjects = sorted(unique_subjects, key=lambda s: subject_counts[s], reverse=True)
    order = []
    subject_array = np.array(subjects)
    for subj in sorted_subjects:
        order.extend(np.where(subject_array == subj)[0].tolist())
    order = np.array(order, dtype=int)

    coords = coords[order]
    subject_array = subject_array[order]
    subjects = subject_array.tolist()
    point_colors = np.array([subject_to_color[s][:3] for s in subjects])

    bg_color = "#0d1117"
    label_bg = "#161b22"
    label_text = "#e6edf3"

    fig, ax = plt.subplots(figsize=(figsize, figsize), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=point_size,
        alpha=0.0,
        edgecolors="none",
        rasterized=True,
    )

    label_positions = []
    for subj in unique_subjects:
        count = subject_counts[subj]
        if count >= min_cluster_size_for_label:
            mask = np.array([s == subj for s in subjects])
            centroid_x = coords[mask, 0].mean()
            centroid_y = coords[mask, 1].mean()
            label_positions.append((centroid_x, centroid_y, subj, count))

    label_positions.sort(key=lambda x: x[3], reverse=True)
    label_texts = []
    for x, y, subj, *_ in label_positions:
        clean_label = get_clean_label(subj)
        text = ax.text(
            x,
            y,
            clean_label,
            fontsize=6,
            fontweight="medium",
            ha="center",
            va="center",
            color=label_text,
            alpha=0.0,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=label_bg,
                alpha=0.0,
                edgecolor="#30363d",
                linewidth=0.5,
            ),
        )
        label_texts.append(text)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_margin = (x_max - x_min) * 0.04
    y_margin = (y_max - y_min) * 0.04
    full_x_min, full_x_max = x_min - x_margin, x_max + x_margin
    full_y_min, full_y_max = y_min - y_margin, y_max + y_margin

    x_range = full_x_max - full_x_min
    y_range = full_y_max - full_y_min
    max_range = max(x_range, y_range)
    if x_range < max_range:
        pad = (max_range - x_range) / 2
        full_x_min -= pad
        full_x_max += pad
    if y_range < max_range:
        pad = (max_range - y_range) / 2
        full_y_min -= pad
        full_y_max += pad

    bounds_center = np.array(
        [(full_x_min + full_x_max) / 2, (full_y_min + full_y_max) / 2],
        dtype=np.float32,
    )
    full_range = np.array([full_x_max - full_x_min, full_y_max - full_y_min], dtype=np.float32)

    center = bounds_center
    delta = coords - center
    spread = np.std(coords, axis=0).mean()
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=spread * 0.015, size=coords.shape)
    max_angle = np.deg2rad(max_rotation_deg)
    depth = np.linalg.norm(delta / np.maximum(full_range, 1e-6), axis=1)
    depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    tilt_vec = np.array([full_range[0] * 0.06, -full_range[1] * 0.04], dtype=np.float32)

    ax.set_aspect("equal", adjustable="box")

    label_coords = np.array([(x, y) for x, y, *_ in label_positions], dtype=np.float32) if label_positions else None

    def update(frame: int):
        t = frame / (frames - 1) if frames > 1 else 1.0
        ease = 0.5 - 0.5 * cos(np.pi * t)
        zoom = start_zoom + (1 - start_zoom) * ease
        angle = max_angle * (1 - ease)
        rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        tilt = tilt_vec * (1 - ease)
        coords_t = center + (delta @ rot.T) + tilt * depth[:, None] + noise * (1 - ease)

        half_span = full_range * zoom / 2
        ax.set_xlim(center[0] - half_span[0], center[0] + half_span[0])
        ax.set_ylim(center[1] - half_span[1], center[1] + half_span[1])

        sizes = point_size * (1.0 + 0.6 * (1 - ease) * (1 - depth))
        scatter.set_offsets(coords_t)
        scatter.set_sizes(sizes)
        scatter.set_alpha(0.05 + (alpha - 0.05) * ease)

        if t >= 0.7:
            label_alpha = min(1.0, (t - 0.7) / 0.3)
            if label_coords is not None:
                label_delta = label_coords - center
                label_depth = np.linalg.norm(
                    label_delta / np.maximum(full_range, 1e-6),
                    axis=1,
                )
                label_depth = (label_depth - label_depth.min()) / max(label_depth.max() - label_depth.min(), 1e-6)
                label_coords_t = center + (label_delta @ rot.T) + tilt * label_depth[:, None]
                for (x, y), text in zip(label_coords_t, label_texts, strict=False):
                    text.set_position((x, y))
                    text.set_alpha(label_alpha)
                    if text.get_bbox_patch() is not None:
                        text.get_bbox_patch().set_alpha(0.9 * label_alpha)

        return [scatter, *label_texts]

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved animation to {output_path}")


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
    point_size: float = typer.Option(
        1.5,
        "--point-size",
        help="Scatter point size.",
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
    animate: bool = typer.Option(
        True,
        "--animate/--no-animate",
        help="Generate an animated GIF for README usage.",
    ),
    gif_frames: int = typer.Option(
        120,
        "--gif-frames",
        help="Number of frames to render for the GIF.",
    ),
    gif_fps: int = typer.Option(
        12,
        "--gif-fps",
        help="Frames per second for the GIF.",
    ),
    gif_max_points: int | None = typer.Option(
        None,
        "--gif-max-points",
        help="Max points to include in the GIF to keep file size reasonable.",
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
        point_size=point_size,
    )

    if animate:
        gif_dir = Path("figs")
        gif_path = gif_dir / f"{output_path.stem}.gif"
        gif_figsize = min(12.0, max(8.0, figsize * 0.6))
        gif_dpi = min(160, dpi)
        create_scatter_animation(
            coords,
            sampled_subjects,
            gif_path,
            figsize=gif_figsize,
            dpi=gif_dpi,
            point_size=max(0.8, point_size * 0.8),
            frames=gif_frames,
            fps=gif_fps,
            max_points=gif_max_points,
            seed=seed,
        )
    logger.info("Visualization complete!")


if __name__ == "__main__":
    app()
