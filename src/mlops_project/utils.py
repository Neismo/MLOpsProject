def format_num_pairs(n: int) -> str:
    """Format number of pairs for output directory name."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    elif n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def build_output_dir_name(model: str, loss: str, num_pairs: int, balanced: bool) -> str:
    """Build output directory name from config values."""
    loss_short = "mnrl" if loss == "MultipleNegativesRankingLoss" else "contrastive"
    pairs_str = format_num_pairs(num_pairs)
    balance_str = "balanced" if balanced else "weighted"
    return f"{model}-{loss_short}-{pairs_str}-{balance_str}"
