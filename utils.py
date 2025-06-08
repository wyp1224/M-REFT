"""Refactored utility functions for knowledge graph evaluation metrics."""

from typing import List, Tuple
import numpy as np

def calculate_rank(
    scores: np.ndarray,
    target_index: int,
    filter_indices: List[int]
) -> int:
    """
    Calculate the rank of a target entity among all possible candidates.

    Args:
        scores: Array of prediction scores for all entities
        target_index: Index of the target entity in scores array
        filter_indices: List of indices to exclude from ranking

    Returns:
        Rank of the target entity (1-based)

    Note:
        This implements the filtered ranking protocol commonly used in KG completion tasks.
        All entities in filter_indices are artificially given lower scores than the target.
    """
    target_score = scores[target_index]
    # Artificially lower scores of filtered entities
    scores[filter_indices] = target_score - 1
    # Calculate rank (handling ties by averaging)
    greater = np.sum(scores > target_score)
    equal = np.sum(scores == target_score)
    return greater + (equal // 2) + 1

def compute_metrics(
    ranks: np.ndarray,
    cutoff_values: Tuple[int, ...] = (1, 3, 10)
) -> Tuple[float, float, float, float, float]:
    """
    Compute standard knowledge graph evaluation metrics from ranks.
    """
    # Input validation
    if len(ranks) == 0:
        raise ValueError("Cannot compute metrics on empty rank array")
    if np.any(ranks < 1):
        raise ValueError("Ranks must be 1-based")

    # Compute basic metrics
    mean_rank = np.mean(ranks)
    mrr = np.mean(1 / ranks)

    # Compute hit@k metrics
    hit10 = np.mean(ranks <= 10)
    hit3 = np.mean(ranks <= 3)
    hit1 = np.mean(ranks <= 1)

    # Additional metrics based on cutoff_values
    if len(cutoff_values) > 3:
        # Dynamically compute additional hit@k metrics if requested
        additional_hits = [np.mean(ranks <= k) for k in cutoff_values[3:]]
        return (mean_rank, mrr, hit10, hit3, hit1) + tuple(additional_hits)

    return mean_rank, mrr, hit10, hit3, hit1

