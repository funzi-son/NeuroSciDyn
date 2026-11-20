"""Unified trajectory evaluation metrics: ADE, RMSE, FDE, Valid Time (Tv)."""

import numpy as np


def compute_trajectory_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    threshold: float = 2.0,
) -> dict[str, float]:
    """Compute ADE, RMSE, FDE, and Valid Time (steps before error >= threshold)."""
    errors = np.linalg.norm(truth - pred, axis=1)
    over = np.where(errors >= threshold)[0]
    valid_time = int(over[0]) if len(over) > 0 else int(len(errors))
    return {
        "ADE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "FDE": float(errors[-1]),
        "Tv": float(valid_time),
    }


def aggregate_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {"ADE": float("nan"), "RMSE": float("nan"), "FDE": float("nan"), "Tv": float("nan")}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}
