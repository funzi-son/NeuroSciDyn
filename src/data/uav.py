"""UAV trajectory CSV loader and Savitzky-Golay velocity computation."""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_velocity(
    pos_seq: np.ndarray,
    dt: float = 0.1,
    use_sg_filter: bool = True,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Compute velocity via Savitzky-Golay derivative or finite differences."""
    if use_sg_filter and len(pos_seq) >= window_length:
        return savgol_filter(
            pos_seq,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=dt,
            axis=0,
        ).astype(np.float32)
    return np.gradient(pos_seq, dt, axis=0).astype(np.float32)


def read_uav_file_list(data_dir: str, list_file: str | None = None) -> list[str]:
    """Return list of CSV paths from a split list file, falling back to all CSVs in data_dir."""
    if list_file and os.path.exists(list_file):
        with open(list_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        matched = [os.path.join(data_dir, n) for n in names if os.path.exists(os.path.join(data_dir, n))]
        if matched:
            return matched
    if os.path.isdir(data_dir):
        return sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        )
    return []


def load_uav_trajectory(file_path: str) -> np.ndarray | None:
    """Load [N, 3] (tx, ty, tz) float32 coordinates from a UAV CSV file."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        return df[["tx", "ty", "tz"]].values.astype(np.float32)
    except Exception:
        return None
