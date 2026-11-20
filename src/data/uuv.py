"""UUV (3DMoTraj) trajectory loader."""

import glob
import os
import numpy as np
import pandas as pd


def load_uuv_tracks(
    data_dir: str,
    scenes: list[str],
    split: str,
    min_len: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Load (positions [N, 3], frame_ids [N], scene_name) tuples for 3DMoTraj scenes."""
    tracks: list[tuple[np.ndarray, np.ndarray, str]] = []
    for scene in scenes:
        folder = os.path.join(data_dir, scene, split)
        if not os.path.isdir(folder):
            continue
        for fp in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            try:
                df = pd.read_csv(
                    fp,
                    sep=r"\s+",
                    header=None,
                    names=["frame_id", "track_id", "x", "y", "z"],
                )
                for _, group in df.groupby("track_id"):
                    group = group.sort_values("frame_id")
                    pos = group[["x", "y", "z"]].values.astype(np.float32)
                    frame_ids = group["frame_id"].values.astype(np.float32)
                    if len(pos) >= min_len:
                        tracks.append((pos, frame_ids, scene))
            except Exception:
                continue
    return tracks
