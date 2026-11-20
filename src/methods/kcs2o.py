"""KC-S2O method adapter (local isotropic normalization + RK4 Neural ODE)."""

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from ..data.uav import load_uav_trajectory, read_uav_file_list
from ..data.uuv import load_uuv_tracks
from ..losses import time_weighted_loss
from ..metrics import aggregate_metrics, compute_trajectory_metrics
from ..models.kcs2o import KCS2OModel
from ..utils import ensure_dir, set_seed, setup_logger
from .base import BaseMethod


def _normalize_window(p_rel: np.ndarray, use_isotropic: bool) -> tuple[np.ndarray, np.ndarray]:
    if use_isotropic:
        scale = max(float(np.linalg.norm(p_rel, axis=1).max()), 1e-3)
        scale_arr = np.array([scale], dtype=np.float32)
    else:
        scale_arr = np.maximum(np.abs(p_rel).max(axis=0), 1e-3).astype(np.float32)
    return (p_rel / scale_arr).astype(np.float32), scale_arr


class _KCS2ODataset(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _build_uav_samples(file_paths: list[str], cfg: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
    seq_len, pred_len = cfg["seq_len"], cfg["pred_len"]
    stride = cfg.get("stride", 10)
    dt = cfg.get("dt", 0.1)
    use_iso = cfg.get("use_isotropic_norm", True)

    samples: list[dict[str, torch.Tensor]] = []
    for fp in file_paths:
        pos = load_uav_trajectory(fp)
        if pos is None or len(pos) <= seq_len + pred_len:
            continue
        for i in range(0, len(pos) - seq_len - pred_len + 1, stride):
            p_win = pos[i : i + seq_len + pred_len]
            anchor = p_win[seq_len - 1].copy()
            p_norm, scale_arr = _normalize_window(p_win - anchor, use_iso)
            v_init = (p_norm[seq_len - 1] - p_norm[seq_len - 2]) / dt
            samples.append(
                {
                    "x": torch.FloatTensor(p_norm[:seq_len]),
                    "y": torch.FloatTensor(p_norm[seq_len:]),
                    "v_init": torch.FloatTensor(v_init),
                    "dt": torch.FloatTensor([dt]),
                    "scale": torch.FloatTensor(scale_arr),
                }
            )
    return samples


def _build_uuv_samples(
    data_dir: str,
    scenes: list[str],
    split: str,
    cfg: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    seq_len, pred_len = cfg["seq_len"], cfg["pred_len"]
    stride = cfg.get("stride", 2) if split == "train" else seq_len + pred_len
    fps = cfg.get("fps", 10.0)
    use_iso = cfg.get("use_isotropic_norm", True)

    tracks = load_uuv_tracks(data_dir, scenes, split, min_len=seq_len + pred_len)
    samples: list[dict[str, torch.Tensor]] = []
    for pos, f_ids, _ in tracks:
        for i in range(0, len(pos) - seq_len - pred_len + 1, stride):
            p_win = pos[i : i + seq_len + pred_len]
            dt = float(np.mean(np.diff(f_ids[i : i + seq_len + pred_len])) / fps)
            if dt <= 0:
                dt = cfg.get("dt", 0.1)
            anchor = p_win[seq_len - 1].copy()
            p_norm, scale_arr = _normalize_window(p_win - anchor, use_iso)
            v_init = (p_norm[seq_len - 1] - p_norm[seq_len - 2]) / dt
            samples.append(
                {
                    "x": torch.FloatTensor(p_norm[:seq_len]),
                    "y": torch.FloatTensor(p_norm[seq_len:]),
                    "v_init": torch.FloatTensor(v_init),
                    "dt": torch.FloatTensor([dt]),
                    "scale": torch.FloatTensor(scale_arr),
                }
            )
    return samples


class KCS2OMethod(BaseMethod):
    name = "kcs2o"
    checkpoint_filename = "best_kcs2o.pth"

    def build_model(self) -> nn.Module:
        cfg = self.config
        return KCS2OModel(
            input_dim=cfg.get("input_dim", 3),
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 3),
            use_kinematic_prior=cfg.get("use_kinematic_prior", True),
            dropout=cfg.get("dropout", 0.2),
        )

    def _resolve_checkpoint(self, checkpoint: str | None) -> str:
        return checkpoint or os.path.join(self.config["save_dir"], self.checkpoint_filename)

    def _train_loop(
        self,
        train_samples: list[dict[str, torch.Tensor]],
        val_samples: list[dict[str, torch.Tensor]],
        ckpt_path: str,
        device: torch.device,
        tag: str,
        logger,
    ) -> float:
        cfg = self.config
        train_loader = DataLoader(_KCS2ODataset(train_samples), batch_size=cfg["batch_size"], shuffle=True, drop_last=len(train_samples) >= cfg["batch_size"])
        val_loader = DataLoader(_KCS2ODataset(val_samples), batch_size=cfg["batch_size"], shuffle=False)

        model = self.build_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-5))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

        best_val_ade = float("inf")
        patience_counter = 0

        for epoch in range(cfg["epochs"]):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                v_init = batch["v_init"].to(device)
                dt = batch["dt"].to(device)

                optimizer.zero_grad()
                pred = model(x, v_init=v_init, dt_batch=dt, predict_steps=cfg["pred_len"])
                loss = time_weighted_loss(
                    pred, y,
                    loss_fn=cfg.get("loss_fn", "smooth_l1"),
                    use_time_weight=cfg.get("use_time_weight", True),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_ade_sum, n_samples = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    v_init = batch["v_init"].to(device)
                    dt = batch["dt"].to(device)
                    scale = batch["scale"].to(device).unsqueeze(1)
                    pred = model(x, v_init=v_init, dt_batch=dt, predict_steps=cfg["pred_len"])
                    dist = torch.norm((pred - y) * scale, dim=-1)
                    val_ade_sum += dist.mean(dim=1).sum().item()
                    n_samples += x.size(0)

            val_ade = val_ade_sum / max(n_samples, 1)
            scheduler.step(val_ade)
            logger.info(f"[{self.name}][{tag}] Ep {epoch+1:03d}/{cfg['epochs']} | Tr.Loss: {train_loss/max(len(train_loader),1):.5f} | Val ADE: {val_ade:.4f}m")

            if val_ade < best_val_ade:
                best_val_ade = val_ade
                torch.save(model.state_dict(), ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.get("patience", 30):
                    logger.info("Early stopping triggered.")
                    break

        return best_val_ade

    def train(self, dataset: str, device: torch.device) -> dict[str, Any]:
        set_seed(self.config.get("seed", 42))
        ensure_dir(self.config["save_dir"])
        logger = setup_logger(self.name)
        cfg = self.config

        if dataset == "uav":
            train_files = read_uav_file_list(cfg["data_dir"], cfg.get("train_list"))
            val_files = read_uav_file_list(cfg["data_dir"], cfg.get("val_list"))
            train_samples = _build_uav_samples(train_files, cfg)
            val_samples = _build_uav_samples(val_files, cfg)
            if not train_samples:
                raise FileNotFoundError(f"No UAV training samples found in {cfg['data_dir']}")
            ckpt_path = self._resolve_checkpoint(None)
            best_ade = self._train_loop(train_samples, val_samples, ckpt_path, device, "UAV", logger)
            return {"checkpoint": ckpt_path, "best_val_ade": best_ade}

        if dataset == "uuv":
            results = {}
            for scene in cfg["scenes"]:
                train_samples = _build_uuv_samples(cfg["data_dir"], [scene], "train", cfg)
                val_samples = _build_uuv_samples(cfg["data_dir"], [scene], "val", cfg)
                if not train_samples:
                    logger.info(f"[{self.name}][UUV] No data for {scene}, skipping.")
                    continue
                ckpt_path = os.path.join(cfg["save_dir"], f"best_kcs2o_{scene}.pth")
                best_ade = self._train_loop(train_samples, val_samples, ckpt_path, device, f"UUV/{scene}", logger)
                results[scene] = {"checkpoint": ckpt_path, "best_val_ade": best_ade}
            return results

        raise ValueError(f"Unsupported dataset: {dataset}")

    def load_for_inference(
        self,
        device: torch.device,
        checkpoint: str | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        ckpt_path = self._resolve_checkpoint(checkpoint)
        model = self.build_model().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        return model, {}

    def _rollout(
        self,
        model: nn.Module,
        raw_coords: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        seq_len, pred_len = self.config["seq_len"], self.config["pred_len"]
        dt = self.config.get("dt", 0.1)
        use_iso = self.config.get("use_isotropic_norm", True)

        if raw_coords is None or len(raw_coords) <= seq_len:
            return None, None, None

        history = raw_coords[:seq_len].copy()
        truth = raw_coords[seq_len:]
        dt_tensor = torch.FloatTensor([[dt]]).to(device)

        curr_hist = history.copy()
        preds_abs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for _ in range(math.ceil(len(truth) / pred_len)):
                anchor = curr_hist[-1].copy()
                norm_hist, scale_arr = _normalize_window(curr_hist - anchor, use_iso)
                v_init = (norm_hist[-1] - norm_hist[-2]) / dt

                x_t = torch.FloatTensor(norm_hist).unsqueeze(0).to(device)
                v_t = torch.FloatTensor(v_init).unsqueeze(0).to(device)
                pred_norm = model(x_t, v_init=v_t, dt_batch=dt_tensor, predict_steps=pred_len)
                pred_abs = pred_norm.squeeze(0).cpu().numpy() * scale_arr + anchor
                preds_abs.extend(pred_abs)
                curr_hist = np.vstack([curr_hist, pred_abs])[-seq_len:]

        return history, truth, np.asarray(preds_abs, dtype=np.float32)[: len(truth)]

    def predict_uav_trajectory(
        self,
        model: nn.Module,
        raw_coords: np.ndarray,
        meta: dict[str, Any],
        device: torch.device,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        return self._rollout(model, raw_coords, device)

    def evaluate(self, dataset: str, device: torch.device, checkpoint: str | None = None) -> dict[str, float]:
        cfg = self.config
        threshold = cfg.get("error_threshold", 2.0)

        if dataset == "uav":
            model, _ = self.load_for_inference(device, checkpoint)
            files = read_uav_file_list(cfg["data_dir"], cfg.get("test_list"))
            records = []
            for fp in files:
                coords = load_uav_trajectory(fp)
                _, truth, pred = self._rollout(model, coords, device)
                if truth is not None and pred is not None:
                    records.append(compute_trajectory_metrics(truth, pred, threshold))
            return aggregate_metrics(records)

        if dataset == "uuv":
            records = []
            for scene in cfg["scenes"]:
                ckpt_path = checkpoint or os.path.join(cfg["save_dir"], f"best_kcs2o_{scene}.pth")
                if not os.path.exists(ckpt_path):
                    continue
                model = self.build_model().to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                model.eval()
                for pos, _, _ in load_uuv_tracks(cfg["data_dir"], [scene], "test", min_len=cfg["seq_len"] + 1):
                    _, truth, pred = self._rollout(model, pos, device)
                    if truth is not None and pred is not None:
                        records.append(compute_trajectory_metrics(truth, pred, threshold))
            return aggregate_metrics(records)

        raise ValueError(f"Unsupported dataset: {dataset}")

    def build_onnx_export_spec(
        self,
        checkpoint: str | None = None,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], dict[str, Any]]:
        net, _ = self.load_for_inference(torch.device("cpu"), checkpoint)
        dt = self.config.get("dt", 0.1)
        pred_len = self.config["pred_len"]
        seq_len = self.config["seq_len"]

        class _OnnxWrap(nn.Module):
            def __init__(self, inner: nn.Module):
                super().__init__()
                self.net = inner

            def forward(self, x: torch.Tensor, v_init: torch.Tensor) -> torch.Tensor:
                dt_batch = torch.full((x.shape[0], 1), dt, device=x.device, dtype=x.dtype)
                return self.net(x, v_init=v_init, dt_batch=dt_batch, predict_steps=pred_len)

        wrapper = _OnnxWrap(net).eval()
        inputs = (torch.randn(1, seq_len, 3) * 0.3, torch.randn(1, 3))
        return wrapper, inputs, ["x", "v_init"], {}
