"""Shared training, evaluation, and rollout logic for KITE and TransKITE (6D pos+vel Seq2Seq)."""

import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset

from ..data.uav import compute_velocity, load_uav_trajectory, read_uav_file_list
from ..data.uuv import load_uuv_tracks
from ..losses import kinematic_physics_loss
from ..metrics import aggregate_metrics, compute_trajectory_metrics
from ..utils import ensure_dir, set_seed, setup_logger
from .base import BaseMethod


class _ListDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _build_uav_samples(
    file_paths: list[str],
    cfg: dict[str, Any],
    training: bool = False,
    pos_max: float | None = None,
    vel_max: float | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float, float]:
    dt = cfg.get("dt", 0.1)
    seq_len = cfg["seq_len"]
    pred_len = cfg["pred_len"]
    stride = cfg.get("stride", 5)
    use_sg = cfg.get("use_sg_filter", True)

    if training:
        pos_list, vel_list = [], []
        for fp in file_paths:
            pos = load_uav_trajectory(fp)
            if pos is None or len(pos) < seq_len + pred_len:
                continue
            pos_list.append(pos)
            vel_list.append(compute_velocity(pos, dt, use_sg))
        all_pos = np.vstack(pos_list)
        all_vel = np.vstack(vel_list)
        pos_max = float(np.linalg.norm(all_pos, axis=1).max())
        vel_max = float(np.linalg.norm(all_vel, axis=1).max())

    assert pos_max is not None and vel_max is not None
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for fp in file_paths:
        pos = load_uav_trajectory(fp)
        if pos is None or len(pos) < seq_len + pred_len:
            continue
        vel = compute_velocity(pos, dt, use_sg)
        traj = np.concatenate([pos / pos_max, vel / vel_max], axis=1)
        for i in range(0, len(traj) - seq_len - pred_len + 1, stride):
            x = traj[i : i + seq_len]
            y = traj[i + seq_len : i + seq_len + pred_len]
            samples.append((torch.FloatTensor(x), torch.FloatTensor(y)))
    return samples, pos_max, vel_max


def _build_uuv_samples(
    data_dir: str,
    scenes: list[str],
    split: str,
    cfg: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    seq_len = cfg["seq_len"]
    pred_len = cfg["pred_len"]
    stride = cfg.get("stride", 2) if split == "train" else seq_len + pred_len
    fps = cfg.get("fps", 10.0)
    use_sg = cfg.get("use_sg_filter", True)

    tracks = load_uuv_tracks(data_dir, scenes, split, min_len=seq_len + pred_len)
    samples: list[dict[str, torch.Tensor]] = []
    for pos, f_ids, _ in tracks:
        for i in range(0, len(pos) - seq_len - pred_len + 1, stride):
            p_win = pos[i : i + seq_len + pred_len]
            dt = float(np.mean(np.diff(f_ids[i : i + seq_len + pred_len])) / fps)
            if dt <= 0:
                dt = cfg.get("dt", 0.1)

            anchor = p_win[seq_len - 1].copy()
            p_rel = p_win - anchor
            scale = np.maximum(np.abs(p_rel).max(axis=0), 1e-3)
            p_norm = p_rel / scale
            v_norm = compute_velocity(p_norm, dt, use_sg)

            x = np.concatenate([p_norm[:seq_len], v_norm[:seq_len]], axis=-1)
            y = np.concatenate([p_norm[seq_len:], v_norm[seq_len:]], axis=-1)
            samples.append(
                {
                    "x": torch.FloatTensor(x),
                    "y": torch.FloatTensor(y),
                    "scale": torch.FloatTensor(scale),
                    "dt": torch.FloatTensor([dt]),
                }
            )
    return samples


def _load_norm_stats(stats_path: str) -> dict[str, float]:
    if stats_path.endswith(".npz"):
        data = np.load(stats_path)
        return {"pos_max": float(np.max(data["pos_max"])), "vel_max": float(np.max(data["vel_max"]))}
    with open(stats_path, "r") as f:
        data = json.load(f)
    return {"pos_max": float(data["pos_max"]), "vel_max": float(data["vel_max"])}


class KinematicSeq2SeqBase(BaseMethod):
    """Base adapter for 6D (pos + vel) sequence-to-sequence models (KITE & TransKITE)."""

    checkpoint_filename: str = "best_model.pth"

    def _resolve_checkpoint(self, checkpoint: str | None) -> str:
        if checkpoint:
            return checkpoint
        return os.path.join(self.config["save_dir"], self.checkpoint_filename)

    def _resolve_stats_path(self, ckpt_path: str) -> str:
        ckpt_dir = os.path.dirname(ckpt_path) or self.config["save_dir"]
        for name in ("norm_stats.json", "norm_stats.npz"):
            candidate = os.path.join(ckpt_dir, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"norm_stats.json or norm_stats.npz not found in {ckpt_dir}")

    def train(self, dataset: str, device: torch.device) -> dict[str, Any]:
        set_seed(self.config.get("seed", 42))
        ensure_dir(self.config["save_dir"])
        logger = setup_logger(self.name)
        if dataset == "uav":
            return self._train_uav(device, logger)
        if dataset == "uuv":
            return self._train_uuv(device, logger)
        raise ValueError(f"Unsupported dataset: {dataset}")

    def _train_uav(self, device: torch.device, logger) -> dict[str, Any]:
        cfg = self.config
        train_files = read_uav_file_list(cfg["data_dir"], cfg.get("train_list"))
        val_files = read_uav_file_list(cfg["data_dir"], cfg.get("val_list"))
        if not train_files:
            raise FileNotFoundError(f"No UAV training files found in {cfg['data_dir']}")

        train_samples, pos_max, vel_max = _build_uav_samples(train_files, cfg, training=True)
        val_samples, _, _ = _build_uav_samples(val_files, cfg, training=False, pos_max=pos_max, vel_max=vel_max)

        with open(os.path.join(cfg["save_dir"], "norm_stats.json"), "w") as f:
            json.dump({"pos_max": pos_max, "vel_max": vel_max}, f, indent=2)

        train_loader = DataLoader(_ListDataset(train_samples), batch_size=cfg["batch_size"], shuffle=True, drop_last=len(train_samples) >= cfg["batch_size"])
        val_loader = DataLoader(_ListDataset(val_samples), batch_size=cfg["batch_size"], drop_last=False)

        model = self.build_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        best_val_rmse = float("inf")
        patience_counter = 0
        ckpt_path = self._resolve_checkpoint(None)

        for epoch in range(cfg["epochs"]):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                noise = cfg.get("input_noise", 0.0)
                bx_in = bx + torch.randn_like(bx) * noise if noise > 0 else bx
                if not cfg.get("use_kinematics", True):
                    bx_in = bx_in[:, :, :3]

                optimizer.zero_grad()
                pred = model(bx_in)
                loss = kinematic_physics_loss(
                    bx_in[:, -1, :3], by, pred, pos_max, vel_max,
                    dt=cfg.get("dt", 0.1),
                    alpha_vel=cfg.get("alpha_vel", 0.3),
                    alpha_cons=cfg.get("alpha_cons", 0.1),
                    alpha_phys=cfg.get("alpha_phys", 0.1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_rmse_sum = 0.0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vx_in = vx if cfg.get("use_kinematics", True) else vx[:, :, :3]
                    pred = model(vx_in)
                    rmse_m = torch.sqrt(torch.mean(((pred[:, :, :3] - vy[:, :, :3]) * pos_max) ** 2))
                    val_rmse_sum += rmse_m.item()

            avg_val_rmse = val_rmse_sum / max(len(val_loader), 1)
            logger.info(f"[{self.name}][UAV] Ep {epoch+1:03d}/{cfg['epochs']} | Tr.Loss: {train_loss/max(len(train_loader),1):.4f} | Val RMSE: {avg_val_rmse:.3f}m")
            scheduler.step()

            if avg_val_rmse < best_val_rmse:
                best_val_rmse = avg_val_rmse
                torch.save(model.state_dict(), ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.get("patience", 25):
                    logger.info("Early stopping triggered.")
                    break

        return {"checkpoint": ckpt_path, "best_val_rmse": best_val_rmse}

    def _train_uuv(self, device: torch.device, logger) -> dict[str, Any]:
        cfg = self.config
        results = {}
        for scene in cfg["scenes"]:
            train_samples = _build_uuv_samples(cfg["data_dir"], [scene], "train", cfg)
            val_samples = _build_uuv_samples(cfg["data_dir"], [scene], "val", cfg)
            if not train_samples:
                logger.info(f"[{self.name}][UUV] No data for {scene}, skipping.")
                continue

            train_loader = DataLoader(_ListDataset(train_samples), batch_size=cfg["batch_size"], shuffle=True)
            val_loader = DataLoader(_ListDataset(val_samples), batch_size=cfg["batch_size"], shuffle=False)

            model = self.build_model().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

            best_val_ade = float("inf")
            patience_counter = 0
            ckpt_path = os.path.join(cfg["save_dir"], f"best_{self.name}_{scene}.pth")

            for epoch in range(cfg["epochs"]):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    bx = batch["x"].to(device)
                    by = batch["y"].to(device)
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = kinematic_physics_loss(
                        bx[:, -1, :3], by, pred, pos_max=1.0, vel_max=1.0,
                        dt=cfg.get("dt", 0.1),
                        alpha_vel=cfg.get("alpha_vel", 0.1),
                        alpha_cons=cfg.get("alpha_cons", 0.1),
                        alpha_phys=cfg.get("alpha_phys", 0.1),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                ade_sum, n_samples = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        bx = batch["x"].to(device)
                        by = batch["y"].to(device)
                        scale = batch["scale"].to(device).unsqueeze(1)
                        pred = model(bx)
                        dist = torch.norm((pred[:, :, :3] - by[:, :, :3]) * scale, dim=-1)
                        ade_sum += dist.mean(dim=1).sum().item()
                        n_samples += bx.size(0)

                val_ade = ade_sum / max(n_samples, 1)
                scheduler.step(val_ade)
                if val_ade < best_val_ade:
                    best_val_ade = val_ade
                    torch.save(model.state_dict(), ckpt_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.get("patience", 30):
                        break

            logger.info(f"[{self.name}][UUV] {scene}: Best Val ADE = {best_val_ade:.4f}m")
            results[scene] = {"checkpoint": ckpt_path, "best_val_ade": best_val_ade}
        return results

    def load_for_inference(
        self,
        device: torch.device,
        checkpoint: str | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        ckpt_path = self._resolve_checkpoint(checkpoint)
        meta = _load_norm_stats(self._resolve_stats_path(ckpt_path))
        model = self.build_model().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        return model, meta

    def predict_uav_trajectory(
        self,
        model: nn.Module,
        raw_coords: np.ndarray,
        meta: dict[str, Any],
        device: torch.device,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        seq_len = self.config["seq_len"]
        pred_len = self.config["pred_len"]
        dt = self.config.get("dt", 0.1)
        use_sg = self.config.get("use_sg_filter", True)
        pos_max, vel_max = meta["pos_max"], meta["vel_max"]

        if raw_coords is None or len(raw_coords) < seq_len + pred_len:
            return None, None, None

        history = raw_coords[:seq_len]
        truth = raw_coords[seq_len:]
        limit_len = len(truth)

        vel = compute_velocity(history, dt, use_sg)
        state = np.concatenate([history / pos_max, vel / vel_max], axis=1)
        curr = torch.FloatTensor(state).unsqueeze(0).to(device)

        preds_scaled: list[list[float]] = []
        num_blocks = math.ceil(limit_len / pred_len)
        model.eval()
        with torch.no_grad():
            for _ in range(num_blocks):
                window = curr[:, -seq_len:, :]
                if not self.config.get("use_kinematics", True):
                    window = window[:, :, :3]
                pred_block = model(window)
                preds_scaled.extend(pred_block.squeeze(0).cpu().numpy().tolist())
                curr = torch.cat([curr[:, pred_len:, :], pred_block], dim=1)

        pred_coords = np.asarray(preds_scaled[:limit_len], dtype=np.float32)[:, :3] * pos_max
        return history, truth, pred_coords

    def evaluate(self, dataset: str, device: torch.device, checkpoint: str | None = None) -> dict[str, float]:
        cfg = self.config
        threshold = cfg.get("error_threshold", 2.0)

        if dataset == "uav":
            model, meta = self.load_for_inference(device, checkpoint)
            files = read_uav_file_list(cfg["data_dir"], cfg.get("test_list"))
            records = []
            for fp in files:
                coords = load_uav_trajectory(fp)
                _, truth, pred = self.predict_uav_trajectory(model, coords, meta, device)
                if truth is not None and pred is not None:
                    records.append(compute_trajectory_metrics(truth, pred, threshold))
            return aggregate_metrics(records)

        if dataset == "uuv":
            seq_len, pred_len = cfg["seq_len"], cfg["pred_len"]
            dt = cfg.get("dt", 0.1)
            use_sg = cfg.get("use_sg_filter", True)
            records = []
            for scene in cfg["scenes"]:
                ckpt_path = checkpoint or os.path.join(cfg["save_dir"], f"best_{self.name}_{scene}.pth")
                if not os.path.exists(ckpt_path):
                    continue
                model = self.build_model().to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                model.eval()
                for pos, _, _ in load_uuv_tracks(cfg["data_dir"], [scene], "test", min_len=seq_len + 1):
                    truth = pos[seq_len:]
                    curr_hist = pos[:seq_len].copy()
                    preds_abs: list[np.ndarray] = []
                    with torch.no_grad():
                        for _ in range(math.ceil(len(truth) / pred_len)):
                            recent = curr_hist[-seq_len:]
                            anchor = recent[-1].copy()
                            p_rel = recent - anchor
                            scale = np.maximum(np.abs(p_rel).max(axis=0), 1e-3)
                            p_norm = p_rel / scale
                            v_norm = compute_velocity(p_norm, dt, use_sg)
                            x = torch.FloatTensor(np.concatenate([p_norm, v_norm], axis=-1)).unsqueeze(0).to(device)
                            out = model(x).squeeze(0).cpu().numpy()[:, :3]
                            pred_abs = out * scale + anchor
                            preds_abs.extend(pred_abs)
                            curr_hist = np.vstack([curr_hist, pred_abs])[-seq_len:]
                    records.append(compute_trajectory_metrics(truth, np.asarray(preds_abs)[: len(truth)], threshold))
            return aggregate_metrics(records)

        raise ValueError(f"Unsupported dataset: {dataset}")

    def build_onnx_export_spec(
        self,
        checkpoint: str | None = None,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], dict[str, Any]]:
        model, meta = self.load_for_inference(torch.device("cpu"), checkpoint)
        in_dim = self.config["input_dim"] * 2 if self.config.get("use_kinematics", True) else self.config["input_dim"]
        example = (torch.randn(1, self.config["seq_len"], in_dim),)
        return model, example, ["x"], meta
