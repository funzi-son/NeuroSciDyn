"""Unified evaluation entrypoint for KITE, TransKITE, and KC-S2O on UAV or UUV datasets.

Examples:
    python evaluate.py --method kite --dataset uav
    python evaluate.py --method kcs2o --dataset uav --checkpoint checkpoints/kcs2o/uav/best_kcs2o.pth
"""

import argparse
from pathlib import Path

from src.methods import METHOD_REGISTRY, get_method
from src.utils import get_device, load_config, setup_logger

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained UAV/UUV trajectory forecasting model.")
    p.add_argument("--method", required=True, choices=list(METHOD_REGISTRY))
    p.add_argument("--dataset", required=True, choices=["uav", "uuv"])
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override")
    p.add_argument("--data-dir", type=str, default=None, help="Override dataset directory")
    p.add_argument("--device", type=str, default=None, help="Device (e.g. cpu, cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "configs" / f"{args.method}_{args.dataset}.yaml"
    cfg = load_config(cfg_path, overrides={"data_dir": args.data_dir})
    device = get_device(args.device)
    logger = setup_logger("evaluate")

    runner = get_method(args.method, cfg)
    metrics = runner.evaluate(args.dataset, device, checkpoint=args.checkpoint)
    logger.info(
        f"[{args.method.upper()}][{args.dataset.upper()}] "
        f"ADE={metrics['ADE']:.4f}m | RMSE={metrics['RMSE']:.4f}m | "
        f"FDE={metrics['FDE']:.4f}m | Tv={metrics['Tv']:.2f} steps"
    )


if __name__ == "__main__":
    main()
