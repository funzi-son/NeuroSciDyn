"""Unified training entrypoint for KITE, TransKITE, and KC-S2O on UAV or UUV datasets.

Examples:
    python train.py --method transkite --dataset uav
    python train.py --method kcs2o --dataset uuv --epochs 50
    python train.py --method kite --dataset uav --config configs/kite_uav.yaml
"""

import argparse
from pathlib import Path

from src.methods import METHOD_REGISTRY, get_method
from src.utils import get_device, load_config, setup_logger

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a UAV/UUV trajectory forecasting model.")
    p.add_argument("--method", required=True, choices=list(METHOD_REGISTRY))
    p.add_argument("--dataset", required=True, choices=["uav", "uuv"])
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (defaults to configs/<method>_<dataset>.yaml)")
    p.add_argument("--data-dir", type=str, default=None, help="Override dataset directory")
    p.add_argument("--save-dir", type=str, default=None, help="Override checkpoint output directory")
    p.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--device", type=str, default=None, help="Device (e.g. cpu, cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "configs" / f"{args.method}_{args.dataset}.yaml"
    cfg = load_config(
        cfg_path,
        overrides={
            "data_dir": args.data_dir,
            "save_dir": args.save_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
    )
    device = get_device(args.device)
    logger = setup_logger("train")
    logger.info(f"Training {args.method.upper()} on {args.dataset.upper()} (device={device})")

    runner = get_method(args.method, cfg)
    summary = runner.train(args.dataset, device)
    logger.info(f"Training finished: {summary}")


if __name__ == "__main__":
    main()
