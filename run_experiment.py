"""Master experiment runner across methods (kite, transkite, kcs2o, all) and datasets (uav, uuv, all).

Examples:
    python run_experiment.py --method all --dataset uav --skip-train
    python run_experiment.py --method transkite --dataset all
"""

import argparse
from pathlib import Path

from src.methods import METHOD_REGISTRY, get_method
from src.utils import get_device, load_config, setup_logger

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run training and/or evaluation across methods and datasets.")
    p.add_argument("--method", default="all", choices=[*METHOD_REGISTRY.keys(), "all"])
    p.add_argument("--dataset", default="uav", choices=["uav", "uuv", "all"])
    p.add_argument("--skip-train", action="store_true", help="Skip training and only run evaluation")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    p.add_argument("--data-dir", type=str, default=None, help="Override dataset directory")
    p.add_argument("--device", type=str, default=None, help="Device (e.g. cpu, cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = list(METHOD_REGISTRY.keys()) if args.method == "all" else [args.method]
    datasets = ["uav", "uuv"] if args.dataset == "all" else [args.dataset]
    device = get_device(args.device)
    logger = setup_logger("experiment")

    summary: list[tuple[str, str, dict[str, float]]] = []
    for ds in datasets:
        for m in methods:
            cfg_path = ROOT / "configs" / f"{m}_{ds}.yaml"
            if not cfg_path.exists():
                logger.info(f"Skipping {m.upper()} on {ds.upper()} (no config at {cfg_path.name})")
                continue
            cfg = load_config(cfg_path, overrides={"epochs": args.epochs, "data_dir": args.data_dir})
            runner = get_method(m, cfg)

            if not args.skip_train:
                logger.info(f"=== Training {m.upper()} on {ds.upper()} ===")
                runner.train(ds, device)

            logger.info(f"=== Evaluating {m.upper()} on {ds.upper()} ===")
            metrics = runner.evaluate(ds, device)
            summary.append((ds, m, metrics))

    print("\n" + "=" * 72)
    print(f"{'Dataset':<10} {'Method':<12} {'ADE (m)':>10} {'RMSE (m)':>10} {'FDE (m)':>10} {'Tv (steps)':>12}")
    print("-" * 72)
    for ds, m, mt in summary:
        print(
            f"{ds.upper():<10} {m.upper():<12} "
            f"{mt['ADE']:>10.4f} {mt['RMSE']:>10.4f} {mt['FDE']:>10.4f} {mt['Tv']:>12.2f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
