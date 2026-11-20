# UAV & UUV Trajectory Forecasting

Unified research codebase and interactive web demo for three 3D trajectory forecasting methods:

- **KITE** — A lightweight kinematics-informed GRU that augments positions with Savitzky–Golay-filtered velocities and is trained with physics-guided kinematic consistency constraints.
- **TransKITE** — A hybrid Transformer-encoder and zero-input GRU-decoder forecaster, trained with physics-informed kinematic constraints to reduce error accumulation over long horizons.
- **KC-S2O** — Kinematics-conditioned sequence-to-ODE forecaster that integrates a learned latent dynamics model from the observed history.

## Repository Structure

```text
NeuroSciDyn/
├── configs/                 # YAML configs per (method, dataset)
├── src/                     # Core package
│   ├── data/                # Shared UAV & UUV loaders
│   ├── models/              # Standalone KITE, TransKITE, KC-S2O architectures
│   ├── methods/             # Method adapters & METHOD_REGISTRY
│   ├── losses.py            # Physics-informed & time-weighted losses
│   ├── metrics.py           # ADE, RMSE, FDE, Valid Time (Tv)
│   └── utils.py             # Seeding, config loading, logging
├── checkpoints/             # Pretrained weights & normalization stats
├── data/                    # Datasets & train/val/test split lists
├── uav-forecasting-demo/    # Interactive web demo
├── train.py                 # Unified training CLI
├── evaluate.py              # Unified evaluation CLI
└── run_experiment.py        # Multi-method / multi-dataset runner
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Availability

The datasets used in this repository are publicly available:

- **UAV**: [Synthetic-UAV-Flight-Trajectories](https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories)
- **UUV**: [3DMoTraj](https://github.com/zhouhao94/3DMoTraj)

## Usage

### Train a single method

```bash
python train.py --method transkite --dataset uav
python train.py --method kcs2o --dataset uuv --epochs 100
python train.py --method kite --dataset uav --config configs/kite_uav.yaml
```

### Evaluate a trained model

```bash
python evaluate.py --method transkite --dataset uav
python evaluate.py --method kcs2o --dataset uav --checkpoint checkpoints/kcs2o/uav/best_kcs2o.pth
```

### Run experiments across all methods

```bash
# Evaluate all 3 methods on UAV using pretrained checkpoints
python run_experiment.py --method all --dataset uav --skip-train

# Train and evaluate all methods on both UAV and UUV
python run_experiment.py --method all --dataset all
```

### Web Demo (UAV only)

