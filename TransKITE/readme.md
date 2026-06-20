# TransKITE Framework: Physics-Informed Trajectory Prediction

A unified framework for physics-informed trajectory prediction across multiple domains (UAV and UUV).

## Directory Structure

KITE_PROJECT/
│
├── model/                      # Network architectures
│   ├── __init__.py
│   └── trans_kite.py           # TransKITE hybrid architecture
│
├── configs/                    # Hyperparameter configurations
│   ├── __init__.py
│   ├── default_config.py       # UAV configuration
│   └── uuv_config.py           # UUV configuration
│
├── core/                       # Agnostic processing core
    ├── __init__.py
│   ├── dataset.py              # Dataloaders for UAV & UUV
│   ├── losses.py               # Physics-informed loss functions
│   ├── metrics.py              # Evaluation metrics (ADE, FDE, RMSE, Valid Time)
│   └── inference.py            # Autoregressive rollout modules
│
├── experiments/                # Output directory for weights and logs
├── visualizations/             # Output directory for trajectory plots
│
├── train.py                    # Unified training script
├── test.py                     # Unified testing script
├── visualize.py                # 2D/3D trajectory visualization script
└── run_experiments.py          # Automated pipeline execution

---

## Usage Guide

All operations are controlled via the Command Line Interface using the `--data` flag.

### 1. Training

To train the full TransKITE model:
```bash
python train.py --data UAV    # Train on aerial trajectories
python train.py --data UUV    # Train on underwater trajectories
```

### 2. Testing & Evaluation

To evaluate the trained models and compute metrics (ADE, FDE, RMSE, Valid Time):
```bash
python test.py --data UAV
python test.py --data UUV
```

### 3. Visualization

To generate 3D spatial plots and 2D per-axis temporal tracking plots for a specific test sample:

```bash
python visualize.py --file_idx 152
```

### 4. Automated Execution

To run the entire pipeline sequentially (Train -> Test) for both domains:
```bash
python run_experiments.py --data ALL
```

