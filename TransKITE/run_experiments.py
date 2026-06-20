### 2. `run_experiments.py`

import subprocess
import argparse
import sys

def run_command(command):
    print(f"\n[{' '.join(command)}] ===============================================")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"Error executing: {' '.join(command)}. Please check the logs.")
        sys.exit(1)

def run_uav_experiments():
    print("\n" + "="*60)
    print("STARTING UAV EXPERIMENTS")
    print("="*60)
    run_command(["python", "train.py", "--data", "UAV"])
    run_command(["python", "test.py", "--data", "UAV"])
    run_command(["python", "visualize.py", "--file_idx", "152"])

def run_uuv_experiments():
    print("\n" + "="*60)
    print("STARTING UUV EXPERIMENTS")
    print("="*60)
    run_command(["python", "train.py", "--data", "UUV"])
    run_command(["python", "test.py", "--data", "UUV"])

def main():
    parser = argparse.ArgumentParser(description="Master Execution Script")
    parser.add_argument('--data', type=str, choices=['UAV', 'UUV', 'ALL'], default='UAV', 
                        help="Select dataset: UAV, UUV, or ALL.")
    args = parser.parse_args()

    print(f"Executing TransKITE pipeline for data mode: {args.data}")

    if args.data in ['UAV', 'ALL']:
        run_uav_experiments()
        
    if args.data in ['UUV', 'ALL']:
        run_uuv_experiments()

    print("\nAll requested experiments completed successfully.")

if __name__ == "__main__":
    main()