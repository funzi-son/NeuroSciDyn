import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader

from configs.uav_config import UAV_CONFIG, get_uav_config
from configs.uuv_config import UUV_CONFIG, get_uuv_config
from core.inference import autoregressive_rollout
from core.metrics import compute_valid_time, compute_ade_fde_rmse
from core.dataset import TransKITE_UUVDataset
from model.trans_kite import TransKITEModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_uav(config):
    stats_path = os.path.join(config['save_dir'], 'norm_stats.json')
    if not os.path.exists(stats_path):
        print("Normalization stats not found. Ensure training is completed.")
        return
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        pos_max, vel_max = stats['pos_max'], stats['vel_max']

    input_dim = config['input_dim'] * 2 if config.get('use_kinematics', True) else config['input_dim']
    model = TransKITEModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], 
                           seq_len=config['seq_len'], pred_len=config['pred_len']).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], "best_transkite.pth"), map_location=DEVICE))
    model.eval()

    with open(config['test_list'], 'r') as f:
        test_files = [line.strip() for line in f.readlines() if line.strip()]

    vt_list, ade_list, fde_list = [], [], []

    for fname in test_files:
        df = pd.read_csv(os.path.join(config['data_dir'], fname))
        raw_coords = df[['tx', 'ty', 'tz']].values.astype(np.float32)
        _, truth, pred = autoregressive_rollout(model, raw_coords, config, pos_max, vel_max, DEVICE)
        if truth is None: continue
        
        vt = compute_valid_time(truth, pred, threshold=2.0)
        ade, fde, _, _ = compute_ade_fde_rmse(truth, pred)
        vt_list.append(vt); ade_list.append(ade); fde_list.append(fde)

    print("\nFINAL RESULTS (UAV):")
    print(f"  ADE: {np.mean(ade_list):.4f} m | FDE: {np.mean(fde_list):.4f} m | VT: {np.mean(vt_list):.2f} steps")
    print("="*60)

def test_uuv(config):
    scenes = config['scenes']
    global_ade_sum, global_fde_sum, global_samples = 0, 0, 0
    scene_results = {}

    print(f"\nEVALUATING UUV (ALL SCENES)")
    input_dim = config['input_dim'] * 2 if config.get('use_kinematics', True) else config['input_dim']
    
    for scene in scenes:
        model_path = os.path.join(config['save_dir'], f"best_transkite_{scene}.pth")
        if not os.path.exists(model_path): 
            print(f"Skipping {scene} (Weights not found)")
            continue

        model = TransKITEModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], 
                               seq_len=config['seq_len'], pred_len=config['pred_len']).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        test_ds = TransKITE_UUVDataset(config['data_dir'], scenes=[scene], split='test', 
                                       seq_len=config['seq_len'], pred_len=config['pred_len'], stride=2, fps=config['fps'], use_sg=config.get('use_sg_filter', True))
        if len(test_ds) == 0: continue
            
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        scene_ade_sum, scene_fde_sum, scene_samples = 0, 0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, y_pos, scale, anchor = batch['x'].to(DEVICE), batch['y_pos'].to(DEVICE), batch['scale'].to(DEVICE).unsqueeze(1), batch['anchor'].to(DEVICE).unsqueeze(1)
                out = model(x)
                pred_pos = out[0] if isinstance(out, tuple) else (out[..., :3] if out.shape[-1] == 6 else out)
                
                pred_abs = (pred_pos * scale) + anchor
                y_abs = (y_pos * scale) + anchor
                dist = torch.norm(pred_abs - y_abs, dim=-1)
                scene_ade_sum += dist.mean(dim=1).sum().item()
                scene_fde_sum += dist[:, -1].sum().item()
                scene_samples += x.size(0)
                
        avg_scene_ade = scene_ade_sum / scene_samples
        avg_scene_fde = scene_fde_sum / scene_samples
        print(f"  {scene.upper()} | Samples: {scene_samples:4d} | ADE: {avg_scene_ade:.4f}m | FDE: {avg_scene_fde:.4f}m")
        scene_results[scene] = {'ADE': avg_scene_ade, 'FDE': avg_scene_fde}
        global_samples += scene_samples; global_ade_sum += avg_scene_ade; global_fde_sum += avg_scene_fde

    if global_samples > 0:
        print("\n" + "="*60)
        print(f" FINAL BENCHMARK (UUV) | GLOBAL ADE: {global_ade_sum/len(scene_results):.4f} m | GLOBAL FDE: {global_fde_sum/len(scene_results):.4f} m")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Unified TransKITE Testing")
    parser.add_argument('--data', type=str, choices=['UAV', 'UUV'], default='UAV')
    args = parser.parse_args()

    # config = DEFAULT_CONFIG.copy() if args.data == 'UAV' else get_uuv_config()
    config = get_uav_config() if args.data == 'UAV' else get_uuv_config()
    config['save_dir'] = f"experiments/{args.data}_TransKITE_Full"

    print("="*60)
    print(f"EVALUATING MODEL: {args.data} Domain")
    print("="*60)

    if args.data == 'UAV': test_uav(config)
    elif args.data == 'UUV': test_uuv(config)

if __name__ == "__main__":
    main()