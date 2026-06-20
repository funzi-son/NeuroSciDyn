import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader

from configs.uav_config import UAV_CONFIG
from configs.uuv_config import UUV_CONFIG
from core.dataset import TransKITE_UUVDataset
from core.inference import autoregressive_rollout
from core.metrics import compute_valid_time
from model.trans_kite import TransKITEModel

# =========================================================================
# 1. UAV VISUALIZATION (1D, 3D Static, 3D Animated)
# =========================================================================
def visualize_uav_trajectory(history, truth, pred, vt, save_dir, file_name):
    print(f"\nGenerating plots for UAV: {file_name}...")
    os.makedirs(save_dir, exist_ok=True)
    
    T = len(history)
    F = len(truth)
    time_hist = np.arange(T)
    time_fut = np.arange(T, T + F)

    # 1. 1D Axis Separated Tracking
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axis_names = ['X', 'Y', 'Z']
    
    for i in range(3):
        axs[i].plot(time_hist, history[:, i], color='black', linestyle='--', linewidth=1.0, alpha=0.6, label='History' if i==0 else "")
        axs[i].plot(time_fut, truth[:, i], color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Ground Truth' if i==0 else "")
        
        if vt > 0:
            axs[i].plot(time_fut[:vt], pred[:vt, i], color='#D62728', linestyle='-', linewidth=1.5, label='TransKITE' if i==0 else "")
            axs[i].scatter(time_fut[vt-1], pred[vt-1, i], color='#D62728', s=25, marker='o', zorder=5)
            
        if vt < F:
            start_idx = max(0, vt-1)
            axs[i].plot(time_fut[start_idx:], pred[start_idx:, i], color='#D62728', linestyle='--', linewidth=1.5, alpha=0.8)
                
        axs[i].set_ylabel(axis_names[i])
        axs[i].grid(True, linestyle=':', alpha=0.5)
    
    axs[2].set_xlabel('Time Step')
    fig.suptitle("TransKITE - Axis Separated Tracking", fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "TransKITE1D.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 3D Static Plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(history[:,0], history[:,1], history[:,2], color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='History')
    ax.plot(truth[:,0], truth[:,1], truth[:,2], color='gray', linestyle='-', linewidth=1.5, label='Ground Truth')
    
    if vt > 0:
        ax.plot(pred[:vt,0], pred[:vt,1], pred[:vt,2], color='#D62728', linestyle='-', linewidth=2.5, label='TransKITE')
        ax.scatter(pred[vt-1,0], pred[vt-1,1], pred[vt-1,2], color='red', s=60, marker='o', depthshade=False, label='Time Valid End')
    
    if vt < F:
        start_idx = max(0, vt-1)
        ax.plot(pred[start_idx:,0], pred[start_idx:,1], pred[start_idx:,2], color='#D62728', linestyle='--', linewidth=2.5, alpha=0.8)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("TransKITE - 3D Trajectory", fontweight='bold', fontsize=14)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "TransKITE3D.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 3D Animated Plot (.GIF)
    print("  -> Rendering 3D Animation (This might take a few seconds)...")
    fig_ani = plt.figure(figsize=(9, 9))
    ax_ani = fig_ani.add_subplot(111, projection='3d')
    
    all_pts = np.vstack([history, truth, pred])
    ax_ani.set_xlim(all_pts[:,0].min() - 1, all_pts[:,0].max() + 1)
    ax_ani.set_ylim(all_pts[:,1].min() - 1, all_pts[:,1].max() + 1)
    ax_ani.set_zlim(all_pts[:,2].min() - 1, all_pts[:,2].max() + 1)
    
    ax_ani.set_xlabel('X')
    ax_ani.set_ylabel('Y')
    ax_ani.set_zlabel('Z')
    ax_ani.set_title("TransKITE - 3D Animated Rollout", fontweight='bold', fontsize=14)
    
    line_hist, = ax_ani.plot([], [], [], color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='History')
    line_gt, = ax_ani.plot([], [], [], color='gray', linestyle='-', linewidth=1.5, label='Ground Truth')
    line_pred_v, = ax_ani.plot([], [], [], color='#D62728', linestyle='-', linewidth=2.5, label='TransKITE (Valid)')
    line_pred_inv, = ax_ani.plot([], [], [], color='#D62728', linestyle='--', linewidth=2.5, alpha=0.8, label='Drift')
    pt_vt, = ax_ani.plot([], [], [], color='red', marker='o', markersize=6, linestyle='None')
    
    ax_ani.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    def update(frame):
        if frame < T:
            line_hist.set_data(history[:frame+1, 0], history[:frame+1, 1])
            line_hist.set_3d_properties(history[:frame+1, 2])
            return line_hist,
        else:
            line_hist.set_data(history[:, 0], history[:, 1])
            line_hist.set_3d_properties(history[:, 2])
            
            f_idx = frame - T + 1
            line_gt.set_data(truth[:f_idx, 0], truth[:f_idx, 1])
            line_gt.set_3d_properties(truth[:f_idx, 2])
            
            if f_idx <= vt:
                line_pred_v.set_data(pred[:f_idx, 0], pred[:f_idx, 1])
                line_pred_v.set_3d_properties(pred[:f_idx, 2])
                if f_idx > 0:
                    pt_vt.set_data([pred[f_idx-1, 0]], [pred[f_idx-1, 1]])
                    pt_vt.set_3d_properties([pred[f_idx-1, 2]])
            else:
                line_pred_v.set_data(pred[:vt, 0], pred[:vt, 1])
                line_pred_v.set_3d_properties(pred[:vt, 2])
                if vt > 0:
                    pt_vt.set_data([pred[vt-1, 0]], [pred[vt-1, 1]])
                    pt_vt.set_3d_properties([pred[vt-1, 2]])
                
                start_idx = max(0, vt-1)
                line_pred_inv.set_data(pred[start_idx:f_idx, 0], pred[start_idx:f_idx, 1])
                line_pred_inv.set_3d_properties(pred[start_idx:f_idx, 2])
                
            return line_hist, line_gt, line_pred_v, line_pred_inv, pt_vt

    ani = FuncAnimation(fig_ani, update, frames=T+F, interval=100, blit=False)
    gif_path = os.path.join(save_dir, "TransKITE_Animated.gif")
    ani.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig_ani)

    print(f"Visualization plots successfully saved to: {save_dir}")

def process_uav(args, device):
    with open(UAV_CONFIG['test_list'], 'r') as f:
        test_files = [line.strip() for line in f.readlines() if line.strip()]
    
    target_file = test_files[args.file_idx]
    file_path = os.path.join(UAV_CONFIG['data_dir'], target_file)
    print(f"\nAnalyzing UAV trajectory file: {target_file}")
    
    df = pd.read_csv(file_path)
    raw_coords = df[['tx', 'ty', 'tz']].values.astype(np.float32)

    save_dir = "experiments/UAV_TransKITE_Full"
    stats_path = os.path.join(save_dir, 'norm_stats.json')
    
    if not os.path.exists(stats_path):
        print(f"Normalization stats not found. Ensure training is completed.")
        return
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        pos_max, vel_max = stats['pos_max'], stats['vel_max']
        
    config_path = os.path.join(save_dir, 'config.json')
    config = json.load(open(config_path, 'r')) if os.path.exists(config_path) else UAV_CONFIG.copy()
        
    input_dim = config['input_dim'] * 2 if config.get('use_kinematics', True) else config['input_dim']
    model = TransKITEModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], seq_len=config['seq_len'], pred_len=config['pred_len']).to(device)
    
    weight_path = os.path.join(save_dir, "best_transkite.pth")
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
    else:
        print(f"Weight file not found at {weight_path}")
        return
        
    inp, trt, pred = autoregressive_rollout(model, raw_coords, config, pos_max, vel_max, device)
    if pred is not None:
        vt = compute_valid_time(trt, pred, threshold=2.0)
        output_dir = f"visualizations/UAV_{args.file_idx}"
        visualize_uav_trajectory(inp, trt, pred, vt, output_dir, target_file)
    else:
        print("Trajectory length is too short for prediction.")

# =========================================================================
# 2. UUV VISUALIZATION (3x3 Grid containing 8 scenes)
# =========================================================================
def process_uuv(device):
    print("\nCollecting UUV trajectories for global visualization...")
    save_dir = "experiments/UUV_TransKITE_Full"
    config_path = os.path.join(save_dir, 'config.json')
    config = json.load(open(config_path, 'r')) if os.path.exists(config_path) else UUV_CONFIG.copy()
    
    scenes = config['scenes']
    collected_plot_data = {}
    input_dim = config['input_dim'] * 2 if config.get('use_kinematics', True) else config['input_dim']
    
    for scene in scenes:
        model_path = os.path.join(save_dir, f"best_transkite_{scene}.pth")
        if not os.path.exists(model_path):
            print(f"  -> [Skipping] Model weights not found for {scene}")
            continue

        model = TransKITEModel(input_dim=input_dim, hidden_dim=config['hidden_dim'], seq_len=config['seq_len'], pred_len=config['pred_len']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        test_ds = TransKITE_UUVDataset(config['data_dir'], scenes=[scene], split='test', 
                                       seq_len=config['seq_len'], pred_len=config['pred_len'], 
                                       stride=2, fps=config['fps'], use_sg=config.get('use_sg_filter', True))
        if len(test_ds) == 0: continue
            
        # Extract only 1 sample (batch_size=1) for plotting
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=True) 

        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                y_pos = batch['y_pos'].to(device)
                scale = batch['scale'].to(device).unsqueeze(1)
                anchor = batch['anchor'].to(device).unsqueeze(1)

                out = model(x)
                pred_pos = out[0] if isinstance(out, tuple) else (out[..., :3] if out.shape[-1] == 6 else out)

                # Denormalize to real-world coordinates
                pred_abs = (pred_pos * scale) + anchor
                y_abs = (y_pos * scale) + anchor
                hist_abs = (x[:, :, :3] * scale) + anchor

                collected_plot_data[scene] = {
                    'hist': hist_abs[0].cpu().numpy(),
                    'truth': y_abs[0].cpu().numpy(),
                    'pred': pred_abs[0].cpu().numpy()
                }
                break # One sample is sufficient for each scene

    if not collected_plot_data:
        print("No UUV data collected for plotting. Ensure training is completed.")
        return

    # Draw 3x3 grid
    print("  -> Rendering 3x3 UUV Global Plot...")
    os.makedirs("visualizations", exist_ok=True)
    fig_all = plt.figure(figsize=(20, 20))
    
    # First subplot reserved for Legend
    ax_legend = fig_all.add_subplot(3, 3, 1)
    ax_legend.axis('off')
    ax_legend.plot([], [], color='gray', linestyle=':', linewidth=2, alpha=0.6, label='History (Input)')
    ax_legend.plot([], [], color='black', linewidth=2, label='Ground Truth')
    ax_legend.plot([], [], color='#D62728', linewidth=2.5, label='TransKITE (Predicted)')
    ax_legend.legend(loc='center', fontsize=16)
    
    for idx, scene in enumerate(scenes):
        if scene not in collected_plot_data: continue
            
        data = collected_plot_data[scene]
        hist, truth, pred = data['hist'], data['truth'], data['pred']
        
        ax_3d = fig_all.add_subplot(3, 3, idx + 2, projection='3d')
        ax_3d.set_box_aspect((1, 1, 0.8)) # Keep standard aspect ratio
        
        ax_3d.plot(hist[:,0], hist[:,1], hist[:,2], color='gray', linestyle=':', linewidth=2, alpha=0.6)
        ax_3d.plot(truth[:,0], truth[:,1], truth[:,2], color='black', linewidth=2)
        ax_3d.plot(pred[:,0], pred[:,1], pred[:,2], color='#D62728', linewidth=2.5) 
        
        # Automatic isotropic scaling for better aspect ratio
        all_pts = np.vstack([hist, truth, pred])
        max_range = np.array([
            all_pts[:,0].max() - all_pts[:,0].min(), 
            all_pts[:,1].max() - all_pts[:,1].min(), 
            all_pts[:,2].max() - all_pts[:,2].min()
        ]).max() / 2.0
        max_range = max(max_range, 0.5) 
        mid_x = (all_pts[:,0].max() + all_pts[:,0].min()) * 0.5
        mid_y = (all_pts[:,1].max() + all_pts[:,1].min()) * 0.5
        mid_z = (all_pts[:,2].max() + all_pts[:,2].min()) * 0.5
        
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
        ax_3d.set_title(f"{scene.upper()}", fontweight='bold', fontsize=14)
        ax_3d.grid(True, linestyle='--', alpha=0.5)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join("visualizations", "UUV_All_Scenes.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig_all)
    print(f"✅ UUV global plot saved to: {plot_path}")

# =========================================================================
# MAIN ROUTER
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="TransKITE Trajectory Visualizer")
    parser.add_argument('--data', type=str, choices=['UAV', 'UUV'], default='UAV', help="Select dataset domain to visualize.")
    parser.add_argument('--file_idx', type=int, default=96, help="Test file index to visualize (UAV only).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.data == 'UAV':
        process_uav(args, device)
    elif args.data == 'UUV':
        process_uuv(device)

if __name__ == "__main__":
    main()