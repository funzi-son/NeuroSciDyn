import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

# Download data: https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories

# ==========================================
# 1. CONFIGURATION & ABLATION STUDY SETTINGS
# ==========================================
CONFIG = {
    'data_dir': 'Synthetic-UAV-Flight-Trajectories', 
    'train_list': 'path_list_train_files.txt',
    'val_list': 'path_list_val_files.txt',
    'test_list': 'path_list_test_files.txt',
    
    'save_dir': 'results_uav_ode_final', 
    'plot_dir': 'results_uav_ode_final/plots',
    
    # ----------------------------------------------------
    # ABLATION STUDY TOGGLES
    # ----------------------------------------------------
    'ablation': {
        # 1. Explicit Kinematic Prior: Set False to remove instantaneous velocity injection (v_init)
        'use_kinematic_prior': True, 
        
        # 2. Time-Weighted Robust Optimization: Set False to remove exponential temporal penalty
        'use_time_weight': True,     
        # Change to 'mse' to utilize standard Mean Squared Error (MSE Only)
        'loss_fn': 'smooth_l1',      
        
        # 3. Isotropic Normalization: Set False to use axis-independent Min-Max scaling (w/o Isotropic Norm)
        'use_isotropic_norm': True   
    },
    
    # UAV time-series configuration
    'seq_len': 50,       
    'pred_len': 20,      
    'stride': 10,
    'dt_fixed': 0.1,     # Fixed delta t for simulated data (10Hz)
    
    # Model configuration
    'input_dim': 3,      
    'hidden_dim': 64,    
    'num_layers': 3,     
    
    # Training configuration
    'batch_size': 64,    
    'epochs': 300,       
    'lr': 1e-3,          
    'weight_decay': 1e-4,
    'patience': 30,      
    'seed': 42,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # Testing configuration
    'error_threshold': 2.0 
}

# Dynamically generate model name to prevent overwriting during ablation studies
def get_model_name():
    suffix = "base"
    if not CONFIG['ablation']['use_kinematic_prior']: suffix += "_no_kinematic"
    if not CONFIG['ablation']['use_time_weight']: suffix += "_no_timeweight"
    if CONFIG['ablation']['loss_fn'] == 'mse': suffix += "_mse"
    if not CONFIG['ablation']['use_isotropic_norm']: suffix += "_no_isotropic"
    return f"uav_model_{suffix}.pth"

CONFIG['model_name'] = get_model_name()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 2. DATA PROCESSING (UAV DATASET)
# ==========================================
class UAV_Dataset_3D(Dataset):
    def __init__(self, list_file, base_dir, seq_len=50, pred_len=20, stride=10, dt=0.1):
        self.samples = []
        
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                file_names = [line.strip() for line in f if line.strip()]
            file_paths = [os.path.join(base_dir, f) for f in file_names]
        else:
            file_paths = []

        print(f"Processing {len(file_paths)} files from {list_file}...")
        print(f"Applying Isotropic Norm: {CONFIG['ablation']['use_isotropic_norm']}")

        for fp in file_paths:
            if not os.path.exists(fp): continue
            try:
                df = pd.read_csv(fp)
                if len(df) <= seq_len + pred_len: continue
                
                pos = df[['tx', 'ty', 'tz']].values.astype(np.float32)
                total_len = len(pos)
                
                for i in range(0, total_len - seq_len - pred_len + 1, stride):
                    p_window = pos[i : i + seq_len + pred_len]
                    
                    anchor = p_window[seq_len - 1].copy()
                    p_rel = p_window - anchor
                    
                    # Ablation: Isotropic Normalization vs Axis-Independent Scaling
                    if CONFIG['ablation']['use_isotropic_norm']:
                        local_scale = float(np.linalg.norm(p_rel, axis=1).max())
                        local_scale = max(local_scale, 1e-3) 
                        scale_arr = np.array([local_scale]) # Shape (1,)
                    else:
                        local_scale = np.abs(p_rel).max(axis=0)
                        local_scale = np.maximum(local_scale, 1e-3)
                        scale_arr = local_scale # Shape (3,)
                    
                    p_norm = p_rel / scale_arr
                    
                    v_init = (p_norm[seq_len - 1] - p_norm[seq_len - 2]) / dt
                    
                    x = torch.FloatTensor(p_norm[:seq_len])
                    y = torch.FloatTensor(p_norm[seq_len:])
                    
                    self.samples.append({
                        'x': x, 
                        'y': y, 
                        'v_init': torch.FloatTensor(v_init),
                        'dt': torch.FloatTensor([dt]),
                        'anchor': torch.FloatTensor(anchor),
                        'scale': torch.FloatTensor(scale_arr)
                    })
            except Exception as e:
                pass

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 3. ARCHITECTURE: KC - S2O 3D
# ==========================================
class ODEFunc3D(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunc3D, self).__init__()
        self.latent_dim = latent_dim 
        self.net = nn.Sequential(
            nn.Linear(3 + latent_dim, 128), 
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 3 + latent_dim) 
        )
        
    def forward(self, t, state):
        out = self.net(state)
        dp = out[:, 0:3]                     
        dh = out[:, 3:3+self.latent_dim]     
        return torch.cat([dp, dh], dim=-1)

def rk4_step(func, t, z, dt, micro_steps=2):
    dt_micro = dt / micro_steps
    for _ in range(micro_steps):
        k1 = func(t, z)
        k2 = func(t + dt_micro/2, z + dt_micro * k1 / 2)
        k3 = func(t + dt_micro/2, z + dt_micro * k2 / 2)
        k4 = func(t + dt_micro, z + dt_micro * k3)
        z = z + (dt_micro / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt_micro
    return z

class VelocityInjected_S2O_3D(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, use_kinematic_prior=True):
        super(VelocityInjected_S2O_3D, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_kinematic_prior = use_kinematic_prior
        
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=CONFIG['num_layers'], batch_first=True, dropout=0.2)
        
        # Ablation: Explicit Kinematic Prior adjustments
        self.expanded_latent_dim = hidden_dim + 3 if self.use_kinematic_prior else hidden_dim
        self.ode_func = ODEFunc3D(self.expanded_latent_dim)

    def forward(self, x, v_init, dt_batch, predict_steps=20):
        _, hn = self.encoder(x)
        h_0 = hn[-1] 
        p_0 = x[:, -1, :3] # Origin coordinates (0,0,0)
        
        # Ablation: Velocity injection logic
        if self.use_kinematic_prior:
            h_prime_0 = torch.cat([v_init, h_0], dim=-1) 
        else:
            h_prime_0 = h_0

        z_t = torch.cat([p_0, h_prime_0], dim=-1)
        
        current_time = torch.zeros_like(dt_batch)
        outputs = []
        for i in range(predict_steps):
            z_t = rk4_step(self.ode_func, current_time, z_t, dt_batch) 
            outputs.append(z_t[:, :3]) 
            current_time = current_time + dt_batch
            
        return torch.stack(outputs, dim=1)

# ==========================================
# 4. METRICS & ROLLING PREDICTION
# ==========================================
def calculate_tv(truth, pred, threshold):
    dist = np.linalg.norm(truth - pred, axis=1)
    over_idx = np.where(dist > threshold)[0]
    return over_idx[0] if len(over_idx) > 0 else len(dist)

def calculate_metrics(truth, pred, threshold):
    dist = np.linalg.norm(truth - pred, axis=1)
    return {
        'ADE': np.mean(dist),
        'RMSE': np.sqrt(np.mean(dist**2)),
        'FDE': dist[-1],
        'Tv': calculate_tv(truth, pred, threshold),
        'Errors': dist
    }

def predict_full_trajectory(model, file_path):
    if not os.path.exists(file_path): return None, None, None
    try:
        df = pd.read_csv(file_path)
        pos = df[['tx', 'ty', 'tz']].values.astype(np.float32)
    except: return None, None, None

    total_len = len(pos)
    if total_len <= CONFIG['seq_len']: return None, None, None

    truth_future_pos = pos[CONFIG['seq_len']:]
    target_len = len(truth_future_pos)
    
    dt_tensor = torch.FloatTensor([CONFIG['dt_fixed']]).to(CONFIG['device'])
    
    model.eval()
    all_preds_abs = []
    
    current_abs_history = pos[:CONFIG['seq_len']].copy()
    num_blocks = math.ceil(target_len / CONFIG['pred_len'])
    
    with torch.no_grad():
        for _ in range(num_blocks):
            anchor = current_abs_history[-1].copy()
            rel_hist = current_abs_history - anchor
            
            # Ablation: Isotropic Normalization vs Axis-Independent Scaling
            if CONFIG['ablation']['use_isotropic_norm']:
                local_scale = float(np.linalg.norm(rel_hist, axis=1).max())
                local_scale = max(local_scale, 1e-3)
                scale_arr = np.array([local_scale])
            else:
                local_scale = np.abs(rel_hist).max(axis=0)
                local_scale = np.maximum(local_scale, 1e-3)
                scale_arr = local_scale
                
            norm_hist = rel_hist / scale_arr
            
            v_init_val = (norm_hist[-1] - norm_hist[-2]) / CONFIG['dt_fixed']
            
            x_tensor = torch.FloatTensor(norm_hist).unsqueeze(0).to(CONFIG['device'])
            v_init_tensor = torch.FloatTensor(v_init_val).unsqueeze(0).to(CONFIG['device'])
            
            pred_norm = model(x_tensor, v_init=v_init_tensor, dt_batch=dt_tensor, predict_steps=CONFIG['pred_len'])
            pred_norm = pred_norm.squeeze(0).cpu().numpy()
            
            # Revert scaling
            pred_abs = (pred_norm * scale_arr) + anchor
            all_preds_abs.extend(pred_abs)
            
            combined = np.vstack([current_abs_history, pred_abs])
            current_abs_history = combined[-CONFIG['seq_len']:]

    preds_merged = np.array(all_preds_abs)[:target_len]
    return pos[:CONFIG['seq_len']], truth_future_pos, preds_merged

# ==========================================
# 5. TRAINING ENGINE
# ==========================================
def run_training():
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"\n{'='*60}")
    print(f"--- START TRAINING: UAV DATASET ---")
    print(f"Model Name: {CONFIG['model_name']}")
    print(f"{'='*60}")

    train_ds = UAV_Dataset_3D(
        CONFIG['train_list'], CONFIG['data_dir'], 
        seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], 
        stride=CONFIG['stride'], dt=CONFIG['dt_fixed']
    )
    if len(train_ds) == 0:
        print("Data not found. Please check the txt file and directory path.")
        return
        
    val_ds = UAV_Dataset_3D(
        CONFIG['val_list'], CONFIG['data_dir'], 
        seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], 
        stride=CONFIG['stride'], dt=CONFIG['dt_fixed']
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], drop_last=False)

    model = VelocityInjected_S2O_3D(
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'],
        use_kinematic_prior=CONFIG['ablation']['use_kinematic_prior']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # Ablation: Loss Function Setup
    if CONFIG['ablation']['loss_fn'] == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=0.1)
    
    # Ablation: Time-Weighted Robust Optimization
    if CONFIG['ablation']['use_time_weight']:
        steps_tensor = torch.arange(CONFIG['pred_len']).float()
        time_weights = torch.exp(0.1 * steps_tensor).to(device).view(1, -1, 1)
    else:
        time_weights = torch.ones(1, CONFIG['pred_len'], 1).to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_ade = float('inf')
    early_stop_counter = 0
    save_model_path = os.path.join(CONFIG['save_dir'], CONFIG['model_name'])
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            v_init = batch['v_init'].to(device)
            dt = batch['dt'].to(device)
            
            optimizer.zero_grad()
            pred = model(x, v_init=v_init, dt_batch=dt, predict_steps=CONFIG['pred_len'])
            
            loss = criterion(pred * time_weights, y * time_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Benchmark Evaluation (ADE / FDE)
        model.eval()
        val_ade_sum, val_fde_sum, total_samples = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                v_init = batch['v_init'].to(device)
                dt = batch['dt'].to(device)
                scale = batch['scale'].to(device).unsqueeze(1) # Shape: (B, 1, 1) or (B, 1, 3)
                
                pred = model(x, v_init=v_init, dt_batch=dt, predict_steps=CONFIG['pred_len'])
                
                # Calculate error in real-world coordinates (meters)
                dist = torch.norm((pred * scale) - (y * scale), dim=-1)
                
                val_ade_sum += dist.mean(dim=1).sum().item()
                val_fde_sum += dist[:, -1].sum().item()
                total_samples += x.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_ade = val_ade_sum / total_samples 
        avg_val_fde = val_fde_sum / total_samples
        
        scheduler.step(avg_val_ade)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch+1:03d}/{CONFIG['epochs']} | LR: {current_lr:.1e} | Tr.Loss: {avg_train_loss:.5f} | Val ADE: {avg_val_ade:.4f} | Val FDE: {avg_val_fde:.4f}")

        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            print(f'   New record ADE: {best_val_ade:.4f}. Model saved!')
            torch.save(model.state_dict(), save_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= CONFIG['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    print(f"Completed! BEST VAL ADE: {best_val_ade:.4f} m\n")

# ==========================================
# 6. GLOBAL EVALUATION
# ==========================================
def evaluate_global():
    model_path = os.path.join(CONFIG['save_dir'], CONFIG['model_name'])
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    model = VelocityInjected_S2O_3D(
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'],
        use_kinematic_prior=CONFIG['ablation']['use_kinematic_prior']
    ).to(CONFIG['device'])
    
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device'], weights_only=True))
    
    with open(CONFIG['test_list'], 'r') as f:
        files = [l.strip() for l in f.readlines() if l.strip()]
        
    print(f"Starting evaluation on all {len(files)} test files (Rolling/Closed-Loop)...")
    
    metrics_list = {'ADE': [], 'RMSE': [], 'FDE': [], 'Tv': []}
    
    for i, fname in enumerate(files):
        path = os.path.join(CONFIG['data_dir'], fname)
        _, truth, pred = predict_full_trajectory(model, path)
        
        if truth is not None:
            m = calculate_metrics(truth, pred, CONFIG['error_threshold'])
            for key in metrics_list:
                metrics_list[key].append(m[key])
                
        if (i+1) % 50 == 0:
            print(f"   ... Processed {i+1} files")
            
    print("\n" + "="*40)
    print(f" EVALUATION RESULTS: {CONFIG['model_name']}")
    print("="*40)
    print(f"Global ADE  : {np.mean(metrics_list['ADE']):.4f}m +/- {np.std(metrics_list['ADE']):.4f}m")
    print(f"Global RMSE : {np.mean(metrics_list['RMSE']):.4f}m +/- {np.std(metrics_list['RMSE']):.4f}m")
    print(f"Global FDE  : {np.mean(metrics_list['FDE']):.4f}m +/- {np.std(metrics_list['FDE']):.4f}m")
    print(f"Global Tv   : {np.mean(metrics_list['Tv']):.2f} steps +/- {np.std(metrics_list['Tv']):.2f} steps")
    print("="*40)

# ==========================================
# 7. RANDOM PLOTTING EXPORT
# ==========================================
def export_random_plots(num_files=5):
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)
    model_path = os.path.join(CONFIG['save_dir'], CONFIG['model_name'])
    
    if not os.path.exists(model_path):
        print("Model not found for plotting!")
        return

    model = VelocityInjected_S2O_3D(
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'],
        use_kinematic_prior=CONFIG['ablation']['use_kinematic_prior']
    ).to(CONFIG['device'])
    
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device'], weights_only=True))
    
    with open(CONFIG['test_list'], 'r') as f:
        files = [l.strip() for l in f.readlines() if l.strip()]
    if not files: return
    
    selected_files = random.sample(files, min(num_files, len(files)))
    
    print(f"\nExporting {len(selected_files)} random prediction plots to {CONFIG['plot_dir']}...")

    for idx, file_name in enumerate(selected_files):
        path = os.path.join(CONFIG['data_dir'], file_name)
        hist, truth, pred = predict_full_trajectory(model, path)
        if hist is None: continue
        
        m = calculate_metrics(truth, pred, CONFIG['error_threshold'])
        tv = m['Tv']
        
        print(f"[{idx+1}/{len(selected_files)}] Plotting FILE: {file_name} (ADE: {m['ADE']:.4f})")

        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(4, 2, figure=fig)
        
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_err = fig.add_subplot(gs[0, 1])
        ax_x = fig.add_subplot(gs[1, 1])
        ax_y = fig.add_subplot(gs[2, 1])
        ax_z = fig.add_subplot(gs[3, 1])
        
        ax_3d.set_box_aspect((0.5, 0.5, 0.5)) 

        steps = np.arange(len(truth))
        valid_end = min(tv + 1, len(pred))

        ax_3d.plot(hist[:,0], hist[:,1], hist[:,2], 'k:', alpha=0.3, label='History')
        ax_3d.plot(truth[:,0], truth[:,1], truth[:,2], 'k-', linewidth=2, label='Ground Truth')
        
        ax_3d.plot(pred[:valid_end,0], pred[:valid_end,1], pred[:valid_end,2], 'r-', linewidth=2.5, label='Prediction')
        if tv < len(pred)-1:
            ax_3d.plot(pred[tv:,0], pred[tv:,1], pred[tv:,2], 'r--', linewidth=1.5, alpha=0.7, label='Invalid Zone')
            ax_3d.scatter([pred[tv,0]], [pred[tv,1]], [pred[tv,2]], c='red', s=40)
        
        ax_3d.set_title("3D Trajectory")
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        ax_3d.legend()

        error = m['Errors']
        ax_err.plot(steps, error, 'r-', label='Error (m)')
        ax_err.axhline(CONFIG['error_threshold'], color='orange', linestyle='--', label=f"Threshold ({CONFIG['error_threshold']}m)")
        ax_err.axvline(tv, color='blue', linestyle=':', label=f"Tv={tv}")
        ax_err.fill_between(steps, 0, error, where=(error <= CONFIG['error_threshold']), color='green', alpha=0.1)
        ax_err.fill_between(steps, 0, error, where=(error > CONFIG['error_threshold']), color='red', alpha=0.1)
        ax_err.set_title("Displacement Error")
        ax_err.legend()

        components = [(ax_x, 0, 'X'), (ax_y, 1, 'Y'), (ax_z, 2, 'Z')]
        for ax, comp_idx, title in components:
            ax.plot(steps, truth[:, comp_idx], 'k-', linewidth=2, label='Ground Truth')
            ax.plot(steps[:valid_end], pred[:valid_end, comp_idx], 'r-', linewidth=2.5, label='Prediction')
            if tv < len(pred)-1:
                ax.plot(steps[tv:], pred[tv:, comp_idx], 'r--', linewidth=1.5, alpha=0.7)
            
            ax.set_ylabel(f"{title} (m)")
            ax.grid(True, alpha=0.3)
            if comp_idx == 0: ax.legend(loc='upper right')
            if comp_idx == 2: ax.set_xlabel("Time Step (Future)")

        plt.tight_layout()
        save_name = f"eval_plot_{CONFIG['model_name'].replace('.pth', '')}_{file_name.replace('.csv', '.png')}"
        save_path = os.path.join(CONFIG['plot_dir'], save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"All random plots saved successfully to {CONFIG['plot_dir']}.")

# ==========================================
# 8. EXECUTION
# ==========================================
if __name__ == "__main__":
    # run_training()
    evaluate_global()
    export_random_plots(num_files=5)
