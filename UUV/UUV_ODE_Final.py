import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')
#Download dataset: https://github.com/zhouhao94/3DMoTraj

# ==========================================
# 1. CONFIGURATION & ABLATION STUDY
# ==========================================
CONFIG = {
    'data_dir': '3DMoTraj',
    'scenes': [f'3Dscene{i}' for i in range(1, 9)],
    'save_dir': 'results_uuv_ode_final',
    'plot_dir': 'results_uuv_ode_final/plots',
    
    # ----------------------------------------------------
    # ABLATION STUDY TOGGLES
    # ----------------------------------------------------
    'ablation': {
        # 1. Explicit Kinematic Prior: Set False to remove instantaneous velocity injection (v_init)
        'use_kinematic_prior': True, 
        
        # 2. Time-Weighted Robust Optimization: Set False to remove exponential temporal penalty
        'use_time_weight': True,     
        # Change to 'mse' to utilize standard Mean Squared Error (MSE Only)
        'loss_fn': 'mse',      
        
        # 3. Isotropic Normalization: Set False to use axis-independent Min-Max scaling
        'use_isotropic_norm': True   
    },
    
    'seq_len': 8,       
    'pred_len': 12,      
    'stride': 2,
    'original_fps': 10.0,
    
    'input_dim': 3,      
    'hidden_dim': 64,  
    'num_layers': 3,
    
    'batch_size': 64,
    'epochs': 300,
    'lr': 1e-3,  
    'weight_decay': 1e-4, 
    'patience': 30, 
    'seed': 42,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def get_model_name(scene):
    suffix = "base"
    if not CONFIG['ablation']['use_kinematic_prior']: suffix += "_no_kinematic"
    if not CONFIG['ablation']['use_time_weight']: suffix += "_no_timeweight"
    if CONFIG['ablation']['loss_fn'] == 'mse': suffix += "_mse"
    if not CONFIG['ablation']['use_isotropic_norm']: suffix += "_no_isotropic"
    return f"best_s2o_{scene}_{suffix}.pth"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 2. DATA PROCESSING
# ==========================================
def load_motraj_data(data_dir, scenes, split, seq_len, pred_len, stride):
    samples = []
    for scene in scenes:
        folder_path = os.path.join(data_dir, scene, split)
        if not os.path.exists(folder_path): continue
            
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        for f in txt_files:
            try:
                df = pd.read_csv(f, sep=r'\s+', header=None, names=['frame_id', 'track_id', 'x', 'y', 'z'])
                
                for track_id, group in df.groupby('track_id'):
                    group = group.sort_values('frame_id')
                    
                    pos = group[['x', 'y', 'z']].values.astype(np.float32)
                    frame_ids = group['frame_id'].values.astype(np.float32)
                    
                    total_len = len(pos)
                    if total_len < seq_len + pred_len: continue
                    
                    for i in range(0, total_len - seq_len - pred_len + 1, stride):
                        p_window = pos[i : i + seq_len + pred_len]
                        f_window = frame_ids[i : i + seq_len + pred_len]
                        samples.append((p_window, f_window, scene))
            except Exception as e: pass
    return samples

class MoTrajDataset3D_Benchmark(Dataset):
    def __init__(self, data_dir, scenes, split, seq_len=8, pred_len=12, stride=2, fps=10.0):
        actual_stride = stride if split == 'train' else seq_len + pred_len
        raw_samples = load_motraj_data(data_dir, scenes, split, seq_len, pred_len, actual_stride)
        self.samples = []
        
        print(f"[{split.upper()}] Processing {len(raw_samples)} samples (8-in/12-out)...")
        
        for p_window, f_window, scene in raw_samples:
            anchor = p_window[seq_len - 1].copy()
            p_rel = p_window - anchor
            
            # Ablation: Isotropic vs Axis-Independent Normalization
            if CONFIG['ablation']['use_isotropic_norm']:
                local_scale = float(np.linalg.norm(p_rel, axis=1).max())
                local_scale = max(local_scale, 1.0) 
                scale_arr = np.array([local_scale])
            else:
                local_scale = np.abs(p_rel).max(axis=0)
                local_scale = np.maximum(local_scale, 1.0)
                scale_arr = local_scale

            p_norm = p_rel / scale_arr
            
            avg_frame_step = np.mean(np.diff(f_window))
            dt = avg_frame_step / fps
            
            v_init = (p_norm[seq_len - 1] - p_norm[seq_len - 2]) / dt
            
            x = torch.FloatTensor(p_norm[:seq_len])
            y = torch.FloatTensor(p_norm[seq_len:])
            
            self.samples.append({
                'x': x, 
                'y': y, 
                'v_init': torch.FloatTensor(v_init),
                'dt': torch.FloatTensor([dt]),
                'scale': torch.FloatTensor(scale_arr),
                'anchor': torch.FloatTensor(anchor),
                'scene': scene
            })

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
    def __init__(self, input_dim=3, hidden_dim=128, use_kinematic_prior=True):
        super(VelocityInjected_S2O_3D, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_kinematic_prior = use_kinematic_prior
        
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Ablation: Explicit Kinematic Prior
        self.expanded_latent_dim = hidden_dim + 3 if self.use_kinematic_prior else hidden_dim
        self.ode_func = ODEFunc3D(self.expanded_latent_dim)

    def forward(self, x, v_init, dt_batch, predict_steps=12):
        _, hn = self.encoder(x)
        h_0 = hn[-1] 
        p_0 = x[:, -1, :3] 
        
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
# 4. TRAINING ENGINE
# ==========================================
def run_training_for_scene(target_scene):
    set_seed(CONFIG['seed'])
    device = CONFIG['device']
    print(f"\n{'='*60}")
    print(f"--- START TRAINING: {target_scene.upper()} ---")
    print(f"{'='*60}")

    train_ds = MoTrajDataset3D_Benchmark(
        CONFIG['data_dir'], [target_scene], split='train', 
        seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], 
        stride=CONFIG['stride'], fps=CONFIG['original_fps']
    )
    if len(train_ds) == 0: 
        print(f"Train data not found for {target_scene}. Skipping.")
        return
        
    val_ds = MoTrajDataset3D_Benchmark(
        CONFIG['data_dir'], [target_scene], split='val', 
        seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], 
        stride=CONFIG['stride'], fps=CONFIG['original_fps']
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], drop_last=False)

    model = VelocityInjected_S2O_3D(
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'],
        use_kinematic_prior=CONFIG['ablation']['use_kinematic_prior']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # Ablation: Loss Function
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
    save_model_path = os.path.join(CONFIG['save_dir'], get_model_name(target_scene))
    
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
        
        model.eval()
        val_ade_sum, val_fde_sum, total_samples = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                v_init = batch['v_init'].to(device)
                dt = batch['dt'].to(device)
                scale = batch['scale'].to(device).unsqueeze(1) 
                
                pred = model(x, v_init=v_init, dt_batch=dt, predict_steps=CONFIG['pred_len'])
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
            
    print(f"Completed {target_scene}! BEST VAL ADE: {best_val_ade:.4f} m\n")

# ==========================================
# 5. OPEN-LOOP EVALUATION
# ==========================================
def run_evaluation():
    device = CONFIG['device']
    print(f"\n{'='*60}")
    print(f"LAUNCHING OPEN-LOOP EVALUATION (8-in / 12-out) ON {device}")
    print(f"{'='*60}")

    global_ade_sum = 0
    global_fde_sum = 0
    global_samples = 0
    
    scene_results = {}

    for scene in CONFIG['scenes']:
        model_path = os.path.join(CONFIG['save_dir'], get_model_name(scene))
        
        if not os.path.exists(model_path):
            print(f"Skipping {scene}: Weights not found at {model_path}")
            continue

        test_ds = MoTrajDataset3D_Benchmark(
            CONFIG['data_dir'], [scene], split='test', 
            seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], 
            stride=CONFIG['stride'], fps=CONFIG['original_fps']
        )
        
        if len(test_ds) == 0:
            print(f"Skipping {scene}: No valid test data.")
            continue
            
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

        model = VelocityInjected_S2O_3D(
            input_dim=CONFIG['input_dim'], 
            hidden_dim=CONFIG['hidden_dim'],
            use_kinematic_prior=CONFIG['ablation']['use_kinematic_prior']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        scene_ade_sum = 0
        scene_fde_sum = 0
        scene_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                v_init = batch['v_init'].to(device)
                dt = batch['dt'].to(device)
                scale = batch['scale'].to(device).unsqueeze(1) 
                
                pred = model(x, v_init=v_init, dt_batch=dt, predict_steps=CONFIG['pred_len'])
                
                pred_real = pred * scale
                y_real = y * scale
                
                dist = torch.norm(pred_real - y_real, dim=-1)
                
                scene_ade_sum += dist.mean(dim=1).sum().item()
                scene_fde_sum += dist[:, -1].sum().item()
                scene_samples += x.size(0)
                
        avg_scene_ade = scene_ade_sum / scene_samples
        avg_scene_fde = scene_fde_sum / scene_samples
        
        print(f"  {scene.upper()} | Samples: {scene_samples:4d} | ADE: {avg_scene_ade:.4f} m | FDE: {avg_scene_fde:.4f} m")
        
        scene_results[scene] = {'ADE': avg_scene_ade, 'FDE': avg_scene_fde}
        
        global_samples += scene_samples
        global_ade_sum += avg_scene_ade
        global_fde_sum += avg_scene_fde

    if global_samples > 0:
        final_global_ade = global_ade_sum / len(scene_results)
        final_global_fde = global_fde_sum / len(scene_results)
        
        print(f"\n{'='*60}")
        print(f" GLOBAL BENCHMARK RESULTS (Average of Test Data)")
        print(f"{'='*60}")
        print(f"Total evaluated samples : {global_samples}")
        print(f"Global Mean ADE         : {final_global_ade:.4f} meters")
        print(f"Global Mean FDE         : {final_global_fde:.4f} meters")
        print(f"{'='*60}")
        
        print("\n[Copy-paste for LaTeX Table]")
        latex_str = "Ours & "
        for scene in CONFIG['scenes']:
            if scene in scene_results:
                latex_str += f"{scene_results[scene]['ADE']:.2f}/{scene_results[scene]['FDE']:.2f} & "
            else:
                latex_str += "-/- & "
        latex_str += f"\\textbf{{{final_global_ade:.2f}/{final_global_fde:.2f}}} \\\\"
        print(latex_str)

# ==========================================
# 6. RANDOM PLOTTING EXPORT
# ==========================================
def export_random_plots(num_plots=5):
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)
    device = CONFIG['device']
    
    # Load all test data across all scenes
    full_test_ds = MoTrajDataset3D_Benchmark(
        CONFIG['data_dir'], CONFIG['scenes'], split='test', 
        seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'], 
        stride=CONFIG['stride'], fps=CONFIG['original_fps']
    )
    
    if len(full_test_ds) == 0:
        print("No test data found for plotting.")
        return

    # Cache loaded models to avoid reloading
    models_cache = {}
    
    indices = random.sample(range(len(full_test_ds)), min(num_plots, len(full_test_ds)))
    print(f"\nExporting {len(indices)} random prediction plots to {CONFIG['plot_dir']}...")

    for i, idx in enumerate(indices):
        sample = full_test_ds[idx]
        scene = sample['scene']
        
        if scene not in models_cache:
            model_path = os.path.join(CONFIG['save_dir'], get_model_name(scene))
            if not os.path.exists(model_path):
                continue
            
            model = VelocityInjected_S2O_3D(
                input_dim=CONFIG['input_dim'], 
                hidden_dim=CONFIG['hidden_dim'],
                use_kinematic_prior=CONFIG['ablation']['use_kinematic_prior']
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            models_cache[scene] = model
            
        model = models_cache[scene]
        
        x = sample['x'].unsqueeze(0).to(device)
        v_init = sample['v_init'].unsqueeze(0).to(device)
        dt = sample['dt'].unsqueeze(0).to(device)
        scale = sample['scale'].numpy()
        anchor = sample['anchor'].numpy()
        y_true_norm = sample['y'].numpy()
        x_norm = sample['x'].cpu().numpy()

        with torch.no_grad():
            pred_norm = model(x, v_init=v_init, dt_batch=dt, predict_steps=CONFIG['pred_len'])
            pred_norm = pred_norm.squeeze(0).cpu().numpy()

        # Reconstruct absolute coordinates
        hist_abs = (x_norm * scale) + anchor
        truth_abs = (y_true_norm * scale) + anchor
        pred_abs = (pred_norm * scale) + anchor

        # Calculate Distances
        dist = np.linalg.norm(pred_abs - truth_abs, axis=1)
        ade = np.mean(dist)

        # Plotting
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(4, 2, figure=fig)
        
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        ax_err = fig.add_subplot(gs[0, 1])
        ax_x = fig.add_subplot(gs[1, :])
        ax_y = fig.add_subplot(gs[2, :])
        ax_z = fig.add_subplot(gs[3, :])
        
        ax_3d.set_box_aspect((0.5, 0.5, 0.5))

        hist_steps = np.arange(CONFIG['seq_len'])
        pred_steps = np.arange(CONFIG['seq_len'], CONFIG['seq_len'] + CONFIG['pred_len'])

        # 1. 3D Trajectory
        ax_3d.plot(hist_abs[:,0], hist_abs[:,1], hist_abs[:,2], 'k:', alpha=0.5, label='History')
        ax_3d.plot(truth_abs[:,0], truth_abs[:,1], truth_abs[:,2], 'k-', linewidth=2, label='Ground Truth')
        ax_3d.plot(pred_abs[:,0], pred_abs[:,1], pred_abs[:,2], 'r-', linewidth=2.5, label='Prediction')
        ax_3d.scatter(anchor[0], anchor[1], anchor[2], c='blue', s=50, label='Anchor', zorder=5)

        ax_3d.set_title(f"3D Trajectory - {scene} (Sample {idx})")
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        ax_3d.legend()

        # 2. Error Plot
        ax_err.plot(pred_steps, dist, 'r-', label='Error (m)')
        ax_err.set_title(f"Displacement Error (ADE: {ade:.4f}m)")
        ax_err.set_xlabel("Future Steps")
        ax_err.legend()

        # 3. XYZ Components
        components = [(ax_x, 0, 'X'), (ax_y, 1, 'Y'), (ax_z, 2, 'Z')]
        for ax, comp_idx, title in components:
            ax.plot(hist_steps, hist_abs[:, comp_idx], 'k:', alpha=0.5, label='History')
            ax.plot(pred_steps, truth_abs[:, comp_idx], 'k-', linewidth=2, label='Ground Truth')
            ax.plot(pred_steps, pred_abs[:, comp_idx], 'r-', linewidth=2.5, label='Prediction')
            
            ax.set_ylabel(f"{title} (m)")
            ax.grid(True, alpha=0.3)
            if comp_idx == 0: ax.legend(loc='upper right')
            if comp_idx == 2: ax.set_xlabel("Time Step")

        plt.tight_layout()
        save_name = f"eval_plot_{scene}_sample_{idx}.png"
        save_path = os.path.join(CONFIG['plot_dir'], save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"All random plots saved successfully to {CONFIG['plot_dir']}.")

# ==========================================
# 7. EXECUTION
# ==========================================
if __name__ == "__main__":
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # # 1. Run training for all scenes
    # for scene in CONFIG['scenes']:
    #     run_training_for_scene(scene)
        
    # 2. Run the global evaluation on the test set
    run_evaluation()
    
    # 3. Export 5 random prediction plots
    # export_random_plots(num_plots=5)
