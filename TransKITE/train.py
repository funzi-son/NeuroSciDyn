import argparse
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

from configs.uav_config import UAV_CONFIG, get_uav_config
from configs.uuv_config import UUV_CONFIG, get_uuv_config

# from configs.default_config import DEFAULT_CONFIG
# from configs.uuv_config import get_uuv_config
from core.dataset import build_pos_vel_lists, read_path_list, TrajectoryDataset, TransKITE_UUVDataset
from core.losses import compute_total_loss
from model.trans_kite import TransKITEModel

import logging

def setup_logger(save_dir, filename):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def train_uav(config, logger, device):
    train_files = read_path_list(config['data_dir'], config['train_list'])
    val_files = read_path_list(config['data_dir'], config['val_list'])

    train_samples, pos_max, vel_max = build_pos_vel_lists(train_files, config, training=True)
    val_samples, _, _ = build_pos_vel_lists(val_files, config, training=False, pos_max=pos_max, vel_max=vel_max)

    train_loader = DataLoader(TrajectoryDataset(train_samples), batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TrajectoryDataset(val_samples), batch_size=config['batch_size'], drop_last=False)

    with open(os.path.join(config['save_dir'], 'norm_stats.json'), 'w') as f:
        json.dump({"pos_max": float(pos_max), "vel_max": float(vel_max)}, f)

    input_dim = config['input_dim'] * 2 if config.get('use_kinematics', True) else config['input_dim']
    model = TransKITEModel(
        input_dim=input_dim, hidden_dim=config['hidden_dim'], 
        seq_len=config['seq_len'], pred_len=config['pred_len']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            noise_level = config.get('input_noise', 0.0)
            bx_noisy = bx + torch.randn_like(bx) * noise_level if noise_level > 0 else bx

            optimizer.zero_grad()
            bx_input = bx_noisy if config.get('use_kinematics', True) else bx_noisy[:, :, :3]
            pred = model(bx_input)
            
            loss = compute_total_loss(bx_noisy[:, -1, :3], by, pred, config, pos_max, vel_max)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss, val_rmse_m = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vx_input = vx if config.get('use_kinematics', True) else vx[:, :, :3]
                pred = model(vx_input)
                
                loss = compute_total_loss(vx[:, -1, :3], vy, pred, config, pos_max, vel_max)
                val_loss += loss.item()
                rmse_m = torch.sqrt(torch.mean(((pred[:, :, :3] - vy[:, :, :3]) * pos_max)**2))
                val_rmse_m += rmse_m.item()
        
        avg_val_rmse = val_rmse_m / len(val_loader)
        logger.info(f"Epoch {epoch+1:03d} | Tr.Loss: {train_loss/len(train_loader):.4f} | Val RMSE: {avg_val_rmse:.3f}m")
        scheduler.step()

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), os.path.join(config['save_dir'], "best_transkite.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info("Early Stopping triggered.")
                break

def train_uuv(config, logger, device):
    scenes = config['scenes']
    alpha_vel = config.get('alpha', 0.1)
    beta_con = config.get('beta', 0.1)
    gamma_loc = config.get('gamma', 0.1)
    
    for scene in scenes:
        logger.info(f"\n--- Training UUV Scene: {scene.upper()} ---")
        train_ds = TransKITE_UUVDataset(config['data_dir'], scenes=[scene], split='train', 
                                        seq_len=config['seq_len'], pred_len=config['pred_len'], 
                                        stride=config['stride'], fps=config['fps'], use_sg=config.get('use_sg_filter', True))
        val_ds = TransKITE_UUVDataset(config['data_dir'], scenes=[scene], split='val', 
                                      seq_len=config['seq_len'], pred_len=config['pred_len'], 
                                      stride=config['stride'], fps=config['fps'], use_sg=config.get('use_sg_filter', True))
        
        if len(train_ds) == 0: 
            logger.info(f"No data for {scene}, skipping.")
            continue
            
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], drop_last=False)
        
        input_dim = config['input_dim'] * 2 if config.get('use_kinematics', True) else config['input_dim']
        model = TransKITEModel(
            input_dim=input_dim, hidden_dim=config['hidden_dim'], 
            seq_len=config['seq_len'], pred_len=config['pred_len']
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience'], min_lr=config['min_lr'])
        
        steps_tensor = torch.arange(config['pred_len']).float()
        time_weights = torch.exp(0.1 * steps_tensor).to(device).view(1, -1, 1)
        
        best_val_ade = float('inf')
        early_stop_counter = 0
        scene_weight_path = os.path.join(config['save_dir'], f"best_transkite_{scene}.pth")
        
        for epoch in range(config['epochs']):
            model.train()
            train_loss_total = 0
            for batch in train_loader:
                x = batch['x'].to(device)
                y_pos, y_vel, dt = batch['y_pos'].to(device), batch['y_vel'].to(device), batch['dt'].to(device).unsqueeze(-1)
                
                optimizer.zero_grad()
                out = model(x)
                if isinstance(out, tuple):
                    pred_pos, pred_vel = out
                elif out.shape[-1] == 6:
                    pred_pos, pred_vel = out[..., :3], out[..., 3:]
                else:
                    pred_pos = out
                    pred_vel = torch.zeros_like(pred_pos)
                
                loss_pos = (F.mse_loss(pred_pos, y_pos, reduction='none') * time_weights).mean()
                loss_vel = (F.mse_loss(pred_vel, y_vel, reduction='none') * time_weights).mean()
                
                p_anchor = torch.zeros_like(pred_pos[:, 0:1, :])
                p_euler = torch.cat([p_anchor, pred_pos[:, :-1, :]], dim=1) + pred_vel * dt
                loss_loc = F.mse_loss(pred_pos, p_euler)
                
                p_global = torch.cumsum(pred_vel * dt, dim=1)
                loss_con = F.mse_loss(pred_pos, p_global)
                
                loss = loss_pos + alpha_vel * loss_vel + gamma_loc * loss_loc + beta_con * loss_con
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_total += loss.item()
                
            model.eval()
            val_ade_sum, total_samples = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y_pos, scale = batch['x'].to(device), batch['y_pos'].to(device), batch['scale'].to(device).unsqueeze(1)
                    out = model(x)
                    pred_pos = out[0] if isinstance(out, tuple) else (out[..., :3] if out.shape[-1] == 6 else out)
                    dist = torch.norm((pred_pos * scale) - (y_pos * scale), dim=-1)
                    val_ade_sum += dist.mean(dim=1).sum().item()
                    total_samples += x.size(0)
                    
            avg_train_loss = train_loss_total / len(train_loader)
            avg_val_ade = val_ade_sum / total_samples if total_samples > 0 else float('inf')
            
            scheduler.step(avg_val_ade)
            
            if avg_val_ade < best_val_ade:
                best_val_ade = avg_val_ade
                torch.save(model.state_dict(), scene_weight_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"[{scene}] Ep {epoch+1:03d} | LR: {optimizer.param_groups[0]['lr']:.1e} | Tr.Loss: {avg_train_loss:.5f} | Val ADE: {avg_val_ade:.4f}m")
                
            if early_stop_counter >= config['patience']: 
                logger.info(f"Early stopping at Epoch {epoch+1}")
                break

def main():
    parser = argparse.ArgumentParser(description="Unified TransKITE Training")
    parser.add_argument('--data', type=str, choices=['UAV', 'UUV'], default='UAV', help="Select dataset domain.")
    args = parser.parse_args()

    # config = DEFAULT_CONFIG.copy() if args.data == 'UAV' else get_uuv_config()
    config = get_uav_config() if args.data == 'UAV' else get_uuv_config()
    config['save_dir'] = f"experiments/{args.data}_TransKITE_Full"

    os.makedirs(config['save_dir'], exist_ok=True)
    logger = setup_logger(config['save_dir'], "train_log.txt")
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    logger.info("="*60)
    logger.info(f"INITIALIZING TRAINING: {args.data} Domain")
    logger.info("="*60)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == 'UAV':
        train_uav(config, logger, device)
    elif args.data == 'UUV':
        train_uuv(config, logger, device)

if __name__ == "__main__":
    main()