import torch
import numpy as np
import math
from scipy.signal import savgol_filter
from .dataset import compute_velocity

def autoregressive_rollout(model, raw_coords, config, pos_max, vel_max, device):
    """Autoregressive unrolling for UAV datasets (Global Scaling)."""
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    dt = config.get('dt', 0.1)
    use_sg = config.get('use_sg_filter', True)
    
    if len(raw_coords) < seq_len + 10:
        return None, None, None

    input_history = raw_coords[:seq_len]
    limit_len = len(raw_coords) - seq_len
    truth_future = raw_coords[seq_len : seq_len + limit_len]

    vel = compute_velocity(input_history, dt, use_sg)
    n_pos = input_history / pos_max
    n_vel = vel / vel_max
    state_scaled = np.concatenate([n_pos, n_vel], axis=1)
    
    current_seq_tensor = torch.FloatTensor(state_scaled).unsqueeze(0).to(device)
    predictions_scaled = []
    num_blocks = int(np.ceil(limit_len / pred_len))
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_blocks):
            window = current_seq_tensor[:, -seq_len:, :].clone()
            if not config.get('use_kinematics', True):
                window = window[:, :, :3]
                
            pred_block = model(window)
            
            block_np = pred_block.detach().cpu().numpy()[0]
            predictions_scaled.extend(block_np.tolist())
            
            remaining = current_seq_tensor[:, pred_len:, :]
            current_seq_tensor = torch.cat((remaining, pred_block), dim=1)

    predictions_scaled = predictions_scaled[:limit_len]
    pred_coords = np.array(predictions_scaled)[:, :3] * pos_max 

    return input_history, truth_future, pred_coords


def autoregressive_rollout_uuv(model, raw_coords, seq_len, pred_len, dt, device, use_sg=True):
    """Autoregressive unrolling for UUV datasets with dynamic local scaling."""
    if len(raw_coords) <= seq_len:
        return None, None, None

    history = raw_coords[:seq_len].copy()
    truth_future = raw_coords[seq_len:]
    target_len = len(truth_future)
    
    current_history = history.copy()
    all_preds_abs = []
    
    num_blocks = math.ceil(target_len / pred_len)
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_blocks):
            recent_hist = current_history[-seq_len:]
            
            anchor = recent_hist[-1].copy()
            p_rel = recent_hist - anchor
            local_scale = np.abs(p_rel).max(axis=0)
            scale = np.maximum(local_scale, 1e-3) 
            p_norm = p_rel / scale
            
            if use_sg and len(p_norm) >= 11:
                v_norm = savgol_filter(p_norm, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
            else:
                v_norm = np.gradient(p_norm, dt, axis=0)
            
            x_kinematic = np.concatenate([p_norm, v_norm], axis=-1)
            x_tensor = torch.FloatTensor(x_kinematic).unsqueeze(0).to(device) 
            
            out = model(x_tensor)
            pred_norm_pos = out[0] if isinstance(out, tuple) else out
            pred_norm_pos = pred_norm_pos.squeeze(0).cpu().numpy() 
            
            pred_abs = (pred_norm_pos * scale) + anchor
            all_preds_abs.extend(pred_abs)
            current_history = np.vstack([current_history, pred_abs])[-seq_len:]

    pred_final = np.array(all_preds_abs)[:target_len]
    return history, truth_future, pred_final