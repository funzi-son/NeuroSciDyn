import torch
import torch.nn as nn

def pos_from_vel(pos0, vel, dt=0.1):
    """Integrates velocity to compute relative position mapping."""
    pos_rel = torch.cumsum(vel * dt, dim=1)   
    return pos0 + pos_rel

def compute_total_loss(pos0, truth, pred, config, pos_max_val, vel_max_val):
    """Computes hybrid data-driven and physics-informed constraints."""
    criterion = nn.MSELoss()
    
    # 1. Data Loss (Normalized Space)
    pos_loss = criterion(pred[:, :, :3], truth[:, :, :3])
    vel_loss = criterion(pred[:, :, 3:], truth[:, :, 3:]) if pred.size(2) == 6 else 0
    total_loss = pos_loss + 0.3 * vel_loss
    
    # Denormalize to apply physical equations accurately
    pos_pred_denorm = pred[:, :, :3] * pos_max_val
    vel_pred_denorm = pred[:, :, 3:] * vel_max_val
    dt = config.get('dt', 0.1)

    # 2. Kinematics Consistency Loss (Global Constraints)
    alpha_cons = config.get('alpha_cons', 0.1)
    if alpha_cons > 0 and pred.size(2) == 6:
        pos0_denorm = pos0.view(-1, 1, 3) * pos_max_val
        pos_from_vel_pred = pos_from_vel(pos0_denorm, vel_pred_denorm, dt)
        pos_from_vel_norm = pos_from_vel_pred / pos_max_val
        total_loss += alpha_cons * criterion(pos_from_vel_norm, pred[:, :, :3])

    # 3. Local-Update Constraint Loss (Euler Integration)
    alpha_phys = config.get('alpha_phys', 0.1)
    if alpha_phys > 0 and pred.size(2) == 6:
        v_current = vel_pred_denorm[:, :-1, :]
        pos_physics = pos_pred_denorm[:, :-1, :] + v_current * dt
        
        phys_loss = criterion(pred[:, 1:, :3], pos_physics / pos_max_val)
        total_loss += alpha_phys * phys_loss

    return total_loss