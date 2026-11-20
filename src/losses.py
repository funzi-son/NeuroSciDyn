"""Loss functions for KITE, TransKITE, and KC-S2O."""

import torch
import torch.nn as nn


def pos_from_vel(pos0: torch.Tensor, vel: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Integrate velocity over time to reconstruct position from initial position."""
    return pos0 + torch.cumsum(vel * dt, dim=1)


def kinematic_physics_loss(
    pos0: torch.Tensor,
    truth: torch.Tensor,
    pred: torch.Tensor,
    pos_max: float | torch.Tensor,
    vel_max: float | torch.Tensor,
    dt: float = 0.1,
    alpha_vel: float = 0.3,
    alpha_cons: float = 0.1,
    alpha_phys: float = 0.1,
) -> torch.Tensor:
    """Hybrid data + global kinematic consistency + Euler local-update loss (KITE / TransKITE)."""
    criterion = nn.MSELoss()

    pos_loss = criterion(pred[:, :, :3], truth[:, :, :3])
    vel_loss = criterion(pred[:, :, 3:], truth[:, :, 3:]) if pred.size(2) == 6 else 0.0
    total_loss = pos_loss + alpha_vel * vel_loss

    if pred.size(2) != 6:
        return total_loss

    pos_pred_denorm = pred[:, :, :3] * pos_max
    vel_pred_denorm = pred[:, :, 3:] * vel_max

    if alpha_cons > 0:
        pos0_denorm = pos0.view(-1, 1, 3) * pos_max
        pos_from_vel_pred = pos_from_vel(pos0_denorm, vel_pred_denorm, dt)
        total_loss = total_loss + alpha_cons * criterion(pos_from_vel_pred / pos_max, pred[:, :, :3])

    if alpha_phys > 0 and pred.size(1) > 1:
        pos_physics = pos_pred_denorm[:, :-1, :] + vel_pred_denorm[:, :-1, :] * dt
        total_loss = total_loss + alpha_phys * criterion(pred[:, 1:, :3], pos_physics / pos_max)

    return total_loss


def time_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_fn: str = "smooth_l1",
    use_time_weight: bool = True,
) -> torch.Tensor:
    """Time-weighted Smooth L1 or MSE loss used by KC-S2O."""
    criterion = nn.MSELoss() if loss_fn == "mse" else nn.SmoothL1Loss(beta=0.1)
    pred_len = pred.size(1)
    if use_time_weight:
        steps = torch.arange(pred_len, device=pred.device, dtype=pred.dtype)
        weights = torch.exp(0.1 * steps).view(1, -1, 1)
    else:
        weights = torch.ones(1, pred_len, 1, device=pred.device, dtype=pred.dtype)
    return criterion(pred * weights, target * weights)
