"""KC-S2O: Kinematic-Conditioned Sequence-to-ODE 3D Architecture."""

import torch
import torch.nn as nn


class ODEFunc3D(nn.Module):
    """Autonomous neural ODE vector field over [position (3), latent (D)]."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(3 + latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 3 + latent_dim),
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        dp = out[:, 0:3]
        dh = out[:, 3 : 3 + self.latent_dim]
        return torch.cat([dp, dh], dim=-1)


def rk4_step(
    func: nn.Module,
    t: torch.Tensor,
    z: torch.Tensor,
    dt: torch.Tensor,
    micro_steps: int = 2,
) -> torch.Tensor:
    """Fixed-step 4th-order Runge-Kutta integrator."""
    dt_micro = dt / micro_steps
    for _ in range(micro_steps):
        k1 = func(t, z)
        k2 = func(t + dt_micro / 2, z + dt_micro * k1 / 2)
        k3 = func(t + dt_micro / 2, z + dt_micro * k2 / 2)
        k4 = func(t + dt_micro, z + dt_micro * k3)
        z = z + (dt_micro / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt_micro
    return z


class KCS2OModel(nn.Module):
    """Velocity-Injected S2O 3D: GRU encoder + kinematic prior + RK4 Neural ODE."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_kinematic_prior: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_kinematic_prior = use_kinematic_prior

        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.expanded_latent_dim = hidden_dim + 3 if use_kinematic_prior else hidden_dim
        self.ode_func = ODEFunc3D(self.expanded_latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        v_init: torch.Tensor,
        dt_batch: torch.Tensor,
        predict_steps: int = 20,
    ) -> torch.Tensor:
        _, hn = self.encoder(x)
        h_0 = hn[-1]
        p_0 = x[:, -1, :3]

        h_prime_0 = torch.cat([v_init, h_0], dim=-1) if self.use_kinematic_prior else h_0
        z_t = torch.cat([p_0, h_prime_0], dim=-1)

        current_time = torch.zeros_like(dt_batch)
        outputs = []
        for _ in range(predict_steps):
            z_t = rk4_step(self.ode_func, current_time, z_t, dt_batch)
            outputs.append(z_t[:, :3])
            current_time = current_time + dt_batch

        return torch.stack(outputs, dim=1)
