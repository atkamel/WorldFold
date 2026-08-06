"""Conditional DDPM over cloth angle fields -- the Diffusion Perception Model
(DPM) of EXPERIMENTS.md sections 1-2, scaled to the 11x11 angle observation.

Given a partial observation (masked field + binary mask), the model denoises
a full field conditioned on it. Sampling K times yields K occlusion-consistent
hypotheses; their spread is a per-cell uncertainty map (Experiment 2's gate
signal).

Design notes:
- Fields are normalized to [-1, 1] (angle / pi) for diffusion; samples are
  clamped and rescaled back to radians.
- epsilon-prediction MLP (the field is only 121 scalars; a UNet is overkill),
  sinusoidal timestep embedding, cosine beta schedule, T=100.
- Inpainting consistency: at every reverse step the VISIBLE cells of x_t are
  replaced with the forward-noised observation (RePaint-style), and the final
  sample copies the observed values verbatim -- only hidden cells are ever
  generated.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from cloth_angles.model.world_model import _wrapped_residual


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AngleFieldDPM(nn.Module):
    """Conditional denoiser: predicts the noise in x_t given (masked field, mask, t)."""

    def __init__(self, grid_size: int, timesteps: int = 100, hidden_dim: int = 512,
                 time_embed_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.n2 = grid_size * grid_size
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.n2 * 3 + time_embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.n2),
        )

    # ---- normalization ----
    @staticmethod
    def to_norm(field_rad: torch.Tensor) -> torch.Tensor:
        return field_rad / math.pi

    @staticmethod
    def from_norm(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(-1.0, 1.0) * math.pi

    # ---- core ----
    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor,
                     cond_field: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        """All field tensors flattened [batch, N*N]; t: [batch] int64."""
        temb = self.time_embed(t)
        inp = torch.cat([x_t, cond_field, cond_mask, temb], dim=-1)
        return self.net(inp)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cumprod[t][:, None]
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def loss(self, field_rad: torch.Tensor, masked_field_rad: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        """field_rad/masked_field_rad: [batch, N*N] radians; mask: [batch, N*N]
        float, 1 = hidden. Standard epsilon-prediction MSE."""
        x0 = self.to_norm(field_rad)
        cond = self.to_norm(masked_field_rad)
        batch = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.predict_eps(x_t, t, cond, mask)
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample_k(self, masked_field_rad: torch.Tensor, mask: torch.Tensor,
                  k: int = 8) -> torch.Tensor:
        """masked_field_rad: [N*N] or [1, N*N] radians; mask: same shape, 1 = hidden.
        Returns [k, N*N] radians: k occlusion-consistent full-field hypotheses.
        Visible cells are constrained to the observation at every reverse step
        and copied verbatim into the final samples."""
        cond = self.to_norm(masked_field_rad.reshape(1, self.n2)).repeat(k, 1)
        m = mask.reshape(1, self.n2).float().repeat(k, 1)
        visible = 1.0 - m
        x0_obs = cond  # hidden cells are 0 there; only visible entries are used

        x = torch.randn(k, self.n2, device=cond.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((k,), step, dtype=torch.long, device=cond.device)
            # RePaint-style: keep the visible region on the observed trajectory
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = m * x + visible * self.q_sample(x0_obs, t, noise)

            eps_hat = self.predict_eps(x, t, cond, m)
            alpha = self.alphas[step]
            ac = self.alphas_cumprod[step]
            mean = (x - (1.0 - alpha) / (1.0 - ac).sqrt() * eps_hat) / alpha.sqrt()
            if step > 0:
                x = mean + self.betas[step].sqrt() * torch.randn_like(x)
            else:
                x = mean

        fields = self.from_norm(x)
        observed = masked_field_rad.reshape(1, self.n2).expand(k, -1)
        return m * fields + visible * observed   # visible cells exactly as observed

    @torch.no_grad()
    def variance_map(self, masked_field_rad: torch.Tensor, mask: torch.Tensor,
                      k: int = 8) -> torch.Tensor:
        """Per-cell variance across k hypotheses, [N, N]. High where the DPM is
        uncertain about the hidden geometry -- Experiment 2's gate signal."""
        samples = self.sample_k(masked_field_rad, mask, k)
        return samples.var(dim=0).reshape(self.grid_size, self.grid_size)


def wrapped_abs_error(a_rad: torch.Tensor, b_rad: torch.Tensor) -> torch.Tensor:
    """|shortest-path angular difference|, elementwise."""
    return _wrapped_residual(a_rad, b_rad).abs()


# ---- checkpointing (mirrors the repo's schema-validated pattern) ----

def save_dpm_checkpoint(path, model: AngleFieldDPM, step: int, grid_size: int,
                         angle_unit: str, angle_convention: str, model_config: dict) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "grid_size": grid_size,
        "angle_unit": angle_unit,
        "angle_convention": angle_convention,
        "model_config": model_config,
    }, path)


def load_dpm_checkpoint(path, expected_grid_size: int, device=None) -> tuple[AngleFieldDPM, dict]:
    checkpoint = torch.load(path, map_location=device or "cpu")
    if checkpoint.get("grid_size") != expected_grid_size:
        raise ValueError(f"DPM checkpoint grid_size {checkpoint.get('grid_size')} != expected {expected_grid_size}")
    cfg = checkpoint["model_config"]
    model = AngleFieldDPM(
        grid_size=checkpoint["grid_size"],
        timesteps=cfg["timesteps"],
        hidden_dim=cfg["hidden_dim"],
        time_embed_dim=cfg["time_embed_dim"],
    )
    if device is not None:
        model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
