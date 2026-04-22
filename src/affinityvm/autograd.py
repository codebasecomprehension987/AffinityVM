"""
Custom autograd bridge: GNN latent -> Rust MD engine -> potential energy scalar.

Forward runs all MD steps in Rust (GIL released) and returns a differentiable
scalar energy. Backward calls energy_grad_latent() on the Rust engine, which
computes dE/d_latent via forward finite differences over the latent dimensions.

The Rust Engine is kept alive by storing it in engine_cache, a plain Python list
held on ctx. This prevents GC from freeing the engine before backward() completes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Function

from affinityvm.engine import AffinityEngine


class MDFunction(Function):
    """
    Custom autograd function: latent -> MD potential energy.

    Usage::
        energy = MDFunction.apply(latent, positions, masses, n_steps, dt, engine_cache)
    """

    @staticmethod
    def forward(
        ctx,
        latent:       torch.Tensor,   # [256] float32, may be on GPU
        positions:    torch.Tensor,   # [N, 3] float32, CPU
        masses:       torch.Tensor,   # [N]    float32, CPU
        n_steps:      int,
        dt:           float,
        engine_cache: list,           # 1-element list; populated here for GC safety
    ) -> torch.Tensor:
        lat_np = latent.detach().cpu().numpy().astype(np.float32, copy=False)
        pos_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        mas_np = masses.detach().cpu().numpy().astype(np.float32, copy=False)

        engine     = AffinityEngine(pos_np, mas_np, lat_np, dt=dt)
        energy_val = engine.step(n_steps)

        # engine_cache[0] keeps the Rust engine alive until backward() finishes
        engine_cache[0]  = engine
        ctx.engine_cache = engine_cache
        ctx.save_for_backward(latent.detach().cpu())

        return torch.tensor(energy_val, dtype=torch.float32, device=latent.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (latent_cpu,) = ctx.saved_tensors
        engine = ctx.engine_cache[0]

        lat_np = latent_cpu.numpy().astype(np.float32, copy=False)
        g_out  = grad_output.item()

        grad_np     = engine.energy_grad_latent(lat_np, g_out)
        grad_latent = torch.from_numpy(grad_np)

        if grad_output.device.type != "cpu":
            grad_latent = grad_latent.to(grad_output.device)

        # One gradient per forward() argument; non-differentiable args return None
        return grad_latent, None, None, None, None, None


# ── Convenience wrapper ───────────────────────────────────────────────────────

def md_energy(
    latent:    torch.Tensor,
    positions: torch.Tensor,
    masses:    torch.Tensor,
    n_steps:   int   = 10_000,
    dt:        float = 0.001,
) -> tuple[torch.Tensor, list]:
    """
    Differentiable MD energy evaluation.

    Returns:
        energy       — scalar tensor, differentiable w.r.t. latent
        engine_cache — 1-element list holding the Rust Engine; must remain
                       in scope until backward() completes
    """
    engine_cache: list = [None]
    energy = MDFunction.apply(latent, positions, masses, n_steps, dt, engine_cache)
    return energy, engine_cache
