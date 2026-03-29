# src/affinityvm/autograd.py
"""
MDFunction — torch.autograd.Function that bridges Python↔Rust for backprop.

Forward:
  1. Take GNN latent (256-dim, on GPU).
  2. Copy to CPU numpy (zero-copy when already on CPU).
  3. Call Engine.step(n_steps) — 10,000 MD steps entirely in Rust, GIL released.
  4. Return total potential energy as a scalar tensor.

Backward:
  ∂loss/∂latent = (∂loss/∂E) · (∂E/∂latent)
  ∂E/∂latent is computed by Engine.energy_grad_latent(), which internally
  calls Enzyme-autodiff'd Rust code (or finite-diff fallback).

Memory safety contract:
  The Rust Engine object is stored in ctx.saved_state (a plain Python object
  reference).  Python's GC cannot collect it while ctx lives on the autograd
  graph — which persists until the backward pass completes.  The PyO3
  PhantomData<'py> in the Rust binding enforces this at compile time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Function

from affinityvm.engine import AffinityEngine


class MDFunction(Function):
    """
    Custom autograd function: latent → MD potential energy.

    Usage::
        energy = MDFunction.apply(latent, positions, masses, n_steps, engine_cache)
    """

    @staticmethod
    def forward(
        ctx,
        latent:       torch.Tensor,          # [256] float32, may be on GPU
        positions:    torch.Tensor,          # [N, 3] float32, CPU
        masses:       torch.Tensor,          # [N]    float32, CPU
        n_steps:      int,
        dt:           float,
        engine_cache: list,                  # mutable 1-element list; populated here
    ) -> torch.Tensor:
        # ── Transfer to CPU numpy (zero-copy if already CPU float32) ──────────
        lat_np  = latent.detach().cpu().numpy().astype(np.float32, copy=False)
        pos_np  = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        mas_np  = masses.detach().cpu().numpy().astype(np.float32, copy=False)

        # ── Build Rust engine ─────────────────────────────────────────────────
        engine = AffinityEngine(pos_np, mas_np, lat_np, dt=dt)

        # ── Run MD — blocking Rust call, GIL released internally ─────────────
        energy_val = engine.step(n_steps)

        # ── Store for backward ────────────────────────────────────────────────
        # engine_cache[0] = engine prevents GC while backward graph is alive
        engine_cache[0] = engine
        ctx.engine_cache = engine_cache       # keep reference in ctx
        ctx.save_for_backward(latent.detach().cpu())
        ctx.n_atoms = pos_np.shape[0]

        return torch.tensor(energy_val, dtype=torch.float32, device=latent.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        engine_cache = ctx.engine_cache
        engine = engine_cache[0]

        (latent_cpu,) = ctx.saved_tensors
        lat_np = latent_cpu.numpy().astype(np.float32, copy=False)
        g_out  = grad_output.item()

        # ── Compute ∂E/∂latent via Rust VJP ──────────────────────────────────
        grad_np = engine.energy_grad_latent(lat_np, g_out)  # numpy [256]
        grad_latent = torch.from_numpy(grad_np)

        # Move gradient to original device
        if grad_output.device.type != "cpu":
            grad_latent = grad_latent.to(grad_output.device)

        # Gradients for: latent, positions, masses, n_steps, dt, engine_cache
        return grad_latent, None, None, None, None, None


# ─── Convenience wrapper ─────────────────────────────────────────────────────

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
        energy (scalar tensor, differentiable w.r.t. latent)
        engine_cache (1-element list holding the Rust Engine — keep alive!)
    """
    engine_cache: list = [None]
    energy = MDFunction.apply(latent, positions, masses, n_steps, dt, engine_cache)
    return energy, engine_cache
