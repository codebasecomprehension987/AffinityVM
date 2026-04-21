"""Tests for MDFunction — the custom torch.autograd.Function."""

import gc

import numpy as np
import pytest
import torch


def _make_latent(n_atoms: int = 2) -> torch.Tensor:
    """Latent with argon LJ params for the first n_atoms atoms."""
    lat = torch.zeros(256, dtype=torch.float32)
    for i in range(n_atoms):
        lat[i * 3]     = 0.238  # epsilon
        lat[i * 3 + 1] = 3.401  # sigma
        lat[i * 3 + 2] = 0.0   # charge
    return lat.requires_grad_(True)


_POS = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=torch.float32)
_MAS = torch.tensor([39.948, 39.948], dtype=torch.float32)


def test_forward_returns_scalar():
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    energy, _ = md_energy(_make_latent(), _POS, _MAS, n_steps=50, dt=0.001)
    assert energy.shape == ()
    assert torch.isfinite(energy)


def test_backward_gradient_shape():
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    lat = _make_latent()
    energy, _ = md_energy(lat, _POS, _MAS, n_steps=10, dt=0.001)
    energy.backward()

    assert lat.grad is not None
    assert lat.grad.shape == (256,)
    assert torch.all(torch.isfinite(lat.grad)), "Non-finite gradient"


def test_gradient_numerical():
    """Finite-difference check: dE/dlat[0] vs autograd (loose tolerance for float32)."""
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    eps = 1e-3

    lat0 = _make_latent()
    e0, _ = md_energy(lat0.detach(), _POS, _MAS, n_steps=0, dt=0.001)

    lat_p = _make_latent()
    with torch.no_grad():
        lat_p[0] += eps
    e_p, _ = md_energy(lat_p.detach(), _POS, _MAS, n_steps=0, dt=0.001)

    fd_grad = (e_p.item() - e0.item()) / eps

    lat_a = _make_latent()
    e_a, _ = md_energy(lat_a, _POS, _MAS, n_steps=0, dt=0.001)
    e_a.backward()
    ag_grad = lat_a.grad[0].item()

    assert abs(fd_grad - ag_grad) < 0.5, (
        f"Numerical grad {fd_grad:.4f} vs autograd {ag_grad:.4f}"
    )

def test_engine_survives_gc():
    """Rust Engine must remain alive through GC until backward() completes."""
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    lat = _make_latent()
    energy, cache = md_energy(lat, _POS, _MAS, n_steps=10)
    gc.collect()  # must not free the engine — cache holds the ref
    energy.backward()
    assert lat.grad is not None
