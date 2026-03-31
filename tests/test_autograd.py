# tests/test_autograd.py
"""
Tests for the MDFunction custom autograd.

Validates:
  - forward pass returns a scalar tensor
  - backward pass returns a gradient with correct shape
  - gradient is finite
  - numerical gradient check (finite-difference vs autograd)
"""

import numpy as np
import pytest
import torch


def make_simple_latent(n_atoms: int = 2) -> torch.Tensor:
    lat = torch.zeros(256, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for i in range(n_atoms):
            lat[i * 3]     = 0.238   # epsilon
            lat[i * 3 + 1] = 3.401   # sigma
            lat[i * 3 + 2] = 0.0     # charge
    lat.requires_grad_(True)
    return lat


def test_md_function_forward():
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    lat = make_simple_latent(2)
    pos = torch.tensor([[0., 0., 0.], [3.8, 0., 0.]], dtype=torch.float32)
    mas = torch.tensor([39.948, 39.948], dtype=torch.float32)

    energy, cache = md_energy(lat, pos, mas, n_steps=50, dt=0.001)
    assert energy.shape == ()
    assert torch.isfinite(energy)


def test_md_function_backward():
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    lat = make_simple_latent(2)
    pos = torch.tensor([[0., 0., 0.], [3.8, 0., 0.]], dtype=torch.float32)
    mas = torch.tensor([39.948, 39.948], dtype=torch.float32)

    energy, cache = md_energy(lat, pos, mas, n_steps=10, dt=0.001)
    energy.backward()

    assert lat.grad is not None, "No gradient on latent"
    assert lat.grad.shape == (256,)
    assert torch.all(torch.isfinite(lat.grad)), "Non-finite gradient"


def test_gradient_numerical():
    """
    Loose numerical gradient check.
    Computes finite-difference ∂E/∂lat[0] and compares to autograd.
    """
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    eps = 1e-3
    pos = torch.tensor([[0., 0., 0.], [3.8, 0., 0.]], dtype=torch.float32)
    mas = torch.tensor([39.948, 39.948], dtype=torch.float32)

    # E(lat)
    lat0 = make_simple_latent(2)
    e0, _ = md_energy(lat0.detach(), pos, mas, n_steps=0, dt=0.001)

    # E(lat + eps * e_0)
    lat_p = make_simple_latent(2)
    with torch.no_grad():
        lat_p[0] += eps
    e_p, _ = md_energy(lat_p.detach(), pos, mas, n_steps=0, dt=0.001)

    fd_grad = (e_p.item() - e0.item()) / eps

    # Autograd grad
    lat_a = make_simple_latent(2)
    e_a, _ = md_energy(lat_a, pos, mas, n_steps=0, dt=0.001)
    e_a.backward()
    ag_grad = lat_a.grad[0].item()

    # Allow loose tolerance (finite-diff + float32 noise)
    assert abs(fd_grad - ag_grad) < 0.5 or abs(fd_grad) < 1e-6, (
        f"Numerical grad {fd_grad:.4f} vs autograd {ag_grad:.4f}"
    )


def test_engine_not_gc_during_backward():
    """Engine object must survive until backward completes."""
    import gc
    pytest.importorskip("md_engine")
    from affinityvm.autograd import md_energy

    lat = make_simple_latent(2)
    pos = torch.tensor([[0., 0., 0.], [3.8, 0., 0.]], dtype=torch.float32)
    mas = torch.tensor([39.948, 39.948], dtype=torch.float32)

    energy, cache = md_energy(lat, pos, mas, n_steps=10)
    gc.collect()   # trigger GC — engine must still be alive via cache ref
    energy.backward()
    assert lat.grad is not None
