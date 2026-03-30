# tests/test_engine.py
"""Unit tests for AffinityEngine and the MD integration."""

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_system():
    """Two-atom system (argon dimer) near LJ minimum."""
    positions = np.array([[0.0, 0.0, 0.0],
                          [3.8, 0.0, 0.0]], dtype=np.float32)
    masses    = np.array([39.948, 39.948], dtype=np.float32)
    latent    = np.zeros(256, dtype=np.float32)
    # Atom 0 params: ε=0.238, σ=3.401 (argon); atom 1 same
    latent[0] = 0.238;  latent[1] = 3.401;  latent[2] = 0.0
    latent[3] = 0.238;  latent[4] = 3.401;  latent[5] = 0.0
    return positions, masses, latent


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_import():
    """Engine module is importable (may skip if Rust not built)."""
    pytest.importorskip("md_engine", reason="md_engine Rust extension not built")


def test_engine_construction(simple_system):
    md_engine = pytest.importorskip("md_engine")
    pos, mas, lat = simple_system
    engine = md_engine.Engine(pos, mas, lat, 0.001)
    assert engine is not None


def test_step_returns_energy(simple_system):
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat, dt=0.001)
    e = eng.step(100)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_energy_is_negative_near_min(simple_system):
    """Argon dimer at ~3.8 Å should have negative LJ energy."""
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat, dt=0.0)
    eng.step(0)
    e = eng.total_energy()
    assert e < 0, f"Expected negative energy near LJ min, got {e}"


def test_forces_shape(simple_system):
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(1)
    f = eng.forces
    assert f.shape == (2, 3), f"Expected (2,3) forces, got {f.shape}"


def test_forces_newton3(simple_system):
    """Forces must sum to zero (Newton's 3rd law)."""
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(0)
    f = eng.forces
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-4,
                               err_msg="Net force not zero — Newton 3 violated")


def test_gradient_shape(simple_system):
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(10)
    grad = eng.energy_grad_latent(lat, 1.0)
    assert grad.shape == (256,)
    assert np.all(np.isfinite(grad))


def test_step_count(simple_system):
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(500)
    assert eng.step_count == 500


def test_latent_dim_validation():
    """Engine must raise on wrong latent dimension."""
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos = np.zeros((2, 3), dtype=np.float32)
    mas = np.ones(2, dtype=np.float32)
    bad_lat = np.zeros(10, dtype=np.float32)   # wrong dim
    with pytest.raises(ValueError):
        AffinityEngine(pos, mas, bad_lat)


def test_positions_shape_validation():
    md_engine = pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    bad_pos = np.zeros((4,), dtype=np.float32)  # should be (N,3)
    mas     = np.ones(4, dtype=np.float32)
    lat     = np.zeros(256, dtype=np.float32)
    with pytest.raises(ValueError):
        AffinityEngine(bad_pos, mas, lat)
