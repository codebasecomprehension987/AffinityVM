"""Unit tests for AffinityEngine and the MD integration."""

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_system():
    """Two-atom argon dimer near the LJ energy minimum (~3.8 Å)."""
    positions = np.array([[0.0, 0.0, 0.0],
                          [3.8, 0.0, 0.0]], dtype=np.float32)
    masses    = np.array([39.948, 39.948], dtype=np.float32)
    latent    = np.zeros(256, dtype=np.float32)
    # Atom 0: eps=0.238, sigma=3.401 (argon LJ params); atom 1 same
    latent[0] = 0.238;  latent[1] = 3.401;  latent[2] = 0.0
    latent[3] = 0.238;  latent[4] = 3.401;  latent[5] = 0.0
    return positions, masses, latent


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_import():
    """Engine module is importable (skipped if Rust extension not built)."""
    pytest.importorskip("md_engine", reason="md_engine Rust extension not built")


def test_engine_construction(simple_system):
    md_engine = pytest.importorskip("md_engine")
    pos, mas, lat = simple_system
    assert md_engine.Engine(pos, mas, lat, 0.001) is not None


def test_step_returns_energy(simple_system):
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    e = AffinityEngine(pos, mas, lat, dt=0.001).step(100)
    assert isinstance(e, float)
    assert np.isfinite(e)


def test_energy_is_negative_near_min(simple_system):
    """Argon dimer at ~3.8 Å should have negative LJ energy."""
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat, dt=0.001)
    eng.step(0)
    e = eng.total_energy()
    assert e < 0, f"Expected negative energy near LJ minimum, got {e}"


def test_forces_shape(simple_system):
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(1)
    assert eng.forces.shape == (2, 3)


def test_forces_newton3(simple_system):
    """Net force must be zero (Newton's third law)."""
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(0)
    np.testing.assert_allclose(
        eng.forces.sum(axis=0), 0.0, atol=1e-4,
        err_msg="Net force not zero — Newton 3 violated",
    )


def test_gradient_shape(simple_system):
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(10)
    grad = eng.energy_grad_latent(lat, 1.0)
    assert grad.shape == (256,)
    assert np.all(np.isfinite(grad))


def test_step_count(simple_system):
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos, mas, lat = simple_system
    eng = AffinityEngine(pos, mas, lat)
    eng.step(500)
    assert eng.step_count == 500


def test_latent_dim_validation():
    """Engine must raise ValueError on wrong latent dimension."""
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    pos = np.zeros((2, 3), dtype=np.float32)
    mas = np.ones(2, dtype=np.float32)
    with pytest.raises(ValueError):
        AffinityEngine(pos, mas, np.zeros(10, dtype=np.float32))


def test_positions_shape_validation():
    """Engine must raise ValueError on non-(N,3) positions."""
    pytest.importorskip("md_engine")
    from affinityvm.engine import AffinityEngine
    with pytest.raises(ValueError):
        AffinityEngine(
            np.zeros((4,), dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.zeros(256, dtype=np.float32),
        )
