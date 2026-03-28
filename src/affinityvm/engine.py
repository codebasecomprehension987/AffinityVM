# src/affinityvm/engine.py
"""
AffinityEngine — thin Python wrapper around the Rust `md_engine` PyO3 module.

Handles:
  - Lazy import of the compiled extension (graceful error if not built)
  - Type validation and shape checks before handing off to Rust
  - Convenience methods for Python callers
"""

from __future__ import annotations

import numpy as np

try:
    import md_engine as _md_engine
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    _md_engine = None   # type: ignore[assignment]


LATENT_DIM = 256


class AffinityEngine:
    """
    Wraps the Rust `md_engine.Engine`.

    Args:
        positions: np.ndarray[float32, (N, 3)]  — initial atom positions in Å
        masses:    np.ndarray[float32, (N,)]    — atom masses in amu
        latent:    np.ndarray[float32, (256,)]  — GNN latent vector
        dt:        float                         — MD timestep in ps
    """

    def __init__(
        self,
        positions: np.ndarray,
        masses:    np.ndarray,
        latent:    np.ndarray,
        dt:        float = 0.001,
    ) -> None:
        if not _HAS_RUST:
            raise RuntimeError(
                "Rust md_engine extension not found. "
                "Run `maturin develop --release` in the md_engine/ directory."
            )
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        masses    = np.ascontiguousarray(masses,    dtype=np.float32)
        latent    = np.ascontiguousarray(latent,    dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {positions.shape}")
        if masses.ndim != 1 or masses.shape[0] != positions.shape[0]:
            raise ValueError("masses must be (N,) matching positions rows")
        if latent.shape != (LATENT_DIM,):
            raise ValueError(f"latent must be ({LATENT_DIM},), got {latent.shape}")

        self._engine = _md_engine.Engine(positions, masses, latent, dt)
        self._n_atoms = positions.shape[0]

    # ── MD control ───────────────────────────────────────────────────────────

    def step(self, n: int = 10_000) -> float:
        """
        Run `n` MD steps in Rust (GIL released).

        Returns total potential energy (kcal/mol).
        """
        return float(self._engine.step(n))

    def total_energy(self) -> float:
        """Return total potential energy without running additional steps."""
        return float(self._engine.total_energy())

    # ── State access ─────────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """Current positions [N, 3] Å."""
        return np.array(self._engine.positions())

    @property
    def forces(self) -> np.ndarray:
        """Current forces [N, 3] kcal/(mol·Å)."""
        return np.array(self._engine.forces())

    @property
    def step_count(self) -> int:
        """Total number of MD steps completed."""
        return int(self._engine.step_count())

    # ── Gradient ─────────────────────────────────────────────────────────────

    def energy_grad_latent(self, latent: np.ndarray, grad_output: float = 1.0) -> np.ndarray:
        """
        Compute ∂E/∂latent scaled by grad_output.

        Args:
            latent:       np.ndarray[float32, (256,)]
            grad_output:  upstream gradient scalar

        Returns:
            np.ndarray[float32, (256,)]
        """
        latent = np.ascontiguousarray(latent, dtype=np.float32)
        return np.array(
            self._engine.energy_grad_latent(latent, float(grad_output))
        )

    # ── Representation ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AffinityEngine(n_atoms={self._n_atoms}, "
            f"steps={self.step_count}, "
            f"E={self.total_energy():.4f} kcal/mol)"
        )
