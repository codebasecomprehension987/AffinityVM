"""
AffinityEngine — thin Python wrapper around the Rust `md_engine` PyO3 extension.

Responsibilities:
  - Lazy import with a clear error if the extension is not built
  - Shape and dtype validation before handing off to Rust
  - Convenience properties mirroring the Rust Engine API
"""

from __future__ import annotations

import numpy as np

try:
    import md_engine as _md_engine
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    _md_engine = None  # type: ignore[assignment]


LATENT_DIM = 256


class AffinityEngine:
    """
    Wraps the Rust `md_engine.Engine`.

    Args:
        positions: float32 array (N, 3) — initial atom positions in Angstroms
        masses:    float32 array (N,)   — atom masses in amu
        latent:    float32 array (256,) — GNN latent vector
        dt:        MD timestep in ps
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
            raise ValueError("masses must be (N,) matching positions row count")
        if latent.shape != (LATENT_DIM,):
            raise ValueError(f"latent must be ({LATENT_DIM},), got {latent.shape}")

        self._engine  = _md_engine.Engine(positions, masses, latent, dt)
        self._n_atoms = positions.shape[0]

    # ── MD control ───────────────────────────────────────────────────────────

    def step(self, n: int = 10_000) -> float:
        """Run `n` MD steps in Rust (GIL released). Returns total potential energy (kcal/mol)."""
        return float(self._engine.step(n))

    def total_energy(self) -> float:
        """Return current total potential energy without advancing the simulation."""
        return float(self._engine.total_energy())

    # ── State access ─────────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """Current positions [N, 3] in Angstroms."""
        return np.array(self._engine.positions())

    @property
    def forces(self) -> np.ndarray:
        """Current forces [N, 3] in kcal/(mol*Angstrom)."""
        return np.array(self._engine.forces())

    @property
    def step_count(self) -> int:
        """Total MD steps completed."""
        return int(self._engine.step_count())

    # ── Gradient ─────────────────────────────────────────────────────────────

    def energy_grad_latent(self, latent: np.ndarray, grad_output: float = 1.0) -> np.ndarray:
        """
        Compute dE/d_latent scaled by grad_output (VJP for backprop).

        Args:
            latent:      float32 array (256,)
            grad_output: upstream gradient scalar

        Returns:
            float32 array (256,)
        """
        latent = np.ascontiguousarray(latent, dtype=np.float32)
        return np.array(self._engine.energy_grad_latent(latent, float(grad_output)))

    def __repr__(self) -> str:
        return (
            f"AffinityEngine(n_atoms={self._n_atoms}, "
            f"steps={self.step_count}, "
            f"E={self.total_energy():.4f} kcal/mol)"
        )
