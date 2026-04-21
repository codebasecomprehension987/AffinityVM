"""
AffinityPipeline — end-to-end pIC50 prediction.

Flow:
  SMILES
    -> MoleculeFeaturizer -> PyG Data + positions + masses
    -> LigandGNN          -> latent (256-dim, GPU)
    -> MDFunction.apply   -> MD energy (scalar, differentiable)
    -> RegressionHead     -> pIC50 prediction

The MD energy is concatenated with the GNN latent before regression,
providing both learned structural priors and physics-grounded energetics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from affinityvm.model import LigandGNN, LATENT_DIM
from affinityvm.autograd import md_energy
from affinityvm.featurizer import MoleculeFeaturizer

try:
    from torch_geometric.data import Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ── Regression head ───────────────────────────────────────────────────────────

class RegressionHead(nn.Module):
    """(latent || energy) -> pIC50."""

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = 0.15):
        super().__init__()
        in_dim = latent_dim + 1  # +1 for the MD energy scalar
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, latent: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        e = energy.view(-1, 1) / 100.0  # rough normalisation into ~[-1, 1] range
        return self.net(torch.cat([latent, e], dim=-1)).squeeze(-1)


# ── Pipeline config ───────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    n_md_steps: int   = 10_000
    dt:         float = 0.001   # ps
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int   = 32
    max_atoms:  int   = 80


# ── Main pipeline ─────────────────────────────────────────────────────────────

class AffinityPipeline(nn.Module):
    """
    Full AffinityVM inference + training pipeline.

    Args:
        config: PipelineConfig (defaults used if None)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__()
        self.config     = config or PipelineConfig()
        self.featurizer = MoleculeFeaturizer(max_atoms=self.config.max_atoms)
        self.gnn        = LigandGNN()
        self.head       = RegressionHead()
        # Populated each forward pass to prevent GC of Rust engines during backward
        self._engine_caches: list = []

    def forward(self, smiles_list: list[str]) -> torch.Tensor:
        """
        Predict pIC50 for a list of SMILES strings.

        Returns:
            predictions: float32 tensor of shape [B]
        """
        device       = self.config.device
        feat_outputs = [self.featurizer.featurize(s) for s in smiles_list]

        batch   = Batch.from_data_list([fo["data"] for fo in feat_outputs]).to(device)
        latents = self.gnn(batch)  # [B, 256]

        energies, caches = [], []
        for i, fo in enumerate(feat_outputs):
            pos_i = torch.from_numpy(fo["positions"])  # [N, 3] CPU
            mas_i = torch.from_numpy(fo["masses"])      # [N]    CPU
            e_i, cache_i = md_energy(
                latents[i], pos_i, mas_i,
                n_steps=self.config.n_md_steps,
                dt=self.config.dt,
            )
            energies.append(e_i)
            caches.append(cache_i)

        self._engine_caches = caches  # keep alive until backward

        energy_tensor = torch.stack(energies).to(device)  # [B]
        return self.head(latents, energy_tensor)

    def predict(self, smiles_list: list[str]) -> np.ndarray:
        """Inference-only prediction. Returns a numpy array of pIC50 values."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(smiles_list)
        return preds.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({
            "gnn_state":  self.gnn.state_dict(),
            "head_state": self.head.state_dict(),
            "config":     self.config,
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "AffinityPipeline":
        ckpt     = torch.load(path, map_location=map_location, weights_only=False)
        pipeline = cls(config=ckpt["config"])
        pipeline.gnn.load_state_dict(ckpt["gnn_state"])
        pipeline.head.load_state_dict(ckpt["head_state"])
        return pipeline
