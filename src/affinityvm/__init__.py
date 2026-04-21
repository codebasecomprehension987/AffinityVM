"""
AffinityVM: binding affinity prediction with a differentiable Rust MD engine.

Architecture:
  GNN (PyTorch Geometric) → 256-dim latent
  Rust md_engine (AVX-512 SIMD) → MD trajectory + energy
  torch.autograd.Function (custom VJP) → end-to-end gradients
"""

from affinityvm.engine import AffinityEngine
from affinityvm.model import LigandGNN
from affinityvm.autograd import MDFunction
from affinityvm.pipeline import AffinityPipeline
from affinityvm.featurizer import MoleculeFeaturizer

__version__ = "0.1.0"
__all__ = [
    "AffinityEngine",
    "LigandGNN",
    "MDFunction",
    "AffinityPipeline",
    "MoleculeFeaturizer",
]
