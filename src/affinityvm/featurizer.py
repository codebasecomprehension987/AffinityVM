"""
MoleculeFeaturizer — converts SMILES / RDKit Mol to a PyTorch Geometric Data object.

Node features (44-dim):
  - One-hot atom type       (18): 17 element symbols + other
  - One-hot degree           (7): degrees 0-5 + other
  - One-hot formal charge    (6): -2, -1, 0, +1, +2 + other
  - Hybridisation            (5): SP, SP2, SP3, SP3D, SP3D2
  - Aromaticity              (1)
  - Ring membership          (1)
  - One-hot H count          (6): 0-4 + other

Edge features (10-dim):
  - Bond type (single, double, triple, aromatic — one-hot)  (4)
  - Conjugated                                              (1)
  - In ring                                                 (1)
  - Stereo (NONE, ANY, Z, E — one-hot)                     (4)

Also computes initial 3-D coordinates (MMFF94 or RDKit distance geometry)
and masses for the Rust engine.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import HybridizationType, BondType, BondStereo
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

# 17 named element types + "other" bucket = 18-dim one-hot
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P",
              "Si", "B", "Se", "Te", "As", "Sn", "Ge", "Cu", "other"]

HYBRIDISATION_TYPES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]

BOND_TYPES   = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
STEREO_TYPES = [BondStereo.STEREONONE, BondStereo.STEREOANY,
                BondStereo.STEREOZ,    BondStereo.STEREOE]

# Atomic masses (amu) keyed by symbol; fallback 12.0 for unknowns
MASSES: dict[str, float] = {
    "H": 1.008,  "C": 12.011, "N": 14.007, "O": 15.999,
    "S": 32.06,  "F": 18.998, "Cl": 35.45, "Br": 79.904,
    "I": 126.90, "P": 30.974, "Si": 28.085, "B": 10.811,
    "Se": 78.96, "Te": 127.60, "As": 74.922, "Sn": 118.71,
    "Ge": 72.630, "Cu": 63.546,
}


def _one_hot(value, choices: list) -> list[int]:
    return [int(value == c) for c in choices] + [int(value not in choices)]


def atom_features(atom) -> list[float]:
    feats: list[float] = []
    feats += _one_hot(atom.GetSymbol(), ATOM_TYPES[:-1])                       # 18
    feats += _one_hot(atom.GetDegree(), list(range(6)))                         # 7  (0-5 + other)
    feats += _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])               # 6  (-2..+2 + other)
    feats += [int(atom.GetHybridization() == h) for h in HYBRIDISATION_TYPES]  # 5
    feats += [int(atom.GetIsAromatic())]                                        # 1
    feats += [int(atom.IsInRing())]                                             # 1
    feats += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])                   # 6  (0-4 + other)
    return feats                                                                # total: 44


def bond_features(bond) -> list[float]:
    feats: list[float] = []
    feats += [int(bond.GetBondType() == b) for b in BOND_TYPES]    # 4
    feats += [int(bond.GetIsConjugated())]                          # 1
    feats += [int(bond.IsInRing())]                                 # 1
    feats += [int(bond.GetStereo() == s) for s in STEREO_TYPES]    # 4
    return feats                                                    # 10


class MoleculeFeaturizer:
    """Converts SMILES or RDKit Mol to PyG Data + coordinate arrays."""

    def __init__(self, max_atoms: int = 80, embed_attempts: int = 5):
        if not _HAS_RDKIT:
            raise ImportError("rdkit is required for MoleculeFeaturizer")
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for MoleculeFeaturizer")
        self.max_atoms      = max_atoms
        self.embed_attempts = embed_attempts

    # ── Main entry point ──────────────────────────────────────────────────────

    def featurize(self, smiles_or_mol) -> dict:
        """
        Returns a dict with keys:
          data       — torch_geometric.data.Data
          positions  — np.ndarray[float32, (N, 3)]  3-D coords in Å
          masses     — np.ndarray[float32, (N,)]
          n_atoms    — int
          smiles     — canonical SMILES
        """
        mol = self._to_mol(smiles_or_mol)
        mol = Chem.AddHs(mol)

        positions = self._embed_3d(mol)

        # Strip Hs for graph features but retain heavy-atom coords
        mol_noH     = Chem.RemoveHs(mol)
        heavy_idx   = [a.GetIdx() for a in mol_noH.GetAtoms()]
        pos_heavy   = positions[heavy_idx]

        x              = torch.tensor([atom_features(a) for a in mol_noH.GetAtoms()],
                                      dtype=torch.float)
        edge_index, edge_attr = self._build_edges(mol_noH)
        masses_arr     = np.array(
            [MASSES.get(a.GetSymbol(), 12.0) for a in mol_noH.GetAtoms()],
            dtype=np.float32,
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=mol_noH.GetNumAtoms(),
        )

        return {
            "data":      data,
            "positions": pos_heavy.astype(np.float32),
            "masses":    masses_arr,
            "n_atoms":   mol_noH.GetNumAtoms(),
            "smiles":    Chem.MolToSmiles(mol_noH),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_mol(x):
        if isinstance(x, str):
            mol = Chem.MolFromSmiles(x)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {x!r}")
            return mol
        return x

    def _embed_3d(self, mol_with_Hs) -> np.ndarray:
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        ps.numThreads = 0
        for _ in range(self.embed_attempts):
            if AllChem.EmbedMolecule(mol_with_Hs, ps) == 0:
                break
        if mol_with_Hs.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_with_Hs, AllChem.ETKDG())
        try:
            AllChem.MMFFOptimizeMolecule(mol_with_Hs, maxIters=500)
        except Exception:
            pass
        conf = mol_with_Hs.GetConformer()
        return np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol_with_Hs.GetNumAtoms())],
            dtype=np.float32,
        )

    @staticmethod
    def _build_edges(mol) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols, attrs = [], [], []
        for bond in mol.GetBonds():
            i, j  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat  = bond_features(bond)
            rows += [i, j];  cols += [j, i];  attrs += [feat, feat]
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr  = torch.tensor(attrs,        dtype=torch.float)
        return edge_index, edge_attr
