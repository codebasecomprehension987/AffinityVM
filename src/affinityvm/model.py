"""
LigandGNN — PyTorch Geometric message-passing network.

Encodes a molecular graph (atoms as nodes, bonds as edges) into a
256-dimensional latent vector. The first N_atoms * 3 dims are decoded
by the Rust engine as per-atom force-field parameters (eps, sigma, q).

Architecture:
  5 x GATv2Conv layers (128 hidden -> 256 out)
  Global mean + max pooling
  2-layer MLP head -> latent(256)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


LATENT_DIM = 256
HIDDEN_DIM = 128
# Node feature breakdown (44 total) — must match featurizer.atom_features():
#   atom type one-hot  (18): 17 element types + other
#   degree one-hot      (7): degrees 0-5 + other
#   formal charge       (6): -2 to +2 + other
#   hybridisation       (5): SP / SP2 / SP3 / SP3D / SP3D2
#   aromaticity         (1)
#   ring membership     (1)
#   H count one-hot     (6): 0-4 + other
N_ATOM_FEATURES = 44
N_EDGE_FEATURES = 10   # bond type (4) + conjugated (1) + ring (1) + stereo (4)
N_GNN_LAYERS    = 5
DROPOUT         = 0.10


class GATv2Block(nn.Module):
    """GATv2 layer with residual projection and LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, edge_dim: int = HIDDEN_DIM):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for LigandGNN")
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=DROPOUT,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv(x, edge_index, edge_attr))
        return self.norm(h + self.proj(x))


class LigandGNN(nn.Module):
    """
    Molecular graph encoder -> 256-dim latent.

    Latent layout: latent[i*3 : i*3+3] = (eps_i, sigma_i, q_i) for atom i
    (up to MAX_ATOMS atoms). Remaining dims encode global molecular context
    consumed by the downstream regression head.
    """

    MAX_ATOMS = 80  # latent encodes FF params for up to 80 atoms

    def __init__(
        self,
        in_node_dim: int = N_ATOM_FEATURES,
        in_edge_dim: int = N_EDGE_FEATURES,
        hidden_dim:  int = HIDDEN_DIM,
        latent_dim:  int = LATENT_DIM,
        n_layers:    int = N_GNN_LAYERS,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required")

        self.latent_dim = latent_dim

        self.node_emb = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.SiLU(),
        )

        # GATv2 stack; final layer doubles width for richer pooling
        dims = [hidden_dim] * n_layers + [hidden_dim * 2]
        self.gnn_layers = nn.ModuleList([
            GATv2Block(dims[i], dims[i + 1])
            for i in range(n_layers)
        ])

        pool_dim = hidden_dim * 2
        self.pool_norm = nn.LayerNorm(pool_dim * 2)  # mean || max
        self.head = nn.Sequential(
            nn.Linear(pool_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, data: "Data") -> torch.Tensor:
        """
        Args:
            data: PyG Data — x [N, node_dim], edge_index [2, E],
                  edge_attr [E, edge_dim], batch [N]
        Returns:
            latent [B, 256]
        """
        if data.x.size(0) > self.MAX_ATOMS * (data.batch.max().item() + 1):
            raise ValueError(
                f"Batch contains a molecule exceeding MAX_ATOMS={self.MAX_ATOMS}. "
                f"Filter oversized molecules before batching."
            )
        x          = data.x.float()
        edge_index = data.edge_index
        edge_attr  = data.edge_attr.float() if data.edge_attr is not None else None
        batch      = data.batch

        x = self.node_emb(x)
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)

        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_pool = self.pool_norm(torch.cat([x_mean, x_max], dim=-1))

        return self.head(x_pool)  # [B, 256]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
