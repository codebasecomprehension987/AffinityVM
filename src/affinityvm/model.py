# src/affinityvm/model.py
"""
LigandGNN — PyTorch Geometric message-passing network.

Encodes a molecular graph (atoms as nodes, bonds as edges) into a
256-dimensional latent vector.  The first N_atoms * 3 dims are decoded
by the Rust engine as per-atom force-field parameters (ε, σ, q).

Architecture:
  5 × GATv2Conv layers (128 hidden → 256 out)
  Global mean + max pooling
  2-layer MLP head → latent(256)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


LATENT_DIM = 256
HIDDEN_DIM = 128
N_ATOM_FEATURES = 36   # one-hot element (18) + degree (6) + charge (1) + ...
N_EDGE_FEATURES = 10   # bond type (4) + ring (1) + stereo (4) + conjugated (1)
N_GNN_LAYERS   = 5
DROPOUT         = 0.10


class GATv2Block(nn.Module):
    """GATv2 layer + residual + LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, edge_dim: int = N_EDGE_FEATURES):
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
        self.norm  = nn.LayerNorm(out_dim)
        self.proj  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.act(h)
        return self.norm(h + self.proj(x))


class LigandGNN(nn.Module):
    """
    Molecular graph encoder → 256-dim latent.

    The latent vector is structured so that
      latent[i*3 : i*3+3] = (ε_i, σ_i, q_i)
    for atom i (up to MAX_ATOMS atoms).
    Remaining dimensions encode global molecular context used by
    downstream affinity regression layers.
    """

    MAX_ATOMS = 80   # latent can encode up to 80 atoms' FF params

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

        # Node embedding
        self.node_emb = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge embedding
        self.edge_emb = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.SiLU(),
        )

        # GATv2 message-passing stack
        dims = [hidden_dim] + [hidden_dim] * (n_layers - 1) + [hidden_dim * 2]
        self.gnn_layers = nn.ModuleList([
            GATv2Block(dims[i], dims[i + 1], edge_dim=hidden_dim)
            for i in range(n_layers)
        ])

        # Graph-level pooling → latent
        pool_dim = hidden_dim * 2
        self.pool_norm = nn.LayerNorm(pool_dim * 2)  # mean + max concatenated
        self.head = nn.Sequential(
            nn.Linear(pool_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, data: "Data") -> torch.Tensor:
        """
        Args:
            data: PyG Data with fields:
                  x         [N, in_node_dim]
                  edge_index [2, E]
                  edge_attr  [E, in_edge_dim]
                  batch      [N] (molecule index within batch)

        Returns:
            latent: [B, 256]  (B = batch size)
        """
        x         = data.x.float()
        edge_index = data.edge_index
        edge_attr  = data.edge_attr.float() if data.edge_attr is not None else None
        batch      = data.batch

        x         = self.node_emb(x)
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)

        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Global pooling
        x_mean = global_mean_pool(x, batch)  # [B, D]
        x_max  = global_max_pool(x, batch)   # [B, D]
        x_pool = self.pool_norm(torch.cat([x_mean, x_max], dim=-1))

        return self.head(x_pool)             # [B, 256]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
