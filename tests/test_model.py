# tests/test_model.py
"""Unit tests for LigandGNN."""

import pytest
import torch


def test_pyg_import():
    pytest.importorskip("torch_geometric", reason="torch_geometric not installed")


def test_gnn_forward():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data, Batch
    from affinityvm.model import LigandGNN, N_ATOM_FEATURES, N_EDGE_FEATURES

    torch.manual_seed(0)
    model = LigandGNN()
    model.eval()

    # Toy graph: 5 atoms, 4 edges
    x          = torch.randn(5, N_ATOM_FEATURES)
    edge_index  = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
    edge_attr   = torch.randn(4, N_EDGE_FEATURES)
    batch       = torch.zeros(5, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    out  = model(data)

    assert out.shape == (1, 256), f"Expected (1, 256), got {out.shape}"
    assert torch.all(torch.isfinite(out))


def test_gnn_batch():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data, Batch
    from affinityvm.model import LigandGNN, N_ATOM_FEATURES, N_EDGE_FEATURES

    model = LigandGNN()
    model.eval()

    graphs = []
    for _ in range(4):
        n = torch.randint(3, 10, ()).item()
        e = torch.randint(2, n * 2, ()).item()
        graphs.append(Data(
            x=torch.randn(n, N_ATOM_FEATURES),
            edge_index=torch.randint(0, n, (2, e)),
            edge_attr=torch.randn(e, N_EDGE_FEATURES),
        ))

    batch = Batch.from_data_list(graphs)
    out   = model(batch)
    assert out.shape == (4, 256)


def test_gnn_gradient_flows():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from affinityvm.model import LigandGNN, N_ATOM_FEATURES, N_EDGE_FEATURES

    model = LigandGNN()
    x     = torch.randn(4, N_ATOM_FEATURES)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
    edge_attr  = torch.randn(4, N_EDGE_FEATURES)
    batch      = torch.zeros(4, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    out  = model(data)
    loss = out.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            pytest.fail(f"Parameter {name} has no gradient")
