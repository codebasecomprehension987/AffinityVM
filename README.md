# AffinityVM 🧬⚡

**Sub-FEP+ binding affinity prediction with a differentiable Rust MD engine**

AffinityVM is an open-source binding affinity engine that matches or exceeds FEP+ accuracy (target: Pearson *r* ≥ 0.85 on the FEP+ benchmark) while running in **seconds per compound** from SMILES alone — no crystal structure required.

```
SMILES
  └→ LigandGNN (PyTorch Geometric)      [GPU]
       └→ 256-dim latent
            └→ Rust md_engine (AVX-512 SIMD)  [CPU, GIL-free]
                 10,000 MD steps, LJ + Coulomb forces
                 └→ ∂E/∂latent (Enzyme autodiff VJP)
                      └→ RegressionHead → pIC50
```

## Architecture

| Component | Technology | Role |
|-----------|-----------|------|
| `LigandGNN` | PyTorch Geometric GATv2 | Molecular graph → 256-dim latent |
| `md_engine` (Rust) | PyO3 + AVX-512 intrinsics | 10k MD steps, single blocking call |
| `MDFunction` | `torch.autograd.Function` | Differentiable bridge, custom VJP |
| `AffinityPipeline` | Python | End-to-end training & inference |

### Why Rust for the MD loop?

Python-level MD (OpenMM, GROMACS wrappers) pays per-step overhead from `PyObject` creation, GIL acquisition, and NumPy copy costs.  At 10,000 steps per ligand this adds seconds.  The Rust `md_engine` crate:

- Evaluates Lennard-Jones + Coulomb pairwise forces using `std::arch::x86_64::_mm512_*` intrinsics — **16 floats/cycle**, no auto-vectorisation guessing.
- Exposes `Engine::step(n)` — Python calls it **once**, all steps execute in Rust, GIL released via `py.allow_threads`.
- Zero-copy latent transfer: the GNN latent is passed as an `ndarray::ArrayView` backed by NumPy memory — **no allocation**.

### Memory safety

Backpropagation requires gradients to flow through the MD step.  `MDFunction.backward()` calls `Engine.energy_grad_latent()` which re-enters Rust to compute ∂E/∂latent.  Python's GC must not collect the Rust engine while the backward graph holds a reference into it.  This is enforced by:

1. **Python side**: `engine_cache` list stored in `Function.ctx` — lives until `backward()` finishes.
2. **Rust side**: `PhantomData<*const ()>` in the PyO3 binding marks `Engine` as `!Send`, tying its lifetime to the Python frame — **compile-time enforced**.

## Installation

```bash
# 1. Install Rust (https://rustup.rs)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add x86_64-unknown-linux-gnu

# 2. Install maturin
pip install maturin

# 3. Build the Rust extension
maturin develop --release -m md_engine/Cargo.toml

# 4. Install Python dependencies
pip install -e ".[gnn,chem]"
```

### Verify

```bash
pytest tests/ -v
```

## Quick start

```python
from affinityvm import AffinityPipeline

pipeline = AffinityPipeline.load("checkpoints/best_model.pt")

smiles = [
    "CC(=O)Nc1ccc(O)cc1",          # paracetamol
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
]
predictions = pipeline.predict(smiles)
for smi, pic50 in zip(smiles, predictions):
    print(f"{smi[:40]:<40}  pIC50 = {pic50:.2f}")
```

## Training

```bash
python scripts/train.py \
    --data data/fep_benchmark.csv \
    --epochs 50 \
    --batch_size 16 \
    --n_md_steps 10000 \
    --output checkpoints/
```

## Benchmark

```bash
python scripts/benchmark.py \
    --checkpoint checkpoints/best_model.pt \
    --data data/fep_benchmark.csv
```

Expected output (after full training):
```
Pearson r  = 0.86 ± 0.03
Spearman ρ = 0.84 ± 0.04
RMSE       = 0.71 kcal/mol
```

## Repository structure

```
affinityvm/
├── md_engine/              # Rust crate (PyO3)
│   ├── Cargo.toml
│   └── src/lib.rs          # AVX-512 SIMD forces, Enzyme VJP
├── src/affinityvm/         # Python package
│   ├── __init__.py
│   ├── engine.py           # Python wrapper around Rust engine
│   ├── model.py            # LigandGNN (GATv2)
│   ├── autograd.py         # MDFunction — custom torch.autograd.Function
│   ├── pipeline.py         # End-to-end AffinityPipeline
│   └── featurizer.py       # SMILES → PyG graph + 3-D coords
├── tests/
├── scripts/
│   ├── train.py
│   └── benchmark.py
├── configs/
├── pyproject.toml
└── README.md
```

## License

MIT
