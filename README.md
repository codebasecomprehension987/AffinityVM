# AffinityVM

**Binding affinity prediction with a differentiable Rust MD engine**

AffinityVM combines a PyTorch Geometric GNN with a Rust molecular-dynamics engine
to predict pIC50 from SMILES alone — no crystal structure required. Gradients flow
end-to-end through the MD step via a custom `torch.autograd.Function` and a
forward finite-difference VJP.

> **Status:** Research codebase. No pre-trained weights or benchmarks are included.
> Results depend entirely on the training data and hyperparameter choices you supply.

```
SMILES
  -> LigandGNN (PyTorch Geometric)
       -> 256-dim latent
            -> Rust md_engine (SIMD when built with --features avx512)   [CPU, GIL-free]
                 10,000 MD steps, LJ + Coulomb forces
                 -> dE/d_latent (forward finite-difference VJP)
                      -> RegressionHead -> pIC50
```

---

## Architecture

| Component | Technology | Role |
|---|---|---|
| `LigandGNN` | PyTorch Geometric GATv2 | Molecular graph → 256-dim latent |
| `md_engine` (Rust) | PyO3 + optional AVX-512 intrinsics | 10k MD steps, single blocking call |
| `MDFunction` | `torch.autograd.Function` | Differentiable bridge, forward finite-difference VJP |
| `RegressionHead` | PyTorch MLP | Latent + energy scalar → pIC50 |
| `AffinityPipeline` | Python | End-to-end training and inference |

---

## Why Rust for the MD loop?

Python-level MD pays per-step overhead from `PyObject` creation, GIL acquisition,
and NumPy copy costs. At 10,000 steps per ligand this adds up. The Rust `md_engine` crate:

- Evaluates Lennard-Jones + Coulomb pairwise forces using `std::arch::x86_64::_mm512_*`
  intrinsics when built with `--features avx512`; falls back to a scalar i<j loop on
  all other targets.
- Exposes `Engine::step(n)` — Python calls it **once**, all steps run in Rust,
  GIL released via `py.allow_threads`.
- Zero-copy latent transfer: the GNN latent is passed as an `ndarray::ArrayView`
  backed by NumPy memory — no allocation.

---

## Memory safety

Backpropagation requires gradients to flow through the MD step.
`MDFunction.backward()` calls `Engine.energy_grad_latent()` which re-enters Rust
to compute dE/d_latent. Python's GC must not collect the Rust engine while the
backward graph holds a reference into it. This is enforced at two levels:

1. **Python side** — `engine_cache` list stored on `Function.ctx`, keeping the
   engine alive until `backward()` completes.
2. **Rust side** — `PhantomData<*const ()>` in the PyO3 binding marks `Engine`
   as `!Send`, tying its lifetime to the Python frame at compile time.

---

## Installation

**Prerequisites:** Rust toolchain via [rustup](https://rustup.rs) and `maturin`.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build the Rust extension — scalar build (all platforms)
maturin develop --release -m md_engine/Cargo.toml

# Build with AVX-512 SIMD (x86-64 only)
maturin develop --release -m md_engine/Cargo.toml --features avx512

# Install Python dependencies
pip install -e ".[gnn,chem]"
```

### Verify

```bash
pytest tests/ -v
```

---

## Quick start

```python
from affinityvm import AffinityPipeline

pipeline = AffinityPipeline.load("checkpoints/best_model.pt")

smiles = [
    "CC(=O)Nc1ccc(O)cc1",                        # paracetamol
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",       # testosterone
]

predictions = pipeline.predict(smiles)
for smi, pic50 in zip(smiles, predictions):
    print(f"{smi[:40]:<40}  pIC50 = {pic50:.2f}")
```

> A checkpoint must be produced by running training before `load()` will work.
> No pre-trained checkpoint is included in this repository.

---

## Training

```bash
affinityvm-train \
    --data       data/fep_benchmark.csv \
    --epochs     50                     \
    --batch_size 16                     \
    --n_md_steps 10000                  \
    --output     checkpoints/
```

`configs/default.yaml` supplies defaults for all parameters. Any CLI argument
passed explicitly overrides the corresponding config value.

---

## Benchmarking

```bash
affinityvm-benchmark \
    --checkpoint checkpoints/best_model.pt      \
    --data       data/fep_benchmark.csv         \
    --output     results/benchmark_results.json
```

Results — RMSE, MAE, Pearson r, Spearman ρ, and per-molecule errors — are written
to the output JSON. No pre-trained checkpoint or experimental results are included
in this repository; the numbers you see depend entirely on the data and training
run you supply.

---

## Repository structure

```
AffinityVM/
├── md_engine/                    # Rust crate (PyO3)
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                # LJ + Coulomb forces, optional AVX-512, finite-difference VJP
│
├── src/affinityvm/               # Python package
│   ├── __init__.py
│   ├── engine.py                 # Python wrapper around the Rust engine
│   ├── featurizer.py             # SMILES → PyG graph + 3-D coordinates + masses
│   ├── model.py                  # LigandGNN — GATv2 encoder → 256-dim latent
│   ├── autograd.py               # MDFunction — differentiable MD bridge
│   ├── pipeline.py               # AffinityPipeline — end-to-end train & infer
│   └── scripts/
│       ├── train.py              # affinityvm-train CLI entry point
│       └── benchmark.py          # affinityvm-benchmark CLI entry point
│
├── tests/
│   ├── test_autograd.py          # MDFunction forward, backward, and GC-safety tests
│   ├── test_engine.py            # Rust engine construction and step tests
│   └── test_model.py             # LigandGNN shape and gradient tests
│
├── configs/
│   └── default.yaml              # Default hyperparameters for training
│
├── pyproject.toml
└── README.md
```

---

## License

MIT
