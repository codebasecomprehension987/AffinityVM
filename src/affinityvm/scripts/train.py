"""
AffinityVM training script.

Usage:
    python -m affinityvm.scripts.train --data data/fep_benchmark.csv \\
                                       --epochs 50 --batch_size 16 \\
                                       --lr 3e-4 --n_md_steps 10000 \\
                                       --output checkpoints/

    # or via the installed CLI entry point:
    affinityvm-train --data data/fep_benchmark.csv --output checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from affinityvm.pipeline import AffinityPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[list[str], np.ndarray]:
    """Load SMILES + pIC50 from a CSV with columns: smiles, pic50."""
    smiles_list, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            smiles_list.append(row["smiles"])
            labels.append(float(row["pic50"]))
    return smiles_list, np.array(labels, dtype=np.float32)


def minibatches(smiles: list[str], labels: np.ndarray, batch_size: int):
    idx = np.random.permutation(len(smiles))
    for start in range(0, len(idx), batch_size):
        sel = idx[start : start + batch_size]
        yield [smiles[i] for i in sel], labels[sel]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    log.info("Device: %s", device)

    config   = PipelineConfig(n_md_steps=args.n_md_steps, dt=0.001,
                               device=device, batch_size=args.batch_size)
    pipeline = AffinityPipeline(config=config).to(device)

    smiles_all, labels_all = load_csv(args.data)
    log.info("Loaded %d molecules from %s", len(smiles_all), args.data)

    n      = len(smiles_all)
    perm   = np.random.permutation(n)
    split  = int(0.8 * n)
    smiles_train = [smiles_all[i] for i in perm[:split]]
    labels_train = labels_all[perm[:split]]
    smiles_val   = [smiles_all[i] for i in perm[split:]]
    labels_val   = labels_all[perm[split:]]

    optimiser = AdamW(pipeline.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn   = nn.MSELoss()

    best_val_rmse = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        pipeline.train()
        sq_errors: list[float] = []

        for batch_smiles, batch_labels in minibatches(smiles_train, labels_train, args.batch_size):
            y_true = torch.tensor(batch_labels, dtype=torch.float32, device=device)
            try:
                y_pred = pipeline(batch_smiles)
                loss   = loss_fn(y_pred, y_true)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
                optimiser.step()
                sq_errors.append(loss.item() * len(batch_labels))
            except Exception as exc:
                log.warning("Skipping batch: %s", exc)

        scheduler.step()
        n_train    = len(smiles_train)
        train_rmse = float(np.sqrt(sum(sq_errors) / n_train)) if sq_errors else float("nan")

        # ── Validate ──────────────────────────────────────────────────────
        pipeline.eval()
        val_preds: list[float] = []
        for i in range(0, len(smiles_val), args.batch_size):
            try:
                with torch.no_grad():
                    preds = pipeline(smiles_val[i : i + args.batch_size]).cpu().numpy()
                val_preds.extend(preds.tolist())
            except Exception as exc:
                log.warning("Validation batch error: %s", exc)

        n_val = len(val_preds)
        if n_val:
            val_rmse = float(np.sqrt(np.mean(
                (np.array(val_preds) - labels_val[:n_val]) ** 2
            )))
            pearson = float(np.corrcoef(val_preds, labels_val[:n_val])[0, 1])
        else:
            val_rmse = pearson = float("nan")

        log.info(
            "Epoch %3d/%d | train_RMSE=%.4f | val_RMSE=%.4f | Pearson=%.4f | lr=%.2e",
            epoch, args.epochs, train_rmse, val_rmse, pearson,
            scheduler.get_last_lr()[0],
        )
        history.append({"epoch": epoch, "train_rmse": train_rmse,
                        "val_rmse": val_rmse, "pearson": pearson})

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            ckpt = output_dir / "best_model.pt"
            pipeline.save(str(ckpt))
            log.info("  Saved checkpoint -> %s", ckpt)

    hist_path = output_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training complete. History -> %s", hist_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AffinityVM")
    p.add_argument("--data",       required=True, help="CSV with smiles,pic50 columns")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--n_md_steps", type=int,   default=10_000)
    p.add_argument("--output",     default="checkpoints/")
    p.add_argument("--cpu",        action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
