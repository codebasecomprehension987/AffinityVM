# scripts/train.py
"""
AffinityVM training script.

Usage:
    python scripts/train.py --data data/fep_benchmark.csv \
                            --epochs 50 --batch_size 16 \
                            --lr 3e-4 --n_md_steps 10000 \
                            --output checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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


# ── Data loader ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[list[str], np.ndarray]:
    """Load SMILES + pIC50 from CSV (columns: smiles, pic50)."""
    smiles_list, labels = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"])
            labels.append(float(row["pic50"]))
    return smiles_list, np.array(labels, dtype=np.float32)


def minibatches(smiles: list[str], labels: np.ndarray, batch_size: int):
    idx = np.random.permutation(len(smiles))
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        yield [smiles[i] for i in batch_idx], labels[batch_idx]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        n_md_steps=args.n_md_steps,
        dt=0.001,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        batch_size=args.batch_size,
    )
    log.info("Device: %s", config.device)

    pipeline = AffinityPipeline(config=config).to(config.device)
    smiles_all, labels_all = load_csv(args.data)
    log.info("Loaded %d molecules from %s", len(smiles_all), args.data)

    # 80/20 split
    n = len(smiles_all)
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:split], perm[split:]
    smiles_train = [smiles_all[i] for i in train_idx]
    labels_train = labels_all[train_idx]
    smiles_val   = [smiles_all[i] for i in val_idx]
    labels_val   = labels_all[val_idx]

    optimiser  = AdamW(pipeline.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler  = CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn    = nn.MSELoss()

    best_val_rmse = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        pipeline.train()
        train_losses: list[float] = []

        for batch_smiles, batch_labels in minibatches(smiles_train, labels_train, args.batch_size):
            y_true = torch.tensor(batch_labels, dtype=torch.float32, device=config.device)
            try:
                y_pred = pipeline(batch_smiles)
                loss   = loss_fn(y_pred, y_true)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
                optimiser.step()
                train_losses.append(loss.item())
            except Exception as exc:
                log.warning("Skipping batch due to error: %s", exc)

        scheduler.step()
        train_rmse = float(np.sqrt(np.mean(train_losses))) if train_losses else float("nan")

        # ── Validate ───────────────────────────────────────────────────────
        pipeline.eval()
        val_preds = []
        for i in range(0, len(smiles_val), args.batch_size):
            batch = smiles_val[i : i + args.batch_size]
            try:
                with torch.no_grad():
                    preds = pipeline(batch).cpu().numpy()
                val_preds.extend(preds.tolist())
            except Exception as exc:
                log.warning("Validation batch error: %s", exc)

        if val_preds:
            val_rmse = float(np.sqrt(np.mean(
                (np.array(val_preds) - labels_val[:len(val_preds)]) ** 2
            )))
            pearson  = float(np.corrcoef(val_preds, labels_val[:len(val_preds)])[0, 1])
        else:
            val_rmse, pearson = float("nan"), float("nan")

        log.info(
            "Epoch %3d/%d | train_RMSE=%.4f | val_RMSE=%.4f | Pearson=%.4f | lr=%.2e",
            epoch, args.epochs, train_rmse, val_rmse, pearson,
            scheduler.get_last_lr()[0],
        )

        row = {"epoch": epoch, "train_rmse": train_rmse,
               "val_rmse": val_rmse, "pearson": pearson}
        history.append(row)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            ckpt_path = output_dir / "best_model.pt"
            pipeline.save(str(ckpt_path))
            log.info("  ✓ Saved best checkpoint → %s", ckpt_path)

    # Save history
    hist_path = output_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training complete. History → %s", hist_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AffinityVM")
    p.add_argument("--data",       required=True,  help="CSV with smiles,pic50 columns")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--n_md_steps", type=int,   default=10_000)
    p.add_argument("--output",     default="checkpoints/")
    p.add_argument("--cpu",        action="store_true", help="Force CPU")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
