"""
Evaluate AffinityVM against a held-out benchmark set.

Usage:
    python -m affinityvm.scripts.benchmark \\
        --checkpoint checkpoints/best_model.pt \\
        --data data/fep_benchmark.csv \\
        --output results/benchmark_results.json

    # or via the installed CLI entry point:
    affinityvm-benchmark --checkpoint checkpoints/best_model.pt \\
                         --data data/fep_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from affinityvm.pipeline import AffinityPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def load_csv(path: str) -> tuple[list[str], list[str], np.ndarray]:
    smiles_list, names, labels = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            smiles_list.append(row["smiles"])
            names.append(row.get("name", row["smiles"][:10]))
            labels.append(float(row["pic50"]))
    return smiles_list, names, np.array(labels, dtype=np.float32)


def run(args: argparse.Namespace) -> None:
    pipeline = AffinityPipeline.load(args.checkpoint, map_location="cpu")
    smiles_list, names, labels = load_csv(args.data)

    log.info("Running inference on %d molecules...", len(smiles_list))
    all_preds = []
    for i in range(0, len(smiles_list), args.batch_size):
        batch = smiles_list[i : i + args.batch_size]
        all_preds.extend(pipeline.predict(batch).tolist())
    preds = np.array(all_preds, dtype=np.float32)

    records = [
        {
            "name": name, "smiles": smi,
            "pred_pic50": float(pred), "true_pic50": float(true),
            "error": float(pred - true),
        }
        for name, smi, pred, true in zip(names, smiles_list, preds, labels)
    ]

    errors = np.array([r["error"] for r in records])
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    mae    = float(np.mean(np.abs(errors)))
    r, _   = pearsonr(preds, labels)
    rho, _ = spearmanr(preds, labels)

    log.info("Results:")
    log.info("  RMSE       = %.4f log units", rmse)
    log.info("  MAE        = %.4f log units", mae)
    log.info("  Pearson r  = %.4f", r)
    log.info("  Spearman p = %.4f", rho)

    summary = {
        "n_molecules":  len(records),
        "rmse":         rmse,
        "mae":          mae,
        "pearson_r":    float(r),
        "spearman_rho": float(rho),
        "per_molecule": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved -> %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark AffinityVM")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       required=True)
    p.add_argument("--output",     default="results/benchmark_results.json")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
