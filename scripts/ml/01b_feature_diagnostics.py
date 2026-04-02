#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _common import available_training_features, safe_mkdir, save_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    out_dir = safe_mkdir(Path(args.out_dir))
    df = pd.read_csv(args.table_csv)
    num_cols, _ = available_training_features(df, include_cluster=False, include_baseline=True)
    corr_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corr = df[corr_cols].corr(method="spearman")
    corr.to_csv(out_dir / "spearman_correlation.csv", encoding="utf-8-sig")
    missing = df[corr_cols].isna().mean().sort_values(ascending=False)
    missing.to_csv(out_dir / "missing_share.csv", header=["missing_share"], encoding="utf-8-sig")

    top = corr_cols[: args.top_n]
    if top:
        plt.figure(figsize=(max(10, len(top) * 0.35), max(8, len(top) * 0.35)))
        im = plt.imshow(corr.loc[top, top], vmin=-1, vmax=1)
        plt.xticks(range(len(top)), top, rotation=90)
        plt.yticks(range(len(top)), top)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Spearman correlation")
        plt.tight_layout()
        plt.savefig(out_dir / "spearman_correlation_heatmap.png", dpi=160)
        plt.close()

    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.9:
                pairs.append({"feature_a": a, "feature_b": b, "spearman": float(v)})
    pairs_df = pd.DataFrame(pairs).sort_values("spearman", key=lambda s: s.abs(), ascending=False) if pairs else pd.DataFrame(columns=["feature_a", "feature_b", "spearman"])
    pairs_df.to_csv(out_dir / "high_corr_pairs.csv", index=False, encoding="utf-8-sig")

    save_json(out_dir / "feature_diagnostics_report.json", {
        "n_numeric": len(corr_cols),
        "corr_csv": str(out_dir / "spearman_correlation.csv"),
        "high_corr_pairs_csv": str(out_dir / "high_corr_pairs.csv"),
    })
    print({"n_numeric": len(corr_cols), "high_corr_pairs": int(len(pairs_df))})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
