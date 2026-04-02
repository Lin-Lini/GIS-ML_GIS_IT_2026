#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import available_training_features, load_config, safe_mkdir, save_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config")
    args = ap.parse_args()
    try:
        import hdbscan
    except Exception as e:
        raise SystemExit("Не установлен hdbscan. Выполни: pip install hdbscan") from e

    cfg = load_config(args.config)
    hc = cfg["hdbscan"]
    out_dir = safe_mkdir(Path(args.out_dir))
    df = pd.read_csv(args.table_csv)
    numeric, _ = available_training_features(df, include_cluster=False, include_baseline=False)
    numeric = [c for c in numeric if c not in {"year_gap","n_missing_primary"}]
    if not numeric:
        raise SystemExit("Не найдены признаки для HDBSCAN")
    X = df[numeric].copy()
    prep = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(int(hc["pca_components"]), max(2, min(len(numeric), len(df)-1)))))
    ])
    Z = prep.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(hc["min_cluster_size"]), min_samples=int(hc["min_samples"]), prediction_data=True)
    cid = clusterer.fit_predict(Z)
    out = df.copy()
    out["cluster_id"] = cid.astype(int)
    out["cluster_prob"] = getattr(clusterer, "probabilities_", np.full(len(out), np.nan))
    out["cluster_outlier"] = getattr(clusterer, "outlier_scores_", np.full(len(out), np.nan))
    out_csv = out_dir / "parcel_year_clustered.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    cl_cols = ["cluster_id","cluster_prob","cluster_outlier"] + [c for c in ["weak_target","baseline_risk","change_share","texture_anomaly_share","water_share","delta_ndvi_mean","delta_ndwi_mean"] if c in out.columns]
    out.groupby("cluster_id", dropna=False)[cl_cols].agg(["count","mean"]).to_csv(out_dir / "cluster_summary.csv", encoding="utf-8-sig")
    joblib.dump({"preprocessor": prep, "clusterer": clusterer, "features": numeric}, out_dir / "hdbscan_artifacts.joblib")
    save_json(out_dir / "hdbscan_report.json", {"rows": int(len(out)), "csv": str(out_csv), "n_clusters_without_noise": int(len(set(cid)) - (1 if -1 in cid else 0)), "noise_rows": int((cid==-1).sum())})
    print({"csv": str(out_csv), "n_clusters": int(len(set(cid)) - (1 if -1 in cid else 0))})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
