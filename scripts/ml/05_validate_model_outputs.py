#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from _common import prepare_parcels_with_area, read_geo, robust_clip_prob, safe_mkdir, save_json, slugify_area


PRED_COLUMNS = [
    "pred_baseline_risk",
    "pred_logit",
    "pred_catboost",
    "pred_ml_mean",
    "pred_ml_final",
    "pred_logit_oof",
    "pred_catboost_oof",
]


def _metric_row(y, p, name):
    yy = np.asarray(y, dtype=int)
    pp = robust_clip_prob(p)
    if len(np.unique(yy)) < 2:
        return {"model": name, "roc_auc": None, "pr_auc": None, "logloss": None, "brier": None}
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(yy, pp)),
        "pr_auc": float(average_precision_score(yy, pp)),
        "logloss": float(log_loss(yy, pp)),
        "brier": float(brier_score_loss(yy, pp)),
    }


def _make_internal_metrics(df, out_dir):
    labeled = df[df["weak_target"].isin([0, 1])].copy() if "weak_target" in df.columns else pd.DataFrame()
    if labeled.empty:
        return pd.DataFrame()
    rows = []
    for col in PRED_COLUMNS:
        if col in labeled.columns:
            rows.append(_metric_row(labeled["weak_target"], labeled[col], col))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_dir / "internal_model_vs_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def _join_geometries(results_dir, df):
    parcels = prepare_parcels_with_area(results_dir)
    p = parcels[[c for c in ["parcel_id", "area", "geometry"] if c in parcels.columns]].copy()
    p["parcel_id"] = p["parcel_id"].astype(str)
    out = df.copy()
    out["parcel_id"] = out["parcel_id"].astype(str)
    if "area" in out.columns:
        gdf = p.merge(out, on=["parcel_id", "area"], how="right")
    else:
        gdf = p.merge(out, on="parcel_id", how="right")
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=parcels.crs)


def _external_stats(gdf, masks, out_dir, top_quantile, min_inside, min_outside):
    if gdf.crs != masks.crs:
        masks = masks.to_crs(gdf.crs)
    joined = gpd.sjoin(
        gdf,
        masks[["geometry", *[c for c in ["mask_name", "area"] if c in masks.columns]]],
        predicate="intersects",
        how="left",
    )
    joined["inside_mask"] = joined["index_right"].notna()

    area_col = "area"
    if area_col not in joined.columns:
        area_col = "area_left" if "area_left" in joined.columns else area_col
    year_col = "year"
    if year_col not in joined.columns:
        year_col = "year_left" if "year_left" in joined.columns else year_col

    pred_cols = [c for c in ["pred_baseline_risk", "pred_logit", "pred_catboost", "pred_ml_mean", "pred_ml_final"] if c in joined.columns]
    rows = []
    warnings = []
    for (area, year), sub in joined.groupby([area_col, year_col], dropna=False):
        inside = sub[sub["inside_mask"]].copy()
        outside = sub[~sub["inside_mask"]].copy()
        row = {
            "area": area,
            "year": year,
            "n_total": int(len(sub)),
            "n_inside": int(len(inside)),
            "n_outside": int(len(outside)),
        }
        if len(inside) < min_inside or len(outside) < min_outside:
            warnings.append({
                "area": area,
                "year": int(year) if pd.notna(year) else None,
                "n_inside": int(len(inside)),
                "n_outside": int(len(outside)),
                "warning": f"thin_validation_sample: need inside>={min_inside} and outside>={min_outside}",
            })
        for col in pred_cols:
            thr = sub[col].quantile(1 - top_quantile)
            inside_mean = float(inside[col].mean()) if len(inside) else np.nan
            outside_mean = float(outside[col].mean()) if len(outside) else np.nan
            inside_top = float((inside[col] >= thr).mean()) if len(inside) else np.nan
            outside_top = float((outside[col] >= thr).mean()) if len(outside) else np.nan
            row[f"{col}_inside_mean"] = inside_mean
            row[f"{col}_outside_mean"] = outside_mean
            row[f"{col}_inside_minus_outside"] = inside_mean - outside_mean if np.isfinite(inside_mean) and np.isfinite(outside_mean) else np.nan
            row[f"{col}_lift"] = inside_mean / max(outside_mean, 1e-9) if len(inside) and len(outside) else np.nan
            row[f"{col}_topq_inside_share"] = inside_top
            row[f"{col}_topq_outside_share"] = outside_top
            row[f"{col}_topq_enrichment"] = inside_top / max(outside_top, 1e-9) if np.isfinite(inside_top) and np.isfinite(outside_top) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["area", "year"])
    out.to_csv(out_dir / "external_mask_validation.csv", index=False, encoding="utf-8-sig")
    save_json(out_dir / "external_mask_validation_warnings.json", {"warnings": warnings})
    return out, warnings


def _plot_timeseries(ext, out_dir):
    for area, sub in ext.groupby("area"):
        area_slug = slugify_area(str(area))
        plt.figure(figsize=(10, 6))
        for col, label in [("pred_ml_final_inside_mean", "ml_final_inside_mean"), ("pred_baseline_risk_inside_mean", "baseline_inside_mean")]:
            if col in sub.columns:
                plt.plot(sub["year"], sub[col], marker="o", label=label)
        plt.title(f"{area}: inside-mask mean score")
        plt.xlabel("year")
        plt.ylabel("value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{area_slug}_inside_mask_timeseries.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        for col, label in [("pred_ml_final_lift", "ml_final_lift"), ("pred_baseline_risk_lift", "baseline_lift")]:
            if col in sub.columns:
                plt.plot(sub["year"], sub[col], marker="o", label=label)
        plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
        plt.title(f"{area}: lift inside literature mask")
        plt.xlabel("year")
        plt.ylabel("lift")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{area_slug}_inside_mask_lift.png", dpi=160)
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--scored-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--masks-geojson")
    ap.add_argument("--top-quantile", type=float)
    ap.add_argument("--min-inside", type=int)
    ap.add_argument("--min-outside", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()

    out_dir = safe_mkdir(Path(args.out_dir))
    df = pd.read_csv(args.scored_csv)
    internal = _make_internal_metrics(df, out_dir)

    top_quantile = 0.1
    min_inside = 3
    min_outside = 10
    if args.config:
        from _common import load_config
        cfg = load_config(args.config)
        top_quantile = float(cfg.get("validation", {}).get("top_quantile", top_quantile))
        min_inside = int(cfg.get("validation", {}).get("min_inside", min_inside))
        min_outside = int(cfg.get("validation", {}).get("min_outside", min_outside))
    if args.top_quantile is not None:
        top_quantile = float(args.top_quantile)
    if args.min_inside is not None:
        min_inside = int(args.min_inside)
    if args.min_outside is not None:
        min_outside = int(args.min_outside)

    external = None
    warnings = []
    if args.masks_geojson:
        gdf = _join_geometries(args.results_dir, df)
        masks = read_geo(args.masks_geojson)
        external, warnings = _external_stats(gdf, masks, out_dir, top_quantile, min_inside, min_outside)
        if not external.empty:
            _plot_timeseries(external, out_dir)

    report = {
        "internal_metrics_csv": str(out_dir / "internal_model_vs_baseline_metrics.csv") if not internal.empty else None,
        "external_metrics_csv": str(out_dir / "external_mask_validation.csv") if external is not None and not external.empty else None,
        "external_warnings_json": str(out_dir / "external_mask_validation_warnings.json") if args.masks_geojson else None,
        "rows_scored": int(len(df)),
        "top_quantile": top_quantile,
        "min_inside": min_inside,
        "min_outside": min_outside,
        "n_external_warnings": int(len(warnings)),
    }
    save_json(out_dir / "validation_report.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
