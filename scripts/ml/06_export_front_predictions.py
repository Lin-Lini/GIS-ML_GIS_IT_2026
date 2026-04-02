#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _common import prepare_parcels_with_area, safe_mkdir, save_json, slugify_area


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--scored-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = safe_mkdir(Path(args.out_dir))
    df = pd.read_csv(args.scored_csv)
    df["parcel_id"] = df["parcel_id"].astype(str)

    parcels = prepare_parcels_with_area(args.results_dir)
    p = parcels[[c for c in ["parcel_id", "area", "geometry"] if c in parcels.columns]].copy()
    p["parcel_id"] = p["parcel_id"].astype(str)

    items = []
    sanity = []
    for (area, year), sub in df.groupby(["area", "year"], dropna=False):
        area_slug = slugify_area(str(area))
        p_area = p[p["area"] == area].copy()
        merged = p_area.merge(sub, on=["parcel_id", "area"], how="inner")
        gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=parcels.crs)

        rank_sources = [c for c in ["pred_ml_final", "pred_ml_mean", "pred_catboost", "pred_logit", "pred_baseline_risk"] if c in gdf.columns]
        for c in rank_sources:
            gdf[f"{c}_rank_pct"] = gdf[c].rank(pct=True, method="average")

        if "pred_ml_final" in gdf.columns and "pred_baseline_risk" in gdf.columns:
            gdf["ml_minus_baseline"] = gdf["pred_ml_final"] - gdf["pred_baseline_risk"]
            gdf["ml_vs_baseline_abs_gap"] = gdf["ml_minus_baseline"].abs()
            gdf["ml_attention_flag"] = (
                (gdf["pred_ml_final_rank_pct"] >= 0.9) | (gdf["pred_baseline_risk_rank_pct"] >= 0.9)
            ) & (gdf["ml_vs_baseline_abs_gap"] >= 0.15)
        else:
            gdf["ml_attention_flag"] = False

        if "pred_logit" in gdf.columns and "pred_catboost" in gdf.columns:
            gdf["logit_catboost_disagreement"] = (gdf["pred_logit"] - gdf["pred_catboost"]).abs()

        folder = safe_mkdir(out_dir / "areas" / area_slug)
        csv_path = folder / f"{area_slug}_{int(year)}_parcel_ml_scores.csv"
        geojson_path = folder / f"{area_slug}_{int(year)}_parcel_ml_scores.geojson"

        preferred = [
            "parcel_id",
            "pred_ml_final",
            "pred_ml_mean",
            "pred_logit",
            "pred_catboost",
            "pred_baseline_risk",
            "ml_minus_baseline",
            "ml_vs_baseline_abs_gap",
            "logit_catboost_disagreement",
            "ml_attention_flag",
            "cluster_id",
            "cluster_prob",
            "weak_target",
            "label_type",
            "sample_weight",
        ]
        keep = [c for c in preferred if c in gdf.columns]
        keep += [c for c in gdf.columns if c.endswith("_rank_pct") and c not in keep]

        gdf[keep].to_csv(csv_path, index=False, encoding="utf-8-sig")
        gdf[[*keep, "geometry"]].to_file(geojson_path, driver="GeoJSON")
        items.append({"area": area, "year": int(year), "rows": int(len(gdf)), "csv": str(csv_path), "geojson": str(geojson_path)})
        sanity.append({
            "area": str(area),
            "year": int(year),
            "rows_scored": int(len(sub)),
            "rows_exported": int(len(gdf)),
            "unique_parcels_exported": int(gdf["parcel_id"].nunique()),
        })

    save_json(out_dir / "frontend_ml_export_manifest.json", {"items": items, "sanity": sanity})
    print({"items": len(items), "manifest": str(out_dir / 'frontend_ml_export_manifest.json')})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
