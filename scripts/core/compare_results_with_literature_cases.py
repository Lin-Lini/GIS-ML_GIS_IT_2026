#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
import matplotlib.pyplot as plt

def read_band(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        return arr, ds.transform, ds.crs

def zonal_stats(arr, transform, geoms):
    mask = features.geometry_mask(geoms, out_shape=arr.shape, transform=transform, invert=True)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(np.max(vals)),
    }

def discover_years(results_dir: Path, area_ru: str):
    pat = f"{area_ru}_*_risk_score.tif"
    years = []
    for p in sorted((results_dir / "dynamics").glob(pat)):
        try:
            year = int(p.stem.split("_")[1])
            years.append(year)
        except Exception:
            pass
    return years

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--cases-geojson", required=True)
    ap.add_argument("--masks-geojson", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--point-buffer-m", type=float, default=250.0, help="Buffer around evidence points for point-level sampling")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = gpd.read_file(args.cases_geojson)
    masks = gpd.read_file(args.masks_geojson)

    rows = []

    for area_ru in sorted(cases["area_ru"].unique()):
        years = discover_years(results_dir, area_ru)
        if not years:
            continue

        area_cases = cases[cases["area_ru"] == area_ru].copy()
        area_masks = masks[masks["area_ru"] == area_ru].copy()

        for year in years:
            risk_path = results_dir / "dynamics" / f"{area_ru}_{year}_risk_score.tif"
            hot_path = results_dir / "masks" / f"{area_ru}_{year}_hotspot_mask.tif"
            wocc_path = results_dir / "dynamics" / f"{area_ru}_{year}_water_occurrence.tif"

            risk, transform, crs = read_band(risk_path)
            hotspot, _, _ = read_band(hot_path)
            water_occ, _, _ = read_band(wocc_path)

            if area_masks.crs != crs:
                area_masks_proj = area_masks.to_crs(crs)
            else:
                area_masks_proj = area_masks

            if area_cases.crs != crs:
                area_cases_proj = area_cases.to_crs(crs)
            else:
                area_cases_proj = area_cases

            for _, row in area_masks_proj.iterrows():
                stats_risk = zonal_stats(risk, transform, [row.geometry])
                stats_hot = zonal_stats(hotspot, transform, [row.geometry])
                stats_wat = zonal_stats(water_occ, transform, [row.geometry])
                rows.append({
                    "evidence_type": "mask",
                    "case_id": row["mask_id"],
                    "area_ru": area_ru,
                    "year": year,
                    "source_area_km2": row["source_area_km2"],
                    "risk_mean": stats_risk["mean"],
                    "risk_p90": stats_risk["p90"],
                    "risk_max": stats_risk["max"],
                    "hotspot_share": stats_hot["mean"],
                    "water_occurrence_mean": stats_wat["mean"],
                })

            # point buffers
            for _, crow in area_cases_proj.iterrows():
                geom = crow.geometry.buffer(args.point_buffer_m)
                stats_risk = zonal_stats(risk, transform, [geom])
                stats_hot = zonal_stats(hotspot, transform, [geom])
                stats_wat = zonal_stats(water_occ, transform, [geom])
                rows.append({
                    "evidence_type": "point_buffer",
                    "case_id": crow["case_id"],
                    "area_ru": area_ru,
                    "year": year,
                    "source_area_km2": np.nan,
                    "risk_mean": stats_risk["mean"],
                    "risk_p90": stats_risk["p90"],
                    "risk_max": stats_risk["max"],
                    "hotspot_share": stats_hot["mean"],
                    "water_occurrence_mean": stats_wat["mean"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No comparable layers found.")

    csv_path = out_dir / "literature_validation_stats.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Slide-friendly charts
    for area_ru in sorted(df["area_ru"].unique()):
        dfa = df[(df["area_ru"] == area_ru) & (df["evidence_type"] == "mask")].copy()
        if dfa.empty:
            continue
        x = dfa["year"].astype(int).tolist()
        y1 = dfa["risk_mean"].tolist()
        y2 = dfa["hotspot_share"].tolist()

        plt.figure(figsize=(8, 4.5))
        plt.plot(x, y1, marker="o", label="risk_mean")
        plt.plot(x, y2, marker="s", label="hotspot_share")
        plt.title(f"{area_ru}: literature-derived validation mask vs model outputs")
        plt.xlabel("year")
        plt.ylabel("value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{area_ru}_validation_mask_timeseries.png", dpi=180)
        plt.close()

    summary = {
        "status": "ok",
        "csv": str(csv_path),
        "rows": int(len(df)),
        "charts_dir": str(out_dir),
    }
    (out_dir / "run_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
