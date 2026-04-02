#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_config, minmax01, quantile_by_group, read_geo, safe_mkdir, save_json


NEG_CURRENT_FEATURES = [
    "change_share",
    "texture_anomaly_share",
    "water_share",
    "persistence_water_share",
]


def _abs_quantile_by_area(df: pd.DataFrame, feature: str, q: float) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df.groupby("area")[feature].transform(
        lambda s: s.abs().quantile(q) if s.notna().any() else np.nan
    )


def _safe_bool(s: pd.Series | None, index: pd.Index) -> pd.Series:
    if s is None:
        return pd.Series(False, index=index)
    return s.fillna(False).astype(bool)


def _attach_literature_hits(df: pd.DataFrame, masks_path: str | None) -> pd.DataFrame:
    out = df.copy()
    out["literature_mask_hit"] = False
    if not masks_path or "centroid_lon" not in out.columns or "centroid_lat" not in out.columns:
        return out
    try:
        import geopandas as gpd
    except Exception:
        return out
    pts = gpd.GeoDataFrame(
        out[["area", "parcel_id"]].copy(),
        geometry=gpd.points_from_xy(out["centroid_lon"], out["centroid_lat"]),
        crs="EPSG:4326",
    )
    masks = read_geo(masks_path).to_crs("EPSG:4326")
    joined = gpd.sjoin(pts, masks[["geometry"]], predicate="intersects", how="left")
    out["literature_mask_hit"] = joined["index_right"].notna().to_numpy()
    return out


def _build_positive_conditions(df: pd.DataFrame, qh: float) -> dict[str, pd.Series]:
    cond: dict[str, pd.Series] = {}
    if "change_share" in df.columns:
        cond["high_change"] = df["change_share"] >= quantile_by_group(df, "area", "change_share", qh)
    else:
        cond["high_change"] = pd.Series(False, index=df.index)

    if "texture_anomaly_share" in df.columns:
        cond["high_texture"] = df["texture_anomaly_share"] >= quantile_by_group(df, "area", "texture_anomaly_share", qh)
    else:
        cond["high_texture"] = pd.Series(False, index=df.index)

    if "water_share" in df.columns:
        water_hi = df["water_share"] >= quantile_by_group(df, "area", "water_share", qh)
        if "persistence_water_share" in df.columns:
            persist_hi = df["persistence_water_share"] >= quantile_by_group(df, "area", "persistence_water_share", qh)
            cond["high_water"] = water_hi | persist_hi
        else:
            cond["high_water"] = water_hi
    else:
        cond["high_water"] = pd.Series(False, index=df.index)

    if "delta_ndvi_mean" in df.columns:
        cond["veg_loss"] = df["delta_ndvi_mean"] <= quantile_by_group(df, "area", "delta_ndvi_mean", 1 - qh)
    else:
        cond["veg_loss"] = pd.Series(False, index=df.index)

    if "delta_ndwi_mean" in df.columns:
        cond["wet_shift"] = df["delta_ndwi_mean"] >= quantile_by_group(df, "area", "delta_ndwi_mean", qh)
    else:
        cond["wet_shift"] = pd.Series(False, index=df.index)

    if "roll2__change_share" in df.columns:
        cond["persistent_signal"] = df["roll2__change_share"] >= quantile_by_group(df, "area", "roll2__change_share", 0.75)
    else:
        cond["persistent_signal"] = pd.Series(False, index=df.index)

    return cond


def _build_negative_conditions(df: pd.DataFrame, cfg_labels: dict) -> tuple[dict[str, pd.Series], pd.Series, pd.Series, pd.Series]:
    qlow = float(cfg_labels["negative_current_quantile"])
    qabs = float(cfg_labels["negative_abs_delta_quantile"])

    cond: dict[str, pd.Series] = {}
    if "change_share" in df.columns:
        cond["low_change"] = df["change_share"] <= quantile_by_group(df, "area", "change_share", qlow)
    else:
        cond["low_change"] = pd.Series(False, index=df.index)

    if "texture_anomaly_share" in df.columns:
        cond["low_texture"] = df["texture_anomaly_share"] <= quantile_by_group(df, "area", "texture_anomaly_share", qlow)
    else:
        cond["low_texture"] = pd.Series(False, index=df.index)

    if "water_share" in df.columns:
        low_water = df["water_share"] <= quantile_by_group(df, "area", "water_share", qlow)
        if "persistence_water_share" in df.columns:
            low_persist = df["persistence_water_share"] <= quantile_by_group(df, "area", "persistence_water_share", qlow)
            cond["low_water"] = low_water & low_persist
        else:
            cond["low_water"] = low_water
    else:
        cond["low_water"] = pd.Series(False, index=df.index)

    if "delta_ndvi_mean" in df.columns:
        thr_ndvi = _abs_quantile_by_area(df, "delta_ndvi_mean", qabs)
        cond["stable_ndvi"] = df["delta_ndvi_mean"].isna() | (df["delta_ndvi_mean"].abs() <= thr_ndvi)
    else:
        cond["stable_ndvi"] = pd.Series(True, index=df.index)

    if "delta_ndwi_mean" in df.columns:
        thr_ndwi = _abs_quantile_by_area(df, "delta_ndwi_mean", qabs)
        cond["stable_ndwi"] = df["delta_ndwi_mean"].isna() | (df["delta_ndwi_mean"].abs() <= thr_ndwi)
    else:
        cond["stable_ndwi"] = pd.Series(True, index=df.index)

    if bool(cfg_labels.get("negative_require_low_hotspot", True)) and "hotspot_share" in df.columns:
        cond["low_hotspot"] = df["hotspot_share"].fillna(0) <= quantile_by_group(df, "area", "hotspot_share", 0.5)
    else:
        cond["low_hotspot"] = pd.Series(True, index=df.index)

    row_feature_ready = pd.Series(0, index=df.index, dtype=float)
    for c in NEG_CURRENT_FEATURES:
        if c in df.columns:
            row_feature_ready = row_feature_ready + df[c].notna().astype(int)
    row_feature_ready = row_feature_ready >= max(3, min(4, len([c for c in NEG_CURRENT_FEATURES if c in df.columns])))

    current_stable = (
        cond["low_change"]
        & cond["low_texture"]
        & cond["low_water"]
        & cond["stable_ndvi"]
        & cond["stable_ndwi"]
        & cond["low_hotspot"]
        & row_feature_ready
    )

    parcel_obs = df.assign(_ready=row_feature_ready).groupby(["area", "parcel_id"])["_ready"].transform("sum")
    parcel_stable_count = df.assign(_stable=current_stable).groupby(["area", "parcel_id"])["_stable"].transform("sum")
    parcel_stable_share = parcel_stable_count / parcel_obs.replace(0, np.nan)

    return cond, current_stable, parcel_obs, parcel_stable_share


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config")
    ap.add_argument("--masks-geojson")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lc = cfg["labels"]
    out_dir = safe_mkdir(Path(args.out_dir))
    df = pd.read_csv(args.table_csv)
    df = _attach_literature_hits(df, args.masks_geojson)

    qh = float(lc["positive_quantile"])
    pos_cond = _build_positive_conditions(df, qh)
    neg_cond, current_stable, parcel_obs, parcel_stable_share = _build_negative_conditions(df, lc)

    pos_hits = sum(pos_cond[k].astype(int) for k in ["high_change", "high_texture", "high_water", "veg_loss", "wet_shift", "persistent_signal"])
    min_hits = int(lc["min_positive_hits"])
    positive = (pos_hits >= min_hits) & (pos_cond["high_change"] | pos_cond["high_texture"] | pos_cond["high_water"])
    if bool(lc.get("require_dynamic_signal", True)):
        positive = positive & (pos_cond["veg_loss"] | pos_cond["wet_shift"] | pos_cond["persistent_signal"])

    negative = current_stable & (parcel_obs >= int(lc["negative_min_obs"])) & (parcel_stable_share >= float(lc["negative_min_stable_share"]))
    if bool(lc.get("negative_require_no_literature_hit", True)):
        negative = negative & (~_safe_bool(df.get("literature_mask_hit"), df.index))

    if "n_missing_primary" in df.columns:
        max_missing = int(lc.get("max_missing_primary_for_label", 999))
        eligible = pd.to_numeric(df["n_missing_primary"], errors="coerce").fillna(max_missing + 1) <= max_missing
        positive = positive & eligible
        negative = negative & eligible
    else:
        eligible = pd.Series(True, index=df.index)

    target = pd.Series(np.nan, index=df.index)
    label_type = pd.Series("uncertain", index=df.index)
    target[positive] = 1
    label_type[positive] = "confident_positive"
    target[negative] = 0
    label_type[negative] = "confident_negative"
    target[positive & negative] = np.nan
    label_type[positive & negative] = "conflict"

    sample_weight = pd.Series(1.0, index=df.index)
    sample_weight[positive] = (1.0 + 0.10 * (pos_hits[positive] - min_hits).clip(lower=0)).clip(upper=1.5)
    literature_hit = _safe_bool(df.get("literature_mask_hit"), df.index)
    sample_weight[positive & literature_hit] = sample_weight[positive & literature_hit] + float(lc.get("literature_bonus_weight", 0.1))
    sample_weight[negative] = 1.1

    df = df.copy()
    df["label_is_eligible"] = eligible.astype(bool)
    df["positive_rule_hits"] = pos_hits
    df["negative_current_stable"] = current_stable.astype(bool)
    df["negative_parcel_obs"] = pd.to_numeric(parcel_obs, errors="coerce")
    df["negative_parcel_stable_share"] = pd.to_numeric(parcel_stable_share, errors="coerce")
    df["weak_target"] = target
    df["label_type"] = label_type
    df["sample_weight"] = sample_weight
    df["baseline_risk"] = minmax01(df["risk_score_mean"]) if "risk_score_mean" in df.columns else np.nan

    out_csv = out_dir / "parcel_year_labeled.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "settings": lc,
        "counts": df["label_type"].value_counts(dropna=False).to_dict(),
        "by_area": df.groupby(["area", "label_type"]).size().unstack(fill_value=0).to_dict(orient="index") if "area" in df.columns else {},
        "eligible_rows": int(df["label_is_eligible"].sum()),
        "negative_parcel_stats": {
            "median_obs": float(pd.Series(parcel_obs).dropna().median()) if pd.Series(parcel_obs).notna().any() else None,
            "median_stable_share": float(pd.Series(parcel_stable_share).dropna().median()) if pd.Series(parcel_stable_share).notna().any() else None,
        },
        "csv": str(out_csv),
    }
    save_json(out_dir / "weak_label_report.json", summary)
    print({"csv": str(out_csv), "counts": summary["counts"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
