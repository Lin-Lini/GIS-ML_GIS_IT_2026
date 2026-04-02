from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

AREA_SLUG = {"Амга": "amga", "Юнкор": "yunkor", "Юнкюр": "yunkyur", "amga": "amga", "yunkor": "yunkor", "yunkyur": "yunkyur"}
AREA_CANON = {
    "амга": "Амга",
    "amga": "Амга",
    "юнкор": "Юнкор",
    "yunkor": "Юнкор",
    "юнкюр": "Юнкор",
    "yunkyur": "Юнкор",
}
AN_RE = re.compile(r"^(?P<area>[^_]+)_(?P<year>\d{4})_parcel_stats\.csv$", re.IGNORECASE)
VALID_RE = re.compile(r"^(?P<area>[^_]+)_(?P<year>\d{4})_parcel_stats_valid\.csv$", re.IGNORECASE)

FEATURE_REGEX = {
    "parcel_id": [r"^parcel_id$", r"^id$", r"^cad_num$"],
    "risk_score_mean": [r"^risk_score_mean$", r"^risk_mean$", r"risk.*mean"],
    "hotspot_share": [r"^hotspot_share$", r"hotspot.*share"],
    "water_occurrence_mean": [r"^water_occurrence_mean$", r"water_occurrence.*mean"],
    "ndvi_mean": [r"^ndvi_mean$", r"ndvi.*mean", r"ndvi.*avg"],
    "ndwi_mean": [r"^ndwi_mean$", r"ndwi.*mean", r"ndwi.*avg"],
    "osavi_mean": [r"^osavi_mean$", r"osavi.*mean", r"savi.*mean"],
    "brightness_mean": [r"^brightness_mean$", r"brightness.*mean"],
    "nir_red_ratio_mean": [r"^nir_red_ratio_mean$", r"nir_red_ratio.*mean", r"nir.*red.*ratio.*mean"],
    "red_green_ratio_mean": [r"^red_green_ratio_mean$", r"red_green_ratio.*mean", r"red.*green.*ratio.*mean"],
    "water_share": [r"^water_share$", r"water.*share$"],
    "persistence_water_share": [r"^persistence_water_share$", r"persistence.*water.*share"],
    "change_share": [r"^change_share$", r"change.*share$"],
    "texture_anomaly_share": [r"^texture_anomaly_share$", r"texture.*anomaly.*share"],
    "delta_ndvi_mean": [r"^delta_ndvi_mean$", r"delta_ndvi.*mean", r"d1__ndvi_mean$"],
    "delta_ndwi_mean": [r"^delta_ndwi_mean$", r"delta_ndwi.*mean", r"d1__ndwi_mean$"],
    "slope_mean": [r"^slope_mean$", r"slope.*mean"],
    "tpi_mean": [r"^tpi_mean$", r"tpi.*mean"],
    "roughness_mean": [r"^roughness_mean$", r"roughness.*mean"],
    "curvature_mean": [r"^curvature_mean$", r"curvature.*mean"],
    "tri_mean": [r"^tri_mean$", r"tri.*mean"],
    "dem_mean": [r"^dem_mean$", r"dem.*mean"],
    "aspect_sin_mean": [r"^aspect_sin_mean$", r"aspect_sin.*mean"],
    "aspect_cos_mean": [r"^aspect_cos_mean$", r"aspect_cos.*mean"],
    "area_m2": [r"^area_m2$", r"^shape_area$", r"geom_area"],
    "cad_num": [r"^cad_num$"],
}

PRIMARY_CURRENT = [
    "ndvi_mean", "ndwi_mean", "osavi_mean", "brightness_mean", "nir_red_ratio_mean", "red_green_ratio_mean",
    "water_share", "persistence_water_share", "change_share", "texture_anomaly_share", "water_occurrence_mean",
    "slope_mean", "tpi_mean", "roughness_mean", "curvature_mean", "tri_mean", "dem_mean"
]
PRIMARY_DYNAMIC = ["delta_ndvi_mean", "delta_ndwi_mean"]
BASELINE_ONLY = ["risk_score_mean", "hotspot_share"]


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonicalize_area(area: Any) -> str | None:
    if area is None or (isinstance(area, float) and np.isnan(area)):
        return None
    txt = str(area).strip()
    if not txt:
        return None
    key = txt.lower().replace("ё", "е")
    return AREA_CANON.get(key, txt)


def slugify_area(area: str) -> str:
    canon = canonicalize_area(area) or str(area)
    return AREA_SLUG.get(canon, re.sub(r"[^A-Za-z0-9]+", "_", canon).strip("_").lower() or "area")


def load_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    cfg = load_json(here / "config.default.json", default={})
    if path:
        cfg = deep_update(cfg, load_json(path, default={}))
    return cfg


def parse_area_year_from_filename(name: str) -> tuple[str, int] | None:
    m = AN_RE.match(name)
    if not m:
        return None
    return canonicalize_area(m.group("area")) or m.group("area"), int(m.group("year"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip()).replace("%", "pct") for c in out.columns]
    return out


def _find_best_match(cols: list[str], patterns: list[str]) -> str | None:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        exact = [c for c in cols if rx.fullmatch(c)]
        if exact:
            return exact[0]
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        partial = [c for c in cols if rx.search(c)]
        if partial:
            return sorted(partial, key=lambda x: ("mean" not in x.lower(), len(x)))[0]
    return None


def apply_feature_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = normalize_columns(df)
    cols = list(out.columns)
    mapping = {}
    for canon, pats in FEATURE_REGEX.items():
        if canon in out.columns:
            mapping[canon] = canon
            continue
        found = _find_best_match(cols, pats)
        if found is not None:
            out[canon] = out[found]
            mapping[canon] = found
    return out, mapping


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _coerce_boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "yes", "да", "y"})


def load_valid_flags(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    parts = []
    for p in sorted((results_dir / "analytics").glob("*_parcel_stats_valid.csv")):
        m = VALID_RE.match(p.name)
        if not m:
            continue
        area, year = canonicalize_area(m.group("area")) or m.group("area"), int(m.group("year"))
        df = normalize_columns(pd.read_csv(p))
        pid = next((c for c in ["parcel_id", "cad_num", "id"] if c in df.columns), None)
        val = next((c for c in ["is_valid_for_full_analytics", "is_valid", "valid", "ok"] if c in df.columns), None)
        if pid is None or val is None:
            continue
        keep = df[[pid, val]].copy()
        keep.columns = ["parcel_id", "is_valid_for_full_analytics"]
        keep["parcel_id"] = keep["parcel_id"].astype(str)
        keep["is_valid_for_full_analytics"] = _coerce_boolish(keep["is_valid_for_full_analytics"])
        keep["area"] = area
        keep["year"] = year
        parts.append(keep)
    if not parts:
        return pd.DataFrame(columns=["area", "year", "parcel_id", "is_valid_for_full_analytics"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["area", "year", "parcel_id"])


def load_parcel_stats(results_dir: str | Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    results_dir = Path(results_dir)
    parts = []
    aliases = {}
    for p in sorted((results_dir / "analytics").glob("*_parcel_stats.csv")):
        parsed = parse_area_year_from_filename(p.name)
        if parsed is None:
            continue
        area, year = parsed
        df, amap = apply_feature_aliases(pd.read_csv(p))
        aliases[p.name] = amap
        if "parcel_id" not in df.columns:
            raise ValueError(f"В {p.name} не найден parcel_id")
        df["parcel_id"] = df["parcel_id"].astype(str)
        df["area"] = area
        df["year"] = year
        parts.append(df)
    if not parts:
        raise FileNotFoundError("Не найдены *_parcel_stats.csv")
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = _coerce_numeric(out, list(set(PRIMARY_CURRENT + PRIMARY_DYNAMIC + BASELINE_ONLY + ["area_m2", "aspect_sin_mean", "aspect_cos_mean"])))
    valid = load_valid_flags(results_dir)
    if not valid.empty:
        out = out.merge(valid, on=["area", "year", "parcel_id"], how="left")
    else:
        out["is_valid_for_full_analytics"] = np.nan
    return out, aliases


def read_geo(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if str(path).lower().endswith(".zip"):
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _pick_area_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [c for c in ["area", "area_name", "name", "zone", "label"] if c in gdf.columns]
    if candidates:
        return candidates[0]
    for c in gdf.columns:
        if c == "geometry":
            continue
        vals = gdf[c].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        uniq = set(vals.str.lower().str.replace("ё", "е"))
        if uniq & {"амга", "юнкор", "юнкюр", "amga", "yunkor", "yunkyur"}:
            return c
    raise ValueError("В area_aoi не найден столбец с названием зоны")


def load_area_aoi(results_dir: str | Path) -> gpd.GeoDataFrame:
    results_dir = Path(results_dir)
    candidates = [
        results_dir / "aoi" / "area_aoi.gpkg",
        results_dir / "aoi" / "area_aoi.geojson",
        results_dir / "areas" / "area_aoi.gpkg",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Не найден area_aoi.gpkg/geojson в results/aoi")
    gdf = normalize_columns(read_geo(path))
    area_col = _pick_area_column(gdf)
    gdf = gdf[[area_col, "geometry"]].copy().rename(columns={area_col: "area"})
    gdf["area"] = gdf["area"].map(canonicalize_area)
    gdf = gdf[gdf["area"].notna()].copy()
    if gdf.empty:
        raise ValueError("В area_aoi нет распознанных зон Амга/Юнкор")
    return gdf


def prepare_parcels_with_area(results_dir: str | Path) -> gpd.GeoDataFrame:
    results_dir = Path(results_dir)
    path = results_dir / "parcels" / "parcels_clipped.gpkg"
    parcels = normalize_columns(read_geo(path))
    pid = next((c for c in ["parcel_id", "cad_num", "id"] if c in parcels.columns), None)
    if pid is None:
        raise ValueError("В parcels_clipped.gpkg не найден parcel_id")
    keep = [pid, "geometry"]
    for c in ["area_m2", "cad_num"]:
        if c in parcels.columns and c not in keep:
            keep.append(c)
    parcels = parcels[keep].copy().rename(columns={pid: "parcel_id"})
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)

    aoi = load_area_aoi(results_dir)
    if parcels.crs != aoi.crs:
        aoi = aoi.to_crs(parcels.crs)

    probe = parcels.copy()
    probe["geometry"] = probe.geometry.representative_point()
    joined = gpd.sjoin(probe[["parcel_id", "geometry"]], aoi[["area", "geometry"]], how="left", predicate="within")
    if joined["area"].isna().any():
        miss = joined[joined["area"].isna()][["parcel_id", "geometry"]]
        if len(miss):
            miss2 = gpd.sjoin(miss, aoi[["area", "geometry"]], how="left", predicate="intersects")
            recovered = miss2[["parcel_id", "area"]].dropna().drop_duplicates("parcel_id")
            joined = joined.drop(columns=["area"]).merge(recovered, on="parcel_id", how="left")

    area_map = joined[["parcel_id", "area"]].dropna().drop_duplicates("parcel_id")
    parcels = parcels.merge(area_map, on="parcel_id", how="left")

    try:
        gg = parcels
        if gg.crs is not None and getattr(gg.crs, "is_geographic", False):
            gg = gg.to_crs(3857)
        cen = gg.geometry.centroid
        cen = gpd.GeoSeries(cen, crs=gg.crs).to_crs(4326)
        parcels["centroid_lon"] = cen.x
        parcels["centroid_lat"] = cen.y
    except Exception:
        pass

    return parcels


def enforce_area_consistency(results_dir: str | Path, df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    parcels = prepare_parcels_with_area(results_dir)
    lookup_cols = [c for c in ["parcel_id", "area", "area_m2", "cad_num", "centroid_lon", "centroid_lat"] if c in parcels.columns]
    lookup = pd.DataFrame(parcels[lookup_cols]).copy().rename(columns={"area": "area_from_geometry"})

    out = df.copy()
    out["parcel_id"] = out["parcel_id"].astype(str)
    out["area"] = out["area"].map(canonicalize_area)
    out = out.merge(lookup, on="parcel_id", how="left")

    if "area_m2" not in out.columns and "area_m2_y" in out.columns:
        out["area_m2"] = out["area_m2_y"]
    elif "area_m2_x" in out.columns and "area_m2_y" in out.columns:
        out["area_m2"] = out["area_m2_x"].where(out["area_m2_x"].notna(), out["area_m2_y"])

    if "cad_num" not in out.columns and "cad_num_y" in out.columns:
        out["cad_num"] = out["cad_num_y"]
    elif "cad_num_x" in out.columns and "cad_num_y" in out.columns:
        out["cad_num"] = out["cad_num_x"].where(out["cad_num_x"].notna(), out["cad_num_y"])

    mismatch = out["area_from_geometry"].notna() & (out["area"] != out["area_from_geometry"])
    missing_geom_area = out["area_from_geometry"].isna().sum()
    dropped = int(mismatch.sum())
    if strict:
        out = out[~mismatch].copy()
        if out["area_from_geometry"].notna().any():
            out["area"] = out["area_from_geometry"].where(out["area_from_geometry"].notna(), out["area"])
    else:
        out["area"] = out["area_from_geometry"].where(out["area_from_geometry"].notna(), out["area"])

    info = {
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "dropped_area_mismatch": dropped,
        "rows_without_geometry_area": int(missing_geom_area),
        "unique_parcels_by_area": {str(k): int(v) for k, v in out.groupby("area")["parcel_id"].nunique().to_dict().items()},
    }
    return out, info


def try_join_parcels(results_dir: str | Path, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out, info = enforce_area_consistency(results_dir, df, strict=True)
    return out, info


def add_temporal_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy().sort_values(["area", "parcel_id", "year"]).reset_index(drop=True)
    grp = out.groupby(["area", "parcel_id"], sort=False)
    for c in features:
        if c not in out.columns:
            continue
        out[f"prev1__{c}"] = grp[c].shift(1)
        out[f"d1__{c}"] = out[c] - out[f"prev1__{c}"]
        out[f"roll2__{c}"] = grp[c].transform(lambda s: s.rolling(2, min_periods=1).mean())
    out["prev_year"] = grp["year"].shift(1)
    out["year_gap"] = out["year"] - out["prev_year"]
    if "area_m2" in out.columns:
        out["log_area_m2"] = np.log1p(pd.to_numeric(out["area_m2"], errors="coerce").clip(lower=0))
    out["n_missing_primary"] = out[[c for c in features if c in out.columns]].isna().sum(axis=1)
    return out


def get_feature_inventory(df: pd.DataFrame) -> dict[str, list[str]]:
    current = [c for c in PRIMARY_CURRENT if c in df.columns]
    dynamic = [c for c in PRIMARY_DYNAMIC if c in df.columns]
    baseline = [c for c in BASELINE_ONLY if c in df.columns]
    temporal = [c for c in df.columns if c.startswith(("prev1__", "d1__", "roll2__"))]
    misc = [c for c in ["log_area_m2", "year_gap", "n_missing_primary", "area_m2", "aspect_sin_mean", "aspect_cos_mean", "is_valid_for_full_analytics", "centroid_lon", "centroid_lat"] if c in df.columns]
    return {"current": current, "dynamic": dynamic, "baseline": baseline, "temporal": temporal, "misc": misc}


def quantile_by_group(df: pd.DataFrame, group_col: str, feature: str, q: float) -> pd.Series:
    return df.groupby(group_col)[feature].transform(lambda s: s.quantile(q) if s.notna().any() else np.nan)


def robust_clip_prob(x):
    arr = np.asarray(x, dtype=float)
    return np.clip(arr, 1e-6, 1 - 1e-6)


def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    lo = float(np.nanmin(s))
    hi = float(np.nanmax(s))
    if math.isclose(lo, hi):
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def available_training_features(df: pd.DataFrame, include_cluster: bool = True, include_baseline: bool = False):
    inv = get_feature_inventory(df)
    numeric = inv["current"] + inv["dynamic"] + inv["temporal"] + [c for c in inv["misc"] if c not in {"is_valid_for_full_analytics", "centroid_lon", "centroid_lat"}]
    if include_baseline:
        numeric += inv["baseline"]
    if include_cluster:
        for c in ["cluster_prob", "cluster_outlier"]:
            if c in df.columns:
                numeric.append(c)
    seen = set()
    numeric = [x for x in numeric if not (x in seen or seen.add(x))]
    cat = [c for c in ["area"] if c in df.columns]
    if include_cluster and "cluster_id" in df.columns:
        cat.append("cluster_id")
    return numeric, cat
