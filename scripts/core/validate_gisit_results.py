#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry
except Exception:  # pragma: no cover
    gpd = None
    BaseGeometry = object

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover
    rasterio = None
    transform_bounds = None

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "ERROR": 2}

EXPECTED_SUBDIRS = [
    "analytics",
    "aoi",
    "catalog",
    "composites",
    "dynamics",
    "indices",
    "masks",
    "parcels",
    "terrain",
    "textures",
]

RASTER_EXTS = {".tif", ".tiff"}
VECTOR_EXTS = {".gpkg", ".geojson", ".json", ".shp"}
TABLE_EXTS = {".csv"}

INDEX_FEATURES = {
    "brightness",
    "ndvi",
    "ndwi",
    "nir_red_ratio",
    "osavi",
    "red_green_ratio",
}
MASK_FEATURES = {
    "change_mask",
    "hotspot_mask",
    "persistence_water_mask",
    "texture_anomaly_mask",
    "water_mask",
}
TERRAIN_FEATURES = {
    "aspect_cos",
    "aspect_sin",
    "curvature",
    "dem",
    "roughness",
    "slope",
    "tpi",
    "tri",
}
TEXTURE_CHANNELS = {"nir", "pc1", "red"}
TEXTURE_FEATURES = {
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_entropy",
    "glcm_homogeneity",
    "local_std",
    "local_var",
}
TEXTURE_WINDOWS = {"w5", "w7"}
DYNAMIC_PAIR_FEATURES = {"delta_ndvi", "delta_ndwi", "water_growth"}
DYNAMIC_YEAR_FEATURES = {"risk_score", "water_occurrence"}

EXPECTED_PARCEL_COLUMNS = {"cad_num", "status", "c_cost", "area_m2", "utl_id", "utl_doc"}

RANGE_RULES = [
    (re.compile(r"(?:^|_)ndvi(?:_|\.)", re.IGNORECASE), -1.2, 1.2, "NDVI должен быть близок к диапазону [-1, 1]"),
    (re.compile(r"(?:^|_)ndwi(?:_|\.)", re.IGNORECASE), -1.2, 1.2, "NDWI должен быть близок к диапазону [-1, 1]"),
    (re.compile(r"(?:^|_)osavi(?:_|\.)", re.IGNORECASE), -1.2, 1.2, "OSAVI должен быть близок к диапазону [-1, 1]"),
    (re.compile(r"(?:^|_)aspect_cos(?:_|\.)", re.IGNORECASE), -1.05, 1.05, "aspect_cos должен быть в пределах [-1, 1]"),
    (re.compile(r"(?:^|_)aspect_sin(?:_|\.)", re.IGNORECASE), -1.05, 1.05, "aspect_sin должен быть в пределах [-1, 1]"),
    (re.compile(r"(?:^|_)delta_ndvi(?:_|\.)", re.IGNORECASE), -2.0, 2.0, "delta_ndvi не должен уходить в аномальные значения"),
    (re.compile(r"(?:^|_)delta_ndwi(?:_|\.)", re.IGNORECASE), -2.0, 2.0, "delta_ndwi не должен уходить в аномальные значения"),
    (re.compile(r"(?:^|_)glcm_homogeneity(?:_|\.)", re.IGNORECASE), 0.0, 1.1, "GLCM homogeneity обычно лежит в [0, 1]"),
]

NON_NEGATIVE_RULES = [
    (re.compile(r"(?:^|_)brightness(?:_|\.)", re.IGNORECASE), "brightness не должен быть отрицательным"),
    (re.compile(r"(?:^|_)dem(?:_|\.)", re.IGNORECASE), "DEM не должен быть отрицательным в рамках этой постановки"),
    (re.compile(r"(?:^|_)slope(?:_|\.)", re.IGNORECASE), "slope не должен быть отрицательным"),
    (re.compile(r"(?:^|_)roughness(?:_|\.)", re.IGNORECASE), "roughness не должен быть отрицательным"),
    (re.compile(r"(?:^|_)tri(?:_|\.)", re.IGNORECASE), "TRI не должен быть отрицательным"),
    (re.compile(r"(?:^|_)local_std(?:_|\.)", re.IGNORECASE), "local_std не должен быть отрицательным"),
    (re.compile(r"(?:^|_)local_var(?:_|\.)", re.IGNORECASE), "local_var не должен быть отрицательным"),
    (re.compile(r"(?:^|_)glcm_entropy(?:_|\.)", re.IGNORECASE), "GLCM entropy не должен быть отрицательным"),
    (re.compile(r"(?:^|_)glcm_contrast(?:_|\.)", re.IGNORECASE), "GLCM contrast не должен быть отрицательным"),
    (re.compile(r"(?:^|_)glcm_dissimilarity(?:_|\.)", re.IGNORECASE), "GLCM dissimilarity не должен быть отрицательным"),
    (re.compile(r"(?:^|_)water_occurrence(?:_|\.)", re.IGNORECASE), "water_occurrence не должен быть отрицательным"),
    (re.compile(r"(?:^|_)risk_score(?:_|\.)", re.IGNORECASE), "risk_score не должен быть отрицательным"),
    (re.compile(r"(?:^|_)water_growth(?:_|\.)", re.IGNORECASE), "water_growth не должен быть отрицательным"),
]

BINARY_HINT_RULES = [
    re.compile(r"(?:^|_)change_mask(?:_|\.)", re.IGNORECASE),
    re.compile(r"(?:^|_)hotspot_mask(?:_|\.)", re.IGNORECASE),
    re.compile(r"(?:^|_)persistence_water_mask(?:_|\.)", re.IGNORECASE),
    re.compile(r"(?:^|_)texture_anomaly_mask(?:_|\.)", re.IGNORECASE),
    re.compile(r"(?:^|_)water_mask(?:_|\.)", re.IGNORECASE),
]


@dataclass
class Issue:
    severity: str
    code: str
    path: str
    message: str
    details: Optional[Dict[str, Any]] = None


class Reporter:
    def __init__(self) -> None:
        self.issues: List[Issue] = []

    def add(self, severity: str, code: str, path: Path | str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.issues.append(Issue(severity=severity, code=code, path=str(path), message=message, details=details))

    def info(self, code: str, path: Path | str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.add("INFO", code, path, message, details)

    def warn(self, code: str, path: Path | str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.add("WARNING", code, path, message, details)

    def error(self, code: str, path: Path | str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.add("ERROR", code, path, message, details)

    def counts(self) -> Dict[str, int]:
        c = Counter(x.severity for x in self.issues)
        return {k: c.get(k, 0) for k in ["INFO", "WARNING", "ERROR"]}


@dataclass
class RasterMeta:
    path: str
    relpath: str
    folder: str
    site: Optional[str]
    year: Optional[int]
    year2: Optional[int]
    season: Optional[str]
    feature: Optional[str]
    crs: Optional[str]
    width: Optional[int]
    height: Optional[int]
    count: Optional[int]
    dtype: Optional[str]
    nodata: Any
    bounds: Optional[Tuple[float, float, float, float]]
    transform: Optional[Tuple[float, float, float, float, float, float]]
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    finite_fraction: Optional[float]
    sampled_pixels: int
    unique_preview: List[float]


@dataclass
class VectorMeta:
    path: str
    relpath: str
    folder: str
    feature_count: Optional[int]
    crs: Optional[str]
    geometry_types: List[str]
    invalid_count: Optional[int]
    empty_count: Optional[int]
    bounds: Optional[Tuple[float, float, float, float]]
    columns: List[str]
    duplicate_geometries: Optional[int]


@dataclass
class TableMeta:
    path: str
    relpath: str
    folder: str
    rows: Optional[int]
    cols: Optional[int]
    columns: List[str]
    duplicate_rows: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Полный аудит папки results для GIS'IT: проверка структуры, растров, векторов, таблиц и согласованности данных."
    )
    p.add_argument("--results-dir", required=True, help="Путь к папке results")
    p.add_argument("--out-dir", default=None, help="Куда сохранить отчет. По умолчанию <results-dir>/validation_report")
    p.add_argument("--sample-size", type=int, default=512, help="Максимальный размер стороны для downsample при чтении растров")
    p.add_argument("--max-unique-preview", type=int, default=12, help="Сколько уникальных значений сохранять для превью")
    p.add_argument("--fail-on-error", action="store_true", help="Возвращать код 2, если найдены ошибки")
    return p.parse_args()


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(x) for x in obj]
    return obj


def load_manifest(manifest_path: Path) -> List[str]:
    if not manifest_path.exists():
        return []
    lines = []
    for raw in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if s:
            lines.append(s.replace("\\", "/"))
    return lines


def classify_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if parts else ""


def parse_raster_name(name: str) -> Dict[str, Any]:
    stem = Path(name).stem
    m_pair = re.match(r"^(?P<site>[^_]+)_(?P<year1>\d{4})_(?P<year2>\d{4})_(?P<feature>.+)$", stem)
    if m_pair:
        return {
            "site": m_pair.group("site"),
            "year": int(m_pair.group("year1")),
            "year2": int(m_pair.group("year2")),
            "season": None,
            "feature": m_pair.group("feature"),
        }
    m = re.match(r"^(?P<site>[^_]+)_(?P<year>\d{4})_(?P<rest>.+)$", stem)
    if not m:
        return {"site": None, "year": None, "year2": None, "season": None, "feature": None}
    site = m.group("site")
    year = int(m.group("year"))
    rest = m.group("rest")
    season = None
    feature = rest
    season_candidates = ["annual", "early", "mid", "late"]
    for s in season_candidates:
        prefix = s + "_"
        if rest.startswith(prefix):
            season = s
            feature = rest[len(prefix):]
            break
    if feature.endswith("_ms_composite"):
        feature = "ms_composite"
    return {"site": site, "year": year, "year2": None, "season": season, "feature": feature}


def transform_bounds_safe(src_crs: Any, dst_crs: Any, bounds: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    if not bounds or src_crs is None or dst_crs is None or transform_bounds is None:
        return None
    try:
        return tuple(transform_bounds(src_crs, dst_crs, *bounds, densify_pts=21))
    except Exception:
        return None


def boxes_intersect(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]) -> bool:
    if not a or not b:
        return False
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def bounds_area(bounds: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
    if not bounds:
        return None
    w = max(0.0, bounds[2] - bounds[0])
    h = max(0.0, bounds[3] - bounds[1])
    return w * h


def intersection_area(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
    if not a or not b:
        return None
    xmin = max(a[0], b[0])
    ymin = max(a[1], b[1])
    xmax = min(a[2], b[2])
    ymax = min(a[3], b[3])
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    return (xmax - xmin) * (ymax - ymin)


def geom_types_series(gdf: Any) -> List[str]:
    try:
        values = sorted({str(x) for x in gdf.geometry.geom_type.dropna().unique().tolist()})
        return values
    except Exception:
        return []


def duplicate_geometry_count(gdf: Any) -> int:
    try:
        hashes = gdf.geometry.apply(lambda g: g.wkb_hex if g is not None and not g.is_empty else None)
        return int(hashes.duplicated().sum())
    except Exception:
        return 0


def open_vector(path: Path, reporter: Reporter, root: Path) -> Optional[VectorMeta]:
    if gpd is None:
        reporter.error("DEPENDENCY", path, "geopandas не установлен")
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        reporter.error("VECTOR_OPEN", path, f"Не удалось открыть векторный файл: {e}")
        return None

    rel = str(path.relative_to(root))
    folder = classify_path(path, root)
    feature_count = int(len(gdf))
    crs = str(gdf.crs) if gdf.crs is not None else None
    invalid_count = None
    empty_count = None
    bounds = None
    try:
        invalid_count = int((~gdf.geometry.is_valid.fillna(False)).sum())
        empty_count = int((gdf.geometry.is_empty.fillna(False) | gdf.geometry.isna()).sum())
        if feature_count > 0:
            b = gdf.total_bounds.tolist()
            bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except Exception:
        pass
    meta = VectorMeta(
        path=str(path),
        relpath=rel,
        folder=folder,
        feature_count=feature_count,
        crs=crs,
        geometry_types=geom_types_series(gdf),
        invalid_count=invalid_count,
        empty_count=empty_count,
        bounds=bounds,
        columns=list(gdf.columns),
        duplicate_geometries=duplicate_geometry_count(gdf),
    )

    if feature_count == 0:
        reporter.error("VECTOR_EMPTY", path, "Векторный файл пустой")
    if not crs:
        reporter.error("VECTOR_NO_CRS", path, "У векторного файла отсутствует CRS")
    if invalid_count and invalid_count > 0:
        reporter.error("VECTOR_INVALID", path, "Есть невалидные геометрии", {"invalid_count": invalid_count})
    if empty_count and empty_count > 0:
        reporter.error("VECTOR_EMPTY_GEOMS", path, "Есть пустые или отсутствующие геометрии", {"empty_count": empty_count})
    if meta.duplicate_geometries and meta.duplicate_geometries > 0:
        reporter.warn("VECTOR_DUP_GEOM", path, "Есть дублирующиеся геометрии", {"duplicate_geometries": meta.duplicate_geometries})

    lower_name = path.name.lower()
    if "parcels" in lower_name:
        missing = sorted(EXPECTED_PARCEL_COLUMNS - set(gdf.columns))
        if missing:
            reporter.warn("PARCEL_COLUMNS", path, "В parcels не хватает ожидаемых колонок", {"missing": missing})
        if set(meta.geometry_types) - {"Polygon", "MultiPolygon"}:
            reporter.error("PARCEL_GEOMTYPE", path, "В parcels ожидаются только Polygon/MultiPolygon", {"geometry_types": meta.geometry_types})

    if folder == "aoi":
        if set(meta.geometry_types) - {"Polygon", "MultiPolygon"}:
            reporter.error("AOI_GEOMTYPE", path, "AOI должен содержать только Polygon/MultiPolygon", {"geometry_types": meta.geometry_types})
        if bounds and (bounds[0] == bounds[2] or bounds[1] == bounds[3]):
            reporter.error("AOI_ZERO_BOUNDS", path, "AOI имеет нулевые размеры по bounds")

    return meta


def pick_sample_band(ds: Any) -> int:
    return 1


def read_raster_sample(ds: Any, sample_size: int) -> np.ndarray:
    band = pick_sample_band(ds)
    out_h = max(1, min(ds.height, sample_size))
    out_w = max(1, min(ds.width, sample_size))
    arr = ds.read(band, out_shape=(out_h, out_w), masked=True)
    return np.asarray(arr)


def raster_unique_preview(arr: np.ndarray, max_unique_preview: int) -> List[float]:
    data = arr.compressed() if np.ma.isMaskedArray(arr) else arr.reshape(-1)
    if data.size == 0:
        return []
    uniq = np.unique(data)
    uniq = uniq[:max_unique_preview]
    out = []
    for x in uniq.tolist():
        sx = safe_float(x)
        if sx is not None:
            out.append(sx)
    return out


def open_raster(path: Path, reporter: Reporter, root: Path, sample_size: int, max_unique_preview: int) -> Optional[RasterMeta]:
    if rasterio is None:
        reporter.error("DEPENDENCY", path, "rasterio не установлен")
        return None
    try:
        with rasterio.open(path) as ds:
            rel = str(path.relative_to(root))
            folder = classify_path(path, root)
            parsed = parse_raster_name(path.name)
            crs = ds.crs.to_string() if ds.crs else None
            bounds = tuple(float(x) for x in ds.bounds)
            transform = (ds.transform.a, ds.transform.b, ds.transform.c, ds.transform.d, ds.transform.e, ds.transform.f)
            arr = read_raster_sample(ds, sample_size)
            data = arr.compressed() if np.ma.isMaskedArray(arr) else arr.reshape(-1)
            finite = data[np.isfinite(data)] if data.size else data
            sampled_pixels = int(data.size)
            mn = mx = mean = std = None
            finite_fraction = None
            if sampled_pixels > 0:
                finite_fraction = float(finite.size / max(1, sampled_pixels))
            if finite.size > 0:
                mn = float(np.min(finite))
                mx = float(np.max(finite))
                mean = float(np.mean(finite))
                std = float(np.std(finite))
            unique_preview = raster_unique_preview(arr, max_unique_preview)

            meta = RasterMeta(
                path=str(path),
                relpath=rel,
                folder=folder,
                site=parsed.get("site"),
                year=parsed.get("year"),
                year2=parsed.get("year2"),
                season=parsed.get("season"),
                feature=parsed.get("feature"),
                crs=crs,
                width=int(ds.width),
                height=int(ds.height),
                count=int(ds.count),
                dtype=str(ds.dtypes[0]) if ds.count else None,
                nodata=ds.nodata,
                bounds=bounds,
                transform=transform,
                min=mn,
                max=mx,
                mean=mean,
                std=std,
                finite_fraction=finite_fraction,
                sampled_pixels=sampled_pixels,
                unique_preview=unique_preview,
            )

            if ds.width <= 0 or ds.height <= 0:
                reporter.error("RASTER_SIZE", path, "Некорректные размеры растра")
            if not crs:
                reporter.error("RASTER_NO_CRS", path, "У растра отсутствует CRS")
            if ds.count <= 0:
                reporter.error("RASTER_NO_BANDS", path, "У растра нет ни одного канала")
            if finite_fraction is None or finite_fraction == 0:
                reporter.error("RASTER_ALL_NODATA", path, "Выборка растра состоит только из nodata/NaN")
            elif finite_fraction < 0.1:
                reporter.warn("RASTER_MOSTLY_NODATA", path, "В растре слишком мало валидных пикселей", {"finite_fraction": finite_fraction})
            if mn is not None and mx is not None and mn == mx:
                reporter.warn("RASTER_CONSTANT", path, "Растр выглядит константным", {"value": mn})

            apply_raster_value_rules(meta, reporter)
            return meta
    except Exception as e:
        reporter.error("RASTER_OPEN", path, f"Не удалось открыть растр: {e}")
        return None


def apply_raster_value_rules(meta: RasterMeta, reporter: Reporter) -> None:
    name = Path(meta.path).name
    mn = meta.min
    mx = meta.max
    if mn is None or mx is None:
        return

    for regex, lo, hi, text in RANGE_RULES:
        if regex.search(name):
            if mn < lo or mx > hi:
                reporter.warn("RASTER_RANGE", meta.path, text, {"min": mn, "max": mx, "expected": [lo, hi]})

    for regex, text in NON_NEGATIVE_RULES:
        if regex.search(name) and mn < -1e-6:
            reporter.warn("RASTER_NEGATIVE", meta.path, text, {"min": mn})

    for regex in BINARY_HINT_RULES:
        if regex.search(name):
            uniq = set(round(x) for x in meta.unique_preview if x is not None)
            if not uniq.issubset({0, 1, 255}):
                reporter.warn("MASK_NOT_BINARY", meta.path, "Маска содержит значения вне ожидаемого бинарного набора {0,1,255}", {"unique_preview": meta.unique_preview})
            break

    if re.search(r"(?:^|_)risk_score(?:_|\.)", name, re.IGNORECASE):
        if mx > 1.05 and mx <= 100.5:
            reporter.info("RISK_SCORE_SCALE", meta.path, "risk_score похож на шкалу 0..100, это допустимо если так задумано", {"max": mx})
        elif mx > 100.5:
            reporter.warn("RISK_SCORE_HIGH", meta.path, "risk_score уходит выше 100", {"max": mx})

    if re.search(r"(?:^|_)water_occurrence(?:_|\.)", name, re.IGNORECASE):
        if mx > 1.05 and mx <= 100.5:
            reporter.info("WATER_OCCURRENCE_SCALE", meta.path, "water_occurrence похож на шкалу 0..100, это допустимо если так задумано", {"max": mx})
        elif mx > 100.5:
            reporter.warn("WATER_OCCURRENCE_HIGH", meta.path, "water_occurrence уходит выше 100", {"max": mx})


def open_table(path: Path, reporter: Reporter, root: Path) -> Optional[TableMeta]:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception as e:
            reporter.error("TABLE_OPEN", path, f"Не удалось открыть CSV: {e}")
            return None

    rel = str(path.relative_to(root))
    folder = classify_path(path, root)
    rows, cols = df.shape
    meta = TableMeta(
        path=str(path),
        relpath=rel,
        folder=folder,
        rows=int(rows),
        cols=int(cols),
        columns=list(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
    )

    if rows == 0:
        reporter.error("TABLE_EMPTY", path, "CSV пустой")
    if cols == 0:
        reporter.error("TABLE_NO_COLS", path, "CSV не содержит столбцов")
    if meta.duplicate_rows and meta.duplicate_rows > 0:
        reporter.warn("TABLE_DUP_ROWS", path, "В CSV есть дублирующиеся строки", {"duplicate_rows": meta.duplicate_rows})
    if len(set(df.columns)) != len(df.columns):
        reporter.error("TABLE_DUP_COLS", path, "В CSV есть дублирующиеся имена колонок")
    if any(str(c).startswith("Unnamed:") for c in df.columns):
        reporter.warn("TABLE_UNNAMED", path, "В CSV есть безымянные колонки")
    return meta


def discover_files(root: Path) -> Dict[str, List[Path]]:
    files: Dict[str, List[Path]] = {"raster": [], "vector": [], "table": [], "other": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf in RASTER_EXTS:
            files["raster"].append(path)
        elif suf in VECTOR_EXTS:
            files["vector"].append(path)
        elif suf in TABLE_EXTS:
            files["table"].append(path)
        else:
            files["other"].append(path)
    return files


def check_root_structure(root: Path, reporter: Reporter) -> None:
    if not root.exists():
        reporter.error("ROOT_MISSING", root, "Папка results не существует")
        return
    if not root.is_dir():
        reporter.error("ROOT_NOT_DIR", root, "Путь results не является директорией")
        return

    for name in EXPECTED_SUBDIRS:
        p = root / name
        if not p.exists():
            reporter.error("SUBDIR_MISSING", p, f"Отсутствует обязательная подпапка {name}")
        elif not p.is_dir():
            reporter.error("SUBDIR_NOT_DIR", p, f"{name} существует, но это не папка")

    for name in ["manifest.txt", "repair_terrain_risk_report.json", "run_report.json"]:
        p = root / name
        if not p.exists():
            reporter.warn("ROOT_FILE_MISSING", p, f"В корне нет файла {name}")


def select_aoi_meta(vector_metas: List[VectorMeta], reporter: Reporter) -> Optional[VectorMeta]:
    preferred = [
        "aoi/area_aoi.gpkg",
        "aoi/aoi_union.geojson",
        "aoi/scene_footprints.gpkg",
    ]
    rel_to_meta = {m.relpath.replace("\\", "/"): m for m in vector_metas}
    for key in preferred:
        if key in rel_to_meta:
            reporter.info("AOI_SOURCE", key, f"Базовый AOI для проверок: {key}")
            return rel_to_meta[key]
    if vector_metas:
        reporter.warn("AOI_SOURCE_FALLBACK", vector_metas[0].path, "Базовый AOI не найден, использую первый вектор из папки aoi")
        return vector_metas[0]
    reporter.error("AOI_MISSING", "aoi", "Не удалось найти ни одного AOI-вектора для пространственных проверок")
    return None


def check_manifest(root: Path, reporter: Reporter, discovered: Dict[str, List[Path]]) -> None:
    manifest_path = root / "manifest.txt"
    manifest = load_manifest(manifest_path)
    if not manifest:
        return
    actual = {str(p.relative_to(root)).replace("\\", "/") for paths in discovered.values() for p in paths}
    manifest_set = set(manifest)
    missing_from_disk = sorted(manifest_set - actual)
    extra_on_disk = sorted(actual - manifest_set)
    if missing_from_disk:
        reporter.warn("MANIFEST_MISSING_FILES", manifest_path, "В manifest перечислены файлы, которых нет на диске", {"count": len(missing_from_disk), "examples": missing_from_disk[:20]})
    if extra_on_disk:
        reporter.info("MANIFEST_EXTRA_FILES", manifest_path, "На диске есть файлы, которых нет в manifest", {"count": len(extra_on_disk), "examples": extra_on_disk[:20]})


def check_bounds_against_aoi(aoi_meta: Optional[VectorMeta], raster_metas: List[RasterMeta], vector_metas: List[VectorMeta], reporter: Reporter) -> None:
    if not aoi_meta or not aoi_meta.bounds or not aoi_meta.crs:
        return
    aoi_bounds = aoi_meta.bounds
    aoi_crs = aoi_meta.crs

    for meta in raster_metas:
        if not meta.bounds:
            continue
        rb = meta.bounds
        if meta.crs != aoi_crs:
            rb = transform_bounds_safe(meta.crs, aoi_crs, meta.bounds) if meta.crs else None
        if rb is None:
            reporter.warn("RASTER_BOUNDS_REPROJECT", meta.path, "Не удалось привести bounds растра к CRS AOI")
            continue
        if not boxes_intersect(rb, aoi_bounds):
            reporter.error("RASTER_OUTSIDE_AOI", meta.path, "Bounds растра не пересекают AOI")
            continue
        aoi_area = bounds_area(aoi_bounds)
        inter_area = intersection_area(rb, aoi_bounds)
        rb_area = bounds_area(rb)
        if aoi_area and rb_area and inter_area is not None and rb_area > 0:
            overlap_fraction = inter_area / rb_area
            if overlap_fraction < 0.05:
                reporter.warn("RASTER_LOW_AOI_OVERLAP", meta.path, "У bounds растра очень маленькое пересечение с AOI", {"overlap_fraction": overlap_fraction})

    for meta in vector_metas:
        if meta.folder not in {"parcels", "analytics"}:
            continue
        if not meta.bounds or not meta.crs:
            continue
        vb = meta.bounds
        if meta.crs != aoi_crs:
            if gpd is None:
                reporter.warn("VECTOR_BOUNDS_REPROJECT", meta.path, "Не удалось привести bounds вектора к CRS AOI")
                continue
            try:
                from shapely.geometry import box
                vb = tuple(gpd.GeoSeries([box(*meta.bounds)], crs=meta.crs).to_crs(aoi_crs).total_bounds.tolist())
            except Exception:
                reporter.warn("VECTOR_BOUNDS_REPROJECT", meta.path, "Не удалось привести bounds вектора к CRS AOI")
                continue
        if not boxes_intersect(vb, aoi_bounds):
            reporter.error("VECTOR_OUTSIDE_AOI", meta.path, "Bounds векторного слоя не пересекают AOI")


def check_group_alignment(raster_metas: List[RasterMeta], reporter: Reporter) -> None:
    groups: Dict[Tuple[str, str, int], List[RasterMeta]] = defaultdict(list)
    for m in raster_metas:
        if m.site and m.year and m.crs and m.width and m.height and m.transform:
            groups[(m.folder, m.site, m.year)].append(m)
    for key, items in groups.items():
        by_shape = Counter((m.width, m.height, m.crs, m.transform) for m in items)
        if len(by_shape) > 1 and len(items) > 1:
            worst = by_shape.most_common()
            reporter.warn(
                "GROUP_ALIGNMENT",
                f"{key[0]}/{key[1]}_{key[2]}",
                "Внутри одной группы есть разные сетки/CRS/transform. Для моделей это потенциально опасно.",
                {"variants": [{"grid": list(k), "count": v} for k, v in worst[:10]]},
            )


def check_expected_family_completeness(raster_metas: List[RasterMeta], reporter: Reporter) -> None:
    terrain = defaultdict(set)
    masks = defaultdict(set)
    indices = defaultdict(set)
    textures = defaultdict(set)
    dyn_year = defaultdict(set)
    dyn_pair = defaultdict(set)
    composites = defaultdict(set)

    for m in raster_metas:
        if not m.site or not m.year or not m.feature:
            continue
        base = (m.site, m.year)
        if m.folder == "terrain":
            terrain[base].add(m.feature)
        elif m.folder == "masks":
            masks[base].add(m.feature)
        elif m.folder == "indices":
            indices[(m.site, m.year, m.season)].add(m.feature)
        elif m.folder == "composites":
            composites[(m.site, m.year)].add(m.season)
        elif m.folder == "textures":
            textures[base].add(m.feature)
        elif m.folder == "dynamics":
            if m.year2:
                dyn_pair[(m.site, m.year, m.year2)].add(m.feature)
            else:
                dyn_year[base].add(m.feature)

    for key, feats in terrain.items():
        missing = sorted(TERRAIN_FEATURES - feats)
        if missing:
            reporter.warn("TERRAIN_MISSING", f"terrain/{key[0]}_{key[1]}", "В наборе terrain не хватает ожидаемых слоев", {"missing": missing, "present": sorted(feats)})

    for key, feats in masks.items():
        missing = sorted(MASK_FEATURES - feats)
        if missing:
            reporter.warn("MASKS_MISSING", f"masks/{key[0]}_{key[1]}", "В наборе masks не хватает ожидаемых масок", {"missing": missing, "present": sorted(feats)})

    for key, feats in dyn_year.items():
        missing = sorted(DYNAMIC_YEAR_FEATURES - feats)
        if missing:
            reporter.warn("DYNAMICS_YEAR_MISSING", f"dynamics/{key[0]}_{key[1]}", "В динамике по году не хватает risk_score/water_occurrence", {"missing": missing, "present": sorted(feats)})

    for key, feats in dyn_pair.items():
        missing = sorted(DYNAMIC_PAIR_FEATURES - feats)
        if missing:
            reporter.warn("DYNAMICS_PAIR_MISSING", f"dynamics/{key[0]}_{key[1]}_{key[2]}", "В динамике по паре лет не хватает ожидаемых слоев", {"missing": missing, "present": sorted(feats)})

    for key, feats in indices.items():
        missing = sorted(INDEX_FEATURES - feats)
        if missing:
            reporter.warn("INDICES_MISSING", f"indices/{key[0]}_{key[1]}_{key[2]}", "Для сезона не хватает части спектральных индексов", {"missing": missing, "present": sorted(feats)})

    for key, seasons in composites.items():
        if not seasons:
            reporter.warn("COMPOSITE_EMPTY", f"composites/{key[0]}_{key[1]}", "Для site-year нет ни одного composite сезона")

    expected_texture_count = len(TEXTURE_CHANNELS) * len(TEXTURE_FEATURES) * len(TEXTURE_WINDOWS)
    for key, feats in textures.items():
        if len(feats) < expected_texture_count:
            reporter.warn("TEXTURES_MISSING", f"textures/{key[0]}_{key[1]}", "В texture-наборе меньше слоев, чем ожидается по паттерну channel x feature x window", {"present_count": len(feats), "expected_count": expected_texture_count})


def check_tables_against_vectors(table_metas: List[TableMeta], vector_metas: List[VectorMeta], reporter: Reporter) -> None:
    by_stem_vector = defaultdict(list)
    for vm in vector_metas:
        by_stem_vector[Path(vm.path).stem].append(vm)

    for tm in table_metas:
        stem = Path(tm.path).stem
        candidates = by_stem_vector.get(stem, [])
        if candidates:
            reporter.info("TABLE_VECTOR_PAIR", tm.path, "Найден одноименный векторный слой для CSV")

        lower = Path(tm.path).name.lower()
        if "parcel_stats" in lower and tm.rows is not None and tm.rows <= 0:
            reporter.error("PARCEL_STATS_EMPTY", tm.path, "parcel_stats CSV пустой")
        if "valid" in lower and tm.rows is not None and tm.rows == 0:
            reporter.error("VALID_EMPTY", tm.path, "*_valid.csv пустой")
        if "front" in lower and tm.rows is not None and tm.rows == 0:
            reporter.error("FRONT_EMPTY", tm.path, "*_front.csv пустой")


def check_catalog_outputs(table_metas: List[TableMeta], reporter: Reporter) -> None:
    wanted = {
        "scene_catalog_from_inventory.csv",
        "scene_catalog_scan.csv",
        "scene_catalog_selected_ms.csv",
    }
    found = {Path(t.path).name for t in table_metas if t.folder == "catalog"}
    missing = sorted(wanted - found)
    if missing:
        reporter.warn("CATALOG_MISSING", "catalog", "В catalog отсутствуют ожидаемые CSV", {"missing": missing})


def check_json_reports(root: Path, reporter: Reporter) -> Dict[str, Any]:
    report_info: Dict[str, Any] = {}
    for name in ["run_report.json", "repair_terrain_risk_report.json", "catalog/inventory_report.json"]:
        path = root / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            report_info[name] = data
            reporter.info("JSON_OK", path, "JSON-отчет успешно прочитан")
        except Exception as e:
            reporter.error("JSON_BAD", path, f"Не удалось прочитать JSON: {e}")
    return report_info


def severity_rank(sev: str) -> int:
    return SEVERITY_ORDER.get(sev, 99)


def build_markdown_summary(root: Path, reporter: Reporter, raster_metas: List[RasterMeta], vector_metas: List[VectorMeta], table_metas: List[TableMeta]) -> str:
    counts = reporter.counts()
    issues_sorted = sorted(reporter.issues, key=lambda x: (-severity_rank(x.severity), x.path, x.code))
    top_errors = [x for x in issues_sorted if x.severity == "ERROR"][:50]
    top_warnings = [x for x in issues_sorted if x.severity == "WARNING"][:80]

    lines = []
    lines.append("# Отчет по валидации results")
    lines.append("")
    lines.append(f"- Папка: `{root}`")
    lines.append(f"- Растров: {len(raster_metas)}")
    lines.append(f"- Векторов: {len(vector_metas)}")
    lines.append(f"- Таблиц CSV: {len(table_metas)}")
    lines.append(f"- Ошибок: {counts['ERROR']}")
    lines.append(f"- Предупреждений: {counts['WARNING']}")
    lines.append(f"- Информационных сообщений: {counts['INFO']}")
    lines.append("")
    lines.append("## Что проверялось")
    lines.append("")
    lines.append("- структура папок и наличие ключевых файлов")
    lines.append("- открываемость raster/vector/csv")
    lines.append("- CRS, bounds, размеры, число каналов, nodata")
    lines.append("- валидность геометрий и типы геометрий")
    lines.append("- пересечение с AOI")
    lines.append("- диапазоны значений для индексов, terrain-слоев и масок")
    lines.append("- полнота семейств слоев: terrain, masks, indices, dynamics, textures")
    lines.append("- согласованность сеток внутри групп site-year")
    lines.append("- базовая согласованность analytics/catalog/json-отчетов")
    lines.append("")
    lines.append("## Критичные проблемы")
    lines.append("")
    if not top_errors:
        lines.append("Критичных ошибок не найдено.")
    else:
        for issue in top_errors:
            lines.append(f"- [{issue.code}] `{issue.path}`: {issue.message}")
    lines.append("")
    lines.append("## Предупреждения")
    lines.append("")
    if not top_warnings:
        lines.append("Предупреждений нет.")
    else:
        for issue in top_warnings:
            lines.append(f"- [{issue.code}] `{issue.path}`: {issue.message}")
    lines.append("")
    lines.append("## Интерпретация")
    lines.append("")
    if counts["ERROR"] == 0:
        lines.append("Датасет выглядит пригодным для следующего этапа, но предупреждения все равно стоит разобрать перед финальной сборкой признаков и обучением модели.")
    else:
        lines.append("До обучения модели набор лучше не считать окончательно надежным: критичные ошибки означают риск сломать признаки, пространственное соответствие или сам train/inference pipeline.")
    lines.append("")
    lines.append("## Файлы отчета")
    lines.append("")
    lines.append("- `validation_summary.json` - агрегированный JSON")
    lines.append("- `validation_issues.csv` - плоская таблица проблем")
    lines.append("- `validation_inventory.csv` - инвентаризация файлов и метаданных")
    lines.append("- `validation_report.md` - этот markdown-отчет")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, reporter: Reporter, root: Path, raster_metas: List[RasterMeta], vector_metas: List[VectorMeta], table_metas: List[TableMeta], json_reports: Dict[str, Any]) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows: List[Dict[str, Any]] = []
    for m in raster_metas:
        d = asdict(m)
        d["kind"] = "raster"
        inventory_rows.append(json_ready(d))
    for m in vector_metas:
        d = asdict(m)
        d["kind"] = "vector"
        inventory_rows.append(json_ready(d))
    for m in table_metas:
        d = asdict(m)
        d["kind"] = "table"
        inventory_rows.append(json_ready(d))

    issues_rows = [json_ready(asdict(x)) for x in reporter.issues]

    summary = {
        "results_dir": str(root),
        "counts": reporter.counts(),
        "n_rasters": len(raster_metas),
        "n_vectors": len(vector_metas),
        "n_tables": len(table_metas),
        "json_reports_read": sorted(json_reports.keys()),
        "issues": issues_rows,
    }

    json_path = out_dir / "validation_summary.json"
    issues_csv_path = out_dir / "validation_issues.csv"
    inv_csv_path = out_dir / "validation_inventory.csv"
    md_path = out_dir / "validation_report.md"

    json_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(issues_rows).to_csv(issues_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory_rows).to_csv(inv_csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_markdown_summary(root, reporter, raster_metas, vector_metas, table_metas), encoding="utf-8")

    return {
        "summary_json": json_path,
        "issues_csv": issues_csv_path,
        "inventory_csv": inv_csv_path,
        "report_md": md_path,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.results_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "validation_report")
    reporter = Reporter()

    check_root_structure(root, reporter)
    if not root.exists() or not root.is_dir():
        write_outputs(out_dir, reporter, root, [], [], [], {})
        return 2

    discovered = discover_files(root)
    check_manifest(root, reporter, discovered)

    vector_metas: List[VectorMeta] = []
    for path in discovered["vector"]:
        meta = open_vector(path, reporter, root)
        if meta:
            vector_metas.append(meta)

    raster_metas: List[RasterMeta] = []
    for path in discovered["raster"]:
        meta = open_raster(path, reporter, root, args.sample_size, args.max_unique_preview)
        if meta:
            raster_metas.append(meta)

    table_metas: List[TableMeta] = []
    for path in discovered["table"]:
        meta = open_table(path, reporter, root)
        if meta:
            table_metas.append(meta)

    aoi_vectors = [m for m in vector_metas if m.folder == "aoi"]
    aoi_meta = select_aoi_meta(aoi_vectors, reporter)

    check_bounds_against_aoi(aoi_meta, raster_metas, vector_metas, reporter)
    check_group_alignment(raster_metas, reporter)
    check_expected_family_completeness(raster_metas, reporter)
    check_tables_against_vectors(table_metas, vector_metas, reporter)
    check_catalog_outputs(table_metas, reporter)
    json_reports = check_json_reports(root, reporter)

    outputs = write_outputs(out_dir, reporter, root, raster_metas, vector_metas, table_metas, json_reports)

    counts = reporter.counts()
    print(json.dumps({
        "results_dir": str(root),
        "out_dir": str(out_dir),
        "n_rasters": len(raster_metas),
        "n_vectors": len(vector_metas),
        "n_tables": len(table_metas),
        "counts": counts,
        "files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))

    if args.fail_on_error and counts["ERROR"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
