#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной GIS-пайплайн проекта GIS_IT_2026 / ArcticLens.

Назначение скрипта:
1. Найти и каталогизировать сцены Канопус.
2. Построить рабочую AOI по фактическим footprint сцен.
3. Подготовить годовые и сезонные композиты на единой сетке.
4. Рассчитать спектральные, текстурные, рельефные и динамические признаки.
5. Сформировать интерпретируемый baseline-индекс risk_score и тематические маски.
6. Выполнить parcel-level агрегацию признаков по полигонам участков.
7. Сохранить воспроизводимые артефакты в виде GeoTIFF, GeoJSON, GPKG и CSV.

"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --- Внешние библиотеки для пространственной, растровой и табличной обработки данных. ---
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio import features
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage as ndi
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows


# Регулярное выражение для разбора имени папки сцены Канопус и извлечения даты, времени, уровня и типа продукта.
SCENE_RE = re.compile(
    r'.*KANOPUS_(?P<date>\d{8})_(?P<time>\d{6})_\d+\.(?P<level>L\d)\.(?P<product>MS|PMS|PAN)\.SCN01',
    re.IGNORECASE,
)

# Границы сезонных интервалов, используемых при построении сезонных композитов.
DEFAULT_SEASONS = {
    "early": ((5, 1), (6, 15)),
    "mid": ((6, 16), (7, 31)),
    "late": ((8, 1), (9, 30)),
}

# Пороговые значения для формирования ключевых тематических масок.
CORE_MASK_THRESHOLDS = {
    "water_occurrence_persistence": 0.6,
    "risk_hotspot_quantile": 0.9,
    "texture_anomaly_quantile": 0.9,
}

# Веса компонент авторского baseline-индекса risk_score.
RISK_WEIGHTS = {
    "water_growth": 0.25,
    "vegetation_loss": 0.20,
    "texture_anomaly": 0.20,
    "terrain_susceptibility": 0.15,
    "temporal_instability": 0.20,
}


@dataclass
class SceneRecord:
    """Структура с метаданными одной найденной сцены и путями к связанным файлам."""
    scene_pkg: str
    area: str
    work_folder: str
    scene_folder: str
    tif_path: str
    gbd_path: str
    date: str
    time: str
    year: int
    level: str
    product: str
    score: int


def log(msg: str) -> None:
    """Печатает сервисное сообщение в stdout без буферизации."""
    print(msg, flush=True)


def safe_mkdir(path: Path) -> Path:
    """Создает каталог со всеми родительскими директориями и возвращает его путь."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_float32(arr: np.ndarray) -> np.ndarray:
    """Приводит массив к типу float32 для единообразной работы с растровыми слоями."""
    return np.asarray(arr, dtype=np.float32)


def nanpercentile(arr: np.ndarray, q: float) -> float:
    """Считает процентиль только по конечным значениям массива."""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def robust_minmax(arr: np.ndarray, q_low: float = 5, q_high: float = 95) -> np.ndarray:
    """Нормирует массив в диапазон [0, 1] по устойчивым квантилям, игнорируя выбросы."""
    vals = arr[np.isfinite(arr)]
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if vals.size == 0:
        return out
    lo = np.percentile(vals, q_low)
    hi = np.percentile(vals, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[np.isfinite(arr)] = 0.0
        return out
    out[np.isfinite(arr)] = np.clip((arr[np.isfinite(arr)] - lo) / (hi - lo), 0.0, 1.0)
    return out


def write_json(path: Path, obj: object) -> None:
    """Сохраняет объект Python в JSON-файл с UTF-8 и читаемыми отступами."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Читает векторный слой из обычного пути или напрямую из ZIP-архива."""
    path = str(path)
    if path.lower().endswith(".zip"):
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def save_vector(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Сохраняет GeoDataFrame в подходящем формате по расширению файла."""
    if path.suffix.lower() == ".geojson":
        gdf.to_file(path, driver="GeoJSON")
    elif path.suffix.lower() in {".gpkg"}:
        gdf.to_file(path, driver="GPKG")
    else:
        gdf.to_file(path)


def fix_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Удаляет пустые геометрии, исправляет невалидные полигоны и оставляет только Polygon/MultiPolygon."""
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.reset_index(drop=True)
    return gdf


def parse_area(work_folder: str) -> str:
    """Определяет имя территории по названию рабочей папки."""
    s = work_folder.lower()
    if "амга" in s:
        return "Амга"
    if "юнкор" in s:
        return "Юнкор"
    return "unknown"


def parse_scene_folder(scene_folder: str) -> Optional[Dict[str, str]]:
    """Разбирает имя папки сцены и извлекает дату, время, год, уровень и тип продукта."""
    m = SCENE_RE.search(scene_folder)
    if not m:
        return None
    date = pd.to_datetime(m.group("date"), format="%Y%m%d", errors="coerce")
    if pd.isna(date):
        return None
    return {
        "date": date.strftime("%Y-%m-%d"),
        "time": m.group("time"),
        "year": int(date.year),
        "level": m.group("level").upper(),
        "product": m.group("product").upper(),
    }


def score_scene(product: str, level: str) -> int:
    """Назначает сцене приоритет: MS и L2 считаются предпочтительными."""
    product_score = {"MS": 0, "PMS": 10, "PAN": 20}.get(product.upper(), 50)
    level_score = {"L2": 0, "L1": 1}.get(level.upper(), 5)
    return product_score + level_score


# --- Этап 1. Поиск и каталогизация сцен. ---
def discover_scene_dirs(data_root: Path) -> List[SceneRecord]:
    """Рекурсивно ищет сцены в data_root и собирает метаданные по каждому валидному набору."""
    rows: List[SceneRecord] = []
    for tif_path in data_root.rglob("*.tif"):
        scene_dir = tif_path.parent
        scene_folder = scene_dir.name
        meta = parse_scene_folder(scene_folder)
        if meta is None:
            continue
        gbd_candidates = sorted(scene_dir.glob("*.GBD.shp"))
        if not gbd_candidates:
            continue
        parts = tif_path.relative_to(data_root).parts
        if len(parts) < 4:
            continue
        scene_pkg = parts[0]
        work_folder = parts[1]
        area = parse_area(work_folder)
        rows.append(
            SceneRecord(
                scene_pkg=scene_pkg,
                area=area,
                work_folder=work_folder,
                scene_folder=scene_folder,
                tif_path=str(tif_path),
                gbd_path=str(gbd_candidates[0]),
                date=meta["date"],
                time=meta["time"],
                year=meta["year"],
                level=meta["level"],
                product=meta["product"],
                score=score_scene(meta["product"], meta["level"]),
            )
        )
    return rows


def scene_catalog_from_inventory(inventory_csv: Path) -> pd.DataFrame:
    """Строит каталог сцен на основе inventory CSV, если он передан отдельно."""
    df = pd.read_csv(inventory_csv)
    df["path"] = df["FullName"].astype(str).str.replace("\\", "/", regex=False)
    rows = []
    for rel in df.loc[df["Type"].eq("FILE"), "path"]:
        p = Path(rel)
        if p.suffix.lower() != ".tif":
            continue
        meta = parse_scene_folder(p.parent.name)
        if meta is None:
            continue
        parts = p.parts
        if len(parts) < 4:
            continue
        gbd_name = p.parent.name + ".GBD.shp"
        rows.append(
            {
                "scene_pkg": parts[-4],
                "work_folder": parts[-3],
                "scene_folder": parts[-2],
                "tif_name": p.name,
                "gbd_name": gbd_name,
                "area": parse_area(parts[-3]),
                "date": meta["date"],
                "time": meta["time"],
                "year": meta["year"],
                "level": meta["level"],
                "product": meta["product"],
                "score": score_scene(meta["product"], meta["level"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["area", "date", "time", "score"]).reset_index(drop=True)


def choose_best_ms_scenes(catalog: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только мультиспектральные сцены и выбирает лучшую сцену на acquisition."""
    if catalog.empty:
        return catalog
    df = catalog.copy()
    df = df[df["product"].eq("MS")]
    if df.empty:
        raise ValueError("В каталоге не найдено ни одной мультиспектральной сцены MS.")
    df = df.sort_values(["area", "date", "time", "score", "scene_pkg"])
    df = df.groupby(["area", "date", "time"], as_index=False).first()
    return df.reset_index(drop=True)


def build_inventory_report(inventory_csv: Optional[Path], out_dir: Path) -> Optional[pd.DataFrame]:
    """Создает сводный отчет по inventory CSV и сохраняет его в каталог результатов."""
    if inventory_csv is None or not inventory_csv.exists():
        return None
    catalog = scene_catalog_from_inventory(inventory_csv)
    if catalog.empty:
        return catalog
    report = {
        "rows_total": int(len(catalog)),
        "by_area": catalog["area"].value_counts().to_dict(),
        "by_product": catalog["product"].value_counts().to_dict(),
        "by_level": catalog["level"].value_counts().to_dict(),
        "unique_acquisitions_by_area": catalog.groupby("area")[["date", "time"]].apply(lambda x: len(x.drop_duplicates())).to_dict(),
        "by_year": catalog["year"].value_counts().sort_index().to_dict(),
    }
    catalog.to_csv(out_dir / "scene_catalog_from_inventory.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "inventory_report.json", report)
    return catalog


def catalog_to_df(rows: Sequence[SceneRecord]) -> pd.DataFrame:
    """Преобразует список SceneRecord в табличный DataFrame."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in rows]).sort_values(["area", "date", "time", "score"]).reset_index(drop=True)


def assign_season(ts: pd.Timestamp, seasons: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]) -> Optional[str]:
    """Относит дату съемки к одному из заранее заданных сезонных интервалов."""
    if pd.isna(ts):
        return None
    for name, ((m1, d1), (m2, d2)) in seasons.items():
        start = pd.Timestamp(year=ts.year, month=m1, day=d1)
        end = pd.Timestamp(year=ts.year, month=m2, day=d2)
        if start <= ts <= end:
            return name
    return None


# --- Этап 2. Построение AOI по footprint сцен и подготовка рабочей parcel-маски. ---
def read_footprints(scene_catalog: pd.DataFrame) -> gpd.GeoDataFrame:
    """Читает GBD-footprint для выбранных сцен и собирает их в один GeoDataFrame."""
    rows = []
    for row in scene_catalog.itertuples():
        gdf = read_vector(row.gbd_path)
        gdf = fix_polygons(gdf)
        if gdf.empty:
            continue
        geom = unary_union(list(gdf.geometry))
        rows.append({"area": row.area, "date": row.date, "time": row.time, "scene_pkg": row.scene_pkg, "geometry": geom})
    if not rows:
        raise ValueError("Не удалось собрать ни одного footprint из GBD shapefile.")
    gdf = gpd.GeoDataFrame(rows, crs=read_vector(scene_catalog.iloc[0]["gbd_path"]).crs)
    return fix_polygons(gdf)


def dissolve_aoi(footprints: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Строит AOI по территориям и общий union AOI на основе footprint сцен."""
    per_area = footprints.dissolve(by="area", as_index=False)
    union = gpd.GeoDataFrame([{"aoi_name": "all", "geometry": unary_union(list(per_area.geometry))}], crs=per_area.crs)
    return fix_polygons(per_area), fix_polygons(union)


def clip_parcels_to_aoi(parcel_path: Optional[Path], aoi_union: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
    """Обрезает слой участков по union AOI и при необходимости назначает parcel_id."""
    if parcel_path is None:
        return None
    parcels = read_vector(parcel_path)
    parcels = fix_polygons(parcels)
    if parcels.crs != aoi_union.crs:
        parcels = parcels.to_crs(aoi_union.crs)
    clipped = gpd.overlay(parcels, aoi_union[["geometry"]], how="intersection")
    clipped = fix_polygons(clipped)
    if "parcel_id" not in clipped.columns:
        clipped["parcel_id"] = np.arange(1, len(clipped) + 1)
    return clipped.reset_index(drop=True)


def bounds_union(bounds_list: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Возвращает объединяющий bounding box для списка границ."""
    left = min(b[0] for b in bounds_list)
    bottom = min(b[1] for b in bounds_list)
    right = max(b[2] for b in bounds_list)
    top = max(b[3] for b in bounds_list)
    return (left, bottom, right, top)


def area_bounds_in_crs(geoms: Iterable, src_crs, dst_crs) -> Tuple[float, float, float, float]:
    """Переводит границы геометрий в целевую CRS и объединяет их в один extent."""
    bds = []
    for geom in geoms:
        bds.append(transform_bounds(src_crs, dst_crs, *geom.bounds, densify_pts=21))
    return bounds_union(bds)


# --- Этап 3. Подготовка единой пространственной сетки и перепроекция сцен. ---
def choose_reference_scene(scene_paths: Sequence[str]) -> Tuple[dict, float, float]:
    """Выбирает опорную сцену с наилучшим пространственным разрешением."""
    best = None
    best_res = None
    for path in scene_paths:
        with rasterio.open(path) as src:
            rx = abs(src.transform.a)
            ry = abs(src.transform.e)
            cand = (min(rx, ry), src.meta.copy())
            if best is None or cand[0] < best_res:
                best = cand[1]
                best_res = cand[0]
    if best is None:
        raise ValueError("Нет сцен для выбора опорной сетки.")
    return best, best["transform"].a, abs(best["transform"].e)


def build_reference_grid(scene_paths: Sequence[str], clip_gdf: gpd.GeoDataFrame) -> Tuple[Affine, int, int, object]:
    """Формирует единую расчетную сетку по AOI и разрешению опорной сцены."""
    ref_meta, resx, resy = choose_reference_scene(scene_paths)
    dst_crs = ref_meta["crs"]
    if clip_gdf.crs != dst_crs:
        clip_gdf = clip_gdf.to_crs(dst_crs)
    bounds = area_bounds_in_crs(clip_gdf.geometry, clip_gdf.crs, dst_crs) if clip_gdf.crs != dst_crs else bounds_union([g.bounds for g in clip_gdf.geometry])
    width = max(1, int(math.ceil((bounds[2] - bounds[0]) / abs(resx))))
    height = max(1, int(math.ceil((bounds[3] - bounds[1]) / abs(resy))))
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    return transform, width, height, dst_crs


def read_reprojected_stack(
    scene_paths: Sequence[str],
    transform: Affine,
    width: int,
    height: int,
    dst_crs,
    band_map: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Читает нужные каналы сцен, перепроецирует их на общую сетку и собирает стек."""
    band_order = ["blue", "green", "red", "nir"]
    idxs = [band_map[k] for k in band_order]
    stack = []
    valid_masks = []
    for path in scene_paths:
        with rasterio.open(path) as src:
            data = np.full((len(idxs), height, width), np.nan, dtype=np.float32)
            for i, src_band in enumerate(idxs):
                dest = np.full((height, width), np.nan, dtype=np.float32)
                src_arr = src.read(src_band).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    src_arr[src_arr == nodata] = np.nan
                reproject(
                    source=src_arr,
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                data[i] = dest
            valid = np.isfinite(data).all(axis=0)
            stack.append(data)
            valid_masks.append(valid)
    full = np.stack(stack, axis=0)
    valid = np.stack(valid_masks, axis=0)
    return full, valid


def rasterize_mask(clip_gdf: gpd.GeoDataFrame, transform: Affine, width: int, height: int, dst_crs) -> np.ndarray:
    """Растеризует AOI в булеву маску в координатах расчетной сетки."""
    if clip_gdf.crs != dst_crs:
        clip_gdf = clip_gdf.to_crs(dst_crs)
    geoms = [mapping(g) for g in clip_gdf.geometry]
    mask = features.geometry_mask(geoms, out_shape=(height, width), transform=transform, invert=True)
    return mask


def median_composite(stack: np.ndarray, valid: np.ndarray, clip_mask: np.ndarray) -> np.ndarray:
    """Строит медианный композит по стеку сцен с учетом валидности и маски AOI."""
    comp = np.nanmedian(stack, axis=0).astype(np.float32)
    comp[:, ~clip_mask] = np.nan
    missing = ~np.isfinite(comp).all(axis=0)
    comp[:, missing] = np.nan
    return comp


def write_multiband(path: Path, arr: np.ndarray, transform: Affine, crs, nodata: float = np.nan) -> None:
    """Сохраняет многоканальный GeoTIFF со стандартными параметрами компрессии."""
    safe_mkdir(path.parent)
    meta = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(np.float32))


def write_singleband(path: Path, arr: np.ndarray, transform: Affine, crs, nodata: float = np.nan) -> None:
    """Упрощенная обертка для сохранения одноканального растрового слоя."""
    write_multiband(path, arr[np.newaxis, ...], transform, crs, nodata=nodata)


def eps(n: float = 1e-6) -> float:
    """Возвращает малую константу для защиты от деления на ноль."""
    return n


# --- Этап 4. Расчет спектральных, текстурных и рельефных признаков. ---
def compute_indices(comp: np.ndarray, soil_index: str = "osavi") -> Dict[str, np.ndarray]:
    """Рассчитывает базовые спектральные индексы и простые отношения каналов."""
    blue, green, red, nir = [comp[i].astype(np.float32) for i in range(4)]
    ndvi = (nir - red) / (nir + red + eps())
    ndwi = (green - nir) / (green + nir + eps())
    if soil_index.lower() == "savi":
        savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps())
        soil_name = "savi"
        soil_arr = savi
    else:
        osavi = 1.16 * (nir - red) / (nir + red + 0.16 + eps())
        soil_name = "osavi"
        soil_arr = osavi
    nir_red = nir / (red + eps())
    red_green = red / (green + eps())
    brightness = np.nanmean(np.stack([blue, green, red, nir], axis=0), axis=0).astype(np.float32)
    out = {
        "ndvi": ndvi.astype(np.float32),
        "ndwi": ndwi.astype(np.float32),
        soil_name: soil_arr.astype(np.float32),
        "nir_red_ratio": nir_red.astype(np.float32),
        "red_green_ratio": red_green.astype(np.float32),
        "brightness": brightness,
    }
    return out


def compute_pc1(comp: np.ndarray) -> np.ndarray:
    """Считает первую главную компоненту по многоканальному композиту."""
    bands, h, w = comp.shape
    X = comp.reshape(bands, -1).T
    mask = np.isfinite(X).all(axis=1)
    out = np.full(h * w, np.nan, dtype=np.float32)
    if mask.sum() < 10:
        return out.reshape(h, w)
    Xv = X[mask]
    Xv = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + 1e-6)
    cov = np.cov(Xv, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    pc1 = Xv @ vecs[:, np.argmax(vals)]
    out[mask] = pc1.astype(np.float32)
    return out.reshape(h, w)


def local_mean(arr: np.ndarray, size: int) -> np.ndarray:
    """Считает локальное среднее в окне фиксированного размера."""
    valid = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    sm = ndi.uniform_filter(filled, size=size, mode="nearest")
    cnt = ndi.uniform_filter(valid, size=size, mode="nearest")
    out = sm / np.maximum(cnt, 1e-6)
    out[cnt <= 0] = np.nan
    return out.astype(np.float32)


def local_std_var(arr: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Считает локальное стандартное отклонение и дисперсию."""
    mean = local_mean(arr, size)
    sq = local_mean(arr ** 2, size)
    var = np.maximum(sq - mean ** 2, 0.0).astype(np.float32)
    std = np.sqrt(var).astype(np.float32)
    return std, var


def local_range(arr: np.ndarray, size: int) -> np.ndarray:
    """Считает локальный размах значений в окне."""
    maxv = ndi.maximum_filter(np.where(np.isfinite(arr), arr, -np.inf), size=size, mode="nearest")
    minv = ndi.minimum_filter(np.where(np.isfinite(arr), arr, np.inf), size=size, mode="nearest")
    out = (maxv - minv).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def laplacian(arr: np.ndarray) -> np.ndarray:
    """Считает лапласиан по растру как меру локальной кривизны/резкости изменений."""
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    out = ndi.laplace(filled, mode="nearest").astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    """Считает модуль градиента по двум направлениям."""
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    gx = ndi.sobel(filled, axis=1, mode="nearest")
    gy = ndi.sobel(filled, axis=0, mode="nearest")
    out = np.hypot(gx, gy).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def quantize_valid(arr: np.ndarray, levels: int = 16) -> np.ndarray:
    """Квантует непрерывный растр в ограниченное число уровней для расчета GLCM."""
    vals = arr[np.isfinite(arr)]
    out = np.zeros(arr.shape, dtype=np.uint8)
    if vals.size == 0:
        return out
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    scaled[~np.isfinite(arr)] = 0
    out = np.round(scaled * (levels - 1)).astype(np.uint8)
    return out


def glcm_metrics(
    arr: np.ndarray,
    window: int,
    levels: int = 16,
    downsample: int = 2,
    stride: int = 2,
) -> Dict[str, np.ndarray]:
    """Рассчитывает GLCM-метрики текстуры в локальном окне и возвращает их как растры."""
    arr = arr.astype(np.float32)
    if downsample > 1:
        arr_small = arr[::downsample, ::downsample]
    else:
        arr_small = arr
    q = quantize_valid(arr_small, levels=levels)
    h, w = q.shape
    out_shape = (h, w)
    contrast = np.full(out_shape, np.nan, dtype=np.float32)
    homogeneity = np.full(out_shape, np.nan, dtype=np.float32)
    dissimilarity = np.full(out_shape, np.nan, dtype=np.float32)
    entropy = np.full(out_shape, np.nan, dtype=np.float32)
    if h < window or w < window:
        return {
            "contrast": np.full(arr.shape, np.nan, dtype=np.float32),
            "homogeneity": np.full(arr.shape, np.nan, dtype=np.float32),
            "dissimilarity": np.full(arr.shape, np.nan, dtype=np.float32),
            "entropy": np.full(arr.shape, np.nan, dtype=np.float32),
        }
    views = view_as_windows(q, (window, window), step=stride)
    coords = []
    metrics = []
    for i in range(views.shape[0]):
        for j in range(views.shape[1]):
            win = views[i, j]
            glcm = graycomatrix(
                win,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=levels,
                symmetric=True,
                normed=True,
            )
            c = graycoprops(glcm, "contrast").mean()
            hmg = graycoprops(glcm, "homogeneity").mean()
            dis = graycoprops(glcm, "dissimilarity").mean()
            p = glcm.astype(np.float64)
            p = p / max(p.sum(), 1e-12)
            ent = -(p * np.log2(np.clip(p, 1e-12, None))).sum()
            coords.append((i * stride + window // 2, j * stride + window // 2))
            metrics.append((c, hmg, dis, ent))
    for (r, c), (v1, v2, v3, v4) in zip(coords, metrics):
        contrast[r, c] = v1
        homogeneity[r, c] = v2
        dissimilarity[r, c] = v3
        entropy[r, c] = v4
    filled_metrics = []
    for tgt in [contrast, homogeneity, dissimilarity, entropy]:
        mask = np.isfinite(tgt)
        if not mask.any():
            filled = np.full_like(arr_small, np.nan, dtype=np.float32)
        else:
            idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            filled = tgt[tuple(idx)].astype(np.float32)
        if downsample > 1:
            zoom_f = (arr.shape[0] / filled.shape[0], arr.shape[1] / filled.shape[1])
            filled = ndi.zoom(filled, zoom_f, order=1).astype(np.float32)
            filled = filled[: arr.shape[0], : arr.shape[1]]
        filled_metrics.append(filled.astype(np.float32))
    return {
        "contrast": filled_metrics[0],
        "homogeneity": filled_metrics[1],
        "dissimilarity": filled_metrics[2],
        "entropy": filled_metrics[3],
    }


def otsu_threshold(arr: np.ndarray) -> float:
    """Оценивает порог по методу Оцу для одномерного распределения значений."""
    vals = arr[np.isfinite(arr)]
    if vals.size < 16:
        return 0.0
    hist, bin_edges = np.histogram(vals, bins=256)
    hist = hist.astype(np.float64)
    prob = hist / np.maximum(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_edges[:-1])
    mu_t = mu[-1]
    sigma = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    idx = np.nanargmax(sigma)
    return float(bin_edges[idx])


# --- Этап 5. Формирование тематических масок и baseline-индекса risk_score. ---
def water_mask_from_ndwi(ndwi: np.ndarray, force_threshold: Optional[float] = None) -> np.ndarray:
    """Строит бинарную маску воды по NDWI и автоматическому или ручному порогу."""
    thr = force_threshold if force_threshold is not None else otsu_threshold(ndwi)
    out = np.zeros(ndwi.shape, dtype=np.uint8)
    out[np.isfinite(ndwi) & (ndwi > thr)] = 1
    return out


def terrain_from_dem(
    dem_path: Path,
    transform: Affine,
    width: int,
    height: int,
    dst_crs,
    tri_window: int = 3,
    tpi_window: int = 11,
) -> Dict[str, np.ndarray]:
    """Перепроецирует DEM на общую сетку и рассчитывает terrain-производные."""
    with rasterio.open(dem_path) as src:
        dem = np.full((height, width), np.nan, dtype=np.float32)
        src_arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            src_arr[src_arr == nodata] = np.nan
        reproject(
            source=src_arr,
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    dx = abs(transform.a)
    dy = abs(transform.e)
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem)).astype(np.float32)
    gy, gx = np.gradient(filled, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    aspect_sin = np.sin(aspect).astype(np.float32)
    aspect_cos = np.cos(aspect).astype(np.float32)
    curv = laplacian(dem)
    local_dem_mean = local_mean(dem, tpi_window)
    tpi = (dem - local_dem_mean).astype(np.float32)
    maxf = ndi.maximum_filter(filled, size=tri_window, mode="nearest")
    minf = ndi.minimum_filter(filled, size=tri_window, mode="nearest")
    roughness = (maxf - minf).astype(np.float32)
    neigh = ndi.generic_filter(filled, function=lambda x: np.sqrt(np.mean((x - x[x.size // 2]) ** 2)), size=tri_window, mode="nearest")
    tri = neigh.astype(np.float32)
    slope_deg = np.degrees(slope).astype(np.float32)
    for arr in [slope_deg, aspect_sin, aspect_cos, curv, tpi, roughness, tri]:
        arr[~np.isfinite(dem)] = np.nan
    return {
        "dem": dem.astype(np.float32),
        "slope": slope_deg,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
        "curvature": curv.astype(np.float32),
        "tpi": tpi,
        "roughness": roughness,
        "tri": tri,
    }


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """Размечает связные компоненты в бинарной маске и возвращает их размеры."""
    labels, n = ndi.label(mask.astype(bool))
    if n == 0:
        return labels.astype(np.int32), {}
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    return labels.astype(np.int32), {int(i): int(c) for i, c in zip(ids, counts)}


def zone_stats(values: np.ndarray) -> Dict[str, float]:
    """Считает агрегированные статистики по зоне: mean, std, min, max, p90, p95."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p90": np.nan,
            "p95": np.nan,
        }
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
    }


# --- Этап 6. Parcel-level агрегация признаков по участкам. ---
def zonal_table(
    parcels: gpd.GeoDataFrame,
    raster_layers: Dict[str, np.ndarray],
    mask_layers: Dict[str, np.ndarray],
    hotspot_labels: np.ndarray,
    hotspot_sizes: Dict[int, int],
    transform: Affine,
    crs,
) -> pd.DataFrame:
    """Формирует parcel-level таблицу статистик по растру и тематическим маскам."""
    rows = []
    if parcels.crs != crs:
        parcels = parcels.to_crs(crs)
    pixel_area = abs(transform.a * transform.e)
    for idx, row in parcels.iterrows():
        geom_mask = features.geometry_mask([mapping(row.geometry)], out_shape=next(iter(raster_layers.values())).shape, transform=transform, invert=True)
        rec = {"parcel_id": row.get("parcel_id", idx + 1)}
        for key, arr in raster_layers.items():
            st = zone_stats(np.where(geom_mask, arr, np.nan))
            for k, v in st.items():
                rec[f"{key}_{k}"] = v
        for key, arr in mask_layers.items():
            pix = arr[geom_mask]
            rec[f"{key}_share"] = float(np.nanmean(pix)) if pix.size else np.nan
        labels = hotspot_labels[geom_mask]
        ids = [int(i) for i in np.unique(labels) if i > 0]
        rec["hotspot_count"] = int(len(ids))
        if ids:
            sizes = [hotspot_sizes[i] * pixel_area for i in ids]
            rec["mean_hotspot_size_m2"] = float(np.mean(sizes))
        else:
            rec["mean_hotspot_size_m2"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def subtract_or_nan(curr: Optional[np.ndarray], prev: Optional[np.ndarray]) -> np.ndarray:
    """Вычитает предыдущий слой из текущего, либо возвращает NaN-слой, если пары нет."""
    if curr is None or prev is None:
        shape = curr.shape if curr is not None else prev.shape
        return np.full(shape, np.nan, dtype=np.float32)
    return (curr - prev).astype(np.float32)


def normalize_component(arr: Optional[np.ndarray], invert: bool = False) -> np.ndarray:
    """Нормирует компоненту риска и при необходимости инвертирует шкалу."""
    if arr is None:
        return np.array([], dtype=np.float32)
    out = robust_minmax(arr)
    if invert:
        mask = np.isfinite(out)
        out[mask] = 1.0 - out[mask]
    return out.astype(np.float32)


def combine_risk(components: Dict[str, np.ndarray]) -> np.ndarray:
    """Объединяет нормированные компоненты в итоговый risk_score по весам."""
    ref = next(iter(components.values()))
    total = np.zeros(ref.shape, dtype=np.float32)
    weight_sum = np.zeros(ref.shape, dtype=np.float32)
    for name, arr in components.items():
        w = float(RISK_WEIGHTS.get(name, 0.0))
        if arr.size == 0 or w <= 0:
            continue
        mask = np.isfinite(arr)
        total[mask] += arr[mask] * w
        weight_sum[mask] += w
    out = np.full(ref.shape, np.nan, dtype=np.float32)
    mask = weight_sum > 0
    out[mask] = total[mask] / weight_sum[mask]
    return out


def write_manifest(path: Path, items: Sequence[str]) -> None:
    """Сохраняет список сформированных файлов в простой текстовый manifest."""
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def composite_for_group(
    group_df: pd.DataFrame,
    clip_gdf: gpd.GeoDataFrame,
    out_path: Path,
    band_map: Dict[str, int],
) -> Tuple[np.ndarray, Affine, object]:
    """Полный цикл построения композита для одного набора сцен."""
    scene_paths = group_df["tif_path"].tolist()
    transform, width, height, dst_crs = build_reference_grid(scene_paths, clip_gdf)
    clip_mask = rasterize_mask(clip_gdf, transform, width, height, dst_crs)
    stack, valid = read_reprojected_stack(scene_paths, transform, width, height, dst_crs, band_map=band_map)
    comp = median_composite(stack, valid, clip_mask)
    write_multiband(out_path, comp, transform, dst_crs)
    return comp, transform, dst_crs


def save_layer_dict(layer_dict: Dict[str, np.ndarray], out_dir: Path, stem_prefix: str, transform: Affine, crs) -> List[str]:
    """Сохраняет словарь растровых слоев на диск и возвращает список созданных файлов."""
    outputs = []
    for name, arr in layer_dict.items():
        path = out_dir / f"{stem_prefix}_{name}.tif"
        write_singleband(path, arr.astype(np.float32), transform, crs)
        outputs.append(str(path))
    return outputs


# --- Главный конвейер обработки всего проекта. ---
def build_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    """Основной конвейер обработки: от сцен и AOI до risk_score и parcel-level аналитики."""
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "catalog")
    safe_mkdir(out_dir / "aoi")
    safe_mkdir(out_dir / "composites")
    safe_mkdir(out_dir / "terrain")
    safe_mkdir(out_dir / "indices")
    safe_mkdir(out_dir / "textures")
    safe_mkdir(out_dir / "dynamics")
    safe_mkdir(out_dir / "masks")
    safe_mkdir(out_dir / "analytics")
    safe_mkdir(out_dir / "parcels")

    # Если передан inventory CSV, строим отдельный каталог и сводный отчет по поставке.
    inventory_catalog = build_inventory_report(Path(args.inventory_csv) if args.inventory_csv else None, out_dir / "catalog")

    # Сканируем реальную файловую структуру и собираем каталог сцен.
    rows = discover_scene_dirs(Path(args.data_root))
    catalog = catalog_to_df(rows)
    if catalog.empty:
        raise ValueError("Не найдено ни одной валидной сцены под указанным data-root.")
    catalog.to_csv(out_dir / "catalog" / "scene_catalog_scan.csv", index=False, encoding="utf-8-sig")

    # Оставляем только лучшие мультиспектральные сцены по каждой дате и времени съемки.
    selected = choose_best_ms_scenes(catalog)
    selected["date"] = pd.to_datetime(selected["date"])
    selected["season"] = selected["date"].apply(lambda x: assign_season(x, DEFAULT_SEASONS))
    selected.to_csv(out_dir / "catalog" / "scene_catalog_selected_ms.csv", index=False, encoding="utf-8-sig")

    # По GBD-файлам строим footprint сцен и из них формируем рабочую AOI.
    footprints = read_footprints(selected)
    aoi_area, aoi_union = dissolve_aoi(footprints)
    save_vector(footprints, out_dir / "aoi" / "scene_footprints.gpkg")
    save_vector(aoi_area, out_dir / "aoi" / "area_aoi.gpkg")
    save_vector(aoi_union, out_dir / "aoi" / "aoi_union.geojson")

    # Если передан слой участков, обрезаем его по union AOI и готовим к parcel-level аналитике.
    parcels = clip_parcels_to_aoi(Path(args.parcel_mask) if args.parcel_mask else None, aoi_union)
    if parcels is not None:
        save_vector(parcels, out_dir / "parcels" / "parcels_clipped.gpkg")

    band_map = {"blue": 1, "green": 2, "red": 3, "nir": 4}
    if args.band_map:
        band_map = json.loads(args.band_map)

    report: Dict[str, object] = {
        "areas": sorted(selected["area"].dropna().unique().tolist()),
        "inventory_catalog_rows": 0 if inventory_catalog is None else int(len(inventory_catalog)),
        "scanned_scene_rows": int(len(catalog)),
        "selected_ms_rows": int(len(selected)),
        "outputs": [],
    }

    area_year_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    area_year_masks: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    area_year_meta: Dict[Tuple[str, int], Dict[str, object]] = {}
    area_year_textures: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    for area in sorted(selected["area"].dropna().unique()):
        area_aoi = aoi_area[aoi_area["area"] == area]
        area_df = selected[selected["area"] == area].copy()
        years = sorted(area_df["date"].dt.year.unique().tolist())
        for year in years:
            group = area_df[area_df["date"].dt.year == year].copy()
            if group.empty:
                continue
            # Для каждой территории и года строим медианный годовой композит.
            comp_path = out_dir / "composites" / f"{area}_{year}_annual_ms_composite.tif"
            comp, transform, crs = composite_for_group(group, area_aoi, comp_path, band_map)
            report["outputs"].append(str(comp_path))
            # Сразу после композита рассчитываем базовые спектральные индексы.
            idx = compute_indices(comp, soil_index=args.soil_index)
            idx_dir = out_dir / "indices"
            report["outputs"].extend(save_layer_dict(idx, idx_dir, f"{area}_{year}_annual", transform, crs))
            pc1 = compute_pc1(comp) if args.compute_pc1 else None
            tex_input = {
                "nir": comp[3],
                "red": comp[2],
            }
            if pc1 is not None:
                tex_input["pc1"] = pc1
            # Текстурные признаки считаются по NIR, Red и при необходимости по PC1.
            tex_layers: Dict[str, np.ndarray] = {}
            for src_name, src_arr in tex_input.items():
                for win in [5, 7]:
                    std, var = local_std_var(src_arr, win)
                    tex_layers[f"{src_name}_local_std_w{win}"] = std
                    tex_layers[f"{src_name}_local_var_w{win}"] = var
                    glcm = glcm_metrics(src_arr, window=win, levels=args.texture_levels, downsample=args.texture_downsample, stride=args.texture_stride)
                    for gname, garr in glcm.items():
                        tex_layers[f"{src_name}_glcm_{gname}_w{win}"] = garr
            area_year_textures[(area, year)] = tex_layers
            report["outputs"].extend(save_layer_dict(tex_layers, out_dir / "textures", f"{area}_{year}", transform, crs))

            # При наличии DEM рассчитываем terrain-производные.
            if args.dem:
                terrain = terrain_from_dem(Path(args.dem), transform, comp.shape[2], comp.shape[1], crs)
                report["outputs"].extend(save_layer_dict(terrain, out_dir / "terrain", f"{area}_{year}", transform, crs))
            else:
                terrain = {}

            # Водяная маска строится по NDWI и затем используется дальше в динамике и risk_score.
            water_mask = water_mask_from_ndwi(idx["ndwi"], force_threshold=args.water_threshold)
            area_year_cache[(area, year)] = {
                "comp_blue": comp[0],
                "comp_green": comp[1],
                "comp_red": comp[2],
                "comp_nir": comp[3],
                **idx,
                **terrain,
            }
            area_year_masks[(area, year)] = {
                "water_mask": water_mask.astype(np.float32),
            }
            area_year_meta[(area, year)] = {"transform": transform, "crs": crs}

            write_singleband(out_dir / "masks" / f"{area}_{year}_water_mask.tif", water_mask.astype(np.float32), transform, crs)
            report["outputs"].append(str(out_dir / "masks" / f"{area}_{year}_water_mask.tif"))

            # Дополнительно считаем сезонные композиты и индексы, если для сезона есть сцены.
            for season in ["early", "mid", "late"]:
                sg = group[group["season"] == season]
                if sg.empty:
                    continue
                s_path = out_dir / "composites" / f"{area}_{year}_{season}_ms_composite.tif"
                s_comp, s_transform, s_crs = composite_for_group(sg, area_aoi, s_path, band_map)
                s_idx = compute_indices(s_comp, soil_index=args.soil_index)
                report["outputs"].append(str(s_path))
                report["outputs"].extend(save_layer_dict(s_idx, out_dir / "indices", f"{area}_{year}_{season}", s_transform, s_crs))

        prev_key = None
        water_masks = []
        year_order = sorted([k[1] for k in area_year_cache.keys() if k[0] == area])
        for year in year_order:
            curr_key = (area, year)
            curr = area_year_cache[curr_key]
            curr_mask = area_year_masks[curr_key]
            transform = area_year_meta[curr_key]["transform"]
            crs = area_year_meta[curr_key]["crs"]

            if prev_key is None:
                d_ndvi = np.full(curr["ndvi"].shape, np.nan, dtype=np.float32)
                d_ndwi = np.full(curr["ndwi"].shape, np.nan, dtype=np.float32)
                water_growth = np.full(curr["ndwi"].shape, np.nan, dtype=np.float32)
            else:
                prev = area_year_cache[prev_key]
                prev_mask = area_year_masks[prev_key]
                d_ndvi = subtract_or_nan(curr["ndvi"], prev["ndvi"])
                d_ndwi = subtract_or_nan(curr["ndwi"], prev["ndwi"])
                water_growth = (curr_mask["water_mask"] - prev_mask["water_mask"]).astype(np.float32)
                write_singleband(out_dir / "dynamics" / f"{area}_{prev_key[1]}_{year}_delta_ndvi.tif", d_ndvi, transform, crs)
                write_singleband(out_dir / "dynamics" / f"{area}_{prev_key[1]}_{year}_delta_ndwi.tif", d_ndwi, transform, crs)
                write_singleband(out_dir / "dynamics" / f"{area}_{prev_key[1]}_{year}_water_growth.tif", water_growth, transform, crs)
                report["outputs"].extend([
                    str(out_dir / "dynamics" / f"{area}_{prev_key[1]}_{year}_delta_ndvi.tif"),
                    str(out_dir / "dynamics" / f"{area}_{prev_key[1]}_{year}_delta_ndwi.tif"),
                    str(out_dir / "dynamics" / f"{area}_{prev_key[1]}_{year}_water_growth.tif"),
                ])

            water_masks.append(curr_mask["water_mask"])
            tex = area_year_textures[curr_key]
            texture_driver = tex.get("nir_local_std_w7")
            if texture_driver is None:
                texture_driver = next(iter(tex.values()))
            # Выделяем зоны повышенной текстурной аномальности по верхнему квантилю.
            tex_q = nanpercentile(texture_driver, CORE_MASK_THRESHOLDS["texture_anomaly_quantile"] * 100)
            tex_mask = (texture_driver >= tex_q).astype(np.uint8)
            area_year_masks[curr_key]["high_texture_anomaly_mask"] = tex_mask.astype(np.float32)
            write_singleband(out_dir / "masks" / f"{area}_{year}_texture_anomaly_mask.tif", tex_mask.astype(np.float32), transform, crs)
            report["outputs"].append(str(out_dir / "masks" / f"{area}_{year}_texture_anomaly_mask.tif"))

            # Считаем накопленную водную встречаемость и производную persistence_water_mask.
            occurrence = np.nanmean(np.stack(water_masks, axis=0), axis=0).astype(np.float32)
            persistence = (occurrence >= CORE_MASK_THRESHOLDS["water_occurrence_persistence"]).astype(np.uint8)
            area_year_masks[curr_key]["water_occurrence"] = occurrence
            area_year_masks[curr_key]["persistence_water_mask"] = persistence.astype(np.float32)
            write_singleband(out_dir / "dynamics" / f"{area}_{year}_water_occurrence.tif", occurrence, transform, crs)
            write_singleband(out_dir / "masks" / f"{area}_{year}_persistence_water_mask.tif", persistence.astype(np.float32), transform, crs)
            report["outputs"].extend([
                str(out_dir / "dynamics" / f"{area}_{year}_water_occurrence.tif"),
                str(out_dir / "masks" / f"{area}_{year}_persistence_water_mask.tif"),
            ])

            # Change-mask формируется по сильным межгодовым изменениям NDVI/NDWI и приросту воды.
            if np.isfinite(d_ndvi).any() or np.isfinite(d_ndwi).any():
                dn_thr = nanpercentile(np.abs(d_ndvi), 90)
                dw_thr = nanpercentile(np.abs(d_ndwi), 90)
                change_mask = (
                    ((np.abs(d_ndvi) >= dn_thr) & np.isfinite(d_ndvi))
                    | ((np.abs(d_ndwi) >= dw_thr) & np.isfinite(d_ndwi))
                    | (np.abs(water_growth) > 0)
                ).astype(np.uint8)
            else:
                change_mask = np.zeros(curr["ndvi"].shape, dtype=np.uint8)
            area_year_masks[curr_key]["change_mask"] = change_mask.astype(np.float32)
            write_singleband(out_dir / "masks" / f"{area}_{year}_change_mask.tif", change_mask.astype(np.float32), transform, crs)
            report["outputs"].append(str(out_dir / "masks" / f"{area}_{year}_change_mask.tif"))

            terrain_driver = None
            if "slope" in curr and "tpi" in curr:
                terrain_driver = robust_minmax(np.nanmean(np.stack([curr["slope"], np.abs(curr["tpi"])], axis=0), axis=0))
            elif "slope" in curr:
                terrain_driver = robust_minmax(curr["slope"])
            else:
                terrain_driver = np.zeros(curr["ndvi"].shape, dtype=np.float32)

            tmp_stack = np.stack([np.abs(d_ndvi), np.abs(d_ndwi)], axis=0)
            valid_cnt = np.isfinite(tmp_stack).sum(axis=0)
            tmp_sum = np.nansum(tmp_stack, axis=0)
            tmp_instability = np.full(tmp_sum.shape, np.nan, dtype=np.float32)
            tmp_instability[valid_cnt > 0] = (tmp_sum[valid_cnt > 0] / valid_cnt[valid_cnt > 0]).astype(np.float32)

            # Составляем нормированные компоненты авторского индекса риска.
            comps = {
                "water_growth": normalize_component(np.maximum(water_growth, 0.0)),
                "vegetation_loss": normalize_component(-d_ndvi),
                "texture_anomaly": normalize_component(texture_driver),
                "terrain_susceptibility": terrain_driver.astype(np.float32),
                "temporal_instability": normalize_component(tmp_instability),
            }
            # Итоговый risk_score объединяет все компоненты с заданными весами.
            risk = combine_risk(comps)
            rq = nanpercentile(risk, CORE_MASK_THRESHOLDS["risk_hotspot_quantile"] * 100)
            hotspot_mask = (risk >= rq).astype(np.uint8)
            labels, sizes = connected_components(hotspot_mask)
            area_year_masks[curr_key]["hotspot_mask"] = hotspot_mask.astype(np.float32)
            write_singleband(out_dir / "dynamics" / f"{area}_{year}_risk_score.tif", risk.astype(np.float32), transform, crs)
            write_singleband(out_dir / "masks" / f"{area}_{year}_hotspot_mask.tif", hotspot_mask.astype(np.float32), transform, crs)
            report["outputs"].extend([
                str(out_dir / "dynamics" / f"{area}_{year}_risk_score.tif"),
                str(out_dir / "masks" / f"{area}_{year}_hotspot_mask.tif"),
            ])

            # Если есть полигоны участков, агрегируем по ним все ключевые признаки и маски.
            if parcels is not None:
                raster_layers = {
                    "ndvi": curr["ndvi"],
                    "ndwi": curr["ndwi"],
                    args.soil_index: curr[args.soil_index],
                    "brightness": curr["brightness"],
                    "delta_ndvi": d_ndvi,
                    "delta_ndwi": d_ndwi,
                    "risk_score": risk,
                }
                if "slope" in curr:
                    raster_layers["slope"] = curr["slope"]
                    raster_layers["tpi"] = curr["tpi"]
                    raster_layers["curvature"] = curr["curvature"]
                mask_layers = {
                    "water": curr_mask["water_mask"],
                    "change": change_mask.astype(np.float32),
                    "texture_anomaly": tex_mask.astype(np.float32),
                    "hotspot": hotspot_mask.astype(np.float32),
                    "persistence_water": persistence.astype(np.float32),
                }
                zt = zonal_table(parcels, raster_layers, mask_layers, labels, sizes, transform, crs)
                zt.to_csv(out_dir / "analytics" / f"{area}_{year}_parcel_stats.csv", index=False, encoding="utf-8-sig")
                joined = parcels.merge(zt, on="parcel_id", how="left")
                save_vector(joined, out_dir / "analytics" / f"{area}_{year}_parcel_stats.gpkg")
                report["outputs"].extend([
                    str(out_dir / "analytics" / f"{area}_{year}_parcel_stats.csv"),
                    str(out_dir / "analytics" / f"{area}_{year}_parcel_stats.gpkg"),
                ])

            prev_key = curr_key

    # В конце сохраняем сводный JSON-отчет и текстовый manifest со списком всех артефактов.
    write_json(out_dir / "run_report.json", report)
    write_manifest(out_dir / "manifest.txt", report["outputs"])
    return report


# --- Встроенный self-test для быстрой проверки воспроизводимости контура. ---
def create_test_scene(scene_dir: Path, arr: np.ndarray, transform: Affine, crs="EPSG:4326") -> None:
    """Создает синтетическую тестовую сцену и соответствующий footprint для self-test."""
    safe_mkdir(scene_dir)
    tif = scene_dir / f"{scene_dir.name}.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(arr.astype(np.float32))
    poly = gpd.GeoDataFrame(
        [{"geometry": box(transform.c, transform.f + transform.e * arr.shape[1], transform.c + transform.a * arr.shape[2], transform.f)}],
        crs=crs,
    )
    poly.to_file(scene_dir / f"{scene_dir.name}.GBD.shp")


def run_self_test(base_script: Path) -> Dict[str, object]:
    """Запускает минимальный автономный тест пайплайна на синтетических данных."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_root = tmp / "data"
        parcels_path = tmp / "parcels.geojson"
        dem_path = tmp / "dem.tif"
        out_dir = tmp / "out"
        safe_mkdir(data_root)

        transform = from_bounds(120.0, 60.0, 120.1, 60.1, 48, 48)
        x = np.linspace(0, 1, 48)
        y = np.linspace(0, 1, 48)
        xx, yy = np.meshgrid(x, y)

        def synth(year_shift: float) -> np.ndarray:
            blue = 0.10 + 0.02 * xx + 0.01 * yy
            green = 0.15 + 0.05 * xx
            red = 0.18 + 0.05 * yy + year_shift
            nir = 0.35 + 0.08 * xx - 0.04 * yy - year_shift
            water = ((xx - 0.7) ** 2 + (yy - 0.3) ** 2) < (0.12 + year_shift / 2) ** 2
            nir = nir.copy()
            green = green.copy()
            red = red.copy()
            nir[water] -= 0.20
            green[water] += 0.12
            red[water] -= 0.04
            return np.stack([blue, green, red, nir], axis=0).astype(np.float32)

        s1 = data_root / "102_2026_113_1111111" / "1111111_13.01.26_Амга" / "KV6_00000_00000-01_KANOPUS_20220512_014309_20.L2.MS.SCN01"
        s2 = data_root / "102_2026_113_1111112" / "1111112_13.01.26_Амга" / "KV6_00000_00000-01_KANOPUS_20230509_013320_20.L2.MS.SCN01"
        s3 = data_root / "102_2026_113_1111113" / "1111113_13.01.26_Амга" / "KV6_00000_00000-01_KANOPUS_20230703_013706_20.L2.MS.SCN01"

        create_test_scene(s1, synth(0.00), transform)
        create_test_scene(s2, synth(0.03), transform)
        create_test_scene(s3, synth(0.05), transform)

        parcels = gpd.GeoDataFrame(
            [
                {"parcel_id": 1, "geometry": box(120.01, 60.01, 120.05, 60.05)},
                {"parcel_id": 2, "geometry": box(120.05, 60.02, 120.09, 60.08)},
            ],
            crs="EPSG:4326",
        )
        parcels.to_file(parcels_path, driver="GeoJSON")

        dem = (150 + 10 * xx + 20 * yy + 2 * np.sin(xx * 10) * np.cos(yy * 8)).astype(np.float32)
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            height=dem.shape[0],
            width=dem.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(dem[np.newaxis, ...])

        args = argparse.Namespace(
            data_root=str(data_root),
            inventory_csv=None,
            parcel_mask=str(parcels_path),
            out_dir=str(out_dir),
            dem=str(dem_path),
            band_map=None,
            soil_index="osavi",
            compute_pc1=True,
            texture_levels=16,
            texture_downsample=2,
            texture_stride=2,
            water_threshold=None,
        )
        report = build_pipeline(args)
        must_exist = [
            out_dir / "catalog" / "scene_catalog_scan.csv",
            out_dir / "aoi" / "aoi_union.geojson",
            out_dir / "composites" / "Амга_2022_annual_ms_composite.tif",
            out_dir / "indices" / "Амга_2022_annual_ndvi.tif",
            out_dir / "terrain" / "Амга_2022_slope.tif",
            out_dir / "masks" / "Амга_2023_hotspot_mask.tif",
            out_dir / "analytics" / "Амга_2023_parcel_stats.csv",
            out_dir / "run_report.json",
        ]
        missing = [str(p) for p in must_exist if not p.exists()]
        if missing:
            raise RuntimeError("Self-test failed. Missing files: " + ", ".join(missing))
        return {"status": "ok", "out_dir": str(out_dir), "outputs": report["outputs"][:8]}


# --- CLI: описание параметров запуска. ---
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Описывает аргументы командной строки для обычного запуска и self-test."""
    parser = argparse.ArgumentParser(prog="gisit_permafrost_data_pipeline.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--data-root", required=True)
    p_run.add_argument("--inventory-csv")
    p_run.add_argument("--parcel-mask")
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--dem")
    p_run.add_argument("--band-map", help='JSON, например {"blue":1,"green":2,"red":3,"nir":4}')
    p_run.add_argument("--soil-index", choices=["savi", "osavi"], default="osavi")
    p_run.add_argument("--compute-pc1", action="store_true")
    p_run.add_argument("--texture-levels", type=int, default=16)
    p_run.add_argument("--texture-downsample", type=int, default=2)
    p_run.add_argument("--texture-stride", type=int, default=2)
    p_run.add_argument("--water-threshold", type=float)

    p_test = sub.add_parser("self-test")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа: разбирает аргументы и запускает нужный режим работы."""
    args = parse_args(argv)
    if args.command == "self-test":
        report = run_self_test(Path(__file__))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        report = build_pipeline(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

