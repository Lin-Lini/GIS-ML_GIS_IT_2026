from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

SEARCH_PATTERNS = ["*.GBD.shp", "*.gbd.shp", "*.Shp", "*.SHP"]
SCENE_EXTS = {".zip", ".7z", ".rar"}


@dataclass
class SceneFootprintRecord:
    scene_id: str
    source_type: str
    source_path: str
    gbd_path: str
    feature_count: int
    geometry_count: int
    area_km2: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Собирает AOI по всем footprint-слоям GBD из сцен и разбивает покрытие на независимые зоны."
    )
    p.add_argument("--data-root", required=True, help="Корневая папка с данными организатора")
    p.add_argument("--out-dir", required=True, help="Куда сохранить scene footprints, AOI zones и таблицы")
    p.add_argument(
        "--target-crs",
        default="EPSG:4326",
        help="CRS для финального экспорта footprints/zones (по умолчанию EPSG:4326)",
    )
    p.add_argument(
        "--metric-crs",
        default="EPSG:6933",
        help="Метрика для площадей/буферов/кластеризации (по умолчанию EPSG:6933)",
    )
    p.add_argument(
        "--merge-gap-m",
        type=float,
        default=0.0,
        help="Буфер в метрах для склейки почти соприкасающихся покрытий в одну зону. 0 = не склеивать.",
    )
    p.add_argument(
        "--min-zone-area-km2",
        type=float,
        default=0.01,
        help="Отсечь очень мелкие артефакты после dissolve/explode. По умолчанию 0.01 км².",
    )
    p.add_argument(
        "--prefer-zips",
        action="store_true",
        help="Если есть и архив, и распакованная папка одной сцены, в первую очередь брать архив.",
    )
    return p.parse_args()


def normalize_geom(geom):
    if geom is None or geom.is_empty:
        return None
    geom = make_valid(geom)
    if geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return None
        return unary_union(polys)
    return None


def find_gbd_in_dir(root: Path) -> List[Path]:
    hits: List[Path] = []
    for pattern in SEARCH_PATTERNS:
        hits.extend(root.rglob(pattern))
    # убираем дубли при пересечении шаблонов
    uniq = []
    seen = set()
    for p in sorted(hits):
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def candidate_scene_dirs(data_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in data_root.rglob("*"):
        if p.is_dir():
            gbd = find_gbd_in_dir(p)
            if gbd:
                out.append(p)
    # оставляем только минимальные каталоги, где есть GBD, чтобы не брать всех родителей
    out_sorted = sorted(out, key=lambda x: len(x.parts))
    keep: List[Path] = []
    for p in out_sorted:
        if not any(parent in keep for parent in p.parents):
            keep.append(p)
    return keep


def candidate_archives(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"])


def find_gbd_members(zf: zipfile.ZipFile) -> List[str]:
    names = zf.namelist()
    hits = [n for n in names if n.lower().endswith(".gbd.shp")]
    return sorted(hits)


def extract_shapefile_bundle(zf: zipfile.ZipFile, shp_member: str, tmpdir: Path) -> Path:
    base = shp_member[:-4]
    need_suffixes = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    extracted = None
    names = set(zf.namelist())
    for suf in need_suffixes:
        member = base + suf
        if member in names:
            zf.extract(member, path=tmpdir)
            if suf == ".shp":
                extracted = tmpdir / member
    if extracted is None or not extracted.exists():
        raise FileNotFoundError(f"Не удалось извлечь shapefile bundle для {shp_member}")
    return extracted


def read_gbd(shp_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError(f"GBD без CRS: {shp_path}")
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        return gdf
    gdf["geometry"] = gdf.geometry.apply(normalize_geom)
    gdf = gdf[gdf.geometry.notna()].copy()
    return gdf


def dissolve_scene_geometry(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    geom = normalize_geom(geom)
    return geom


def pick_scene_id(source_path: Path, gbd_path_str: str) -> str:
    name = source_path.stem if source_path.is_file() else source_path.name
    if name.lower().endswith(".gbd"):
        name = name[:-4]
    if name in {"MS", "PAN", "PMS", "SCN01", "SCN02"}:
        parts = Path(gbd_path_str).parts
        for part in reversed(parts):
            if part.upper().startswith("KANOPUS") or part.startswith("102_"):
                return part.rsplit(".", 1)[0]
    return name


def scene_from_dir(scene_dir: Path, metric_crs: str) -> Optional[tuple[SceneFootprintRecord, object, str]]:
    gbd_files = find_gbd_in_dir(scene_dir)
    if not gbd_files:
        return None
    best = gbd_files[0]
    gdf = read_gbd(best)
    geom = dissolve_scene_geometry(gdf)
    if geom is None:
        return None
    tmp = gpd.GeoDataFrame({"geometry": [geom]}, crs=gdf.crs).to_crs(metric_crs)
    area_km2 = float(tmp.geometry.area.iloc[0] / 1_000_000.0)
    rec = SceneFootprintRecord(
        scene_id=pick_scene_id(scene_dir, str(best)),
        source_type="dir",
        source_path=str(scene_dir),
        gbd_path=str(best),
        feature_count=int(len(gdf)),
        geometry_count=1,
        area_km2=area_km2,
    )
    return rec, geom, gdf.crs.to_string()


def scene_from_zip(zip_path: Path, metric_crs: str) -> Optional[tuple[SceneFootprintRecord, object, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        members = find_gbd_members(zf)
        if not members:
            return None
        shp_member = members[0]
        with tempfile.TemporaryDirectory() as td:
            shp = extract_shapefile_bundle(zf, shp_member, Path(td))
            gdf = read_gbd(shp)
            geom = dissolve_scene_geometry(gdf)
            if geom is None:
                return None
            tmp = gpd.GeoDataFrame({"geometry": [geom]}, crs=gdf.crs).to_crs(metric_crs)
            area_km2 = float(tmp.geometry.area.iloc[0] / 1_000_000.0)
            rec = SceneFootprintRecord(
                scene_id=pick_scene_id(zip_path, shp_member),
                source_type="zip",
                source_path=str(zip_path),
                gbd_path=shp_member,
                feature_count=int(len(gdf)),
                geometry_count=1,
                area_km2=area_km2,
            )
            return rec, geom, gdf.crs.to_string()


def build_scene_footprints(data_root: Path, metric_crs: str, prefer_zips: bool) -> gpd.GeoDataFrame:
    rows = []
    geoms = []
    used = set()

    dirs = candidate_scene_dirs(data_root)
    zips = candidate_archives(data_root)

    ordered: List[Path] = []
    if prefer_zips:
        ordered.extend(zips)
        ordered.extend(dirs)
    else:
        ordered.extend(dirs)
        ordered.extend(zips)

    for p in ordered:
        try:
            result = scene_from_zip(p, metric_crs) if p.is_file() else scene_from_dir(p, metric_crs)
        except Exception:
            continue
        if result is None:
            continue
        rec, geom, crs = result
        key = (rec.scene_id, rec.source_type)
        if rec.scene_id in used:
            continue
        used.add(rec.scene_id)
        rows.append(asdict(rec) | {"src_crs": crs})
        geoms.append(geom)

    if not rows:
        raise RuntimeError("Не найдено ни одного корректного GBD footprint слоя")

    # приводим все геометрии к CRS первого слоя
    base_crs = rows[0]["src_crs"]
    fixed = []
    for row, geom in zip(rows, geoms):
        g = gpd.GeoDataFrame({"geometry": [geom]}, crs=row["src_crs"]) if row["src_crs"] != base_crs else gpd.GeoDataFrame({"geometry": [geom]}, crs=base_crs)
        if row["src_crs"] != base_crs:
            g = g.to_crs(base_crs)
        fixed.append(g.geometry.iloc[0])

    gdf = gpd.GeoDataFrame(rows, geometry=fixed, crs=base_crs)
    return gdf


def build_zones(scene_gdf: gpd.GeoDataFrame, target_crs: str, metric_crs: str, merge_gap_m: float, min_zone_area_km2: float) -> gpd.GeoDataFrame:
    metric = scene_gdf.to_crs(metric_crs)
    union_geom = unary_union([g for g in metric.geometry if g is not None and not g.is_empty])
    union_geom = normalize_geom(union_geom)
    if union_geom is None:
        raise RuntimeError("После union не осталось валидной геометрии")

    if merge_gap_m > 0:
        union_geom = union_geom.buffer(merge_gap_m).buffer(-merge_gap_m)
        union_geom = normalize_geom(union_geom)

    zones = gpd.GeoDataFrame(geometry=[union_geom], crs=metric_crs).explode(index_parts=False).reset_index(drop=True)
    zones["geometry"] = zones.geometry.apply(normalize_geom)
    zones = zones[zones.geometry.notna()].copy()
    zones["area_km2"] = zones.geometry.area / 1_000_000.0
    zones = zones[zones["area_km2"] >= min_zone_area_km2].copy().reset_index(drop=True)
    zones["zone_id"] = [f"ZONE_{i:03d}" for i in range(1, len(zones) + 1)]
    zones = zones[["zone_id", "area_km2", "geometry"]].to_crs(target_crs)
    return zones


def assign_scenes_to_zones(scene_gdf: gpd.GeoDataFrame, zones_gdf: gpd.GeoDataFrame, metric_crs: str) -> pd.DataFrame:
    s = scene_gdf.to_crs(metric_crs).copy()
    z = zones_gdf.to_crs(metric_crs).copy()
    rows = []
    for _, srow in s.iterrows():
        best_zone = None
        best_area = -1.0
        for _, zrow in z.iterrows():
            inter = srow.geometry.intersection(zrow.geometry)
            if inter.is_empty:
                continue
            area = inter.area
            if area > best_area:
                best_area = area
                best_zone = zrow.zone_id
        rows.append({
            "scene_id": srow.scene_id,
            "source_type": srow.source_type,
            "source_path": srow.source_path,
            "zone_id": best_zone,
            "footprint_area_km2": float(srow.geometry.area / 1_000_000.0),
            "intersection_area_km2": float(best_area / 1_000_000.0) if best_zone else 0.0,
        })
    return pd.DataFrame(rows)


def write_outputs(scene_gdf: gpd.GeoDataFrame, zones_gdf: gpd.GeoDataFrame, scene_zone_df: pd.DataFrame, out_dir: Path, target_crs: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    aoi_dir = out_dir / "aoi"
    catalog_dir = out_dir / "catalog"
    aoi_dir.mkdir(exist_ok=True)
    catalog_dir.mkdir(exist_ok=True)

    scene_out = scene_gdf.to_crs(target_crs).copy()
    scene_out.to_file(aoi_dir / "all_scene_footprints.gpkg", driver="GPKG")
    scene_out.to_file(aoi_dir / "all_scene_footprints.geojson", driver="GeoJSON")

    zones_out = zones_gdf.to_crs(target_crs).copy()
    zones_out.to_file(aoi_dir / "all_aoi_zones.gpkg", driver="GPKG")
    zones_out.to_file(aoi_dir / "all_aoi_zones.geojson", driver="GeoJSON")

    union_geom = unary_union(list(zones_out.geometry))
    union_gdf = gpd.GeoDataFrame({"name": ["all_scene_union"], "geometry": [union_geom]}, crs=target_crs)
    union_gdf.to_file(aoi_dir / "all_aoi_union.geojson", driver="GeoJSON")
    union_gdf.to_file(aoi_dir / "all_aoi_union.gpkg", driver="GPKG")

    scene_zone_df.to_csv(catalog_dir / "scene_to_zone.csv", index=False, encoding="utf-8-sig")

    zone_summary = scene_zone_df.groupby("zone_id", dropna=False).agg(
        scene_count=("scene_id", "nunique"),
        footprint_area_km2_sum=("footprint_area_km2", "sum"),
        overlap_area_km2_sum=("intersection_area_km2", "sum"),
    ).reset_index()
    zone_summary = zone_summary.merge(zones_out[["zone_id", "area_km2"]], on="zone_id", how="left")
    zone_summary.to_csv(catalog_dir / "zone_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "scene_count": int(len(scene_out)),
        "zone_count": int(len(zones_out)),
        "target_crs": target_crs,
        "zones": zones_out[["zone_id", "area_km2"]].to_dict(orient="records"),
    }
    (catalog_dir / "aoi_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    if not data_root.exists():
        raise SystemExit(f"Нет data-root: {data_root}")

    scene_gdf = build_scene_footprints(data_root, metric_crs=args.metric_crs, prefer_zips=args.prefer_zips)
    zones_gdf = build_zones(
        scene_gdf,
        target_crs=args.target_crs,
        metric_crs=args.metric_crs,
        merge_gap_m=args.merge_gap_m,
        min_zone_area_km2=args.min_zone_area_km2,
    )
    scene_zone_df = assign_scenes_to_zones(scene_gdf, zones_gdf, metric_crs=args.metric_crs)
    write_outputs(scene_gdf, zones_gdf, scene_zone_df, out_dir=out_dir, target_crs=args.target_crs)

    print(json.dumps({
        "status": "ok",
        "scene_count": int(len(scene_gdf)),
        "zone_count": int(len(zones_gdf)),
        "zones": zones_gdf[["zone_id", "area_km2"]].to_dict(orient="records"),
        "out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
