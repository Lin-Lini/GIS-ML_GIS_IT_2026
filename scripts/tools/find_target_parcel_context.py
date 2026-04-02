from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


AREA_ALIASES = {
    "амга": "Амга",
    "amga": "Амга",
    "юнкор": "Юнкор",
    "yunkor": "Юнкор",
    "юнкюр": "Юнкор",
}


def norm_area(s: str) -> str:
    key = s.strip().lower()
    return AREA_ALIASES.get(key, s.strip())



def find_existing(*paths: str) -> Path:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError("Не найден ни один из ожидаемых файлов: " + ", ".join(paths))



def load_parcels(results_dir: Path) -> gpd.GeoDataFrame:
    p = find_existing(
        results_dir / "parcels" / "parcels_clipped.gpkg",
        results_dir / "vectors" / "parcels_clipped.gpkg",
        results_dir / "global" / "parcels_clipped.geojson",
    )
    gdf = gpd.read_file(p)
    if "parcel_id" not in gdf.columns:
        raise ValueError(f"В {p} нет колонки parcel_id")
    return gdf



def load_aoi(results_dir: Path) -> gpd.GeoDataFrame | None:
    candidates = [
        results_dir / "aoi" / "area_aoi.gpkg",
        results_dir / "aoi" / "area_aoi.geojson",
        results_dir / "vectors" / "area_aoi.gpkg",
        results_dir / "vectors" / "area_aoi.geojson",
        results_dir / "global" / "aoi_union.geojson",
    ]
    for p in candidates:
        if p.exists():
            return gpd.read_file(p)
    return None



def ensure_area_column(parcels: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if "area" in parcels.columns:
        parcels = parcels.copy()
        parcels["area"] = parcels["area"].astype(str)
        return parcels
    if aoi is None or "area" not in aoi.columns:
        return parcels
    p = parcels[[c for c in parcels.columns if c != "area"]].copy()
    a = aoi[["area", "geometry"]].copy()
    if p.crs != a.crs:
        a = a.to_crs(p.crs)
    joined = gpd.sjoin(p, a, how="left", predicate="intersects")
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")
    return joined



def select_target(parcels: gpd.GeoDataFrame, parcel_id: int, area: str | None) -> gpd.GeoDataFrame:
    gdf = parcels.copy()
    gdf["parcel_id_num"] = gdf["parcel_id"].astype(str).str.extract(r"(\d+)")[0]
    gdf = gdf[gdf["parcel_id_num"].astype(float).astype("Int64") == parcel_id]
    if area:
        if "area" not in gdf.columns:
            raise ValueError("В слое parcels нет колонки area, а фильтр по area был задан")
        gdf = gdf[gdf["area"].astype(str) == area]
    if gdf.empty:
        raise ValueError(f"Участок parcel_id={parcel_id} area={area!r} не найден")
    return gdf.drop(columns=["parcel_id_num"], errors="ignore")



def build_context_bbox(target: gpd.GeoDataFrame, meters: float) -> gpd.GeoDataFrame:
    g = target.to_crs(3857)
    minx, miny, maxx, maxy = g.total_bounds
    geom = box(minx - meters, miny - meters, maxx + meters, maxy + meters)
    out = gpd.GeoDataFrame({"name": ["context_bbox"]}, geometry=[geom], crs=3857)
    return out.to_crs(target.crs)



def save_png(target: gpd.GeoDataFrame, context_bbox: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame | None, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    cb = context_bbox.to_crs(3857)
    tg = target.to_crs(3857)
    if aoi is not None and not aoi.empty:
        ao = aoi.to_crs(3857)
        ao.clip(cb).boundary.plot(ax=ax, linewidth=1.0)
    context_bbox.to_crs(3857).boundary.plot(ax=ax, linewidth=1.0, linestyle="--")
    tg.boundary.plot(ax=ax, linewidth=2.5)
    tg.plot(ax=ax, alpha=0.35)
    minx, miny, maxx, maxy = cb.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title("Контекст участка", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def main() -> None:
    ap = argparse.ArgumentParser(description="Найти проблемный участок, сохранить его контур, bbox и PNG-контекст.")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--parcel-id", type=int, required=True)
    ap.add_argument("--area", default="Амга")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--buffer-m", type=float, default=1200.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    area = norm_area(args.area) if args.area else None
    parcels = ensure_area_column(load_parcels(results_dir), load_aoi(results_dir))
    aoi = load_aoi(results_dir)
    target = select_target(parcels, args.parcel_id, area)

    if aoi is not None and "area" in aoi.columns and area:
        aoi = aoi[aoi["area"].astype(str) == area]

    context_bbox = build_context_bbox(target, args.buffer_m)

    target_wgs84 = target.to_crs(4326)
    bbox_wgs84 = context_bbox.to_crs(4326)
    centroid = target_wgs84.geometry.iloc[0].centroid
    minx, miny, maxx, maxy = bbox_wgs84.total_bounds

    target_geojson = out_dir / f"parcel_{args.parcel_id}_{area}_{args.year}.geojson"
    bbox_geojson = out_dir / f"parcel_{args.parcel_id}_{area}_{args.year}_bbox.geojson"
    png_path = out_dir / f"parcel_{args.parcel_id}_{area}_{args.year}_context.png"
    info_json = out_dir / f"parcel_{args.parcel_id}_{area}_{args.year}_links.json"

    target_wgs84.to_file(target_geojson, driver="GeoJSON")
    bbox_wgs84.to_file(bbox_geojson, driver="GeoJSON")
    save_png(target, context_bbox, aoi, png_path)

    google = f"https://www.google.com/maps?q={centroid.y:.6f},{centroid.x:.6f}"
    yandex = f"https://yandex.ru/maps/?ll={centroid.x:.6f}%2C{centroid.y:.6f}&z=16"

    info = {
        "parcel_id": args.parcel_id,
        "area": area,
        "year": args.year,
        "centroid_lat": round(float(centroid.y), 6),
        "centroid_lon": round(float(centroid.x), 6),
        "bbox_wgs84": {
            "min_lon": round(float(minx), 6),
            "min_lat": round(float(miny), 6),
            "max_lon": round(float(maxx), 6),
            "max_lat": round(float(maxy), 6),
        },
        "google_maps_url": google,
        "yandex_maps_url": yandex,
        "target_geojson": str(target_geojson),
        "bbox_geojson": str(bbox_geojson),
        "png": str(png_path),
    }
    info_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
