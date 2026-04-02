import argparse
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin


AREA_ALIASES = {
    "amga": ["amga", "амга"],
    "yunkor": ["yunkor", "юнкор", "yunkyur", "юнкюр"],
}


def normalize_text(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", str(s).lower())


def detect_area_year(path: Path) -> Tuple[Optional[str], Optional[int]]:
    stem = normalize_text(path.stem)
    area = None
    for canonical, aliases in AREA_ALIASES.items():
        if any(normalize_text(a) in stem for a in aliases):
            area = canonical
            break
    m = re.search(r"(20\d{2})", path.stem)
    year = int(m.group(1)) if m else None
    return area, year


def load_palette(png_path: str) -> np.ndarray:
    img = np.array(Image.open(png_path).convert("RGBA"))
    rgb = img[:, :, :3].astype(np.float32)
    stds = rgb.std(axis=0).mean(axis=1)
    cols = np.where(stds > 5)[0]
    if len(cols) == 0:
        x0, x1 = 0, img.shape[1]
    else:
        x0, x1 = int(cols.min()), int(cols.max()) + 1
    return np.round(img[:, x0:x1, :3].mean(axis=1)).astype(np.uint8)


def colorize(data: np.ndarray, valid_mask: np.ndarray, pal: np.ndarray, vmin: float, vmax: float):
    rgb = np.zeros((3, data.shape[0], data.shape[1]), dtype=np.uint8)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    t = (data.astype(np.float32) - np.float32(vmin)) / np.float32(vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.round(t * (len(pal) - 1)).astype(np.int32)
    rgb[0, valid_mask] = pal[idx[valid_mask], 0]
    rgb[1, valid_mask] = pal[idx[valid_mask], 1]
    rgb[2, valid_mask] = pal[idx[valid_mask], 2]
    mask_u8 = (valid_mask * 255).astype(np.uint8)
    return rgb, mask_u8


def build_overviews(dst, width: int, height: int, resampling_name: str = "nearest"):
    factors = []
    m = max(width, height)
    f = 2
    while m / f >= 256:
        factors.append(f)
        f *= 2
    if factors:
        resampling = Resampling.nearest if resampling_name == "nearest" else Resampling.bilinear
        dst.build_overviews(factors, resampling=resampling)
        dst.update_tags(ns="rio_overview", resampling=resampling_name)


def choose_template_raster(results_dir: Path, area: Optional[str], year: Optional[int]) -> Optional[Path]:
    candidates = list(results_dir.rglob("*.tif"))
    if not candidates:
        return None

    def score(p: Path) -> Tuple[int, int]:
        s = normalize_text(str(p))
        sc = 0
        if "riskscore" in s:
            sc += 100
        elif "risk" in s:
            sc += 40
        if year is not None and str(year) in s:
            sc += 30
        if area is not None:
            if any(normalize_text(alias) in s for alias in AREA_ALIASES.get(area, [])):
                sc += 30
        return sc, -len(str(p))

    candidates = sorted(candidates, key=score, reverse=True)
    best = candidates[0]
    best_score = score(best)[0]
    return best if best_score > 0 else None


def make_fallback_grid(gdf: gpd.GeoDataFrame, res: float) -> Tuple[rasterio.Affine, int, int]:
    minx, miny, maxx, maxy = gdf.total_bounds
    width = max(1, int(math.ceil((maxx - minx) / res)))
    height = max(1, int(math.ceil((maxy - miny) / res)))
    transform = from_origin(minx, maxy, res, res)
    return transform, width, height


def rasterize_field(
    gdf: gpd.GeoDataFrame,
    field: str,
    out_shape: Tuple[int, int],
    transform,
    fill_value: float,
) -> np.ndarray:
    vals = gdf[field].astype(float)
    shapes = ((geom, float(val)) for geom, val in zip(gdf.geometry, vals) if geom is not None and np.isfinite(val))
    arr = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        fill=fill_value,
        transform=transform,
        dtype="float32",
        all_touched=False,
    )
    return arr


def process_one(
    src_geojson: Path,
    out_dir: Path,
    field: str,
    template_raster: Optional[Path],
    fallback_res: float,
    palette: Optional[np.ndarray],
    vmin: float,
    vmax: float,
    overwrite: bool,
):
    area, year = detect_area_year(src_geojson)
    stem = src_geojson.stem
    raw_out = out_dir / f"{stem}_{field}_heatmap.tif"
    rgb_out = out_dir / f"{stem}_{field}_heatmap_colored_rgb_masked.tif"

    if raw_out.exists() and (palette is None or rgb_out.exists()) and not overwrite:
        print(f"[SKIP] {stem}")
        return

    gdf = gpd.read_file(src_geojson)
    if field not in gdf.columns:
        raise ValueError(f"Field not found in {src_geojson.name}: {field}")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[np.isfinite(gdf[field].astype(float))].copy()
    if gdf.empty:
        raise ValueError(f"No valid geometries with finite field '{field}' in {src_geojson.name}")

    template_profile = None
    template_used = None
    if template_raster is not None:
        template_used = template_raster
    
    if template_used is not None:
        with rasterio.open(template_used) as tpl:
            if gdf.crs is None:
                raise ValueError(f"GeoJSON has no CRS: {src_geojson}")
            if tpl.crs is None:
                raise ValueError(f"Template raster has no CRS: {template_used}")
            if str(gdf.crs) != str(tpl.crs):
                gdf = gdf.to_crs(tpl.crs)
            transform = tpl.transform
            width = tpl.width
            height = tpl.height
            template_profile = tpl.profile.copy()
    else:
        if gdf.crs is None:
            raise ValueError(f"GeoJSON has no CRS: {src_geojson}")
        transform, width, height = make_fallback_grid(gdf, fallback_res)

    nodata = np.float32(-9999.0)
    arr = rasterize_field(gdf, field, (height, width), transform, fill_value=float(nodata))
    valid = np.isfinite(arr) & (arr != nodata)

    raw_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": gdf.crs,
        "transform": transform,
        "nodata": float(nodata),
        "compress": "DEFLATE",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    if template_profile is not None:
        raw_profile["crs"] = template_profile.get("crs", gdf.crs)

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raw_out, "w", **raw_profile) as dst:
        dst.write(arr, 1)
        dst.write_mask((valid * 255).astype(np.uint8))
        dst.update_tags(
            source_geojson=str(src_geojson),
            template_raster=str(template_used) if template_used else "",
            area=area or "",
            year=str(year) if year is not None else "",
            field=field,
            layer_type="parcel_ml_heatmap",
            note="Parcel-level field rasterized to grid; this is not a native pixel-wise ML model.",
        )
        build_overviews(dst, width, height, "nearest")

    if palette is not None:
        rgb, mask_u8 = colorize(arr, valid, palette, vmin, vmax)
        rgb_profile = raw_profile.copy()
        rgb_profile.update(
            dtype="uint8",
            count=3,
            nodata=None,
            predictor=2,
            interleave="pixel",
            photometric="RGB",
        )
        with rasterio.open(rgb_out, "w", **rgb_profile) as dst:
            dst.write(rgb)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.write_mask(mask_u8)
            dst.update_tags(
                source_geojson=str(src_geojson),
                template_raster=str(template_used) if template_used else "",
                area=area or "",
                year=str(year) if year is not None else "",
                field=field,
                palette_min=str(vmin),
                palette_max=str(vmax),
                layer_type="colored_parcel_ml_heatmap",
                note="Parcel-level field rasterized to grid and colorized; not a native pixel-wise ML raster.",
            )
            build_overviews(dst, width, height, "nearest")

    print(f"[OK] {src_geojson.name} -> {raw_out.name}" + (f" + {rgb_out.name}" if palette is not None else ""))


def main():
    ap = argparse.ArgumentParser(
        description="Batch rasterize parcel-level ML scores from GeoJSON to GeoTIFF heatmaps."
    )
    ap.add_argument("--in-dir", required=True, help="Folder with *parcel_ml_scores.geojson files")
    ap.add_argument("--out-dir", required=True, help="Output folder for heatmap GeoTIFFs")
    ap.add_argument("--field", default="pred_ml_final", help="Numeric field to rasterize")
    ap.add_argument("--pattern", default="*parcel_ml_scores.geojson", help="GeoJSON filename glob")
    ap.add_argument("--results-dir", default=None, help="Optional results folder to auto-find matching template risk_score raster")
    ap.add_argument("--template-raster", default=None, help="Optional single template raster for all outputs")
    ap.add_argument("--palette", default=None, help="Optional PNG/JPG palette for colored RGB output")
    ap.add_argument("--min", dest="vmin", type=float, default=0.0, help="Palette minimum")
    ap.add_argument("--max", dest="vmax", type=float, default=1.0, help="Palette maximum")
    ap.add_argument("--fallback-res", type=float, default=10.5, help="Fallback pixel size if template raster is not found")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.exists():
        raise SystemExit(f"[ERROR] Input folder not found: {in_dir}")

    files = sorted(p for p in in_dir.rglob(args.pattern) if p.is_file())
    if not files:
        raise SystemExit(f"[ERROR] No GeoJSON files found by pattern {args.pattern!r} in {in_dir}")

    palette = load_palette(args.palette) if args.palette else None
    results_dir = Path(args.results_dir) if args.results_dir else None
    template_raster = Path(args.template_raster) if args.template_raster else None

    created = 0
    failed = 0
    for src in files:
        try:
            local_template = template_raster
            if local_template is None and results_dir is not None:
                area, year = detect_area_year(src)
                local_template = choose_template_raster(results_dir, area, year)
                if local_template is None:
                    print(f"[WARN] Template not found for {src.name}; fallback grid will be used")
                else:
                    print(f"[INFO] Template for {src.name}: {local_template}")
            process_one(
                src_geojson=src,
                out_dir=out_dir,
                field=args.field,
                template_raster=local_template,
                fallback_res=args.fallback_res,
                palette=palette,
                vmin=args.vmin,
                vmax=args.vmax,
                overwrite=args.overwrite,
            )
            created += 1
        except Exception as e:
            print(f"[FAIL] {src.name}: {e}")
            failed += 1

    print()
    print("[SUMMARY]")
    print(f"processed={len(files)}")
    print(f"created={created}")
    print(f"failed={failed}")
    print(f"out_dir={out_dir}")
    if failed > 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
