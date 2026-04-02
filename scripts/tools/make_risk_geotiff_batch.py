import argparse
from pathlib import Path
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling, ColorInterp
from PIL import Image


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


def colorize(data: np.ndarray, mask: np.ndarray, pal: np.ndarray, vmin: float, vmax: float):
    valid = (~mask) & np.isfinite(data)
    rgb = np.zeros((3, data.shape[0], data.shape[1]), dtype=np.uint8)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    t = (data.astype(np.float32) - np.float32(vmin)) / np.float32(vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.round(t * (len(pal) - 1)).astype(np.int32)
    rgb[0, valid] = pal[idx[valid], 0]
    rgb[1, valid] = pal[idx[valid], 1]
    rgb[2, valid] = pal[idx[valid], 2]
    mask_u8 = (valid * 255).astype(np.uint8)
    return rgb, mask_u8


def build_overviews(dst, width: int, height: int, resampling_name: str):
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


def process_one(src_path: Path, dst_path: Path, pal: np.ndarray, vmin: float, vmax: float, resampling: str):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        data = src.read(1, masked=True)
        rgb, mask_u8 = colorize(data.filled(np.nan), np.ma.getmaskarray(data), pal, vmin, vmax)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=3,
            nodata=None,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            interleave="pixel",
            photometric="RGB",
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(rgb)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.write_mask(mask_u8)
            dst.update_tags(
                source_raster=str(src_path),
                palette_source="image gradient",
                palette_min=str(vmin),
                palette_max=str(vmax),
                layer_type="colored_risk_score",
            )
            build_overviews(dst, src.width, src.height, resampling)


def find_inputs(in_dir: Path, pattern: str, suffix: str):
    files = sorted(p for p in in_dir.rglob(pattern) if p.is_file())
    out = []
    for p in files:
        stem_lower = p.stem.lower()
        if stem_lower.endswith(suffix.lower()):
            continue
        if stem_lower.endswith("_colored_rgb_masked"):
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Batch colorize risk score GeoTIFF rasters into frontend-ready RGB GeoTIFFs.")
    ap.add_argument("--in-dir", required=True, help="Folder with risk score GeoTIFFs")
    ap.add_argument("--palette", required=True, help="PNG/JPG with vertical gradient palette")
    ap.add_argument("--out-dir", default=None, help="Output folder. Default: <in-dir>/colored_risk_geotiff")
    ap.add_argument("--pattern", default="*risk_score*.tif", help="Glob pattern for input GeoTIFFs")
    ap.add_argument("--suffix", default="_colored_rgb_masked", help="Suffix for output files")
    ap.add_argument("--min", dest="vmin", type=float, default=0.0, help="Palette minimum")
    ap.add_argument("--max", dest="vmax", type=float, default=1.0, help="Palette maximum")
    ap.add_argument("--resampling", default="nearest", choices=["nearest", "bilinear"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] Input folder not found: {in_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "colored_risk_geotiff"
    pal = load_palette(args.palette)
    inputs = find_inputs(in_dir, args.pattern, args.suffix)

    if not inputs:
        print(f"[ERROR] No files found by pattern {args.pattern!r} in {in_dir}")
        sys.exit(2)

    print(f"[INFO] Found {len(inputs)} input file(s)")
    print(f"[INFO] Output folder: {out_dir}")

    ok = 0
    skipped = 0
    failed = 0

    for src_path in inputs:
        dst_name = f"{src_path.stem}{args.suffix}.tif"
        dst_path = out_dir / dst_name

        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] {dst_path.name} already exists")
            skipped += 1
            continue

        try:
            process_one(src_path, dst_path, pal, args.vmin, args.vmax, args.resampling)
            print(f"[OK] {src_path.name} -> {dst_path.name}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {src_path.name}: {e}")
            failed += 1

    print()
    print("[SUMMARY]")
    print(f"created={ok}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"out_dir={out_dir}")

    if failed > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
