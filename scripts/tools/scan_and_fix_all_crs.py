from pathlib import Path
import argparse
import shutil
import json
import pandas as pd
import geopandas as gpd
import fiona
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds

R_EXT = {".tif", ".tiff", ".vrt"}
V_EXT = {".gpkg", ".geojson", ".json", ".shp"}

def crs_str(x):
    if not x:
        return ""
    try:
        return CRS.from_user_input(x).to_string()
    except:
        return str(x)

def crs_epsg(x):
    if not x:
        return None
    try:
        return CRS.from_user_input(x).to_epsg()
    except:
        return None

def b4326(c, b):
    try:
        out = transform_bounds(CRS.from_user_input(c), "EPSG:4326", *b, densify_pts=21)
        return [round(v, 8) for v in out]
    except:
        return None

def fmt_bounds(b):
    if b is None:
        return ""
    return json.dumps([round(float(x), 8) for x in b], ensure_ascii=False)

def rs_for_file(p):
    n = p.name.lower()
    bilinear_keys = [
        "composite", "colored", "brightness", "ndvi", "ndwi", "osavi",
        "ratio", "dem", "slope", "roughness", "curvature", "tpi", "tri",
        "aspect", "contrast", "dissimilarity", "entropy", "homogeneity",
        "std", "var"
    ]
    if any(k in n for k in bilinear_keys):
        return Resampling.bilinear
    return Resampling.nearest

def scan_raster(p, rel):
    rows = []
    try:
        with rasterio.open(p) as src:
            c = src.crs
            b = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            rows.append({
                "kind": "raster",
                "path": str(rel),
                "layer": "",
                "crs": crs_str(c),
                "epsg": crs_epsg(c),
                "web_ready": crs_epsg(c) in (4326, 3857),
                "bounds_native": fmt_bounds(b),
                "bounds_4326": fmt_bounds(b4326(c, b)),
                "count": src.count,
                "dtype": ",".join(src.dtypes),
                "error": ""
            })
    except Exception as e:
        rows.append({
            "kind": "raster",
            "path": str(rel),
            "layer": "",
            "crs": "",
            "epsg": None,
            "web_ready": False,
            "bounds_native": "",
            "bounds_4326": "",
            "count": None,
            "dtype": "",
            "error": str(e)
        })
    return rows

def scan_vector(p, rel):
    rows = []
    try:
        layers = fiona.listlayers(p) if p.suffix.lower() == ".gpkg" else [None]
        for lyr in layers:
            with fiona.open(p, layer=lyr) as src:
                c = src.crs_wkt or src.crs
                b = src.bounds
                rows.append({
                    "kind": "vector",
                    "path": str(rel),
                    "layer": lyr or "",
                    "crs": crs_str(c),
                    "epsg": crs_epsg(c),
                    "web_ready": crs_epsg(c) in (4326, 3857),
                    "bounds_native": fmt_bounds(b),
                    "bounds_4326": fmt_bounds(b4326(c, b)),
                    "count": len(src),
                    "dtype": "",
                    "error": ""
                })
    except Exception as e:
        rows.append({
            "kind": "vector",
            "path": str(rel),
            "layer": "",
            "crs": "",
            "epsg": None,
            "web_ready": False,
            "bounds_native": "",
            "bounds_4326": "",
            "count": None,
            "dtype": "",
            "error": str(e)
        })
    return rows

def fix_raster(src_p, dst_p, target="EPSG:4326", force=False):
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_p) as src:
        if src.crs is None:
            return "skip_no_crs"
        if crs_str(src.crs) == target and not force:
            shutil.copy2(src_p, dst_p)
            return "copied_same_crs"
        tr, w, h = calculate_default_transform(src.crs, target, src.width, src.height, *src.bounds)
        pr = src.profile.copy()
        pr.update(crs=target, transform=tr, width=w, height=h, compress="deflate")
        rs = rs_for_file(src_p)
        with rasterio.open(dst_p, "w", **pr) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=tr,
                    dst_crs=target,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                    resampling=rs
                )
        return f"reprojected_{rs.name}"

def fix_vector(src_p, dst_p, target="EPSG:4326", force=False):
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    ext = dst_p.suffix.lower()
    if src_p.suffix.lower() == ".gpkg":
        layers = fiona.listlayers(src_p)
        if dst_p.exists():
            dst_p.unlink()
        for i, lyr in enumerate(layers):
            gdf = gpd.read_file(src_p, layer=lyr)
            if gdf.crs is None:
                continue
            if crs_str(gdf.crs) != target or force:
                gdf = gdf.to_crs(target)
            mode = "w" if i == 0 else "a"
            gdf.to_file(dst_p, layer=lyr, driver="GPKG", mode=mode)
        return "done_gpkg"
    gdf = gpd.read_file(src_p)
    if gdf.crs is None:
        return "skip_no_crs"
    if crs_str(gdf.crs) != target or force:
        gdf = gdf.to_crs(target)
    if ext in {".geojson", ".json"}:
        gdf.to_file(dst_p, driver="GeoJSON")
        return "done_geojson"
    if ext == ".shp":
        gdf.to_file(dst_p, driver="ESRI Shapefile")
        return "done_shp"
    if ext == ".gpkg":
        gdf.to_file(dst_p, driver="GPKG")
        return "done_gpkg"
    return "skip_unsupported"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--report-csv", default="crs_report.csv")
    ap.add_argument("--summary-txt", default="crs_summary.txt")
    ap.add_argument("--fix-root", default="")
    ap.add_argument("--target-crs", default="EPSG:4326")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    rasters = []
    vectors = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rel = p.relative_to(root)
        if ext in R_EXT:
            rasters.append((p, rel))
            rows.extend(scan_raster(p, rel))
        elif ext in V_EXT:
            vectors.append((p, rel))
            rows.extend(scan_vector(p, rel))

    df = pd.DataFrame(rows)
    df.to_csv(args.report_csv, index=False, encoding="utf-8-sig")

    lines = []
    lines.append(f"root: {root}")
    lines.append(f"target_crs: {args.target_crs}")
    lines.append(f"rows_total: {len(df)}")
    lines.append(f"errors: {int((df['error'] != '').sum()) if len(df) else 0}")
    lines.append("")
    if len(df):
        lines.append("by_kind:")
        lines.append(df["kind"].value_counts(dropna=False).to_string())
        lines.append("")
        lines.append("by_epsg:")
        lines.append(df["epsg"].fillna("None").value_counts(dropna=False).to_string())
        lines.append("")
        lines.append("not_web_ready:")
        bad = df[(df["crs"] != "") & (~df["web_ready"])]
        if len(bad):
            lines.append(bad[["kind", "path", "layer", "crs", "epsg"]].to_string(index=False))
        else:
            lines.append("none")
    Path(args.summary_txt).write_text("\n".join(lines), encoding="utf-8")

    print(f"report_csv: {args.report_csv}")
    print(f"summary_txt: {args.summary_txt}")

    if args.fix_root:
        fix_root = Path(args.fix_root)
        out_rows = []
        for p, rel in rasters:
            try:
                dst = fix_root / rel
                act = fix_raster(p, dst, target=args.target_crs, force=args.force)
                out_rows.append({"kind": "raster", "path": str(rel), "action": act, "out": str(dst)})
                print("RASTER", rel, "->", act)
            except Exception as e:
                out_rows.append({"kind": "raster", "path": str(rel), "action": f"error: {e}", "out": ""})
                print("RASTER", rel, "->", e)

        for p, rel in vectors:
            try:
                dst = fix_root / rel
                act = fix_vector(p, dst, target=args.target_crs, force=args.force)
                out_rows.append({"kind": "vector", "path": str(rel), "action": act, "out": str(dst)})
                print("VECTOR", rel, "->", act)
            except Exception as e:
                out_rows.append({"kind": "vector", "path": str(rel), "action": f"error: {e}", "out": ""})
                print("VECTOR", rel, "->", e)

        pd.DataFrame(out_rows).to_csv(fix_root / "fix_actions.csv", index=False, encoding="utf-8-sig")
        print(f"fix_root: {fix_root}")

if __name__ == "__main__":
    main()