#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import PRIMARY_CURRENT, PRIMARY_DYNAMIC, add_temporal_features, get_feature_inventory, load_parcel_stats, safe_mkdir, save_json, try_join_parcels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = safe_mkdir(Path(args.out_dir))
    df, aliases = load_parcel_stats(args.results_dir)
    df, area_filter_info = try_join_parcels(args.results_dir, df)
    df = add_temporal_features(df, PRIMARY_CURRENT + PRIMARY_DYNAMIC)

    csv_path = out_dir / "parcel_year_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(out_dir / "parcel_year_table.parquet", index=False)
        parquet_path = str(out_dir / "parcel_year_table.parquet")
    except Exception:
        parquet_path = None

    meta = {
        "rows": int(len(df)),
        "areas": sorted(df["area"].dropna().astype(str).unique().tolist()),
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()),
        "features": get_feature_inventory(df),
        "source_aliases": aliases,
        "area_filter_info": area_filter_info,
        "rows_by_area_year": {
            f"{area}_{year}": int(n)
            for (area, year), n in df.groupby(["area", "year"], dropna=False).size().to_dict().items()
        },
        "unique_parcels_by_area": {
            str(k): int(v) for k, v in df.groupby("area")["parcel_id"].nunique().to_dict().items()
        },
        "csv": str(csv_path),
        "parquet": parquet_path,
    }
    save_json(out_dir / "parcel_year_table_meta.json", meta)
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
