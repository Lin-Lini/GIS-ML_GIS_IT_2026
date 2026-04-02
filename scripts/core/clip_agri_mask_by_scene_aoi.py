from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


STRICT_KEEP = re.compile(
    r'(?:\bпашн\w*\b|'
    r'\bпастбищ\w*\b|'
    r'\bсенокос\w*\b|\bсенокош\w*\b|'
    r'сельскохозяйственн\w*\s+использован|'
    r'сельскохозяйственн\w*\s+производств|'
    r'сельхозпроизводств|'
    r'сельскохозяйственн\w*\s+угод|'
    r'полев\w*\s+участ)',
    flags=re.IGNORECASE,
)

STRICT_DROP = re.compile(
    r'(?:кфх|фермер|животновод|скотовод|коневод|оленевод|сайылык|'
    r'личн\w*\s+подсобн\w*\s+хозяйств(?!.*полев\w*\s+участ)|'
    r'\bлпх\b(?!.*полев\w*\s+участ)|'
    r'под\s+личное\s+подсобное\s+хозяйство(?!.*полев)|'
    r'под\s+сельскохозяйственные\s+объекты|'
    r'фонд\s+перераспределен)',
    flags=re.IGNORECASE,
)


def zip_sidecars(base_path: Path, zip_path: Path) -> None:
    suffixes = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for suf in suffixes:
            p = base_path.with_suffix(suf)
            if p.exists():
                z.write(p, arcname=p.name)


def read_shapefile_from_zip(shp_zip: Path) -> gpd.GeoDataFrame:
    stem = shp_zip.stem
    candidates = [
        f'zip://{shp_zip}!{stem}.shp',
        f'zip://{shp_zip}!yakutia_agri_field_core.shp',
        f'zip://{shp_zip}!yakutia_agri_strict_fields.shp',
    ]
    for cand in candidates:
        try:
            return gpd.read_file(cand)
        except Exception:
            continue
    raise FileNotFoundError(f'Не удалось прочитать shapefile из архива: {shp_zip}')


def strict_filter_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    doc = out['utl_doc'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True)
    mask = doc.str.contains(STRICT_KEEP, regex=True) & ~doc.str.contains(STRICT_DROP, regex=True)
    return out.loc[mask].copy()


def extract_scene_aoi(scene_zip: Path, tmp_root: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(scene_zip) as z:
        members = [n for n in z.namelist() if '.GBD.' in n and n.endswith(('.shp', '.shx', '.dbf', '.prj', '.cpg'))]
        if not members:
            raise FileNotFoundError(f'В архиве не найден GBD shapefile: {scene_zip}')
        subdir = tmp_root / scene_zip.stem
        subdir.mkdir(parents=True, exist_ok=True)
        for member in members:
            with z.open(member) as src, open(subdir / Path(member).name, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        shp = next(subdir.glob('*.GBD.shp'))
    aoi = gpd.read_file(shp)
    aoi = aoi.to_crs(4326)
    aoi['scene_zip'] = scene_zip.name
    return aoi[['scene_zip', 'geometry']]


def safe_name_from_zip(scene_zip: Path) -> str:
    lower = scene_zip.name.lower()
    if 'амга' in lower or '3926868' in lower:
        return 'amga'
    if 'юнкор' in lower or 'юнк' in lower or '3927029' in lower:
        return 'yunkor'
    return scene_zip.stem.lower()


def export_vector(gdf: gpd.GeoDataFrame, base_name: str, out_dir: Path) -> None:
    shp_base = out_dir / base_name
    gdf.to_file(shp_base.with_suffix('.shp'), driver='ESRI Shapefile', encoding='UTF-8')
    zip_sidecars(shp_base, out_dir / f'{base_name}_shp.zip')

    geojson_path = out_dir / f'{base_name}.geojson'
    gdf.to_file(geojson_path, driver='GeoJSON')
    with zipfile.ZipFile(out_dir / f'{base_name}_geojson.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(geojson_path, arcname=geojson_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description='Обрезка сельхоз-маски по AOI сцен Канопус.')
    parser.add_argument('--base_shp_zip', required=True, help='Архив с базовым shapefile сельхозмаски.')
    parser.add_argument('--scene_zips', required=True, nargs='+', help='Один или несколько архивов со сценами Канопус.')
    parser.add_argument('--out_dir', required=True, help='Папка для результатов.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_gdf = read_shapefile_from_zip(Path(args.base_shp_zip)).to_crs(4326)
    strict_gdf = strict_filter_fields(base_gdf)
    strict_gdf = strict_gdf[~strict_gdf.geometry.is_empty & strict_gdf.geometry.notna()].copy()

    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)

        aoi_layers = []
        stats_rows = []

        for scene_zip_str in args.scene_zips:
            scene_zip = Path(scene_zip_str)
            aoi = extract_scene_aoi(scene_zip, tmp_root)
            area_name = safe_name_from_zip(scene_zip)
            aoi_layers.append(aoi.assign(area_name=area_name))

            clip = gpd.overlay(strict_gdf, aoi[['geometry']], how='intersection')
            clip = clip[~clip.geometry.is_empty & clip.geometry.notna()].copy()
            clip['aoi_name'] = area_name

            export_vector(clip, f'{area_name}_strict_fields_by_scene_aoi', out_dir)
            stats_rows.append({'layer': f'{area_name}_strict_fields_by_scene_aoi', 'count': len(clip)})

        aoi_all = pd.concat(aoi_layers, ignore_index=True)
        aoi_all = gpd.GeoDataFrame(aoi_all, geometry='geometry', crs='EPSG:4326')
        export_vector(aoi_all, 'scene_aoi_footprints', out_dir)
        stats_rows.append({'layer': 'scene_aoi_footprints', 'count': len(aoi_all)})

        union_geom = aoi_all.unary_union
        union_gdf = gpd.GeoDataFrame({'name': ['amga_yunkor_scene_union']}, geometry=[union_geom], crs='EPSG:4326')
        export_vector(union_gdf, 'scene_aoi_union', out_dir)
        stats_rows.append({'layer': 'scene_aoi_union', 'count': len(union_gdf)})

        combo = gpd.overlay(strict_gdf, union_gdf[['geometry']], how='intersection')
        combo = combo[~combo.geometry.is_empty & combo.geometry.notna()].copy()
        combo['aoi_name'] = 'amga_yunkor_scene_union'
        export_vector(combo, 'amga_yunkor_strict_fields_by_scene_union', out_dir)
        stats_rows.append({'layer': 'amga_yunkor_strict_fields_by_scene_union', 'count': len(combo)})

    summary = pd.DataFrame(stats_rows)
    summary.to_csv(out_dir / 'summary.csv', index=False, encoding='utf-8-sig')

    readme = f"""Результаты обрезки сельхозмаски по AOI сцен Канопус.

Входные данные:
- базовый слой: {args.base_shp_zip}
- сцены: {', '.join(args.scene_zips)}

Логика:
1. Читается базовый shapefile с сельхозмаской.
2. Поверх него применяется более строгий текстовый фильтр по полю utl_doc:
   - оставляются прямые полевые и сельхозформулировки (пашня, пастбища, сенокосы, сельхозиспользование, сельхозпроизводство, полевой участок);
   - отсекаются более широкие и спорные сущности (КФХ, животноводство, сайылыки, неполевое ЛПХ и т.д.).
3. Из каждого архива сцены извлекается GBD shapefile, который используется как AOI покрытия сцены.
4. Строгая сельхозмаска обрезается по каждому AOI отдельно и по объединению AOI.
5. Результаты сохраняются в Shapefile и GeoJSON.

Практически:
- для фронта удобнее брать GeoJSON-файлы по Амге и Юнкору по отдельности;
- для общей аналитики можно использовать объединённый слой amga_yunkor_strict_fields_by_scene_union.
"""
    (out_dir / 'README.txt').write_text(readme, encoding='utf-8')

    print(f'Готово. Результаты в: {out_dir}')


if __name__ == '__main__':
    main()
