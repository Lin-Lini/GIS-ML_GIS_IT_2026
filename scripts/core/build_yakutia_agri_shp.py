from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import from_wkt, make_valid

SRC_ZIP = Path('/mnt/data/Земельный Кадастр_Якутия.zip')
OUT_DIR = Path('/mnt/data/yakutia_agri_layers')
OUT_DIR.mkdir(exist_ok=True)

INCLUDE_RE = re.compile(
    r'(?:сельскох|сельхоз|пастбищ|сенокос|сенокош|пашн|пахот|'
    r'крестьян|фермер|кфх|'
    r'личн\w*\s+подсобн\w*\s+хозяйств|\bлпх\b|'
    r'животновод|скотовод|коневод|табун|оленевод|'
    r'сайылык|сайылыч|'
    r'полев\w*\s+участ|'
    r'сельскохозяйственн\w*\s+угод|'
    r'фонд\s+перераспределен|'
    r'выращиван\w*\s+зернов|'
    r'агрокомплекс|'
    r'пастбища\s+и\s+сенокосы)',
    flags=re.IGNORECASE,
)

EXCLUDE_RE = re.compile(
    r'(?:садов|огород|дач|жил|дом|ижс|улиц|дорог|лес|кладбищ|'
    r'связ|энерг|транспорт|гидротех|шлюз|дамб|канал|водопровод|'
    r'полигон|отход|скотомогиль|биотерм|часовн|кордон|аэродром|'
    r'котельн|подстанц|вл-|лэп|\bтп\b|баня|охот|рыбал|'
    r'развлеч|культур|истор|ритуал|гараж|коммун|автосерв|азс|'
    r'карьер|строитель|промышлен|склад|грэс|медицин|санатор|лагер|'
    r'парк|свинар|убойн|летн\w*\s+усад|усадьб|кордона?)',
    flags=re.IGNORECASE,
)

COLS = ['CAD_N', 'STATUS', 'C_COST', 'AREA', 'UTL_ID', 'UTL_DOC', 'OBJ_WKT']
WRITE_COLS = ['cad_num', 'status', 'c_cost', 'area_m2', 'utl_id', 'utl_doc', 'geometry']


def read_inner_csv(outer_zip: zipfile.ZipFile, inner_name: str) -> pd.DataFrame:
    data = outer_zip.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(data)) as z2:
        csv_name = z2.namelist()[0]
        return pd.read_csv(io.BytesIO(z2.read(csv_name)), skipinitialspace=True, dtype=str, usecols=COLS)


def finalize_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    df = df.copy()
    df['geometry'] = from_wkt(df['OBJ_WKT'].fillna(''))
    gdf = gpd.GeoDataFrame(df.drop(columns=['OBJ_WKT']), geometry='geometry', crs='EPSG:4326')
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].apply(make_valid)

    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()

    gdf.rename(
        columns={
            'CAD_N': 'cad_num',
            'STATUS': 'status',
            'C_COST': 'c_cost',
            'AREA': 'area_m2',
            'UTL_ID': 'utl_id',
            'UTL_DOC': 'utl_doc',
        },
        inplace=True,
    )

    gdf['area_m2'] = pd.to_numeric(gdf['area_m2'], errors='coerce')
    gdf['c_cost'] = pd.to_numeric(gdf['c_cost'], errors='coerce')
    gdf['utl_doc'] = gdf['utl_doc'].fillna('').astype(str)
    return gdf[WRITE_COLS]


def zip_sidecars(base_path: Path, zip_path: Path) -> None:
    suffixes = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for suf in suffixes:
            p = base_path.with_suffix(suf)
            if p.exists():
                z.write(p, arcname=p.name)


def main() -> None:
    with zipfile.ZipFile(SRC_ZIP) as outer:
        inner_names = [n for n in outer.namelist() if n.endswith('.csv.zip')]

        extended_parts = []
        core_parts = []

        for inner in inner_names:
            df = read_inner_csv(outer, inner)
            df['OBJ_WKT'] = df['OBJ_WKT'].fillna('')
            df['UTL_DOC'] = df['UTL_DOC'].fillna('')
            has_geom = df['OBJ_WKT'].ne('')
            ext_mask = has_geom & (df['UTL_ID'] == '003001000000')
            core_mask = ext_mask & df['UTL_DOC'].str.contains(INCLUDE_RE, na=False) & ~df['UTL_DOC'].str.contains(EXCLUDE_RE, na=False)

            if ext_mask.any():
                extended_parts.append(df.loc[ext_mask, COLS].copy())
            if core_mask.any():
                core_parts.append(df.loc[core_mask, COLS].copy())

    gdf_ext = finalize_gdf(pd.concat(extended_parts, ignore_index=True))
    gdf_core = finalize_gdf(pd.concat(core_parts, ignore_index=True))

    ext_base = OUT_DIR / 'yakutia_agri_all_003001'
    core_base = OUT_DIR / 'yakutia_agri_field_core'

    # write using a minimal, stable field set
    gdf_ext.to_file(ext_base.with_suffix('.shp'), driver='ESRI Shapefile', encoding='UTF-8')
    gdf_core.to_file(core_base.with_suffix('.shp'), driver='ESRI Shapefile', encoding='UTF-8')

    zip_sidecars(ext_base, OUT_DIR / 'yakutia_agri_all_003001_shp.zip')
    zip_sidecars(core_base, OUT_DIR / 'yakutia_agri_field_core_shp.zip')

    summary = pd.DataFrame([
        {
            'layer': 'yakutia_agri_all_003001',
            'count': len(gdf_ext),
            'crs': str(gdf_ext.crs),
            'description': 'Все полигоны с геометрией из категории UTL_ID=003001000000.',
        },
        {
            'layer': 'yakutia_agri_field_core',
            'count': len(gdf_core),
            'crs': str(gdf_core.crs),
            'description': 'Более узкий слой полей/угодий: пашня, пастбища, сенокосы, КФХ, ЛПХ на полевых участках, животноводство и др.; дачи/садоводство/инфраструктура отсечены.',
        },
    ])
    summary.to_csv(OUT_DIR / 'summary.csv', index=False)

    readme = f'''Слои построены из архива: {SRC_ZIP.name}

1) yakutia_agri_all_003001_shp.zip
   - единый shapefile по всем объектам с геометрией и UTL_ID=003001000000
   - количество объектов: {len(gdf_ext)}

2) yakutia_agri_field_core_shp.zip
   - единый shapefile по более полевому ядру сельхозугодий
   - количество объектов: {len(gdf_core)}

CRS: EPSG:4326

Практически:
- если нужен максимально широкий охват сельхозназначения, бери yakutia_agri_all_003001
- если нужен более чистый слой именно полей/угодий для AOI и spatial join, бери yakutia_agri_field_core
'''
    (OUT_DIR / 'README.txt').write_text(readme, encoding='utf-8')

    bundle = OUT_DIR / 'yakutia_agri_shapefiles_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [
            'yakutia_agri_all_003001.shp', 'yakutia_agri_all_003001.shx', 'yakutia_agri_all_003001.dbf', 'yakutia_agri_all_003001.prj', 'yakutia_agri_all_003001.cpg',
            'yakutia_agri_field_core.shp', 'yakutia_agri_field_core.shx', 'yakutia_agri_field_core.dbf', 'yakutia_agri_field_core.prj', 'yakutia_agri_field_core.cpg',
            'yakutia_agri_all_003001_shp.zip', 'yakutia_agri_field_core_shp.zip', 'summary.csv', 'README.txt'
        ]:
            p = OUT_DIR / name
            if p.exists():
                z.write(p, arcname=name)

    print('extended_count', len(gdf_ext))
    print('core_count', len(gdf_core))
    print('bundle', bundle)


if __name__ == '__main__':
    main()
