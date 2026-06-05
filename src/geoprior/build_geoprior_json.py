#!/usr/bin/env python3
"""
Build COCO-style train/val/test JSONs for geo-prior training.

Source data (all paths/identifiers come from src/geoprior/config.py / .env):
  - <BQ_DATASET>.gbif_occurrence_location  (lat/lon/eventdate per gbif_id)
  - <BQ_DATASET>.gbif_inat_occurrences     (species_name per gbif_id)
  - $GEOPRIOR_SPLITS_DIR/{val,test}.csv    (vision-model val/test photo lists)
  - frozen geoprior_categ_map.json         (species_name -> class_id, 12317 classes)

Output JSONs (written to $GEOPRIOR_DATA_DIR):
  - train.json   (all geocoded - val/test - bad_coords - sparse_species)
  - val.json     (vision val_set photos intersected with geocoded data)
  - test.json    (vision test_set photos intersected with geocoded data)

Filters applied (per the create_moth_filtered_geoprior_json.ipynb precedent):
  train:
    - exclude any gbif_id in val.csv or test.csv
    - require non-NULL lat/lon/eventdate
    - drop lat OR lon values that appear > 1000 times (placeholder coords)
    - drop species with <= MIN_OCC_PER_SPECIES geocoded occurrences
      (default 0 = keep all species)
  val/test:
    - require non-NULL lat/lon/eventdate (only)
"""
import json
import time

import pandas as pd
from google.cloud import bigquery

from src.geoprior import config

SPLITS_DIR        = config.SPLITS_DIR
GEOPRIOR_DIR      = config.DATA_DIR
CATEG_MAP_PATH    = config.CATEG_MAP_PATH    # frozen species -> class_id (12317)
SAME_COORD_THRESHOLD = 1000   # drop coord values appearing > this many times
MIN_OCC_PER_SPECIES  = 0     # 0 disables the filter (keep all species)

DATE_FORMAT = '%Y-%m-%d %H:%M:%S+00:00'


def t(msg, start=None):
    now = time.time()
    if start is None:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
    else:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}  ({now - start:.1f}s)', flush=True)
    return now


def load_split_gbif_ids(csv_path):
    df = pd.read_csv(csv_path, usecols=['gbif_id'])
    return set(df['gbif_id'].astype(int).tolist())


def fetch_all_geocoded(client):
    """Pull all valid geocoded occurrences with species_name."""
    q = f'''
    SELECT
      o.gbifID                              AS gbif_id,
      o.verbatimSpeciesScientificName       AS species_name,
      l.decimallatitude                     AS latitude,
      l.decimallongitude                    AS longitude,
      l.eventdate                           AS event_date
    FROM `{config.TBL_OCCURRENCES}` o
    JOIN `{config.TBL_LOCATION}` l
      ON l.gbif_id = o.gbifID
    WHERE l.decimallatitude  IS NOT NULL
      AND l.decimallongitude IS NOT NULL
      AND l.eventdate        IS NOT NULL
      AND o.verbatimSpeciesScientificName IS NOT NULL
    '''
    job = client.query(q)
    df = job.to_dataframe()
    return df, job.total_bytes_billed


def format_date_str(ts):
    if pd.isna(ts):
        return None
    # eventdate is a pandas Timestamp w/ TZ
    try:
        return ts.strftime(DATE_FORMAT)
    except Exception:
        return None


def build_json(df, categ_map, drop_label):
    """Convert a filtered DataFrame to the COCO-style JSON the dataloader expects."""
    images = []
    annotations = []
    n_unmapped = 0
    for row in df.itertuples(index=False):
        cid = categ_map.get(row.species_name)
        if cid is None:
            n_unmapped += 1
            continue
        images.append({
            'id': int(row.gbif_id),
            'latitude':  float(row.latitude),
            'longitude': float(row.longitude),
            'date':      format_date_str(row.event_date),
        })
        annotations.append({
            'image_id':   int(row.gbif_id),
            'category_id': int(cid),
        })
    # Categories array: keep full 12317-class space (don't shrink to species-in-this-file)
    n_classes = len(categ_map)
    categories = [{'id': i} for i in range(n_classes)]
    print(f'    [{drop_label}] kept {len(images):,} rows, dropped {n_unmapped} unmapped-species rows')
    return {'images': images, 'annotations': annotations, 'categories': categories}


def main():
    start = t('Loading category map and split gbif_id sets')
    with open(CATEG_MAP_PATH) as f:
        categ_map = json.load(f)
    print(f'  Category map: {len(categ_map):,} species -> class_id')

    val_ids  = load_split_gbif_ids(SPLITS_DIR / 'val.csv')
    test_ids = load_split_gbif_ids(SPLITS_DIR / 'test.csv')
    val_test_ids = val_ids | test_ids
    print(f'  val.csv:  {len(val_ids):,} unique gbif_ids')
    print(f'  test.csv: {len(test_ids):,} unique gbif_ids')
    print(f'  val ∪ test: {len(val_test_ids):,} unique gbif_ids')

    t('Fetching all geocoded occurrences from BQ', start)
    fetch_start = time.time()
    client = bigquery.Client(project=config.BQ_PROJECT)
    df, bytes_billed = fetch_all_geocoded(client)
    t(f'BQ fetch done. Rows: {len(df):,}. Bytes scanned: {bytes_billed/1e6:.1f} MB (~${bytes_billed/1e12*5:.4f})', fetch_start)

    t('Computing placeholder-coord filter (lat/lon appearing > 1000 times)')
    lat_counts = df['latitude'].value_counts()
    lon_counts = df['longitude'].value_counts()
    bad_lats = set(lat_counts[lat_counts > SAME_COORD_THRESHOLD].index)
    bad_lons = set(lon_counts[lon_counts > SAME_COORD_THRESHOLD].index)
    print(f'  Bad lats (placeholder values): {len(bad_lats)} → e.g. {sorted(bad_lats)[:5]}')
    print(f'  Bad lons (placeholder values): {len(bad_lons)} → e.g. {sorted(bad_lons)[:5]}')

    t('=== Building TRAIN set ===')
    train = df.copy()
    print(f'  Before filters:       {len(train):,}')
    train = train[~train['gbif_id'].isin(val_test_ids)]
    print(f'  After excluding v/t:  {len(train):,}')
    train = train[~train['latitude'].isin(bad_lats) & ~train['longitude'].isin(bad_lons)]
    print(f'  After dropping bad coords: {len(train):,}')
    spc_counts = train['species_name'].value_counts()
    kept_species = set(spc_counts[spc_counts > MIN_OCC_PER_SPECIES].index)
    train = train[train['species_name'].isin(kept_species)]
    print(f'  After dropping species ≤ {MIN_OCC_PER_SPECIES} occ: {len(train):,}')
    print(f'  Species kept in train: {len(kept_species):,} (of {len(categ_map):,} total)')

    train_json = build_json(train, categ_map, 'train')

    t('=== Building VAL set ===')
    val = df[df['gbif_id'].isin(val_ids)].copy()
    print(f'  Initial (valid lat/lon/date intersected with val.csv): {len(val):,}')
    val_json = build_json(val, categ_map, 'val')

    t('=== Building TEST set ===')
    test = df[df['gbif_id'].isin(test_ids)].copy()
    print(f'  Initial (valid lat/lon/date intersected with test.csv): {len(test):,}')
    test_json = build_json(test, categ_map, 'test')

    t('Writing JSONs to disk')
    for name, obj in [('train', train_json), ('val', val_json), ('test', test_json)]:
        p = GEOPRIOR_DIR / f'{name}.json'
        with open(p, 'w') as f:
            json.dump(obj, f)
        print(f'  Wrote {p}  ({p.stat().st_size/1e6:.1f} MB,  {len(obj["images"]):,} images)')

    t('Done.', start)


if __name__ == '__main__':
    main()
