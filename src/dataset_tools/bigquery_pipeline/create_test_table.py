#!/usr/bin/env python3
"""
Create a stratified test table for clean.py testing.

Includes all 3 duplicate types (exact dupes, same-taxon multi-gbif,
multi-taxon conflicts) plus a clean random sample. Total ~35-40K rows.
"""
import argparse
from google.cloud import bigquery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="BQ dataset name (e.g. global_all_leps_2605)")
    parser.add_argument("--project", default="leps-ai")
    parser.add_argument("--source-table", default="training_images")
    parser.add_argument("--output-table", default="training_images_test")
    parser.add_argument("--max-type1-photos", type=int, default=10_000,
                        help="Cap on type-1 duplicate photo_ids (each appears ×2 rows)")
    parser.add_argument("--clean-sample", type=int, default=10_000,
                        help="Number of clean (non-duplicate) rows to include")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    src = f"`{args.project}.{args.dataset}.{args.source_table}`"
    dst_ref = f"{args.project}.{args.dataset}.{args.output_table}"
    dst = f"`{dst_ref}`"

    query = f"""
    CREATE OR REPLACE TABLE {dst} AS

    WITH
    -- Type 1: exact duplicate rows (same photo_id + taxon + gbif_id appears >1 time)
    type1_photos AS (
      SELECT photo_id
      FROM {src}
      GROUP BY photo_id, inat_taxon_id, gbif_id
      HAVING COUNT(*) > 1
      LIMIT {args.max_type1_photos}
    ),

    -- Type 2: same photo_id mapped to multiple taxa AND multiple gbif_ids
    type2_photos AS (
      SELECT photo_id
      FROM {src}
      GROUP BY photo_id
      HAVING COUNT(DISTINCT inat_taxon_id) > 1 AND COUNT(DISTINCT gbif_id) > 1
    ),

    -- Type 3: same photo_id + same taxon, but multiple gbif_ids
    type3_photos AS (
      SELECT photo_id
      FROM {src}
      GROUP BY photo_id
      HAVING COUNT(DISTINCT inat_taxon_id) = 1 AND COUNT(DISTINCT gbif_id) > 1
    ),

    all_dup_photos AS (
      SELECT photo_id FROM type1_photos
      UNION DISTINCT
      SELECT photo_id FROM type2_photos
      UNION DISTINCT
      SELECT photo_id FROM type3_photos
    ),

    -- All rows belonging to any duplicate photo_id
    dup_rows AS (
      SELECT t.*
      FROM {src} t
      INNER JOIN all_dup_photos d USING (photo_id)
    ),

    -- Clean rows: photo_ids not involved in any duplication
    clean_rows AS (
      SELECT t.*
      FROM {src} t
      WHERE t.photo_id NOT IN (SELECT photo_id FROM all_dup_photos)
      LIMIT {args.clean_sample}
    )

    SELECT * FROM dup_rows
    UNION ALL
    SELECT * FROM clean_rows
    """

    print(f"Creating {dst_ref} ...")
    print(f"  source : {args.project}.{args.dataset}.{args.source_table}")
    print(f"  type-1 cap: {args.max_type1_photos:,} photo_ids")
    print(f"  clean sample: {args.clean_sample:,} rows")
    print()

    job = client.query(query)
    job.result()

    ref = client.get_table(dst_ref)
    print(f"Done — {ref.num_rows:,} rows written to {dst_ref}")


if __name__ == "__main__":
    main()
