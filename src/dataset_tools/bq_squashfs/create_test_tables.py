#!/usr/bin/env python3
"""
Create small BQ test tables for testing download_images.py without touching production.

Creates:
  test_training_images          — 50 rows sampled from training_images, fetch_status='pending'
  test_training_images_downloads — empty table, same schema as training_images_downloads

Usage:
    python create_test_tables.py
    python create_test_tables.py --n-rows 100
"""

import argparse
from google.cloud import bigquery

BQ_PROJECT = "leps-ai"
BQ_DATASET = "global_butterflies_2604"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rows", type=int, default=50,
                        help="Number of rows to sample from training_images (default: 50)")
    args = parser.parse_args()

    client = bigquery.Client(project=BQ_PROJECT)
    prefix = f"{BQ_PROJECT}.{BQ_DATASET}"

    # ── test_training_images ─────────────────────────────────────────────────
    print(f"Creating {prefix}.test_training_images ({args.n_rows} rows)...")
    client.query(f"""
        CREATE OR REPLACE TABLE `{prefix}.test_training_images` AS
        SELECT
            photo_id,
            gbif_id,
            inat_taxon_id,
            dataset_source_uuid,
            absolute_url,
            relative_local_path,
            'pending'            AS fetch_status,
            CAST(NULL AS INT64)  AS image_width,
            CAST(NULL AS INT64)  AS image_height,
            CAST(NULL AS INT64)  AS image_size,
            CAST(NULL AS BOOL)   AS corrupted
        FROM `{prefix}.training_images`
        WHERE fetch_status = 'downloaded'
        LIMIT {args.n_rows}
    """).result()

    n = client.get_table(f"{prefix}.test_training_images").num_rows
    print(f"  Created: {n} rows, all fetch_status='pending'")

    # ── test_training_images_downloads ───────────────────────────────────────
    print(f"Creating {prefix}.test_training_images_downloads (empty)...")
    client.query(f"""
        CREATE OR REPLACE TABLE `{prefix}.test_training_images_downloads`
        (
            dataset_source_uuid STRING,
            fetch_status        STRING,
            image_width         INT64,
            image_height        INT64,
            image_size          INT64,
            corrupted           BOOL
        )
    """).result()
    print("  Created: 0 rows")

    print("\nDone. Run test with:")
    print(f"  python download_images.py \\")
    print(f"      --staging-dir /scratch/$USER/test_download \\")
    print(f"      --num-jobs 1 --task-id 0 \\")
    print(f"      --num-workers 8 --chunk-size {args.n_rows} \\")
    print(f"      --limit {args.n_rows} --table-prefix test_")


if __name__ == "__main__":
    main()
