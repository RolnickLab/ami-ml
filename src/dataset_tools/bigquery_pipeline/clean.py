#!/usr/bin/env python3
"""
Clean a training_images BQ table by removing duplicates and sparse taxa.

Duplicate types handled in order:
  1. Exact rows          — same (photo_id, inat_taxon_id, gbif_id), keep one row
  2. Same-taxon multi-gbif — same (photo_id, inat_taxon_id), multiple gbif_ids → keep MIN gbif_id
  3. Multi-taxon conflict  — same photo_id maps to multiple taxa → drop all or keep lowest taxon_id
  4. Min-images filter     — drop taxa with fewer than N images (default: off)

Overwrites the source table in-place unless --dry-run is passed.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from google.cloud import bigquery


def _step3_cte(strategy):
    if strategy == "drop":
        return "SELECT * FROM step2 WHERE photo_id NOT IN (SELECT photo_id FROM multi_taxon_photos)"
    else:  # keep-lowest-taxon-id
        return (
            "SELECT * FROM step2\n"
            "  QUALIFY ROW_NUMBER() OVER (PARTITION BY photo_id ORDER BY inat_taxon_id, gbif_id) = 1"
        )


def _cte_chain(src_ref, strategy, min_images):
    min_filter = f"t.cnt >= {min_images}" if min_images > 0 else "TRUE"
    return f"""
src AS (
  SELECT * FROM `{src_ref}`
),
step1 AS (
  -- Remove exact duplicate rows: same photo+taxon+gbif, keep one
  SELECT * FROM src
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY photo_id, inat_taxon_id, gbif_id
    ORDER BY dataset_source_uuid
  ) = 1
),
step2 AS (
  -- Same photo+taxon with multiple gbif_ids: keep the MIN gbif_id
  SELECT * FROM step1
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY photo_id, inat_taxon_id
    ORDER BY gbif_id
  ) = 1
),
multi_taxon_photos AS (
  -- Photo IDs that still map to more than one taxon after step2
  SELECT photo_id FROM step2
  GROUP BY photo_id
  HAVING COUNT(DISTINCT inat_taxon_id) > 1
),
step3 AS (
  -- Resolve multi-taxon conflicts
  {_step3_cte(strategy)}
),
taxa_img_count AS (
  SELECT inat_taxon_id, COUNT(*) AS cnt FROM step3 GROUP BY inat_taxon_id
),
step4 AS (
  -- Drop taxa below min-images threshold
  SELECT s.* FROM step3 s
  JOIN taxa_img_count t USING (inat_taxon_id)
  WHERE {min_filter}
)"""


def run_count_query(client, src_ref, strategy, min_images):
    """Return per-stage row/taxa counts in a single BQ query."""
    chain = _cte_chain(src_ref, strategy, min_images)
    query = f"""
WITH {chain}
SELECT stage, n, taxa FROM (
  SELECT 'input'     AS stage, COUNT(*) AS n, COUNT(DISTINCT inat_taxon_id) AS taxa FROM src UNION ALL
  SELECT 'step1'     AS stage, COUNT(*) AS n, COUNT(DISTINCT inat_taxon_id) AS taxa FROM step1 UNION ALL
  SELECT 'step2'     AS stage, COUNT(*) AS n, COUNT(DISTINCT inat_taxon_id) AS taxa FROM step2 UNION ALL
  SELECT 'step3'     AS stage, COUNT(*) AS n, COUNT(DISTINCT inat_taxon_id) AS taxa FROM step3 UNION ALL
  SELECT 'step4'     AS stage, COUNT(*) AS n, COUNT(DISTINCT inat_taxon_id) AS taxa FROM step4 UNION ALL
  SELECT 'conflicts' AS stage, COUNT(*) AS n, 0 AS taxa FROM multi_taxon_photos
)
ORDER BY stage
"""
    results = {row.stage: {"rows": row.n, "taxa": row.taxa}
               for row in client.query(query).result()}
    return results


def run_write_query(client, src_ref, dst_ref, strategy, min_images):
    chain = _cte_chain(src_ref, strategy, min_images)
    query = f"CREATE OR REPLACE TABLE `{dst_ref}` AS\nWITH {chain}\nSELECT * FROM step4"
    client.query(query).result()


def print_report(counts, strategy, min_images, dst_ref, dry_run):
    c = counts
    step_labels = [
        ("input",  "Input rows"),
        ("step1",  "Exact duplicates removed      (same photo+taxon+gbif, keep one)"),
        ("step2",  "Same-taxon multi-gbif resolved (keep MIN gbif_id)"),
        ("step3",  f"Multi-taxon conflicts handled  (strategy={strategy})"),
        ("step4",  f"Min-images-per-taxon filter    (threshold={min_images or 'off'})"),
    ]

    print()
    print("=" * 70)
    prev = None
    for key, label in step_labels:
        rows = c[key]["rows"]
        taxa = c[key]["taxa"]
        if prev is None:
            print(f"  {label}")
            print(f"    rows={rows:>10,}   taxa={taxa:>8,}")
        else:
            removed = prev - rows
            pct = removed / prev * 100 if prev else 0
            print(f"  {label}")
            print(f"    rows={rows:>10,}   removed={removed:>8,}  ({pct:.1f}%)")
        prev = rows

    # Step 3 extra detail
    n_conflict_photos = c["conflicts"]["rows"]
    print(f"    [{n_conflict_photos:,} photo_ids had conflicting taxa]")

    total_removed = c["input"]["rows"] - c["step4"]["rows"]
    total_pct = total_removed / c["input"]["rows"] * 100 if c["input"]["rows"] else 0
    print()
    print(f"  {'─'*60}")
    print(f"  Total removed : {total_removed:>10,}  ({total_pct:.1f}%)")
    print(f"  Output rows   : {c['step4']['rows']:>10,}")
    print(f"  Taxa before   : {c['input']['taxa']:>10,}")
    print(f"  Taxa after    : {c['step4']['taxa']:>10,}")
    dry_tag = "  [DRY RUN — table not written]" if dry_run else ""
    print(f"  Output table  : {dst_ref}{dry_tag}")
    print("=" * 70)
    print()


def build_log(counts, args, dst_ref, dry_run, started_at):
    c = counts
    return {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "project": args.project,
        "table": args.table,
        "output_table": dst_ref,
        "dry_run": dry_run,
        "multi_taxon_strategy": args.multi_taxon_strategy,
        "min_images_per_taxon": args.min_images_per_taxon,
        "steps": {
            "exact_duplicates": {
                "before": c["input"]["rows"],
                "after": c["step1"]["rows"],
                "removed": c["input"]["rows"] - c["step1"]["rows"],
            },
            "same_taxon_multi_gbif": {
                "before": c["step1"]["rows"],
                "after": c["step2"]["rows"],
                "removed": c["step1"]["rows"] - c["step2"]["rows"],
            },
            "multi_taxon_conflicts": {
                "before": c["step2"]["rows"],
                "after": c["step3"]["rows"],
                "removed": c["step2"]["rows"] - c["step3"]["rows"],
                "conflict_photo_ids": c["conflicts"]["rows"],
            },
            "min_images_filter": {
                "before": c["step3"]["rows"],
                "after": c["step4"]["rows"],
                "removed": c["step3"]["rows"] - c["step4"]["rows"],
                "threshold": args.min_images_per_taxon,
            },
        },
        "summary": {
            "input_rows": c["input"]["rows"],
            "output_rows": c["step4"]["rows"],
            "total_removed": c["input"]["rows"] - c["step4"]["rows"],
            "taxa_before": c["input"]["taxa"],
            "taxa_after": c["step4"]["taxa"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="BQ dataset name (e.g. global_all_leps_2605)")
    parser.add_argument("--project", default="leps-ai")
    parser.add_argument("--table", default="training_images",
                        help="Table to clean (overwritten in-place)")
    parser.add_argument("--min-images-per-taxon", type=int, default=0,
                        help="Drop taxa with fewer images than this (0 = off)")
    parser.add_argument("--multi-taxon-strategy",
                        choices=["drop", "keep-lowest-taxon-id"], default="drop",
                        help="How to handle a photo mapped to multiple taxa")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without writing the table")
    parser.add_argument("--log-file", help="Write JSON log to this path")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    src_ref = f"{args.project}.{args.dataset}.{args.table}"
    dst_ref = src_ref  # overwrite in-place

    client = bigquery.Client(project=args.project)

    print(f"clean.py  {'[DRY RUN] ' if args.dry_run else ''}started at {started_at}")
    print(f"  table              : {src_ref}")
    print(f"  multi-taxon        : {args.multi_taxon_strategy}")
    print(f"  min-images-per-taxon: {args.min_images_per_taxon or 'off'}")
    print()
    print("Running count queries ...")

    counts = run_count_query(client, src_ref, args.multi_taxon_strategy, args.min_images_per_taxon)
    print_report(counts, args.multi_taxon_strategy, args.min_images_per_taxon, dst_ref, args.dry_run)

    if not args.dry_run:
        print("Writing cleaned table ...")
        run_write_query(client, src_ref, dst_ref, args.multi_taxon_strategy, args.min_images_per_taxon)
        print(f"Done — {counts['step4']['rows']:,} rows written to {dst_ref}")

    log = build_log(counts, args, dst_ref, args.dry_run, started_at)
    if args.log_file:
        with open(args.log_file, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Log written to {args.log_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
