#!/usr/bin/env python3
"""
Export BigQuery query results to a CSV file.

Reads a SQL query from a file, executes it against BigQuery, and streams
results row-by-row to a CSV file. Generic — works with any SELECT query.

Usage:
    python bq_export.py \\
        --query-file queries/global_min25occ.sql \\
        --output /project/.../global_min25occ.csv

    python bq_export.py \\
        --query-file queries/global_max2000img.sql \\
        --output /project/.../global_max2000img.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from google.cloud import bigquery

LOG_EVERY = 500_000


def log(msg: str) -> None:
    print(msg, flush=True)


def export(query_file: Path, output: Path, project: str) -> None:
    query = query_file.read_text().strip()
    if not query:
        log(f"ERROR: query file {query_file} is empty", file=sys.stderr)
        sys.exit(1)

    client = bigquery.Client(project=project)

    log(f"Query file : {query_file}")
    log(f"Output     : {output}")
    log(f"Submitting query ...")

    job = client.query(query)
    log(f"Job ID     : {job.job_id}")
    log("Streaming rows to CSV ...")

    output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rows_written = 0
    unique_species: set[str] = set()
    unique_occurrences: set[int] = set()

    with open(output, "w", newline="") as f:
        writer = None
        for row in job.result():
            row_dict = dict(row)

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
                writer.writeheader()

            writer.writerow(row_dict)
            rows_written += 1

            if "species_name" in row_dict:
                unique_species.add(row_dict["species_name"])
            if "gbif_id" in row_dict:
                unique_occurrences.add(row_dict["gbif_id"])

            if rows_written % LOG_EVERY == 0:
                elapsed = (time.perf_counter() - t0) / 60
                rate = rows_written / (time.perf_counter() - t0)
                log(f"  {rows_written:,} rows  {elapsed:.1f} min  ({rate:,.0f} rows/s)")

    if rows_written == 0:
        log("ERROR: query returned 0 rows", file=sys.stderr)
        sys.exit(1)

    elapsed = (time.perf_counter() - t0) / 60
    size_mb = output.stat().st_size / 1024**2

    log(f"\n=== Export complete ===")
    log(f"  Rows written    : {rows_written:,}")
    if unique_species:
        log(f"  Unique species  : {len(unique_species):,}")
    if unique_occurrences:
        log(f"  Unique gbif_ids : {len(unique_occurrences):,}")
    log(f"  Output          : {output}  ({size_mb:.0f} MB)")
    log(f"  Elapsed         : {elapsed:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query-file", required=True, type=Path,
                        help="Path to a .sql file containing the SELECT query")
    parser.add_argument("--output",     required=True, type=Path,
                        help="Path to write the output CSV file")
    parser.add_argument("--project",    default="leps-ai",
                        help="GCP project (default: leps-ai)")
    args = parser.parse_args()

    if not args.query_file.exists():
        print(f"ERROR: query file not found: {args.query_file}", file=sys.stderr)
        sys.exit(1)

    export(
        query_file=args.query_file,
        output=args.output,
        project=args.project,
    )


if __name__ == "__main__":
    main()
