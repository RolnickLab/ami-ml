#!/usr/bin/env python
"""
Bridge script: reads a DwC-A file to extract verbatimScientificName (or another
name column) and joins it onto an existing annotations CSV produced by the
ami-dataset clean-dataset step.

This is needed because load_dwca_data() in src/dataset_tools/utils.py does not
include verbatimScientificName in its column selection.

Outputs:
  - Augmented annotations CSV with the label column added
  - Category map JSON (species_name -> integer_id)
"""

import argparse
import json
import sys

import pandas as pd
from dwca.read import DwCAReader


def print_dwca_summary(occ_df: pd.DataFrame) -> None:
    """Print summary statistics for the DwC-A occurrence data."""
    print("\n=== DwC-A Summary ===")
    print(f"  Total occurrences: {len(occ_df)}")

    for col in [
        "verbatimScientificName",
        "scientificName",
        "species",
        "family",
        "order",
    ]:
        if col in occ_df.columns:
            n_unique = occ_df[col].dropna().nunique()
            n_missing = occ_df[col].isna().sum()
            print(f"  Unique {col}: {n_unique}  (missing: {n_missing})")

    if "eventDate" in occ_df.columns:
        dates = pd.to_datetime(occ_df["eventDate"], errors="coerce").dropna()
        if len(dates) > 0:
            print(f"  Date range: {dates.min().date()} to {dates.max().date()}")

    for coord_col in ["decimalLatitude", "decimalLongitude"]:
        if coord_col in occ_df.columns:
            vals = pd.to_numeric(occ_df[coord_col], errors="coerce").dropna()
            if len(vals) > 0:
                lo = "{:.2f}".format(vals.min())
                hi = "{:.2f}".format(vals.max())
                print(f"  {coord_col}: {lo} to {hi}")

    print()


def report_missing_labels(occ_df: pd.DataFrame, label_column: str) -> None:
    """Report occurrences where the label column is missing/empty."""
    missing_mask = occ_df[label_column].isna() | (
        occ_df[label_column].astype(str).str.strip() == ""
    )
    n_missing = missing_mask.sum()

    if n_missing == 0:
        print(f"All occurrences have a value for '{label_column}'.")
        return

    msg = f"\nWARNING: {n_missing} occurrences missing '{label_column}'"
    print(msg)
    alt_cols = [
        c
        for c in [
            "scientificName",
            "species",
            "acceptedScientificName",
            "verbatimScientificName",
        ]
        if c in occ_df.columns and c != label_column
    ]

    missing_rows = occ_df[missing_mask].head(20)
    for _, row in missing_rows.iterrows():
        alt_info = ", ".join(f"{c}={row.get(c, 'N/A')}" for c in alt_cols)
        print(f"  coreid={row['id']}: {alt_info}")

    if n_missing > 20:
        print(f"  ... and {n_missing - 20} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Augment annotations CSV with species names from DwC-A"
    )
    parser.add_argument(
        "--dwca-file", required=True, help="Path to Darwin Core Archive zip file"
    )
    parser.add_argument(
        "--annotations-csv",
        required=True,
        help="CSV from clean-dataset step (must have 'coreid' column)",
    )
    parser.add_argument(
        "--output-csv", required=True, help="Path to save augmented annotations CSV"
    )
    parser.add_argument(
        "--category-map-json", required=True, help="Path to save category map JSON"
    )
    parser.add_argument(
        "--label-column",
        default="verbatimScientificName",
        help="DwC-A column to use as species label (default: verbatimScientificName)",
    )
    args = parser.parse_args()

    # --- Read DwC-A ---
    print(f"Reading DwC-A: {args.dwca_file}")
    with DwCAReader(args.dwca_file) as dwca:
        occ_df = dwca.pd_read(
            "occurrence.txt", parse_dates=True, on_bad_lines="skip", low_memory=False
        )
        media_df = dwca.pd_read(
            "multimedia.txt", parse_dates=True, on_bad_lines="skip", low_memory=False
        )

    print(f"  Occurrences: {len(occ_df)}, Multimedia records: {len(media_df)}")
    print_dwca_summary(occ_df)

    # --- Check label column exists ---
    if args.label_column not in occ_df.columns:
        available = [
            c for c in occ_df.columns if "name" in c.lower() or "species" in c.lower()
        ]
        print(f"ERROR: Column '{args.label_column}' missing from occurrence data.")
        print(f"  Available name-related columns: {available}")
        sys.exit(1)

    report_missing_labels(occ_df, args.label_column)

    # --- Build coreid -> label mapping ---
    # The 'id' column in occurrence.txt corresponds to 'coreid' in multimedia/annotations
    name_map = occ_df[["id", args.label_column]].drop_duplicates()
    name_map = name_map.rename(columns={"id": "coreid"})

    # --- Read annotations CSV ---
    print(f"Reading annotations: {args.annotations_csv}")
    annotations = pd.read_csv(args.annotations_csv)
    print(f"  Rows: {len(annotations)}")

    if "coreid" not in annotations.columns:
        print("ERROR: annotations CSV does not have a 'coreid' column.")
        print(f"  Available columns: {list(annotations.columns)}")
        sys.exit(1)

    # --- Join ---
    # Ensure coreid types match for the merge
    annotations["coreid"] = annotations["coreid"].astype(str)
    name_map["coreid"] = name_map["coreid"].astype(str)

    merged = annotations.merge(name_map, on="coreid", how="left")

    # --- Drop rows with missing labels ---
    missing_mask = merged[args.label_column].isna() | (
        merged[args.label_column].astype(str).str.strip() == ""
    )
    n_dropped = missing_mask.sum()
    if n_dropped > 0:
        print(f"WARNING: Dropping {n_dropped} rows with missing '{args.label_column}'")
    merged = merged[~missing_mask].copy()

    print(f"  Rows after join and filter: {len(merged)}")

    # --- Save augmented CSV ---
    merged.to_csv(args.output_csv, index=False)
    print(f"Saved augmented annotations: {args.output_csv}")

    # --- Build and save category map ---
    species_list = sorted(merged[args.label_column].unique())
    category_map = {name: idx for idx, name in enumerate(species_list)}

    with open(args.category_map_json, "w") as f:
        json.dump(category_map, f, indent=2)
    print(f"Saved category map ({len(category_map)} species): {args.category_map_json}")

    # --- Per-species image count summary ---
    print(f"\n=== Per-species image counts ({args.label_column}) ===")
    counts = merged[args.label_column].value_counts().sort_index()
    for species, count in counts.items():
        print(f"  {species}: {count}")

    print(f"\nTotal images: {len(merged)}")
    print(f"Total species: {len(category_map)}")


if __name__ == "__main__":
    main()
