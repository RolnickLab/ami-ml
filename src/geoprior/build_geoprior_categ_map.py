#!/usr/bin/env python3
"""
Build / verify the geo-prior category map (species_name -> class_id).

This is the FROZEN class-space contract for the geo-prior FCNet model. The
trained model's output indices are bound to this exact alphabetical ordering,
so the committed ``geoprior_categ_map.json`` must NOT change once a model has
been trained against it. This script regenerates the map from BigQuery and
*verifies* it against the committed artifact; it refuses to silently overwrite
on drift (use --write to (re)materialise the artifacts intentionally).

Source (BigQuery, project ``leps-ai``)
--------------------------------------
  - leps-ai.global_butterflies_2604.gbif_inat_occurrences
        gbifID -> verbatimSpeciesScientificName
  - leps-ai.global_butterflies_2604.gbif_occurrence_location
        gbif_id -> decimallatitude / decimallongitude / eventdate
        (itself derived from the public GBIF mirror, snapshot
         ``public_gbif_2026-05``; see src/dataset_tools/bq_squashfs/README.md)

Class-space definition
----------------------
  Every species (``verbatimSpeciesScientificName``) with >= 1 geocoded
  occurrence (non-null lat, lon, eventdate), sorted alphabetically with
  Python's default ``sorted()`` over the species-name strings, enumerated
  0..N-1. As of the 2026-05 snapshot this is 12,317 classes.

Outputs (only written with --write)
-----------------------------------
  geoprior_categ_map.json          species -> class_id (int)   [FROZEN artifact]
  geoprior_label_map.json          class_id (str) -> species   (HF-style reverse)
  geoprior_metadata.json           species -> {class_id, n_geocoded_occurrences}
  master_species_list.txt          sorted species, one per line
  master_species_with_counts.json  species -> n_geocoded_occurrences

Usage
-----
  # verify the committed frozen map still matches BigQuery (default; no writes)
  python -m src.geoprior.build_geoprior_categ_map

  # (re)generate every artifact into the configured data dir
  python -m src.geoprior.build_geoprior_categ_map --write
"""
import argparse
import json
import sys
from pathlib import Path

from google.cloud import bigquery

from src.geoprior import config

# Counts the geocoded occurrences per species. The WHERE clause mirrors
# build_geoprior_json.py::fetch_all_geocoded so the class space is exactly the
# set of species that survive into the geo-prior JSON pipeline.
SPECIES_COUNT_QUERY = f"""
SELECT
  o.verbatimSpeciesScientificName AS species_name,
  COUNT(*)                        AS n_geocoded_occurrences
FROM `{config.TBL_OCCURRENCES}` o
JOIN `{config.TBL_LOCATION}` l
  ON l.gbif_id = o.gbifID
WHERE l.decimallatitude  IS NOT NULL
  AND l.decimallongitude IS NOT NULL
  AND l.eventdate        IS NOT NULL
  AND o.verbatimSpeciesScientificName IS NOT NULL
GROUP BY species_name
"""


def fetch_species_counts():
    """Return {species_name: n_geocoded_occurrences} and bytes billed."""
    client = bigquery.Client(project=config.BQ_PROJECT)
    job = client.query(SPECIES_COUNT_QUERY)
    rows = list(job.result())
    counts = {r["species_name"]: int(r["n_geocoded_occurrences"]) for r in rows}
    return counts, (job.total_bytes_billed or 0)


def build_maps(counts):
    """Derive every artifact from the {species: count} dict (alphabetical)."""
    species = sorted(counts.keys())
    categ_map = {s: i for i, s in enumerate(species)}
    label_map = {str(i): s for s, i in categ_map.items()}
    metadata = {
        s: {"class_id": categ_map[s], "n_geocoded_occurrences": counts[s]}
        for s in species
    }
    with_counts = {s: counts[s] for s in species}
    return species, categ_map, label_map, metadata, with_counts


def _diff_categ_map(frozen, regenerated):
    """Human-readable summary of how a regenerated map differs from frozen."""
    fk, rk = set(frozen), set(regenerated)
    added = sorted(rk - fk)
    removed = sorted(fk - rk)
    reindexed = sorted(s for s in (fk & rk) if frozen[s] != regenerated[s])
    return added, removed, reindexed


def verify_against(frozen_path, regenerated, label):
    """Compare a regenerated dict to an on-disk JSON. Return True if identical."""
    frozen_path = Path(frozen_path)
    if not frozen_path.exists():
        print(f"  [{label}] frozen file not found: {frozen_path} (skipping)")
        return None
    frozen = json.loads(frozen_path.read_text())
    if frozen == regenerated:
        print(
            f"  [{label}] VERIFY OK — {len(regenerated):,} entries, identical to {frozen_path}"
        )
        return True
    print(f"  [{label}] VERIFY FAILED — differs from {frozen_path}")
    if label == "categ_map":
        added, removed, reindexed = _diff_categ_map(frozen, regenerated)
        print(f"      added species:    {len(added)}  e.g. {added[:5]}")
        print(f"      removed species:  {len(removed)}  e.g. {removed[:5]}")
        print(f"      reindexed (id changed): {len(reindexed)}  e.g. {reindexed[:5]}")
    else:
        print(
            f"      frozen has {len(frozen):,} entries, regenerated has {len(regenerated):,}"
        )
    return False


def write_artifacts(out_dir, species, categ_map, label_map, metadata, with_counts):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geoprior_categ_map.json").write_text(json.dumps(categ_map))
    (out_dir / "geoprior_label_map.json").write_text(json.dumps(label_map))
    (out_dir / "geoprior_metadata.json").write_text(json.dumps(metadata))
    (out_dir / "master_species_with_counts.json").write_text(json.dumps(with_counts))
    (out_dir / "master_species_list.txt").write_text("\n".join(species) + "\n")
    for name in (
        "geoprior_categ_map.json",
        "geoprior_label_map.json",
        "geoprior_metadata.json",
        "master_species_with_counts.json",
        "master_species_list.txt",
    ):
        p = out_dir / name
        print(f"  wrote {p}  ({p.stat().st_size/1e3:.0f} KB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frozen",
        default=str(config.CATEG_MAP_PATH),
        help="Path to the committed frozen geoprior_categ_map.json to verify against",
    )
    ap.add_argument(
        "--verify-counts",
        default=None,
        help="Optional path to an existing master_species_with_counts.json "
        "to validate the BQ count query against",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Materialise all five artifacts into --out-dir",
    )
    ap.add_argument(
        "--out-dir",
        default=str(config.DATA_DIR),
        help="Directory to write artifacts to (with --write)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Write even if verification against --frozen fails",
    )
    args = ap.parse_args()

    print(
        f"Querying BigQuery (project={config.BQ_PROJECT}) for geocoded species counts ..."
    )
    counts, billed = fetch_species_counts()
    species, categ_map, label_map, metadata, with_counts = build_maps(counts)
    total_occ = sum(counts.values())
    print(
        f"  {len(species):,} species, {total_occ:,} geocoded occurrences, "
        f"scanned {billed/1e6:.1f} MB (~${billed/1e12*5:.4f})"
    )

    print("Verifying ...")
    ok = verify_against(args.frozen, categ_map, "categ_map")
    if args.verify_counts:
        verify_against(args.verify_counts, with_counts, "counts")

    if args.write:
        if ok is False and not args.force:
            print(
                "Refusing to --write: regenerated map differs from the frozen "
                "artifact. Re-run with --force only if you intend to retire the "
                "current class space (and retrain the model)."
            )
            sys.exit(1)
        print(f"Writing artifacts to {args.out_dir} ...")
        write_artifacts(
            args.out_dir, species, categ_map, label_map, metadata, with_counts
        )

    # Exit non-zero on a real mismatch so CI / callers can catch drift.
    if ok is False and not args.force:
        sys.exit(1)


if __name__ == "__main__":
    main()
