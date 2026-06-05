# `geoprior_categ_map.json` — provenance

**Frozen class-space contract** for the geo-prior FCNet model: `species_name -> class_id`
(0..12316). The trained model's output indices are bound to this exact alphabetical
ordering, so this file MUST NOT change while a model trained against it is in use.

## Where it comes from

Built from BigQuery (project `leps-ai`):

| Table | Provides |
|---|---|
| `leps-ai.global_butterflies_2604.gbif_inat_occurrences` | `gbifID` → `verbatimSpeciesScientificName` |
| `leps-ai.global_butterflies_2604.gbif_occurrence_location` | `gbif_id` → `decimallatitude` / `decimallongitude` / `eventdate` (derived from the public GBIF mirror, snapshot `public_gbif_2026-05`; see `src/dataset_tools/bq_squashfs/README.md`) |

**Definition:** every species (`verbatimSpeciesScientificName`) with ≥ 1 geocoded
occurrence (non-null lat, lon, eventdate), sorted alphabetically (Python `sorted()`),
enumerated 0..N-1.

**Snapshot (2026-05):** 12,317 species over 6,864,466 geocoded occurrences.

## Regenerate / verify

```bash
# verify this committed file still matches BigQuery (no writes)
python src/dataset_tools/build_geoprior_categ_map.py \
    --frozen research/geoprior/geoprior_categ_map.json

# (re)materialise all artifacts (categ_map + label_map + metadata + master lists)
python src/dataset_tools/build_geoprior_categ_map.py \
    --write --out-dir /mnt/melabbas/data/geoprior
```

The builder refuses to overwrite this artifact if the regenerated map differs
(use `--force` only when intentionally retiring the class space, which requires
retraining the model).
