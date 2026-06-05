# geoprior_fagner

Geo-prior network (SINR-style **FCNet**) used by the geo-prior pipeline. The
model predicts `p(species | lat, lon, date)` and is consumed by
`src/geoprior/train_geoprior.py` and `src/geoprior/predict_geoprior.py`.

## Source

Taken from Fagner Cunha's lepsAI `geo_prior` package:

- **Upstream:** https://github.com/mihow/fagner-lepsAI (`geo_prior/`)
- **Commit:** `ff4ccd1555f1ac463c79c884b2a2218062f936a0`
- **License:** Apache-2.0 (© 2022 Fagner Cunha; © 2023 Rolnick Lab, Mila). The
  original license headers are kept intact in each module.

## Contents

| Module | What it provides |
|---|---|
| `models.py` | `FCNet` (the geo-prior network) and `ResLayer` |
| `losses.py` | `weighted_binary_cross_entropy`, `log_loss` |
| `dataloader.py` | `LocationDataset`, `BalancedSampler`, `RandSpatioTemporalGenerator`, location/date encoding |

## Why it lives here

The trained geo-prior model's weights are tied to this exact network
definition, so the code is kept in-tree (frozen at the commit above) rather
than pulled from an external clone at runtime. This keeps the pipeline
self-contained and reproducible — no machine-specific paths. To pick up
upstream changes, re-copy from the upstream commit and bump the reference
above (and expect to retrain, since the architecture contract changes).
