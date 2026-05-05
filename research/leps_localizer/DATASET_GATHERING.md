# Arthropod Localization Dataset Gathering

> **Provenance.** Originated in the `chroma-backend` private repo at
> `detector_dataset/DATASET.md` (branch `worktree-detector-training`).
> Copied here unmodified except for this header so the leps localizer
> training repo is self-contained. The source-of-truth pipeline still
> lives in chroma-backend because it needs FG production DB credentials.
> Re-sync:
>
>     cp ~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/DATASET.md \
>        research/leps_localizer/DATASET_GATHERING.md
>
> **Splits in this repo.** Originally all four sets were locked as
> held-out eval. For the leps localizer first model, the three FG sets
> are repartitioned into **train + val** (deterministic 90/10 by
> `md5(seed:photo_id)`); **leeds-butterflies stays held-out eval**.
> See `DESIGN.md` § "Held-out test set" for rationale.

How `detector_dataset/` produces COCO-format object-detection data for
arthropod localization. Two pipelines, one common output schema, so a
downstream eval harness can run the same detector against both and
report per-source metrics.

- **FG**: bboxes drawn by Fieldguide users via the upload crop tool.
  Real-world distribution; weak to noisy labels but representative of
  the kind of imagery the detector will actually see.
- **Leeds**: bboxes derived from human-curated segmentation masks
  in the Leeds Butterfly Dataset (BMVC 2009). Held-out reference with
  tight, clean ground truth.

Both produce a single COCO category — `{id: 1, name: "arthropod",
supercategory: "organism"}` — so models trained or evaluated on one are
swappable on the other.

---

## Output schema

Each built dataset directory contains:

```
<dataset>/
├── annotations.json    # COCO file
├── images/             # actual image files (FG: downloaded; Leeds: symlinked)
├── manifest.jsonl      # FG only — pre-COCO row-per-photo record
├── fetch.log           # FG only — one JSON line per fetch attempt
└── overlays/           # `detector-dataset overlays …` output (optional)
```

`annotations.json` is valid COCO with a few non-standard fields prefixed
`fg_` or `leeds_` for provenance:

| FG image field         | Meaning |
|---|---|
| `fg_photo_id`          | FG `photos.mongo_id` |
| `fg_species_id`        | FG `categories.mongo_id` of the leaf taxon |
| `fg_species_parents`   | parents array (root-to-leaf taxonomic chain) |
| `fg_photographer_id`   | FG `users.mongo_id` |
| `fg_image_url_s3`      | original AWS S3 URL on `production-chroma` |
| `fg_image_url_arbutus` | mirror URL on `fieldguide-production` (Arbutus, Compute Canada) |
| `fg_created_at`        | photo creation timestamp |

| FG annotation field    | Meaning |
|---|---|
| `fg_bbox_source`       | always `"user_crop_info"` — bbox came from the photo's `photo_images.crop_info` |

| Leeds image field      | Meaning |
|---|---|
| `leeds_species_id`     | category id (`001`–`010`) parsed from the filename |
| `leeds_species_name`   | scientific name from the README mapping |
| `leeds_image_stem`     | filename stem (matches the `_seg0.png` mask) |

| Leeds annotation field | Meaning |
|---|---|
| `leeds_bbox_source`    | always `"tight_from_mask"` |

The `info` block on every COCO file records `build_seed`,
`fg_root_category_id`, and `manifest_sha256` (FG) or `source_url`
(Leeds) so the build is auditable.

---

## FG pipeline

Three stages run end-to-end via `uv run detector-dataset build <name>`:

### 1. Extract (`extract.py`)

A single SQL query against the FG production DB selects candidate photos.
The body of the query is in `extract.py:26-52`; the operative filter is:

```sql
JOIN photo_images pi ON pi.photo_id = p.mongo_id AND pi.sort_order = 0
JOIN categories c    ON c.mongo_id = p.category_mongo_id
WHERE
  pi.crop_info IS NOT NULL                                  -- user drew a bbox
  AND pi.crop_info::jsonb ? 'width'
  AND (pi.crop_info->>'width')::numeric < pi.width          -- not a no-op crop
  AND c.parents @> ARRAY[<root_category_id>]::text[]        -- in the target clade
  AND pi.width  >= 400
  AND pi.height >= 400
ORDER BY md5(photo_id || <seed>)                            -- deterministic shuffle
LIMIT <hard_cap>                                            -- candidate pool size
```

A few details that matter:

- `pi.sort_order = 0` is required. FG `photo_images` allows multiple rows
  per `photo_id` (variants like web vs. mobile); without this filter we
  get duplicate manifest entries.
- The photo→category join uses `category_mongo_id` (varchar), not
  `category_id` (int FK). The mongo-id path is the canonical one.
- `c.parents @> ARRAY[…]::text[]` uses the GIN index on the
  `categories.parents` array. Performance is dominated by the `ORDER BY
  md5(…)` sort, not the filter.
- `(crop_info->>'width')::numeric` — some `crop_info.width` values are
  stored as float strings (e.g. `"1456.05"`); never cast to `::int`.

After the SQL returns up to `hard_cap` candidate rows, Python-side
processing in `extract.process_rows` does:

1. **Bbox validation** (`bbox.py`): convert `crop_info` `{x,y,w,h}`
   to int `[x, y, w, h]`, check `(x, y) >= 0` and `(x+w, y+h) <=
   (image_w, image_h)`.
2. **Area-fraction filter** (`extract.py:22-23`): drop bboxes outside
   `[MIN_AREA_FRACTION, MAX_AREA_FRACTION]` (currently `[0.01, 0.95]`).
   This filters trivial empty crops and full-frame "crops" that aren't
   really cropping.
3. **Sampling cap** (`sampling.py`): selects up to `target_count` rows
   under a per-(species, user) cap that starts at 1 and bumps to 2,
   then 3, … until either the target is reached or the pool is
   saturated. This guarantees diversity for small targets while
   gracefully accepting concentration when the target outgrows the
   distinct-pair pool.

Output: `manifest.jsonl` (one JSON row per kept photo).

### 2. Fetch (`fetch.py`)

For each manifest entry, attempt to download the image with a four-step
fallback so bandwidth costs and origin pressure are minimized:

1. **AWS S3 anonymous** — read from `production-chroma.s3.amazonaws.com`
   (some images are publicly readable).
2. **Arbutus anonymous** — same key on the
   `fieldguide-production` bucket at `object-arbutus.cloud.computecanada.ca`
   (Compute Canada object store; cheaper egress).
3. **Arbutus authenticated** — same bucket with the
   `ARBUTUS_ACCESS_KEY_ID` / `ARBUTUS_SECRET_ACCESS_KEY` credentials
   from `.env`.
4. **Cache** — if the destination file already exists from a prior
   build, reuse it.

After download the file is opened with PIL and its dimensions are
checked against `manifest.image_w / image_h`. Any mismatch (e.g. an
image rotated post-upload) skips the photo. Each attempt is logged
to `fetch.log` as one JSON line for forensics.

Output: `images/<photo_id>.jpg`.

### 3. Assemble (`assemble.py`)

Reads `manifest.jsonl`, intersects with the actually-fetched images, and
writes `annotations.json`. The COCO `info.manifest_sha256` is the
digest of the raw manifest, so two builds with the same configs are
verifiable identical.

---

## FG dataset configs (`configs.py`)

Three named specs cover the planned phases:

| name | clade root (mongo_id, common name) | target | hard cap |
|---|---|---|---|
| `leps-butterflies-500` | `552e76f2…` Butterflies (Papilionoidea) — 38K subtree cats | 500 | 5,000 |
| `leps-2000`            | `5926f024…` Butterflies & Moths (Lepidoptera) — 218K cats | 2,000 | 20,000 |
| `arthropoda-all`       | `531507a0…` Arthropods (Arthropoda) — 1.1M cats | unlimited | 200,000 |

**FG categories use common-name labels, not Latin.** The
`detector-dataset taxonomy` helper hardcodes the Latin names and returns
zero hits against production — those names exist only as data junk.
Real clade roots were resolved by walking the `parents[]` array from the
Leps stream's seed categories ("Butterflies", "Moths") up to Animals.
Don't trust the `taxonomy.py` output as-is.

---

## Leeds pipeline

Single stage: `uv run detector-dataset build-leeds`.

1. Download `leedsbutterfly_dataset_v1.0.zip` from
   `https://zenodo.org/records/7558895/files/leedsbutterfly_dataset_v1.0.zip`
   (cached locally).
2. Extract.
3. For each image, read its `*_seg0.png` segmentation mask, derive a
   tight COCO bbox, write the COCO entry.

The mask handling has two non-obvious details, both required for
correctness:

- **Read raw palette indices.** Leeds masks are mode `"P"` PNGs with
  index values `0..3`. `Image.open(p).convert("L")` would map those
  through luminance to `0/76/150/255`, breaking any rule that depends on
  the exact index value. Use `np.array(Image.open(p))` directly.
- **Apply the README's foreground rule.** The Leeds v1.0 README states
  *"Pixels with values 1 and 3 represent foreground pixels, and others
  background pixels."* Treating any non-zero pixel as foreground (the
  obvious shortcut) is wrong: value 2 pixels are scattered around image
  edges and inflate the bbox to the full frame. The
  `leeds_foreground()` helper applies the correct rule.

Source images are symlinked (not copied) into `<dataset>/images/` so the
COCO `file_name` references resolve when readers join them against the
COCO root.

---

## Reproducibility & seeds

Each build is parameterized by `seed` (default 42). The seed feeds two
places:

- The DB query's `ORDER BY md5(photo_id || seed)` — deterministically
  shuffles the candidate pool.
- The Python sampler's tie-breaking when multiple rows compete for the
  same (species, user) slot.

Two builds with the same `(spec, seed, target_count, hard_cap)` against
the same DB snapshot will produce byte-identical manifests; the COCO
file's `info.manifest_sha256` is the verification anchor.

---

## Known limitations

- **FG bbox quality is variable.** Crops are user-drawn at upload time
  via a 640×640 square widget. Most are reasonable; a non-trivial
  fraction enclose the subject plus significant background, and a few
  are essentially full-frame (frac > 0.9). No automatic post-filter
  catches these — manual vetting is the gate. Run
  `detector-dataset overlays <name>` to render `frac<F>_<photo_id>.jpg`
  thumbnails sorted by area fraction.

- **Area-fraction skew.** The user-crop convention biases the dataset
  toward subject-fills-frame images: in a 1000-sample pull from
  Papilionoidea, ~49% of bboxes had `frac >= 0.5` and only ~13% had
  `frac < 0.25` (the "small subject, real localization challenge"
  bucket). This reflects the real input distribution to the detector
  (many uploads are tight phone close-ups), so it isn't filtered out;
  downstream eval should report per-bucket metrics if the spread
  matters.

- **One image per (species, user) at the floor.** The cap algorithm
  bumps to 2, 3, … only when the target can't be hit at cap=1. For
  small targets (e.g. 100 from Papilionoidea), this nearly guarantees
  one-photo-per-photographer-per-species; for large targets it
  concentrates around prolific contributors.

- **Pre-rotation EXIF mismatches.** A small fraction of FG photos have
  `pi.width / pi.height` swapped relative to the actual image bytes
  (likely EXIF orientation not applied at storage time). The fetcher
  catches these as dimension mismatches and skips them; in a 1000
  pull this accounted for 1 skipped image.

---

## File map

```
detector_dataset/
├── DATASET.md                            ← this file
├── README.md                             ← quick start
├── pyproject.toml
├── src/detector_dataset/
│   ├── bbox.py        - crop_info → COCO xywh, validation
│   ├── urls.py        - S3 ↔ Arbutus URL conversion (handles 4 FG URL formats)
│   ├── manifest.py    - JSONL reader/writer for the intermediate manifest
│   ├── sampling.py    - per-(species, user) cap algorithm
│   ├── configs.py     - dataset specs + clade IDs
│   ├── db.py          - read-only psycopg connection helper
│   ├── taxonomy.py    - root-category lookup (broken, see "configs" above)
│   ├── extract.py     - Stage 1: SQL → manifest.jsonl
│   ├── fetch.py       - Stage 2: download images with fallback
│   ├── assemble.py    - Stage 3: manifest → annotations.json
│   ├── leeds.py       - Leeds Zenodo → annotations.json
│   ├── overlays.py    - bbox overlay rendering
│   └── cli.py         - argparse entrypoint
└── tests/             - 50 unit tests, all pure (no DB or network)
```

---

## Rebuilding from scratch

```bash
cd detector_dataset
uv sync
cp .env.example .env  # fill PSQL_URL + ARBUTUS_* from chroma-backend creds

# FG: requires production DB access + Arbutus credentials
uv run detector-dataset build leps-butterflies-500 --target 1000

# Leeds: requires only outbound HTTPS to zenodo.org
uv run detector-dataset build-leeds

# Visual vetting:
uv run detector-dataset overlays leps-butterflies-500
uv run detector-dataset overlays --dataset-dir datasets/leeds-butterflies
```

A 1000-image FG pull takes ~3 minutes on a good network (the bottleneck
is the deterministic-sort SQL query, ~30s, then sequential image
fetches). Leeds is ~1 minute total once the Zenodo zip is cached.
