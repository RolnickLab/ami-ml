# Session prompt — independent iNat-RG eval for the two newly-exported global butterfly classifiers

## Goal

Run an **independent accuracy + PyTorch↔CoreML parity** evaluation of the two
classifiers that just shipped to the LepsAI iOS app, using the last ~1000
research-grade (RG) butterfly observations from iNaturalist as the eval set.
The CoreML conversion has only been smoke-tested (spec loads, outputs named
`logits`) — we don't yet know whether the converted models actually predict
the same thing as the PyTorch reference on real photos, or how either one
performs out-of-distribution against fresh iNat uploads. This eval closes
both gaps.

## Models under test

Both are timm `resnet50` trained by Mohamed Elabbas, exported with
`research/leps_localizer/scripts/export_classifier.py --raw-output --ios-bundle`,
ImageNet preprocessing baked into the *category map + side-files*, NOT into the
graph (the LepsAI iOS contract).

| modelID | n_classes | input_size | weights (HF) | S3 zip | sha256 |
|---|---|---|---|---|---|
| `global-butterflies-resnet50-512` | 8851 species (binomial) | 512 | `hf://mohammedelabbas/global-butterflies-max1000img-512/resnet50_20260504_135731_checkpoint.pt` | https://object-arbutus.cloud.computecanada.ca/ami-models/mobile/global-butterflies-resnet50-512.zip | (re-fetch with `curl -sI` or `sha256sum`) |
| `global-butterflies-subspecies-resnet50-128` | 19411 subspecies (trinomial, e.g. `"Heliconius melpomene martinae"`) | 128 | `hf://mohammedelabbas/global-butterflies-subspecies-max1000img-128/resnet50_<ts>_checkpoint.pt` (resolve from wandb `x54e20om` or HF README) | https://object-arbutus.cloud.computecanada.ca/ami-models/mobile/global-butterflies-subspecies-resnet50-128.zip | `dfeb5db95007b52c54c4f375dc4669995bd2225da06392c2f9f1761f456f6a91` |

Local staging dirs (with `EXPORT_MANIFEST.json`, mlpackage, category-map,
model-info, ONNX):

- `/home/michael/Projects/LepsAI/models/staging/export-20260508T234922Z/` (species)
- `/home/michael/Projects/LepsAI/models/staging/export-20260512T054653Z/` (subspecies)

**Both** share identical inference preprocessing (`src/classification/dataloader.py:18-29`
+ val transform), and **must** be evaluated with that exact pipeline — NOT the
`Resize → CenterCrop` recipe in `research/leps_localizer/scripts/validate_export.py:53-68`,
which was a hold-over from the timm default.

The required eval transform (per `src/classification/dataloader.py:18-29`):

```python
import PIL
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _pad_to_square(image: PIL.Image.Image) -> PIL.Image.Image:
    w, h = image.size
    if h < w:
        return transforms.Pad(padding=[0, 0, 0, w - h])(image)   # pad bottom
    if h > w:
        return transforms.Pad(padding=[0, 0, h - w, 0])(image)   # pad right
    return image

def eval_transform(input_size: int):
    return transforms.Compose([
        transforms.Lambda(_pad_to_square),                        # top-left anchor, black
        transforms.Resize((input_size, input_size)),              # NOT Resize+CenterCrop
        transforms.ToTensor(),                                    # /255
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
```

`fill=0` (black) on `transforms.Pad` is the default. The training data has the
original image anchored at (0,0) of a square canvas. Symmetric/center padding
would silently degrade accuracy.

## Eval set composition

### Source

iNaturalist v1 public API, no auth required.

```
GET https://api.inaturalist.org/v1/observations
  ?quality_grade=research
  &taxon_id=47222               # Papilionoidea = butterflies (NOT moths)
  &photo_license=cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nd,cc-by-nc-sa,cc-by-nc-nd
  &has[]=photos
  &order=desc
  &order_by=observed_on
  &per_page=200                 # API max
```

Paginate via `id_below=<smallest_observation_id_from_prev_page>` (the `page`
parameter is capped at 10000 results — `id_below` has no cap and is the
recommended pagination). Stop after the target N is reached.

### Filter rules

Drop any observation where:

- `taxon` is null, or `taxon.name` is not a binomial / trinomial scientific name
  (skip family/genus-only RG records — rare but they exist)
- `photos` is empty, or the first photo has no `medium`-sized URL
- `license_code` is null (unlicensed photo — don't cache or distribute)
- `observed_on` is null or in the future (defensive)

### Target N

Start at **N=1000**. The full 1000 should fit in <5 min of PyTorch inference on
a laptop CPU and ~10 min of CoreML on the macOS VM (per prior smoke-test
timing of 200–270 ms/img on QEMU). Scale to N=10000 only if N=1000 passes the
parity gates and we want tighter per-family confidence intervals.

### Bias notes (mention in `report.md`)

iNat is heavily skewed toward:

- **North America** + **Western Europe** (English-speaking iNat user base, plus
  pl@ntnet-style adoption in EU). African / SE-Asian / South American
  butterflies are *severely* underrepresented even though Mohamed's training
  set has them.
- **Day-flying species and showy taxa**. Hesperiidae (skippers) and small
  Lycaenidae are underrepresented relative to Nymphalidae and Papilionidae.
- **Recent dates → late-spring / summer hemisphere bias**. "Last 1000 RG" right
  now = mostly southern-hemisphere autumn observers. Pulling at a different
  date will shift the family mix. Worth re-running quarterly.

If a family has <30 samples in the pulled set, suppress its per-family
accuracy line in the report (too noisy) and surface the sample count.

### Storage layout

```
output/
└── inat_rg_eval_<YYYYMMDD>/
    ├── observations.ndjson          # one row per pulled iNat obs
    ├── photos/
    │   └── <inat_observation_id>.jpg
    ├── pytorch_per_sample.jsonl
    │   # one row per (obs_id, model_id) with logits hash + top-K
    ├── coreml_per_sample.jsonl
    │   # same schema as pytorch_per_sample.jsonl, populated on the mac VM
    ├── metrics.json                 # aggregate numbers
    └── report.md                    # human-readable summary
```

`<YYYYMMDD>` is the UTC date the iNat pull started. Use
`/mnt/butterflies-fg-2026-05/eval/classifier_inat_rg_<YYYYMMDD>/` on the
workspace VM if running there (faster network for iNat downloads), otherwise
`~/Projects/AMI/ami-ml/output/inat_rg_eval_<YYYYMMDD>/` locally.

## Eval protocol

### Step 1 — fetch iNat observation metadata

```
uv run python research/leps_localizer/scripts/eval_classifier_inat_rg.py fetch \
    --target-n 1000 \
    --taxon-id 47222 \
    --out-dir output/inat_rg_eval_<YYYYMMDD>/
```

For each kept observation, write one NDJSON row:

```json
{
  "id": 197345678,
  "observed_on": "2026-05-11",
  "taxon_id": 60535,
  "taxon_name": "Vanessa atalanta",
  "taxon_rank": "species",
  "ancestors": [
    {"id": 47120, "rank": "phylum", "name": "Arthropoda"},
    {"id": 47158, "rank": "class", "name": "Insecta"},
    {"id": 47157, "rank": "order", "name": "Lepidoptera"},
    {"id": 47222, "rank": "superfamily", "name": "Papilionoidea"},
    {"id": 47224, "rank": "family", "name": "Nymphalidae"},
    {"id": 51578, "rank": "subfamily", "name": "Nymphalinae"},
    {"id": 50940, "rank": "genus", "name": "Vanessa"}
  ],
  "photo_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/.../medium.jpg",
  "photo_license": "cc-by-nc",
  "place_guess": "Vermont, USA",
  "latitude": 44.26,
  "longitude": -72.58
}
```

Replace `square` in the iNat photo URL with `medium` (longest side 500px — big
enough for 512² and 128² downstream, small enough to keep total disk under
100 MB for N=1000).

Respect iNat's stated 100 req/min rate limit. With `per_page=200`, 1000 obs =
5 requests after filtering — well under the limit. Add a 0.5 s sleep between
requests anyway to be polite.

### Step 2 — download photos

```
uv run python research/leps_localizer/scripts/eval_classifier_inat_rg.py download-photos \
    --obs-ndjson output/inat_rg_eval_<YYYYMMDD>/observations.ndjson \
    --photos-dir output/inat_rg_eval_<YYYYMMDD>/photos/
```

- Skip if `<id>.jpg` already exists and is non-empty.
- HTTP 404 / 403 → write `<id>.MISSING` sentinel, skip in downstream steps,
  count in `metrics.json`.
- Save the file as `<inat_observation_id>.jpg` regardless of original
  extension — the medium endpoint always returns JPEG.

### Step 3 — PyTorch reference inference

```
# species (512²)
uv run python research/leps_localizer/scripts/eval_classifier_inat_rg.py infer-pytorch \
    --export-dir /home/michael/Projects/LepsAI/models/staging/export-20260508T234922Z/ \
    --obs-ndjson output/inat_rg_eval_<YYYYMMDD>/observations.ndjson \
    --photos-dir output/inat_rg_eval_<YYYYMMDD>/photos/ \
    --out-jsonl output/inat_rg_eval_<YYYYMMDD>/pytorch_per_sample.jsonl \
    --top-k 5

# subspecies (128²) — separate invocation
uv run python research/leps_localizer/scripts/eval_classifier_inat_rg.py infer-pytorch \
    --export-dir /home/michael/Projects/LepsAI/models/staging/export-20260512T054653Z/ \
    --obs-ndjson output/inat_rg_eval_<YYYYMMDD>/observations.ndjson \
    --photos-dir output/inat_rg_eval_<YYYYMMDD>/photos/ \
    --out-jsonl output/inat_rg_eval_<YYYYMMDD>/pytorch_per_sample.jsonl \
    --top-k 5 \
    --append
```

Per-sample JSONL row schema (one row per `(obs_id, model_id)`):

```json
{
  "obs_id": 197345678,
  "model_id": "global-butterflies-resnet50-512",
  "runtime": "pytorch",
  "input_size": 512,
  "logits_sha256": "ab12...",        // sha256 of fp32 logits bytes, for parity
  "logits_l2": 142.7,                 // L2 norm, sanity-check
  "top_k": [
    {"class_index": 5423, "label": "Vanessa atalanta", "logit": 14.2, "prob": 0.81},
    ...
  ],
  "elapsed_ms": 41.2
}
```

Implementation notes:

- Reuse `_resolve_hf` and `_load_label_list` from
  `research/leps_localizer/scripts/export_classifier.py:107-155` for the
  weights + label map. The `EXPORT_MANIFEST.json` written by the exporter
  has `weights_resolved` and `label_map_resolved` paths already cached on
  disk — load from there first, fall back to HF re-fetch.
- The PyTorch model from the manifest is `--raw-output` (no in-graph
  normalization), so the transform above is the one to feed it. This matches
  what the CoreML model will receive on-device.
- Load on GPU if available (`torch.cuda.is_available()`) and batch in groups
  of 32. Move logits to CPU/numpy before the per-sample write.
- `logits_sha256` is a cheap content hash — same fp32 bytes between PyTorch
  and CoreML → bitwise identical (won't happen, but a useful canary).
  Real parity check is on logit *values* in step 5, not the hash.

### Step 4 — CoreML inference on the macOS VM

CoreML runtime requires macOS. The mac VM is available via
`ssh macos-vm` (the user has a configured alias).

```
# from the Linux laptop, sync the inputs:
rsync -a output/inat_rg_eval_<YYYYMMDD>/ macos-vm:~/inat_rg_eval_<YYYYMMDD>/
rsync -a /home/michael/Projects/LepsAI/models/staging/export-20260508T234922Z/ macos-vm:~/models/export-species/
rsync -a /home/michael/Projects/LepsAI/models/staging/export-20260512T054653Z/ macos-vm:~/models/export-subspecies/

# run inference on the mac VM:
ssh macos-vm 'cd ~/inat_rg_eval_<YYYYMMDD> && python3 eval_classifier_inat_rg.py infer-coreml \
    --mlpackage ~/models/export-species/global-butterflies-resnet50-512.mlpackage \
    --category-map ~/models/export-species/global-butterflies-resnet50-512-category-map.json \
    --input-size 512 \
    --obs-ndjson observations.ndjson \
    --photos-dir photos/ \
    --out-jsonl coreml_per_sample.jsonl \
    --top-k 5'

# repeat for subspecies (128), --append

# pull results back:
rsync -a macos-vm:~/inat_rg_eval_<YYYYMMDD>/coreml_per_sample.jsonl \
    output/inat_rg_eval_<YYYYMMDD>/
```

The script's `infer-coreml` subcommand should be runnable standalone on
macOS (no `uv`, no `src/` imports — just `coremltools`, `Pillow`, `numpy`).
Keep it copyable as a single file. The `eval_classifier_inat_rg.py`
file might need to be self-contained enough to scp over, OR — preferred —
the script lives in the repo and the mac VM has it via the
shared-development checkout. Check `.claude/notes/workspace-vm.md` and
`.claude/skills/macos-guest/SKILL.md` for the canonical sync pattern; do
not invent a new one.

CoreML preprocessing on the VM-side must replicate the same
pad-to-square + resize + normalize pipeline as PyTorch. Use Pillow for the
pad + resize (anchored top-left, black fill), then convert to
`(1, 3, H, W)` fp32 numpy array, normalize with ImageNet mean/std, feed as
`{"input": arr}` (matches the `ct.TensorType(name="input")` from
`export_classifier.py:259`).

Read output as `out["logits"]` (1-D length-N array of logits — no softmax in
the graph for `--raw-output` models). Compute softmax + top-K in numpy.

### Step 5 — aggregate

```
uv run python research/leps_localizer/scripts/eval_classifier_inat_rg.py aggregate \
    --eval-dir output/inat_rg_eval_<YYYYMMDD>/
```

Reads `pytorch_per_sample.jsonl` + `coreml_per_sample.jsonl` + `observations.ndjson`,
joins on `obs_id`, writes `metrics.json` and `report.md`.

**Metrics to compute (per model):**

- `pytorch_top1_species_acc`: top-1 binomial match. For subspecies model, strip
  predicted trinomial to first 2 tokens (`"Heliconius melpomene martinae"` →
  `"Heliconius melpomene"`) before comparing.
- `pytorch_top5_species_acc`: any of top-5 binomials matches ground-truth.
- `pytorch_top1_subspecies_acc` *(subspecies model only)*: exact trinomial
  match. Ground-truth needs to be a trinomial — for binomial GT, skip the
  observation from this metric and record the count.
- `pytorch_top5_subspecies_acc` *(subspecies model only)*.
- `coreml_top1_species_acc`, `coreml_top5_species_acc` — same as above but
  from the CoreML output.
- `pt_coreml_top1_agreement`: % of obs where PT top-1 class_index ==
  CoreML top-1 class_index.
- `pt_coreml_logit_cosine_sim`: mean / p95 / min cosine similarity between
  the two full logit vectors (computed per-sample, then aggregated).
- `pt_coreml_max_abs_logit_diff`: max |logit_pt - logit_coreml| across all
  (obs, class) pairs. Direct analogue of `validate_export.py`'s parity gate.
- Per-family breakdown (top-1 species acc for both PT and CoreML):
  Nymphalidae, Pieridae, Lycaenidae, Papilionidae, Hesperiidae, Riodinidae.
  Pull family from the `ancestors` array (`rank == "family"`). Suppress
  families with <30 samples.

**Report format (`report.md`):**

```markdown
# iNat-RG independent classifier eval — <run_date>

Pulled 1000 RG butterfly observations from iNat between <oldest_observed_on>
and <newest_observed_on>. Geographic distribution: <top 5 countries from
place_guess>. Family distribution: Nymphalidae 412, Pieridae 198, ...

## Summary

| Model | PT top-1 (sp) | PT top-5 (sp) | CoreML top-1 (sp) | CoreML top-5 (sp) | PT↔CoreML top-1 agree | mean cos-sim | max |Δlogit| |
|---|---|---|---|---|---|---|---|
| global-butterflies-resnet50-512 | 0.71 | 0.89 | 0.71 | 0.89 | 99.8% | 0.99998 | 7.3e-04 |
| global-butterflies-subspecies-resnet50-128 | 0.43 | 0.69 | ... | ... | ... | ... | ... |

## Per-family top-1 species accuracy

| Family | n | PT (sp 512) | PT (sub 128 → bi) | CoreML (sp 512) | CoreML (sub 128 → bi) |
|---|---|---|---|---|---|
| Nymphalidae | 412 | 0.78 | 0.55 | 0.78 | 0.55 |
| Pieridae | 198 | 0.69 | 0.41 | ... | ... |
| ...

## Subspecies model — trinomial accuracy

Only `<n_with_trinomial_gt>` of <N> observations have subspecies-level
ground-truth in iNat. Top-1 subspecies accuracy on that subset: <x>.
(Note: low denominator — most iNat RG IDs stop at species rank.)

## Parity diagnostics

PyTorch ↔ CoreML top-1 agreement: <x>% (gate: ≥99%)
Max |logit_pt - logit_coreml|: <y> (gate: <1e-2)
Mean cosine similarity: <z>

<MISMATCHES table — first 10 obs where PT and CoreML disagree on top-1,
with both predictions and the ground-truth>

## Caveats

- iNat geographic bias: <N from NA / EU / rest>
- Subspecies model: most iNat RG IDs are species-level, so the
  trinomial-accuracy denominator is small.
- Photo quality varies: iNat thumbnails can be 500px on the long side
  after the `medium` rewrite — the species model expects 512×512 and may
  lose detail on the short side after pad-to-square. Re-run with the
  `large` (1024px) iNat photos if accuracy is suspiciously low.
```

## Script layout

```
research/leps_localizer/scripts/eval_classifier_inat_rg.py
```

Subcommands (use `argparse` subparsers, no Click):

- `fetch` — pull iNat metadata → `observations.ndjson`
- `download-photos` — download iNat medium photos → `photos/<id>.jpg`
- `infer-pytorch` — PyTorch reference inference → `pytorch_per_sample.jsonl`
- `infer-coreml` — CoreML inference on mac VM → `coreml_per_sample.jsonl`
  (standalone-runnable: no `src/` imports, no `uv`)
- `aggregate` — join + metrics + report → `metrics.json`, `report.md`

Reuse from `research/leps_localizer/scripts/export_classifier.py`:

- `_resolve_hf` (`:107-132`) — fetch HF artifacts to local cache
- `_load_label_list` (`:141-155`) — parse the `label_map.json` schema variants
- Manifest reader pattern from `validate_export.py:38-42`

If the `_resolve_hf` + `_load_label_list` helpers feel valuable as shared
utility, promote them to `src/classification/io.py` — but **defer that
decision to inline judgment**; don't refactor pre-emptively. Inline copies
are fine for this script.

The CoreML subcommand needs to be runnable without the ami-ml repo's
src tree (mac VM may have an older checkout or none). Either:

1. Make `infer-coreml` only import stdlib + `coremltools` + `numpy` +
   `Pillow`, and have it parse `EXPORT_MANIFEST.json` standalone, OR
2. Sync the script file alongside the eval dir each run.

Option 1 is preferred — keeps the script copy-paste portable.

## Validation gates

A run "passes" if all of these hold for **both** models:

1. **PT ↔ CoreML top-1 class_index agreement ≥ 99%** across the full 1000
   observations. Below 95% = conversion bug, investigate before trusting
   any iOS-side claim of the model working.
2. **Max |logit_pt - logit_coreml| < 1e-2** across all (obs, class) pairs.
   Looser than the 1e-3 gate in `validate_export.py:114` because real iNat
   images have noise (JPEG decode + resize jitter) — the dummy zero-tensor
   parity test in that script tolerates 1e-3 because it has no noise. Use
   `validate_export.py`'s gate for synthetic inputs; this one for real
   images.
3. **Mean PT↔CoreML cosine similarity > 0.9999** on the logit vectors.
4. Per-family accuracy is **reported, not gated** — this is descriptive
   (where does each model fail?), not pass/fail.

If gate (1) or (2) fails, do NOT claim the iOS-side conversion is good. The
likely failure modes:

- CoreML preprocessing on the mac VM script doesn't match PT exactly (e.g.
  PIL vs torchvision resize interpolation differs — use bilinear in both)
- The `ct.TensorType(name="input")` graph was traced with the wrong
  example tensor shape
- FP16 quantization on the CoreML compute units silently degraded — re-try
  with `compute_units=ct.ComputeUnit.CPU_ONLY` to isolate

## Cost-aware notes

| Step | Compute | Wall time | Disk / network |
|---|---|---|---|
| `fetch` | 5 iNat API requests | ~5 s + 0.5 s/req throttle | ~2 MB JSON |
| `download-photos` | 1000 HTTPS downloads | ~3-5 min (parallelize 16-way) | ~100 MB (medium @ 500px) |
| `infer-pytorch` (sp 512) | CPU: ~5 min, GPU (3090): ~30 s | | logits 8851 × fp32 × 1000 = 35 MB |
| `infer-pytorch` (sub 128) | CPU: ~1 min, GPU: ~5 s | | logits 19411 × fp32 × 1000 = 78 MB |
| `infer-coreml` (sp 512) | mac VM QEMU: ~10 min (200–270 ms/img) | | (same logit volume) |
| `infer-coreml` (sub 128) | mac VM QEMU: ~3-5 min | | |
| `aggregate` | ~10 s | | metrics.json + report.md = few KB |

Total disk for a 1000-obs run: ~250 MB (mostly logits). At N=10000 this is
2.5 GB — fine on the workspace VM, watch the laptop disk. Per-sample JSONL
files compress well (`gzip` → ~10× shrink) since logit floats are highly
compressible. Consider gzipping the JSONLs in `aggregate` if running at
N=10k.

iNat API has no auth requirement at this volume but check the request
volume against https://api.inaturalist.org/v1/docs/#!/Observations/get_observations
the first time the script runs — they have asked third parties to keep
below 100 req/min, and we are well under that with `per_page=200`.

## Followups (defer; do not do in this session)

- **Confidence calibration**. Per `~/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/project_confidence_calibration.md`,
  the predicted probabilities aren't calibrated — a "0.9 confidence" prediction
  from the subspecies model is probably not actually 90% accurate. Reuse the
  per-sample JSONL from this eval as input to a temperature-scaling pass once
  the basic accuracy numbers are in.
- **Scale to N=10000**. Only if N=1000 passes the parity gates. Useful for
  tighter per-family confidence intervals and discovering long-tail families
  (Riodinidae, Hedylidae) that have <30 samples at N=1000.
- **eButterfly observations as a second independent set**. eB has a different
  geographic skew (heavy on eastern North America, more contributor-vetted),
  which would surface different failure modes. Pull from the eB API or BigQuery
  `leps-ai.global_butterflies_2604.training_images` filtered to source=eb +
  date > training_cutoff (where training_cutoff is per `EXPORT_MANIFEST.json`
  → `weights_resolved` mtime, roughly).
- **CoreML ↔ ONNX parity**. ONNX runs on the Linux side and was already
  validated against PyTorch zero-tensor in `validate_export.py`. Adding ONNX
  to the per-sample JSONL here would give a transitive check
  (PT ↔ ONNX ↔ CoreML) and isolate where any drift comes from. Worth doing
  if the CoreML gate fails and the cause is unclear.
- **Hard-case curation**. Once we have per-sample JSONL with the mismatches,
  curate the worst 20 confidently-wrong predictions into a small visual
  report. Per memory `feedback_danaus_not_useful_test.md`, monarchs always
  classify right — the signal lives in the lycaenids / fritillaries /
  skippers, and curating that subset gives sharper feedback to Mohamed than
  an aggregate number.

## Key references

- **Preprocessing source of truth:** `src/classification/dataloader.py:18-29` (pad_to_square) + val transform in the same file
- **Export script (helper reuse):** `research/leps_localizer/scripts/export_classifier.py:107-155` (`_resolve_hf`, `_load_label_list`)
- **CoreML graph contract (input name, output name, raw_output flag):** `research/leps_localizer/scripts/export_classifier.py:240-312`
- **Prior parity check (replace with this, on real images):** `research/leps_localizer/scripts/validate_export.py`
- **Bundle contract memory:** `~/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/reference_lepsai_ios_classifier_bundle.md`
- **Danaus signal warning:** `~/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/feedback_danaus_not_useful_test.md`
- **Confidence-calibration backlog:** `~/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/project_confidence_calibration.md`
- **Mohamed training-size lookup:** `~/.claude/projects/-home-michael-Projects-AMI-ami-ml/memory/feedback_mohamed_models_check_training_size.md`
- **Sibling iOS-side prompt (same models, different angle):** `~/Projects/LepsAI/docs/claude/prompts/global-butterflies-subspecies-128-integration.md`
- **macOS VM access pattern:** `.claude/skills/macos-guest/SKILL.md` and `.claude/notes/workspace-vm.md`
- **iNat API docs:** https://api.inaturalist.org/v1/docs/#!/Observations/get_observations
- **Local model staging dirs:** `/home/michael/Projects/LepsAI/models/staging/export-20260508T234922Z/` (species), `/home/michael/Projects/LepsAI/models/staging/export-20260512T054653Z/` (subspecies)
