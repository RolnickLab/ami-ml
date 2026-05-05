# Lepidoptera Localizer Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a fast butterfly bbox detector on Fieldguide data. First model = YOLOv11s on the Arbutus 2026 `ami-workspace-02-gpu` H100 box. Subsequent models = RT-DETRv2 + torchvision Faster R-CNN, compared on the same locked eval set.

**Architecture:**
- Two repos. **Extract pipeline** in the `chroma-backend` worktree (Fieldguide DB lives there) extends `detector_dataset/` with a new `assemble-trainset` stage that emits `index.jsonl` + uploads to `s3://ami-trainingdata/ai-for-leps/localization/<name>/` on Arbutus 2026.
- **Training pipeline** in `ami-ml` (branch `feat/leps-localizer-training`) adds Ultralytics + RT-DETR deps, an index→torch dataset adapter, thin trainer wrappers, and an `eval_on_locked.py` that runs against the 4 frozen eval indexes.
- Training runs on `ami-workspace-02-gpu` (H100 24GB), reading from a FUSE mount of `s3://ami-trainingdata/`.

**Tech Stack:** Python 3.10+, uv, Ultralytics 8.x (YOLOv11), `transformers` for RT-DETR, torchvision detection (existing `src/localization/training.py`), `wandb`, `psycopg` 3, `boto3` / `s5cmd`, `mountpoint-s3` (FUSE), `pytest`.

---

## Spec reference

Source spec: `~/Projects/AMI/ami-ml/research/leps_localizer/DESIGN.md` on branch `feat/leps-localizer-training` (commits `31abd75`, `690243d`, `63bb7be`, `d3f3571`).

Locked eval datasets (do NOT train on):
- `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/datasets/arthropoda/leps-butterflies-500/`
- `…/leps-butterflies-small-1000/`
- `…/leps-butterflies-medsmall-1000/`
- `…/leeds-butterflies/`

Total ~3,800 photos. Their `photo_id`s must be excluded from any training pull.

---

## File map

### chroma-backend worktree (extract side)

| Path | Action | Responsibility |
|---|---|---|
| `detector_dataset/src/detector_dataset/index_jsonl.py` | create | `IndexRecord` dataclass + writer; canonical schema |
| `detector_dataset/src/detector_dataset/exclude.py` | create | Read `photo_id`s from a list of COCO files (or JSONL files) |
| `detector_dataset/src/detector_dataset/upload.py` | create | Upload images + index to S3 (new arbutus 2026 endpoint) using `s5cmd` |
| `detector_dataset/src/detector_dataset/assemble_trainset.py` | create | Orchestrator: extract → fetch → write index.jsonl → upload |
| `detector_dataset/src/detector_dataset/configs.py` | modify | Add `butterflies-fg-2026-05` Spec; add `exclude_photo_ids_from` field on Spec; add `s3_destination` field |
| `detector_dataset/src/detector_dataset/cli.py` | modify | Add `assemble-trainset` and `backfill-eval-jsonl` subcommands |
| `detector_dataset/tests/test_index_jsonl.py` | create | Round-trip JSONL ↔ records, schema enforcement |
| `detector_dataset/tests/test_exclude.py` | create | Excludes work for both COCO and JSONL inputs |

### ami-ml repo (training side)

| Path | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `[project.optional-dependencies] detection = [ultralytics, transformers, pyarrow, ...]` |
| `src/localization/leps_data.py` | create | `LepsLocalizerDataset` class reads `index.jsonl`/parquet, yields `(image, target)` tuples for torchvision; YOLO export helper |
| `src/localization/yolo_train.py` | create | Click CLI: train YOLOv11s/v8s on a given index URL |
| `src/localization/rtdetr_train.py` | create | Click CLI: train RT-DETRv2 on a given index URL |
| `src/localization/eval_on_locked.py` | create | Click CLI: load checkpoint, run against the 4 eval indexes, write JSONL of preds + a markdown report |
| `src/localization/metrics.py` | modify (extend) | Add `containment_iou`, `best_of_n_match`, `recall_by_bucket` helpers |
| `tests/localization/test_leps_data.py` | create | Index loader correctness, split filtering |
| `tests/localization/test_metrics.py` | create | Containment, best-of-n, area-frac bucketing |
| `research/leps_localizer/configs/yolo11s.yaml` | create | Ultralytics data config pointing at the FUSE mount |
| `research/leps_localizer/configs/rtdetr.yaml` | create | RT-DETR config |
| `research/leps_localizer/scripts/train_yolo.sh` | create | Shell wrapper for VM job |
| `research/leps_localizer/scripts/train_rtdetr.sh` | create | Shell wrapper for VM job |
| `research/leps_localizer/scripts/train_frcnn.sh` | create | Shell wrapper for VM job |
| `research/leps_localizer/scripts/eval_locked.sh` | create | Run eval after each training |
| `research/leps_localizer/scripts/setup_workspace_vm.sh` | create | One-shot VM bootstrap (uv install, repo clone, FUSE mount, wandb login, sanity-check) |
| `research/leps_localizer/reports/` | create dir | Empty; reports land here per run |

### ami-devops repo (infra glue)

| Path | Action | Responsibility |
|---|---|---|
| `systemd/storage/ami-trainingdata-newcloud-s3.service` | create | mountpoint-s3 unit pointing at `object-arbutus.alliancecan.ca`, mounting at `/mnt/s3-trainingdata` |

---

## Phase 1 — Extract pipeline (chroma-backend, beast or laptop)

Working directory for tasks 1–7: `~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset/`

Branch: stay on `worktree-detector-training`.

### Task 1: `IndexRecord` dataclass + JSONL writer

**Files:**
- Create: `src/detector_dataset/index_jsonl.py`
- Test: `tests/test_index_jsonl.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_jsonl.py
import json
from pathlib import Path
from detector_dataset.index_jsonl import IndexRecord, write_index, read_index

def test_round_trip(tmp_path: Path):
    rec = IndexRecord(
        dataset_name="butterflies-fg-2026-05",
        dataset_version="v1.0.0",
        data_source="fieldguide-prod",
        extracted_at="2026-05-05T12:00:00Z",
        extract_commit_sha="abc123",
        image_key="images/ab/abcdef.jpg",
        sha256="abcdef",
        photo_id="60a1f2",
        category_id="552e76f291201b5ddbcbf77b",
        parents=["a", "b", "c"],
        user_id="u1",
        width=4032,
        height=3024,
        bbox_xyxy=[100.0, 200.0, 500.0, 600.0],
        bbox_area_fraction=0.073,
        bbox_is_square=True,
        created_at="2024-09-01T00:00:00Z",
        split="train",
    )
    out = tmp_path / "index.jsonl"
    write_index([rec], out)
    rows = list(read_index(out))
    assert rows == [rec]
    # JSON shape sanity
    line = out.read_text().splitlines()[0]
    parsed = json.loads(line)
    assert parsed["bbox_area_fraction"] == 0.073
    assert parsed["split"] == "train"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_index_jsonl.py::test_round_trip -v
```
Expected: `ModuleNotFoundError: detector_dataset.index_jsonl`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/detector_dataset/index_jsonl.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class IndexRecord:
    dataset_name: str
    dataset_version: str
    data_source: str
    extracted_at: str
    extract_commit_sha: str
    image_key: str
    sha256: str
    photo_id: str
    category_id: str
    parents: list[str]
    user_id: str
    width: int
    height: int
    bbox_xyxy: list[float]
    bbox_area_fraction: float
    bbox_is_square: bool
    created_at: str
    split: str


def write_index(records: Iterable[IndexRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")


def read_index(path: Path) -> Iterator[IndexRecord]:
    field_names = {f.name for f in fields(IndexRecord)}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            extra = data.keys() - field_names
            if extra:
                raise ValueError(f"unknown fields in index.jsonl: {sorted(extra)}")
            yield IndexRecord(**data)
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_index_jsonl.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/detector_dataset/index_jsonl.py tests/test_index_jsonl.py
git commit -m "feat(detector-dataset): canonical IndexRecord JSONL reader/writer"
```

### Task 2: Photo-ID exclusion loader (COCO + JSONL inputs)

**Files:**
- Create: `src/detector_dataset/exclude.py`
- Test: `tests/test_exclude.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exclude.py
import json
from pathlib import Path
from detector_dataset.exclude import load_excluded_photo_ids


def test_load_from_coco(tmp_path: Path):
    coco = {
        "images": [
            {"id": 1, "file_name": "x.jpg", "fg_photo_id": "p1"},
            {"id": 2, "file_name": "y.jpg", "fg_photo_id": "p2"},
            {"id": 3, "file_name": "z.jpg"},  # leeds row, no fg_photo_id
        ],
        "annotations": [],
        "categories": [],
    }
    p = tmp_path / "coco.json"
    p.write_text(json.dumps(coco))
    assert load_excluded_photo_ids([p]) == {"p1", "p2"}


def test_load_from_jsonl(tmp_path: Path):
    p = tmp_path / "index.jsonl"
    p.write_text(
        json.dumps({"photo_id": "p3", "split": "val"}) + "\n"
        + json.dumps({"photo_id": "p4", "split": "train"}) + "\n"
    )
    assert load_excluded_photo_ids([p]) == {"p3", "p4"}


def test_load_mixed(tmp_path: Path):
    coco = {"images": [{"fg_photo_id": "p1"}], "annotations": [], "categories": []}
    pc = tmp_path / "a.json"
    pc.write_text(json.dumps(coco))
    pj = tmp_path / "b.jsonl"
    pj.write_text(json.dumps({"photo_id": "p2"}) + "\n")
    assert load_excluded_photo_ids([pc, pj]) == {"p1", "p2"}
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_exclude.py -v
```
Expected: `ModuleNotFoundError: detector_dataset.exclude`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/detector_dataset/exclude.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable


def load_excluded_photo_ids(paths: Iterable[Path]) -> set[str]:
    """Return the set of FG photo_ids to exclude.

    Accepts either COCO JSON files (looks at images[*].fg_photo_id) or
    newline-delimited JSON files (looks at each line's photo_id).
    Files without either field are skipped silently.
    """
    excluded: set[str] = set()
    for path in paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    pid = json.loads(line).get("photo_id")
                    if pid:
                        excluded.add(pid)
        else:  # treat as COCO
            data = json.loads(path.read_text())
            for img in data.get("images", []):
                pid = img.get("fg_photo_id")
                if pid:
                    excluded.add(pid)
    return excluded
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_exclude.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add src/detector_dataset/exclude.py tests/test_exclude.py
git commit -m "feat(detector-dataset): load excluded photo IDs from COCO and JSONL"
```

### Task 3: Spec gains exclude + S3 destination fields

**Files:**
- Modify: `src/detector_dataset/configs.py`

- [ ] **Step 1: Add fields to `Spec` dataclass**

Add to the existing `Spec` dataclass (keep current fields, append):

```python
    # New optional fields
    exclude_photo_ids_from: list[str] = field(default_factory=list)
    s3_destination: str | None = None  # e.g. "s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05"
```

`field` import: `from dataclasses import dataclass, field`.

- [ ] **Step 2: Add `butterflies-fg-2026-05` Spec**

In the `CONFIGS` dict, add:

```python
"butterflies-fg-2026-05": Spec(
    name="butterflies-fg-2026-05",
    target_count=20_000,
    hard_cap=80_000,
    clade_id="552e76f291201b5ddbcbf77b",
    clade_name="Butterflies",
    min_area_fraction=0.0,
    max_area_fraction=0.95,
    per_species_user_cap_start=1,
    seed="2026-05-05",
    exclude_photo_ids_from=[
        "../detector_dataset/datasets/arthropoda/leps-butterflies-500/coco.json",
        "../detector_dataset/datasets/arthropoda/leps-butterflies-small-1000/coco.json",
        "../detector_dataset/datasets/arthropoda/leps-butterflies-medsmall-1000/coco.json",
        "../detector_dataset/datasets/leeds-butterflies/coco.json",
    ],
    s3_destination="s3://ami-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05",
),
```

(Adjust paths if `CONFIGS` lives at a different cwd. Cross-check with current `Spec` field names — keep existing names exact.)

- [ ] **Step 3: Run all existing tests to confirm no regression**

```
uv run pytest -v
```
Expected: existing tests pass; the new fields default to empty so they don't disturb existing configs.

- [ ] **Step 4: Commit**

```
git add src/detector_dataset/configs.py
git commit -m "feat(detector-dataset): butterflies-fg-2026-05 config with eval-set exclusion"
```

### Task 4: `assemble_trainset` orchestrator

**Files:**
- Create: `src/detector_dataset/assemble_trainset.py`

This composes existing extract + fetch + adds index.jsonl emission. Tests live in Task 5 (CLI test).

- [ ] **Step 1: Implement orchestrator**

```python
# src/detector_dataset/assemble_trainset.py
from __future__ import annotations
import datetime as dt
import hashlib
import random
import subprocess
from pathlib import Path
from typing import Iterable

from .configs import Spec
from .exclude import load_excluded_photo_ids
from .extract import process_rows, fetch_candidate_rows  # existing
from .fetch import fetch_one_image  # existing
from .index_jsonl import IndexRecord, write_index


def _git_sha(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _split_assignment(photo_id: str, val_fraction: float, seed: str) -> str:
    """Deterministic per-photo split based on photo_id hash."""
    h = hashlib.md5(f"{seed}:{photo_id}".encode()).hexdigest()
    bucket = int(h, 16) / 16**32  # uniform [0,1)
    return "val" if bucket < val_fraction else "train"


def assemble(
    spec: Spec,
    output_dir: Path,
    val_fraction: float = 0.1,
    repo_root: Path | None = None,
) -> Path:
    """Run the full extract → fetch → index pipeline. Returns path to index.jsonl."""
    repo_root = repo_root or Path(__file__).resolve().parents[3]

    excluded = load_excluded_photo_ids(spec.exclude_photo_ids_from)
    print(f"[assemble] excluding {len(excluded)} photo_ids from eval set")

    rows = list(fetch_candidate_rows(spec))
    print(f"[assemble] fetched {len(rows)} candidate rows")
    rows = [r for r in rows if r["photo_id"] not in excluded]
    print(f"[assemble] {len(rows)} after exclusion")

    sampled = process_rows(rows, spec)  # existing per-species/user cap logic
    print(f"[assemble] {len(sampled)} after sampling")

    extracted_at = dt.datetime.now(dt.UTC).isoformat()
    sha = _git_sha(repo_root)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[IndexRecord] = []
    for row in sampled:
        # fetch_one_image returns local path + sha256
        path, sha256 = fetch_one_image(row, images_dir)
        if path is None:
            continue
        bbox = row["bbox_xyxy"]
        w, h = row["width"], row["height"]
        area_frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        records.append(
            IndexRecord(
                dataset_name=spec.name,
                dataset_version="v1.0.0",
                data_source="fieldguide-prod",
                extracted_at=extracted_at,
                extract_commit_sha=sha,
                image_key=f"images/{sha256[:2]}/{sha256}.jpg",
                sha256=sha256,
                photo_id=row["photo_id"],
                category_id=row["category_id"],
                parents=row["parents"],
                user_id=row["user_id"],
                width=w,
                height=h,
                bbox_xyxy=list(map(float, bbox)),
                bbox_area_fraction=float(area_frac),
                bbox_is_square=abs(bw - bh) < 2,
                created_at=row["created_at"].isoformat() if row.get("created_at") else "",
                split=_split_assignment(row["photo_id"], val_fraction, spec.seed),
            )
        )

    index_path = output_dir / "index.jsonl"
    write_index(records, index_path)

    # write manifests
    train = [r.image_key for r in records if r.split == "train"]
    val = [r.image_key for r in records if r.split == "val"]
    (output_dir / "manifest_train.txt").write_text("\n".join(train) + "\n")
    (output_dir / "manifest_val.txt").write_text("\n".join(val) + "\n")
    print(f"[assemble] wrote {index_path} with {len(records)} records")
    return index_path
```

(If existing `fetch_one_image` signature differs, adapt the call. The orchestrator's job is only to glue and emit JSONL; existing extract/fetch logic is unchanged.)

- [ ] **Step 2: Smoke-run on 10 images locally**

```
uv run python -c "
from pathlib import Path
from detector_dataset.configs import CONFIGS
from detector_dataset.assemble_trainset import assemble
spec = CONFIGS['butterflies-fg-2026-05']
spec = type(spec)(**{**spec.__dict__, 'target_count': 10, 'hard_cap': 50})
assemble(spec, Path('/tmp/leps-smoke'))
"
```

Expected: prints exclusion count, candidate count, sampled count, and writes 10-line `/tmp/leps-smoke/index.jsonl` plus train/val manifests.

- [ ] **Step 3: Spot-check the JSONL**

```
head -1 /tmp/leps-smoke/index.jsonl | python -m json.tool
```

Expected: a single record with all 18 schema fields populated, including `bbox_area_fraction` and `bbox_is_square`.

- [ ] **Step 4: Commit**

```
git add src/detector_dataset/assemble_trainset.py
git commit -m "feat(detector-dataset): assemble-trainset orchestrator emitting index.jsonl"
```

### Task 5: CLI subcommands `assemble-trainset` and `backfill-eval-jsonl`

**Files:**
- Modify: `src/detector_dataset/cli.py`
- Create: `src/detector_dataset/backfill_eval.py`

- [ ] **Step 1: Implement `backfill_eval.py`**

```python
# src/detector_dataset/backfill_eval.py
"""Re-emit the 4 locked eval datasets as index.jsonl alongside their COCO files."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path

from .index_jsonl import IndexRecord, write_index


def backfill_one_coco(coco_path: Path, dataset_name: str, data_source: str, sha: str) -> Path:
    data = json.loads(coco_path.read_text())
    images = {img["id"]: img for img in data["images"]}
    annotations = {a["image_id"]: a for a in data["annotations"]}  # 1 ann per image (single-class)
    extracted_at = dt.datetime.now(dt.UTC).isoformat()
    records: list[IndexRecord] = []
    for img_id, img in images.items():
        ann = annotations.get(img_id)
        if not ann:
            continue
        x, y, bw, bh = ann["bbox"]
        bbox_xyxy = [float(x), float(y), float(x + bw), float(y + bh)]
        w, h = img["width"], img["height"]
        area_frac = (bw * bh) / (w * h)
        records.append(
            IndexRecord(
                dataset_name=dataset_name,
                dataset_version="v1.0.0",
                data_source=data_source,
                extracted_at=extracted_at,
                extract_commit_sha=sha,
                image_key=img["file_name"],
                sha256=img.get("fg_sha256", img.get("sha256", "")),
                photo_id=img.get("fg_photo_id", ""),
                category_id=img.get("fg_category_id", ""),
                parents=img.get("fg_parents", []),
                user_id=img.get("fg_user_id", ""),
                width=w,
                height=h,
                bbox_xyxy=bbox_xyxy,
                bbox_area_fraction=float(area_frac),
                bbox_is_square=abs(bw - bh) < 2,
                created_at=img.get("fg_created_at", ""),
                split="eval",
            )
        )
    out = coco_path.with_name("index.jsonl")
    write_index(records, out)
    return out
```

- [ ] **Step 2: Wire CLI subcommands**

Add to `cli.py`:

```python
@cli.command("assemble-trainset")
@click.argument("config_name")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def cli_assemble_trainset(config_name: str, output_dir: Path):
    from .configs import CONFIGS
    from .assemble_trainset import assemble
    spec = CONFIGS[config_name]
    assemble(spec, output_dir)


@cli.command("backfill-eval-jsonl")
@click.argument("coco_path", type=click.Path(exists=True, path_type=Path))
@click.option("--dataset-name", required=True)
@click.option("--data-source", required=True, type=click.Choice(["fieldguide-prod", "leeds"]))
def cli_backfill_eval(coco_path: Path, dataset_name: str, data_source: str):
    import subprocess
    from .backfill_eval import backfill_one_coco
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    out = backfill_one_coco(coco_path, dataset_name, data_source, sha)
    click.echo(f"wrote {out}")
```

- [ ] **Step 3: Backfill all 4 eval datasets**

```
cd ~/Projects/Fieldguide/chroma-backend/.claude/worktrees/detector-training/detector_dataset
uv run detector-dataset backfill-eval-jsonl datasets/arthropoda/leps-butterflies-500/coco.json --dataset-name leps-butterflies-500 --data-source fieldguide-prod
uv run detector-dataset backfill-eval-jsonl datasets/arthropoda/leps-butterflies-small-1000/coco.json --dataset-name leps-butterflies-small-1000 --data-source fieldguide-prod
uv run detector-dataset backfill-eval-jsonl datasets/arthropoda/leps-butterflies-medsmall-1000/coco.json --dataset-name leps-butterflies-medsmall-1000 --data-source fieldguide-prod
uv run detector-dataset backfill-eval-jsonl datasets/leeds-butterflies/coco.json --dataset-name leeds-butterflies --data-source leeds
```

Expected: 4 new `index.jsonl` files alongside their `coco.json` siblings.

- [ ] **Step 4: Spot-check one**

```
head -1 datasets/arthropoda/leps-butterflies-500/index.jsonl | python -m json.tool
```

Expected: full schema, `split == "eval"`, `bbox_area_fraction` populated.

- [ ] **Step 5: Commit**

```
git add src/detector_dataset/cli.py src/detector_dataset/backfill_eval.py datasets/arthropoda/leps-butterflies-*/index.jsonl datasets/leeds-butterflies/index.jsonl
git commit -m "feat(detector-dataset): backfill index.jsonl for the 4 locked eval datasets"
```

### Task 6: S3 upload helper

**Files:**
- Create: `src/detector_dataset/upload.py`

- [ ] **Step 1: Implement uploader**

```python
# src/detector_dataset/upload.py
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

NEW_ARBUTUS_ENDPOINT = "https://object-arbutus.alliancecan.ca"


def upload_dataset(local_dir: Path, s3_destination: str, profile: str = "ami-newcloud") -> None:
    """Upload a local dataset directory to s3://... using s5cmd.

    Required env: ~/.aws/credentials must have a profile named `profile` with
    keys for the new arbutus endpoint. Set AWS_ENDPOINT_URL=NEW_ARBUTUS_ENDPOINT
    before running, or rely on the profile's endpoint_url.
    """
    if not shutil.which("s5cmd"):
        raise RuntimeError("s5cmd not found in PATH")
    env = os.environ | {
        "AWS_PROFILE": profile,
        "AWS_ENDPOINT_URL": NEW_ARBUTUS_ENDPOINT,
        "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
    }
    cmd = ["s5cmd", "--endpoint-url", NEW_ARBUTUS_ENDPOINT, "sync", f"{local_dir}/", s3_destination + "/"]
    subprocess.run(cmd, env=env, check=True)
```

(If the new-cloud bucket isn't yet created, surface that as an error from the CLI in the next task. Bucket creation is a one-time openstack/openrc operation handled in the VM bootstrap script.)

- [ ] **Step 2: Wire into the orchestrator**

Modify `assemble_trainset.assemble` to call `upload_dataset(output_dir, spec.s3_destination)` if `spec.s3_destination` is set.

- [ ] **Step 3: Commit**

```
git add src/detector_dataset/upload.py src/detector_dataset/assemble_trainset.py
git commit -m "feat(detector-dataset): s5cmd-based upload to new arbutus 2026 S3"
```

### Task 7: Pull `butterflies-fg-2026-05` (20k)

- [ ] **Step 1: Smoke at target=200**

```
uv run detector-dataset assemble-trainset butterflies-fg-2026-05 --output-dir /tmp/leps-200 \
  -- --target-count 200
```

Adjust CLI to accept `--target-count` override if it doesn't already, otherwise edit the spec inline. Expected: 200 images extracted, exclusion count printed, index.jsonl written.

- [ ] **Step 2: Full pull (target=20000)**

```
uv run detector-dataset assemble-trainset butterflies-fg-2026-05 \
  --output-dir ~/datasets/butterflies-fg-2026-05
```

Expected: ~20k images on local disk, ~20k records in `index.jsonl`, manifests written. Runtime: hours, depending on network. Use `nohup` if launching from staging.

- [ ] **Step 3: Verify exclusion**

```
python - <<'PY'
import json
excluded = set()
for p in [
    "datasets/arthropoda/leps-butterflies-500/index.jsonl",
    "datasets/arthropoda/leps-butterflies-small-1000/index.jsonl",
    "datasets/arthropoda/leps-butterflies-medsmall-1000/index.jsonl",
    "datasets/leeds-butterflies/index.jsonl",
]:
    excluded |= {json.loads(l)["photo_id"] for l in open(p) if l.strip()}
got = {json.loads(l)["photo_id"] for l in open("/home/michael/datasets/butterflies-fg-2026-05/index.jsonl")}
print("eval ∩ train =", len(excluded & got))
PY
```

Expected: `eval ∩ train = 0`.

- [ ] **Step 4: Upload to S3**

```
uv run detector-dataset assemble-trainset butterflies-fg-2026-05 --output-dir ~/datasets/butterflies-fg-2026-05 --upload
```

(If the orchestrator already uploads, skip this step. Otherwise add an `--upload` flag in Task 6 and re-run.)

- [ ] **Step 5: Commit any config tweaks**

```
git add -p
git commit -m "chore(detector-dataset): butterflies-fg-2026-05 spec final tuning post-pull"
```

---

## Phase 2 — VM bootstrap

Working directory for tasks 8–9: `~/Projects/AMI/ami-devops` and `~/Projects/AMI/ami-ml`.

### Task 8: New-cloud S3 mount on `ami-workspace-02-gpu`

**Files:**
- Create: `~/Projects/AMI/ami-devops/systemd/storage/ami-trainingdata-newcloud-s3.service`

- [ ] **Step 1: Write the systemd unit**

```ini
# systemd/storage/ami-trainingdata-newcloud-s3.service
[Unit]
Description=mountpoint-s3 for ami-trainingdata bucket on new arbutus 2026 endpoint
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
User=debian
Group=debian
Environment=AWS_PROFILE=ami-newcloud
Environment=AWS_ENDPOINT_URL=https://object-arbutus.alliancecan.ca
Environment=AWS_REQUEST_CHECKSUM_CALCULATION=when_required
Environment=AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
ExecStartPre=/bin/mkdir -p /mnt/s3-trainingdata
ExecStart=/usr/bin/mount-s3 ami-trainingdata /mnt/s3-trainingdata \
  --endpoint-url https://object-arbutus.alliancecan.ca \
  --force-path-style \
  --allow-other --allow-delete \
  --max-threads 32 --part-size 16777216
ExecStop=/bin/fusermount -u /mnt/s3-trainingdata
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: SSH to box and install**

```
ssh ami-workspace-02-gpu

# on the box:
cd ~/ami-devops && git pull
sudo ln -sf ~/ami-devops/systemd/storage/ami-trainingdata-newcloud-s3.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ami-trainingdata-newcloud-s3.service
ls /mnt/s3-trainingdata/
```

Expected: `ai-for-leps/` listed (or the bucket is empty, in which case create the prefix via `aws s3api`).

- [ ] **Step 3: Commit unit file**

```
cd ~/Projects/AMI/ami-devops
git add systemd/storage/ami-trainingdata-newcloud-s3.service
git commit -m "feat(storage): mount-s3 unit for new arbutus 2026 ami-trainingdata bucket"
```

### Task 9: VM bootstrap script

**Files:**
- Create: `research/leps_localizer/scripts/setup_workspace_vm.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# research/leps_localizer/scripts/setup_workspace_vm.sh
# Run once on ami-workspace-02-gpu after FUSE mount is up.
set -euo pipefail

# 1. uv
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.cargo/env || source ~/.local/bin/env || true
fi

# 2. Clone ami-ml
if [ ! -d ~/ami-ml ]; then
  GIT_AUTHOR_NAME="$(whoami) on workspace-02-gpu" \
  GIT_COMMITTER_NAME="$(whoami) on workspace-02-gpu" \
    git clone git@github.com:RolnickLab/ami-ml.git ~/ami-ml
fi
cd ~/ami-ml
git fetch origin
git checkout feat/leps-localizer-training
git pull

# 3. Install (with detection extra)
uv sync --extra dev --extra research --extra detection

# 4. wandb
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY not set; run 'uv run wandb login' interactively"
fi

# 5. Sanity checks
nvidia-smi | head -10
ls /mnt/s3-trainingdata/ai-for-leps/localization/
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
uv run python -c "from ultralytics import YOLO; print('ultralytics ok')"
```

- [ ] **Step 2: Run from local laptop**

```
cd ~/Projects/AMI/ami-ml
git push origin feat/leps-localizer-training
ssh ami-workspace-02-gpu 'bash -s' < research/leps_localizer/scripts/setup_workspace_vm.sh
```

Expected: clones repo, syncs deps, prints `cuda: True`, lists FUSE-mounted localization datasets.

- [ ] **Step 3: Commit**

```
git add research/leps_localizer/scripts/setup_workspace_vm.sh
git commit -m "feat(leps-localizer): VM bootstrap script for ami-workspace-02-gpu"
```

---

## Phase 3 — ami-ml side: deps + index adapter

Working directory: `~/Projects/AMI/ami-ml/`. Do this on the laptop, then push.

### Task 10: Add detection extras to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional dep group**

In the `[project.optional-dependencies]` (or equivalent uv section), add:

```toml
detection = [
  "ultralytics>=8.3.0",
  "transformers>=4.45.0",
  "pyarrow>=15.0.0",
  "boto3>=1.34.0",
  "s3fs>=2024.6.0",
]
```

- [ ] **Step 2: Sync**

```
uv sync --extra dev --extra research --extra detection
```

Expected: resolves cleanly.

- [ ] **Step 3: Commit**

```
git add pyproject.toml uv.lock
git commit -m "feat(localization): add detection extra (ultralytics, transformers, pyarrow)"
```

### Task 11: `LepsLocalizerDataset` — index.jsonl → torchvision targets

**Files:**
- Create: `src/localization/leps_data.py`
- Test: `tests/localization/test_leps_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/localization/test_leps_data.py
import json
from pathlib import Path
from PIL import Image
from src.localization.leps_data import LepsLocalizerDataset


def _make_fixture(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images" / "ab"
    images_dir.mkdir(parents=True)
    img = Image.new("RGB", (200, 100), color=(255, 0, 0))
    img.save(images_dir / "abcdef.jpg")
    rec = {
        "dataset_name": "test", "dataset_version": "v1", "data_source": "fg",
        "extracted_at": "2026-05-05T00:00:00Z", "extract_commit_sha": "abc",
        "image_key": "images/ab/abcdef.jpg", "sha256": "abcdef", "photo_id": "p1",
        "category_id": "c1", "parents": ["a"], "user_id": "u1",
        "width": 200, "height": 100,
        "bbox_xyxy": [10.0, 10.0, 60.0, 60.0],
        "bbox_area_fraction": 0.125, "bbox_is_square": True,
        "created_at": "2024-09-01T00:00:00Z", "split": "train",
    }
    rec2 = {**rec, "photo_id": "p2", "image_key": "images/ab/abcdef.jpg", "split": "val"}
    (tmp_path / "index.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec2) + "\n")
    return tmp_path


def test_train_split_only(tmp_path: Path):
    root = _make_fixture(tmp_path)
    ds = LepsLocalizerDataset(root, split="train")
    assert len(ds) == 1
    image, target = ds[0]
    assert image.size == (200, 100)
    assert target["boxes"].tolist() == [[10.0, 10.0, 60.0, 60.0]]
    assert target["labels"].tolist() == [1]


def test_filter_by_area_fraction(tmp_path: Path):
    root = _make_fixture(tmp_path)
    ds = LepsLocalizerDataset(root, split="train", min_area_fraction=0.5)
    assert len(ds) == 0
```

- [ ] **Step 2: Run test, see it fail**

```
uv run pytest tests/localization/test_leps_data.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement**

```python
# src/localization/leps_data.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class LepsLocalizerDataset(Dataset):
    """Reads index.jsonl + local images dir; yields torchvision-style targets.

    Single class only — `labels` is always 1 (foreground 'arthropod').
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        min_area_fraction: float = 0.0,
        max_area_fraction: float = 1.0,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        records = []
        with (self.root / "index.jsonl").open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["split"] != split:
                    continue
                if not (min_area_fraction <= r["bbox_area_fraction"] <= max_area_fraction):
                    continue
                records.append(r)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img_path = self.root / r["image_key"]
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor([r["bbox_xyxy"]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "photo_id": r["photo_id"]}
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
```

- [ ] **Step 4: Run tests, see them pass**

```
uv run pytest tests/localization/test_leps_data.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add src/localization/leps_data.py tests/localization/test_leps_data.py
git commit -m "feat(localization): LepsLocalizerDataset for index.jsonl-driven training"
```

### Task 12: `metrics.py` extensions (containment, best-of-N, area-frac buckets)

**Files:**
- Modify: `src/localization/metrics.py`
- Create: `tests/localization/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/localization/test_metrics.py
import torch
from src.localization.metrics import (
    iou_xyxy, containment_iou, best_of_n_match, recall_by_bucket,
)


def test_iou_perfect():
    a = torch.tensor([0.0, 0.0, 10.0, 10.0])
    assert iou_xyxy(a, a) == 1.0


def test_containment_inside_square_gt():
    # tight rect prediction fully inside square gt
    pred = torch.tensor([10.0, 10.0, 20.0, 20.0])  # 100 area
    gt = torch.tensor([0.0, 0.0, 30.0, 30.0])      # 900 area
    # iou is small, but containment(pred ∩ gt / area(gt)) is small too,
    # so use containment(pred ∩ gt / area(pred)) — predicted region fully in gt
    assert containment_iou(pred, gt) == 1.0


def test_best_of_n_takes_max_iou():
    preds = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 10.0, 10.0]])
    gt = torch.tensor([0.0, 0.0, 10.0, 10.0])
    assert best_of_n_match(preds, gt, iou_thr=0.5) is True


def test_recall_by_bucket():
    matches = [True, False, True, False]
    fracs = [0.05, 0.05, 0.5, 0.5]
    buckets = [(0.0, 0.1), (0.1, 1.0)]
    assert recall_by_bucket(matches, fracs, buckets) == {(0.0, 0.1): 0.5, (0.1, 1.0): 0.5}
```

- [ ] **Step 2: Run test, see it fail**

```
uv run pytest tests/localization/test_metrics.py -v
```

- [ ] **Step 3: Implement**

```python
# src/localization/metrics.py  (append; keep existing helpers)
import torch


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def containment_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Pred area inside gt as fraction of pred area.

    Handles the FG-square-gt vs tight-rect-pred mismatch: a tight prediction
    fully inside a square gt scores 1.0 here even if IoU is low.
    """
    px1, py1, px2, py2 = pred.tolist()
    gx1, gy1, gx2, gy2 = gt.tolist()
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = (px2 - px1) * (py2 - py1)
    return inter / pred_area if pred_area > 0 else 0.0


def best_of_n_match(preds: torch.Tensor, gt: torch.Tensor, iou_thr: float = 0.5) -> bool:
    """Single-subject task: any predicted box near gt counts as a hit."""
    if len(preds) == 0:
        return False
    return max(iou_xyxy(p, gt) for p in preds) >= iou_thr


def recall_by_bucket(
    matches: list[bool],
    area_fractions: list[float],
    buckets: list[tuple[float, float]],
) -> dict[tuple[float, float], float]:
    """Return per-bucket recall."""
    out: dict[tuple[float, float], float] = {}
    for lo, hi in buckets:
        idxs = [i for i, f in enumerate(area_fractions) if lo <= f < hi]
        if not idxs:
            out[(lo, hi)] = float("nan")
            continue
        out[(lo, hi)] = sum(matches[i] for i in idxs) / len(idxs)
    return out
```

- [ ] **Step 4: Pass**

```
uv run pytest tests/localization/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```
git add src/localization/metrics.py tests/localization/test_metrics.py
git commit -m "feat(localization): containment, best-of-N, area-frac bucket metrics"
```

### Task 13: `eval_on_locked.py` — checkpoint → 4 eval indexes → markdown report

**Files:**
- Create: `src/localization/eval_on_locked.py`

- [ ] **Step 1: Implement**

```python
# src/localization/eval_on_locked.py
"""Run a saved checkpoint against the 4 locked eval indexes.

Outputs:
  reports/<arch>-<run-id>.md  — overall mAP@50, mAP@50-95, per-bucket recall,
                                IoU + containment per dataset, best-of-N hit rate.
  reports/<arch>-<run-id>.jsonl — one prediction record per eval image.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import click
import torch
from PIL import Image

from .leps_data import LepsLocalizerDataset
from .metrics import iou_xyxy, containment_iou, best_of_n_match, recall_by_bucket


EVAL_DATASETS = {
    "leps-butterflies-500":         "/mnt/s3-trainingdata/ai-for-leps/localization/eval-locked/leps-butterflies-500",
    "leps-butterflies-small-1000":  "/mnt/s3-trainingdata/ai-for-leps/localization/eval-locked/leps-butterflies-small-1000",
    "leps-butterflies-medsmall-1000":"/mnt/s3-trainingdata/ai-for-leps/localization/eval-locked/leps-butterflies-medsmall-1000",
    "leeds-butterflies":            "/mnt/s3-trainingdata/ai-for-leps/localization/eval-locked/leeds-butterflies",
}
BUCKETS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.00)]


def _load_predictor(arch: str, ckpt: Path):
    if arch == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
        def predict(image: Image.Image):
            r = model.predict(image, verbose=False)[0]
            return r.boxes.xyxy.cpu()
        return predict
    elif arch == "rtdetr":
        # transformers RT-DETR loader
        raise NotImplementedError("wire in Task 16")
    elif arch == "frcnn":
        from .utils import load_model
        m = load_model(model_type="fasterrcnn_resnet50_fpn_v2", ckpt_path=str(ckpt), num_classes=2)
        m.eval()
        from torchvision.transforms.v2 import functional as F
        def predict(image: Image.Image):
            t = F.to_image_tensor(image)
            t = F.convert_dtype(t, torch.float32)
            with torch.no_grad():
                preds = m([t.cuda() if torch.cuda.is_available() else t])
            return preds[0]["boxes"].cpu()
        return predict
    raise ValueError(f"unknown arch: {arch}")


@click.command()
@click.option("--arch", required=True, type=click.Choice(["yolo", "rtdetr", "frcnn"]))
@click.option("--ckpt", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--run-id", required=True)
@click.option("--out-dir", default="research/leps_localizer/reports", type=click.Path(path_type=Path))
def main(arch: str, ckpt: Path, run_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    predict = _load_predictor(arch, ckpt)

    report_lines = [f"# Eval report — {arch} / {run_id}\n", f"checkpoint: `{ckpt}`\n"]
    preds_jsonl = (out_dir / f"{arch}-{run_id}.jsonl").open("w")

    for ds_name, ds_root in EVAL_DATASETS.items():
        ds = LepsLocalizerDataset(Path(ds_root), split="eval")
        matches: list[bool] = []
        ious: list[float] = []
        contains: list[float] = []
        fracs: list[float] = []
        for image, target in ds:
            pred_boxes = predict(image)
            gt = target["boxes"][0]
            best_iou = max([iou_xyxy(p, gt) for p in pred_boxes], default=0.0)
            best_contain = max([containment_iou(p, gt) for p in pred_boxes], default=0.0)
            hit = best_of_n_match(pred_boxes, gt, iou_thr=0.5)
            matches.append(hit)
            ious.append(best_iou)
            contains.append(best_contain)
            fracs.append((gt[2] - gt[0]) * (gt[3] - gt[1]) / (image.size[0] * image.size[1]))
            preds_jsonl.write(json.dumps({
                "dataset": ds_name,
                "photo_id": target["photo_id"],
                "best_iou": best_iou,
                "best_containment": best_contain,
                "hit_at_iou50": hit,
                "n_preds": int(len(pred_boxes)),
            }) + "\n")
        recall_total = sum(matches) / len(matches) if matches else float("nan")
        bucket_recall = recall_by_bucket(matches, fracs, BUCKETS)
        mean_iou = sum(ious) / len(ious) if ious else float("nan")
        mean_contain = sum(contains) / len(contains) if contains else float("nan")
        report_lines.append(
            f"\n## {ds_name}  (n={len(matches)})\n"
            f"- best-of-N recall@IoU≥0.5: **{recall_total:.3f}**\n"
            f"- mean best IoU: {mean_iou:.3f}\n"
            f"- mean best containment: {mean_contain:.3f}\n"
            f"- per-bucket recall: " + ", ".join(
                f"[{lo:.2f},{hi:.2f}): {v:.3f}" for (lo, hi), v in bucket_recall.items()
            ) + "\n"
        )

    preds_jsonl.close()
    report_path = out_dir / f"{arch}-{run_id}.md"
    report_path.write_text("\n".join(report_lines))
    click.echo(f"wrote {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit (no test yet — exercised by Phase 4)**

```
git add src/localization/eval_on_locked.py
git commit -m "feat(localization): eval_on_locked.py runs 4 eval datasets, emits markdown + jsonl"
```

---

## Phase 4 — YOLO training run (on `ami-workspace-02-gpu`)

Working directory: `~/ami-ml/` on the VM.

### Task 14: YOLO data config + manifest converter

**Files:**
- Create: `research/leps_localizer/configs/yolo11s.yaml`
- Create: `src/localization/yolo_train.py`

- [ ] **Step 1: Write the YOLO data yaml**

```yaml
# research/leps_localizer/configs/yolo11s.yaml
path: /mnt/s3-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05
train: manifest_train.txt
val:   manifest_val.txt
names:
  0: arthropod
```

- [ ] **Step 2: Implement YOLO trainer wrapper**

```python
# src/localization/yolo_train.py
"""YOLOv11s/v8s wrapper. Reads index.jsonl, writes Ultralytics-format labels
on first run (cached on disk), then trains."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import click

from .leps_data import LepsLocalizerDataset


def materialize_yolo_labels(dataset_root: Path) -> None:
    """Write YOLO-format .txt labels next to each image based on index.jsonl.
    Format: <class_idx> <cx> <cy> <w> <h>  (normalized 0-1)
    """
    labels_dir = dataset_root / "labels"
    if labels_dir.exists() and (labels_dir / ".done").exists():
        return
    labels_dir.mkdir(exist_ok=True)
    with (dataset_root / "index.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            x1, y1, x2, y2 = r["bbox_xyxy"]
            w, h = r["width"], r["height"]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            stem = Path(r["image_key"]).stem
            (labels_dir / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    (labels_dir / ".done").touch()


@click.command()
@click.option("--data", required=True, type=click.Path(path_type=Path), help="path to yolo data .yaml")
@click.option("--dataset-root", required=True, type=click.Path(path_type=Path))
@click.option("--model", default="yolo11s.pt")
@click.option("--imgsz", default=640)
@click.option("--epochs", default=100)
@click.option("--batch", default=32)
@click.option("--name", required=True)
@click.option("--project", default="leps_localizer")
@click.option("--device", default="0")
def main(data, dataset_root, model, imgsz, epochs, batch, name, project, device):
    from ultralytics import YOLO
    materialize_yolo_labels(Path(dataset_root))
    m = YOLO(model)
    m.train(
        data=str(data), imgsz=imgsz, epochs=epochs, batch=batch,
        name=name, project=project, device=device,
        optimizer="AdamW", lr0=0.001, cos_lr=True, warmup_epochs=3,
        patience=20, mosaic=1.0, mixup=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        save=True, plots=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add CLI entry**

In `pyproject.toml`:

```toml
[project.scripts]
ami-localizer-yolo = "src.localization.yolo_train:main"
ami-localizer-eval = "src.localization.eval_on_locked:main"
```

(Keep existing entries; append.)

- [ ] **Step 4: Push**

```
uv sync --extra detection
git add src/localization/yolo_train.py research/leps_localizer/configs/yolo11s.yaml pyproject.toml uv.lock
git commit -m "feat(localization): YOLOv11s training wrapper + entry script"
git push origin feat/leps-localizer-training
```

### Task 15: Train YOLOv11s on `ami-workspace-02-gpu`

- [ ] **Step 1: SSH to VM, pull, run smoke**

```
ssh ami-workspace-02-gpu
cd ~/ami-ml && git pull
uv sync --extra dev --extra research --extra detection
GIT_AUTHOR_NAME='Michael Bunsen' GIT_COMMITTER_NAME='Michael Bunsen' \
GIT_AUTHOR_EMAIL='michael@mixedneeds.com' GIT_COMMITTER_EMAIL='michael@mixedneeds.com' \
  uv run ami-localizer-yolo \
  --data research/leps_localizer/configs/yolo11s.yaml \
  --dataset-root /mnt/s3-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05 \
  --epochs 3 --name yolo11s-smoke
```

Expected: completes 3 epochs, prints final mAP, no errors.

- [ ] **Step 2: Full run**

```
nohup uv run ami-localizer-yolo \
  --data research/leps_localizer/configs/yolo11s.yaml \
  --dataset-root /mnt/s3-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05 \
  --epochs 100 --name yolo11s-butterflies-v0 \
  > ~/runs/yolo11s-v0.log 2>&1 &
```

Monitor: `tail -f ~/runs/yolo11s-v0.log`. Expected runtime: ~12-24 hours.

- [ ] **Step 3: Eval after training**

```
uv run ami-localizer-eval \
  --arch yolo \
  --ckpt /home/debian/ami-ml/leps_localizer/yolo11s-butterflies-v0/weights/best.pt \
  --run-id v0
```

Expected: 4 eval reports written to `research/leps_localizer/reports/yolo-v0.md` and `.jsonl`.

- [ ] **Step 4: Mirror artefacts**

```
aws s3 sync \
  /home/debian/ami-ml/leps_localizer/yolo11s-butterflies-v0/ \
  s3://ami-trainingdata/ai-for-leps/localization/runs/yolo11s-butterflies-v0/

# from beast (later, optional):
rsync -av --progress \
  ami-workspace-02-gpu:/home/debian/ami-ml/leps_localizer/yolo11s-butterflies-v0/ \
  /media/michael/ZWEIBEL/MODELS/leps_localizer/yolo11s-butterflies-v0/
```

- [ ] **Step 5: Commit the report**

```
cd ~/ami-ml
git add research/leps_localizer/reports/yolo-v0.md research/leps_localizer/reports/yolo-v0.jsonl
git commit -m "report(leps-localizer): YOLOv11s v0 eval results"
git push origin feat/leps-localizer-training
```

---

## Phase 5 — RT-DETRv2 + FRCNN runs

### Task 16: RT-DETRv2 trainer

**Files:**
- Create: `src/localization/rtdetr_train.py`
- Create: `research/leps_localizer/configs/rtdetr.yaml`

- [ ] **Step 1: Implement RT-DETR trainer using `transformers.RTDetrForObjectDetection`**

(Skeleton matching `yolo_train.py` shape; full implementation deferred until YOLO is validated. If `transformers` RT-DETR API changes, fall back to the official `rtdetrv2_pytorch` repo as a vendored submodule.)

- [ ] **Step 2: Wire `eval_on_locked.py` rtdetr branch**

Replace the `NotImplementedError` in `eval_on_locked.py::_load_predictor` with the matching loader.

- [ ] **Step 3: Train + eval (on VM, parallel to or after YOLO)**

```
nohup uv run ami-localizer-rtdetr \
  --data /mnt/s3-trainingdata/ai-for-leps/localization/butterflies-fg-2026-05 \
  --name rtdetr-butterflies-v0 --epochs 50 \
  > ~/runs/rtdetr-v0.log 2>&1 &
```

Then `ami-localizer-eval --arch rtdetr --ckpt … --run-id v0`.

- [ ] **Step 4: Commit reports**

```
git add research/leps_localizer/reports/rtdetr-v0.md research/leps_localizer/reports/rtdetr-v0.jsonl
git commit -m "report(leps-localizer): RT-DETRv2 v0 eval results"
```

### Task 17: Faster R-CNN run via existing `src/localization/training.py`

**Files:**
- Modify: `src/localization/training.py` (add path to read `index.jsonl` instead of the legacy single-JSON-per-dir format)
- Or: write a small adapter that converts `index.jsonl` to the legacy `{filename: [bboxes, labels]}` JSON and points `--data_dir` at it

- [ ] **Step 1: Adapter or modification**

(Choose the lighter path: adapter shell script that emits the legacy JSON from `index.jsonl`, since `training.py` is treated as upstream.)

- [ ] **Step 2: Train**

```
uv run python src/localization/training.py \
  --model_type fasterrcnn_resnet50_fpn_v2 \
  --pretrained_backbone True \
  --num_epochs 30 --early_stop 10 --lr 0.005 \
  --batch_size 8 --num_workers 6 \
  --data_dir /home/debian/leps-frcnn/ \
  --save_dir /home/debian/runs/frcnn-v0 \
  --wandb_project leps_localizer
```

- [ ] **Step 3: Eval + commit**

Same protocol as Tasks 13/15.

```
uv run ami-localizer-eval --arch frcnn --ckpt /home/debian/runs/frcnn-v0/best.pt --run-id v0
git add research/leps_localizer/reports/frcnn-v0.md
git commit -m "report(leps-localizer): Faster R-CNN v0 eval results"
```

### Task 18: Comparison report

**Files:**
- Create: `research/leps_localizer/reports/comparison-2026-MM-DD.md`

- [ ] **Step 1: Write the comparison**

Manual or scripted (small Python that reads the 3 `*.jsonl` and tabulates). Markdown table comparing per-eval-set recall@IoU≥0.5, mean IoU, mean containment, per-bucket recall.

- [ ] **Step 2: Commit + push**

```
git add research/leps_localizer/reports/comparison-2026-MM-DD.md
git commit -m "report(leps-localizer): 3-architecture comparison"
git push origin feat/leps-localizer-training
```

---

## Definition of done

- [ ] `butterflies-fg-2026-05` uploaded to S3 with `index.jsonl`, manifests, README
- [ ] 4 eval datasets re-emitted as `index.jsonl` under `eval-locked/` on S3
- [ ] YOLOv11s trained, evaluated, report committed
- [ ] RT-DETRv2 trained, evaluated, report committed
- [ ] Faster R-CNN ResNet50 FPN v2 trained, evaluated, report committed
- [ ] Side-by-side comparison report at `research/leps_localizer/reports/comparison-*.md`
- [ ] Best checkpoint exported to ONNX
- [ ] Inference benchmark on `beast` 2× RTX 3090 for the best model: img/s, GPU util, peak VRAM
