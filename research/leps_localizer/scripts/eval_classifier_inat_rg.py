# flake8: noqa: E221, E231
"""Independent iNat-RG eval for the two global-butterfly classifiers
(species + subspecies) — PyTorch reference vs ONNX runtime.

Subcommands:

  fetch           — pull last N RG butterfly observations from iNat API
                    → observations.jsonl
  download-photos — download medium photos to photos/<obs_id>.jpg
  infer-pytorch   — PyTorch reference inference → pytorch_<model>.jsonl
  infer-onnx      — onnxruntime CPU inference → onnx_<model>.jsonl
  aggregate       — join PT + ONNX + groundtruth → metrics.json + report.md

Preprocessing is the canonical pad-to-square (top-left anchor, black fill)
→ Resize(N,N) → ToTensor (/255) → ImageNet mean/std normalize. Matches
`src/classification/dataloader.py:18-29` (val transform). Both species and
subspecies models were trained this way.

Run order:
    uv run python research/leps_localizer/scripts/eval_classifier_inat_rg.py fetch --n 1000
    uv run python ... download-photos
    uv run python ... infer-pytorch --model species
    uv run python ... infer-pytorch --model subspecies
    uv run python ... infer-onnx --model species
    uv run python ... infer-onnx --model subspecies
    uv run python ... aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------- constants ----------------------------- #

INAT_API = "https://api.inaturalist.org/v1/observations"
INAT_TAXON_BUTTERFLIES = 47224  # Papilionoidea (NOT 47222 — that's Apoidea/bees)
USER_AGENT = "ami-ml-research/0.1 (michael@mixedneeds.com)"

# iNat family IDs (resolved via /v1/taxa?q=<name>&rank=family on 2026-05-12)
INAT_FAMILY_IDS = {
    47922: "Nymphalidae",
    47923: "Lycaenidae",
    59166: "Riodinidae",
    48508: "Pieridae",
    47223: "Papilionidae",
    47653: "Hesperiidae",
    # additional moth-adjacent families that occasionally appear in butterfly searches
    # (Hedylidae 49641 — American moth-butterflies, very rare)
    49641: "Hedylidae",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_CONFIGS = {
    "species": {
        "model_id": "global-butterflies-resnet50-512",
        "arch": "resnet50",
        "num_classes": 8851,
        "input_size": 512,
        "weights": "/home/michael/.cache/ami-ml/hf/mohammedelabbas/global-butterflies-max1000img-512/resnet50_20260504_135731_checkpoint.pt",
        "label_map": "/home/michael/.cache/ami-ml/hf/mohammedelabbas/global-butterflies-max1000img-512/label_map.json",
        "onnx": "/home/michael/Projects/LepsAI/models/staging/export-20260508T234922Z/global-butterflies-resnet50-512.onnx",
    },
    "subspecies": {
        "model_id": "global-butterflies-subspecies-resnet50-128",
        "arch": "resnet50",
        "num_classes": 19411,
        "input_size": 128,
        "weights": "/home/michael/.cache/ami-ml/hf/mohammedelabbas/global-butterflies/resnet50_epoch30_checkpoint.pt",
        "label_map": "/home/michael/.cache/ami-ml/hf/mohammedelabbas/global-butterflies/label_map.json",
        "onnx": "/home/michael/Projects/LepsAI/models/staging/global-butterflies-subspecies-resnet50-128.onnx",
    },
}

FAMILIES_TO_REPORT = [
    "Nymphalidae",
    "Pieridae",
    "Lycaenidae",
    "Papilionidae",
    "Hesperiidae",
    "Riodinidae",
]


# ----------------------------- shared utils ----------------------------- #


def _load_label_list(label_map_path: Path, num_classes: int) -> list[str]:
    raw = json.loads(label_map_path.read_text())
    if isinstance(raw, list):
        labels = list(raw)
    elif isinstance(raw, dict):
        try:
            labels = [raw[str(i)] for i in range(num_classes)]
        except KeyError as e:
            raise ValueError(f"label_map missing index {e}") from e
    else:
        raise ValueError(f"unexpected label_map type: {type(raw).__name__}")
    if len(labels) != num_classes:
        raise ValueError(f"label_map has {len(labels)} entries, expected {num_classes}")
    return labels


def _pad_to_square_pil(img):
    """Top-left anchored pad to square with black fill — matches
    `src/classification/dataloader.py:18-29`."""
    from torchvision import transforms

    w, h = img.size
    if h < w:
        return transforms.Pad(padding=[0, 0, 0, w - h])(img)
    if h > w:
        return transforms.Pad(padding=[0, 0, h - w, 0])(img)
    return img


def _eval_transform(input_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Lambda(_pad_to_square_pil),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _binomial(name: str) -> str:
    parts = name.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name


def _http_get_json(url: str, max_retries: int = 3, retry_sleep_s: float = 2.0):
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            print(f"  http retry {attempt + 1}/{max_retries}: {e}", file=sys.stderr)
            time.sleep(retry_sleep_s * (attempt + 1))
    raise RuntimeError(f"http get failed after {max_retries} tries: {last_err}")


# ----------------------------- fetch ----------------------------- #


def cmd_fetch(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "observations.jsonl"

    target_n = args.n
    per_page = 200
    id_below = None
    kept: list[dict] = []
    n_pages = 0
    n_seen = 0
    n_dropped = {
        "no_taxon": 0,
        "bad_rank": 0,
        "no_photos": 0,
        "no_license": 0,
        "future_date": 0,
        "no_observed_on": 0,
    }

    print(f"fetching up to {target_n} RG butterfly observations from iNat...")
    today = datetime.now(timezone.utc).date()

    while len(kept) < target_n:
        params = {
            "taxon_id": INAT_TAXON_BUTTERFLIES,
            "quality_grade": "research",
            "order": "desc",
            "order_by": "observed_on",
            "per_page": per_page,
            "photos": "true",
        }
        if id_below is not None:
            params["id_below"] = id_below
        url = INAT_API + "?" + urllib.parse.urlencode(params)
        data = _http_get_json(url)
        results = data.get("results", [])
        n_pages += 1
        n_seen += len(results)
        if not results:
            print(f"  no more results after page {n_pages}")
            break

        for r in results:
            taxon = r.get("taxon")
            if not taxon:
                n_dropped["no_taxon"] += 1
                continue
            rank = taxon.get("rank")
            if rank not in ("species", "subspecies"):
                n_dropped["bad_rank"] += 1
                continue
            photos = r.get("photos") or []
            if not photos:
                n_dropped["no_photos"] += 1
                continue
            license_code = photos[0].get("license_code")
            if not license_code:
                n_dropped["no_license"] += 1
                continue
            observed_on = r.get("observed_on")
            if not observed_on:
                n_dropped["no_observed_on"] += 1
                continue
            try:
                obs_date = datetime.fromisoformat(observed_on).date()
                if obs_date > today:
                    n_dropped["future_date"] += 1
                    continue
            except ValueError:
                n_dropped["no_observed_on"] += 1
                continue

            photo_url = photos[0].get("url") or ""
            # iNat returns the square thumbnail URL; swap to medium (~500px long side)
            photo_url_medium = photo_url.replace("/square.", "/medium.")
            # fallback for non-square original URLs
            if photo_url_medium == photo_url and "square" in photo_url:
                photo_url_medium = photo_url.replace("square", "medium")

            # iNat's observations endpoint returns `ancestor_ids` (list of int IDs)
            # but not `ancestors` (list of {id,rank,name}). Resolve family via
            # hardcoded butterfly-family ID map.
            ancestor_ids = taxon.get("ancestor_ids") or []
            family = None
            for aid in ancestor_ids:
                if aid in INAT_FAMILY_IDS:
                    family = INAT_FAMILY_IDS[aid]
                    break
            # genus = first token of binomial name
            genus = (taxon.get("name") or "").split()[0]

            row = {
                "inat_obs_id": r.get("id"),
                "taxon_name": taxon.get("name"),
                "taxon_id": taxon.get("id"),
                "taxon_rank": rank,
                "family": family,
                "genus": genus,
                "ancestor_ids": ancestor_ids,
                "photo_url_medium": photo_url_medium,
                "observed_on": observed_on,
                "place_guess": r.get("place_guess"),
                "license_code": license_code,
            }
            kept.append(row)
            if len(kept) >= target_n:
                break

        # pagination cursor: smallest id in this page becomes next id_below
        if results:
            id_below = min(r["id"] for r in results)
        print(
            f"  page {n_pages}: results={len(results)} kept_total={len(kept)} "
            f"id_below_next={id_below}"
        )
        time.sleep(0.5)  # be polite

    with obs_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    print("\nfetch summary:")
    print(f"  pages fetched: {n_pages}")
    print(f"  raw observations seen: {n_seen}")
    print(f"  kept: {len(kept)}")
    print(f"  dropped: {n_dropped}")
    print(f"  -> {obs_path}")
    return 0


# ----------------------------- download-photos ----------------------------- #


def _download_one(args_tuple):
    """Worker for parallel download: returns (obs_id, status, err)."""
    obs_id, url, target, tmp = args_tuple
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.rename(target)
        return (obs_id, "ok", None)
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return (obs_id, "fail", f"{type(e).__name__}: {e}")


def cmd_download_photos(args: argparse.Namespace) -> int:
    """Parallel downloader. iNat photos live on AWS S3 (no API rate limit) —
    a network round-trip per request is ~1-2s, so 16-way parallelism is safe
    and brings 1000 photos to ~1-2 minutes instead of ~30."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_dir = Path(args.out_dir)
    obs_path = out_dir / "observations.jsonl"
    photos_dir = out_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in obs_path.read_text().splitlines() if line]
    workers = getattr(args, "workers", 16)
    print(f"downloading {len(rows)} photos to {photos_dir}/ (parallel={workers})")

    todo = []
    n_skip = 0
    for row in rows:
        obs_id = row["inat_obs_id"]
        url = row["photo_url_medium"]
        if not url:
            continue
        target = photos_dir / f"{obs_id}.jpg"
        if target.exists() and target.stat().st_size > 0:
            n_skip += 1
            continue
        tmp = target.with_suffix(".jpg.part")
        todo.append((obs_id, url, target, tmp))

    print(f"  {n_skip} already on disk, {len(todo)} to fetch")

    n_ok = 0
    n_fail = 0
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            obs_id, status, err = fut.result()
            if status == "ok":
                n_ok += 1
            else:
                n_fail += 1
                if n_fail <= 10:
                    print(f"  obs {obs_id}: FAIL {err}")
            if i % 100 == 0:
                elapsed = time.monotonic() - t_start
                rate = i / max(elapsed, 1e-3)
                eta = (len(todo) - i) / max(rate, 1e-3)
                print(
                    f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail} "
                    f"rate={rate:.1f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

    print(
        f"\ndownload summary: ok={n_ok} skip={n_skip} fail={n_fail} "
        f"elapsed={time.monotonic() - t_start:.0f}s"
    )
    return 0


# ----------------------------- pytorch inference ----------------------------- #


def _load_pytorch_model(cfg: dict, device: str):
    import timm
    import torch

    model = timm.create_model(
        cfg["arch"], pretrained=False, num_classes=cfg["num_classes"]
    )
    ckpt = torch.load(cfg["weights"], map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


def _topk_from_logits(logits, labels: list[str], k: int):
    import numpy as np

    arr = logits.flatten()
    idx = np.argsort(arr)[::-1][:k]
    return (
        [int(i) for i in idx],
        [float(arr[i]) for i in idx],
        [labels[int(i)] for i in idx],
    )


def cmd_infer_pytorch(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from PIL import Image

    cfg = MODEL_CONFIGS[args.model]
    out_dir = Path(args.out_dir)
    obs_path = out_dir / "observations.jsonl"
    photos_dir = out_dir / "photos"
    out_path = out_dir / f"pytorch_{args.model}.jsonl"

    rows = [json.loads(line) for line in obs_path.read_text().splitlines() if line]
    labels = _load_label_list(Path(cfg["label_map"]), cfg["num_classes"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {cfg['model_id']} on {device}")
    model = _load_pytorch_model(cfg, device)
    transform = _eval_transform(cfg["input_size"])

    batch_size = args.batch_size
    n_ok = 0
    n_missing = 0
    n_err = 0
    t0 = time.monotonic()

    with out_path.open("w") as fout, torch.no_grad():
        batch_imgs = []
        batch_meta = []

        def _flush():
            nonlocal n_ok
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs).to(device)
            t_start = time.monotonic()
            logits = model(x).cpu().numpy().astype(np.float32)
            elapsed_per = (time.monotonic() - t_start) * 1000.0 / len(batch_imgs)
            for (obs_id, _), row_logits in zip(batch_meta, logits):
                top5_idx, top5_val, top5_names = _topk_from_logits(
                    row_logits, labels, 5
                )
                rec = {
                    "inat_obs_id": obs_id,
                    "model": args.model,
                    "model_id": cfg["model_id"],
                    "runtime": "pytorch",
                    "input_size": cfg["input_size"],
                    "logits_top5_idx": top5_idx,
                    "logits_top5_val": [round(v, 6) for v in top5_val],
                    "top1_pred_name": top5_names[0],
                    "top5_pred_names": top5_names,
                    "elapsed_ms": round(elapsed_per, 3),
                }
                fout.write(json.dumps(rec) + "\n")
                n_ok += 1
            batch_imgs.clear()
            batch_meta.clear()

        for i, row in enumerate(rows):
            obs_id = row["inat_obs_id"]
            photo = photos_dir / f"{obs_id}.jpg"
            if not photo.exists() or photo.stat().st_size == 0:
                n_missing += 1
                continue
            try:
                img = Image.open(photo).convert("RGB")
                x = transform(img)
            except Exception as e:
                print(f"  obs {obs_id}: decode/transform FAIL {type(e).__name__}: {e}")
                n_err += 1
                continue
            batch_imgs.append(x)
            batch_meta.append((obs_id, row))
            if len(batch_imgs) >= batch_size:
                _flush()
                if (i + 1) % (batch_size * 4) == 0:
                    print(f"  [{i + 1}/{len(rows)}] ok={n_ok}")

        _flush()

    print(
        f"\npytorch {args.model} done: ok={n_ok} missing={n_missing} err={n_err} "
        f"elapsed={time.monotonic() - t0:.0f}s -> {out_path}"
    )
    return 0


# ----------------------------- onnx inference ----------------------------- #


def cmd_infer_onnx(args: argparse.Namespace) -> int:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    cfg = MODEL_CONFIGS[args.model]
    out_dir = Path(args.out_dir)
    obs_path = out_dir / "observations.jsonl"
    photos_dir = out_dir / "photos"
    out_path = out_dir / f"onnx_{args.model}.jsonl"

    rows = [json.loads(line) for line in obs_path.read_text().splitlines() if line]
    labels = _load_label_list(Path(cfg["label_map"]), cfg["num_classes"])

    print(f"loading ONNX {cfg['onnx']} (CPU)")
    sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    transform = _eval_transform(cfg["input_size"])

    n_ok = 0
    n_missing = 0
    n_err = 0
    t0 = time.monotonic()

    with out_path.open("w") as fout:
        for i, row in enumerate(rows):
            obs_id = row["inat_obs_id"]
            photo = photos_dir / f"{obs_id}.jpg"
            if not photo.exists() or photo.stat().st_size == 0:
                n_missing += 1
                continue
            try:
                img = Image.open(photo).convert("RGB")
                x = transform(img).unsqueeze(0).numpy().astype(np.float32)
            except Exception as e:
                print(f"  obs {obs_id}: decode/transform FAIL {type(e).__name__}: {e}")
                n_err += 1
                continue
            t_start = time.monotonic()
            logits = sess.run(None, {in_name: x})[0].astype(np.float32).flatten()
            elapsed_ms = (time.monotonic() - t_start) * 1000.0
            top5_idx, top5_val, top5_names = _topk_from_logits(logits, labels, 5)
            rec = {
                "inat_obs_id": obs_id,
                "model": args.model,
                "model_id": cfg["model_id"],
                "runtime": "onnx",
                "input_size": cfg["input_size"],
                "logits_top5_idx": top5_idx,
                "logits_top5_val": [round(v, 6) for v in top5_val],
                "top1_pred_name": top5_names[0],
                "top5_pred_names": top5_names,
                "elapsed_ms": round(elapsed_ms, 3),
            }
            fout.write(json.dumps(rec) + "\n")
            n_ok += 1
            if (i + 1) % 100 == 0:
                print(
                    f"  [{i + 1}/{len(rows)}] ok={n_ok} elapsed={time.monotonic() - t0:.0f}s"
                )

    print(
        f"\nonnx {args.model} done: ok={n_ok} missing={n_missing} err={n_err} "
        f"elapsed={time.monotonic() - t0:.0f}s -> {out_path}"
    )
    return 0


# ----------------------------- parity (joint pt+onnx logit run) ----------------------------- #


def cmd_parity(args: argparse.Namespace) -> int:
    """Run PT and ONNX inference for a single model on the same images,
    in lockstep, computing per-image full-vector cosine sim, top-5 |Δlogit|,
    top-1 agreement. Writes parity_<model>.jsonl with per-image stats."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image

    cfg = MODEL_CONFIGS[args.model]
    out_dir = Path(args.out_dir)
    obs_path = out_dir / "observations.jsonl"
    photos_dir = out_dir / "photos"
    out_path = out_dir / f"parity_{args.model}.jsonl"

    rows = [json.loads(line) for line in obs_path.read_text().splitlines() if line]
    _load_label_list(Path(cfg["label_map"]), cfg["num_classes"])

    # For parity we want the cleanest possible comparison; cuda reduction order
    # differs from CPU and produces large nominal |Δlogit| (1.0+) at the cost of
    # readability, while cosine sim stays at 1.0 and top-1 agreement at 99.9+%.
    # Force CPU for both sides to expose the actual ONNX export drift.
    device = (
        "cpu"
        if getattr(args, "pt_device", "cpu") == "cpu"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading PT model on {device}")
    pt_model = _load_pytorch_model(cfg, device)
    print("loading ONNX (CPU)")
    sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    transform = _eval_transform(cfg["input_size"])

    n_ok = 0
    n_missing = 0
    n_err = 0
    t0 = time.monotonic()

    with out_path.open("w") as fout, torch.no_grad():
        for i, row in enumerate(rows):
            obs_id = row["inat_obs_id"]
            photo = photos_dir / f"{obs_id}.jpg"
            if not photo.exists() or photo.stat().st_size == 0:
                n_missing += 1
                continue
            try:
                img = Image.open(photo).convert("RGB")
                x_t = transform(img).unsqueeze(0)
            except Exception as e:
                print(f"  obs {obs_id}: decode/transform FAIL {type(e).__name__}: {e}")
                n_err += 1
                continue
            x_np = x_t.numpy().astype(np.float32)
            pt_logits = (
                pt_model(x_t.to(device)).cpu().numpy().astype(np.float32).flatten()
            )
            onnx_logits = (
                sess.run(None, {in_name: x_np})[0].astype(np.float32).flatten()
            )
            # per-image stats
            diff = pt_logits - onnx_logits
            max_abs_diff = float(np.max(np.abs(diff)))
            cos_sim = float(
                np.dot(pt_logits, onnx_logits)
                / (np.linalg.norm(pt_logits) * np.linalg.norm(onnx_logits) + 1e-12)
            )
            pt_top1 = int(np.argmax(pt_logits))
            onnx_top1 = int(np.argmax(onnx_logits))
            pt_top5 = np.argsort(pt_logits)[::-1][:5]
            onnx_top5 = np.argsort(onnx_logits)[::-1][:5]
            top5_abs_diff = float(
                np.max(
                    np.abs(np.concatenate([pt_logits[pt_top5] - onnx_logits[pt_top5]]))
                )
            )
            rec = {
                "inat_obs_id": obs_id,
                "model": args.model,
                "pt_top1_idx": pt_top1,
                "onnx_top1_idx": onnx_top1,
                "agree": pt_top1 == onnx_top1,
                "max_abs_logit_diff_full": max_abs_diff,
                "max_abs_logit_diff_top5": top5_abs_diff,
                "cosine_sim": cos_sim,
                "pt_top5_idx": [int(i) for i in pt_top5],
                "onnx_top5_idx": [int(i) for i in onnx_top5],
            }
            fout.write(json.dumps(rec) + "\n")
            n_ok += 1
            if (i + 1) % 100 == 0:
                print(
                    f"  [{i + 1}/{len(rows)}] ok={n_ok} elapsed={time.monotonic() - t0:.0f}s"
                )

    print(
        f"\nparity {args.model} done: ok={n_ok} missing={n_missing} err={n_err} "
        f"elapsed={time.monotonic() - t0:.0f}s -> {out_path}"
    )
    return 0


# ----------------------------- aggregate ----------------------------- #


def _percentile(arr, p):
    import numpy as np

    if not len(arr):
        return None
    return float(np.percentile(arr, p))


def cmd_aggregate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    obs_path = out_dir / "observations.jsonl"
    obs_rows = [json.loads(line) for line in obs_path.read_text().splitlines() if line]
    obs_by_id = {r["inat_obs_id"]: r for r in obs_rows}

    def load_jsonl(path: Path) -> dict[int, dict]:
        if not path.exists():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            out[r["inat_obs_id"]] = r
        return out

    pt_species = load_jsonl(out_dir / "pytorch_species.jsonl")
    pt_sub = load_jsonl(out_dir / "pytorch_subspecies.jsonl")
    onnx_species = load_jsonl(out_dir / "onnx_species.jsonl")
    onnx_sub = load_jsonl(out_dir / "onnx_subspecies.jsonl")
    parity_species = load_jsonl(out_dir / "parity_species.jsonl")
    parity_sub = load_jsonl(out_dir / "parity_subspecies.jsonl")

    print(
        f"loaded: obs={len(obs_rows)} pt_sp={len(pt_species)} pt_sub={len(pt_sub)} "
        f"onnx_sp={len(onnx_species)} onnx_sub={len(onnx_sub)} "
        f"par_sp={len(parity_species)} par_sub={len(parity_sub)}"
    )

    # -------- accuracy --------
    def acc_for(infer_rows: dict[int, dict], comparison: str) -> dict:
        """comparison: 'species' (compare binomials) or 'subspecies' (compare trinomials)."""
        top1 = 0
        top5 = 0
        n = 0
        n_missing_gt = 0
        per_family_top1 = {f: [0, 0] for f in FAMILIES_TO_REPORT}
        for obs_id, ir in infer_rows.items():
            obs = obs_by_id.get(obs_id)
            if obs is None:
                continue
            gt = obs["taxon_name"]
            if not gt:
                n_missing_gt += 1
                continue
            if comparison == "species":
                gt_cmp = _binomial(gt)
                preds_cmp = [_binomial(p) for p in ir["top5_pred_names"]]
            elif comparison == "subspecies":
                # only count if GT is trinomial
                if len(gt.split()) < 3:
                    continue
                gt_cmp = gt
                preds_cmp = list(ir["top5_pred_names"])
            else:
                raise ValueError(comparison)
            t1 = preds_cmp[0] == gt_cmp
            t5 = gt_cmp in preds_cmp
            top1 += int(t1)
            top5 += int(t5)
            n += 1
            fam = obs.get("family")
            if fam in per_family_top1:
                per_family_top1[fam][0] += int(t1)
                per_family_top1[fam][1] += 1
        return {
            "n_eval": n,
            "n_missing_gt": n_missing_gt,
            "top1": (top1 / n) if n else None,
            "top5": (top5 / n) if n else None,
            "top1_count": top1,
            "top5_count": top5,
            "per_family": {
                f: {
                    "n": v[1],
                    "top1": (v[0] / v[1]) if v[1] else None,
                    "top1_count": v[0],
                }
                for f, v in per_family_top1.items()
            },
        }

    metrics: dict = {
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_observations": len(obs_rows),
        "models": {},
        "parity": {},
        "sample_composition": {},
    }

    # sample composition
    family_counts: dict[str, int] = {}
    rank_counts = {"species": 0, "subspecies": 0}
    date_range = [None, None]  # min, max
    place_counts: dict[str, int] = {}
    for o in obs_rows:
        fam = o.get("family") or "Unknown"
        family_counts[fam] = family_counts.get(fam, 0) + 1
        rank_counts[o["taxon_rank"]] = rank_counts.get(o["taxon_rank"], 0) + 1
        d = o.get("observed_on")
        if d:
            if date_range[0] is None or d < date_range[0]:
                date_range[0] = d
            if date_range[1] is None or d > date_range[1]:
                date_range[1] = d
        # top-level country guess: take last comma-separated segment
        pg = o.get("place_guess") or ""
        if pg:
            country = pg.split(",")[-1].strip() or "Unknown"
            place_counts[country] = place_counts.get(country, 0) + 1
    metrics["sample_composition"] = {
        "family_counts": dict(sorted(family_counts.items(), key=lambda kv: -kv[1])),
        "rank_counts": rank_counts,
        "date_min": date_range[0],
        "date_max": date_range[1],
        "top_places": dict(sorted(place_counts.items(), key=lambda kv: -kv[1])[:10]),
    }

    metrics["models"]["species"] = {
        "model_id": MODEL_CONFIGS["species"]["model_id"],
        "pytorch": acc_for(pt_species, "species"),
        "onnx": acc_for(onnx_species, "species"),
    }
    metrics["models"]["subspecies"] = {
        "model_id": MODEL_CONFIGS["subspecies"]["model_id"],
        "pytorch_species": acc_for(pt_sub, "species"),
        "pytorch_subspecies": acc_for(pt_sub, "subspecies"),
        "onnx_species": acc_for(onnx_sub, "species"),
        "onnx_subspecies": acc_for(onnx_sub, "subspecies"),
    }

    # -------- parity --------
    def parity_stats(parity_rows: dict[int, dict]) -> dict:
        if not parity_rows:
            return {}
        agree = [int(r["agree"]) for r in parity_rows.values()]
        max_diff_full = [r["max_abs_logit_diff_full"] for r in parity_rows.values()]
        max_diff_top5 = [r["max_abs_logit_diff_top5"] for r in parity_rows.values()]
        cos_sim = [r["cosine_sim"] for r in parity_rows.values()]
        return {
            "n": len(parity_rows),
            "top1_agreement_rate": sum(agree) / len(agree),
            "top1_disagreements": len(agree) - sum(agree),
            "max_abs_logit_diff_full_max": max(max_diff_full),
            "max_abs_logit_diff_full_mean": sum(max_diff_full) / len(max_diff_full),
            "max_abs_logit_diff_full_p95": _percentile(max_diff_full, 95),
            "max_abs_logit_diff_top5_max": max(max_diff_top5),
            "max_abs_logit_diff_top5_mean": sum(max_diff_top5) / len(max_diff_top5),
            "max_abs_logit_diff_top5_p95": _percentile(max_diff_top5, 95),
            "cosine_sim_mean": sum(cos_sim) / len(cos_sim),
            "cosine_sim_p5": _percentile(cos_sim, 5),
            "cosine_sim_min": min(cos_sim),
        }

    metrics["parity"]["species"] = parity_stats(parity_species)
    metrics["parity"]["subspecies"] = parity_stats(parity_sub)

    # write metrics.json
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics -> {metrics_path}")

    # -------- report.md --------
    lines: list[str] = []
    lines.append(f"# iNat-RG independent classifier eval — {metrics['run_date']}")
    lines.append("")
    sc = metrics["sample_composition"]
    lines.append(
        f"Pulled **{metrics['n_observations']}** RG butterfly observations from iNat between "
        f"**{sc['date_min']}** and **{sc['date_max']}**."
    )
    lines.append("")
    lines.append(
        f"Rank distribution: species={sc['rank_counts'].get('species', 0)}, "
        f"subspecies={sc['rank_counts'].get('subspecies', 0)}"
    )
    lines.append("")
    fam_str = ", ".join(f"{k}={v}" for k, v in list(sc["family_counts"].items())[:8])
    lines.append(f"Top families: {fam_str}")
    lines.append("")
    place_str = ", ".join(f"{k}={v}" for k, v in list(sc["top_places"].items())[:5])
    lines.append(f"Top countries: {place_str}")
    lines.append("")

    # summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | runtime | n_eval | top1 species | top5 species | top1 subspecies | top5 subspecies |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    sp_pt = metrics["models"]["species"]["pytorch"]
    sp_on = metrics["models"]["species"]["onnx"]
    lines.append(
        f"| global-butterflies-resnet50-512 | pytorch | {sp_pt['n_eval']} | "
        f"{sp_pt['top1']:.4f} | {sp_pt['top5']:.4f} | n/a | n/a |"
    )
    lines.append(
        f"| global-butterflies-resnet50-512 | onnx    | {sp_on['n_eval']} | "
        f"{sp_on['top1']:.4f} | {sp_on['top5']:.4f} | n/a | n/a |"
    )
    sub_pt_sp = metrics["models"]["subspecies"]["pytorch_species"]
    sub_pt_sub = metrics["models"]["subspecies"]["pytorch_subspecies"]
    sub_on_sp = metrics["models"]["subspecies"]["onnx_species"]
    sub_on_sub = metrics["models"]["subspecies"]["onnx_subspecies"]
    lines.append(
        f"| global-butterflies-subspecies-resnet50-128 | pytorch | {sub_pt_sp['n_eval']} | "
        f"{sub_pt_sp['top1']:.4f} | {sub_pt_sp['top5']:.4f} | "
        f"{'%.4f' % sub_pt_sub['top1'] if sub_pt_sub['top1'] is not None else 'n/a (no trinom GT)'} | "
        f"{'%.4f' % sub_pt_sub['top5'] if sub_pt_sub['top5'] is not None else 'n/a'} |"
    )
    lines.append(
        f"| global-butterflies-subspecies-resnet50-128 | onnx    | {sub_on_sp['n_eval']} | "
        f"{sub_on_sp['top1']:.4f} | {sub_on_sp['top5']:.4f} | "
        f"{'%.4f' % sub_on_sub['top1'] if sub_on_sub['top1'] is not None else 'n/a (no trinom GT)'} | "
        f"{'%.4f' % sub_on_sub['top5'] if sub_on_sub['top5'] is not None else 'n/a'} |"
    )
    lines.append("")
    lines.append(
        f"Subspecies eval denominators: PT={sub_pt_sub['n_eval']}, ONNX={sub_on_sub['n_eval']} "
        f"(only obs with trinomial GT count)."
    )
    lines.append("")

    # parity table
    lines.append("## PT ↔ ONNX parity")
    lines.append("")
    lines.append(
        "| Model | n | top-1 agreement | mean cos-sim | min cos-sim | mean \\|Δlogit\\| (full) | p95 \\|Δlogit\\| (full) | max \\|Δlogit\\| (full) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ("species", "subspecies"):
        p = metrics["parity"][name]
        if not p:
            continue
        lines.append(
            f"| {name} | {p['n']} | {p['top1_agreement_rate']:.4f} ({p['top1_disagreements']} disagreements) | "
            f"{p['cosine_sim_mean']:.6f} | {p['cosine_sim_min']:.6f} | "
            f"{p['max_abs_logit_diff_full_mean']:.2e} | "
            f"{p['max_abs_logit_diff_full_p95']:.2e} | "
            f"{p['max_abs_logit_diff_full_max']:.2e} |"
        )
    lines.append("")
    lines.append(
        "Gate: top-1 agreement ≥ 0.99 (warn at <0.95), cosine similarity > 0.9999."
    )
    lines.append("")

    # per-family
    lines.append("## Per-family top-1 species accuracy")
    lines.append("")
    lines.append(
        "| Family | n | PT species (sp 512) | ONNX species (sp 512) | PT sub→binomial (sub 128) | ONNX sub→binomial (sub 128) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for fam in FAMILIES_TO_REPORT:
        sp_pt_f = metrics["models"]["species"]["pytorch"]["per_family"].get(fam, {})
        sp_on_f = metrics["models"]["species"]["onnx"]["per_family"].get(fam, {})
        sub_pt_f = metrics["models"]["subspecies"]["pytorch_species"]["per_family"].get(
            fam, {}
        )
        sub_on_f = metrics["models"]["subspecies"]["onnx_species"]["per_family"].get(
            fam, {}
        )
        n = sp_pt_f.get("n", 0)
        if n < 30:
            lines.append(f"| {fam} | {n} | (suppressed, n<30) | | | |")
            continue

        def fmt(v):
            return "n/a" if v is None else f"{v:.4f}"

        lines.append(
            f"| {fam} | {n} | {fmt(sp_pt_f.get('top1'))} | {fmt(sp_on_f.get('top1'))} | "
            f"{fmt(sub_pt_f.get('top1'))} | {fmt(sub_on_f.get('top1'))} |"
        )
    lines.append("")

    # caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- iNat is heavily skewed toward English-speaking + Western Europe users. "
        "Geographic top-5 above gives a rough sense; African / SE-Asian taxa are "
        "severely underrepresented vs. Mohamed's training set."
    )
    lines.append(
        "- iNat photos here are the `medium` rewrite (~500 px on long side). The species "
        "model expects 512×512 input and may lose short-side detail after pad-to-square."
    )
    lines.append(
        "- Most iNat RG IDs are species-level — the subspecies-vs-trinomial-GT accuracy "
        "denominator is small. Per-family top-1 uses the **binomial-strip** comparison "
        "so the subspecies model's family numbers are directly comparable to the species "
        "model."
    )
    lines.append("- Per-family with n<30 is suppressed (too noisy).")
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"report -> {report_path}")
    return 0


# ----------------------------- argparse ----------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="pull RG butterfly observations from iNat")
    p_fetch.add_argument("--n", type=int, default=1000)
    p_fetch.add_argument("--out-dir", type=str, required=True)
    p_fetch.set_defaults(func=cmd_fetch)

    p_dl = sub.add_parser("download-photos", help="download medium photos")
    p_dl.add_argument("--out-dir", type=str, required=True)
    p_dl.add_argument("--workers", type=int, default=16)
    p_dl.set_defaults(func=cmd_download_photos)

    p_pt = sub.add_parser("infer-pytorch")
    p_pt.add_argument("--model", choices=["species", "subspecies"], required=True)
    p_pt.add_argument("--out-dir", type=str, required=True)
    p_pt.add_argument("--batch-size", type=int, default=32)
    p_pt.set_defaults(func=cmd_infer_pytorch)

    p_on = sub.add_parser("infer-onnx")
    p_on.add_argument("--model", choices=["species", "subspecies"], required=True)
    p_on.add_argument("--out-dir", type=str, required=True)
    p_on.set_defaults(func=cmd_infer_onnx)

    p_par = sub.add_parser(
        "parity", help="joint PT+ONNX run for full-vector parity stats"
    )
    p_par.add_argument("--model", choices=["species", "subspecies"], required=True)
    p_par.add_argument("--out-dir", type=str, required=True)
    p_par.add_argument(
        "--pt-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="device for the PT side. CPU recommended for clean parity; cuda's "
        "reduction order differs from ONNX-CPU and inflates max |Δlogit|.",
    )
    p_par.set_defaults(func=cmd_parity)

    p_ag = sub.add_parser("aggregate")
    p_ag.add_argument("--out-dir", type=str, required=True)
    p_ag.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
