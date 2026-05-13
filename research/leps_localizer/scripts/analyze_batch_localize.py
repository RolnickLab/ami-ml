# flake8: noqa: E221, E231
"""Analyze a batch_localize NDJSON output for benchmark / sanity stats.

Computes:
- detect / zero_det / missing rates
- score distribution (mean, p10/50/90, min/max)
- area_fraction distribution (mean, p10/50/90, min/max)
- area_fraction bucket counts matching METHODS_AND_RESULTS §1.6 binning
- failure flags (low_conf <0.3, tiny <0.005)
- by-source breakdown if `source` is joined back (not done here — pass --worklist
  to add it)
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def _area_bin(af: float) -> str:
    if af < 0.005:
        return "tiny<0.005"
    if af < 0.01:
        return "micro<0.01"
    if af < 0.05:
        return "<0.05"
    if af < 0.10:
        return "<0.10"
    if af < 0.25:
        return "<0.25"
    if af < 0.50:
        return "<0.50"
    return ">=0.50"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ndjson", type=Path)
    p.add_argument("--low-conf", type=float, default=0.30)
    p.add_argument("--tiny-area", type=float, default=0.005)
    args = p.parse_args()

    rows = []
    with args.ndjson.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    n = len(rows)
    n_zero = sum(1 for r in rows if not r["predicted_bbox_xyxy"])
    n_detected = n - n_zero
    scores = [r["score"] for r in rows if r["score"] is not None]
    afs = []
    for r in rows:
        if not r["predicted_bbox_xyxy"]:
            continue
        x1, y1, x2, y2 = r["predicted_bbox_xyxy"]
        w = r.get("image_width")
        h = r.get("image_height")
        if not w or not h:
            continue
        af = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / (w * h)
        afs.append(af)

    print(f"=== analyze {args.ndjson} ===")
    print(f"n_rows        : {n}")
    print(f"n_detected    : {n_detected} ({100 * n_detected / n:.1f}%)")
    print(f"n_zero_det    : {n_zero} ({100 * n_zero / n:.1f}%)")
    if scores:
        print(
            f"score         : mean={statistics.mean(scores):.3f} "
            f"p10={_pct(scores, 10):.3f} p50={_pct(scores, 50):.3f} "
            f"p90={_pct(scores, 90):.3f} min={min(scores):.3f} max={max(scores):.3f}"
        )
        n_low = sum(1 for s in scores if s < args.low_conf)
        print(
            f"low_conf (<{args.low_conf}) : {n_low}/{len(scores)} ({100 * n_low / len(scores):.1f}%)"
        )
    if afs:
        print(
            f"area_fraction : mean={statistics.mean(afs):.4f} "
            f"p10={_pct(afs, 10):.4f} p50={_pct(afs, 50):.4f} "
            f"p90={_pct(afs, 90):.4f} min={min(afs):.4f} max={max(afs):.4f}"
        )
        n_tiny = sum(1 for af in afs if af < args.tiny_area)
        print(
            f"tiny (<{args.tiny_area}) : {n_tiny}/{len(afs)} ({100 * n_tiny / len(afs):.1f}%)"
        )
        bins = Counter(_area_bin(af) for af in afs)
        print("area_fraction bins:")
        order = [
            "tiny<0.005",
            "micro<0.01",
            "<0.05",
            "<0.10",
            "<0.25",
            "<0.50",
            ">=0.50",
        ]
        for k in order:
            c = bins.get(k, 0)
            print(f"  {k:>12}: {c:5d} ({100 * c / len(afs):5.1f}%)")
    # n_detections distribution
    nds = Counter(r.get("n_detections_above_thresh", 0) for r in rows)
    print("n_detections_above_thresh dist (top 10):")
    for k, c in sorted(nds.items())[:10]:
        print(f"  {k}: {c}")


if __name__ == "__main__":
    main()
