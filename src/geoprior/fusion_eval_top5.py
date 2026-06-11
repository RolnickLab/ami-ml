#!/usr/bin/env python3
"""
Path A — Top-5 fusion eval.

For each image in the cached CNN predictions CSV, we have:
  - the ground-truth species_name
  - the CNN's predicted top-1 (species_name + score)
  - the CNN's top-5 (species_name pipe-separated + scores pipe-separated)

We don't have full 8,851-class probability vectors, so we approximate fusion
*within* the top-5 candidates:
  1. Look up geo-prior probability for each of the 5 candidates (via species
     -> geoprior class_id, from geoprior_categ_map.json).
  2. Apply replace_value=1.0 for any candidate species missing from the
     geoprior class space (the 7 max1000 species with zero geocoded data).
  3. If valid=True for this image: fused_i = cnn_i * prior_i; else: fused_i = cnn_i.
  4. New top-1 = argmax over the 5 fused scores.

This captures fusion's effect when CNN is uncertain enough that the right
answer is in its top-5 but maybe at rank 2-5. It does NOT capture cases
where the right answer is at rank 6+ — for that we'd need full inference
(Path B).
"""
import csv
import json
import os
import time

import numpy as np

from src.geoprior import config


def load_categ_map(path):
    """Load {species_name: class_id} mapping."""
    with open(path) as f:
        return json.load(f)


def load_prior_for_gbif(prior_dir, gbif_id):
    """Returns (prior_vector, valid_flag) or (None, None) if file missing."""
    p_path = os.path.join(prior_dir, "preds", f"{gbif_id}.npy")
    v_path = os.path.join(prior_dir, "valid", f"{gbif_id}.npy")
    if not os.path.exists(p_path):
        return None, None
    return np.load(p_path), np.load(v_path)


def eval_split(split, csv_path, prior_dir, geoprior_categ_map, max_rows=None):
    print(f"\n===== {split.upper()} =====")
    print(f"  CSV:   {csv_path}")
    print(f"  Prior: {prior_dir}")

    n_total = 0
    n_baseline_correct = 0
    n_fused_correct = 0
    n_top5_hit = 0
    n_flipped_to_correct = 0
    n_flipped_to_wrong = 0
    n_no_prior_file = 0
    n_valid = 0
    n_unmapped_species_in_top5 = 0  # candidates not in geoprior

    # for macro-F1
    tp_baseline = {}
    fp_baseline = {}
    fn_baseline = {}
    tp_fused = {}
    fp_fused = {}
    fn_fused = {}

    t0 = time.time()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gbif_id = row["gbif_id"]
            gt_sp = row["species_name"]
            baseline_pred = row["predicted_species"]
            top5_sp = row["top5_species"].split("|")
            top5_scores = np.array(
                [float(x) for x in row["top5_scores"].split("|")], dtype=np.float32
            )

            # baseline
            n_total += 1
            if baseline_pred == gt_sp:
                n_baseline_correct += 1
            if gt_sp in top5_sp:
                n_top5_hit += 1

            # load prior
            prior_vec, valid = load_prior_for_gbif(prior_dir, gbif_id)
            if prior_vec is None:
                # no prior file (image's gbif_id wasn't in val.json — no geocoded match)
                n_no_prior_file += 1
                fused_pred = baseline_pred
            else:
                # gather prior values for top-5
                prior_for_top5 = np.empty(5, dtype=np.float32)
                for i, sp in enumerate(top5_sp):
                    cid = geoprior_categ_map.get(sp)
                    if cid is None:
                        prior_for_top5[i] = 1.0  # missing species → no info
                        n_unmapped_species_in_top5 += 1
                    else:
                        prior_for_top5[i] = prior_vec[cid]
                # apply fusion
                if valid > 0.5:
                    n_valid += 1
                    fused = top5_scores * prior_for_top5
                else:
                    fused = top5_scores.copy()
                # renormalize within top-5
                s = fused.sum()
                if s > 0:
                    fused = fused / s
                fused_pred = top5_sp[int(np.argmax(fused))]

            if fused_pred == gt_sp:
                n_fused_correct += 1
            if fused_pred != baseline_pred:
                if fused_pred == gt_sp:
                    n_flipped_to_correct += 1
                elif baseline_pred == gt_sp:
                    n_flipped_to_wrong += 1

            # macro-F1 accounting (per-class)
            def bump(d, key):
                d[key] = d.get(key, 0) + 1

            if baseline_pred == gt_sp:
                bump(tp_baseline, gt_sp)
            else:
                bump(fp_baseline, baseline_pred)
                bump(fn_baseline, gt_sp)
            if fused_pred == gt_sp:
                bump(tp_fused, gt_sp)
            else:
                bump(fp_fused, fused_pred)
                bump(fn_fused, gt_sp)

            if max_rows and n_total >= max_rows:
                break

    elapsed = time.time() - t0

    def macro_f1(tp, fp, fn):
        # F1 = 2TP / (2TP + FP + FN), per class, averaged
        classes = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
        f1s = []
        for c in classes:
            t = tp.get(c, 0)
            p = fp.get(c, 0)
            n = fn.get(c, 0)
            denom = 2 * t + p + n
            f1s.append(2 * t / denom if denom else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0

    baseline_acc = n_baseline_correct / n_total
    fused_acc = n_fused_correct / n_total
    top5_hit = n_top5_hit / n_total
    delta = fused_acc - baseline_acc

    print(f"  Rows processed:     {n_total:,}")
    print(f"  Time:               {elapsed:.1f}s  ({n_total/elapsed:.0f} rows/s)")
    print(f"  Valid (had geo):    {n_valid:,}  ({100*n_valid/n_total:.2f}%)")
    print(f"  Missing prior file: {n_no_prior_file:,}")
    print(
        f"  Unmapped species (top-5 candidate ∉ geoprior): {n_unmapped_species_in_top5:,}"
    )
    print()
    print(f"  Baseline top-1:   {baseline_acc:.4%}  ({n_baseline_correct:,} correct)")
    print(f"  Fused    top-1:   {fused_acc:.4%}  ({n_fused_correct:,} correct)")
    print(f"  Δ top-1:          {delta:+.4%}  ({delta*100:+.2f} pp)")
    print(f"  Top-5 ceiling:    {top5_hit:.4%}  (GT in CNN top-5)")
    print()
    print("  Flips (CNN→fused changed prediction):")
    print(f"    to correct:   {n_flipped_to_correct:,}")
    print(f"    to wrong:     {n_flipped_to_wrong:,}")
    print(f"    net flips:    {n_flipped_to_correct - n_flipped_to_wrong:+,}")
    print()
    print(f"  Macro-F1 baseline: {macro_f1(tp_baseline, fp_baseline, fn_baseline):.4f}")
    print(f"  Macro-F1 fused:    {macro_f1(tp_fused, fp_fused, fn_fused):.4f}")

    return {
        "n_total": n_total,
        "baseline_acc": baseline_acc,
        "fused_acc": fused_acc,
        "delta": delta,
        "top5_hit": top5_hit,
        "flips_correct": n_flipped_to_correct,
        "flips_wrong": n_flipped_to_wrong,
    }


def main():
    geoprior_categ_map = load_categ_map(config.CATEG_MAP_PATH)
    print(f"Loaded geoprior_categ_map: {len(geoprior_categ_map):,} species -> class_id")

    res_val = eval_split(
        "val",
        str(config.CLF_VAL_PREDS),
        str(config.GEOPRIOR_VAL_PREDS),
        geoprior_categ_map,
    )
    res_test = eval_split(
        "test",
        str(config.CLF_TEST_PREDS),
        str(config.GEOPRIOR_TEST_PREDS),
        geoprior_categ_map,
    )

    print("\n===== SUMMARY =====")
    bva, fva, dva = res_val["baseline_acc"], res_val["fused_acc"], res_val["delta"]
    bte, fte, dte = res_test["baseline_acc"], res_test["fused_acc"], res_test["delta"]
    print(f"  baseline (val):   {bva:.4%}")
    print(f"  fused    (val):   {fva:.4%}  (delta = {dva:+.4%})")
    print(f"  baseline (test):  {bte:.4%}")
    print(f"  fused    (test):  {fte:.4%}  (delta = {dte:+.4%})")


if __name__ == "__main__":
    main()
