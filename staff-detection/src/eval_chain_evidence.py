"""Chain-level tag evidence vs truth, per half of the video, from a cross-fitted auto run.

Needs a run of: staff_detect.py --tag auto --tag-crossfit (each half scored by the model trained on the
other half). For each half, the evidence threshold is chosen on the OTHER half's chains (the half the
scoring model was trained on) and then applied. Truth per chain and half: the staff if the chain owns
the GT chest point on at least half of its labelled frames there. Writes the chosen thresholds next to
the models (tag_classifier_fold{k}.json, and their mean in tag_classifier.json) for staff_detect.py.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from detections import rows_per_frame
from eval_detection import load_gt_points
from fisheye import ANGLE_STEPS, owning_box, to_panorama_point
from tag_model import EVIDENCE_VARIANTS, llr, load_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output folder of staff_detect.py --tag auto --tag-crossfit")
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--no-save", action="store_true", help="do not write the thresholds")
    return parser.parse_args()


def staff_chain_per_frame(detections, chains, gt_points):
    """{labelled frame: chain owning the GT chest point (None if no box owns it)}."""
    frame_rows = rows_per_frame(detections)
    center, radius = tuple(detections["center"]), float(detections["radius"])
    owners = {}
    for frame, point in gt_points.items():
        rows = [r for r in frame_rows[frame] if chains[r] >= 0] if frame < len(frame_rows) else []
        pano = to_panorama_point(*point, center, radius)
        owner = None if pano is None or not rows else owning_box(pano, detections["box"][rows], ANGLE_STEPS)
        owners[frame] = None if owner is None else int(chains[rows[owner]])
    return owners


def chain_halves(detections, chains, ratios, owners, split):
    """Per (chain, half): evidence, frames scored, labelled frames present, of which on the GT point, frame range."""
    stats = defaultdict(lambda: {"evidence": 0.0, "scored": 0, "frames": set()})
    for row, chain in enumerate(chains):
        if chain < 0:
            continue
        frame = int(detections["frame"][row])
        entry = stats[(int(chain), int(frame >= split))]
        entry["frames"].add(frame)
        if not np.isnan(ratios[row]):
            entry["evidence"] += float(ratios[row])
            entry["scored"] += 1
    for (chain, half), entry in stats.items():
        labelled = [f for f in entry["frames"] if f in owners]
        entry["labelled"] = len(labelled)
        entry["on_point"] = sum(owners[f] == chain for f in labelled)
        entry["truth"] = None if not labelled else entry["on_point"] / len(labelled) >= 0.5
    return stats


def choose_threshold(entries):
    """Evidence threshold with the best balanced accuracy on chains of known truth (ties: the higher one)."""
    known = [(e["evidence"], e["truth"]) for e in entries if e["truth"] is not None and e["scored"]]
    staff = [v for v, t in known if t]
    others = [v for v, t in known if not t]
    if not staff or not others:
        return 0.0, float("nan")
    values = sorted({v for v, _ in known})
    candidates = [values[0] - 1] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1]
    best = max(candidates, key=lambda t: (np.mean([v >= t for v in staff]) + np.mean([v < t for v in others]), t))
    accuracy = (np.mean([v >= best for v in staff]) + np.mean([v < best for v in others])) / 2
    return float(best), float(accuracy)


def row_ratios(detections, scores, split, variant):
    """Log-likelihood ratio per detection row: rows in the second half were scored by fold0 (trained on
    the first half) and vice versa, so each uses its own model's table."""
    folds = [load_model(MODEL_DIR / f"tag_classifier_fold{k}.npz") for k in (0, 1)]
    ratios = np.full(len(scores), np.nan)
    for row in np.flatnonzero(~np.isnan(scores)):
        model = folds[0] if detections["frame"][row] >= split else folds[1]
        ratios[row] = llr(model, scores[row], variant)
    return ratios


def evaluate_variant(detections, chains, scores, owners, split, variant, verbose):
    stats = chain_halves(detections, chains, row_ratios(detections, scores, split, variant), owners, split)
    halves = {h: [] for h in (0, 1)}
    for (chain, half), entry in stats.items():
        entry["chain"] = chain
        entry["range"] = f"{min(entry['frames'])}-{max(entry['frames'])}"
        halves[half].append(entry)
    # half 1 is scored by fold0 (trained on half 0): its threshold is chosen on half 0, and vice versa
    thresholds = {0: choose_threshold(halves[0])[0], 1: choose_threshold(halves[1])[0]}
    totals = np.zeros(4, int)
    for half, fold in ((0, 1), (1, 0)):
        name = f"{'First' if half == 0 else 'Second'} half (frames {'<' if half == 0 else '>='} {split})"
        totals += report_half(name, halves[half], thresholds[fold], verbose)
    found, staff, false, others = totals
    balanced = (found / staff + (1 - false / others)) / 2 if staff and others else float("nan")
    return thresholds, totals, balanced


def report_half(name, entries, threshold, verbose=True):
    known = [e for e in entries if e["truth"] is not None and e["scored"]]
    staff = [e for e in known if e["truth"]]
    others = [e for e in known if not e["truth"]]
    found = sum(e["evidence"] >= threshold for e in staff)
    false = sum(e["evidence"] >= threshold for e in others)
    if not verbose:
        return found, len(staff), false, len(others)
    print(f"\n{name}: threshold {threshold:+.2f} (chosen on the other half); chains with labelled frames and scores")
    print(f"{'chain':>7}{'frames':>13}{'scored':>8}{'evidence':>10}{'per frame':>11}  {'truth':<22}decision")
    for e in sorted(known, key=lambda e: -e["evidence"]):
        truth = f"{'STAFF' if e['truth'] else 'other'} ({e['on_point']} of {e['labelled']})"
        decision = "staff" if e["evidence"] >= threshold else "-"
        mark = "" if (e["evidence"] >= threshold) == e["truth"] else "   <- wrong"
        print(f"{e['chain']:>7}{e['range']:>13}{e['scored']:>8}{e['evidence']:>+10.2f}"
              f"{e['evidence'] / e['scored']:>+11.3f}  {truth:<22}{decision}{mark}")
    print(f"  staff chains found {found} of {len(staff)}; other chains called staff {false} of {len(others)}")
    return found, len(staff), false, len(others)


def main():
    args = parse_args()
    run = Path(args.run)
    detections = dict(np.load(run / "detections.npz"))
    chains = np.load(run / "chains.npz")["chain"]
    tags = np.load(run / "tag_scores.npz")
    if not bool(tags["crossfit"]):
        raise SystemExit("This run was not cross-fitted; rerun staff_detect.py with --tag auto --tag-crossfit")
    split = int(tags["split"])
    owners = staff_chain_per_frame(detections, chains, load_gt_points(args.gt))

    print("Evidence variants (thresholds chosen on the other half; held-out chains):")
    results = {}
    for variant in EVIDENCE_VARIANTS:
        thresholds, totals, balanced = evaluate_variant(detections, chains, tags["score"], owners, split, variant, False)
        results[variant] = (thresholds, balanced)
        print(f"  {variant:<17} staff chains found {totals[0]} of {totals[1]}, other chains called staff "
              f"{totals[2]} of {totals[3]}, balanced accuracy {balanced:.2f}")
    variant = max(results, key=lambda v: results[v][1])
    print(f"\nBest: {variant}. Per-chain evidence vs truth:")
    thresholds, totals, _ = evaluate_variant(detections, chains, tags["score"], owners, split, variant, True)
    print(f"\nOverall: staff chains found {totals[0]} of {totals[1]}; other chains called staff {totals[2]} of {totals[3]}")

    if not args.no_save:
        for fold, threshold in thresholds.items():
            (MODEL_DIR / f"tag_classifier_fold{fold}.json").write_text(
                json.dumps({"variant": variant, "threshold": threshold}))
        (MODEL_DIR / "tag_classifier.json").write_text(
            json.dumps({"variant": variant, "threshold": float(np.mean(list(thresholds.values())))}))
        print(f"Saved the variant and thresholds to {MODEL_DIR}/tag_classifier*.json")


if __name__ == "__main__":
    main()
