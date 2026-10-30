"""Compare one or more staff_detect.py runs (their staff.csv) with the hand labels in data/gt.csv."""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSITION_TOLERANCE = 0.10  # a predicted point is a hit within this fraction of the radius


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="output folders of staff_detect.py")
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="") as f:
        return {int(row["frame"]): row for row in csv.DictReader(f)}


def presence_metrics(gt, pred):
    """Counts over frames that are not "ignore" in the ground truth."""
    tp = fp = fn = tn = 0
    for frame, row in gt.items():
        if row["status"] == "ignore":
            continue
        truth = row["status"] == "present"
        predicted = frame in pred and pred[frame]["present"] == "1"
        tp += truth and predicted
        fp += predicted and not truth
        fn += truth and not predicted
        tn += not truth and not predicted
    total = tp + fp + fn + tn
    return {"frames scored": total, "GT present": tp + fn, "GT absent": fp + tn,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "accuracy": (tp + tn) / total if total else float("nan")}


def position_metrics(gt, pred, radius):
    """Labelled frames: hit if predicted present within POSITION_TOLERANCE x radius of the GT point."""
    distances, labelled = [], 0
    for frame, row in gt.items():
        if row["status"] != "present" or not row["x"]:
            continue
        labelled += 1
        p = pred.get(frame)
        if p is None or p["present"] != "1":
            continue
        distances.append(np.hypot(float(p["x"]) - float(row["x"]), float(p["y"]) - float(row["y"])))
    distances = np.array(distances)
    hits = int(np.sum(distances <= POSITION_TOLERANCE * radius))
    return {"labelled frames": labelled, "predicted on them": len(distances),
            f"hits (<= {POSITION_TOLERANCE:.2f} R)": hits,
            "position hit rate": hits / labelled if labelled else float("nan"),
            "median distance (px)": float(np.median(distances)) if len(distances) else float("nan")}


def evaluate(run, gt):
    run = Path(run)
    radius = float(np.load(run / "detections.npz")["radius"])
    pred = read_rows(run / "staff.csv")
    return {**presence_metrics(gt, pred), **position_metrics(gt, pred, radius)}, radius


def staff_chain_report(run, gt):
    """Per staff chain: frame range, tag hits, and on its labelled frames how often it is on the GT point."""
    run = Path(run)
    detections, chains = np.load(run / "detections.npz"), np.load(run / "chains.npz")
    radius = float(detections["radius"])
    hits = dict(zip(chains["hit_chain"].tolist(), chains["hit_count"].tolist()))
    gt_points = {f: (float(r["x"]), float(r["y"])) for f, r in gt.items() if r["status"] == "present" and r["x"]}
    print(f"\n{run.name}: staff chains")
    print(f"{'chain':>7}{'frames':>14}{'tag hits':>10}{'on GT point (labelled frames)':>32}")
    for chain in chains["staff_chain"].tolist():
        rows = np.flatnonzero(chains["chain"] == chain)
        frames = detections["frame"][rows]
        on_point = labelled = 0
        for frame in sorted(set(frames.tolist()) & set(gt_points)):
            # if the chain has two boxes in a frame, use the more confident one (as the pipeline does)
            in_frame = rows[frames == frame]
            best = in_frame[np.argmax(detections["conf"][in_frame])]
            labelled += 1
            on_point += np.hypot(*(detections["chest"][best] - gt_points[frame])) <= POSITION_TOLERANCE * radius
        print(f"{chain:>7}{f'{frames.min()}-{frames.max()}':>14}{hits[chain]:>10}{f'{on_point} of {labelled}':>32}")


def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}" if value <= 1 else f"{value:.1f}"
    return str(value)


def main():
    args = parse_args()
    gt = read_rows(args.gt)
    results = [evaluate(run, gt) for run in args.runs]
    names = [Path(run).name for run in args.runs]
    print(f"radius: {', '.join(f'{n} {r:.0f} px' for n, (_, r) in zip(names, results))}\n")
    width = max(len(n) for n in names) + 2
    print(f"{'metric':<26}" + "".join(f"{n:>{width}}" for n in names))
    for key in results[0][0]:
        print(f"{key:<26}" + "".join(f"{fmt(r[key]):>{width}}" for r, _ in results))
    for run in args.runs:
        if (Path(run) / "chains.npz").exists():
            staff_chain_report(run, gt)


if __name__ == "__main__":
    main()
