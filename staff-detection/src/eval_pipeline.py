"""Compare one or more staff_detect.py runs (their staff.csv) with the hand labels in data/gt.csv."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from detections import rows_per_frame
from fisheye import ANGLE_STEPS, owning_box, pad_seam, to_panorama_point, unwarp

ROOT = Path(__file__).resolve().parent.parent
POSITION_TOLERANCE = 0.10  # a predicted point is a hit within this fraction of the radius
COVERAGE_SAVED = 4  # missed frames saved as images per run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="output folders of staff_detect.py")
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--coverage", nargs=2, type=int, metavar=("START", "END"),
                        help="also check detection coverage on labelled frames START..END: does any box "
                             "own the GT point? (e.g. 0 315 for the seated stretch)")
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"),
                        help="video, for the images of --coverage misses")
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
    # auto runs store summed evidence (log-likelihood ratio) instead of hit counts
    evidence = dict(zip(chains["hit_chain"].tolist(), chains["evidence"].tolist())) if "evidence" in chains else {}
    # auto runs also store how many frames of each chain were scored, so the hit rate can be shown
    scored = dict(zip(chains["hit_chain"].tolist(), chains["scored_count"].tolist())) if "scored_count" in chains else {}
    gt_points = {f: (float(r["x"]), float(r["y"])) for f, r in gt.items() if r["status"] == "present" and r["x"]}
    print(f"\n{run.name}: staff chains")
    print(f"{'chain':>7}{'frames':>14}{'evidence' if evidence else 'tag hits':>10}{'scored':>8}"
          f"{'on GT point (labelled frames)':>32}")
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
        scored_text = str(scored[chain]) if scored else "-"
        tag_text = f"{evidence[chain]:+.2f}" if evidence else str(hits[chain])
        print(f"{chain:>7}{f'{frames.min()}-{frames.max()}':>14}{tag_text:>10}{scored_text:>8}"
              f"{f'{on_point} of {labelled}':>32}")


def coverage_check(run, gt, start, end, video):
    """On labelled frames start..end, does any cached box own the GT point? Saves a few misses."""
    run = Path(run)
    detections = dict(np.load(run / "detections.npz"))
    frame_rows = rows_per_frame(detections)
    center, radius = tuple(detections["center"]), float(detections["radius"])
    use_pano = bool(detections["use_pano"])
    points = {f: (float(r["x"]), float(r["y"])) for f, r in gt.items()
              if start <= f <= end and f < len(frame_rows) and r["status"] == "present" and r["x"]}
    missed = []
    for frame, (x, y) in sorted(points.items()):
        point = to_panorama_point(x, y, center, radius) if use_pano else (x, y)
        owner = None if point is None else owning_box(
            point, detections["box"][frame_rows[frame]], ANGLE_STEPS if use_pano else None)
        if owner is None:
            missed.append(frame)
    print(f"\n{run.name}: coverage {start}-{end}: a box owns the GT point on "
          f"{len(points) - len(missed)} of {len(points)} labelled frames; missed {missed or 'none'}")
    if missed:
        save_coverage_misses(run / "coverage_misses", missed[::max(1, len(missed) // COVERAGE_SAVED)][:COVERAGE_SAVED],
                             points, detections, frame_rows, video)


def save_coverage_misses(out, frames, points, detections, frame_rows, video):
    """Original frame with the GT point, above the panorama with the boxes and the GT point."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    center, radius = tuple(detections["center"]), float(detections["radius"])
    cap = cv2.VideoCapture(video)
    for index in range(max(frames) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if index not in frames:
            continue
        x, y = (round(v) for v in points[index])
        original = frame.copy()
        cv2.drawMarker(original, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.putText(original, f"frame {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        view = pad_seam(unwarp(frame, center, radius)) if detections["use_pano"] else frame.copy()
        for x1, y1, x2, y2 in detections["box"][frame_rows[index]].astype(int):
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        point = to_panorama_point(x, y, center, radius) if detections["use_pano"] else (x, y)
        if point is not None:
            cv2.drawMarker(view, (round(point[0]), round(point[1])), (0, 0, 255), cv2.MARKER_CROSS, 28, 3)
        width = max(original.shape[1], view.shape[1])
        pad = lambda image: cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out / f"frame_{index:04d}.jpg"), np.vstack([pad(original), pad(view)]))
    cap.release()
    print(f"  saved {sorted(frames)} to {out}")


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
    if args.coverage:
        for run in args.runs:
            coverage_check(run, gt, *args.coverage, args.video)


if __name__ == "__main__":
    main()
