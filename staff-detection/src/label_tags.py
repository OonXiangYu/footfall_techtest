"""Hand-label name-tag visibility (y / n / u) on the upright crops from tag_crops.py."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from eval_detection import ROOT
from tag_crops import EVERY, iter_crops

WINDOW = "label tags"
DISPLAY_SIZE = 720
LABEL_KEYS = {ord("y"): "y", ord("n"): "n", ord("u"): "u"}
DENSE_MARGIN = 10  # frames around each "y" run for --dense
SPARSE_LABELS = ROOT / "data" / "tag_gt.csv"
LABEL_COLORS = {"y": (0, 255, 0), "n": (0, 0, 255), "u": (0, 200, 255), None: (200, 200, 200)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--detections", default=str(ROOT / "outputs" / "detections.npz"))
    parser.add_argument("--out", help="labels file (default data/tag_gt.csv, or data/tag_gt_dense.csv with --dense)")
    parser.add_argument("--dense", action="store_true",
                        help=f"label every frame within +-{DENSE_MARGIN} frames of each 'y' run in data/tag_gt.csv")
    args = parser.parse_args()
    if args.out is None:
        args.out = str(ROOT / "data" / ("tag_gt_dense.csv" if args.dense else "tag_gt.csv"))
    return args


def load_labels(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        labels = {int(row["frame"]): row["label"] for row in csv.DictReader(f)}
    print(f"Resumed from {path}: {len(labels)} labels")
    return labels


def save_labels(path, labels):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        for frame in sorted(labels):
            writer.writerow([frame, labels[frame]])
    print(f"Saved {path}")


# consecutive y frames (EVERY apart) joined into "start-end" strings
def runs(frames):
    groups = []
    for frame in sorted(frames):
        if groups and frame - groups[-1][-1] == EVERY:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return [f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups]


def y_runs(labels):
    """Runs of "y" frames in the every-EVERY-frames labels, as [first, last] lists."""
    groups = []
    for frame in sorted(f for f, label in labels.items() if label == "y"):
        if groups and frame - groups[-1][-1] == EVERY:
            groups[-1][-1] = frame
        else:
            groups.append([frame, frame])
    return groups


def dense_frames(sparse_labels):
    """Every frame within DENSE_MARGIN frames of a "y" run."""
    frames = set()
    for first, last in y_runs(sparse_labels):
        frames.update(range(max(0, first - DENSE_MARGIN), last + DENSE_MARGIN + 1))
    return frames


def render(crop, index, position, count, current, used_fallback):
    image = cv2.resize(crop, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
    lines = [(f"frame {index}  [{position + 1}/{count}]", (255, 255, 255)),
             (f"label: {current or '-'}", LABEL_COLORS[current])]
    if used_fallback:
        lines.append(("interp (crop may be off)", (0, 165, 255)))
    for row, (text, color) in enumerate(lines):
        origin = (12, 36 + row * 36)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return image


def first_unlabelled(crops, labels):
    return next((i for i, (index, _, _) in enumerate(crops) if index not in labels), 0)


def main():
    args = parse_args()
    frames = None
    labels = load_labels(args.out)
    if args.dense:
        sparse = load_labels(SPARSE_LABELS)
        frames = dense_frames(sparse)
        if not labels:
            # start from the every-5-frames labels that fall inside the dense ranges
            labels = {f: label for f, label in sparse.items() if f in frames}
            print(f"Dense mode: {len(y_runs(sparse))} y runs -> {len(frames)} frames; "
                  f"{len(labels)} pre-filled from {SPARSE_LABELS.name}")
    print("Cutting crops...")
    crops = list(iter_crops(args.video, args.gt, args.detections, frames=frames))
    position = first_unlabelled(crops, labels)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while True:
        index, crop, used_fallback = crops[position]
        cv2.imshow(WINDOW, render(crop, index, position, len(crops), labels.get(index), used_fallback))
        key = cv2.waitKey(30) & 0xFF
        # closing the window with the mouse is treated like q so no work is lost
        if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in LABEL_KEYS:
            labels[index] = LABEL_KEYS[key]
            position = min(position + 1, len(crops) - 1)
        elif key == ord("d"):
            position = min(position + 1, len(crops) - 1)
        elif key == ord("a"):
            position = max(position - 1, 0)
    cv2.destroyAllWindows()

    save_labels(args.out, labels)
    counts = Counter(labels.values())
    print(f"y: {counts['y']}  n: {counts['n']}  u: {counts['u']}  "
          f"(unlabelled: {len(crops) - len(labels)})")
    print("Tag visible (y) runs:", ", ".join(runs(f for f, label in labels.items() if label == "y")) or "none")


if __name__ == "__main__":
    main()
