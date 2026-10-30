"""Evaluate the classical tag score: staff crops (y / n) vs chest crops of other people."""

import argparse
import csv
import heapq

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_detection import ROOT
from detections import rows_per_frame
from tag_crops import EVERY, box_chest_point, interpolate_chest, snap_to_detection, upright_crop
from tag_detect import chest_square, kernel_for_tag, measure_tag_size, tag_score

REVIEW_DIR = ROOT / "outputs" / "tag_review"
OTHER_MIN_DIST = 0.2  # other people: box centre farther than this fraction of the radius from the staff
TOP_OTHERS, BOTTOM_Y = 20, 10
THRESHOLDS = range(120, 251, 10)  # where the scores actually lie (medians 160-210 on the first run)
GROUPS = [  # (key, title, color): categorical slots 1-3 of the dataviz reference palette
    ("y", "staff, tag visible (y)", "#2a78d6"),
    ("n", "staff, tag not visible (n)", "#eb6834"),
    ("others", "other people", "#1baf7a"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--tag-gt", default=str(ROOT / "data" / "tag_gt.csv"))
    parser.add_argument("--detections", default=str(ROOT / "outputs" / "detections.npz"))
    return parser.parse_args()


def load_tag_labels(path):
    with open(path, newline="") as f:
        return {int(row["frame"]): row["label"] for row in csv.DictReader(f)}


# original-scale crops for the staff (with label) and for other people, every EVERY frames
def collect_crops(args, tag_labels):
    chest = interpolate_chest(args.gt)
    detections = dict(np.load(args.detections))
    frame_rows = rows_per_frame(detections)
    center, radius = tuple(detections["center"]), float(detections["radius"])
    wanted = {f for f in chest if f % EVERY == 0 and tag_labels.get(f) in ("y", "n")}

    staff, others = [], []  # (frame, crop, used_fallback) / (frame, crop)
    cap = cv2.VideoCapture(args.video)
    index = 0
    while index <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            guess = np.array(chest[index])
            point, used_fallback = snap_to_detection(guess, frame_rows[index], detections)
            staff.append((index, upright_crop(frame, tuple(point), center, scale=1), used_fallback))
            for row in frame_rows[index]:
                if np.hypot(*(detections["point"][row] - guess)) > OTHER_MIN_DIST * radius:
                    chest_point = box_chest_point(detections["pano_box"][row], center, radius)
                    others.append((index, upright_crop(frame, tuple(chest_point), center, scale=1)))
        index += 1
    cap.release()
    return staff, others


def choose_kernel(staff, tag_labels):
    sizes = [measure_tag_size(crop) for frame, crop, fallback in staff
             if tag_labels[frame] == "y" and not fallback]
    (width, height, used), kernel = kernel_for_tag(sizes)
    print(f"Tag size on {used} of {len(sizes)} 'y' crops (detection-centred, blob < measuring kernel): "
          f"median {width:.0f} x {height:.0f} px -> kernel {kernel} x {kernel}")
    return kernel


def print_distributions(scores, decimals=0):
    print(f"\n{'group':<30}{'count':>7}{'median':>8}{'p90':>7}")
    for key, title, _ in GROUPS:
        values = scores[key]
        print(f"{title:<30}{len(values):>7}{np.median(values):>8.{decimals}f}"
              f"{np.percentile(values, 90):>7.{decimals}f}")


def print_threshold_table(scores, thresholds=THRESHOLDS):
    print(f"\n{'threshold':>9}{'y found':>10}{'staff-n fire':>14}{'others fire':>13}")
    for threshold in thresholds:
        rates = [np.mean(np.array(scores[key]) >= threshold) for key, _, _ in GROUPS]
        print(f"{threshold:>9.2f}{rates[0]:>10.0%}{rates[1]:>14.0%}{rates[2]:>13.0%}")


def plot_histograms(scores, path):
    bins = np.arange(0, 256, 5)
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(9, 7), sharex=True)
    for ax, (key, title, color) in zip(axes, GROUPS):
        ax.hist(scores[key], bins=bins, color=color, edgecolor="#fcfcfb", linewidth=1)
        ax.set_title(f"{title}  (n={len(scores[key])}, median {np.median(scores[key]):.0f})",
                     loc="left", fontsize=10, color="#0b0b0b")
        ax.set_ylabel("crops", color="#52514e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("tag score (max top-hat / black-hat response in the chest square)", color="#52514e")
    fig.suptitle("Tag score by group", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def review_image(crop, score, kind, position, frame):
    scale = 3
    image = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    x0, y0, x1, y1 = chest_square(crop)
    cv2.rectangle(image, (x0 * scale, y0 * scale), (x1 * scale, y1 * scale), (255, 255, 255), 1)
    cv2.drawMarker(image, (position[0] * scale, position[1] * scale), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    text = f"f{frame} {kind} {score}"
    cv2.putText(image, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def save_review(scored_y, scored_others):
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in REVIEW_DIR.glob("*.jpg"):
        old.unlink()
    for rank, (score, kind, position, frame, crop) in enumerate(scored_others, 1):
        path = REVIEW_DIR / f"others_top{rank:02d}_score{score}_f{frame:04d}.jpg"
        cv2.imwrite(str(path), review_image(crop, score, kind, position, frame))
    for rank, (score, kind, position, frame, crop) in enumerate(scored_y, 1):
        path = REVIEW_DIR / f"y_low{rank:02d}_score{score}_f{frame:04d}.jpg"
        cv2.imwrite(str(path), review_image(crop, score, kind, position, frame))


def main():
    args = parse_args()
    tag_labels = load_tag_labels(args.tag_gt)
    staff, others = collect_crops(args, tag_labels)
    kernel = choose_kernel(staff, tag_labels)

    scores = {"y": [], "n": [], "others": []}
    scored_y = []
    for frame, crop, _ in staff:
        score, kind, position = tag_score(crop, kernel)
        scores[tag_labels[frame]].append(score)
        if tag_labels[frame] == "y":
            scored_y.append((score, kind, position, frame, crop))
    scored_others = []
    for frame, crop in others:
        score, kind, position = tag_score(crop, kernel)
        scores["others"].append(score)
        scored_others.append((score, kind, position, frame, crop))
    ignored = sum(1 for label in tag_labels.values() if label == "u")
    print(f"Staff crops: {len(scores['y'])} y, {len(scores['n'])} n ({ignored} u ignored); "
          f"other-people crops: {len(scores['others'])}")

    print_distributions(scores)
    print_threshold_table(scores)
    plot_histograms(scores, ROOT / "outputs" / "tag_scores.png")
    save_review(heapq.nsmallest(BOTTOM_Y, scored_y, key=lambda s: s[0]),
                heapq.nlargest(TOP_OTHERS, scored_others, key=lambda s: s[0]))
    print(f"\nSaved {ROOT / 'outputs' / 'tag_scores.png'} and review crops to {REVIEW_DIR}")


if __name__ == "__main__":
    main()
