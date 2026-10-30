"""Person detection on the original fisheye frame vs on the unwarped panorama (with seam padding).

    python eval/eval_detection.py

YOLOv8n at the pipeline's settings (imgsz 1440, conf 0.25) on the hand-labelled frames of data/gt.csv:
a hit is a frame where some person box contains the labelled chest point. Also prints the boxes per frame
and the detector time per frame.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detections import detect_view
from fisheye import ANGLE_STEPS, circle_for, point_in_any_box, to_panorama_point
from labels import DATA, ROOT, load_gt_points

IMGSZ = 1440
CONF = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DATA / "sample.mp4"))
    parser.add_argument("--gt", default=str(DATA / "gt.csv"))
    return parser.parse_args()


def read_frames(video_path, wanted):
    """{index: frame} for the wanted frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames, index = {}, 0
    while index <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
        index += 1
    cap.release()
    return frames


def evaluate_view(model, frames, gt_points, circle, use_pano, conf=CONF):
    name = "panorama" if use_pano else "original frame"
    detect_view(model, frames[min(gt_points)], circle, use_pano, IMGSZ, conf)  # warm-up, not timed
    hits, times, box_counts = 0, [], []
    for index, (x, y) in tqdm(sorted(gt_points.items()), desc=name, leave=False):
        _, boxes, elapsed_ms = detect_view(model, frames[index], circle, use_pano, IMGSZ, conf)
        point = to_panorama_point(x, y, *circle) if use_pano else (x, y)
        times.append(elapsed_ms)
        box_counts.append(len(boxes))
        hits += point is not None and point_in_any_box(point, boxes, ANGLE_STEPS if use_pano else None)
    return {"name": name, "frames": len(gt_points), "hits": int(hits),
            "boxes": float(np.mean(box_counts)), "ms": float(np.mean(times))}


def main():
    args = parse_args()
    gt_points = load_gt_points(args.gt)
    frames = read_frames(args.video, set(gt_points) | {0})
    gt_points = {i: p for i, p in gt_points.items() if i in frames}
    circle = circle_for(frames[0])
    print(f"Fisheye circle: center ({circle[0][0]:.0f}, {circle[0][1]:.0f}), radius {circle[1]:.0f}; "
          f"YOLOv8n imgsz {IMGSZ}, conf {CONF}")
    model = YOLO(str(ROOT / "yolov8n.pt"))
    rows = [evaluate_view(model, frames, gt_points, circle, use_pano) for use_pano in (False, True)]
    print(f"\n{'view':<16}{'GT frames':>10}{'hits':>6}{'hit rate':>10}{'boxes/frame':>13}{'ms/frame':>10}")
    for row in rows:
        print(f"{row['name']:<16}{row['frames']:>10}{row['hits']:>6}{row['hits'] / row['frames']:>10.1%}"
              f"{row['boxes']:>13.1f}{row['ms']:>10.1f}")


if __name__ == "__main__":
    main()
