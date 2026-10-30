import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from detections import choose_view, detect_view
from fisheye import ANGLE_STEPS, fallback_circle, find_fisheye_circle, point_in_any_box, to_panorama_point

ROOT = Path(__file__).resolve().parent.parent
MISSES_DIR = ROOT / "outputs" / "misses"
PROBE_FRAMES = 50
DEFAULT_IMGSZ = 1440
DEFAULT_CONF = 0.25
SETTINGS = [(640, 0.25), (1440, 0.25), (1440, 0.10), (1600, 0.25)]  # (imgsz, conf) for compare


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--unwarp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--compare", action="store_true",
                        help="evaluate every entry in SETTINGS instead of --imgsz/--conf")
    return parser.parse_args()

# read from gt.csv for those for eval
def load_gt_points(path):
    with open(path, newline="") as f:
        return {int(row["frame"]): (int(row["x"]), int(row["y"]))
                for row in csv.DictReader(f)
                if row["status"] == "present" and row["x"] and row["y"]}

# read the frames and just save thos needed
def read_frames(video_path, wanted):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frames, index = {}, 0
    with tqdm(total=max(wanted) + 1, desc="Reading") as progress:
        while index <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = frame
            index += 1
            progress.update(1)
    cap.release()
    return frames

# save those frame who miss with detail
def save_miss(path, image, boxes, points):
    image = image.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y in points:
        cv2.drawMarker(image, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.imwrite(str(path), image)

# evalution
def evaluate_setting(model, frames, gt_points, circle, use_pano, imgsz, conf):
    name = f"{'pano' if use_pano else 'orig'}_imgsz{imgsz}_conf{conf:.2f}"
    miss_dir = MISSES_DIR / name
    miss_dir.mkdir(parents=True, exist_ok=True)
    for old in miss_dir.glob("*.jpg"):
        old.unlink()

    detect_view(model, frames[min(gt_points)], circle, use_pano, imgsz, conf)  # warm-up, not timed
    pano_width = ANGLE_STEPS if use_pano else None

    hits, times, box_counts, outside = 0, [], [], 0
    for index, (x, y) in tqdm(sorted(gt_points.items()), desc=name, leave=False):
        image, boxes, elapsed_ms = detect_view(model, frames[index], circle, use_pano, imgsz, conf)
        point = to_panorama_point(x, y, *circle) if use_pano else (x, y)
        times.append(elapsed_ms)
        box_counts.append(len(boxes))
        if point is None:
            outside += 1
        if point is not None and point_in_any_box(point, boxes, pano_width):
            hits += 1
            continue
        marks = [] if point is None else [point]
        if point is not None and use_pano and point[0] + pano_width < image.shape[1]:
            marks.append((point[0] + pano_width, point[1]))  # same point inside the seam padding
        save_miss(miss_dir / f"frame_{index:04d}.jpg", image, boxes, marks)
    if outside:
        print(f"  {name}: {outside} GT point(s) lie outside the unwarp circle (counted as misses)")
    return {"name": name, "frames": len(gt_points), "hits": hits,
            "mean_boxes": float(np.mean(box_counts)), "mean_ms": float(np.mean(times))}


def print_table(rows):
    print(f"\n{'setting':<26}{'GT frames':>10}{'hits':>6}{'hit rate':>10}{'boxes/frame':>13}{'ms/frame':>10}")
    for row in rows:
        print(f"{row['name']:<26}{row['frames']:>10}{row['hits']:>6}"
              f"{row['hits'] / row['frames']:>10.1%}{row['mean_boxes']:>13.1f}{row['mean_ms']:>10.1f}")


def main():
    args = parse_args()
    gt_points = load_gt_points(args.gt)
    if not gt_points:
        raise SystemExit(f"No present frames with a point in {args.gt}")
    frames = read_frames(args.video, set(gt_points) | set(range(PROBE_FRAMES)))
    gt_points = {i: p for i, p in gt_points.items() if i in frames}

    circle = find_fisheye_circle(frames[0])
    if circle:
        print(f"Fisheye circle: center ({circle[0][0]:.1f}, {circle[0][1]:.1f}), radius {circle[1]:.1f}")
    else:
        circle = fallback_circle(frames[0])
        print(f"Fisheye circle: none found, panorama falls back to frame center "
              f"({circle[0][0]:.0f}, {circle[0][1]:.0f}), radius {circle[1]:.0f}")

    model = YOLO(str(ROOT / "yolov8n.pt"))
    probe_frames = [frames[i] for i in range(PROBE_FRAMES) if i in frames]
    rows = []
    for imgsz, conf in SETTINGS if args.compare else [(args.imgsz, args.conf)]:
        print(f"imgsz {imgsz} / conf {conf}")
        use_pano = choose_view(args.unwarp, model, probe_frames, circle, imgsz, conf)
        rows.append(evaluate_setting(model, frames, gt_points, circle, use_pano, imgsz, conf))
    print_table(rows)
    print(f"Missed frames saved under {MISSES_DIR}")


if __name__ == "__main__":
    main()
