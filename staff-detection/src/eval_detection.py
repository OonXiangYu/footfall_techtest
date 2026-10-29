"""Measure how often YOLO finds the staff member at the hand-labelled points in gt.csv."""

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MISSES_DIR = ROOT / "outputs" / "misses"
ANGLE_STEPS = 1440  # panorama width: one column per 0.25 degree
PROBE_FRAMES = 50
SETTINGS = [(640, 0.25), (1440, 0.25), (1440, 0.10)]  # (imgsz, conf)
BORDER_THRESHOLD = 20
SEAM_PAD = 0.10  # fraction of panorama columns repeated past the right edge
SEAM_OVERLAP = 0.6  # overlap (intersection / smaller box) above which two seam boxes are one person


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--unwarp", choices=["auto", "on", "off"], default="auto")
    return parser.parse_args()


def load_gt_points(path):
    """Frames that are "present" and have a clicked point -> (x, y)."""
    with open(path, newline="") as f:
        return {int(row["frame"]): (int(row["x"]), int(row["y"]))
                for row in csv.DictReader(f)
                if row["status"] == "present" and row["x"] and row["y"]}


def read_frames(video_path, wanted):
    """Read the video sequentially and keep only the frames whose index is in `wanted`."""
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


def find_fisheye_circle(frame):
    """Return (center, radius) of the fisheye image circle, or None if there is no clear one."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = (gray > BORDER_THRESHOLD).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))

    circle = np.zeros_like(bright)
    cv2.circle(circle, (round(cx), round(cy)), round(radius), 255, -1)
    outside = circle == 0
    # Without a real dark border the "circle" just encloses the whole frame (normal camera,
    # or a fisheye whose image circle is cropped by the frame).
    if outside.mean() < 0.05 or (bright[outside] == 0).mean() < 0.9:
        return None
    # The bright region should fill most of the circle that is inside the frame.
    if (bright[~outside] > 0).mean() < 0.7:
        return None
    return (cx, cy), radius


def fallback_circle(frame):
    """Frame center, radius reaching the left and right edges."""
    h, w = frame.shape[:2]
    return (w / 2, h / 2), max(w, h) / 2


def unwarp(frame, center, radius):
    """Fisheye -> panorama: angle along x, distance from the center along y (rim at the top)."""
    polar = cv2.warpPolar(frame, (round(radius), ANGLE_STEPS), center, radius, cv2.WARP_POLAR_LINEAR)
    return cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)


def to_panorama_point(x, y, center, radius):
    """Map an original-image point to panorama coordinates, or None if outside the circle.

    warpPolar puts angle on the rows and distance on the columns:
        polar_row = theta * ANGLE_STEPS / (2*pi),   polar_col = rho * polar_width / radius
    with theta = atan2(y - cy, x - cx). Because image y points down, increasing theta turns
    clockwise on screen, which is also the direction warpPolar uses (checked by warping a
    single bright dot and finding it in the output). ROTATE_90_COUNTERCLOCKWISE then maps
    polar (row, col) to panorama (x = row, y = polar_width - 1 - col).
    """
    dx, dy = x - center[0], y - center[1]
    rho = math.hypot(dx, dy)
    if rho >= radius:
        return None
    theta = math.atan2(dy, dx) % (2 * math.pi)
    polar_width = round(radius)
    pano_x = theta * ANGLE_STEPS / (2 * math.pi)
    pano_y = polar_width - 1 - rho * polar_width / radius
    return pano_x, pano_y


def detect(model, image, imgsz, conf):
    """Person boxes as an (N, 4) xyxy array, plus the wall-clock time of the model call in ms."""
    start = time.perf_counter()
    result = model(image, classes=[0], imgsz=imgsz, conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result.boxes.xyxy.cpu().numpy(), elapsed_ms


def pad_seam(pano):
    """Repeat the first SEAM_PAD of the columns after the right edge, so people on the seam are whole."""
    pad = round(pano.shape[1] * SEAM_PAD)
    return np.hstack([pano, pano[:, :pad]])


def overlap(a, b):
    """Intersection area over the smaller box's area."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller


def merge_seam_boxes(boxes, pano_width):
    """Fold detections from the seam padding back onto the panorama.

    A box entirely inside the padding is shifted back by pano_width, unless the same person
    was already found at the left edge. A box that crosses the seam is the only complete view
    of that person, so it is kept as is (x2 > pano_width) and its truncated copy at the left
    edge is dropped.
    """
    inside = [b for b in boxes if b[2] <= pano_width]
    merged = []
    for box in (b for b in boxes if b[2] > pano_width):
        shifted = box - (pano_width, 0, pano_width, 0)
        if box[0] >= pano_width:
            if not any(overlap(b, shifted) > SEAM_OVERLAP for b in inside):
                merged.append(shifted)
        else:
            inside = [b for b in inside if overlap(b, shifted) <= SEAM_OVERLAP]
            merged.append(box)
    return np.array(inside + merged, dtype=float).reshape(-1, 4)


def detect_view(model, frame, circle, use_pano, imgsz, conf):
    """Run detection on the panorama (seam-padded) or the original frame.

    Returns the image the boxes refer to, the boxes, and the model time in ms.
    """
    if not use_pano:
        boxes, elapsed_ms = detect(model, frame, imgsz, conf)
        return frame, boxes, elapsed_ms
    pano = unwarp(frame, *circle)
    padded = pad_seam(pano)
    boxes, elapsed_ms = detect(model, padded, imgsz, conf)
    return padded, merge_seam_boxes(boxes, pano.shape[1]), elapsed_ms


def count_people(model, frames, circle, use_pano, imgsz, conf):
    return sum(len(detect_view(model, f, circle, use_pano, imgsz, conf)[1]) for f in frames)


def choose_view(mode, model, probe_frames, circle, imgsz, conf):
    """Return True to run detection on the panorama, False for the original frame."""
    if mode != "auto":
        return mode == "on"
    on_pano = count_people(model, probe_frames, circle, True, imgsz, conf)
    on_original = count_people(model, probe_frames, circle, False, imgsz, conf)
    use_pano = on_pano > on_original
    print(f"  auto: first {len(probe_frames)} frames -> {on_pano} people on panorama, "
          f"{on_original} on original -> {'panorama' if use_pano else 'original frame'}")
    return use_pano


def point_in_any_box(point, boxes, wrap_width=None):
    """wrap_width: panorama width, so a point near x=0 also matches boxes that cross the seam."""
    candidates = [point] if wrap_width is None else [point, (point[0] + wrap_width, point[1])]
    return any(x1 <= x <= x2 and y1 <= y <= y2
               for x, y in candidates for x1, y1, x2, y2 in boxes)


def save_miss(path, image, boxes, points):
    image = image.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y in points:
        cv2.drawMarker(image, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.imwrite(str(path), image)


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
    for imgsz, conf in SETTINGS:
        print(f"imgsz {imgsz} / conf {conf}")
        use_pano = choose_view(args.unwarp, model, probe_frames, circle, imgsz, conf)
        rows.append(evaluate_setting(model, frames, gt_points, circle, use_pano, imgsz, conf))
    print_table(rows)
    print(f"Missed frames saved under {MISSES_DIR}")


if __name__ == "__main__":
    main()
