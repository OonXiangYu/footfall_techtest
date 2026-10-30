"""Cut upright crops around the staff member's chest so we can look at the name tag."""

import argparse
import csv
import math

import cv2
import numpy as np

from eval_detection import ROOT, to_original_point
from eval_own_tracker import rows_per_frame

OUT_DIR = ROOT / "outputs" / "tag_sheets"
EVERY = 5
CROP = 160
SCALE = 2
SHEET_COLS, SHEET_ROWS = 8, 5
MAX_SNAP = 0.10  # max distance (fraction of the radius) from the interpolated point to a box centre
CHEST_HEIGHT = 0.30  # chest sits 30% down the box; the panorama box top is the head (rim side)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--detections", default=str(ROOT / "outputs" / "detections.npz"))
    return parser.parse_args()


# chest position for every present frame, interpolated between labelled points
def interpolate_chest(gt_path):
    with open(gt_path, newline="") as f:
        rows = list(csv.DictReader(f))
    present = [int(r["frame"]) for r in rows if r["status"] == "present"]
    labelled = [(int(r["frame"]), int(r["x"]), int(r["y"]))
                for r in rows if r["status"] == "present" and r["x"] and r["y"]]
    frames, xs, ys = zip(*labelled)
    # outside the first/last labelled frame there is nothing to interpolate between
    inside = [f for f in present if frames[0] <= f <= frames[-1]]
    return {f: (float(np.interp(f, frames, xs)), float(np.interp(f, frames, ys))) for f in inside}


# chest point of a panorama box, in original-image coordinates
def box_chest_point(box, center, radius):
    x1, y1, x2, y2 = box
    return np.array(to_original_point((x1 + x2) / 2, y1 + CHEST_HEIGHT * (y2 - y1), center, radius))


# chest point from the detection box nearest the interpolated point; returns (point, used_fallback)
def snap_to_detection(guess, rows, detections):
    center, radius = detections["center"], float(detections["radius"])
    if len(rows):
        distances = np.hypot(*(detections["point"][rows] - guess).T)
        nearest = distances.argmin()
        if distances[nearest] <= MAX_SNAP * radius:
            return box_chest_point(detections["pano_box"][rows[nearest]], center, radius), False
    return guess, True


# 160x160 crop centred on the point, rotated so the centre->point direction points up
def upright_crop(frame, point, center, scale=SCALE):
    # direction angle measured clockwise on screen (image y points down); up is -90 degrees.
    # getRotationMatrix2D turns counter-clockwise on screen, so turning by angle + 90 sends it up.
    angle = math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))
    matrix = cv2.getRotationMatrix2D(point, angle + 90, 1.0)
    matrix[:, 2] += (CROP / 2 - point[0], CROP / 2 - point[1])  # move the point to the crop centre
    crop = cv2.warpAffine(frame, matrix, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    if scale == 1:
        return crop
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def label(crop, index, used_fallback):
    cv2.putText(crop, str(index), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(crop, str(index), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if used_fallback:
        cv2.putText(crop, "interp", (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(crop, "interp", (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    return crop


def save_sheet(crops, number):
    per_sheet = SHEET_COLS * SHEET_ROWS
    crops = crops + [np.zeros_like(crops[0])] * (per_sheet - len(crops))
    rows = [np.hstack(crops[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
    cv2.imwrite(str(OUT_DIR / f"sheet_{number:02d}.jpg"), np.vstack(rows))


# read the video sequentially and yield (frame index, upright crop, used_fallback) every EVERY frames
def iter_crops(video_path, gt_path, detections_path, scale=SCALE):
    chest = interpolate_chest(gt_path)
    last = max(f for f in chest if f % EVERY == 0)
    detections = dict(np.load(detections_path))
    frame_rows = rows_per_frame(detections)
    center = tuple(detections["center"])
    cap = cv2.VideoCapture(video_path)
    index = 0
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in chest and index % EVERY == 0:
            point, used_fallback = snap_to_detection(np.array(chest[index]), frame_rows[index], detections)
            yield index, upright_crop(frame, tuple(point), center, scale), used_fallback
        index += 1
    cap.release()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("sheet_*.jpg"):
        old.unlink()

    crops, sheets, total, fallbacks = [], 0, 0, 0
    for index, crop, used_fallback in iter_crops(args.video, args.gt, args.detections):
        crops.append(label(crop, index, used_fallback))
        total += 1
        fallbacks += used_fallback
        if len(crops) == SHEET_COLS * SHEET_ROWS:
            save_sheet(crops, sheets)
            crops, sheets = [], sheets + 1
    if crops:
        save_sheet(crops, sheets)
        sheets += 1
    print(f"Wrote {total} crops on {sheets} sheet(s) to {OUT_DIR}")
    print(f"{total - fallbacks} crops centred on a detection box, {fallbacks} on the interpolated point")


if __name__ == "__main__":
    main()
