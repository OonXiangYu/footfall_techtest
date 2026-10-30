"""Click the name-tag centre on upright staff crops (tag visible, staff has a pose) -> data/tag_pos.csv.

Keys: left click = tag centre (moves on), s = skip, a / d = previous / next, q = save and quit.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from crops import CROP, upright_matrix
from eval_detection import ROOT
from pose import (LEFT_HIP, LEFT_SHOULDER, POSE_WEIGHTS, RIGHT_HIP, RIGHT_SHOULDER, body_frame, from_body,
                  keypoints_to_original, run_pose, staff_owner)
from tag_crops import interpolate_chest, load_tag_labels

WINDOW = "label tag position"
SCALE = 4
CROP_CENTRE = (0.0, 0.3)  # body coordinates (shoulder widths): just below the shoulder midpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--tag-labels", default=str(ROOT / "data" / "tag_gt.csv"))
    parser.add_argument("--detections", default=str(ROOT / "outputs" / "detections.npz"))
    parser.add_argument("--out", default=str(ROOT / "data" / "tag_pos.csv"))
    return parser.parse_args()


def staff_poses(args):
    """For each "y" frame where the staff member has a usable pose (the same rule as tag_window.py):
    (frame, upright crop, crop matrix, keypoints in original coords, keypoint confidences)."""
    tag_labels = load_tag_labels(args.tag_labels)
    chest = interpolate_chest(args.gt)
    frames = {f for f, label in tag_labels.items() if label == "y" and f in chest}
    detections = np.load(args.detections)
    center, radius = tuple(detections["center"]), float(detections["radius"])
    model = YOLO(str(ROOT / POSE_WEIGHTS))
    items = []
    cap = cv2.VideoCapture(args.video)
    for index in range(max(frames) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if index not in frames:
            continue
        boxes, keypoints, confs = run_pose(model, frame, center, radius)
        owner = staff_owner(boxes, chest[index], center, radius)
        if owner is None:
            continue
        keypoints = keypoints_to_original(keypoints[owner], center, radius)
        body = body_frame(keypoints, confs[owner])
        if body is None:
            continue
        # centre on the pose's chest, so crops without a detection-based centre look the same
        matrix = upright_matrix(from_body(CROP_CENTRE, body), center)
        crop = cv2.warpAffine(frame, matrix, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        items.append((index, crop, matrix, keypoints, confs[owner]))
    cap.release()
    return items


def load_positions(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        positions = {int(r["frame"]): (r["status"], r["x"], r["y"]) for r in csv.DictReader(f)}
    print(f"Resumed from {path}: {len(positions)} entries")
    return positions


def save_positions(path, positions):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "status", "x", "y"])
        for frame in sorted(positions):
            writer.writerow([frame, *positions[frame]])
    print(f"Saved {path}")


def to_crop(point, matrix):
    x, y = matrix @ np.array([point[0], point[1], 1.0])
    return round(x * SCALE), round(y * SCALE)


def render(item, position, number, count):
    index, crop, matrix, keypoints, confs = item
    image = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    for k, name, color in ((LEFT_SHOULDER, "L", (0, 255, 255)), (RIGHT_SHOULDER, "R", (255, 255, 0))):
        x, y = to_crop(keypoints[k], matrix)
        cv2.circle(image, (x, y), 8, color, 2)
        cv2.putText(image, f"{name} {confs[k]:.2f}", (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for k in (LEFT_HIP, RIGHT_HIP):
        cv2.circle(image, to_crop(keypoints[k], matrix), 5, (255, 0, 255), -1)
    status = "-"
    if position is not None:
        status = position[0]
        if position[0] == "tag":
            cv2.drawMarker(image, to_crop((float(position[1]), float(position[2])), matrix),
                           (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    text = f"frame {index} [{number + 1}/{count}]  {status}   click tag centre, s=skip, a/d, q=quit"
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def on_mouse(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    index, _, matrix, _, _ = state["items"][state["number"]]
    # crop pixel -> original image, so the label does not depend on how the crop was cut
    ox, oy = cv2.invertAffineTransform(matrix) @ np.array([x / SCALE, y / SCALE, 1.0])
    state["positions"][index] = ("tag", f"{ox:.1f}", f"{oy:.1f}")
    state["number"] = min(state["number"] + 1, len(state["items"]) - 1)


def main():
    args = parse_args()
    print("Running the pose model on the labelled frames...")
    items = staff_poses(args)
    print(f"{len(items)} crops to label")
    positions = load_positions(args.out)
    state = {"items": items, "positions": positions,
             "number": next((i for i, item in enumerate(items) if item[0] not in positions), 0)}
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    while True:
        item = items[state["number"]]
        cv2.imshow(WINDOW, render(item, positions.get(item[0]), state["number"], len(items)))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s"):
            positions[item[0]] = ("skip", "", "")
            state["number"] = min(state["number"] + 1, len(items) - 1)
        elif key == ord("d"):
            state["number"] = min(state["number"] + 1, len(items) - 1)
        elif key == ord("a"):
            state["number"] = max(state["number"] - 1, 0)
    cv2.destroyAllWindows()
    save_positions(args.out, positions)
    print(f"{sum(p[0] == 'tag' for p in positions.values())} tag positions, "
          f"{sum(p[0] == 'skip' for p in positions.values())} skipped, {len(items) - len(positions)} unlabelled")


if __name__ == "__main__":
    main()
