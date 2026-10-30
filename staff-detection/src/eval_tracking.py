import argparse
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from eval_detection import DEFAULT_IMGSZ, ROOT, load_gt_points, read_frames
from fisheye import (ANGLE_STEPS, fallback_circle, find_fisheye_circle, merge_seam_tracks, owning_box, pad_seam,
                     to_panorama_point, unwarp)

FIRST_FRAME, LAST_FRAME = 300, 600
SETTINGS = [  # (name, conf, tracker config)
    ("A_conf0.25_default", 0.25, "bytetrack.yaml"),
    ("B_conf0.10_default", 0.10, "bytetrack.yaml"),
    ("C_conf0.10_buffer90", 0.10, str(ROOT / "configs" / "bytetrack_buffer90.yaml")),
]

SEAM_PAD_TRACKING = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    return parser.parse_args()

def track_frame(model, frame, circle, conf, tracker, seam_pad):
    pano = unwarp(frame, *circle)
    image = pad_seam(pano) if seam_pad else pano
    result = model.track(image, persist=True, classes=[0], imgsz=DEFAULT_IMGSZ, conf=conf,
                         tracker=tracker, verbose=False)[0]
    if result.boxes.id is None:  # detections that are not yet confirmed tracks have no ID
        return image, np.zeros((0, 4)), []
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()
    if seam_pad:
        boxes, ids = merge_seam_tracks(boxes, ids, pano.shape[1])
    return image, boxes, ids

# gwtting back the staff id
def staff_track_id(point, boxes, ids, wrap_width=None):
    index = owning_box(point, boxes, wrap_width)
    return None if index is None else ids[index]


def id_color(track_id):
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 256, 3))


def draw_frame(image, boxes, ids, index, point, staff_id):
    image = image.copy()
    for (x1, y1, x2, y2), track_id in zip(boxes.astype(int), ids):
        thickness = 4 if track_id == staff_id else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), id_color(track_id), thickness)
        cv2.putText(image, str(track_id), (x1 + 2, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    id_color(track_id), 2, cv2.LINE_AA)
    header = f"frame {index}"
    if point is not None:
        cv2.drawMarker(image, (round(point[0]), round(point[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
        header += f"  staff ID: {staff_id}"
    cv2.putText(image, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def count_id_changes(staff_ids):
    known = [i for i in staff_ids if i is not None]
    return sum(a != b for a, b in zip(known, known[1:]))

# run from frame 300 - 600 with diff setting to evalution
def run_setting(name, conf, tracker, frames, gt_points, circle, seam_pad):
    model = YOLO(str(ROOT / "yolov8n.pt"))  # fresh model, so no tracker state carries over
    wrap_width = ANGLE_STEPS if seam_pad else None
    track_lengths = Counter()
    staff_ids = []
    writer = None
    for index in tqdm(range(FIRST_FRAME, LAST_FRAME + 1), desc=name, leave=False):
        image, boxes, ids = track_frame(model, frames[index], circle, conf, tracker, seam_pad)
        track_lengths.update(ids)
        point, staff_id = None, None
        if index in gt_points:
            point = to_panorama_point(*gt_points[index], *circle)
            staff_id = staff_track_id(point, boxes, ids, wrap_width) if point else None
            staff_ids.append(staff_id)
        annotated = draw_frame(image, boxes, ids, index, point, staff_id)
        if writer is None:
            h, w = annotated.shape[:2]
            path = ROOT / "outputs" / f"track_{name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        writer.write(annotated)
    writer.release()
    return {"name": name, "unique_ids": len(track_lengths),
            "median_length": float(np.median(list(track_lengths.values()))) if track_lengths else 0.0,
            "staff_ids": staff_ids, "id_changes": count_id_changes(staff_ids)}


def print_table(rows):
    print(f"\n{'setting':<22}{'unique IDs':>11}{'median len':>11}{'ID changes':>11}  staff IDs at GT frames")
    for row in rows:
        sequence = " ".join("-" if i is None else str(i) for i in row["staff_ids"])
        print(f"{row['name']:<22}{row['unique_ids']:>11}{row['median_length']:>11.0f}"
              f"{row['id_changes']:>11}  {sequence}")


def main():
    args = parse_args()
    gt_points = {i: p for i, p in load_gt_points(args.gt).items() if FIRST_FRAME <= i <= LAST_FRAME}
    frames = read_frames(args.video, set(range(FIRST_FRAME, LAST_FRAME + 1)))
    circle = find_fisheye_circle(frames[FIRST_FRAME]) or fallback_circle(frames[FIRST_FRAME])
    print(f"Panorama circle: center ({circle[0][0]:.0f}, {circle[0][1]:.0f}), radius {circle[1]:.0f}")
    print(f"Frames {FIRST_FRAME}-{LAST_FRAME}, GT frames in range: {sorted(gt_points)}")
    print(f"Seam padding during tracking: {'on' if SEAM_PAD_TRACKING else 'off'}")

    rows = [run_setting(name, conf, tracker, frames, gt_points, circle, SEAM_PAD_TRACKING)
            for name, conf, tracker in SETTINGS]
    print_table(rows)
    print(f"Videos saved as {ROOT / 'outputs'}/track_<setting>.mp4")


if __name__ == "__main__":
    main()
