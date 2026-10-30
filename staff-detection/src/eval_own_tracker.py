"""Evaluate our own tracker (src/tracker.py) on the cached detections in outputs/detections.npz."""

import argparse
from collections import Counter, deque

import cv2
import numpy as np

from eval_detection import ANGLE_STEPS, ROOT, load_gt_points, to_panorama_point
from eval_tracking import count_id_changes, id_color, staff_track_id
from tracker import Tracker

MAX_MISSED_SECONDS = 1.0
SETTINGS = [  # (name, max_dist as a fraction of the radius, use velocity)
    ("dist0.05", 0.05, True),
    ("dist0.10", 0.10, True),
    ("dist0.20", 0.20, True),
    ("dist0.10_novel", 0.10, False),
]
COMPARE_RANGE = (300, 600)  # same frames as the ByteTrack run
TRAIL_LENGTH = 15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--detections", default=str(ROOT / "outputs" / "detections.npz"))
    return parser.parse_args()


# median and max px per frame between consecutive labelled points
def staff_speed(gt_points):
    frames = sorted(gt_points)
    speeds = [np.hypot(*np.subtract(gt_points[b], gt_points[a])) / (b - a) for a, b in zip(frames, frames[1:])]
    return float(np.median(speeds)), float(np.max(speeds))


# detection row indices for each frame (rows are stored in frame order)
def rows_per_frame(detections):
    frame_of_row = detections["frame"]
    frame_count = int(detections["frame_count"])
    bounds = np.searchsorted(frame_of_row, np.arange(frame_count + 1))
    return [np.arange(bounds[f], bounds[f + 1]) for f in range(frame_count)]


# run the tracker over every frame; returns the track id of every detection row
def run_tracker(detections, frame_rows, max_dist, max_missed, use_velocity):
    tracker = Tracker(max_dist, max_missed, use_velocity)
    ids = np.zeros(len(detections["frame"]), int)
    for rows in frame_rows:
        ids[rows] = tracker.update(detections["point"][rows])
    return ids


# staff track id at each GT frame (None if no box contains the point)
def staff_ids_at_gt(detections, frame_rows, ids, gt_points):
    center, radius = detections["center"], float(detections["radius"])
    staff = {}
    for frame, (x, y) in sorted(gt_points.items()):
        rows = frame_rows[frame]
        point = to_panorama_point(x, y, center, radius)
        staff[frame] = None if point is None else staff_track_id(
            point, detections["pano_box"][rows], ids[rows].tolist(), ANGLE_STEPS)
    return staff


def evaluate(name, detections, frame_rows, gt_points, max_dist_fraction, use_velocity):
    max_dist = max_dist_fraction * float(detections["radius"])
    max_missed = round(MAX_MISSED_SECONDS * float(detections["fps"]))
    ids = run_tracker(detections, frame_rows, max_dist, max_missed, use_velocity)
    staff = staff_ids_at_gt(detections, frame_rows, ids, gt_points)
    in_range = [staff[f] for f in staff if COMPARE_RANGE[0] <= f <= COMPARE_RANGE[1]]
    return {"name": name, "ids": ids, "staff": staff,
            "unique_ids": len(set(ids.tolist())),
            "median_length": float(np.median(list(Counter(ids.tolist()).values()))),
            "changes": count_id_changes(list(staff.values())),
            "changes_in_range": count_id_changes(in_range)}


def print_table(rows):
    print(f"\n{'setting':<16}{'unique IDs':>11}{'median len':>11}{'ID changes':>11}"
          f"{'changes 300-600':>17}")
    for row in rows:
        print(f"{row['name']:<16}{row['unique_ids']:>11}{row['median_length']:>11.0f}"
              f"{row['changes']:>11}{row['changes_in_range']:>17}")
    for row in rows:
        sequence = " ".join("-" if i is None else str(i) for i in row["staff"].values())
        print(f"\n{row['name']} staff IDs at GT frames:\n  {sequence}")


def draw_tracks(frame, points, ids, trails, staff_id, gt_point, index):
    image = frame.copy()
    for track_id, trail in trails.items():
        if len(trail) > 1:
            cv2.polylines(image, [np.array(trail, np.int32)], False, id_color(track_id), 1, cv2.LINE_AA)
    for (x, y), track_id in zip(points.astype(int), ids):
        is_staff = track_id == staff_id
        cv2.circle(image, (x, y), 9 if is_staff else 4, id_color(track_id), -1)
        if is_staff:
            cv2.circle(image, (x, y), 11, (255, 255, 255), 2)
        cv2.putText(image, str(track_id), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 if is_staff else 0.45, id_color(track_id), 2 if is_staff else 1, cv2.LINE_AA)
    if gt_point is not None:
        cv2.drawMarker(image, gt_point, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    header = f"frame {index}  staff ID {staff_id}"
    cv2.putText(image, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return image


# original frames with each track's point, ID and trail; the staff track (as of the last GT frame) is bigger
def save_video(path, video, detections, frame_rows, result, gt_points):
    cap = cv2.VideoCapture(video)
    writer, trails, staff_id = None, {}, None
    for index, rows in enumerate(frame_rows):
        ok, frame = cap.read()
        if not ok:
            break
        points, ids = detections["point"][rows], result["ids"][rows].tolist()
        for point, track_id in zip(points, ids):
            trails.setdefault(track_id, deque(maxlen=TRAIL_LENGTH)).append(point)
        staff_id = result["staff"].get(index, staff_id)
        visible_trails = {i: trails[i] for i in ids}
        image = draw_tracks(frame, points, ids, visible_trails, staff_id, gt_points.get(index), index)
        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(detections["fps"]), (w, h))
        writer.write(image)
    cap.release()
    writer.release()


def main():
    args = parse_args()
    detections = dict(np.load(args.detections))
    gt_points = load_gt_points(args.gt)
    frame_rows = rows_per_frame(detections)

    median_speed, max_speed = staff_speed(gt_points)
    print(f"Staff speed from GT: median {median_speed:.1f} px/frame, max {max_speed:.1f} px/frame")
    radius, fps = float(detections["radius"]), float(detections["fps"])
    print(f"Radius {radius:.0f} px, fps {fps:.2f}, max_missed {round(MAX_MISSED_SECONDS * fps)} frames; "
          f"max_dist 0.05/0.10/0.20 = {0.05 * radius:.0f}/{0.10 * radius:.0f}/{0.20 * radius:.0f} px")

    rows = [evaluate(name, detections, frame_rows, gt_points, fraction, use_velocity)
            for name, fraction, use_velocity in SETTINGS]
    print_table(rows)

    best = min(rows, key=lambda r: (r["changes"], r["unique_ids"]))
    path = ROOT / "outputs" / f"own_track_{best['name']}.mp4"
    save_video(path, args.video, detections, frame_rows, best, gt_points)
    print(f"\nBest setting: {best['name']} -> {path}")


if __name__ == "__main__":
    main()
