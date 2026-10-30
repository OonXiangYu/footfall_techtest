"""Appearance per detection: an HSV colour histogram of the torso, for appearance-aware linking."""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO

from crops import upright_crop
from pose import POSE_WEIGHTS, body_frame, body_patch, keypoints_to_original, run_pose

ROOT = Path(__file__).resolve().parent.parent
TORSO_X, TORSO_Y = (-0.5, 0.5), (0.1, 0.9)  # torso in body coordinates (shoulder widths)
TORSO_SIZE = (24, 24)
CROP_TORSO = np.s_[70:115, 62:98]  # torso region of the 160 x 160 upright chest crop (no pose)
HIST_BINS = (8, 4, 4)  # hue, saturation, value
POSE_MATCH_IOU = 0.3  # a pose box belongs to a detector box if they overlap at least this much
END_SAMPLES = 10  # detections averaged for a track's start / end appearance


def hsv_histogram(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    return hist / max(hist.sum(), 1e-6)


def similarity(a, b):
    """1 - Bhattacharyya distance: 1 for identical histograms, 0 for disjoint ones."""
    return 1.0 - cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)


def box_iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_poses(det_boxes, pose_boxes):
    """{detection index: pose index}, one-to-one by IoU (Hungarian), IoU >= POSE_MATCH_IOU."""
    if len(det_boxes) == 0 or len(pose_boxes) == 0:
        return {}
    iou = np.array([[box_iou(d, p) for p in pose_boxes] for d in det_boxes])
    rows, cols = linear_sum_assignment(-iou)
    return {r: c for r, c in zip(rows, cols) if iou[r, c] >= POSE_MATCH_IOU}


def compute_appearance(video, detections, frame_rows):
    """Torso histogram of every detection row, and whether it came from the pose (else the chest crop)."""
    center, radius = tuple(detections["center"]), float(detections["radius"])
    model = YOLO(str(ROOT / POSE_WEIGHTS))
    hists = np.zeros((len(detections["frame"]), int(np.prod(HIST_BINS))), np.float32)
    from_pose = np.zeros(len(detections["frame"]), bool)
    cap = cv2.VideoCapture(str(video))
    for index in tqdm(range(len(frame_rows)), desc="Appearance"):
        ok, frame = cap.read()
        if not ok:
            break
        rows = frame_rows[index]
        if len(rows) == 0:
            continue
        pose_boxes, keypoints, confs = run_pose(model, frame, center, radius)
        matched = match_poses(detections["box"][rows], pose_boxes)
        for k, row in enumerate(rows):
            body = None
            if k in matched:
                body = body_frame(keypoints_to_original(keypoints[matched[k]], center, radius), confs[matched[k]])
            if body is not None:
                patch = body_patch(frame, body, TORSO_X, TORSO_Y, TORSO_SIZE)
                from_pose[row] = True
            else:
                patch = upright_crop(frame, tuple(detections["chest"][row]), center, scale=1)[CROP_TORSO]
            hists[row] = hsv_histogram(patch)
    cap.release()
    return hists, from_pose


def load_or_compute_appearance(path, video, detections, frame_rows):
    """Cached at path (keyed on the video and the detection rows)."""
    key = json.dumps({"video": str(Path(video).resolve()), "mtime": Path(video).stat().st_mtime,
                      "rows": len(detections["frame"]), "bins": HIST_BINS, "torso": [TORSO_X, TORSO_Y]})
    if Path(path).exists():
        cached = np.load(path)
        if str(cached["key"]) == key:
            return cached["hists"], cached["from_pose"]
    hists, from_pose = compute_appearance(video, detections, frame_rows)
    np.savez(path, key=key, hists=hists, from_pose=from_pose)
    return hists, from_pose


def track_ends(frames, ids, hists):
    """For each track id: (mean histogram of its first END_SAMPLES detections, of its last END_SAMPLES)."""
    rows_of = {}
    for row, track_id in enumerate(ids):
        rows_of.setdefault(track_id, []).append(row)
    ends = {}
    for track_id, rows in rows_of.items():
        rows = sorted(rows, key=lambda r: frames[r])
        ends[track_id] = (hists[rows[:END_SAMPLES]].mean(axis=0), hists[rows[-END_SAMPLES:]].mean(axis=0))
    return ends
