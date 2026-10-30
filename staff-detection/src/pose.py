"""Pose (yolov8n-pose) on the seam-padded panorama, and a body frame built from the shoulders."""

import numpy as np

from fisheye import ANGLE_STEPS, merge_seam_tracks, owning_box, pad_seam, to_original_point, to_panorama_point, unwarp

POSE_WEIGHTS = "yolov8n-pose.pt"
IMGSZ = 1440
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6  # COCO keypoint indices, as the model names them
LEFT_HIP, RIGHT_HIP = 11, 12
KEYPOINT_CONF = 0.5


def run_pose(model, frame, center, radius):
    """Seam-merged pose boxes (padded panorama coords), keypoints (N, 17, 2) and confidences (N, 17)."""
    pano = unwarp(frame, center, radius)
    result = model(pad_seam(pano), imgsz=IMGSZ, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return boxes, np.zeros((0, 17, 2)), np.zeros((0, 17))
    keypoints = result.keypoints.xy.cpu().numpy()
    confs = result.keypoints.conf.cpu().numpy()
    boxes, rows = merge_seam_tracks(boxes, list(range(len(boxes))), pano.shape[1])
    return boxes, keypoints[rows], confs[rows]


def staff_owner(boxes, chest_point, center, radius):
    """Index of the pose box that owns the staff member's (original-image) chest point, or None."""
    point = to_panorama_point(*chest_point, center, radius)
    return None if point is None else owning_box(point, boxes, ANGLE_STEPS)


def keypoints_to_original(keypoints, center, radius):
    """Panorama keypoints -> original image (x past the seam wraps)."""
    return np.array([to_original_point(x, y, center, radius) for x, y in keypoints])


def body_frame(keypoints, confs):
    """(origin, x axis, y axis, shoulder width) in original-image coordinates, or None.

    origin = shoulder midpoint; x axis = the model's left -> right shoulder direction;
    y axis = perpendicular to it, pointing towards the hips; unit = shoulder width.
    Needs both shoulders and at least one hip above KEYPOINT_CONF.
    """
    left, right = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    hips = [keypoints[k] for k in (LEFT_HIP, RIGHT_HIP) if confs[k] > KEYPOINT_CONF]
    if min(confs[LEFT_SHOULDER], confs[RIGHT_SHOULDER]) <= KEYPOINT_CONF or not hips:
        return None
    origin = (left + right) / 2
    width = float(np.linalg.norm(right - left))
    if width < 1:
        return None
    x_axis = (right - left) / width
    y_axis = np.array([-x_axis[1], x_axis[0]])
    if np.dot(np.mean(hips, axis=0) - origin, y_axis) < 0:
        y_axis = -y_axis
    return origin, x_axis, y_axis, width


def to_body(point, frame):
    """Original-image point -> body coordinates (shoulder-width units)."""
    origin, x_axis, y_axis, width = frame
    offset = np.asarray(point, float) - origin
    return np.array([offset @ x_axis, offset @ y_axis]) / width


def from_body(point, frame):
    """Body coordinates -> original-image point."""
    origin, x_axis, y_axis, width = frame
    return origin + width * (point[0] * x_axis + point[1] * y_axis)
