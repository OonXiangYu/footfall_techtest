"""Pose (yolov8n-pose) on the seam-padded panorama, and a body frame built from the shoulders."""

import cv2
import numpy as np

from fisheye import ANGLE_STEPS, merge_seam_tracks, owning_box, pad_seam, to_original_point, to_panorama_point, unwarp

POSE_WEIGHTS = "yolov8n-pose.pt"
IMGSZ = 1440
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6  # COCO keypoint indices, as the model names them
LEFT_HIP, RIGHT_HIP = 11, 12
KEYPOINT_CONF = 0.5
CANVAS_X, CANVAS_Y = (-2.0, 2.0), (-1.0, 2.5)  # body-canvas extent in shoulder widths


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


def body_patch(frame, body, x_range, y_range, size):
    """The body region x_range x y_range (shoulder widths; x along the shoulders, y towards the hips),
    resampled from the original frame to size (width, height) px, in colour."""
    origin, x_axis, y_axis, width = body
    sx = (x_range[1] - x_range[0]) * width / size[0]
    sy = (y_range[1] - y_range[0]) * width / size[1]
    x0, y0 = x_range[0] * width, y_range[0] * width
    # patch pixel (u, v) -> original image: origin + (x0 + u sx) * x_axis + (y0 + v sy) * y_axis
    matrix = np.array([[sx * x_axis[0], sy * y_axis[0], origin[0] + x0 * x_axis[0] + y0 * y_axis[0]],
                       [sx * x_axis[1], sy * y_axis[1], origin[1] + x0 * x_axis[1] + y0 * y_axis[1]]])
    return cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderValue=(0, 0, 0))


def body_canvas(frame, body):
    """Gray image of the body region CANVAS_X x CANVAS_Y at native resolution
    (one shoulder width = body width px)."""
    width = body[3]
    size = (round((CANVAS_X[1] - CANVAS_X[0]) * width), round((CANVAS_Y[1] - CANVAS_Y[0]) * width))
    # ranges matching the rounded size exactly, so one canvas pixel is one original pixel
    x_range = (CANVAS_X[0], CANVAS_X[0] + size[0] / width)
    y_range = (CANVAS_Y[0], CANVAS_Y[0] + size[1] / width)
    return cv2.cvtColor(body_patch(frame, body, x_range, y_range, size), cv2.COLOR_BGR2GRAY)


def to_canvas(point, width):
    return (point[0] - CANVAS_X[0]) * width, (point[1] - CANVAS_Y[0]) * width
