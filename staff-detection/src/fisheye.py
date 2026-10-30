"""Fisheye geometry: image circle, panorama unwarp, point mapping, seam padding and merging."""

import math

import cv2
import numpy as np

ANGLE_STEPS = 1440  # panorama width: one column per 0.25 degree
BORDER_THRESHOLD = 20
SEAM_PAD = 0.10  # fraction of panorama columns repeated past the right edge
SEAM_OVERLAP = 0.6  # overlap (intersection / smaller box) above which two seam boxes are one person


# find the circle on the fish eye
def find_fisheye_circle(frame):
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
    if outside.mean() < 0.05 or (bright[outside] == 0).mean() < 0.9:
        return None
    # the bright region should fill most of the circle that is inside the frame.
    if (bright[~outside] > 0).mean() < 0.7:
        return None
    return (cx, cy), radius


# if cant find the circle this si the default
def fallback_circle(frame):
    h, w = frame.shape[:2]
    return (w / 2, h / 2), max(w, h) / 2


def circle_for(frame):
    return find_fisheye_circle(frame) or fallback_circle(frame)


def unwarp(frame, center, radius):
    polar = cv2.warpPolar(frame, (round(radius), ANGLE_STEPS), center, radius, cv2.WARP_POLAR_LINEAR)
    return cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)


# convert the location of point on the ori img to the wnwarp img and see if its in the frame of YOLO
def to_panorama_point(x, y, center, radius):
    dx, dy = x - center[0], y - center[1]
    rho = math.hypot(dx, dy)
    if rho >= radius:
        return None
    theta = math.atan2(dy, dx) % (2 * math.pi)
    polar_width = round(radius)
    pano_x = theta * ANGLE_STEPS / (2 * math.pi)
    pano_y = polar_width - 1 - rho * polar_width / radius
    return pano_x, pano_y


# exact inverse of to_panorama_point; x past the right edge (seam padding) wraps to the start
def to_original_point(pano_x, pano_y, center, radius):
    polar_width = round(radius)
    theta = (pano_x % ANGLE_STEPS) * 2 * math.pi / ANGLE_STEPS
    rho = (polar_width - 1 - pano_y) * radius / polar_width
    return center[0] + rho * math.cos(theta), center[1] + rho * math.sin(theta)


# duplicate the most left side to the right side to let the people at the cutting side can be completed
def pad_seam(pano):
    pad = round(pano.shape[1] * SEAM_PAD)
    return np.hstack([pano, pano[:, :pad]])


# cal how much two boxes is overlap to determine is the person get encoded twice
def overlap(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller


# after filling the cutting side someone might be box twice so just leaves one
def merge_seam_boxes(boxes, pano_width):
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


def merge_seam_tracks(boxes, ids, pano_width):
    """merge_seam_boxes, keeping each surviving box's ID (any per-row label works, e.g. row indices)."""
    merged = merge_seam_boxes(boxes, pano_width)
    shift = np.array([pano_width, 0, pano_width, 0])
    merged_ids = []
    for box in merged:
        # A merged box is either an original box or one shifted back out of the padding.
        match = np.flatnonzero(np.all(np.isclose(boxes, box), axis=1) |
                               np.all(np.isclose(boxes - shift, box), axis=1))
        merged_ids.append(int(ids[match[0]]))
    return merged, merged_ids


def owning_box(point, boxes, wrap_width=None):
    """Index of the box this point belongs to, or None: among the boxes containing it, the one
    whose centre is nearest. wrap_width is the panorama width, so a point near x=0 also matches
    a box that crosses the seam (x2 > wrap_width); distance is measured to the matching copy."""
    candidates = [point] if wrap_width is None else [point, (point[0] + wrap_width, point[1])]
    best, best_distance = None, math.inf
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        for x, y in candidates:
            if x1 <= x <= x2 and y1 <= y <= y2:
                distance = math.hypot((x1 + x2) / 2 - x, (y1 + y2) / 2 - y)
                if distance < best_distance:
                    best, best_distance = index, distance
    return best


# determine is the point in any box
def point_in_any_box(point, boxes, wrap_width=None):
    return owning_box(point, boxes, wrap_width) is not None
