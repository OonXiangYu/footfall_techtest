"""Upright chest crops around a person, from the original fisheye frame."""

import math

import cv2
import numpy as np

from fisheye import to_original_point

CROP = 160
SCALE = 2
CHEST_HEIGHT = 0.30  # chest sits 30% down the box; the panorama box top is the head (rim side)


# chest point of a panorama box, in original-image coordinates
def box_chest_point(box, center, radius):
    x1, y1, x2, y2 = box
    return np.array(to_original_point((x1 + x2) / 2, y1 + CHEST_HEIGHT * (y2 - y1), center, radius))


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
