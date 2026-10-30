"""Classical name-tag score on an upright chest crop (original scale, from tag_crops.upright_crop)."""

import cv2
import numpy as np

CHEST_FRACTION = 0.4  # side of the central square searched for the tag, as a fraction of the crop
# Default from measure_tag_size on the labelled "y" crops (tag about 15 x 7 px -> kernel_for_tag
# gives 19). eval_tag.py re-measures and prints it; the kernel must be a bit larger than the tag
# so the whole tag counts as foreground for top-hat / black-hat.
KERNEL_SIZE = 19
KERNEL_MARGIN = 1.25
MEASURE_KERNEL = 31  # only bounds the measurement: larger than any plausible tag in the chest square
MEASURE_LEVEL = 0.5  # tag pixels = response above this fraction of the peak


def chest_square(crop, fraction=CHEST_FRACTION):
    """Bounds (x0, y0, x1, y1) of the central square (side = fraction of the crop)."""
    h, w = crop.shape[:2]
    half = round(min(h, w) * fraction / 2)
    return w // 2 - half, h // 2 - half, w // 2 + half, h // 2 + half


def hat_responses(crop, kernel_size):
    """Top-hat (bright on dark: tag on the jacket) and black-hat (dark on light: tag on the white shirt)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return {"tophat": cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
            "blackhat": cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)}


def strongest(crop, kernel_size):
    """(score, kind, (x, y) in crop coordinates, response image) of the strongest response in the chest square."""
    x0, y0, x1, y1 = chest_square(crop)
    best = None
    for kind, response in hat_responses(crop, kernel_size).items():
        region = response[y0:y1, x0:x1]
        y, x = np.unravel_index(region.argmax(), region.shape)
        if best is None or region[y, x] > best[0]:
            best = (int(region[y, x]), kind, (x0 + int(x), y0 + int(y)), response)
    return best


def tag_score(crop, kernel_size=KERNEL_SIZE):
    """Returns (score, which response fired, (x, y) where it fired in crop coordinates)."""
    score, kind, position, _ = strongest(crop, kernel_size)
    return score, kind, position


def measure_tag_size(crop):
    """Width and height (px) of the blob around the strongest chest response, using an oversized kernel."""
    score, _, (x, y), response = strongest(crop, MEASURE_KERNEL)
    mask = (response >= MEASURE_LEVEL * score).astype(np.uint8)
    # close the gaps between the letters so the whole tag is one blob
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    blob = labels[y, x]
    return int(stats[blob, cv2.CC_STAT_WIDTH]), int(stats[blob, cv2.CC_STAT_HEIGHT])


def kernel_for_tag(sizes):
    """Median tag (w, h) and the odd kernel size a bit larger than it.

    Blobs that don't fit in the measuring kernel are something else (a shadow or hair on the
    white shirt fires stronger than the small tag), so they are left out.
    """
    tags = np.array([s for s in sizes if max(s) < MEASURE_KERNEL])
    width, height = np.median(tags, axis=0)
    kernel = int(np.ceil(KERNEL_MARGIN * max(width, height)))
    return (float(width), float(height), len(tags)), kernel + (kernel % 2 == 0)
