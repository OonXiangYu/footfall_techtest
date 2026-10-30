"""Name-tag template (from the task PDF) and multi-scale, multi-angle template matching."""

import re

import cv2
import numpy as np

from tag_detect import chest_square

SEARCH_FRACTION = 0.6  # central square searched; larger than eval_tag's 0.4 because some crops are off-centre
SCALES = [0.8, 1.0, 1.25]
ANGLES = [-15, 0, 15]
TAG_MARGIN = 3  # px around the white card in the PDF photo
TAG_SIZE = (14, 7)  # tag size in the video (px), measured on labelled crops (tag_template.py)


# first embedded JPEG in the PDF (the green-polo photo); PDF photos are stored as raw DCT streams
def extract_pdf_photo(path):
    data = path.read_bytes()
    match = re.search(rb"/Filter\s*/DCTDecode[^>]*?/Length\s+(\d+)[^>]*>>\s*stream\r?\n", data, re.S)
    if match is None:
        raise SystemExit(f"No JPEG image found in {path}")
    raw = data[match.end():match.end() + int(match.group(1))]
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# the tag is the largest white (low-saturation, bright) blob on the polo
def find_pdf_tag(photo):
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    # open first: the thin white collar stripes would otherwise merge with the card;
    # then close over the black bars so the card is one blob
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # a small margin keeps the card's dark top band and its edge against the shirt
    x, y = max(x - TAG_MARGIN, 0), max(y - TAG_MARGIN, 0)
    w, h = w + 2 * TAG_MARGIN, h + 2 * TAG_MARGIN
    return cv2.cvtColor(photo[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (x, y, w, h)


def template_variants(template):
    """(image, scale, angle) for every scale/rotation; rotation keeps the size and replicates the border."""
    variants = []
    for scale in SCALES:
        scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = scaled.shape
        for angle in ANGLES:
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(scaled, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            variants.append((rotated, scale, angle))
    return variants


def match(gray, variants):
    """Best TM_CCOEFF_NORMED over all variants inside the search square: (score, (x, y, w, h), scale, angle)."""
    x0, y0, x1, y1 = chest_square(gray, SEARCH_FRACTION)
    region = gray[y0:y1, x0:x1]
    best = (-1.0, None, None, None)
    for image, scale, angle in variants:
        h, w = image.shape
        result = cv2.matchTemplate(region, image, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        if score > best[0]:
            best = (float(score), (x0 + x, y0 + y, w, h), scale, angle)
    return best
