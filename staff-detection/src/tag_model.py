"""The tag classifier's features and its saved model (logistic regression, scored with numpy only)."""

import cv2
import numpy as np

from pose import to_canvas

PATCH_SIZE = (24, 12)  # (width, height) px of a resampled tag window
FEATURES = ["pixels", "abs", "hog"]
HOG_CELL, HOG_BINS = 6, 9


def window_patch(canvas, width, window, shift=(0, 0)):
    """The window region of a body canvas resampled to PATCH_SIZE (subpixel, replicated border)."""
    centre, half = window
    (x0, y0), (x1, y1) = to_canvas(centre - half, width), to_canvas(centre + half, width)
    sx, sy = (x1 - x0) / PATCH_SIZE[0], (y1 - y0) / PATCH_SIZE[1]
    # patch pixel (u, v) -> canvas (x0 + (u + shift) * sx, y0 + (v + shift) * sy)
    matrix = np.array([[sx, 0, x0 + shift[0] * sx], [0, sy, y0 + shift[1] * sy]])
    return cv2.warpAffine(canvas, matrix, PATCH_SIZE, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def hog(patch):
    """Minimal HOG (OpenCV 5 dropped HOGDescriptor): per-cell histograms of unsigned gradient
    orientation weighted by magnitude, each 2x2-cell block L2-normalised."""
    image = patch.astype(np.float32)
    gx, gy = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = np.hypot(gx, gy), np.rad2deg(np.arctan2(gy, gx)) % 180
    bins = np.minimum((angle / (180 / HOG_BINS)).astype(int), HOG_BINS - 1)
    rows, cols = image.shape[0] // HOG_CELL, image.shape[1] // HOG_CELL
    cells = np.zeros((rows, cols, HOG_BINS), np.float32)
    for r in range(rows):
        for c in range(cols):
            window = np.s_[r * HOG_CELL:(r + 1) * HOG_CELL, c * HOG_CELL:(c + 1) * HOG_CELL]
            cells[r, c] = np.bincount(bins[window].ravel(), magnitude[window].ravel(), HOG_BINS)
    blocks = [cells[r:r + 2, c:c + 2].ravel() for r in range(rows - 1) for c in range(cols - 1)]
    return np.concatenate([b / (np.linalg.norm(b) + 1e-6) for b in blocks])


def features(patch, kind):
    """pixels: contrast-normalised pixels; abs: their magnitude (ignores bright-on-dark vs
    dark-on-light); hog: gradient-orientation histograms."""
    if kind == "hog":
        return hog(patch)
    z = (patch.astype(np.float32) - patch.mean()) / (patch.std() + 1e-6)
    return (np.abs(z) if kind == "abs" else z).ravel()


def load_tag_model(path):
    model = dict(np.load(path))
    model["feature"] = str(model["feature"])
    centre, half = model["window_centre"], model["window_half"]
    # the two side windows, mirrored across the body axis
    model["windows"] = [(centre, half), (np.array([-centre[0], centre[1]]), half)]
    return model


def tag_probability(model, canvas, width):
    """Probability that a tag is in either side window of this body canvas."""
    X = np.array([features(window_patch(canvas, width, w), model["feature"]) for w in model["windows"]])
    return float((1 / (1 + np.exp(-(X @ model["coef"] + model["intercept"])))).max())
