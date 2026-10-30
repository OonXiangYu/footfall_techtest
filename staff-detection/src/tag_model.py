"""Tag detector on the body canvas: logistic regression on small tag-centred patches, slid over two
shoulder-placed side windows. Scored with numpy only (the pipeline does not need scikit-learn)."""

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pose import to_canvas

PATCH_SIZE = (24, 12)  # (width, height) px of a patch; covers 2 x the tag size, tag in the middle
STEP = 2  # px between search positions (at patch resolution)
FEATURES = ["abs", "pixels"]
LLR_BINS = np.linspace(0, 1, 11)
LLR_CLIP = 2.0
EVIDENCE_VARIANTS = ["mixture", "visible", "visible_positive"]


def patch_scale(tag_units):
    """Patch pixels per shoulder width (the patch spans twice the tag)."""
    return PATCH_SIZE[0] / (2 * tag_units[0])


def side_windows(centre, half):
    """The two search windows, mirrored across the body axis (the model's left/right can flip)."""
    centre, half = np.asarray(centre, float), np.asarray(half, float)
    return [(centre, half), (np.array([-centre[0], centre[1]]), half)]


def resample(canvas, width, top_left, size, scale):
    """Canvas region starting at top_left (body units), size px at scale px per shoulder width."""
    x0, y0 = to_canvas(top_left, width)
    step = width / scale  # canvas px per output px
    matrix = np.array([[step, 0, x0], [0, step, y0]])
    return cv2.warpAffine(canvas, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def patch_at(canvas, width, position, tag_units):
    """The patch centred on a body position."""
    scale = patch_scale(tag_units)
    half = np.array(PATCH_SIZE) / 2 / scale
    return resample(canvas, width, np.asarray(position) - half, PATCH_SIZE, scale)


def window_patches(canvas, width, window, tag_units):
    """All patches with their centre inside the window: (patches (N, h, w), centres in body units (N, 2))."""
    scale = patch_scale(tag_units)
    centre, half = window
    patch_half = np.array(PATCH_SIZE) / 2 / scale
    top_left = centre - half - patch_half
    size = tuple(int(round(v)) for v in (2 * (half + patch_half) * scale))
    region = resample(canvas, width, top_left, size, scale)
    views = sliding_window_view(region, PATCH_SIZE[::-1])[::STEP, ::STEP]
    rows, cols = views.shape[:2]
    ys, xs = np.meshgrid(np.arange(rows) * STEP, np.arange(cols) * STEP, indexing="ij")
    centres = np.stack([xs.ravel() + PATCH_SIZE[0] / 2, ys.ravel() + PATCH_SIZE[1] / 2], axis=1) / scale + top_left
    return views.reshape(-1, PATCH_SIZE[1], PATCH_SIZE[0]), centres


def features(patches, kind):
    """abs: magnitude of contrast-normalised pixels (bright-on-dark and dark-on-light alike);
    pixels: the contrast-normalised pixels."""
    patches = np.asarray(patches, np.float32).reshape(len(patches), -1)
    z = (patches - patches.mean(axis=1, keepdims=True)) / (patches.std(axis=1, keepdims=True) + 1e-6)
    return np.abs(z) if kind == "abs" else z


def probabilities(model, X):
    return 1 / (1 + np.exp(-(X @ model["coef"] + model["intercept"])))


def best_position(model, canvas, width):
    """(probability, body position) of the best patch in either side window."""
    best = (-1.0, None)
    for window in side_windows(model["window_centre"], model["window_half"]):
        patches, centres = window_patches(canvas, width, window, model["tag_units"])
        p = probabilities(model, features(patches, model["feature"]))
        k = int(np.argmax(p))
        if p[k] > best[0]:
            best = (float(p[k]), centres[k])
    return best


def llr(model, score, variant=None):
    """Clipped log-likelihood ratio of a score from the stored per-bin tables (see llr_tables)."""
    variant = variant or model.get("evidence_variant") or "mixture"
    k = np.clip(np.digitize(score, LLR_BINS) - 1, 0, len(LLR_BINS) - 2)
    return float(np.clip(model[f"llr_{variant}"][k], -LLR_CLIP, LLR_CLIP))


def llr_tables(staff_y, staff_n, others, visible_share):
    """Per score bin, add-one smoothed:
    mixture:          log P(score | staff) / P(score | other), staff = visible_share x tag-visible
                      frames + the rest tag-hidden frames (the tag is hidden most of the time);
    visible:          log P(score | tag visible on staff) / P(score | other);
    visible_positive: the same, but never below 0 (frames that do not look like a tag add nothing)."""
    def density(scores):
        counts, _ = np.histogram(scores, LLR_BINS)
        return (counts + 1) / (counts.sum() + len(counts))
    other = density(others)
    visible = np.log(density(staff_y) / other)
    mixture = np.log((visible_share * density(staff_y) + (1 - visible_share) * density(staff_n)) / other)
    return {"llr_mixture": mixture, "llr_visible": visible, "llr_visible_positive": np.maximum(visible, 0)}


def save_model(path, **fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)


def load_model(path):
    model = dict(np.load(path))
    model["feature"] = str(model["feature"])
    evidence = Path(path).with_suffix(".json")  # evidence variant and chain threshold, from eval_chain_evidence.py
    chosen = json.loads(evidence.read_text()) if evidence.exists() else {}
    model["chain_threshold"] = chosen.get("threshold")
    model["evidence_variant"] = chosen.get("variant", "mixture")
    return model
