"""Person detection on the fisheye frame (panorama + seam padding, or the original frame)."""

import time

import numpy as np

from fisheye import merge_seam_boxes, merge_seam_tracks, pad_seam, unwarp


# detect and return the box that encoded human
def detect(model, image, imgsz, conf):
    start = time.perf_counter()
    result = model(image, classes=[0], imgsz=imgsz, conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result.boxes.xyxy.cpu().numpy(), elapsed_ms


# check whether need to unwarp as not all the img is capture by fish eye
def detect_view(model, frame, circle, use_pano, imgsz, conf):
    if not use_pano:
        boxes, elapsed_ms = detect(model, frame, imgsz, conf)
        return frame, boxes, elapsed_ms
    pano = unwarp(frame, *circle)
    padded = pad_seam(pano)
    boxes, elapsed_ms = detect(model, padded, imgsz, conf)
    return padded, merge_seam_boxes(boxes, pano.shape[1]), elapsed_ms


def count_people(model, frames, circle, use_pano, imgsz, conf):
    return sum(len(detect_view(model, f, circle, use_pano, imgsz, conf)[1]) for f in frames)


# determine whether test on ori img or unwarp img
def choose_view(mode, model, probe_frames, circle, imgsz, conf):
    if mode != "auto":
        return mode == "on"
    on_pano = count_people(model, probe_frames, circle, True, imgsz, conf)
    on_original = count_people(model, probe_frames, circle, False, imgsz, conf)
    use_pano = on_pano > on_original
    print(f"  auto: first {len(probe_frames)} frames -> {on_pano} people on panorama, "
          f"{on_original} on original -> {'panorama' if use_pano else 'original frame'}")
    return use_pano


def detect_people(model, frame, circle, use_pano, imgsz, conf):
    """Person boxes (panorama coordinates if use_pano, else original) and their confidences."""
    pano = unwarp(frame, *circle) if use_pano else None
    image = pad_seam(pano) if use_pano else frame
    result = model(image, classes=[0], imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    if not use_pano:
        return boxes, confs
    # row indices stand in for IDs, so we learn which original rows survived the merge
    boxes, rows = merge_seam_tracks(boxes, list(range(len(boxes))), pano.shape[1])
    return boxes, confs[rows]


DUPLICATE_INSIDE = 0.85  # a box this much inside another one (of the smaller box's area) ...
DUPLICATE_WIDTH = 0.7  # ... and at least this similar in width is a second box on the same person


def duplicate_rows(detections, frame_rows):
    """Rows to drop: the smaller of two boxes on the same person (e.g. upper body + whole body of
    someone seated), which would otherwise make two tracks swap the person between them."""
    boxes, drop = detections["box"], np.zeros(len(detections["frame"]), bool)
    for rows in frame_rows:
        for i in rows:
            for j in rows:
                if i == j or drop[j]:
                    continue
                a, b = boxes[i], boxes[j]  # is a (the smaller) a duplicate inside b?
                area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
                if area_a > area_b:
                    continue
                inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                widths = sorted((a[2] - a[0], b[2] - b[0]))
                if inter >= DUPLICATE_INSIDE * area_a and widths[0] >= DUPLICATE_WIDTH * widths[1]:
                    drop[i] = True
                    break
    return drop


def keep_rows(detections, keep):
    """The detections with only the kept rows (per-row arrays filtered, the rest unchanged)."""
    rows = len(detections["frame"])
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == rows else v)
            for k, v in detections.items()}


# detection row indices for each frame (rows are stored in frame order)
def rows_per_frame(detections):
    frame_of_row = detections["frame"]
    frame_count = int(detections["frame_count"])
    bounds = np.searchsorted(frame_of_row, np.arange(frame_count + 1))
    return [np.arange(bounds[f], bounds[f + 1]) for f in range(frame_count)]
