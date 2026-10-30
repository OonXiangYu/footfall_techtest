"""Tag template matching inside a small window placed from the shoulders (pose), two-fold on tag clicks.

Needs data/tag_pos.csv from label_tag_pos.py.
"""

import argparse
import csv
import heapq
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from eval_detection import ROOT
from eval_tag import print_distributions, print_threshold_table
from fisheye import to_original_point
from pose import POSE_WEIGHTS, body_frame, keypoints_to_original, run_pose, staff_owner, to_body
from tag_crops import interpolate_chest, load_tag_labels
from tag_match import SCALES, SEARCH_FRACTION, TAG_SIZE, extract_pdf_photo, find_pdf_tag

REVIEW_DIR = ROOT / "outputs" / "tag_review_window"
PEOPLE_CACHE = ROOT / "outputs" / "tag_people.pkl"
OTHER_MIN_DIST = 0.2  # other people: box centre farther than this fraction of the radius from the staff
CANVAS_X, CANVAS_Y = (-2.0, 2.0), (-1.0, 2.5)  # body-canvas extent in shoulder widths
WINDOW_STD = 2.0  # one window: half-size = WINDOW_STD x the clicks' std ...
MIN_WINDOW_TAGS = 1.5  # ... but at least this many tag sizes wide/high in total
SIDE_WINDOW_TAGS = 1.0  # two windows: half-size = this many tag sizes
# share of staff frames with a visible tag (38 y of 268 labelled crops), for the track-level rate
TAG_VISIBLE_SHARE = 0.14
THRESHOLDS = np.arange(0.30, 0.951, 0.05)
TOP_OTHERS, BOTTOM_Y = 20, 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--tag-labels", default=str(ROOT / "data" / "tag_gt.csv"))
    parser.add_argument("--tag-pos", default=str(ROOT / "data" / "tag_pos.csv"))
    parser.add_argument("--detections", default=str(ROOT / "outputs" / "detections.npz"))
    parser.add_argument("--pdf", default=str(ROOT / "data" / "AI Evaluation Test.pdf"))
    parser.add_argument("--windows", choices=["one", "two"], default="two",
                        help="one window from the clicks' mean/std, or two small mirrored windows (one per side)")
    return parser.parse_args()


def load_tag_positions(path):
    with open(path, newline="") as f:
        return {int(r["frame"]): np.array([float(r["x"]), float(r["y"])])
                for r in csv.DictReader(f) if r["status"] == "tag"}


def body_canvas(frame, body):
    """Gray image of the body region: x along the shoulders (left -> right), y towards the hips,
    native resolution (one shoulder width = width px). Returns the image and the width."""
    origin, x_axis, y_axis, width = body
    x0, y0 = CANVAS_X[0] * width, CANVAS_Y[0] * width
    # canvas pixel (u, v) -> original image: origin + (u + x0) * x_axis + (v + y0) * y_axis
    matrix = np.array([[x_axis[0], y_axis[0], origin[0] + x0 * x_axis[0] + y0 * y_axis[0]],
                       [x_axis[1], y_axis[1], origin[1] + x0 * x_axis[1] + y0 * y_axis[1]]])
    size = (round((CANVAS_X[1] - CANVAS_X[0]) * width), round((CANVAS_Y[1] - CANVAS_Y[0]) * width))
    canvas = cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderValue=(0, 0, 0))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)


def to_canvas(point, width):
    return (point[0] - CANVAS_X[0]) * width, (point[1] - CANVAS_Y[0]) * width


def collect_people(args, tag_labels):
    """Every person with a usable pose on the y/n tag frames: staff (with label) and others."""
    detections = np.load(args.detections)
    center, radius = tuple(detections["center"]), float(detections["radius"])
    chest = interpolate_chest(args.gt)
    wanted = {f for f, label in tag_labels.items() if label in ("y", "n") and f in chest}
    model = YOLO(str(ROOT / POSE_WEIGHTS))
    staff, others = [], []  # dicts: frame, label, body, canvas
    cap = cv2.VideoCapture(args.video)
    for index in range(max(wanted) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if index not in wanted:
            continue
        boxes, keypoints, confs = run_pose(model, frame, center, radius)
        owner = staff_owner(boxes, chest[index], center, radius)
        for k, box in enumerate(boxes):
            box_centre = to_original_point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, center, radius)
            is_staff = k == owner
            if not is_staff and np.hypot(*np.subtract(box_centre, chest[index])) <= OTHER_MIN_DIST * radius:
                continue  # too close to the staff to call it someone else
            body = body_frame(keypoints_to_original(keypoints[k], center, radius), confs[k])
            if body is None:
                continue
            person = {"frame": index, "label": tag_labels[index], "body": body, "canvas": body_canvas(frame, body)}
            (staff if is_staff else others).append(person)
    cap.release()
    return staff, others


def collect_people_cached(args, tag_labels):
    """collect_people, cached in outputs/ (the pose pass takes a while); rebuilt when inputs change."""
    key = json.dumps({"files": [(p, Path(p).stat().st_mtime) for p in (args.video, args.gt, args.tag_labels)],
                      "canvas": [CANVAS_X, CANVAS_Y], "other_min_dist": OTHER_MIN_DIST, "weights": POSE_WEIGHTS})
    if PEOPLE_CACHE.exists():
        with open(PEOPLE_CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached["key"] == key:
            return cached["staff"], cached["others"]
    staff, others = collect_people(args, tag_labels)
    with open(PEOPLE_CACHE, "wb") as f:
        pickle.dump({"key": key, "staff": staff, "others": others}, f)
    return staff, others


def windows_from_clicks(points, tag_units, mode):
    """Search windows [(centre, half-size)] in body units from clicked tag positions.

    one: a single window around the clicks (mean +- WINDOW_STD std).
    two: two small windows mirrored across the body axis at (+-|mean x|, mean y), because the
         model's left/right shoulders flip now and then, which mirrors the tag's side.
    """
    points = np.array(points)
    tag = np.array(tag_units)
    if mode == "one":
        mean, std = points.mean(axis=0), points.std(axis=0)
        return [(mean, np.maximum(WINDOW_STD * std, MIN_WINDOW_TAGS * tag / 2))]
    side_x, mean_y = np.abs(points[:, 0]).mean(), points[:, 1].mean()
    half = SIDE_WINDOW_TAGS * tag
    return [(np.array([side_x, mean_y]), half), (np.array([-side_x, mean_y]), half)]


def make_template(pdf_path):
    pdf_tag, _ = find_pdf_tag(extract_pdf_photo(Path(pdf_path)))
    return pdf_tag


def match_in_windows(canvas, width, windows, template, tag_units):
    """Best match over several windows: (score, centre, template size, positions summed over windows)."""
    results = [match_in_window(canvas, width, window, template, tag_units) for window in windows]
    best = max(results, key=lambda r: r[0])
    return best[0], best[1], best[2], sum(r[3] for r in results)


def match_in_window(canvas, width, window, template, tag_units):
    """Best TM_CCOEFF_NORMED (template and its mirror) with the tag centre inside the window.
    Returns (score, tag centre in canvas px, template size px, number of positions searched)."""
    mean, half = window
    tw = max(3, round(tag_units[0] * width))
    th = max(3, round(tag_units[1] * width))
    resized = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
    # canvas region whose template-centre positions cover the window
    cx0, cy0 = to_canvas(mean - half, width)
    cx1, cy1 = to_canvas(mean + half, width)
    x0, y0 = max(0, round(cx0 - tw / 2)), max(0, round(cy0 - th / 2))
    x1, y1 = min(canvas.shape[1], round(cx1 + tw / 2)), min(canvas.shape[0], round(cy1 + th / 2))
    region = canvas[y0:y1, x0:x1]
    if region.shape[0] < th or region.shape[1] < tw:
        return -1.0, None, (tw, th), 0
    best, where, positions = -1.0, None, 0
    for variant in (resized, cv2.flip(resized, 1)):
        result = cv2.matchTemplate(region, variant, cv2.TM_CCOEFF_NORMED)
        positions += result.size
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        if score > best:
            best, where = float(score), (x0 + x + tw / 2, y0 + y + th / 2)
    return best, where, (tw, th), positions


def previous_positions():
    """Search positions per crop of the unwindowed matcher (tag_template.py): 3 scales x 3 angles."""
    side = round(160 * SEARCH_FRACTION)
    total = 0
    for scale in SCALES:
        tw, th = round(TAG_SIZE[0] * scale), round(TAG_SIZE[1] * scale)
        total += 3 * (side - tw + 1) * (side - th + 1)
    return total


def print_track_table(scores, thresholds):
    """Expected per-frame hit rate of the staff (tag visible only part of the time) vs other people."""
    print(f"\nTrack level (staff rate = {TAG_VISIBLE_SHARE:.2f} x y found"
          f" + {1 - TAG_VISIBLE_SHARE:.2f} x staff-n firing)")
    print(f"{'threshold':>9}{'staff rate':>12}{'others rate':>13}{'ratio':>8}")
    for threshold in thresholds:
        y, n, others = (np.mean(np.array(scores[g]) >= threshold) for g in ("y", "n", "others"))
        staff_rate = TAG_VISIBLE_SHARE * y + (1 - TAG_VISIBLE_SHARE) * n
        ratio = staff_rate / others if others else float("inf")
        print(f"{threshold:>9.2f}{staff_rate:>12.1%}{others:>13.1%}{ratio:>8.2f}")


def review_image(person, result, windows, tag_units):
    score, where, (tw, th), _ = result
    width = person["body"][3]
    k = max(1, round(240 / person["canvas"].shape[1]))
    image = cv2.cvtColor(cv2.resize(person["canvas"], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC),
                         cv2.COLOR_GRAY2BGR)
    for mean, half in windows:
        p0, p1 = to_canvas(mean - half, width), to_canvas(mean + half, width)
        cv2.rectangle(image, (round(p0[0] * k), round(p0[1] * k)), (round(p1[0] * k), round(p1[1] * k)),
                      (255, 255, 0), 1)
    for shoulder in ((-0.5, 0), (0.5, 0)):
        cv2.circle(image, tuple(round(v * k) for v in to_canvas(shoulder, width)), 4, (0, 255, 255), -1)
    if where is not None:
        cv2.rectangle(image, (round((where[0] - tw / 2) * k), round((where[1] - th / 2) * k)),
                      (round((where[0] + tw / 2) * k), round((where[1] + th / 2) * k)), (0, 0, 255), 2)
    text = f"f{person['frame']} {person['label'] if 'label' in person else ''} {score:.2f}"
    cv2.putText(image, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return image


def save_review(name, scored_others, scored_y, windows, tag_units):
    out = REVIEW_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for rank, (_, person, result) in enumerate(scored_others, 1):
        cv2.imwrite(str(out / f"others_top{rank:02d}_f{person['frame']:04d}.jpg"),
                    review_image(person, result, windows, tag_units))
    for rank, (_, person, result) in enumerate(scored_y, 1):
        cv2.imwrite(str(out / f"y_low{rank:02d}_f{person['frame']:04d}.jpg"),
                    review_image(person, result, windows, tag_units))


def main():
    args = parse_args()
    tag_labels = load_tag_labels(args.tag_labels)
    clicks = load_tag_positions(args.tag_pos)
    staff, others = collect_people_cached(args, tag_labels)
    staff_y = [p for p in staff if p["label"] == "y"]
    print(f"People with a usable pose: staff {len(staff)} ({len(staff_y)} y), others {len(others)}")

    # 2. tag position in body coordinates
    clicked = [(p, to_body(clicks[p["frame"]], p["body"])) for p in staff_y if p["frame"] in clicks]
    body_points = np.array([b for _, b in clicked])
    widths = np.array([p["body"][3] for p, _ in clicked])
    print(f"\nTag clicks with a staff pose: {len(clicked)}")
    print(f"Tag position (shoulder widths; x along the model's L->R shoulders, y towards the hips): "
          f"mean ({body_points[:, 0].mean():.2f}, {body_points[:, 1].mean():.2f}), "
          f"std ({body_points[:, 0].std():.2f}, {body_points[:, 1].std():.2f})")
    right_side = int(np.sum(body_points[:, 0] > 0))
    print(f"Side: {right_side} of {len(clicked)} on the model's RIGHT-shoulder side, "
          f"{len(clicked) - right_side} on its LEFT-shoulder side")
    tag_units = (TAG_SIZE[0] / np.median(widths), TAG_SIZE[1] / np.median(widths))
    print(f"Median shoulder width {np.median(widths):.1f} px -> tag {tag_units[0]:.2f} x {tag_units[1]:.2f} "
          f"shoulder widths")

    # 3. two folds over the staff y people: window from one half's clicks, y scored on the other half
    template = make_template(args.pdf)
    ordered = sorted(staff_y, key=lambda p: p["frame"])
    folds = [ordered[0::2], ordered[1::2]]
    print(f"\nSearch positions per crop: before {previous_positions()} (unwindowed, 9 variants)")
    for k, (build, test) in enumerate(((folds[0], folds[1]), (folds[1], folds[0]))):
        points = [to_body(clicks[p["frame"]], p["body"]) for p in build if p["frame"] in clicks]
        windows = windows_from_clicks(points, tag_units, args.windows)
        name = f"{args.windows}_window_from_half{k}"
        described = ", ".join(f"centre ({c[0]:.2f}, {c[1]:.2f}) half-size ({h[0]:.2f}, {h[1]:.2f})"
                              for c, h in windows)
        print(f"\n=== {name}: {len(points)} clicks -> {described} shoulder widths; y scored on the other half")
        scored = {"y": [], "n": [], "others": []}
        positions = []
        for group, people in (("y", test), ("n", [p for p in staff if p["label"] == "n"]), ("others", others)):
            for person in people:
                result = match_in_windows(person["canvas"], person["body"][3], windows, template, tag_units)
                scored[group].append((result[0], person, result))
                positions.append(result[3])
        print(f"Search positions per crop now: median {np.median(positions):.0f} (template + mirror)")
        scores = {g: [s for s, _, _ in v] for g, v in scored.items()}
        print_distributions(scores, decimals=2)
        print_threshold_table(scores, THRESHOLDS)
        print_track_table(scores, THRESHOLDS)
        save_review(name, heapq.nlargest(TOP_OTHERS, scored["others"], key=lambda s: s[0]),
                    heapq.nsmallest(BOTTOM_Y, scored["y"], key=lambda s: s[0]), windows, tag_units)
    print(f"\nReview images saved to {REVIEW_DIR}")


if __name__ == "__main__":
    main()
