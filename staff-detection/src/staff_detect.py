"""End-to-end staff detection: detect -> track -> link tracklets -> tag evidence -> per-frame output.

    python src/staff_detect.py --video data/sample.mp4 --out outputs/run [--tag auto|oracle]
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
import itertools
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO

from crops import box_chest_point
from detections import choose_view, detect_people, rows_per_frame
from fisheye import ANGLE_STEPS, circle_for, owning_box, to_original_point, to_panorama_point
from linking import (LINK_GAP, LINK_MARGIN, LINK_RATIO, LINK_SPEED, MAX_LINK_DIST, chain_ids, link_tracks,
                     track_summaries)
from pose import POSE_WEIGHTS, body_canvas, body_frame, keypoints_to_original, run_pose
from tag_crops import interpolate_chest
from tag_model import load_tag_model, tag_probability
from tracker import Tracker

ROOT = Path(__file__).resolve().parent.parent
PROBE_FRAMES = 50
STAFF_COLOR = (0, 200, 0)
OTHER_COLOR = (170, 170, 170)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unwarp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--tag", choices=["auto", "oracle"], default="auto")
    parser.add_argument("--tag-labels", help="tag_gt.csv (y/n/u per frame), required for --tag oracle")
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"),
                        help="hand-labelled chest points; --tag oracle uses them to locate the tag")
    parser.add_argument("--tag-model", default=str(ROOT / "models" / "tag_classifier.npz"),
                        help="--tag auto: tag classifier from tag_classifier.py")
    parser.add_argument("--imgsz", type=int, default=1440)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-dist", type=float, default=0.05,
                        help="tracker gate, fraction of the radius (0.05 ~ staff max speed; wider gates swap people)")
    parser.add_argument("--max-missed", type=float, default=1.0, help="seconds a track may go unmatched")
    parser.add_argument("--link-gap", type=float, default=LINK_GAP, help="max seconds between linked tracklets")
    parser.add_argument("--link-speed", type=float, default=LINK_SPEED,
                        help="link distance per frame of gap, fraction of the radius (~ staff median speed)")
    parser.add_argument("--link-margin", type=float, default=LINK_MARGIN,
                        help="extra link distance, fraction of the radius")
    parser.add_argument("--max-link-dist", type=float, default=MAX_LINK_DIST,
                        help="cap on the link distance, fraction of the radius")
    parser.add_argument("--link-ratio", type=float, default=LINK_RATIO,
                        help="ratio test for links: only link when clearly the best option; 0 turns it off")
    parser.add_argument("--tag-threshold", type=float, default=None,
                        help="--tag auto: classifier probability for a hit (default: the model's operating point)")
    parser.add_argument("--background", type=float, default=None,
                        help="--tag auto: hit rate of non-staff people (default: measured, stored in the model)")
    parser.add_argument("--rate-ratio", type=float, default=2.0,
                        help="--tag auto: a staff chain's hit rate must be this many times the background rate")
    parser.add_argument("--tag-every", type=int, default=5, help="--tag auto: score tags every N frames")
    parser.add_argument("--min-hits", type=int, default=2, help="tag hits for a chain to count as staff")
    args = parser.parse_args()
    if args.tag == "oracle" and not args.tag_labels:
        parser.error("--tag oracle needs --tag-labels")
    return args


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    info = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def read_video(path):
    """Yield (index, frame) sequentially."""
    cap = cv2.VideoCapture(path)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield index, frame
        index += 1
    cap.release()


# ---- stage 1: detection ------------------------------------------------------------------------

def cache_key(args):
    stat = Path(args.video).stat()
    return json.dumps({"video": str(Path(args.video).resolve()), "size": stat.st_size,
                       "mtime": int(stat.st_mtime), "imgsz": args.imgsz, "conf": args.conf,
                       "unwarp": args.unwarp})


def load_cached_detections(path, key):
    if not path.exists():
        return None
    detections = dict(np.load(path))
    return detections if str(detections["key"]) == key else None


def run_detection(args, frame_count):
    model = YOLO(str(ROOT / "yolov8n.pt"))
    frames = read_video(args.video)
    # the first frames decide the circle and the view, then get detected like the rest
    probe = list(itertools.islice(frames, PROBE_FRAMES))
    circle = circle_for(probe[0][1])
    use_pano = choose_view(args.unwarp, model, [f for _, f in probe], circle, args.imgsz, args.conf)
    center, radius = circle

    columns = defaultdict(list)
    for index, frame in tqdm(itertools.chain(probe, frames), total=frame_count, desc="Detecting"):
        boxes, confs = detect_people(model, frame, circle, use_pano, args.imgsz, args.conf)
        for box, conf in zip(boxes, confs):
            mid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if use_pano:
                centre_point = to_original_point(*mid, center, radius)
                chest_point = box_chest_point(box, center, radius)
            else:
                # without the panorama there is no "up" in the box, so the box centre stands in
                centre_point = chest_point = mid
            columns["frame"].append(index)
            columns["box"].append(box)
            columns["point"].append(centre_point)
            columns["chest"].append(chest_point)
            columns["conf"].append(conf)
    return {"frame": np.array(columns["frame"], int),
            "box": np.array(columns["box"], float).reshape(-1, 4),
            "point": np.array(columns["point"], float).reshape(-1, 2),
            "chest": np.array(columns["chest"], float).reshape(-1, 2),
            "conf": np.array(columns["conf"], float),
            "frame_count": frame_count, "center": np.array(center), "radius": radius,
            "use_pano": use_pano}


# ---- stage 2-3: tracking and linking -------------------------------------------------------------

def track(detections, frame_rows, max_dist, max_missed):
    tracker = Tracker(max_dist, max_missed)
    ids = np.zeros(len(detections["frame"]), int)
    for rows in frame_rows:
        ids[rows] = tracker.update(detections["point"][rows])
    return ids


def link(detections, track_ids, max_gap, speed, margin, max_distance, ratio=None):
    summaries = track_summaries(detections["frame"], detections["point"], track_ids)
    successor = link_tracks(summaries, max_gap, speed, margin, max_distance, ratio)
    chains = chain_ids(summaries, successor)
    return np.array([chains[t] for t in track_ids], int), len(summaries), len(successor)


# ---- stage 4: tag evidence -----------------------------------------------------------------------

POSE_MATCH_IOU = 0.3  # a pose box belongs to a detector box if they overlap at least this much


def box_iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_poses(det_boxes, pose_boxes):
    """{detection index: pose index}, one-to-one by IoU (Hungarian), IoU >= POSE_MATCH_IOU."""
    if len(det_boxes) == 0 or len(pose_boxes) == 0:
        return {}
    iou = np.array([[box_iou(d, p) for p in pose_boxes] for d in det_boxes])
    rows, cols = linear_sum_assignment(-iou)
    return {r: c for r, c in zip(rows, cols) if iou[r, c] >= POSE_MATCH_IOU}


def auto_tag_hits(args, detections, frame_rows, chains, model, threshold):
    """Every tag_every frames: pose on the panorama, matched to the detector's boxes; each matched
    person's two shoulder-placed tag windows are scored by the classifier. Returns per chain
    (hits, frames scored)."""
    pose_model = YOLO(str(ROOT / POSE_WEIGHTS))
    center, radius = tuple(detections["center"]), float(detections["radius"])
    hits, scored = Counter(), Counter()
    for index, frame in tqdm(read_video(args.video), total=len(frame_rows), desc="Tag scoring"):
        if index % args.tag_every or index >= len(frame_rows):
            continue
        rows = frame_rows[index]
        pose_boxes, keypoints, confs = run_pose(pose_model, frame, center, radius)
        for det, pose in match_poses(detections["box"][rows], pose_boxes).items():
            body = body_frame(keypoints_to_original(keypoints[pose], center, radius), confs[pose])
            if body is None:
                continue
            chain = int(chains[rows[det]])
            scored[chain] += 1
            if tag_probability(model, body_canvas(frame, body), body[3]) >= threshold:
                hits[chain] += 1
    return hits, scored


def load_tag_labels(path):
    with open(path, newline="") as f:
        return {int(row["frame"]): row["label"] for row in csv.DictReader(f)}


def oracle_tag_hits(args, detections, frame_rows, chains):
    """On frames labelled "y", a hit for the chain of the box the staff chest point belongs to."""
    chest = interpolate_chest(args.gt)
    center, radius = detections["center"], float(detections["radius"])
    use_pano = bool(detections["use_pano"])
    hits = Counter()
    for frame, label in load_tag_labels(args.tag_labels).items():
        if label != "y" or frame not in chest or frame >= len(frame_rows):
            continue
        point = to_panorama_point(*chest[frame], center, radius) if use_pano else chest[frame]
        if point is None:
            continue
        rows = frame_rows[frame]
        owner = owning_box(point, detections["box"][rows], ANGLE_STEPS if use_pano else None)
        if owner is not None:
            hits[int(chains[rows[owner]])] += 1
    return hits


# ---- stage 5: output -----------------------------------------------------------------------------

def staff_per_frame(detections, chains, staff_chains, hits, frame_count, max_gap):
    """Per frame: (x, y, chain) of the staff, or None. Short gaps inside a staff chain are interpolated."""
    tracks = defaultdict(dict)  # chain -> {frame: chest point}
    for row in np.flatnonzero(np.isin(chains, list(staff_chains))):
        frame, chain = int(detections["frame"][row]), int(chains[row])
        # if a chain has two boxes in one frame, keep the more confident one
        current = tracks[chain].get(frame)
        if current is None or detections["conf"][row] > current[1]:
            tracks[chain][frame] = (detections["chest"][row], detections["conf"][row])

    candidates = defaultdict(list)  # frame -> [(hits, chain, point)]
    for chain, points in tracks.items():
        frames = sorted(points)
        for a, b in zip(frames, frames[1:] + [None]):
            candidates[a].append((hits[chain], chain, points[a][0]))
            if b is not None and 1 < b - a <= max_gap:
                for f in range(a + 1, b):
                    t = (f - a) / (b - a)
                    candidates[f].append((hits[chain], chain, (1 - t) * points[a][0] + t * points[b][0]))
    result = [None] * frame_count
    for frame, options in candidates.items():
        _, chain, point = max(options, key=lambda o: o[0])  # most tag evidence wins
        result[frame] = (float(point[0]), float(point[1]), chain)
    return result


def write_csv(path, staff):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "present", "x", "y", "chain"])
        for frame, entry in enumerate(staff):
            if entry is None:
                writer.writerow([frame, 0, "", "", ""])
            else:
                writer.writerow([frame, 1, f"{entry[0]:.1f}", f"{entry[1]:.1f}", entry[2]])


def label(image, text, origin, color, scale=0.5, thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def write_video(path, video, fps, detections, frame_rows, chains, staff):
    writer = None
    for index, frame in read_video(video):
        if index >= len(frame_rows):
            break
        staff_chain = staff[index][2] if staff[index] else None
        for row in frame_rows[index]:
            if chains[row] == staff_chain:
                continue
            x, y = detections["chest"][row].astype(int)
            cv2.circle(frame, (x, y), 4, OTHER_COLOR, -1)
            label(frame, str(chains[row]), (x + 6, y - 4), OTHER_COLOR, 0.4)
        if staff[index]:
            x, y = round(staff[index][0]), round(staff[index][1])
            cv2.circle(frame, (x, y), 12, STAFF_COLOR, 3)
            label(frame, f"STAFF ({staff_chain})", (x + 14, y - 10), STAFF_COLOR, 0.7, 2)
        label(frame, f"frame {index}", (10, 30), (255, 255, 255), 0.8, 2)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
    writer.release()


def write_timeline(path, staff, fps):
    present = np.array([entry is not None for entry in staff], float)
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.fill_between(np.arange(len(present)), present, step="mid", color="#2a78d6", linewidth=0)
    ax.set_yticks([0, 1], ["absent", "present"])
    ax.set_xlim(0, len(present))
    ax.set_xlabel(f"frame ({fps:.0f} fps)", color="#52514e")
    ax.set_title(f"Staff presence: {int(present.sum())} of {len(present)} frames", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


# ---- main ----------------------------------------------------------------------------------------

def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps, frame_count = video_info(args.video)
    print(f"Video: {args.video}  {frame_count} frames at {fps:.2f} fps")
    timings = {}

    start = time.perf_counter()
    cache_path, key = out / "detections.npz", cache_key(args)
    detections = load_cached_detections(cache_path, key)
    if detections is None:
        detections = run_detection(args, frame_count)
        np.savez(cache_path, key=key, **detections)
        print(f"1. detection: {len(detections['frame'])} boxes, cached to {cache_path}")
    else:
        print(f"1. detection: reused {cache_path} ({len(detections['frame'])} boxes)")
    timings["detection"] = time.perf_counter() - start
    frame_count = int(detections["frame_count"])  # frames actually read
    radius = float(detections["radius"])
    print(f"   view: {'panorama' if detections['use_pano'] else 'original frame'}, radius {radius:.0f} px")
    frame_rows = rows_per_frame(detections)

    start = time.perf_counter()
    track_ids = track(detections, frame_rows, args.max_dist * radius, round(args.max_missed * fps))
    timings["tracking"] = time.perf_counter() - start
    print(f"2. tracking: {len(set(track_ids.tolist()))} tracks")

    start = time.perf_counter()
    max_gap = round(args.link_gap * fps)
    speed, margin, cap = args.link_speed * radius, args.link_margin * radius, args.max_link_dist * radius
    print(f"   link gate: min({speed:.1f} px/frame x gap + {margin:.1f} px, {cap:.1f} px); "
          f"at the longest gap ({max_gap} frames) that is {min(speed * max_gap + margin, cap):.1f} px")
    ratio = args.link_ratio if args.link_ratio > 0 else None
    print(f"   link ratio test: {ratio or 'off'}")
    chains, track_count, link_count = link(detections, track_ids, max_gap, speed, margin, cap, ratio)
    timings["linking"] = time.perf_counter() - start
    print(f"3. linking: {link_count} links, {track_count} tracks -> {len(set(chains.tolist()))} chains")

    start = time.perf_counter()
    scored = Counter()
    if args.tag == "auto":
        model = load_tag_model(args.tag_model)
        threshold = args.tag_threshold if args.tag_threshold is not None else float(model["threshold"])
        background = args.background if args.background is not None else float(model["background"])
        print(f"   tag classifier: hit if p >= {threshold}; background rate {background:.1%}; staff chain needs "
              f">= {args.min_hits} hits and a hit rate >= {args.rate_ratio * background:.1%}")
        hits, scored = auto_tag_hits(args, detections, frame_rows, chains, model, threshold)
        staff_chains = {chain for chain, count in hits.items()
                        if count >= args.min_hits and count / scored[chain] >= args.rate_ratio * background}
    else:
        hits = oracle_tag_hits(args, detections, frame_rows, chains)
        staff_chains = {chain for chain, count in hits.items() if count >= args.min_hits}
    timings["tag evidence"] = time.perf_counter() - start
    # chain of every detection row plus the tag evidence, for per-chain evaluation
    extra = {"scored_count": np.array([scored[c] for c in hits], int)} if scored else {}
    np.savez(out / "chains.npz", chain=chains, hit_chain=np.array(list(hits), int),
             hit_count=np.array(list(hits.values()), int), staff_chain=np.array(sorted(staff_chains), int), **extra)
    top = ", ".join(f"{c}:{n}" + (f"/{scored[c]}" if scored else "") for c, n in hits.most_common(8))
    print(f"4. tag evidence ({args.tag}): {len(staff_chains)} staff chain(s) with >= {args.min_hits} hits; "
          f"top chains (id:hits{'/scored' if scored else ''}) {top or 'none'}")

    start = time.perf_counter()
    staff = staff_per_frame(detections, chains, staff_chains, hits, frame_count, max_gap)
    write_csv(out / "staff.csv", staff)
    write_timeline(out / "timeline.png", staff, fps)
    write_video(out / "annotated.mp4", args.video, fps, detections, frame_rows, chains, staff)
    timings["output"] = time.perf_counter() - start
    present = sum(entry is not None for entry in staff)
    print(f"5. output: staff present in {present} of {frame_count} frames -> {out}")

    print("\nStage timings: " + ", ".join(f"{name} {seconds:.1f} s" for name, seconds in timings.items())
          + f", total {sum(timings.values()):.1f} s")


if __name__ == "__main__":
    main()
