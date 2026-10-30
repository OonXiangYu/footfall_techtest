"""End-to-end staff detection: detect -> track -> link tracklets -> staff chains -> per-frame output.

    python src/staff_detect.py --video data/sample.mp4 --out outputs/run [--tag review|oracle]

review (default): a person confirms the staff among candidates ranked by the tag model (semi-automatic);
oracle: the staff chains come from the hand tag labels (data/tag_gt.csv), as a reference.
"""

import argparse
import csv
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from appearance import load_or_compute_appearance, match_poses, similarity, track_ends
from crops import box_chest_point
from detections import choose_view, detect_people, duplicate_rows, keep_rows, rows_per_frame
from fisheye import ANGLE_STEPS, box_outline, circle_for, owning_box, to_original_point, to_panorama_point
from linking import (LINK_GAP, LINK_MARGIN, LINK_RATIO, LINK_SPEED, LONG_DISTANCE, LONG_GAP, MAX_LINK_DIST,
                     chain_ids, link_tracks, track_summaries)
from pose import POSE_WEIGHTS, body_canvas, body_frame, keypoints_to_original, run_pose
from review import (CONFIRM_WINDOW, EXPAND_SIMILARITY, PER_PAGE, candidates, load_confirmed, review,
                    save_confirmed, staff_from_clicks)
from labels import DATA, ROOT, interpolate_chest, load_tag_labels
from tag_model import best_position, load_model
from tracker import Tracker

PROBE_FRAMES = 50
STAFF_COLOR = (0, 200, 0)
OTHER_COLOR = (170, 170, 170)
SWITCH_PENALTY = 0.10  # staff path: cost of changing chain, as a fraction of the radius (in px of jump)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unwarp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--tag", choices=["review", "oracle"], default="review",
                        help="review: a person confirms the staff among ranked candidates (semi-automatic); "
                             "oracle: the hand tag labels (--tag-labels), as a reference")
    parser.add_argument("--confirmed", help="--tag review: start the review page from these confirmations "
                                            "(with --no-review: use them as they are, without the page)")
    parser.add_argument("--no-review", dest="review", action="store_false",
                        help="--tag review --confirmed: do not open the review page")
    parser.add_argument("--expand-similarity", type=float, default=EXPAND_SIMILARITY,
                        help="--tag review: torso similarity for adding chains that look like a confirmed one")
    parser.add_argument("--no-expand", dest="expand", action="store_false",
                        help="--tag review: only the confirmed chains are staff")
    parser.add_argument("--confirm-scope", choices=["chain", "window"], default="chain",
                        help="--tag review: scope of confirmations that do not carry their own (the review page "
                             "stores one per click): the whole linked chain, or only --confirm-window seconds around "
                             "the clicked frame")
    parser.add_argument("--per-page", type=int, default=PER_PAGE,
                        help="--tag review: candidates per review page (4 per row)")
    parser.add_argument("--ui-height", type=int, default=None,
                        help="--tag review: maximum review window height in px (default: fit the screen)")
    parser.add_argument("--confirm-window", type=float, default=CONFIRM_WINDOW,
                        help="--tag review --confirm-scope window: seconds either side of the click")
    parser.add_argument("--tag-labels", help="tag_gt.csv (y/n/u per frame), required for --tag oracle")
    parser.add_argument("--gt", default=str(DATA / "gt.csv"),
                        help="hand-labelled chest points; --tag oracle uses them to locate the tag")
    parser.add_argument("--tag-model", default=str(ROOT / "models" / "tag_classifier.npz"),
                        help="--tag review: tag model that ranks the candidates (tools/train_tag_model.py)")
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
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="keep a box that lies inside another box of similar width (dropped by default)")
    parser.add_argument("--no-link-appearance", dest="link_appearance", action="store_false",
                        help="link without torso appearance (no ratio-test rescue, no still-person pass)")
    parser.add_argument("--long-gap", type=float, default=LONG_GAP,
                        help="still-person linking: max seconds, for close and similar-looking track ends")
    parser.add_argument("--long-distance", type=float, default=LONG_DISTANCE,
                        help="still-person linking: max distance, fraction of the radius")
    parser.add_argument("--link-ratio", type=float, default=LINK_RATIO,
                        help="ratio test for links: only link when clearly the best option; 0 turns it off")
    parser.add_argument("--tag-every", type=int, default=5, help="--tag review: score tags every N frames to rank the candidates")
    parser.add_argument("--max-hold", type=float, default=0.5,
                        help="seconds a staff position may be interpolated without a detection; longer gaps are "
                             "reported as not present")
    parser.add_argument("--switch-penalty", type=float, default=SWITCH_PENALTY,
                        help="staff path: cost of changing chain, fraction of the radius")
    parser.add_argument("--debug", action="store_true",
                        help="annotated video: also draw every other chain with its id (default: only the staff)")
    parser.add_argument("--min-hits", type=int, default=2, help="--tag oracle: tag hits for a chain to count as staff")
    args = parser.parse_args(argv)
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


def frame_size(path):
    cap = cv2.VideoCapture(path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return size


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


def link(detections, track_ids, max_gap, speed, margin, max_distance, ratio=None, hists=None,
         long_gap=None, long_distance=None):
    summaries = track_summaries(detections["frame"], detections["point"], track_ids)
    appearance = None
    if hists is not None:
        ends = track_ends(detections["frame"], track_ids, hists)

        def end_to_start(a, b):  # similarity of a's end and b's start
            return similarity(ends[a][1], ends[b][0])
        appearance = end_to_start
    successor = link_tracks(summaries, max_gap, speed, margin, max_distance, ratio, appearance,
                            long_gap if hists is not None else None, long_distance)
    chains = chain_ids(summaries, successor)
    return np.array([chains[t] for t in track_ids], int), len(summaries), len(successor)


# ---- stage 4: tag evidence -----------------------------------------------------------------------

def score_tags(args, detections, frame_rows, model):
    """Every tag_every frames: pose on the panorama, matched to the detector's boxes; each matched person's
    best tag position in the two side windows is scored. Returns the score per detection row (NaN where not
    scored)."""
    pose_model = YOLO(str(ROOT / POSE_WEIGHTS))
    center, radius = tuple(detections["center"]), float(detections["radius"])
    scores = np.full(len(detections["frame"]), np.nan)
    for index, frame in tqdm(read_video(args.video), total=len(frame_rows), desc="Tag scoring"):
        if index % args.tag_every or index >= len(frame_rows):
            continue
        rows = frame_rows[index]
        pose_boxes, keypoints, confs = run_pose(pose_model, frame, center, radius)
        for det, pose in match_poses(detections["box"][rows], pose_boxes).items():
            body = body_frame(keypoints_to_original(keypoints[pose], center, radius), confs[pose])
            if body is None:
                continue
            scores[rows[det]], _ = best_position(model, body_canvas(frame, body), body[3])
    return scores


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

def staff_per_frame(detections, chains, staff_chains, weights, frame_count, max_hold, switch_penalty):
    """Per frame: (x, y, chain, detection row or None if interpolated) of the staff, or None.

    Every frame with staff-chain detections offers one candidate per staff chain (its most confident box).
    One path through these candidates is chosen by dynamic programming, minimising the summed jumps between
    consecutive picks plus switch_penalty px per change of chain, so overlapping staff chains cannot make
    the output flicker between people. Gaps in the path up to max_hold frames are interpolated; longer gaps
    are not present (a position is never held without a detection for longer). weights is unused here but
    kept so callers can pass confirmed / added information.
    """
    best = {}  # (frame, chain) -> row
    for row in np.flatnonzero(np.isin(chains, list(staff_chains))):
        key = (int(detections["frame"][row]), int(chains[row]))
        if key not in best or detections["conf"][row] > detections["conf"][best[key]]:
            best[key] = row
    per_frame = defaultdict(list)
    for (frame, chain), row in best.items():
        per_frame[frame].append((chain, row))
    frames = sorted(per_frame)
    result = [None] * frame_count
    if not frames:
        return result

    # dynamic programming over the frames that have candidates
    costs = [np.zeros(len(per_frame[frames[0]]))]
    back = [None]
    for prev_frame, frame in zip(frames, frames[1:]):
        prev, cur = per_frame[prev_frame], per_frame[frame]
        prev_points = detections["chest"][[r for _, r in prev]]
        cur_points = detections["chest"][[r for _, r in cur]]
        jump = np.linalg.norm(cur_points[:, None, :] - prev_points[None, :, :], axis=2)
        switch = np.array([[c != p for p, _ in prev] for c, _ in cur], float) * switch_penalty
        total = costs[-1][None, :] + jump + switch
        back.append(total.argmin(axis=1))
        costs.append(total.min(axis=1))
    k = int(np.argmin(costs[-1]))
    path = []
    for i in range(len(frames) - 1, -1, -1):
        path.append((frames[i], *per_frame[frames[i]][k]))
        if back[i] is not None:
            k = int(back[i][k])
    path.reverse()

    for frame, chain, row in path:
        x, y = detections["chest"][row]
        result[frame] = (float(x), float(y), chain, int(row))
    for (a, chain_a, row_a), (b, _, row_b) in zip(path, path[1:]):
        if 1 < b - a <= max_hold + 1:
            pa, pb = detections["chest"][row_a], detections["chest"][row_b]
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                point = (1 - t) * pa + t * pb
                result[f] = (float(point[0]), float(point[1]), chain_a, None)
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


def staff_outline(detections, row):
    """The staff box in the original image: the panorama box mapped back, or the box itself."""
    box = detections["box"][row]
    if bool(detections["use_pano"]):
        return box_outline(box, (tuple(detections["center"]), float(detections["radius"])))
    x1, y1, x2, y2 = box.astype(int)
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)


def staff_label(image, text, origin, scale, thickness):
    """The staff label, moved inside the frame when the person is near an edge."""
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = min(max(origin[0], 4), image.shape[1] - width - 4)
    y = min(max(origin[1], height + 4), image.shape[0] - 4)
    label(image, text, (x, y), STAFF_COLOR, scale, thickness)


def write_video(path, video, fps, detections, frame_rows, chains, staff, debug=False):
    """The staff as a green box labelled "STAFF (x, y)" with the position in original-image pixels (a thin
    marker on short interpolated stretches); with debug, every other chain as a grey point with its id."""
    writer = None
    for index, frame in read_video(video):
        if index >= len(frame_rows):
            break
        entry = staff[index]
        if debug:
            for row in frame_rows[index]:
                if entry and chains[row] == entry[2]:
                    continue
                x, y = detections["chest"][row].astype(int)
                cv2.circle(frame, (x, y), 4, OTHER_COLOR, -1)
                label(frame, str(chains[row]), (x + 6, y - 4), OTHER_COLOR, 0.4)
        if entry:
            # the position as in staff.csv: original-image pixels of the chest point
            x, y = round(entry[0]), round(entry[1])
            name = f"STAFF ({x}, {y})" + (f" chain {entry[2]}" if debug else "")
            if entry[3] is not None:
                outline = staff_outline(detections, entry[3])
                cv2.polylines(frame, [outline], True, STAFF_COLOR, 3)
                top = outline[outline[:, 1].argmin()]
                staff_label(frame, name, (int(top[0]) - 70, int(top[1]) - 8), 0.7, 2)
            else:  # interpolated for at most max-hold: a light marker, no box
                cv2.drawMarker(frame, (x, y), STAFF_COLOR, cv2.MARKER_CROSS, 18, 2)
                staff_label(frame, name, (x + 12, y - 10), 0.6, 1)
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


def frame_ranges(frames):
    """Sorted frame numbers grouped into consecutive (first, last) ranges."""
    groups = []
    for f in frames:
        if groups and f == groups[-1][1] + 1:
            groups[-1][1] = f
        else:
            groups.append([f, f])
    return [tuple(g) for g in groups]


def ranges_text(staff, fps):
    """The frames where the staff member is present (and not found), as ranges, with the totals."""
    present = [f for f, entry in enumerate(staff) if entry is not None]
    absent = [f for f, entry in enumerate(staff) if entry is None]
    fmt = lambda groups: ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in groups) or "none"  # noqa: E731
    return (f"present: {fmt(frame_ranges(present))}\n"
            f"total: {len(present)} of {len(staff)} frames ({len(present) / len(staff):.1%}; "
            f"{len(present) / fps:.1f} of {len(staff) / fps:.1f} s at {fps:.0f} fps)\n"
            f"not found: {fmt(frame_ranges(absent))}\n")


def write_trajectory(path, staff, fps, center, radius, size):
    """The staff path in original-image pixels on a blank background with the fisheye circle, coloured by time;
    gaps where the staff member is not found are dotted grey with their frames."""
    width, height = size
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.add_patch(Circle(center, radius, fill=False, color="#b4b2a9", linewidth=1))
    cmap = LinearSegmentedColormap.from_list("time", plt.get_cmap("Blues")(np.linspace(0.35, 1.0, 256)))
    norm = plt.Normalize(0, (len(staff) - 1) / fps)
    segments, times = [], []
    for f in range(1, len(staff)):
        if staff[f] is not None and staff[f - 1] is not None:
            segments.append([staff[f - 1][:2], staff[f][:2]])
            times.append(f / fps)
    lines = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lines.set_array(np.array(times))
    ax.add_collection(lines)
    gaps = frame_ranges([f for f, entry in enumerate(staff) if entry is None])
    for first, last in gaps:
        before = staff[first - 1] if first > 0 else None
        after = staff[last + 1] if last + 1 < len(staff) else None
        ends = [e[:2] for e in (before, after) if e is not None]
        if len(ends) == 2:
            ax.plot(*zip(*ends), color="#7a7974", linestyle=":", linewidth=1.5)
        for x, y in ends:
            ax.plot(x, y, "o", markersize=6, markerfacecolor="white", markeredgecolor="#52514e")
    # gaps are numbered on the plot (staggered, several can start at the same desk) and listed in a box
    for number, (first, last) in enumerate(gaps, 1):
        before = staff[first - 1] if first > 0 else staff[last + 1]
        ax.annotate(str(number), before[:2], xytext=(-14, 10 + 10 * ((number - 1) % 4)), textcoords="offset points",
                    fontsize=8, color="#52514e", fontweight="bold")
    if gaps:
        listing = "\n".join(f"{n}: frames {a}-{b} ({(b - a + 1) / fps:.1f} s)" for n, (a, b) in enumerate(gaps, 1))
        ax.text(0.99, 0.02, "not found\n" + listing, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color="#52514e", bbox={"facecolor": "white", "edgecolor": "#d3d1c7", "boxstyle": "round"})
    present = [entry for entry in staff if entry is not None]
    for entry, text in ((present[0], "start"), (present[-1], "end")):
        ax.plot(entry[0], entry[1], "s", markersize=7, color="#2c2c2a")
        ax.annotate(text, entry[:2], xytext=(6, -10), textcoords="offset points", fontsize=8, color="#2c2c2a")
    ax.plot([], [], color="#7a7974", linestyle=":", label="not found (gap; open circles: last / next position)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, label="time (s)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image coordinates: y down
    ax.set_aspect("equal")
    ax.set_xlabel("x (px, original image)", color="#52514e")
    ax.set_ylabel("y (px, original image)", color="#52514e")
    ax.set_title(f"Staff path: present in {len(present)} of {len(staff)} frames, {len(gaps)} gap(s)", loc="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


# ---- main ----------------------------------------------------------------------------------------

def build_chains(args, out, fps, frame_count, timings):
    """Stages 1-3: detection (cached), appearance (cached), duplicate removal, tracking, linking."""
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
    all_rows = len(detections["frame"])

    hists = None
    if args.link_appearance or args.tag == "review":
        start = time.perf_counter()
        hists, from_pose = load_or_compute_appearance(out / "appearance.npz", args.video, detections, frame_rows)
        timings["appearance"] = time.perf_counter() - start
        print(f"   appearance: torso colour histograms, {from_pose.mean():.0%} from the pose, rest from the chest crop")
    keep = np.ones(all_rows, bool)
    if not args.keep_duplicates:
        keep = ~duplicate_rows(detections, frame_rows)
        detections = keep_rows(detections, keep)
        frame_rows = rows_per_frame(detections)
        hists = hists[keep] if hists is not None else None
        print(f"   duplicates: dropped {int((~keep).sum())} boxes lying inside another box of similar width")

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
    link_hists = hists if args.link_appearance else None
    if link_hists is not None:
        print(f"   appearance linking: ratio-test rescue; still-person pass up to {args.long_gap:g} s "
              f"within {args.long_distance * radius:.1f} px")
    chains, track_count, link_count = link(detections, track_ids, max_gap, speed, margin, cap, ratio, link_hists,
                                           round(args.long_gap * fps), args.long_distance * radius)
    timings["linking"] = time.perf_counter() - start
    print(f"3. linking: {link_count} links, {track_count} tracks -> {len(set(chains.tolist()))} chains")
    return {"detections": detections, "frame_rows": frame_rows, "keep": keep, "all_rows": all_rows,
            "track_ids": track_ids, "hists": hists, "chains": chains, "radius": radius, "frame_count": frame_count, "max_gap": max_gap,
            "max_hold": round(args.max_hold * fps), "switch_penalty": args.switch_penalty * radius}


def review_scores(args, out, state):
    """Tag scores (all-data model) of every scored detection row, cached in <out>/tag_scores_review.npz."""
    path = out / "tag_scores_review.npz"
    model_path = Path(args.tag_model)
    key = json.dumps({"model": str(model_path.resolve()), "mtime": model_path.stat().st_mtime,
                      "every": args.tag_every, "rows": int(state["keep"].sum())})
    if path.exists():
        cached = np.load(path)
        if str(cached["key"]) == key:
            return cached["score"]
    scores = score_tags(args, state["detections"], state["frame_rows"], load_model(model_path))
    np.savez(path, key=key, score=scores)
    return scores


def review_staff(args, out, fps, state):
    """--tag review: confirmed clicks (from the review page, optionally starting from a confirmed.json, or
    from the file alone with --no-review), each scoped to its whole chain or a window around the click (stored
    per click; --confirm-scope for clicks without one), plus appearance expansion. Returns (chain ids after
    scoping, staff chains, {chain: 2 confirmed / 1 added})."""
    detections, chains, keep = state["detections"], state["chains"], state["keep"]
    kept_row_of = np.full(state["all_rows"], -1)
    kept_row_of[keep] = np.arange(int(keep.sum()))
    previous = load_confirmed(args.confirmed, chains, kept_row_of) if args.confirmed else []
    if args.confirmed:
        print(f"   confirmed: {len(previous)} click(s) on {len({c for c, *_ in previous})} chain(s) "
              f"from {args.confirmed}")
    if args.confirmed and not args.review:
        clicks = [(chain, frame, scope) for chain, frame, _, scope in previous]
    else:
        scores = review_scores(args, out, state)
        ranked = candidates(detections, chains, scores, fps)
        # earlier confirmations: select the matching candidates; ones that are no longer candidates are
        # shown first (labelled "prev") so they can still be removed
        row_index = {c["row"]: i for i, c in enumerate(ranked)}
        extra = [{"row": row, "frame": frame, "chain": chain, "rank": "prev",
                  "score": float(np.nan_to_num(scores[row]))}
                 for chain, frame, row, _ in previous if row not in row_index]
        shown = extra + ranked
        preselected = {}
        for chain, frame, row, scope in previous:
            index = row_index[row] + len(extra) if row in row_index else next(
                i for i, c in enumerate(extra) if c["row"] == row)
            preselected[index] = scope or args.confirm_scope
        print(f"   review: {len(ranked)} candidates ({len(preselected)} already selected); opening the review page")
        selected = review(args.video, detections, chains, shown, bool(detections["use_pano"]), args.per_page,
                          args.ui_height, preselected)
        picked = [(shown[i], scope) for i, scope in selected.items()]
        save_confirmed(out / "confirmed.json", picked, np.flatnonzero(keep))
        clicks = [(c["chain"], c["frame"], scope) for c, scope in picked]
        print(f"   confirmed {len(picked)} candidate(s) on {len({c for c, *_ in clicks})} chain(s) "
              f"({sum(s == 'window' for *_, s in clicks)} with window scope); saved {out / 'confirmed.json'}")
    scoped, staff, weights, added = staff_from_clicks(
        detections, chains, state["hists"], clicks, args.confirm_scope, round(args.confirm_window * fps),
        args.expand_similarity if args.expand else None, state["radius"])
    print(f"   confirm scope: per click (default {args.confirm_scope}, window +-{args.confirm_window:g} s); "
          f"appearance expansion "
          f"{f'(similarity >= {args.expand_similarity}): added {len(added)} chain(s)' if args.expand else 'off'}")
    return scoped, staff, Counter(weights)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps, frame_count = video_info(args.video)
    print(f"Video: {args.video}  {frame_count} frames at {fps:.2f} fps")
    timings = {}
    state = build_chains(args, out, fps, frame_count, timings)
    detections, frame_rows, chains = state["detections"], state["frame_rows"], state["chains"]
    keep, all_rows, frame_count = state["keep"], state["all_rows"], state["frame_count"]

    start = time.perf_counter()
    if args.tag == "review":
        chains, staff_chains, hits = review_staff(args, out, fps, state)
    else:
        hits = oracle_tag_hits(args, detections, frame_rows, chains)
        staff_chains = {chain for chain, count in hits.items() if count >= args.min_hits}
    timings["tag evidence"] = time.perf_counter() - start
    # chain of every detection row plus the tag evidence, for per-chain evaluation
    # one entry per row of detections.npz; -1 for dropped duplicates
    chain_per_row = np.full(all_rows, -1, int)
    chain_per_row[keep] = chains
    np.savez(out / "chains.npz", chain=chain_per_row, hit_chain=np.array(list(hits), int),
             hit_count=np.array([round(v) for v in hits.values()], int),
             staff_chain=np.array(sorted(staff_chains), int))
    if args.tag == "review":
        print(f"4. staff (review): {sum(v == 2 for v in hits.values())} confirmed + "
              f"{sum(v == 1 for v in hits.values())} added chain(s)")
    else:
        top = ", ".join(f"{c}:{n}" for c, n in hits.most_common(8))
        print(f"4. tag evidence (oracle): {len(staff_chains)} staff chain(s) with >= {args.min_hits} hits; "
              f"top chains (id:hits) {top or 'none'}")

    start = time.perf_counter()
    staff = staff_per_frame(detections, chains, staff_chains, hits, frame_count, state["max_hold"],
                            state["switch_penalty"])
    write_csv(out / "staff.csv", staff)
    write_timeline(out / "timeline.png", staff, fps)
    write_trajectory(out / "trajectory.png", staff, fps, tuple(detections["center"]), float(detections["radius"]),
                     frame_size(args.video))
    ranges = ranges_text(staff, fps)
    (out / "staff_ranges.txt").write_text(ranges, encoding="utf-8")
    write_video(out / "annotated.mp4", args.video, fps, detections, frame_rows, chains, staff, args.debug)
    timings["output"] = time.perf_counter() - start
    present = sum(entry is not None for entry in staff)
    print(f"5. output: staff present in {present} of {frame_count} frames -> {out}")

    print("\nStage timings: " + ", ".join(f"{name} {seconds:.1f} s" for name, seconds in timings.items())
          + f", total {sum(timings.values()):.1f} s")
    print(f"\nStaff frames (also in {out / 'staff_ranges.txt'}):\n{ranges}", end="")


if __name__ == "__main__":
    main()
