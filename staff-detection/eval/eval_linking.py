"""Evaluate the tracker gate and the tracklet-linking settings on a staff_detect.py run's detections.

    python eval/eval_linking.py outputs/run

Reads <run>/detections.npz and <run>/appearance.npz. A track (or chain) is "the staff" at a labelled
frame when its box owns the GT chest point; purity = labelled frames where each staff track is on the
GT point, over the labelled frames it exists on (a swap onto someone else lowers it, a break does not).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from appearance import similarity, track_ends
from detections import duplicate_rows, keep_rows, rows_per_frame
from fisheye import ANGLE_STEPS, owning_box, to_panorama_point
from labels import DATA, load_gt_points
from linking import (LINK_GAP, LINK_MARGIN, LINK_RATIO, LINK_SPEED, LONG_DISTANCE, LONG_GAP, MAX_LINK_DIST,
                     chain_ids, link_tracks, track_summaries)
from staff_detect import video_info
from tracker import Tracker

MAX_MISSED_SECONDS = 1.0
# tracker gate (fraction of the radius): ~ the staff max speed; wider gates swap people (low purity)
DEFAULT_GATE = 0.05
SETTINGS = [  # (name, max_dist as a fraction of the radius, use velocity)
    (f"dist{DEFAULT_GATE:.2f}", DEFAULT_GATE, True),
    ("dist0.10", 0.10, True),
    ("dist0.20", 0.20, True),
    (f"dist{DEFAULT_GATE:.2f}_novel", DEFAULT_GATE, False),
]
LINK_SETTINGS = [  # (name, max gap s, ratio or None, appearance rescue, still-person gap s or None, drop duplicates)
    ("gap1.5s", 1.5, None, False, None, False),
    ("ratio0.5", 1.5, 0.5, False, None, False),
    ("gap4s", 4.0, None, False, None, False),
    (f"default (r{LINK_RATIO}, {LINK_GAP:g}s)", LINK_GAP, LINK_RATIO, False, None, False),
    ("default + appearance", LINK_GAP, LINK_RATIO, True, None, False),
    (f"default + app + still {LONG_GAP:g}s", LINK_GAP, LINK_RATIO, True, LONG_GAP, False),
    ("default + dedupe", LINK_GAP, LINK_RATIO, False, None, True),
    ("default + dedupe + app", LINK_GAP, LINK_RATIO, True, None, True),
    (f"dedupe + app + still {LONG_GAP:g}s", LINK_GAP, LINK_RATIO, True, LONG_GAP, True),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output folder of staff_detect.py (its detections.npz and appearance.npz)")
    parser.add_argument("--video", default=str(DATA / "sample.mp4"))
    parser.add_argument("--gt", default=str(DATA / "gt.csv"))
    return parser.parse_args()


def count_id_changes(staff_ids):
    known = [i for i in staff_ids if i is not None]
    return sum(a != b for a, b in zip(known, known[1:]))


# median and max px per frame between consecutive labelled points
def staff_speed(gt_points):
    frames = sorted(gt_points)
    speeds = [np.hypot(*np.subtract(gt_points[b], gt_points[a])) / (b - a) for a, b in zip(frames, frames[1:])]
    return float(np.median(speeds)), float(np.max(speeds))


# run the tracker over every frame; returns the track id of every detection row
def run_tracker(detections, frame_rows, max_dist, max_missed, use_velocity):
    tracker = Tracker(max_dist, max_missed, use_velocity)
    ids = np.zeros(len(detections["frame"]), int)
    for rows in frame_rows:
        ids[rows] = tracker.update(detections["point"][rows])
    return ids


# staff track id at each GT frame (None if no box contains the point)
def staff_ids_at_gt(detections, frame_rows, ids, gt_points):
    center, radius = detections["center"], float(detections["radius"])
    staff = {}
    for frame, (x, y) in sorted(gt_points.items()):
        rows = frame_rows[frame]
        point = to_panorama_point(x, y, center, radius)
        owner = None if point is None else owning_box(point, detections["box"][rows], ANGLE_STEPS)
        staff[frame] = None if owner is None else int(ids[rows][owner])
    return staff


# for each staff track id: (labelled frames it is on the GT point, labelled frames it exists on).
# A break gives several pure ids (on == total); a swap gives an id that is also on someone else (on < total).
def track_purity(staff, ids, frame_rows):
    purity = {}
    for track_id in sorted({i for i in staff.values() if i is not None}):
        present = [f for f in staff if track_id in ids[frame_rows[f]]]
        on_point = sum(staff[f] == track_id for f in present)
        purity[track_id] = (on_point, len(present))
    return purity


def evaluate(name, detections, frame_rows, gt_points, max_dist_fraction, use_velocity):
    max_dist = max_dist_fraction * float(detections["radius"])
    max_missed = round(MAX_MISSED_SECONDS * float(detections["fps"]))
    ids = run_tracker(detections, frame_rows, max_dist, max_missed, use_velocity)
    staff = staff_ids_at_gt(detections, frame_rows, ids, gt_points)
    purity = track_purity(staff, ids, frame_rows)
    on_point, total = (sum(p[i] for p in purity.values()) for i in (0, 1))
    return {"name": name, "ids": ids, "staff": staff, "purity": purity,
            "purity_rate": on_point / total if total else float("nan"),
            "unique_ids": len(set(ids.tolist())),
            "median_length": float(np.median(list(Counter(ids.tolist()).values()))),
            "changes": count_id_changes(list(staff.values()))}


# tracker (DEFAULT_GATE) + linking; the chain that owns the GT point is the staff chain at that frame
def evaluate_linking(name, detections, frame_rows, gt_points, max_gap_seconds, ratio, hists=None, long_gap=None,
                     dedupe=False):
    if dedupe:
        keep = ~duplicate_rows(detections, frame_rows)
        detections = keep_rows(detections, keep)
        frame_rows = rows_per_frame(detections)
        hists = hists[keep] if hists is not None else None
    radius, fps = float(detections["radius"]), float(detections["fps"])
    ids = run_tracker(detections, frame_rows, DEFAULT_GATE * radius, round(MAX_MISSED_SECONDS * fps), True)
    summaries = track_summaries(detections["frame"], detections["point"], ids)
    appearance = None
    if hists is not None:
        ends = track_ends(detections["frame"], ids, hists)
        appearance = lambda a, b: similarity(ends[a][1], ends[b][0])  # noqa: E731  a's end vs b's start
    successor = link_tracks(summaries, round(max_gap_seconds * fps), LINK_SPEED * radius,
                            LINK_MARGIN * radius, MAX_LINK_DIST * radius, ratio, appearance,
                            round(long_gap * fps) if long_gap else None, LONG_DISTANCE * radius)
    chain_of = chain_ids(summaries, successor)
    chains = np.array([chain_of[t] for t in ids], int)
    staff = staff_ids_at_gt(detections, frame_rows, chains, gt_points)
    purity = track_purity(staff, chains, frame_rows)
    on_point, total = (sum(p[i] for p in purity.values()) for i in (0, 1))
    return {"name": name, "links": len(successor), "chains": len(set(chains.tolist())), "staff": staff,
            "purity": purity, "purity_rate": on_point / total if total else float("nan"),
            "covered": sum(s is not None for s in staff.values()), "changes": count_id_changes(list(staff.values()))}


def print_link_table(rows, labelled):
    print(f"\nLinking (tracker gate {DEFAULT_GATE}); staff chain = chain owning the GT point")
    print(f"{'setting':<30}{'purity':>8}{'impure chains':>15}{'covered':>10}{'chain changes':>15}"
          f"{'links':>7}{'chains':>8}")
    for row in rows:
        impure = sum(on < total for on, total in row["purity"].values())
        impure_text = f"{impure}/{len(row['purity'])}"
        covered = f"{row['covered']}/{labelled}"
        print(f"{row['name']:<30}{row['purity_rate']:>8.0%}{impure_text:>15}{covered:>10}{row['changes']:>15}"
              f"{row['links']:>7}{row['chains']:>8}")
    for row in rows:
        impure = [f"{i}:{on}/{total}" for i, (on, total) in row["purity"].items() if on < total]
        print(f"  {row['name']} impure chains (id:on GT/labelled frames present): {' '.join(impure) or 'none'}")


def print_table(rows):
    print("\nTracker gate (max_dist); staff ID = track owning the GT point")
    print(f"{'setting':<18}{'purity':>8}{'impure IDs':>12}{'ID changes':>12}{'unique IDs':>12}{'median len':>12}")
    for row in rows:
        impure = sum(on < total for on, total in row["purity"].values())
        impure_text = f"{impure}/{len(row['purity'])}"
        print(f"{row['name']:<18}{row['purity_rate']:>8.0%}{impure_text:>12}{row['changes']:>12}"
              f"{row['unique_ids']:>12}{row['median_length']:>12.0f}")
    for row in rows:
        sequence = " ".join("-" if i is None else str(i) for i in row["staff"].values())
        print(f"\n{row['name']} staff IDs at GT frames:\n  {sequence}")
        impure = [f"{i}:{on}/{total}" for i, (on, total) in row["purity"].items() if on < total]
        print(f"  also on someone else (id:on GT/labelled frames present): {' '.join(impure) or 'none'}")


def main():
    args = parse_args()
    run = Path(args.run)
    detections = dict(np.load(run / "detections.npz"))
    detections["fps"], _ = video_info(args.video)
    gt_points = load_gt_points(args.gt)
    frame_rows = rows_per_frame(detections)

    median_speed, max_speed = staff_speed(gt_points)
    print(f"Staff speed from GT: median {median_speed:.1f} px/frame, max {max_speed:.1f} px/frame")
    radius, fps = float(detections["radius"]), float(detections["fps"])
    print(f"Radius {radius:.0f} px, fps {fps:.2f}, max_missed {round(MAX_MISSED_SECONDS * fps)} frames; "
          f"max_dist 0.05/0.10/0.20 = {0.05 * radius:.0f}/{0.10 * radius:.0f}/{0.20 * radius:.0f} px")

    rows = [evaluate(name, detections, frame_rows, gt_points, fraction, use_velocity)
            for name, fraction, use_velocity in SETTINGS]
    print_table(rows)

    appearance = run / "appearance.npz"
    hists = np.load(appearance)["hists"] if appearance.exists() else None
    if hists is None:
        print(f"\n(no {appearance}: appearance settings skipped)")
    link_rows = [evaluate_linking(name, detections, frame_rows, gt_points, gap, ratio,
                                  hists if use_appearance else None, long_gap, dedupe)
                 for name, gap, ratio, use_appearance, long_gap, dedupe in LINK_SETTINGS
                 if hists is not None or not use_appearance]
    print_link_table(link_rows, len(gt_points))



if __name__ == "__main__":
    main()
