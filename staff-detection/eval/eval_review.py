"""Simulate the semi-automatic mode (staff_detect.py --tag review) with the hand labels in data/gt.csv.

    python eval/eval_review.py outputs/run

- Candidate ranking: how many of the top 5 / 10 / 20 candidates are really the staff member.
- Simulated users: one who confirms every true staff candidate on page 1 (and on pages 1-2), and the
  earlier one who confirms one candidate per outfit (2 or 3 clicks); each scored with whole-chain and
  window confirm scope, with and without appearance expansion, including the purity of everything
  marked staff.
- Expansion threshold: purity of the chains it adds (on labelled frames) for a range of thresholds.

Uses the run folder's caches (detections.npz, appearance.npz, tag_scores_review.npz).
"""

import argparse
import sys
from pathlib import Path

from eval_run import position_metrics, read_rows, staff_chain_per_frame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from appearance import similarity
from fisheye import ANGLE_STEPS, owning_box, to_panorama_point
from labels import DATA, interpolate_chest, load_gt_points
from review import (CANDIDATES_PER_CHAIN, CONFIRM_WINDOW, EXPAND_MARGIN, EXPAND_SIMILARITY,
                    EXPAND_SPEED, MIN_CHAIN_SECONDS, PER_PAGE, candidates, chain_summaries, expand, staff_from_clicks)
from staff_detect import build_chains, parse_args as pipeline_args, review_scores, staff_per_frame, video_info

THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
TARGET_PURITY = 0.95
NEAR_LABEL = 12  # frames: a labelled frame this close decides a candidate's truth via its chain


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output folder for the review mode (its caches are reused or created)")
    parser.add_argument("--video", default=str(DATA / "sample.mp4"))
    parser.add_argument("--gt", default=str(DATA / "gt.csv"))
    return parser.parse_args()


def candidate_truth(detections, frame_rows, chains, ranked, chest, owners):
    """Is the candidate the staff member? If its chain is present at a labelled frame within
    NEAR_LABEL frames, whether the chain owns the GT point there (chains are ~98% pure, and linear
    interpolation of the walking staff over 25 frames can be 100+ px off); otherwise whether its box
    owns the interpolated GT chest point."""
    center, radius = tuple(detections["center"]), float(detections["radius"])
    frames_of = {}
    for row, chain in enumerate(chains):
        frames_of.setdefault(int(chain), set()).add(int(detections["frame"][row]))
    truth = []
    for c in ranked:
        near = [f for f in owners if abs(f - c["frame"]) <= NEAR_LABEL and f in frames_of[c["chain"]]]
        if near:
            nearest = min(near, key=lambda f: abs(f - c["frame"]))
            truth.append(owners[nearest] == c["chain"])
            continue
        point = chest.get(c["frame"])
        pano = None if point is None else to_panorama_point(*point, center, radius)
        rows = frame_rows[c["frame"]]
        owner = None if pano is None else owning_box(pano, detections["box"][rows], ANGLE_STEPS)
        truth.append(owner is not None and rows[owner] == c["row"])
    return truth


class Scenario:
    """Turns confirmed clicks into staff chains (with a confirm scope and optional expansion) and scores
    them against gt.csv."""

    def __init__(self, state, gt_rows, gt_points, fps):
        self.state, self.gt_rows, self.gt_points = state, gt_rows, gt_points
        self.window = round(CONFIRM_WINDOW * fps)

    def run(self, clicks, scope, threshold):
        s = self.state
        scoped, staff, weights, added = staff_from_clicks(s["detections"], s["chains"], s["hists"], clicks, scope,
                                                          self.window, threshold, s["radius"])
        result = self.score(scoped, staff, weights)
        owners = staff_chain_per_frame(s["detections"], scoped, self.gt_points)
        result["purity"] = chain_purity(staff, scoped, s["detections"]["frame"], owners)
        result["added purity"] = chain_purity(added, scoped, s["detections"]["frame"], owners)
        result["added"] = len(added)
        return result

    def score(self, chains, staff_chains, weights):
        s = self.state
        staff = staff_per_frame(s["detections"], chains, staff_chains, weights, s["frame_count"], s["max_hold"],
                                s["switch_penalty"])
        pred = {f: ({"present": "1", "x": str(e[0]), "y": str(e[1])} if e else {"present": "0", "x": "", "y": ""})
                for f, e in enumerate(staff)}
        metrics = position_metrics(self.gt_rows, pred, s["radius"])
        hits_key = next(k for k in metrics if k.startswith("hits"))
        return {"hits": metrics[hits_key], "median": metrics["median distance (px)"],
                "frames": sum(e is not None for e in staff)}


def chain_purity(chains_marked, chains, frames, owners):
    """(on GT point, present): labelled frames where a marked chain is present, and how many of them it
    owns the GT chest point."""
    present = on_point = 0
    for chain in chains_marked:
        for frame in set(frames[chains == chain].tolist()) & set(owners):
            present += 1
            on_point += owners[frame] == chain
    return on_point, present


def page_user(ranked, truth, pages):
    """Confirms every candidate on the first pages that really is the staff member."""
    shown = PER_PAGE * pages
    return [(c["chain"], c["frame"]) for c, ok in zip(ranked[:shown], truth[:shown]) if ok]


def outfit_user(ranked, truth, info, confirmations, radius):
    """The earlier simulation: one confirmation at a time, the best-ranked true candidate whose chain is
    not yet staff (whole chains + expansion) and does not look like a confirmed chain (another outfit)."""
    clicks, confirmed, staff = [], [], set()
    for _ in range(confirmations):
        open_ = [c for c, ok in zip(ranked, truth) if ok and c["chain"] not in staff]

        def new_outfit(c):
            return all(similarity(info[c["chain"]]["hist"], info[k]["hist"]) < EXPAND_SIMILARITY for k in confirmed)
        pick = next((c for c in open_ if new_outfit(c)), open_[0] if open_ else None)
        if pick is None:
            break
        clicks.append((pick["chain"], pick["frame"]))
        confirmed.append(pick["chain"])
        staff = set(confirmed) | set(expand(confirmed, info, EXPAND_SIMILARITY, EXPAND_SPEED * radius,
                                            EXPAND_MARGIN * radius))
    return clicks


def fraction(pair):
    on_point, present = pair
    return f"{on_point}/{present} = {on_point / present:.0%}" if present else "-"


def main():
    args = parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    sargs = pipeline_args(["--video", args.video, "--out", str(run), "--tag", "review"])
    fps, frame_count = video_info(args.video)
    state = build_chains(sargs, run, fps, frame_count, {})
    detections, chains = state["detections"], state["chains"]
    scores = review_scores(sargs, run, state)
    ranked = candidates(detections, chains, scores, fps)
    gt_points = load_gt_points(args.gt)
    owners = staff_chain_per_frame(detections, chains, gt_points)
    truth = candidate_truth(detections, state["frame_rows"], chains, ranked, interpolate_chest(args.gt), owners)

    per_page = PER_PAGE
    print(f"\nCandidates: {len(ranked)} (each chain >= {MIN_CHAIN_SECONDS:g} s: its best frame first, then up to "
          f"{CANDIDATES_PER_CHAIN} per chain); {sum(truth)} of them are the staff member")
    for k in (5, 10, 20):
        print(f"  top {k:>2}: {sum(truth[:k])} staff  (ranks {[c['rank'] for c, t in zip(ranked[:k], truth[:k]) if t]})")
    for pages in (1, 2):
        shown = per_page * pages
        print(f"  page{'s 1-2' if pages == 2 else ' 1'} ({shown} candidates): {sum(truth[:shown])} staff on "
              f"{len({c['chain'] for c, t in zip(ranked[:shown], truth[:shown]) if t})} chain(s)")

    info = chain_summaries(detections, chains, state["hists"])
    scenario = Scenario(state, read_rows(args.gt), gt_points, fps)
    users = [("page 1, all staff", page_user(ranked, truth, 1)),
             ("pages 1-2, all staff", page_user(ranked, truth, 2)),
             ("one per outfit x2", outfit_user(ranked, truth, info, 2, state["radius"])),
             ("one per outfit x3", outfit_user(ranked, truth, info, 3, state["radius"]))]
    print(f"\nSimulated users; confirm scope chain = whole linked chain, window = +-{CONFIRM_WINDOW:g} s; "
          f"expansion at similarity {EXPAND_SIMILARITY}")
    print(f"{'user':<22}{'clicks':>7}{'scope':>8}{'expansion':>10}{'hits of 55':>12}{'median px':>11}"
          f"{'frames w/ pred':>16}{'staff purity (labelled)':>26}")
    for name, clicks in users:
        first = True
        for scope in ("chain", "window"):
            for threshold in (None, EXPAND_SIMILARITY):
                r = scenario.run(clicks, scope, threshold)
                print(f"{name if first else '':<22}{(len(clicks) if first else ''):>7}{scope:>8}"
                      f"{('+' + str(r['added']) if threshold else 'off'):>10}{r['hits']:>12}{r['median']:>11.1f}"
                      f"{r['frames']:>16}{fraction(r['purity']):>26}")
                first = False

    # expansion threshold: purity of what it adds, for the page-1 user's confirmations (whole chains)
    clicks = users[0][1]
    print(f"\nExpansion threshold sweep (page-1 user, {len(clicks)} clicks, whole chains)")
    print(f"{'similarity':>10}{'added':>7}{'added purity (labelled)':>26}{'hits of 55':>12}{'frames w/ pred':>16}")
    chosen = None
    for threshold in THRESHOLDS:
        r = scenario.run(clicks, "chain", threshold)
        on_point, present = r["added purity"]
        print(f"{threshold:>10.2f}{r['added']:>7}{fraction(r['added purity']):>26}{r['hits']:>12}{r['frames']:>16}")
        if chosen is None and present and on_point / present >= TARGET_PURITY:
            chosen = threshold
    print(f"\nLowest threshold with >= {TARGET_PURITY:.0%} purity of the added chains: {chosen}")


if __name__ == "__main__":
    main()
