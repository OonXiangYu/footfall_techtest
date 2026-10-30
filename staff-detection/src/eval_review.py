"""Simulate the semi-automatic mode (staff_detect.py --tag review) with the hand labels in data/gt.csv.

- Candidate ranking: how many of the top 5 / 10 / 20 candidates are really the staff member.
- Simulated user: confirms the best-ranked true staff candidate, then the next one whose chain is not
  yet staff (another outfit / time), and so on; results after 1, 2 and 3 confirmations, with and
  without appearance expansion.
- Expansion threshold: purity of the chains it adds (on labelled frames) for a range of thresholds.

Uses the run folder's caches (detections.npz, appearance.npz, tag_scores_review.npz).
"""

import argparse
from pathlib import Path

from eval_chain_evidence import staff_chain_per_frame
from eval_detection import load_gt_points
from eval_pipeline import position_metrics, read_rows
from appearance import similarity
from fisheye import ANGLE_STEPS, owning_box, to_panorama_point
from review import (CANDIDATE_SPACING, CANDIDATES_PER_CHAIN, EXPAND_MARGIN, EXPAND_SIMILARITY, EXPAND_SPEED,
                    candidates, chain_summaries, expand)
from staff_detect import build_chains, parse_args as pipeline_args, review_scores, staff_per_frame, video_info
from tag_crops import interpolate_chest

ROOT = Path(__file__).resolve().parent.parent
CONFIRMATIONS = 3
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
TARGET_PURITY = 0.95
NEAR_LABEL = 12  # frames: a labelled frame this close decides a candidate's truth via its chain


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output folder for the review mode (its caches are reused or created)")
    parser.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--max-confirmations", type=int, default=CONFIRMATIONS)
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
    """Everything needed to turn a set of staff chains into scores against gt.csv."""

    def __init__(self, state, gt_rows, info):
        self.state, self.gt_rows, self.info = state, gt_rows, info

    def staff_chains(self, confirmed, threshold):
        added = {}
        if threshold is not None:
            radius = self.state["radius"]
            added = expand(confirmed, self.info, threshold, EXPAND_SPEED * radius, EXPAND_MARGIN * radius)
        weights = {c: 2 for c in confirmed}
        weights.update({c: 1 for c in added})
        return set(confirmed) | set(added), weights, added

    def score(self, staff_chains, weights):
        s = self.state
        staff = staff_per_frame(s["detections"], s["chains"], staff_chains, weights, s["frame_count"], s["max_gap"])
        pred = {f: ({"present": "1", "x": str(e[0]), "y": str(e[1])} if e else {"present": "0", "x": "", "y": ""})
                for f, e in enumerate(staff)}
        metrics = position_metrics(self.gt_rows, pred, s["radius"])
        hits_key = next(k for k in metrics if k.startswith("hits"))
        return {"hits": metrics[hits_key], "median": metrics["median distance (px)"],
                "labelled predicted": metrics["predicted on them"], "frames": sum(e is not None for e in staff)}


def simulate(scenario, ranked, truth, threshold, confirmations=CONFIRMATIONS):
    """A user following the hint: confirm the best-ranked true candidate whose chain is not yet staff
    and does not look like an already confirmed chain (another outfit); if there is none, the best-ranked
    true candidate whose chain is not yet staff."""
    confirmed, steps = [], []
    staff = set()
    info = scenario.info
    for _ in range(confirmations):
        open_ = [c for c, ok in zip(ranked, truth) if ok and c["chain"] not in staff]

        def new_outfit(c):
            return all(similarity(info[c["chain"]]["hist"], info[k]["hist"]) < EXPAND_SIMILARITY for k in confirmed)
        pick = next((c for c in open_ if new_outfit(c)), open_[0] if open_ else None)
        if pick is None:
            break
        confirmed.append(pick["chain"])
        staff, weights, added = scenario.staff_chains(confirmed, threshold)
        steps.append((pick, len(added), scenario.score(staff, weights)))
    return steps


def added_purity(added, chains, frames, owners):
    """Labelled frames where an added chain is present, and how many of them it is on the GT point."""
    present = on_point = 0
    for chain in added:
        chain_frames = set(frames[chains == chain].tolist())
        for frame in chain_frames & set(owners):
            present += 1
            on_point += owners[frame] == chain
    return on_point, present


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
    owners = staff_chain_per_frame(detections, chains, load_gt_points(args.gt))
    truth = candidate_truth(detections, state["frame_rows"], chains, ranked, interpolate_chest(args.gt), owners)

    print(f"\nCandidates: {len(ranked)} (best {CANDIDATES_PER_CHAIN} per chain, top spread >= "
          f"{CANDIDATE_SPACING:g} s apart); "
          f"{sum(truth)} of them are the staff member")
    for k in (5, 10, 20):
        print(f"  top {k:>2}: {sum(truth[:k])} staff  (ranks {[c['rank'] for c, t in zip(ranked[:k], truth[:k]) if t]})")

    scenario = Scenario(state, read_rows(args.gt), chain_summaries(detections, chains, state["hists"]))
    print(f"\nSimulated user (confirms the best-ranked true candidate of a not-yet-confirmed outfit); "
          f"expansion at similarity {EXPAND_SIMILARITY}")
    print(f"{'':<16}{'confirm':>8}{'rank':>6}{'frame':>7}{'added':>7}{'hits of 55':>12}{'median px':>11}"
          f"{'labelled w/ pred':>18}{'frames w/ pred':>16}")
    for name, threshold in (("no expansion", None), ("with expansion", EXPAND_SIMILARITY)):
        for k, (pick, added, result) in enumerate(simulate(scenario, ranked, truth, threshold,
                                                           args.max_confirmations), 1):
            print(f"{name if k == 1 else '':<16}{k:>8}{pick['rank']:>6}{pick['frame']:>7}{added:>7}"
                  f"{result['hits']:>12}{result['median']:>11.1f}{result['labelled predicted']:>18}"
                  f"{result['frames']:>16}")

    # expansion threshold: purity of what it adds, with the 3 confirmations chosen without expansion
    confirmed = [pick["chain"] for pick, _, _ in simulate(scenario, ranked, truth, None)]
    print(f"\nExpansion threshold sweep ({len(confirmed)} confirmed chains: {confirmed})")
    print(f"{'similarity':>10}{'added':>7}{'purity (labelled frames)':>27}{'hits of 55':>12}{'frames w/ pred':>16}")
    chosen = None
    for threshold in THRESHOLDS:
        staff, weights, added = scenario.staff_chains(confirmed, threshold)
        on_point, present = added_purity(added, chains, detections["frame"], owners)
        purity = on_point / present if present else float("nan")
        result = scenario.score(staff, weights)
        print(f"{threshold:>10.2f}{len(added):>7}{f'{on_point}/{present} = {purity:.0%}' if present else '-':>27}"
              f"{result['hits']:>12}{result['frames']:>16}")
        if chosen is None and present and purity >= TARGET_PURITY:
            chosen = threshold
    print(f"\nLowest threshold with >= {TARGET_PURITY:.0%} purity: {chosen}")


if __name__ == "__main__":
    main()
