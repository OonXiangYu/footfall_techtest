"""Key decisions of the pipeline as ablations on the sample video: the current default vs the alternative,
everything else unchanged. Prints one table and saves it to results/decisions.md and results/decisions.csv,
with one bar chart per decision in results/decisions.png.

    python eval/decisions.py [--run outputs/reproduce] [--results results]

1. detection view: original frame vs panorama          (staff hits of 55, boxes / frame)
2. detection conf: 0.10 vs 0.25                          (staff hits of 55, boxes / frame)
3. tracker gate: 0.20 R vs 0.05 R                        (track purity, staff ID changes)
4. linking ratio test: off vs 0.5                        (chain purity, staff chain changes)
5. duplicate-box removal: off vs on                      (chain purity, staff chain changes, the chain on the
                                                          seated stretch)
6. final result with examples/sample_confirmed.json      (hits of 55, median error)
7. oracle reference: --tag oracle with data/tag_gt.csv  (hits of 55, median error)

1-2 run the detector on the 55 labelled frames (as eval_detection.py). 3-7 run the pipeline's stages with
the alternative passed as a staff_detect.py flag (--max-dist, --link-ratio 0, --keep-duplicates), reusing
the detection and appearance caches in --run (made on the first run if missing, about 4 minutes).
7 marks as staff every chain with at least --min-hits (2) hand-labelled tag sightings, as
staff_detect.py --tag oracle does.
A staff hit: a box contains the labelled chest point. Purity and changes are on the labelled frames, where
the staff track / chain is the one whose box owns the labelled point (as eval_linking.py).
"""

import argparse
import contextlib
import csv
import io
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detection import CONF, evaluate_view, read_frames
from eval_linking import count_id_changes, staff_ids_at_gt, track_purity
from eval_run import position_metrics, read_rows
from fisheye import circle_for
from labels import DATA, ROOT, load_gt_points
from staff_detect import (build_chains, oracle_tag_hits, parse_args as pipeline_args, review_staff, staff_per_frame,
                          video_info)

CONFIRMED = ROOT / "examples" / "sample_confirmed.json"
TAG_LABELS = DATA / "tag_gt.csv"  # hand tag labels (y / n / u, every 5th frame) for the oracle reference
LOW_CONF = 0.10
WIDE_GATE = 0.20
SEATED = (0, 347)  # the staff member sits at the desk for these frames at the start of the sample video
DEFAULT_COLOR, ALTERNATIVE_COLOR = "#2a78d6", "#b4b2a9"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DATA / "sample.mp4"))
    parser.add_argument("--gt", default=str(DATA / "gt.csv"))
    parser.add_argument("--run", default=str(ROOT / "outputs" / "reproduce"),
                        help="folder with the detection and appearance caches (created if missing)")
    parser.add_argument("--results", default=str(ROOT / "results"), help="where to save decisions.md/.csv/.png")
    return parser.parse_args()


def row(number, decision, alternative, default, metric, alt_value, default_value, alt_text, default_text,
        chart=False, limit=None):
    """One table row; chart: this metric is the decision's bar chart (limit: its y-axis maximum)."""
    return {"number": number, "decision": decision, "alternative": alternative, "default": default,
            "metric": metric, "alternative_value": alt_value, "default_value": default_value,
            "alternative_text": alt_text, "default_text": default_text, "chart": chart, "limit": limit}


# ---- 1-2: detection on the labelled frames -------------------------------------------------------

def detection_rows(video, gt_points):
    frames = read_frames(video, set(gt_points) | {0})
    gt_points = {i: p for i, p in gt_points.items() if i in frames}
    circle = circle_for(frames[0])
    model = YOLO(str(ROOT / "yolov8n.pt"))
    original = evaluate_view(model, frames, gt_points, circle, False)
    panorama = evaluate_view(model, frames, gt_points, circle, True)
    low_conf = evaluate_view(model, frames, gt_points, circle, True, LOW_CONF)
    total = len(gt_points)

    def hits_text(result):
        return f"{result['hits']} of {total} ({result['hits'] / total:.1%})"

    rows = []
    # charted: the view by staff hits; the conf by boxes / frame (staff hits barely move, the boxes do)
    for number, decision, alt_name, default_name, alt, chart_hits in (
            (1, "detection view", "original frame", "panorama", original, True),
            (2, "detection conf", f"{LOW_CONF:.2f}", f"{CONF:.2f}", low_conf, False)):
        rows += [row(number, decision, alt_name, default_name, f"staff hits of {total}", alt["hits"],
                     panorama["hits"], hits_text(alt), hits_text(panorama), chart=chart_hits, limit=total),
                 row(number, decision, alt_name, default_name, "boxes / frame", round(alt["boxes"], 1),
                     round(panorama["boxes"], 1), f"{alt['boxes']:.1f}", f"{panorama['boxes']:.1f}",
                     chart=not chart_hits, limit=max(alt["boxes"], panorama["boxes"]))]
    return rows


# ---- 3-6: tracking, linking and the final result on the cached detections ------------------------

def pipeline_state(base, overrides, run, fps, frame_count):
    """The pipeline's stages 1-3 (detection and appearance from the cache) with extra staff_detect.py flags."""
    with contextlib.redirect_stdout(io.StringIO()):
        return build_chains(pipeline_args(base + overrides), run, fps, frame_count, {})


def purity(state, ids, gt_points):
    """(rate, on, total, staff changes, {labelled frame: staff id}) for track or chain ids."""
    staff = staff_ids_at_gt(state["detections"], state["frame_rows"], ids, gt_points)
    per_id = track_purity(staff, ids, state["frame_rows"])
    on, total = (sum(p[i] for p in per_id.values()) for i in (0, 1))
    return on / total, on, total, count_id_changes(list(staff.values())), staff


def seated_chain(state, staff):
    """The chain on the labelled point most often in the seated stretch: (first frame, last frame)."""
    owners = Counter(c for f, c in staff.items() if SEATED[0] <= f <= SEATED[1] and c is not None)
    chain = owners.most_common(1)[0][0]
    frames = state["detections"]["frame"][state["chains"] == chain]
    return int(frames.min()), int(frames.max())


def purity_rows(number, decision, alt_name, default_name, kind, alt, default, chart=True):
    def text(result):
        return f"{result[0]:.1%} ({result[1]}/{result[2]})"
    return [row(number, decision, alt_name, default_name, f"{kind} purity", round(alt[0], 4), round(default[0], 4),
                text(alt), text(default), chart=chart, limit=1.0),
            row(number, decision, alt_name, default_name, f"staff {kind} changes", alt[3], default[3],
                str(alt[3]), str(default[3]))]


def pipeline_rows(args, gt_points):
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    fps, frame_count = video_info(args.video)
    base = ["--video", args.video, "--out", str(run), "--gt", args.gt]
    print("   pipeline stages with the default settings (the detections are made first if not cached)")
    default = pipeline_state(base, [], run, fps, frame_count)
    wide_gate = pipeline_state(base, ["--max-dist", str(WIDE_GATE)], run, fps, frame_count)
    no_ratio = pipeline_state(base, ["--link-ratio", "0"], run, fps, frame_count)
    duplicates = pipeline_state(base, ["--keep-duplicates"], run, fps, frame_count)
    default_gate = pipeline_args(base).max_dist

    rows = purity_rows(3, "tracker gate", f"{WIDE_GATE:.2f} R", f"{default_gate:.2f} R", "track",
                       purity(wide_gate, wide_gate["track_ids"], gt_points),
                       purity(default, default["track_ids"], gt_points))
    default_chains = purity(default, default["chains"], gt_points)
    rows += purity_rows(4, "linking ratio test", "off", f"{pipeline_args(base).link_ratio:g}", "chain",
                        purity(no_ratio, no_ratio["chains"], gt_points), default_chains)
    duplicate_chains = purity(duplicates, duplicates["chains"], gt_points)
    # charted: the seated-stretch chain (what duplicate removal changes), not the purity
    rows += purity_rows(5, "duplicate-box removal", "off", "on", "chain", duplicate_chains, default_chains,
                        chart=False)
    spans = [seated_chain(duplicates, duplicate_chains[4]), seated_chain(default, default_chains[4])]
    lengths = [last - first + 1 for first, last in spans]
    seated_label = f"chain on the seated stretch (frames {SEATED[0]}-{SEATED[1]}), frames"
    rows.append(row(5, "duplicate-box removal", "off", "on", seated_label, lengths[0], lengths[1],
                    *(f"{length} ({first}-{last})" for length, (first, last) in zip(lengths, spans)),
                    chart=True, limit=SEATED[1] - SEATED[0] + 1))

    # 6: the final result: the confirmed chains, the staff path, scored on the labelled frames
    with contextlib.redirect_stdout(io.StringIO()):
        confirmed_args = pipeline_args(base + ["--confirmed", str(CONFIRMED), "--no-review"])
        chains, staff_chains, weights = review_staff(confirmed_args, run, fps, default)
    rows += result_rows(6, "final result", CONFIRMED.name, default, chains, staff_chains, weights, args.gt)

    # 7: the oracle reference: staff chains from the hand tag labels (as staff_detect.py --tag oracle)
    oracle_args = pipeline_args(base + ["--tag", "oracle", "--tag-labels", str(TAG_LABELS)])
    hits = oracle_tag_hits(oracle_args, default["detections"], default["frame_rows"], default["chains"])
    staff_chains = {chain for chain, count in hits.items() if count >= oracle_args.min_hits}
    rows += result_rows(7, "oracle (hand tag labels)", TAG_LABELS.name, default, default["chains"], staff_chains,
                        hits, args.gt)
    return rows


def result_rows(number, decision, setting, state, chains, staff_chains, weights, gt_path):
    """Hits and median error of the staff path through the given staff chains, on the labelled frames."""
    staff = staff_per_frame(state["detections"], chains, staff_chains, weights, state["frame_count"],
                            state["max_hold"], state["switch_penalty"])
    pred = {f: {"present": "1", "x": f"{e[0]:.1f}", "y": f"{e[1]:.1f}"} for f, e in enumerate(staff) if e}
    metrics = position_metrics(read_rows(gt_path), pred, state["radius"])
    hits = next(v for k, v in metrics.items() if k.startswith("hits"))
    labelled, median = metrics["labelled frames"], metrics["median distance (px)"]
    return [row(number, decision, "", setting, f"hits of {labelled}", "", hits, "-",
                f"{hits} of {labelled} ({hits / labelled:.1%})", chart=True, limit=labelled),
            row(number, decision, "", setting, "median error (px)", "", round(median, 1), "-", f"{median:.1f}")]


# ---- output -----------------------------------------------------------------------------------

def markdown(rows):
    lines = ["| # | decision: alternative -> default | metric | alternative | default |", "|---|---|---|---|---|"]
    for r in rows:
        decision = (f"{r['decision']}: {r['alternative']} -> **{r['default']}**" if r["alternative"]
                    else f"{r['decision']}: {r['default']}")
        lines.append(f"| {r['number']} | {decision} | {r['metric']} | {r['alternative_text']} | "
                     f"**{r['default_text']}** |")
    return "\n".join(lines)


def save_csv(path, rows):
    columns = ["number", "decision", "alternative", "default", "metric", "alternative_value", "default_value"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_chart(path, rows):
    charted = [r for r in rows if r["chart"]]
    columns = (len(charted) + 1) // 2
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 6.4))
    for ax, r in zip(axes.ravel(), charted):
        names = [r["alternative"], r["default"] + " (default)"] if r["alternative"] else [r["default"]]
        values = ([r["alternative_value"]] if r["alternative"] else []) + [r["default_value"]]
        texts = ([r["alternative_text"]] if r["alternative"] else []) + [r["default_text"]]
        colors = ([ALTERNATIVE_COLOR] if r["alternative"] else []) + [DEFAULT_COLOR]
        bars = ax.bar(names, values, color=colors, width=0.55)
        for bar, text in zip(bars, texts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text.split(" (")[0], ha="center",
                    va="bottom", fontsize=8, color="#2c2c2a")
        if len(bars) == 1:
            ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(0, r["limit"] * 1.12)
        if r["limit"] == 1.0:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{r['number']}. {r['decision']}\n{r['metric']}", loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
        ax.grid(axis="y", color="#e8e6dc", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(charted):]:
        ax.set_visible(False)
    fig.suptitle("Key decisions on the sample video: alternative vs current default (blue)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    args = parse_args()
    if not Path(args.video).exists():
        raise SystemExit(f"No video at {args.video}: the sample video is not included in the repository "
                         f"(see the README)")
    start = time.perf_counter()
    gt_points = load_gt_points(args.gt)
    print("Decisions 1-2: detector on the labelled frames")
    rows = detection_rows(args.video, gt_points)
    print("Decisions 3-6: pipeline stages on the cached detections")
    rows += pipeline_rows(args, gt_points)

    table = markdown(rows)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "decisions.md").write_text(
        "# Key decisions: ablations on the sample video\n\n"
        "Each row changes one setting from the current default (bold) to the alternative, everything else\n"
        "unchanged. Made by `eval/decisions.py`; staff hits and purity are measured on the 55 labelled frames.\n\n"
        + table + "\n", encoding="utf-8")
    save_csv(results / "decisions.csv", rows)
    save_chart(results / "decisions.png", rows)
    print("\n" + table)
    print(f"\nSaved {results / 'decisions.md'}, decisions.csv and decisions.png")
    print(f"Total run time: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
