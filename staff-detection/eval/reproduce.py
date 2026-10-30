"""Reproduce every number in results/summary.md on the sample video, with one command.

    python eval/reproduce.py [--video data/sample.mp4] [--out outputs/reproduce] [--keep-caches]

Runs, in order:
  1. src/staff_detect.py with the final review's confirmations (examples/sample_confirmed.json, --no-review)
  2. eval/eval_detection.py  original frame vs panorama
  3. eval/eval_linking.py    tracker gate and linking settings
  4. eval/eval_run.py        hits, presence and frame-by-frame diagnostics of the run
  5. eval/eval_review.py     simulated review users
Each step's output is shown and saved to <out>/logs/. At the end: the numbers next to the expected
values read from results/summary.md, whether staff.csv is identical to results/staff.csv, and the
run time of each step. Exits with 1 if anything differs.

The run starts from scratch: <out> is cleared first (only a folder this script created, marked with
a .reproduce file). --keep-caches reuses its detections and appearance caches instead (much faster;
the numbers are the same).
"""

import argparse
import contextlib
import filecmp
import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_run import JUMP, FAR, diagnose, evaluate, read_rows
from labels import DATA, ROOT

SUMMARY = ROOT / "results" / "summary.md"
EXPECTED_CSV = ROOT / "results" / "staff.csv"
CONFIRMED = ROOT / "examples" / "sample_confirmed.json"
MARKER = ".reproduce"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DATA / "sample.mp4"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "reproduce"), help="run folder")
    parser.add_argument("--keep-caches", action="store_true",
                        help="reuse the detections and appearance caches of an earlier reproduce run")
    return parser.parse_args()


def prepare(out, keep_caches):
    if out.exists() and not keep_caches:
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by reproduce.py; choose another --out")
        shutil.rmtree(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / MARKER).touch()


def run_step(name, args, log_path):
    """Run a script with this Python, show its output and save it to log_path. Returns seconds taken."""
    print(f"\n===== {name}: {' '.join(str(a) for a in args)}", flush=True)
    start = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    # stdout is shown and logged; stderr (progress bars) is shown only
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
            [sys.executable, *map(str, args)], cwd=ROOT, env=env, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
    if process.returncode:
        raise SystemExit(f"{name} failed (exit code {process.returncode}); see {log_path}")
    return time.perf_counter() - start


def expected_values():
    """{first cell: [other cells]} of every table row in results/summary.md, bold marks removed."""
    rows = {}
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) > 1 and not set(cells[0]) <= set("-"):
            rows[cells[0]] = cells[1:]
    return rows


def detection_values(log_path):
    """{view: (hits, frames, boxes per frame)} from eval_detection.py's table."""
    values = {}
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"(original frame|panorama)\s+(\d+)\s+(\d+)\s+[\d.]+%\s+([\d.]+)", line)
        if match:
            values[match[1]] = (int(match[3]), int(match[2]), float(match[4]))
    return values


def of(part, whole, decimals=None):
    text = f"{part} of {whole}"
    return text if decimals is None else f"{text} ({part / whole:.{decimals}%})"


def reproduced_values(out, gt_path, detection_log):
    metrics, _ = evaluate(out, read_rows(gt_path))
    with contextlib.redirect_stdout(io.StringIO()):  # its full report is in logs/eval_run.txt
        frames = diagnose(out, gt_path, quiet=True)
    hits_key = next(k for k in metrics if k.startswith("hits"))
    detection = detection_values(detection_log)
    claims = frames["claimed by 0/1/2/3+ staff chains"]
    rows = [  # (row in summary.md, column, reproduced value)
        ("hits (prediction within 0.10 of the fisheye radius = 48 px)", 0,
         of(metrics[hits_key], metrics["labelled frames"], 0)),
        ("median distance to the labelled point", 0, f"{metrics['median distance (px)']:.1f} px"),
        ("frames with a prediction", 0, of(metrics["predicted on them"], metrics["labelled frames"])),
        ("frames marked present", 0, of(metrics["TP"] + metrics["FP"], metrics["frames scored"])),
        ("precision / recall", 0, f"{metrics['precision']:.2f} / {metrics['recall']:.2f}"),
        ("present frames within 0.10 R of the interpolated point", 0, frames[f"present frames within {FAR:.2f} R of GT"]),
        ("frames on the wrong person (far from the point, another track on it)", 0,
         str(frames["wrong person (far, another chain on GT)"])),
        ("frames where the point stands still while far from the GT", 0, str(frames["still-and-far frames"])),
        ("frames claimed by two staff tracks at once", 0, str(sum(claims[2:]))),
        ("jumps of more than 0.10 R between frames", 0, str(frames[f"jumps > {JUMP:.2f} R"])),
        ("present frames with no staff box (held or interpolated, at most 0.5 s)", 0,
         str(frames["present but no staff box drawn"])),
    ]
    for row, view in (("original fisheye frame", "original frame"), ("unwarped panorama", "panorama")):
        hits, total, boxes = detection.get(view, (0, 0, float("nan")))
        rows += [(row, 0, of(hits, total, 1) if total else "missing"), (row, 1, f"{boxes:.1f}")]
    return rows


def main():
    args = parse_args()
    if not Path(args.video).exists():
        raise SystemExit(f"No video at {args.video}: the sample video is not included in the repository "
                         f"(see the README); pass --video")
    out = Path(args.out).resolve()
    prepare(out, args.keep_caches)
    logs = out / "logs"
    gt = DATA / "gt.csv"
    steps = [
        ("pipeline", [ROOT / "src" / "staff_detect.py", "--video", args.video, "--out", out,
                      "--confirmed", CONFIRMED, "--no-review"]),
        ("eval_detection", [ROOT / "eval" / "eval_detection.py", "--video", args.video, "--gt", gt]),
        ("eval_linking", [ROOT / "eval" / "eval_linking.py", out, "--video", args.video, "--gt", gt]),
        ("eval_run", [ROOT / "eval" / "eval_run.py", out, "--gt", gt]),
        ("eval_review", [ROOT / "eval" / "eval_review.py", out, "--video", args.video, "--gt", gt]),
    ]
    start = time.perf_counter()
    timings = {name: run_step(name, command, logs / f"{name}.txt") for name, command in steps}
    total = time.perf_counter() - start

    expected = expected_values()
    rows = reproduced_values(out, gt, logs / "eval_detection.txt")
    print("\n===== Reproduced vs results/summary.md")
    print(f"{'metric':<72}{'expected':>18}{'reproduced':>18}")
    mismatches = 0
    for row, column, value in rows:
        cells = expected.get(row)
        want = cells[column] if cells and column < len(cells) else "(not in summary.md)"
        ok = want == value
        mismatches += not ok
        label = row if column == 0 else f"{row}: boxes / frame"
        print(f"{label[:71]:<72}{want:>18}{value:>18}  {'ok' if ok else 'DIFFERS'}")
    same_csv = filecmp.cmp(out / "staff.csv", EXPECTED_CSV, shallow=False)
    mismatches += not same_csv
    print(f"\nstaff.csv identical to results/staff.csv: {'yes' if same_csv else 'NO'}")
    print("Run time: " + ", ".join(f"{name} {seconds:.0f} s" for name, seconds in timings.items())
          + f"; total {total / 60:.1f} min")
    print(f"Logs: {logs}")
    print("All numbers match." if not mismatches else f"{mismatches} value(s) differ.")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
