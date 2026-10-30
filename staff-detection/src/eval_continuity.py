"""Frame-by-frame diagnostics of a staff_detect.py run (what the 1-per-second labelled evaluation misses).

- claims: per frame, how many staff chains have a detection there; how often the output switches chain;
- continuity: frames where the output point jumps more than JUMP x radius from the previous frame, and
  frames marked present with no staff detection behind the point (held / interpolated);
- error against the interpolated GT chest point on every frame (plot: <run>/error_timeline.png);
- wrong-person ranges: present frames whose point is not on the GT person, with the chain that was;
- still-and-far ranges: the output point does not move while it is far from the GT point.
The GT between labelled frames (every 25 frames) is linearly interpolated, so while the staff member
walks it can itself be tens of pixels off; treat single frames with care, ranges are more telling.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detections import rows_per_frame
from fisheye import ANGLE_STEPS, owning_box, to_panorama_point
from tag_crops import interpolate_chest

ROOT = Path(__file__).resolve().parent.parent
JUMP = 0.10  # fraction of the radius
FAR = 0.10  # fraction of the radius: farther than this from the GT point is "not on the person"
STILL = 1.0  # px: the output point moved less than this from the previous frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="output folder of staff_detect.py")
    parser.add_argument("--gt", default=str(ROOT / "data" / "gt.csv"))
    parser.add_argument("--quiet", action="store_true", help="summary only, no range listings")
    return parser.parse_args()


def load_staff(path):
    with open(path, newline="") as f:
        return [None if r["present"] != "1" else (float(r["x"]), float(r["y"]), int(r["chain"]))
                for r in csv.DictReader(f)]


def ranges(frames):
    """Consecutive frames grouped into (first, last)."""
    groups = []
    for f in sorted(frames):
        if groups and f == groups[-1][1] + 1:
            groups[-1][1] = f
        else:
            groups.append([f, f])
    return [tuple(g) for g in groups]


def owner_chain(detections, chains, rows, point):
    center, radius = tuple(detections["center"]), float(detections["radius"])
    rows = [r for r in rows if chains[r] >= 0]
    pano = to_panorama_point(*point, center, radius)
    owner = None if pano is None or not rows else owning_box(pano, detections["box"][rows], ANGLE_STEPS)
    return None if owner is None else int(chains[rows[owner]])


def diagnose(run, gt_path, quiet=False, plot=True):
    run = Path(run)
    detections = dict(np.load(run / "detections.npz"))
    saved = np.load(run / "chains.npz")
    chains, staff_chains = saved["chain"], set(saved["staff_chain"].tolist())
    staff = load_staff(run / "staff.csv")
    frame_rows = rows_per_frame(detections)
    radius = float(detections["radius"])
    gt = interpolate_chest(gt_path)

    claims = np.array([len({int(chains[r]) for r in rows if chains[r] in staff_chains}) for rows in frame_rows])
    present = [f for f, e in enumerate(staff) if e]
    switches = sum(1 for a, b in zip(staff, staff[1:]) if a and b and a[2] != b[2])
    jumps = [f for f in range(1, len(staff)) if staff[f] and staff[f - 1]
             and np.hypot(staff[f][0] - staff[f - 1][0], staff[f][1] - staff[f - 1][1]) > JUMP * radius]
    undrawn = [f for f in present if not any(chains[r] == staff[f][2] for r in frame_rows[f])]
    errors = {f: float(np.hypot(staff[f][0] - gt[f][0], staff[f][1] - gt[f][1])) for f in present if f in gt}
    still_far = [f for f in present if f > 0 and staff[f - 1] and f in errors and errors[f] > FAR * radius
                 and np.hypot(staff[f][0] - staff[f - 1][0], staff[f][1] - staff[f - 1][1]) < STILL]
    # wrong person: far from the GT point AND another chain owns it (the interpolated GT alone drifts while
    # the staff member walks, so distance alone also flags frames where the output is on the right person)
    owners = {f: owner_chain(detections, chains, frame_rows[f], gt[f]) for f in present if f in gt}
    wrong = [f for f in present if f in errors and errors[f] > FAR * radius
             and owners[f] is not None and owners[f] != staff[f][2]]

    # missed: no staff output while some chain owns the GT point (the staff is there, just not marked)
    absent_owner = {f: owner_chain(detections, chains, frame_rows[f], gt[f])
                    for f in range(len(staff)) if not staff[f] and f in gt}
    missed = [f for f, o in absent_owner.items() if o is not None]
    summary = {"frames": len(staff), "present": len(present),
               "claimed by 0/1/2/3+ staff chains": [int(np.sum(claims == k)) for k in (0, 1, 2)] + [int(np.sum(claims >= 3))],
               "chain switches": switches, f"jumps > {JUMP:.2f} R": len(jumps),
               "present but no staff box drawn": len(undrawn),
               "longest run without a box (frames)": max((b - a + 1 for a, b in ranges(undrawn)), default=0),
               "median error vs interpolated GT (px)": float(np.median(list(errors.values()))) if errors else float("nan"),
               f"present frames within {FAR:.2f} R of GT": f"{sum(e <= FAR * radius for e in errors.values())} of {len(errors)}",
               "still-and-far frames": len(still_far),
               "wrong person (far, another chain on GT)": len(wrong),
               "  of which on an unconfirmed chain": sum(owners[f] not in staff_chains for f in wrong),
               "missed (no output, a chain on the GT person)": len(missed)}
    print(f"\n== {run.name}")
    for key, value in summary.items():
        print(f"  {key:<40} {value}")
    if not quiet:
        print(f"\n  still-and-far ranges (point not moving, > {FAR:.2f} R from GT): frames, output chain, box drawn?")
        for a, b in ranges(still_far):
            if b - a + 1 >= 5:
                chain = staff[a][2]
                drawn = sum(any(chains[r] == chain for r in frame_rows[f]) for f in range(a, b + 1))
                print(f"    {a}-{b}  chain {chain}  box drawn on {drawn} of {b - a + 1} frames")
        print(f"\n  wrong-person ranges (>= 5 frames; > {FAR:.2f} R from GT and another chain owns the GT point): "
              f"output chain -> chain on the GT person")
        for a, b in ranges(wrong):
            if b - a + 1 < 5:
                continue
            outputs = sorted({staff[f][2] for f in range(a, b + 1)})
            on_gt = [owners[f] for f in range(a, b + 1)]
            on = sorted({o for o in on_gt if o is not None})
            label = ", ".join(f"{o}{'*' if o in staff_chains else ''} ({on_gt.count(o)})" for o in on) or "none"
            print(f"    {a}-{b}  output {outputs}  ->  on the GT person: {label}")
        print("    (* = a staff chain; counts = frames)")
        print("\n  missed ranges (>= 5 frames; no output, the GT person is on a chain): chain on the GT person")
        for a, b in ranges(missed):
            if b - a + 1 < 5:
                continue
            on_gt = [absent_owner[f] for f in range(a, b + 1)]
            label = ", ".join(f"{o}{'*' if o in staff_chains else ''} ({on_gt.count(o)})"
                              for o in sorted({o for o in on_gt if o is not None}))
            print(f"    {a}-{b}  ->  {label}")
    if plot:
        save_plot(run / "error_timeline.png", staff, errors, undrawn, radius)
    return summary


def save_plot(path, staff, errors, undrawn, radius):
    frames = np.arange(len(staff))
    error = np.array([errors.get(f, np.nan) for f in frames])
    drawn = error.copy()
    drawn[list(undrawn)] = np.nan
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(frames, error, color="#eb6834", linewidth=1, label="no staff box (held / interpolated)")
    ax.plot(frames, drawn, color="#2a78d6", linewidth=1.2, label="staff box drawn")
    ax.axhline(FAR * radius, color="#52514e", linewidth=0.8, linestyle="--")
    ax.text(len(staff), FAR * radius, f" {FAR:.2f} R", va="center", fontsize=8, color="#52514e")
    ax.set_xlim(0, len(staff))
    ax.set_xlabel("frame", color="#52514e")
    ax.set_ylabel("px from GT (interpolated)", color="#52514e")
    ax.set_title("Staff output: distance to the GT chest point per frame (gaps = no staff output)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    args = parse_args()
    diagnose(args.run, args.gt, args.quiet)
    print(f"\nSaved {Path(args.run) / 'error_timeline.png'}")


if __name__ == "__main__":
    main()
