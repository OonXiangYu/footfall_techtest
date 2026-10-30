"""Paths and readers for the hand labels in data/ (shared by the pipeline, the tools and the evaluation)."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_gt_points(path):
    """{frame: (x, y)} of the labelled staff chest points (present frames with a point) in gt.csv."""
    with open(path, newline="") as f:
        return {int(row["frame"]): (int(row["x"]), int(row["y"]))
                for row in csv.DictReader(f)
                if row["status"] == "present" and row["x"] and row["y"]}


def interpolate_chest(gt_path):
    """{frame: (x, y)}: the chest point for every present frame, linearly interpolated between the labelled
    points (every ~25 frames, so it can be tens of px off while the staff member walks)."""
    with open(gt_path, newline="") as f:
        rows = list(csv.DictReader(f))
    present = [int(r["frame"]) for r in rows if r["status"] == "present"]
    labelled = [(int(r["frame"]), int(r["x"]), int(r["y"]))
                for r in rows if r["status"] == "present" and r["x"] and r["y"]]
    frames, xs, ys = zip(*labelled)
    # outside the first/last labelled frame there is nothing to interpolate between
    inside = [f for f in present if frames[0] <= f <= frames[-1]]
    return {f: (float(np.interp(f, frames, xs)), float(np.interp(f, frames, ys))) for f in inside}


def load_tag_labels(path):
    """{frame: "y" / "n" / "u"} from tag_gt.csv or tag_gt_dense.csv."""
    with open(path, newline="") as f:
        return {int(row["frame"]): row["label"] for row in csv.DictReader(f)}


def load_tag_positions(path):
    """{frame: clicked tag centre (x, y)} from tag_pos.csv (skipped frames left out)."""
    with open(path, newline="") as f:
        return {int(r["frame"]): np.array([float(r["x"]), float(r["y"])])
                for r in csv.DictReader(f) if r["status"] == "tag"}
