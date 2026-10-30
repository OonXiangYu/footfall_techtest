"""Train and evaluate the tag detector (tag_model.py) on the sparse + dense tag labels.

Training (per time block, using only that block): positives at the clicked tag positions (data/tag_pos.csv),
random negative positions on staff "n" frames and on other people, then
  1. self-training: unclicked "y" people whose best position is confident become positives;
  2. copy-paste: real tag patches (clicked or confident) pasted into other people's and staff-n tag
     windows, with scale / brightness / blur jitter;
  3. one round of hard-negative mining (the best-scoring positions on negative people).
Evaluation: first half of the video vs second half and swapped (neighbouring frames are near-duplicates).
Saves one model per fold (models/tag_classifier_fold{0,1}.npz) and one trained on everything
(models/tag_classifier.npz) with a score -> log-likelihood-ratio table for chain evidence.
"""

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from eval_detection import ROOT
from eval_tag import print_distributions, print_threshold_table
from pose import to_body
from tag_crops import load_tag_labels
from tag_match import TAG_SIZE
from tag_model import (FEATURES, PATCH_SIZE, best_position, features, llr_tables, patch_at, patch_scale, resample,
                       save_model, side_windows)
from tag_window import collect_people_cached, load_tag_positions

SPARSE_LABELS = ROOT / "data" / "tag_gt.csv"
DENSE_LABELS = ROOT / "data" / "tag_gt_dense.csv"
PEOPLE_CACHE = ROOT / "outputs" / "tag_people_dense.pkl"
MODEL_DIR = ROOT / "models"
WEIGHTS_IMAGE = ROOT / "outputs" / "tag_classifier_weights.png"
C = 0.1  # logistic regression regularisation
NEGATIVES_PER_WINDOW = {"n": 8, "others": 4}  # random negative positions per side window
CONFIDENT = 0.8  # self-training: an unclicked positive's best position counts if at least this sure
PASTES = 400  # copy-paste positives per training block
HARD_NEGATIVES = None  # all: the best-scoring position of every negative person (ablation: better than 200)
THRESHOLDS = np.round(np.arange(0.1, 0.951, 0.1), 2)
MIN_STAFF_RATE = 0.05
RUN_GAP = 5  # y frames at most this far apart belong to the same run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature", choices=FEATURES, help="only this feature type (default: compare all)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def merged_labels():
    """Sparse labels overridden by dense ones; "u" dropped."""
    labels = {**load_tag_labels(SPARSE_LABELS), **load_tag_labels(DENSE_LABELS)}
    return {f: label for f, label in labels.items() if label in ("y", "n")}


def y_runs(frames):
    runs = []
    for frame in sorted(frames):
        if runs and frame - runs[-1][-1] <= RUN_GAP:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


# ---- training samples -------------------------------------------------------------------------

def fit(positives, negatives, kind, tag_units, window_centre, window_half):
    X = features(np.concatenate([positives, negatives]), kind)
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X, y)
    return {"feature": kind, "coef": clf.coef_[0], "intercept": float(clf.intercept_[0]),
            "tag_units": np.array(tag_units), "window_centre": np.array(window_centre),
            "window_half": np.array(window_half)}


def jittered(person, position, tag_units):
    """Patches at a tag position, shifted by -1/0/+1 px and mirrored."""
    step = 1 / patch_scale(tag_units)
    patches = []
    for dx in (-step, 0, step):
        for dy in (-step, 0, step):
            patch = patch_at(person["canvas"], person["body"][3], np.asarray(position) + (dx, dy), tag_units)
            patches += [patch, cv2.flip(patch, 1)]
    return patches


def random_negatives(rng, people, count, centre, half, tag_units):
    patches = []
    for person in people:
        for window_centre, window_half in side_windows(centre, half):
            for _ in range(count):
                position = window_centre + rng.uniform(-1, 1, 2) * window_half
                patches.append(patch_at(person["canvas"], person["body"][3], position, tag_units))
    return patches


def tag_crop(person, position, tag_units):
    """Just the tag (half the patch) around a position, at patch resolution."""
    size = (PATCH_SIZE[0] // 2, PATCH_SIZE[1] // 2)
    scale = patch_scale(tag_units)
    return resample(person["canvas"], person["body"][3], np.asarray(position) - np.array(size) / 2 / scale,
                    size, scale)


def paste(rng, background, tag):
    """A tag pasted into the middle of a background patch with scale, brightness, blur and offset jitter."""
    scale = rng.uniform(0.9, 1.1)
    w, h = max(3, round(tag.shape[1] * scale)), max(2, round(tag.shape[0] * scale))
    tag = cv2.resize(tag, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    tag = np.clip(tag * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10), 0, 255)
    sigma = rng.uniform(0, 0.8)
    if sigma > 0.1:
        tag = cv2.GaussianBlur(tag, (3, 3), sigma)
    patch = background.astype(np.float32).copy()
    x = (PATCH_SIZE[0] - w) // 2 + rng.integers(-1, 2)
    y = (PATCH_SIZE[1] - h) // 2 + rng.integers(-1, 2)
    x, y = int(np.clip(x, 0, PATCH_SIZE[0] - w)), int(np.clip(y, 0, PATCH_SIZE[1] - h))
    patch[y:y + h, x:x + w] = tag
    return patch.astype(np.uint8)


def train(staff, others, clicks, tag_units, kind, rng):
    """The training recipe above, on one block of people. Returns (model, counts of each sample type)."""
    clicked = [p for p in staff if p["label"] == "y" and p["frame"] in clicks]
    click_body = {id(p): to_body(clicks[p["frame"]], p["body"]) for p in clicked}
    points = np.array(list(click_body.values()))
    centre = np.array([np.abs(points[:, 0]).mean(), points[:, 1].mean()])  # mirrored windows: |x|
    half = np.array(tag_units)

    positives = [patch for p in clicked for patch in jittered(p, click_body[id(p)], tag_units)]
    staff_n = [p for p in staff if p["label"] == "n"]
    negatives = (random_negatives(rng, staff_n, NEGATIVES_PER_WINDOW["n"], centre, half, tag_units)
                 + random_negatives(rng, others, NEGATIVES_PER_WINDOW["others"], centre, half, tag_units))
    model = fit(positives, negatives, kind, tag_units, centre, half)

    # 1. self-training on unclicked positives
    sources = [tag_crop(p, click_body[id(p)], tag_units) for p in clicked]
    confident = 0
    for person in (p for p in staff if p["label"] == "y" and p["frame"] not in clicks):
        score, position = best_position(model, person["canvas"], person["body"][3])
        if score >= CONFIDENT:
            positives += jittered(person, position, tag_units)
            sources.append(tag_crop(person, position, tag_units))
            confident += 1

    # 2. copy-paste into staff-n and other people's tag windows
    targets = staff_n + others
    pasted = []
    for _ in range(PASTES):
        person = targets[rng.integers(len(targets))]
        window_centre, window_half = side_windows(centre, half)[rng.integers(2)]
        position = window_centre + rng.uniform(-1, 1, 2) * window_half
        background = patch_at(person["canvas"], person["body"][3], position, tag_units)
        pasted.append(paste(rng, background, sources[rng.integers(len(sources))]))
    model = fit(positives + pasted, negatives, kind, tag_units, centre, half)

    # 3. hard negatives: the best-scoring positions on people without a visible tag
    scored = []
    for person in targets:
        score, position = best_position(model, person["canvas"], person["body"][3])
        scored.append((score, person, position))
    scored.sort(key=lambda s: -s[0])
    hard = [patch_at(p["canvas"], p["body"][3], pos, tag_units)
            for _, p, pos in scored[:HARD_NEGATIVES or len(scored)]]
    model = fit(positives + pasted, negatives + hard, kind, tag_units, centre, half)
    counts = {"clicked": len(clicked), "self-trained": confident, "positive patches": len(positives),
              "pasted": len(pasted), "negatives": len(negatives), "hard negatives": len(hard)}
    return model, counts


# ---- evaluation -------------------------------------------------------------------------------

def score_people(model, staff, others):
    def best(people):
        return [best_position(model, p["canvas"], p["body"][3])[0] for p in people]
    return {"y": best([p for p in staff if p["label"] == "y"]),
            "n": best([p for p in staff if p["label"] == "n"]),
            "others": best(others)}


def rates(scores, threshold, share):
    y, n, others = (np.mean(np.array(scores[g]) >= threshold) if len(scores[g]) else 0.0
                    for g in ("y", "n", "others"))
    return share * y + (1 - share) * n, others


def print_track_table(scores, share):
    print(f"\nTrack level (staff rate = {share:.2f} x y found + {1 - share:.2f} x staff-n firing)")
    print(f"{'threshold':>9}{'staff rate':>12}{'others rate':>13}{'ratio':>8}")
    for threshold in THRESHOLDS:
        staff_rate, others = rates(scores, threshold, share)
        print(f"{threshold:>9.2f}{staff_rate:>12.1%}{others:>13.1%}"
              f"{(staff_rate / others if others else float('inf')):>8.2f}")


def auc(positives, negatives):
    """Probability that a random positive outscores a random negative (ties count half)."""
    p, n = np.asarray(positives)[:, None], np.asarray(negatives)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def best_ratio(scores, share):
    best = (0.0, None)
    for threshold in THRESHOLDS:
        staff_rate, others = rates(scores, threshold, share)
        if staff_rate >= MIN_STAFF_RATE and others > 0 and staff_rate / others > best[0]:
            best = (staff_rate / others, threshold)
    return best


def tag_units_of(staff, clicks):
    widths = [p["body"][3] for p in staff if p["label"] == "y" and p["frame"] in clicks]
    return (TAG_SIZE[0] / np.median(widths), TAG_SIZE[1] / np.median(widths))


def run_folds(kind, staff, others, clicks, split, share, rng):
    print(f"\n##### features: {kind}")
    halves = [lambda f: f < split, lambda f: f >= split]
    tag_units = tag_units_of(staff, clicks)
    folds = []
    for k in (0, 1):
        train_in, test_in = halves[k], halves[1 - k]
        model, counts = train([p for p in staff if train_in(p["frame"])], [p for p in others if train_in(p["frame"])],
                              {f: c for f, c in clicks.items() if train_in(f)}, tag_units, kind, rng)
        held_out = score_people(model, [p for p in staff if test_in(p["frame"])],
                                [p for p in others if test_in(p["frame"])])
        in_sample = score_people(model, [p for p in staff if train_in(p["frame"])],
                                 [p for p in others if train_in(p["frame"])])
        name = "first half" if k == 0 else "second half"
        print(f"\n=== {kind}: trained on the {name} ({', '.join(f'{v} {n}' for n, v in counts.items())}), "
              f"tested on the other half")
        print_distributions(held_out, decimals=2)
        print_threshold_table(held_out, THRESHOLDS)
        print_track_table(held_out, share)
        ratio, threshold = best_ratio(held_out, share)
        aucs = (auc(held_out["y"], held_out["others"]), auc(held_out["y"], held_out["n"]))
        print(f"best staff/others ratio {ratio:.2f} at threshold {threshold}; "
              f"AUC y vs others {aucs[0]:.2f}, y vs staff-n {aucs[1]:.2f}")
        folds.append({"model": model, "held_out": held_out, "in_sample": in_sample, "ratio": ratio, "auc": aucs[0],
                      "threshold": threshold, "trained_on": "first" if k == 0 else "second"})
    return folds


def save_weights_picture(model, staff, clicks, tag_units):
    positives = [patch_at(p["canvas"], p["body"][3], to_body(clicks[p["frame"]], p["body"]), tag_units)
                 for p in staff if p["label"] == "y" and p["frame"] in clicks]
    panels = [("mean patch at the clicked tag positions", np.mean(positives, axis=0), "gray"),
              (f"logistic-regression weights ({model['feature']})", model["coef"].reshape(PATCH_SIZE[::-1]), "RdBu_r")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    for ax, (title, image, cmap) in zip(axes, panels):
        limit = np.abs(image).max() if cmap == "RdBu_r" else None
        shown = ax.imshow(image, cmap=cmap, vmin=-limit if limit else None, vmax=limit, interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(shown, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(WEIGHTS_IMAGE, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    labels = merged_labels()
    sparse = load_tag_labels(SPARSE_LABELS)
    share = sum(v == "y" for v in sparse.values()) / sum(v in ("y", "n") for v in sparse.values())
    defaults = argparse.Namespace(video=str(ROOT / "data" / "sample.mp4"), gt=str(ROOT / "data" / "gt.csv"),
                                  detections=str(ROOT / "outputs" / "detections.npz"))
    staff, others = collect_people_cached(defaults, labels, PEOPLE_CACHE)
    clicks = load_tag_positions(ROOT / "data" / "tag_pos.csv")
    frames = sorted({p["frame"] for p in staff})
    split = (frames[0] + frames[-1]) // 2
    y_frames = [p["frame"] for p in staff if p["label"] == "y"]
    print(f"People with a usable pose: staff {len(staff)} ({len(y_frames)} y, "
          f"{sum(p['label'] == 'n' for p in staff)} n), others {len(others)}")
    print(f"Tag visible on {share:.0%} of the staff frames (sparse labels)")
    for name, in_half in (("first", lambda f: f < split), ("second", lambda f: f >= split)):
        runs = y_runs(f for f in y_frames if in_half(f))
        print(f"{name} half (split at frame {split}): {sum(len(r) for r in runs)} y people in {len(runs)} distinct "
              f"runs, {sum(in_half(f) for f in clicks)} clicks")

    results = {kind: run_folds(kind, staff, others, clicks, split, share, rng)
               for kind in ([args.feature] if args.feature else FEATURES)}
    print("\nSummary per held-out half (AUC y vs others; best staff/others ratio @ threshold):")
    for kind, folds in results.items():
        print(f"  {kind:<7}" + "   ".join(f"AUC {f['auc']:.2f}; {f['ratio']:.2f} @ {f['threshold']}" for f in folds))
    # chain evidence weighs every score, so rank by how well scores separate (AUC), not by one threshold
    kind = max(results, key=lambda k: min(f["auc"] for f in results[k]))
    folds = results[kind]
    print(f"Best (by the worse half's AUC): {kind}")

    tag_units = tag_units_of(staff, clicks)
    for k, fold in enumerate(folds):
        tables = llr_tables(fold["in_sample"]["y"], fold["in_sample"]["n"], fold["in_sample"]["others"], share)
        save_model(MODEL_DIR / f"tag_classifier_fold{k}.npz", **fold["model"], **tables, split=split,
                   trained_on=fold["trained_on"], visible_share=share)
    # the all-data model's LLR table comes from held-out scores (each half scored by the other half's model)
    held = {g: folds[0]["held_out"][g] + folds[1]["held_out"][g] for g in ("y", "n", "others")}
    tables = llr_tables(held["y"], held["n"], held["others"], share)
    model, counts = train(staff, others, clicks, tag_units, kind, rng)
    save_model(MODEL_DIR / "tag_classifier.npz", **model, **tables, split=split, trained_on="all",
               visible_share=share)
    print(f"\nSaved models/tag_classifier_fold0.npz, _fold1.npz and tag_classifier.npz ({kind}; all data: "
          f"{', '.join(f'{v} {n}' for n, v in counts.items())})")
    print("LLR per score bin 0.0-0.1 ... 0.9-1.0 (all-data model, from held-out scores):")
    for name, table in tables.items():
        print(f"  {name[4:]:<17}" + " ".join(f"{v:+.2f}" for v in table))
    save_weights_picture(model, staff, clicks, tag_units)
    print(f"Saved {WEIGHTS_IMAGE}")


if __name__ == "__main__":
    main()
