"""Small tag classifier on the two shoulder-placed tag windows (logistic regression, no deep nets).

Patches: both side windows of tag_window.py ("two" mode), resampled to PATCH_SIZE, grayscale.
Positives: the clicked side on "y" frames. Negatives: both windows on staff "n" frames (same clothes,
no visible tag) and on other people. Evaluated on time blocks (first half of the video vs second
half, then swapped), since neighbouring frames are near-duplicates.
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
from tag_model import FEATURES, PATCH_SIZE, features, window_patch
from tag_window import collect_people_cached, load_tag_positions, print_track_table, windows_from_clicks

C = 0.1  # logistic regression regularisation (few positives)
SHIFTS = (-1, 0, 1)  # positive augmentation: window shifts in patch pixels, plus a mirror
THRESHOLDS = np.round(np.arange(0.1, 0.951, 0.1), 2)
MIN_STAFF_RATE = 0.05  # ignore thresholds where the staff almost never fires
MODEL_PATH = ROOT / "models" / "tag_classifier.npz"
WEIGHTS_IMAGE = ROOT / "outputs" / "tag_classifier_weights.png"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature", choices=FEATURES, help="only this feature type (default: compare all)")
    return parser.parse_args()




def side_of(click_body, windows):
    """Index of the window on the clicked side (windows are at +x and -x)."""
    return 0 if click_body[0] > 0 else 1


def training_set(staff, others, clicks, windows, kind):
    X, y = [], []
    for person in staff:
        canvas, width = person["canvas"], person["body"][3]
        if person["label"] == "y":
            if person["frame"] not in clicks:
                continue  # skipped click: side unknown
            window = windows[side_of(to_body(clicks[person["frame"]], person["body"]), windows)]
            for dx in SHIFTS:
                for dy in SHIFTS:
                    patch = window_patch(canvas, width, window, (dx, dy))
                    for variant in (patch, cv2.flip(patch, 1)):
                        X.append(features(variant, kind))
                        y.append(1)
        else:
            for window in windows:
                X.append(features(window_patch(canvas, width, window), kind))
                y.append(0)
    for person in others:
        for window in windows:
            X.append(features(window_patch(person["canvas"], person["body"][3], window), kind))
            y.append(0)
    return np.array(X), np.array(y)


def train(X, y):
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    return model.fit(X, y)


def person_score(model, person, windows, kind):
    """Tag probability: the larger of the two side windows."""
    X = [features(window_patch(person["canvas"], person["body"][3], w), kind) for w in windows]
    return float(model.predict_proba(np.array(X))[:, 1].max())


def best_ratio(scores):
    """(ratio, threshold) with the best staff/others rate, among thresholds where the staff fires enough."""
    best = (0.0, None)
    for threshold in THRESHOLDS:
        y, n, o = (np.mean(np.array(scores[g]) >= threshold) for g in ("y", "n", "others"))
        staff_rate = 0.14 * y + 0.86 * n
        if staff_rate >= MIN_STAFF_RATE and o > 0 and staff_rate / o > best[0]:
            best = (staff_rate / o, threshold)
    return best


def tag_units_from(staff, clicks):
    widths = [p["body"][3] for p in staff if p["label"] == "y" and p["frame"] in clicks]
    return (TAG_SIZE[0] / np.median(widths), TAG_SIZE[1] / np.median(widths))


def evaluate(kind, staff, others, clicks, split):
    print(f"\n##### features: {kind}")
    ratios, fold_scores = [], []
    blocks = [lambda f: f < split, lambda f: f >= split]
    for k in (0, 1):
        in_train, in_test = blocks[k], blocks[1 - k]
        train_staff = [p for p in staff if in_train(p["frame"])]
        train_clicks = {f: c for f, c in clicks.items() if in_train(f)}
        points = [to_body(train_clicks[p["frame"]], p["body"]) for p in train_staff if p["frame"] in train_clicks]
        windows = windows_from_clicks(points, tag_units_from(train_staff, train_clicks), "two")
        X, y = training_set(train_staff, [p for p in others if in_train(p["frame"])], train_clicks, windows, kind)
        model = train(X, y)
        test_staff = [p for p in staff if in_test(p["frame"])]
        scores = {"y": [person_score(model, p, windows, kind) for p in test_staff if p["label"] == "y"],
                  "n": [person_score(model, p, windows, kind) for p in test_staff if p["label"] == "n"],
                  "others": [person_score(model, p, windows, kind) for p in others if in_test(p["frame"])]}
        name = "first half" if k == 0 else "second half"
        print(f"\n=== {kind}: trained on the {name} ({len(points)} clicks, {int(y.sum())} positive patches "
              f"after augmentation, {int((y == 0).sum())} negative), tested on the other half")
        print_distributions(scores, decimals=2)
        print_threshold_table(scores, THRESHOLDS)
        print_track_table(scores, THRESHOLDS)
        ratio, threshold = best_ratio(scores)
        print(f"best staff/others ratio {ratio:.2f} at threshold {threshold}")
        ratios.append((ratio, threshold))
        fold_scores.append(scores)
    return ratios, fold_scores


def rates(scores, threshold):
    y, n, others = (np.mean(np.array(scores[g]) >= threshold) for g in ("y", "n", "others"))
    return 0.14 * y + 0.86 * n, others


def operating_point(fold_scores):
    """Threshold with the best worse-fold staff/others ratio, and the others' (background) hit rate
    there, averaged over the held-out folds. Returns (threshold, background rate, staff rate)."""
    best = (-1.0, None)
    for threshold in THRESHOLDS:
        fold_rates = [rates(scores, threshold) for scores in fold_scores]
        if min(staff for staff, _ in fold_rates) < MIN_STAFF_RATE:
            continue
        worse = min(staff / others if others else 0.0 for staff, others in fold_rates)
        if worse > best[0]:
            best = (worse, threshold)
    threshold = best[1] if best[1] is not None else THRESHOLDS[-1]
    fold_rates = [rates(scores, threshold) for scores in fold_scores]
    return float(threshold), float(np.mean([o for _, o in fold_rates])), float(np.mean([s for s, _ in fold_rates]))


def save_model_and_weights(kind, staff, others, clicks, threshold, background):
    """Train on everything and save the model (for staff_detect.py) and a picture of its weights."""
    points = [to_body(clicks[p["frame"]], p["body"]) for p in staff if p["label"] == "y" and p["frame"] in clicks]
    tag_units = tag_units_from(staff, clicks)
    windows = windows_from_clicks(points, tag_units, "two")
    X, y = training_set(staff, others, clicks, windows, kind)
    model = train(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_PATH, feature=kind, coef=model.coef_[0], intercept=model.intercept_[0],
             window_centre=windows[0][0], window_half=windows[0][1], patch_size=np.array(PATCH_SIZE),
             threshold=threshold, background=background)
    print(f"\nSaved {MODEL_PATH} ({kind}, windows at (+-{windows[0][0][0]:.2f}, {windows[0][0][1]:.2f}) "
          f"half-size ({windows[0][1][0]:.2f}, {windows[0][1][1]:.2f}) shoulder widths)")

    positives = np.array([features(window_patch(p["canvas"], p["body"][3],
                                                windows[side_of(to_body(clicks[p["frame"]], p["body"]), windows)]),
                                   "pixels")
                          for p in staff if p["label"] == "y" and p["frame"] in clicks])
    panels = [("mean tag patch (clicked side)", positives.mean(axis=0).reshape(PATCH_SIZE[::-1]), "gray")]
    if kind != "hog":
        panels.append((f"logistic-regression weights ({kind})", model.coef_[0].reshape(PATCH_SIZE[::-1]), "RdBu_r"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 2.6))
    for ax, (title, image, cmap) in zip(np.atleast_1d(axes), panels):
        limit = np.abs(image).max() if cmap == "RdBu_r" else None
        shown = ax.imshow(image, cmap=cmap, vmin=-limit if limit else None, vmax=limit, interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(shown, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(WEIGHTS_IMAGE, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Saved {WEIGHTS_IMAGE}")


def main():
    args = parse_args()
    defaults = argparse.Namespace(video=str(ROOT / "data" / "sample.mp4"), gt=str(ROOT / "data" / "gt.csv"),
                                  tag_labels=str(ROOT / "data" / "tag_gt.csv"),
                                  detections=str(ROOT / "outputs" / "detections.npz"))
    staff, others = collect_people_cached(defaults, load_tag_labels(defaults.tag_labels))
    clicks = load_tag_positions(ROOT / "data" / "tag_pos.csv")
    frames = sorted({p["frame"] for p in staff})
    split = (frames[0] + frames[-1]) // 2
    print(f"Time split at frame {split}: clicks {sum(f < split for f in clicks)} before, "
          f"{sum(f >= split for f in clicks)} after")
    results = {kind: evaluate(kind, staff, others, clicks, split) for kind in ([args.feature] if args.feature else FEATURES)}
    print("\nSummary (best staff/others ratio per test half, threshold):")
    for kind, (ratios, _) in results.items():
        print(f"  {kind:<7}" + "   ".join(f"{r:.2f} @ {t}" for r, t in ratios))
    best = max(results, key=lambda kind: min(r for r, _ in results[kind][0]))
    threshold, background, staff_rate = operating_point(results[best][1])
    print(f"Best (by the worse of the two halves): {best}; operating threshold {threshold} -> held-out "
          f"staff rate {staff_rate:.1%}, background (others) rate {background:.1%}")
    save_model_and_weights(best, staff, others, clicks, threshold, background)


if __name__ == "__main__":
    main()
