"""Template matching for the name tag, evaluated on the same groups and crops as eval_tag.py."""

import re

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_detection import ROOT
from eval_tag import GROUPS, collect_crops, load_tag_labels, parse_args, print_distributions, print_threshold_table
from tag_detect import chest_square, kernel_for_tag, measure_tag_size

PDF_PATH = ROOT / "data" / "AI Evaluation Test.pdf"
TEMPLATE_DIR = ROOT / "outputs" / "tag_templates"
REVIEW_DIR = ROOT / "outputs" / "tag_review_template"
SEARCH_FRACTION = 0.6  # central square searched; larger than eval_tag's 0.4 because some crops are off-centre
SCALES = [0.8, 1.0, 1.25]
ANGLES = [-15, 0, 15]
ALIGN_ITERATIONS = 2
TAG_MARGIN = 3  # px around the white card in the PDF photo
THRESHOLDS = np.arange(0.30, 0.951, 0.05)
TOP_OTHERS, BOTTOM_Y = 20, 10


# first embedded JPEG in the PDF (the green-polo photo); PDF photos are stored as raw DCT streams
def extract_pdf_photo(path):
    data = path.read_bytes()
    match = re.search(rb"/Filter\s*/DCTDecode[^>]*?/Length\s+(\d+)[^>]*>>\s*stream\r?\n", data, re.S)
    if match is None:
        raise SystemExit(f"No JPEG image found in {path}")
    raw = data[match.end():match.end() + int(match.group(1))]
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# the tag is the largest white (low-saturation, bright) blob on the polo
def find_pdf_tag(photo):
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    # open first: the thin white collar stripes would otherwise merge with the card;
    # then close over the black bars so the card is one blob
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # a small margin keeps the card's dark top band and its edge against the shirt
    x, y = max(x - TAG_MARGIN, 0), max(y - TAG_MARGIN, 0)
    w, h = w + 2 * TAG_MARGIN, h + 2 * TAG_MARGIN
    return cv2.cvtColor(photo[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), (x, y, w, h)


def template_variants(template):
    """(image, scale, angle) for every scale/rotation; rotation keeps the size and replicates the border."""
    variants = []
    for scale in SCALES:
        scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = scaled.shape
        for angle in ANGLES:
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(scaled, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            variants.append((rotated, scale, angle))
    return variants


def match(gray, variants):
    """Best TM_CCOEFF_NORMED over all variants inside the search square: (score, (x, y, w, h), scale, angle)."""
    x0, y0, x1, y1 = chest_square(gray, SEARCH_FRACTION)
    region = gray[y0:y1, x0:x1]
    best = (-1.0, None, None, None)
    for image, scale, angle in variants:
        h, w = image.shape
        result = cv2.matchTemplate(region, image, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        if score > best[0]:
            best = (float(score), (x0 + x, y0 + y, w, h), scale, angle)
    return best


def standardize(patch):
    patch = patch.astype(np.float32)
    return (patch - patch.mean()) / (patch.std() + 1e-6)


def to_uint8(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def build_video_template(grays, start):
    """Align the tag patches (by matching the current template, scale 1, no rotation) and average them."""
    template = start
    h, w = start.shape
    for _ in range(ALIGN_ITERATIONS):
        patches = []
        for gray in grays:
            _, (x, y, _, _), _, _ = match(gray, [(template, 1.0, 0)])
            patches.append(standardize(gray[y:y + h, x:x + w]))
        # standardize first so a dark-jacket patch and a white-shirt patch weigh the same
        template = to_uint8(np.mean(patches, axis=0))
    return template


def save_template(template, name):
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEMPLATE_DIR / f"{name}.png"), template)
    preview = cv2.resize(template, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(TEMPLATE_DIR / f"{name}_x12.png"), preview)


def split_halves(y_frames):
    """Alternate labelled-y frames into two halves."""
    ordered = sorted(y_frames)
    return set(ordered[0::2]), set(ordered[1::2])


def score_all(staff_grays, other_grays, template):
    variants = template_variants(template)
    return ([match(g, variants) for g in staff_grays], [match(g, variants) for g in other_grays])


def group_scores(staff, staff_results, other_results, tag_labels, y_frames, detection_only):
    scores = {"y": [], "n": [], "others": [r[0] for r in other_results]}
    for (frame, _, used_fallback), result in zip(staff, staff_results):
        if detection_only and used_fallback:
            continue
        label = tag_labels[frame]
        if label == "n" or (label == "y" and frame in y_frames):
            scores[label].append(result[0])
    return scores


def report(name, scores):
    print(f"\n=== {name}")
    print_distributions(scores, decimals=2)
    print_threshold_table(scores, THRESHOLDS)


def plot_histograms(runs, path):
    bins = np.arange(-0.2, 1.0001, 0.025)
    fig, axes = plt.subplots(len(GROUPS), len(runs), figsize=(4.2 * len(runs), 7), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for col, (run_name, scores) in enumerate(runs):
        for row, (key, title, color) in enumerate(GROUPS):
            ax = axes[row, col]
            ax.hist(scores[key], bins=bins, color=color, edgecolor="#fcfcfb", linewidth=0.8)
            ax.set_title(f"{run_name}\n{title} (n={len(scores[key])}, median {np.median(scores[key]):.2f})"
                         if row == 0 else f"{title} (n={len(scores[key])}, median {np.median(scores[key]):.2f})",
                         loc="left", fontsize=8.5, color="#0b0b0b")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("crops", color="#52514e")
        axes[-1, col].set_xlabel("best TM_CCOEFF_NORMED score", color="#52514e")
    fig.suptitle("Template-match score by group (all crops)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def review_image(crop, result, frame):
    score, (x, y, w, h), scale, angle = result
    k = 3
    image = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    x0, y0, x1, y1 = chest_square(crop, SEARCH_FRACTION)
    cv2.rectangle(image, (x0 * k, y0 * k), (x1 * k, y1 * k), (255, 255, 255), 1)
    cv2.rectangle(image, (x * k, y * k), ((x + w) * k, (y + h) * k), (0, 0, 255), 2)
    text = f"f{frame} {score:.2f} s{scale} r{angle}"
    cv2.putText(image, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def save_review(name, staff, staff_results, others, other_results, tag_labels, y_frames):
    out = REVIEW_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    top = sorted(zip(other_results, others), key=lambda p: -p[0][0])[:TOP_OTHERS]
    for rank, (result, (frame, crop)) in enumerate(top, 1):
        cv2.imwrite(str(out / f"others_top{rank:02d}_f{frame:04d}.jpg"), review_image(crop, result, frame))
    ys = [(r, s) for r, s in zip(staff_results, staff) if tag_labels[s[0]] == "y" and s[0] in y_frames]
    for rank, (result, (frame, crop, _)) in enumerate(sorted(ys, key=lambda p: p[0][0])[:BOTTOM_Y], 1):
        cv2.imwrite(str(out / f"y_low{rank:02d}_f{frame:04d}.jpg"), review_image(crop, result, frame))


def main():
    args = parse_args()
    tag_labels = load_tag_labels(args.tag_gt)
    staff, others = collect_crops(args, tag_labels)
    staff_grays = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for _, crop, _ in staff]
    other_grays = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for _, crop in others]

    sizes = [measure_tag_size(crop) for frame, crop, fallback in staff if tag_labels[frame] == "y" and not fallback]
    (tag_w, tag_h, used), _ = kernel_for_tag(sizes)
    tag_size = (round(tag_w), round(tag_h))
    print(f"Measured tag size on {used} 'y' crops: {tag_size[0]} x {tag_size[1]} px")

    # A) the tag from the task PDF
    pdf_tag, box = find_pdf_tag(extract_pdf_photo(PDF_PATH))
    template_a = cv2.resize(pdf_tag, tag_size, interpolation=cv2.INTER_AREA)
    save_template(pdf_tag, "A_pdf_full")
    save_template(template_a, "A_pdf")
    print(f"Template A: PDF tag at {box} ({box[2]} x {box[3]} px) resized to {tag_size[0]} x {tag_size[1]}")

    # B) averaged from the video's y crops, two halves; a y crop is only scored by the other half's template
    y_all = [f for f, _, _ in staff if tag_labels[f] == "y"]
    halves = split_halves(y_all)
    templates_b = []
    for i, half in enumerate(halves):
        grays = [g for (f, _, fallback), g in zip(staff, staff_grays) if f in half and not fallback]
        template = build_video_template(grays, template_a)
        save_template(template, f"B_video_half{i}")
        templates_b.append(template)
        print(f"Template B{i}: averaged from {len(grays)} detection-centred y crops of half {i} "
              f"({len(half)} y crops in that half)")

    runs = [("A_pdf", template_a, set(y_all)),
            ("B_from_half0", templates_b[0], halves[1]),
            ("B_from_half1", templates_b[1], halves[0])]
    plotted = []
    for name, template, y_frames in runs:
        staff_results, other_results = score_all(staff_grays, other_grays, template)
        for detection_only in (False, True):
            scores = group_scores(staff, staff_results, other_results, tag_labels, y_frames, detection_only)
            report(f"{name} - {'detection-centred staff crops only' if detection_only else 'all crops'}", scores)
            if not detection_only:
                plotted.append((name, scores))
        save_review(name, staff, staff_results, others, other_results, tag_labels, y_frames)

    plot_histograms(plotted, ROOT / "outputs" / "tag_scores_template.png")
    print(f"\nSaved templates to {TEMPLATE_DIR}, histogram to {ROOT / 'outputs' / 'tag_scores_template.png'}, "
          f"review crops to {REVIEW_DIR}")


if __name__ == "__main__":
    main()
