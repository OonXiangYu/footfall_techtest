"""Semi-automatic staff identification: rank candidate people by tag score, let a human confirm the
staff on a review page, then expand from the confirmed chains by appearance."""

import json
from pathlib import Path

import cv2
import numpy as np

from appearance import similarity
from crops import upright_crop
from fisheye import box_outline

CANDIDATES_PER_CHAIN = 2  # a chain's best-scoring frames offered for review
CANDIDATE_SPACING = 2.0  # seconds: at most one candidate per window this long
MAX_CANDIDATES = 96
GRID = (4, 3)  # columns x rows per review page
CROP_SIZE, THUMB_SIZE = 200, (200, 150)
LABEL_HEIGHT = 34
WINDOW = "staff review"
HINT = ["Click the staff member (toggle). The staff may change clothes:",
        "confirm one candidate per outfit.   n / p = next / previous page, Enter = done"]

# appearance expansion
# torso-histogram similarity to a confirmed chain: the lowest threshold whose added chains were >= 95% on
# the staff in both simulated users of eval_review.py (rank order: 0.75 ok; outfit-aware: 0.75 = 43%,
# 0.80 = 100%). Small samples: a few labelled frames each.
EXPAND_SIMILARITY = 0.8
EXPAND_SPEED = 0.05  # walking speed for reachability, fraction of the radius per frame (staff max ~0.048)
EXPAND_MARGIN = 0.05  # fraction of the radius
EXPAND_MIN_ROWS = 5  # chains with fewer detections have too noisy a histogram


# ---- candidates ----------------------------------------------------------------------------------

def candidates(detections, chains, scores, fps):
    """Ranked review candidates: each chain's best CANDIDATES_PER_CHAIN scored rows, best score first.
    The top of the list is spread in time (at most one per CANDIDATE_SPACING window); the candidates
    that rule skipped follow, best score first. Returns dicts: row, frame, chain, score, rank."""
    by_chain = {}
    for row in np.flatnonzero(~np.isnan(scores)):
        by_chain.setdefault(int(chains[row]), []).append(row)
    pool = []
    for rows in by_chain.values():
        pool += sorted(rows, key=lambda r: -scores[r])[:CANDIDATES_PER_CHAIN]
    pool.sort(key=lambda r: -scores[r])
    spacing = CANDIDATE_SPACING * fps
    spread, rest = [], []
    for row in pool:
        frame = int(detections["frame"][row])
        candidate = {"row": int(row), "frame": frame, "chain": int(chains[row]), "score": float(scores[row])}
        if all(abs(frame - c["frame"]) >= spacing for c in spread):
            spread.append(candidate)
        else:
            rest.append(candidate)
    chosen = (spread + rest)[:MAX_CANDIDATES]
    for rank, candidate in enumerate(chosen, 1):
        candidate["rank"] = rank
    return chosen


# ---- review page ---------------------------------------------------------------------------------

def candidate_tiles(video, detections, candidates_, use_pano):
    """One image per candidate: the upright chest crop (large) above a thumbnail of the frame with the
    person outlined, and a label line."""
    center, radius = tuple(detections["center"]), float(detections["radius"])
    by_frame = {}
    for i, c in enumerate(candidates_):
        by_frame.setdefault(c["frame"], []).append(i)
    tiles = [None] * len(candidates_)
    cap = cv2.VideoCapture(str(video))
    for index in range(max(by_frame) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        for i in by_frame.get(index, []):
            c = candidates_[i]
            crop = upright_crop(frame, tuple(detections["chest"][c["row"]]), center, scale=1)
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)
            thumb = frame.copy()
            box = detections["box"][c["row"]]
            if use_pano:
                cv2.polylines(thumb, [box_outline(box, (center, radius))], True, (0, 255, 255), 4)
            else:
                cv2.rectangle(thumb, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (0, 255, 255), 4)
            thumb = cv2.resize(thumb, THUMB_SIZE, interpolation=cv2.INTER_AREA)
            label = np.zeros((LABEL_HEIGHT, CROP_SIZE, 3), np.uint8)
            cv2.putText(label, f"#{c['rank']}  f{c['frame']}  s{c['score']:.2f}", (6, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles[i] = np.vstack([label, crop, thumb])
    cap.release()
    return tiles


def page_image(tiles, page, selected):
    columns, rows = GRID
    per_page = columns * rows
    tile_h, tile_w = tiles[0].shape[:2]
    pad = 6
    header = 56
    image = np.full((header + rows * (tile_h + pad) + pad, columns * (tile_w + pad) + pad, 3), 40, np.uint8)
    for line, text in enumerate(HINT):
        cv2.putText(image, text, (pad, 20 + line * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    pages = (len(tiles) + per_page - 1) // per_page
    cv2.putText(image, f"page {page + 1}/{pages}  confirmed {len(selected)}", (image.shape[1] - 230, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for slot in range(per_page):
        i = page * per_page + slot
        if i >= len(tiles):
            break
        x = pad + (slot % columns) * (tile_w + pad)
        y = header + pad + (slot // columns) * (tile_h + pad)
        image[y:y + tile_h, x:x + tile_w] = tiles[i]
        if i in selected:
            cv2.rectangle(image, (x - 3, y - 3), (x + tile_w + 2, y + tile_h + 2), (0, 220, 0), 4)
            cv2.putText(image, "STAFF", (x + tile_w - 70, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)
    return image, (pad, header + pad, tile_w + pad, tile_h + pad)


def review(video, detections, candidates_, use_pano):
    """Show the candidates page by page; returns the indices of the clicked (confirmed) candidates."""
    tiles = candidate_tiles(video, detections, candidates_, use_pano)
    columns, rows = GRID
    per_page = columns * rows
    pages = (len(tiles) + per_page - 1) // per_page
    state = {"page": 0, "selected": set(), "layout": None}

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or state["layout"] is None:
            return
        x0, y0, step_x, step_y = state["layout"]
        column, row = (x - x0) // step_x, (y - y0) // step_y
        if 0 <= column < columns and 0 <= row < rows:
            i = state["page"] * per_page + row * columns + column
            if i < len(tiles):
                state["selected"] ^= {i}

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        image, state["layout"] = page_image(tiles, state["page"], state["selected"])
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("n"):
            state["page"] = min(state["page"] + 1, pages - 1)
        elif key == ord("p"):
            state["page"] = max(state["page"] - 1, 0)
    cv2.destroyAllWindows()
    return sorted(state["selected"])


# ---- confirmations -------------------------------------------------------------------------------

def save_confirmed(path, confirmed, full_row_of):
    """confirmed: candidate dicts; full_row_of: detection row (after dropping duplicates) -> row of detections.npz."""
    entries = [{"frame": c["frame"], "chain": c["chain"], "detection": int(full_row_of[c["row"]])} for c in confirmed]
    Path(path).write_text(json.dumps({"confirmed": entries}, indent=2))


def load_confirmed(path, chains, kept_row_of):
    """Chains of the confirmed detections (by their detections.npz row, so the file survives re-linking).
    kept_row_of: detections.npz row -> row after dropping duplicates, or -1."""
    staff = set()
    for entry in json.loads(Path(path).read_text())["confirmed"]:
        row = kept_row_of[entry["detection"]]
        if row < 0:
            print(f"   confirmed detection {entry['detection']} (frame {entry['frame']}) was dropped; skipped")
            continue
        staff.add(int(chains[row]))
    return staff


# ---- appearance expansion ------------------------------------------------------------------------

def chain_summaries(detections, chains, hists):
    """Per chain: first/last frame, first/last point, mean torso histogram, number of detections."""
    info = {}
    order = np.argsort(detections["frame"], kind="stable")
    for row in order:
        chain = int(chains[row])
        frame, point = int(detections["frame"][row]), detections["point"][row]
        if chain not in info:
            info[chain] = {"first": frame, "last": frame, "first_point": point, "last_point": point,
                           "hist": hists[row].astype(np.float64).copy(), "rows": 1}
        else:
            entry = info[chain]
            entry["last"], entry["last_point"] = frame, point
            entry["hist"] += hists[row]
            entry["rows"] += 1
    for entry in info.values():
        entry["hist"] /= entry["rows"]
    return info


def reachable(a, b, speed, margin):
    """Could one person walk from the end of the earlier chain to the start of the later one?"""
    first, second = (a, b) if a["last"] < b["first"] else (b, a)
    gap = second["first"] - first["last"]
    if gap <= 0:
        return False
    return np.hypot(*(second["first_point"] - first["last_point"])) <= speed * gap + margin


def expand(confirmed, info, threshold, speed, margin):
    """Add chains that look like a confirmed chain (similarity >= threshold to its mean torso
    histogram), do not overlap in time with any staff chain, and are reachable at walking speed from
    one. Greedy, most similar first, until nothing more can be added. Returns {added chain: similarity}."""
    staff = set(confirmed)
    added = {}
    while True:
        options = []
        for chain, entry in info.items():
            if chain in staff or entry["rows"] < EXPAND_MIN_ROWS:
                continue
            sim = max(similarity(entry["hist"], info[c]["hist"]) for c in confirmed)
            if sim < threshold:
                continue
            if any(entry["first"] <= info[s]["last"] and info[s]["first"] <= entry["last"] for s in staff):
                continue
            if any(reachable(entry, info[s], speed, margin) for s in staff):
                options.append((sim, chain))
        if not options:
            return added
        sim, chain = max(options)
        staff.add(chain)
        added[chain] = sim
