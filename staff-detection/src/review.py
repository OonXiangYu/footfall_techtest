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
COLUMNS = 4  # tiles per row on a review page
PER_PAGE = 8  # tiles per page (4 x 2 keeps them large enough to recognise people)
FALLBACK_SCREEN = (1280, 800)  # if the screen size cannot be read
SCREEN_USE = (0.95, 0.85)  # share of the screen the window may take (leaves room for title bar / taskbar)
KEYS_HELP = "Review keys: click a tile = toggle staff, n = next page, p = previous page, Enter = done"
CROP_SIZE, THUMB_SIZE = 200, (200, 150)
LABEL_HEIGHT = 34
WINDOW = "staff review"
HINT = ["Click the staff member (toggle). The staff may change clothes:",
        "confirm one candidate per outfit.   n / p = next / previous page, Enter = done"]

CONFIRM_WINDOW = 5.0  # seconds either side of a click kept as staff with --confirm-scope window

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


def page_image(tiles, page, selected, per_page=PER_PAGE):
    """The page at full size, and its layout (x0, y0, step_x, step_y, columns, rows) in those pixels."""
    columns = min(COLUMNS, per_page)
    rows = (per_page + columns - 1) // columns
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
    return image, (pad, header + pad, tile_w + pad, tile_h + pad, columns, rows)


def screen_size():
    """(width, height) of the screen, or FALLBACK_SCREEN if it cannot be read."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:  # no display / no tkinter: fall back to a size that fits most laptops
        return FALLBACK_SCREEN


def fit_scale(image_size, ui_height=None):
    """Scale (<= 1) that makes a page of image_size (width, height) fit the screen, or ui_height px tall."""
    width, height = image_size
    screen_w, screen_h = screen_size()
    max_w, max_h = screen_w * SCREEN_USE[0], screen_h * SCREEN_USE[1]
    if ui_height:
        max_h = ui_height
    return min(1.0, max_w / width, max_h / height)


def tile_at(x, y, layout, scale, page, per_page, count):
    """Candidate index under a click at (x, y) in the shown (scaled) window, or None."""
    x0, y0, step_x, step_y, columns, rows = layout
    x, y = x / scale, y / scale  # back to full-size page pixels
    column, row = int((x - x0) // step_x), int((y - y0) // step_y)
    if not (0 <= column < columns and 0 <= row < rows):
        return None
    # the gap to the right of / below a tile is not part of it
    if (x - x0) % step_x >= step_x - 6 or (y - y0) % step_y >= step_y - 6:
        return None
    slot = row * columns + column
    i = page * per_page + slot
    return i if slot < per_page and i < count else None


def review(video, detections, candidates_, use_pano, per_page=PER_PAGE, ui_height=None):
    """Show the candidates page by page; returns the indices of the clicked (confirmed) candidates."""
    tiles = candidate_tiles(video, detections, candidates_, use_pano)
    pages = (len(tiles) + per_page - 1) // per_page
    full, _ = page_image(tiles, 0, set(), per_page)
    scale = fit_scale((full.shape[1], full.shape[0]), ui_height)
    print(f"   {KEYS_HELP}")
    print(f"   {len(tiles)} candidates on {pages} page(s) of {per_page}; window scaled to {scale:.0%} "
          f"({round(full.shape[1] * scale)} x {round(full.shape[0] * scale)} px; --ui-height to change)")
    state = {"page": 0, "selected": set(), "layout": None}

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or state["layout"] is None:
            return
        i = tile_at(x, y, state["layout"], scale, state["page"], per_page, len(tiles))
        if i is not None:
            state["selected"] ^= {i}

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        image, state["layout"] = page_image(tiles, state["page"], state["selected"], per_page)
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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
    """The confirmed clicks as [(chain, frame)], found by their detections.npz row so the file survives
    re-linking. kept_row_of: detections.npz row -> row after dropping duplicates, or -1."""
    clicks = []
    for entry in json.loads(Path(path).read_text())["confirmed"]:
        row = kept_row_of[entry["detection"]]
        if row < 0:
            print(f"   confirmed detection {entry['detection']} (frame {entry['frame']}) was dropped; skipped")
            continue
        clicks.append((int(chains[row]), int(entry["frame"])))
    return clicks


def scope_chains(chains, frames, clicks, scope, window_frames):
    """Chain ids with the confirmed scope applied, and the confirmed ids.
    chain: a click confirms its whole linked chain.
    window: only the chain's detections within window_frames of a click on it are confirmed; the rest
    of that chain becomes a separate (unconfirmed) chain."""
    confirmed = {chain for chain, _ in clicks}
    if scope == "chain":
        return chains, confirmed
    scoped = chains.copy()
    offset = int(chains.max()) + 1
    for chain in confirmed:
        clicked = np.array([frame for c, frame in clicks if c == chain])
        rows = np.flatnonzero(chains == chain)
        outside = np.abs(frames[rows][:, None] - clicked[None, :]).min(axis=1) > window_frames
        scoped[rows[outside]] = chain + offset
    return scoped, confirmed


def staff_from_clicks(detections, chains, hists, clicks, scope, window_frames, threshold, radius):
    """Staff from confirmed clicks: (chain ids after scoping, staff chains, {chain: 2 confirmed / 1 added},
    added chains). threshold None: no appearance expansion."""
    scoped, confirmed = scope_chains(chains, detections["frame"], clicks, scope, window_frames)
    added = {}
    if threshold is not None and confirmed:
        info = chain_summaries(detections, scoped, hists)
        added = expand(confirmed, info, threshold, EXPAND_SPEED * radius, EXPAND_MARGIN * radius)
    weights = {c: 2 for c in confirmed}
    weights.update({c: 1 for c in added})
    return scoped, set(confirmed) | set(added), weights, added


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
