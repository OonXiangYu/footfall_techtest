"""Offline tracklet linking: join a track that ends to one that starts shortly after, nearby."""

import numpy as np
from scipy.optimize import linear_sum_assignment

# defaults, as fractions of the fisheye radius (distances) or seconds (gap)
LINK_SPEED = 0.01  # link distance per frame of gap (~ the staff member's median speed)
LINK_MARGIN = 0.05  # extra link distance
MAX_LINK_DIST = 0.20  # cap, so long gaps cannot allow jumps across the room
LINK_GAP = 4.0  # max seconds between a track end and the next start
# ratio test: link only if clearly better than the next-best option; chosen by chain purity
# (eval_own_tracker.py: 98% with ratio + 4 s gap vs 64% for plain 1.5 s linking)
LINK_RATIO = 0.5
# appearance (torso colour histogram similarity, 1 - Bhattacharyya): same person (one track's start
# vs end) >= 0.6 in 73% of cases, two people in the same frame in 4% (measured on this video)
APP_MIN = 0.6
APP_MARGIN = 0.2  # a rescued link must beat every competing candidate's similarity by this much
LONG_GAP = 10.0  # seconds; only for still people (distance <= LONG_DISTANCE) who look the same
LONG_DISTANCE = 0.05  # fraction of the radius


def track_summaries(frames, points, ids):
    """For each track id: (first frame, last frame, first point, last point)."""
    summaries = {}
    for frame, point, track_id in zip(frames, points, ids):
        if track_id not in summaries:
            summaries[track_id] = [frame, frame, point, point]
        else:
            summaries[track_id][1] = frame
            summaries[track_id][3] = point
    return summaries


def link_tracks(summaries, max_gap, speed, margin, max_distance, ratio=None, appearance=None,
                long_gap=None, long_distance=None):
    """One-to-one links end -> start, chosen by the Hungarian method on distance.

    A link from track a to track b is allowed if b starts 1..max_gap frames after a ends and
    the jump is at most min(speed * gap + margin, max_distance) pixels; the cap stops long gaps
    from allowing jumps across the room. With a ratio, a link is also only allowed if it is
    clearly the best option for both tracks (see ratio_test). Returns {track id: next track id}.

    appearance(a, b) -> similarity in [0, 1] of a's end and b's start, optional. With it:
    - a link the ratio test refuses is still allowed if its similarity is at least APP_MIN and
      APP_MARGIN above every competing candidate of both tracks (see appearance_rescue);
    - with long_gap, a second pass links the ends and starts the first pass left over when b
      starts up to long_gap frames later, within long_distance pixels, and looks the same
      (someone seated or standing still who drops out of detection). It is a separate pass so
      these far-in-time candidates never make a normal link look ambiguous.
    """
    def near(gap, distance):
        return gap <= max_gap and distance <= min(speed * gap + margin, max_distance)

    successor = _assign(summaries, sorted(summaries), sorted(summaries), near, ratio, appearance, None)
    if appearance is not None and long_gap is not None:
        ends = sorted(set(summaries) - set(successor))
        starts = sorted(set(summaries) - set(successor.values()))

        def still(gap, distance):
            return max_gap < gap <= long_gap and distance <= long_distance

        successor.update(_assign(summaries, ends, starts, still, ratio, appearance, APP_MIN))
    return successor


def _assign(summaries, ends, starts, allowed_pair, ratio, appearance, min_similarity):
    """One-to-one links from track ends to track starts among the pairs allowed_pair(gap, distance)
    accepts (and, with min_similarity, that look alike), Hungarian on distance."""
    cost = np.full((len(ends), len(starts)), np.inf)
    sim = np.full((len(ends), len(starts)), np.nan)
    for i, a in enumerate(ends):
        _, a_last, _, a_point = summaries[a]
        for j, b in enumerate(starts):
            b_first, _, b_point, _ = summaries[b]
            gap = b_first - a_last
            if gap <= 0:
                continue
            distance = np.hypot(*np.subtract(b_point, a_point))
            if not allowed_pair(gap, distance):
                continue
            if appearance is not None:
                sim[i, j] = appearance(a, b)
            if min_similarity is None or sim[i, j] >= min_similarity:
                cost[i, j] = distance
    allowed = np.isfinite(cost)
    if ratio is not None:
        passed = ratio_test(cost, ratio)
        if appearance is not None:
            passed |= appearance_rescue(cost, sim)
        allowed &= passed
    if not allowed.any():
        return {}
    # forbidden pairs get a cost no allowed pair can reach, then get filtered out below
    rows, cols = linear_sum_assignment(np.where(allowed, cost, 1e12))
    return {ends[r]: starts[c] for r, c in zip(rows, cols) if allowed[r, c]}


def ratio_test(cost, ratio):
    """True where a pair's distance is < ratio x the next-best option of BOTH tracks: the ending
    track's other starts (row) and the starting track's other ends (column). A pair with no
    competitor passes. Ambiguous ends (two people nearby) are left unlinked rather than guessed."""
    passed = np.zeros(cost.shape, bool)
    for i, j in zip(*np.nonzero(np.isfinite(cost))):
        row_others = np.delete(cost[i], j)
        col_others = np.delete(cost[:, j], i)
        passed[i, j] = all(cost[i, j] < ratio * others.min() if np.isfinite(others).any() else True
                           for others in (row_others, col_others))
    return passed


def appearance_rescue(cost, sim):
    """True where an allowed pair looks like the same person and clearly more so than every other
    candidate of both tracks: similarity >= APP_MIN and >= APP_MARGIN above the competitors' best."""
    rescued = np.zeros(cost.shape, bool)
    for i, j in zip(*np.nonzero(np.isfinite(cost))):
        if not sim[i, j] >= APP_MIN:
            continue
        competitors = [sim[i, k] for k in np.flatnonzero(np.isfinite(cost[i])) if k != j]
        competitors += [sim[k, j] for k in np.flatnonzero(np.isfinite(cost[:, j])) if k != i]
        competitors = [s for s in competitors if not np.isnan(s)]
        rescued[i, j] = not competitors or sim[i, j] >= max(competitors) + APP_MARGIN
    return rescued


def chain_ids(summaries, successor):
    """Map every track id to its chain id (the id of the first track in the chain)."""
    has_predecessor = set(successor.values())
    chains = {}
    for start in summaries:
        if start in has_predecessor:
            continue
        track = start
        while track is not None:
            chains[track] = start
            track = successor.get(track)
    return chains
