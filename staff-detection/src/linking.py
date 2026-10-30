"""Offline tracklet linking: join a track that ends to one that starts shortly after, nearby."""

import numpy as np
from scipy.optimize import linear_sum_assignment


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


def link_tracks(summaries, max_gap, speed, margin, max_distance):
    """One-to-one links end -> start, chosen by the Hungarian method on distance.

    A link from track a to track b is allowed if b starts 1..max_gap frames after a ends and
    the jump is at most min(speed * gap + margin, max_distance) pixels; the cap stops long gaps
    from allowing jumps across the room. Returns {track id: next track id}.
    """
    ids = sorted(summaries)
    cost = np.full((len(ids), len(ids)), np.inf)
    for i, a in enumerate(ids):
        _, a_last, _, a_point = summaries[a]
        for j, b in enumerate(ids):
            b_first, _, b_point, _ = summaries[b]
            gap = b_first - a_last
            if 0 < gap <= max_gap:
                distance = np.hypot(*np.subtract(b_point, a_point))
                if distance <= min(speed * gap + margin, max_distance):
                    cost[i, j] = distance
    allowed = np.isfinite(cost)
    if not allowed.any():
        return {}
    # forbidden pairs get a cost no allowed pair can reach, then get filtered out below
    rows, cols = linear_sum_assignment(np.where(allowed, cost, 1e12))
    return {ids[r]: ids[c] for r, c in zip(rows, cols) if allowed[r, c]}


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
