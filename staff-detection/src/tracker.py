"""A small point tracker: constant-velocity prediction + Hungarian matching on distance."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

VELOCITY_SMOOTHING = 0.5  # weight of the old velocity when a track is matched


@dataclass
class Track:
    id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missed: int = 0


class Tracker:
    # max_dist in pixels, max_missed in frames
    def __init__(self, max_dist, max_missed, use_velocity=True):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.use_velocity = use_velocity
        self.tracks = []
        self.next_id = 1

    def predict(self, track):
        return track.position + track.velocity if self.use_velocity else track.position

    # match detection points (N, 2) to tracks; returns the track id for each detection
    def update(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        predicted = np.array([self.predict(t) for t in self.tracks]).reshape(-1, 2)
        cost = np.linalg.norm(predicted[:, None, :] - points[None, :, :], axis=2)
        # forbidden pairs get a cost no allowed pair can reach, then get filtered out below
        rows, cols = linear_sum_assignment(np.where(cost > self.max_dist, 1e9, cost))
        pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= self.max_dist]

        detection_ids = [None] * len(points)
        matched_tracks = set()
        for r, c in pairs:
            track = self.tracks[r]
            step = points[c] - track.position
            track.velocity = VELOCITY_SMOOTHING * track.velocity + (1 - VELOCITY_SMOOTHING) * step
            track.position = points[c]
            track.missed = 0
            matched_tracks.add(r)
            detection_ids[c] = track.id

        for r, track in enumerate(self.tracks):
            if r not in matched_tracks:
                track.position = self.predict(track)
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for c, point in enumerate(points):
            if detection_ids[c] is None:
                self.tracks.append(Track(self.next_id, point))
                detection_ids[c] = self.next_id
                self.next_id += 1
        return detection_ids
