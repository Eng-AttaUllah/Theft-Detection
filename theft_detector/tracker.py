"""Lightweight IoU multi-object tracker (SORT style, no Kalman filter)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TrackingConfig
from .detectors import Detection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float = 0.0
    hits: int = 1
    misses: int = 0
    start_time: float = 0.0
    last_time: float = 0.0
    label: str = "person"
    history: list[tuple[float, int, int]] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2

    @property
    def feet(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, self.box[3]

    @property
    def age(self) -> float:
        return max(0.0, self.last_time - self.start_time)

    @property
    def area(self) -> int:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])

    def overlaps(self, box: tuple[int, int, int, int], min_ratio: float = 0.0) -> float:
        """Fraction of ``box`` covered by this track (or raw IoU when 0)."""
        ix1, iy1 = max(self.box[0], box[0]), max(self.box[1], box[1])
        ix2, iy2 = min(self.box[2], box[2]), min(self.box[3], box[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if min_ratio <= 0:
            area_b = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            return inter / area_b
        area_b = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        return inter / area_b


class Tracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        frame: np.ndarray | None = None,       # unused: this tracker is geometry only
    ) -> list[Track]:
        matched_track: set[int] = set()
        matched_det: set[int] = set()

        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = iou(tr.box, det.box)
                if score >= self.cfg.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for _, ti, di in pairs:
            if ti in matched_track or di in matched_det:
                continue
            track = self.tracks[ti]
            det = detections[di]
            track.box = det.box
            track.confidence = det.confidence
            track.label = det.label
            track.hits += 1
            track.misses = 0
            track.last_time = timestamp
            track.history.append((timestamp, *det.center))
            matched_track.add(ti)
            matched_det.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_track:
                track.misses += 1
                track.last_time = timestamp

        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            cx, cy = det.center
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    box=det.box,
                    confidence=det.confidence,
                    label=det.label,
                    hits=1,
                    misses=0,
                    start_time=timestamp,
                    last_time=timestamp,
                    history=[(timestamp, cx, cy)],
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_misses]
        # keep histories short (2 minutes of positions is plenty)
        cutoff = timestamp - 120.0
        for t in self.tracks:
            while t.history and t.history[0][0] < cutoff:
                t.history.pop(0)
        return list(self.tracks)

    # ------------------------------------------------------------------ #
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.hits >= self.cfg.min_hits and t.misses == 0]

    def live_ids(self) -> set[int]:
        return {t.track_id for t in self.tracks}
