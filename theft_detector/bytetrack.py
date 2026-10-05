"""ByteTrack / BoT-SORT multi-object tracking.

ByteTrack (Zhang et al., ECCV 2022) associates detections in three stages and,
crucially, also uses the *low*-score detections most trackers throw away, so a
half-occluded person keeps her id instead of being dropped:

    stage 1.  live tracks   x high-score detections
    stage 2.  live tracks   x low-score  detections   <- the ByteTrack trick
    stage 3.  unconfirmed   x the remaining high-score detections

Behind it sits a constant-velocity Kalman filter over ``(cx, cy, area, aspect)``
that predicts where a box will be next frame, which is what keeps the IoU alive
while the detector blinks. Lost tracks stay in the matching pool (predicted one
step ahead), so an occlusion recovers the same id instead of minting a new one.

BoT-SORT (Aharon & Barlan, 2023) = ByteTrack + camera-motion compensation +
appearance re-identification. This port keeps the Kalman motion model and adds
lightweight ReID: an HSV colour histogram compared by cosine similarity, used
to pull a lost track back when IoU alone is gone (and it holds lost tracks
longer). Camera-motion compensation is deliberately absent - the shop camera is
fixed. ReID needs no model file, so it works with a plain OpenCV install.

Both trackers expose the same interface as
:class:`theft_detector.tracker.Tracker` (``update`` / ``confirmed`` /
``live_ids``) and emit the same :class:`~theft_detector.tracker.Track` records,
so the behaviour rules do not care which one is running:

    tracking.algorithm = "iou" | "bytetrack" | "botsort"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from .config import TrackingConfig
from .detectors import Detection
from .tracker import Track, iou

# Track states
TRK_NEW = "new"           # one/two hits, not yet reported to the rules
TRK_TRACKED = "tracked"   # matched, safe to use
TRK_LOST = "lost"         # unmatched for a while, waiting to come back


# --------------------------------------------------------------------------- #
# Kalman filter over (cx, cy, s, r) - s = area, r = width/height
# --------------------------------------------------------------------------- #
class KalmanBox:
    """Constant-velocity Kalman filter for one box (SORT/ByteTrack flavour)."""

    POS_STD = 1.0 / 20.0        # how far the box may drift per frame
    VEL_STD = 1.0 / 160.0       # how much its velocity may change
    MEAS_POS = 0.02             # detector noise, relative to the box size
    MEAS_S = 0.05
    MEAS_R = 0.04

    def __init__(self, box: tuple[int, int, int, int]) -> None:
        cx, cy, s, r = to_csr(box)
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0             # velocity is unknown at first
        self.P[3, 3] = 1e-4                 # aspect ratio barely changes

    # ------------------------------------------------------------------ #
    @property
    def box(self) -> tuple[int, int, int, int]:
        return from_csr(self.x[0], self.x[1], self.x[2], self.x[3])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])

    def predict(self) -> tuple[int, int, int, int]:
        """Advance one frame; returns the predicted box."""
        F = np.eye(8)
        F[0, 4] = F[1, 5] = F[2, 6] = 1.0
        self.x = F @ self.x
        self.x[2] = max(self.x[2], 16.0)
        self.P = F @ self.P @ F.T + self._process_noise()
        return self.box

    def update(self, box: tuple[int, int, int, int]) -> None:
        """Pull the state towards a measurement."""
        z = np.array(to_csr(box), dtype=np.float64)
        H = np.eye(4, 8)
        s = max(abs(self.x[2]), 16.0)
        R = np.diag(
            [
                (self.MEAS_POS * s) ** 2 + 1.0,
                (self.MEAS_POS * s) ** 2 + 1.0,
                (self.MEAS_S * s) ** 2 + 1.0,
                (self.MEAS_R * abs(self.x[3])) ** 2 + 1e-3,
            ]
        )
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P
        self.x[2] = max(self.x[2], 16.0)

    # ------------------------------------------------------------------ #
    def _process_noise(self) -> np.ndarray:
        s = max(abs(self.x[2]), 16.0)
        sp, sv = self.POS_STD * s, self.VEL_STD * s
        return np.diag(
            [
                sp * sp, sp * sp, (sp * 0.5) ** 2, (s * 1e-4) ** 2,
                sv * sv, sv * sv, (sv * 0.5) ** 2, 1e-6,
            ]
        )


def to_csr(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, float(w * h), w / float(h)


def from_csr(cx: float, cy: float, s: float, r: float) -> tuple[int, int, int, int]:
    s = max(16.0, s)
    r = max(1e-3, r)
    w = float(np.sqrt(s * r))
    h = s / max(w, 1e-6)
    return (
        int(round(cx - w / 2)),
        int(round(cy - h / 2)),
        int(round(cx + w / 2)),
        int(round(cy + h / 2)),
    )


# --------------------------------------------------------------------------- #
# Lightweight appearance model (BoT-SORT's ReID, without a neural net)
# --------------------------------------------------------------------------- #
def appearance(box: tuple[int, int, int, int], frame: np.ndarray | None) -> np.ndarray | None:
    """HSV colour histogram of a box - our compact ReID signature."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(w, box[2]), min(h, box[3])
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    small = cv2.resize(patch, (32, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.reshape(-1).astype(np.float32)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity of two signatures (0..1)."""
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


# --------------------------------------------------------------------------- #
@dataclass
class STrack:
    """Internal track: what the tracker itself remembers."""

    track_id: int
    kalman: KalmanBox
    score: float
    start_time: float
    last_time: float
    hits: int = 1
    misses: int = 0
    state: str = TRK_NEW
    label: str = "person"
    feat: np.ndarray | None = None
    history: list[tuple[float, int, int]] = field(default_factory=list)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.kalman.box

    @property
    def age(self) -> float:
        return max(0.0, self.last_time - self.start_time)


# --------------------------------------------------------------------------- #
def match_greedy(
    boxes: list[tuple[int, int, int, int]],
    dets: list[Detection],
    min_iou: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one matching on IoU: (pairs, unmatched boxes, unmatched dets)."""
    pairs: list[tuple[float, int, int]] = []
    for ti, box in enumerate(boxes):
        for di, det in enumerate(dets):
            score = iou(box, det.box)
            if score >= min_iou:
                pairs.append((score, ti, di))
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_t: set[int] = set()
    used_d: set[int] = set()
    kept: list[tuple[int, int]] = []
    for score, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        kept.append((ti, di))
        used_t.add(ti)
        used_d.add(di)
    return (
        kept,
        [i for i in range(len(boxes)) if i not in used_t],
        [i for i in range(len(dets)) if i not in used_d],
    )


# --------------------------------------------------------------------------- #
class BaseTracker:
    """Interface shared by the legacy IoU tracker and the ByteTrack family."""

    name = "base"
    lost_multiplier = 1          # BoT-SORT keeps lost tracks around longer

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.stracks: list[STrack] = []        # every live track, any state
        self.removed: list[STrack] = []
        self.REMOVED_KEEP = 256    # how many dropped tracks to keep for inspection
        self._next_id = 1
        self._revived: set[int] = set()        # ids ReID pulled back this frame

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        raise NotImplementedError

    def confirmed(self) -> list[Track]:
        return [t for t in self.output() if t.hits >= self.cfg.min_hits and t.misses == 0]

    def live_ids(self) -> set[int]:
        return {t.track_id for t in self.output()}

    # ------------------------------------------------------------------ #
    def output(self) -> list[Track]:
        """Internal state -> the Track records the rules consume."""
        out: list[Track] = []
        for st in self.stracks:
            out.append(
                Track(
                    track_id=st.track_id,
                    box=st.box,
                    confidence=st.score,
                    hits=st.hits,
                    misses=st.misses,
                    start_time=st.start_time,
                    last_time=st.last_time,
                    label=st.label,
                    history=st.history,
                )
            )
        return out

    # ------------------------------------------------------------------ #
    def _new_strack(self, det: Detection, timestamp: float) -> STrack:
        st = STrack(
            track_id=self._next_id,
            kalman=KalmanBox(det.box),
            score=det.confidence,
            start_time=timestamp,
            last_time=timestamp,
            label=det.label,
            history=[(timestamp, *det.center)],
        )
        self._next_id += 1
        return st

    def _hit(self, st: STrack, det: Detection, timestamp: float) -> None:
        st.kalman.update(det.box)
        st.score = det.confidence
        st.label = det.label
        st.last_time = timestamp
        st.misses = 0
        st.hits += 1
        st.state = TRK_TRACKED if st.hits >= self.cfg.min_hits else TRK_NEW
        st.history.append((timestamp, *det.center))

    def _miss(self, st: STrack) -> None:
        st.misses += 1
        st.state = TRK_LOST

    def _forget(self, st: STrack) -> None:
        """Archive a track we gave up on - bounded, so a 24/7 run cannot leak."""
        self.removed.append(st)
        if len(self.removed) > self.REMOVED_KEEP:
            del self.removed[: len(self.removed) - self.REMOVED_KEEP]

    def _retire(self, timestamp: float) -> None:
        """Drop hopelessly lost tracks, trim their histories, keep pools sane."""
        budget = self.cfg.max_misses * self.lost_multiplier
        live: list[STrack] = []
        seen: set[int] = set()
        for st in self.stracks:
            if st.track_id in seen:
                continue
            seen.add(st.track_id)
            if st.state == TRK_LOST and st.misses > budget:
                st.history.clear()              # it is gone: free its trail
                self._forget(st)
                continue
            live.append(st)
        self.stracks = live
        cutoff = timestamp - 120.0
        for st in self.stracks:
            while st.history and st.history[0][0] < cutoff:
                st.history.pop(0)


# --------------------------------------------------------------------------- #
class ByteTracker(BaseTracker):
    """ByteTrack: three-stage association, Kalman motion model, low-score rescue."""

    name = "bytetrack"
    MIN_IOU = 0.01                # almost any overlap counts as the same object
    LOW_SCORE = 0.05              # detections below this are plain noise

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        thresh = float(self.cfg.track_thresh)
        high = [d for d in detections if d.confidence >= thresh]
        low = [d for d in detections if self.LOW_SCORE <= d.confidence < thresh]
        self._revived.clear()

        for st in self.stracks:                   # everyone moves first
            st.kalman.predict()

        live = [st for st in self.stracks if st.state in (TRK_TRACKED, TRK_LOST)]
        unconfirmed = [st for st in self.stracks if st.state == TRK_NEW]

        # stage 1: live tracks x high-score detections
        pairs, rest_live, rest_high = match_greedy(
            [st.box for st in live], high, self.MIN_IOU
        )
        for ti, di in pairs:
            self._hit(live[ti], high[di], timestamp)

        # stage 2: the leftovers x low-score detections (occluded / blurred)
        left_live = [live[i] for i in rest_live]
        used_low: set[int] = set()
        if left_live and low:
            pairs2, rest_live2, _ = match_greedy(
                [st.box for st in left_live], low, self.MIN_IOU
            )
            for ti, di in pairs2:
                self._hit(left_live[ti], low[di], timestamp)
            used_low = {di for _, di in pairs2}
            left_live = [left_live[i] for i in rest_live2]

        # stage 3: unconfirmed tracks x the still unused high-score detections
        left_high = [high[i] for i in rest_high]
        if unconfirmed and left_high:
            pairs3, rest_unconf, rest_high2 = match_greedy(
                [st.box for st in unconfirmed], left_high, self.MIN_IOU
            )
            for ti, di in pairs3:
                self._hit(unconfirmed[ti], left_high[di], timestamp)
            unmatched_new = [unconfirmed[i] for i in rest_unconf]
            left_high = [left_high[i] for i in rest_high2]
        else:
            unmatched_new = list(unconfirmed)

        # leftovers that nothing claimed
        leftovers = list(left_high) + [
            d for i, d in enumerate(low) if i not in used_low
        ]
        leftovers = self._recover(leftovers, timestamp, frame, left_live)

        # state transitions
        for st in left_live:
            if st.track_id in self._revived:
                continue                       # ReID pulled it back (BoT-SORT)
            self._miss(st)
        for st in unmatched_new:
            st.misses += 1
            if st.misses > 2:                     # one-off false positive
                st.history.clear()
                self._forget(st)
                if st in self.stracks:
                    self.stracks.remove(st)
                continue
            st.state = TRK_LOST                   # give it another chance
        for det in leftovers:
            if det.confidence >= thresh:
                self.stracks.append(self._new_strack(det, timestamp))

        self._remember_appearance(timestamp, frame)
        self._retire(timestamp)
        return self.output()

    # ------------------------------------------------------------------ #
    def _recover(
        self,
        leftovers: list[Detection],
        timestamp: float,
        frame: np.ndarray | None,
        candidates: list[STrack],
    ) -> list[Detection]:
        """Hook: BoT-SORT re-associates leftovers by appearance.

        ``candidates`` are the tracks that failed IoU this frame - they would
        be marked lost if nothing claims them.
        """
        return leftovers

    def _remember_appearance(self, timestamp: float, frame: np.ndarray | None) -> None:
        """Hook: BoT-SORT keeps an appearance signature per track."""


# --------------------------------------------------------------------------- #
class BoTSortTracker(ByteTracker):
    """ByteTrack + colour-histogram ReID (fixed camera, so no CMC step)."""

    name = "botsort"
    lost_multiplier = 3           # an identified person may vanish longer

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        self._frame = frame
        return super().update(detections, timestamp, frame)

    # ------------------------------------------------------------------ #
    def _recover(
        self,
        leftovers: list[Detection],
        timestamp: float,
        frame: np.ndarray | None,
        candidates: list[STrack],
    ) -> list[Detection]:
        if not leftovers or frame is None:
            return leftovers
        threshold = float(self.cfg.reid_threshold)
        margin = float(self.cfg.reid_max_distance)
        lost = [st for st in candidates if st.feat is not None]
        # most recently seen first: it is the likeliest to still be there
        lost.sort(key=lambda s: s.last_time, reverse=True)

        remaining = list(leftovers)
        for st in lost:
            best_i, best = -1, 0.0
            for i, det in enumerate(remaining):
                if det.confidence < self.cfg.track_thresh or det.label != st.label:
                    continue
                sim = similarity(st.feat, appearance(det.box, frame))
                if sim < threshold:
                    continue
                # ReID is a tie-breaker, not permission to teleport
                if not (
                    iou(st.box, det.box) > 0.0
                    or _centers_within(st.box, det.box, margin)
                ):
                    continue
                if sim > best:
                    best, best_i = sim, i
            if best_i < 0:
                continue
            det = remaining.pop(best_i)
            self._hit(st, det, timestamp)
            self._revived.add(st.track_id)        # do not mark it missing below
            feat = appearance(det.box, frame)
            if feat is not None:
                st.feat = feat
        return remaining

    # ------------------------------------------------------------------ #
    def _remember_appearance(self, timestamp: float, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        for st in self.stracks:
            feat = appearance(st.box, frame)
            if feat is None:
                continue
            if st.feat is None:
                st.feat = feat
            else:                                  # slow EMA: robust to lighting
                st.feat = (0.9 * st.feat + 0.1 * feat).astype(np.float32)
                cv2.normalize(st.feat, st.feat)


def _centers_within(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: float
) -> bool:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return abs(ax - bx) <= margin and abs(ay - by) <= margin


# --------------------------------------------------------------------------- #
def build_tracker(cfg: TrackingConfig) -> BaseTracker:
    """``tracking.algorithm`` -> tracker instance."""
    algo = (getattr(cfg, "algorithm", "bytetrack") or "bytetrack").lower()
    if algo in ("iou", "sort", "legacy"):
        from .tracker import Tracker

        return Tracker(cfg)                       # type: ignore[return-value]
    if algo in ("bytetrack", "byte"):
        return ByteTracker(cfg)
    if algo in ("botsort", "bot-sort"):
        return BoTSortTracker(cfg)
    raise ValueError(f"unknown tracking.algorithm {algo!r} (expected iou|bytetrack|botsort)")


def iter_labels(tracks: Iterable[Track]) -> set[str]:
    """Convenience for logs/tests: the labels a batch of tracks carries."""
    return {getattr(t, "label", "person") for t in tracks}
