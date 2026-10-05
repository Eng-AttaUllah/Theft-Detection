"""Object IDs - tracking the *things* a person can take.

Person IDs come from the person tracker (ByteTrack / BoT-SORT). This module
runs a **second** tracker over the non-person detections of a YOLO/SSD model
and turns their trajectories into a small state machine:

    on_shelf   still sitting where it was first seen (inside a shelf zone)
    carried    held by somebody - it sits on top of a live person track
    moved      displaced from its anchor without a carrier (pushed/slid)
    gone       the track vanished while it was being carried

Anchor + zone + carrier are what the ``object_taken`` rule and the theft
confidence stage consume: "the bottle that stood on shelf_a for 10 minutes is
suddenly on person #4 and then leaves the frame" is a much sharper signal than
"some pixels changed".

With the ``hog`` / ``motion`` backends there are no object detections at all,
so this stage stays silent and everything else behaves as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bytetrack import ByteTracker
from .config import TrackingConfig
from .detectors import Detection
from .tracker import Track, iou
from .zones import Zone

# object states
ON_SHELF = "on_shelf"
CARRIED = "carried"
MOVED = "moved"
IDLE = "idle"


# --------------------------------------------------------------------------- #
@dataclass
class ObjectTrack:
    """One tracked thing and its story."""

    object_id: int
    label: str
    box: tuple[int, int, int, int]
    anchor: tuple[int, int, int, int]
    state: str = IDLE
    alive: bool = True
    carrier: int | None = None          # person track id holding it, if any
    anchor_zone: str | None = None      # shelf/register zone it started in
    carried_since: float | None = None
    released_at: float | None = None
    moved_distance: float = 0.0
    confidence: float = 0.0
    start_time: float = 0.0
    last_time: float = 0.0
    hits: int = 0

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2

    @property
    def size(self) -> int:
        return max(1, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    @property
    def anchor_size(self) -> int:
        return max(1, (self.anchor[2] - self.anchor[0]) * (self.anchor[3] - self.anchor[1]))

    @property
    def age(self) -> float:
        return max(0.0, self.last_time - self.start_time)

    def in_zone(self, zone: Zone, margin: int = 0) -> bool:
        x1, y1, x2, y2 = zone.bbox()
        cx, cy = self.center
        return x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin


# --------------------------------------------------------------------------- #
class ObjectTracker:
    """Person tracker (inherited) + object tracker + the object state machine."""

    def __init__(
        self,
        tracking: TrackingConfig,
        params: dict | None = None,
        keep_seconds: float = 30.0,
    ) -> None:
        # objects deserve a proper Kalman tracker even when people use the
        # legacy one: small boxes move far between frames
        self._tracker = ByteTracker(tracking)
        self.p = {
            "carrier_margin": 80,       # px, person box that counts as holding it
            "move_ratio": 0.35,         # fraction of its own size = "moved"
            "release_seconds": 1.5,     # no carrier for this long -> not carried
            "lost_seconds": 6.0,        # keep a vanished object around this long
        }
        if params:
            for key in list(self.p):
                if key in params:
                    self.p[key] = params[key]
        self.keep_seconds = float(keep_seconds)
        self.objects: dict[int, ObjectTrack] = {}
        self._prev: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: list[Detection],
        persons: list[Track],
        zones: list[Zone],
        timestamp: float,
    ) -> list[ObjectTrack]:
        if not detections and not self.objects:
            return []

        tracks = self._tracker.update(detections, timestamp)
        live = {t.track_id: t for t in self._tracker.confirmed()}

        for tr in tracks:
            if tr.track_id not in live:
                continue
            obj = self.objects.get(tr.track_id)
            if obj is None:
                obj = self._adopt(tr, zones, timestamp)
            self._advance(obj, tr, persons, zones, timestamp)

        # nothing visible here any more: remember it briefly, note if it went
        # missing while somebody was holding it
        for obj in self.objects.values():
            if obj.object_id in live or not obj.alive:
                continue
            obj.alive = False
            if obj.state != CARRIED:
                obj.carrier = None

        self._garbage(timestamp)
        return [
            o for o in self.objects.values()
            if o.alive or (timestamp - o.last_time) <= self.keep_seconds
        ]

    # ------------------------------------------------------------------ #
    def _adopt(self, tr: Track, zones: list[Zone], timestamp: float) -> ObjectTrack:
        obj = ObjectTrack(
            object_id=tr.track_id,
            label=getattr(tr, "label", "object"),
            box=tr.box,
            anchor=tr.box,
            start_time=timestamp,
            last_time=timestamp,
            hits=tr.hits,
            confidence=tr.confidence,
        )
        for z in zones:
            if z.kind in ("shelf", "register") and z.contains(*obj.center):
                obj.anchor_zone = z.name
                obj.state = ON_SHELF
                break
        self.objects[obj.object_id] = obj
        return obj

    # ------------------------------------------------------------------ #
    def _advance(
        self,
        obj: ObjectTrack,
        tr: Track,
        persons: list[Track],
        zones: list[Zone],
        timestamp: float,
    ) -> None:
        obj.box = tr.box
        obj.last_time = timestamp
        obj.hits = tr.hits
        obj.alive = True
        obj.confidence = tr.confidence

        ax, ay = (obj.anchor[0] + obj.anchor[2]) / 2, (obj.anchor[1] + obj.anchor[3]) / 2
        cx, cy = obj.center
        obj.moved_distance = max(obj.moved_distance, math.hypot(cx - ax, cy - ay))

        # who is holding it?
        carrier = self._carrier(obj, persons)
        if carrier is not None:
            obj.carrier = carrier
            obj.released_at = None
            if obj.carried_since is None:
                obj.carried_since = timestamp
        else:
            if obj.released_at is None:
                obj.released_at = timestamp
            if (
                obj.carried_since is not None
                and timestamp - obj.released_at > float(self.p["release_seconds"])
            ):
                obj.carried_since = None
                obj.carrier = None

        moved_enough = obj.moved_distance > float(self.p["move_ratio"]) * math.sqrt(
            obj.anchor_size
        )
        held = (
            obj.carrier is not None
            and obj.carried_since is not None
            and timestamp - obj.carried_since >= 0.4
        )
        if held:
            obj.state = CARRIED
        elif moved_enough:
            obj.state = MOVED
        elif obj.anchor_zone:
            obj.state = ON_SHELF
        else:
            obj.state = IDLE

    # ------------------------------------------------------------------ #
    def _carrier(self, obj: ObjectTrack, persons: list[Track]) -> int | None:
        """Person whose box the object sits in - i.e. who is holding it."""
        margin = float(self.p["carrier_margin"])
        cx, cy = obj.center
        best_id, best_score = None, 0.0
        for p in persons:
            if p.misses:
                continue
            px1, py1, px2, py2 = p.box
            # a held item sits *inside* the holder: plain IoU would punish
            # that (small box inside a big one), so containment wins
            contained = (
                px1 - margin <= cx <= px2 + margin
                and py1 - margin <= cy <= py2 + margin
            )
            score = 0.6 if contained else iou(obj.box, p.box)
            if score > best_score:
                best_id, best_score = p.track_id, score
        return best_id if best_score >= 0.3 else None

    # ------------------------------------------------------------------ #
    def _garbage(self, timestamp: float) -> None:
        dead = [
            oid
            for oid, o in self.objects.items()
            if not o.alive and timestamp - o.last_time > self.keep_seconds
        ]
        for oid in dead:
            self.objects.pop(oid, None)
