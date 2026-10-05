"""Behaviour rules that turn detections/tracks into theft alerts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from .alerts import Event
from .detectors import Detection
from .objects import CARRIED, ObjectTrack
from .tracker import Track
from .zones import Zone


# --------------------------------------------------------------------------- #
@dataclass
class FrameContext:
    """Everything a rule may look at for a single frame."""

    frame: np.ndarray
    gray: np.ndarray
    timestamp: float
    frame_index: int
    fps: float
    zones: list[Zone]
    tracks: list[Track]                      # confirmed tracks only
    detections: list[Detection] = field(default_factory=list)
    objects: list[ObjectTrack] = field(default_factory=list)   # object-ID stage
    person_mask: np.ndarray | None = None    # uint8, full resolution
    flow_mag: np.ndarray | None = None       # optical flow magnitude map
    flow_scale: float = 1.0                  # full-res coords -> flow coords
    brightness: float = 128.0
    sharpness: float = 0.0
    global_diff: float = 0.0

    # ------------------------------------------------------------------ #
    def zone_of(self, point: tuple[int, int], kinds: Iterable[str] | None) -> Zone | None:
        for z in self.zones:
            if kinds and z.kind not in kinds:
                continue
            if z.contains(*point):
                return z
        return None

    def person_near(self, zone: Zone, margin: int = 150) -> bool:
        """True when somebody's *feet* are close to the zone (ground plane).

        Using the feet rather than the whole box matters: a customer at the
        counter has their body right under the shelf zone but stands far away
        from it on the floor. The margin only has to bridge the distance the
        camera projects between a shelf face and the floor in front of it.
        """
        x1, y1, x2, y2 = zone.bbox()
        for t in self.tracks:
            fx, fy = t.feet
            dx = max(x1 - fx, 0, fx - x2)
            dy = max(y1 - fy, 0, fy - y2)
            if dx * dx + dy * dy <= margin * margin:
                return True
        return False

    def track_in_zone(self, zone: Zone) -> bool:
        for t in self.tracks:
            if zone.contains(*t.feet) or zone.contains(*t.center):
                return True
            x1, y1, x2, y2 = zone.bbox()
            if _intersects(t.box, (x1, y1, x2, y2)):
                return True
        return False


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# --------------------------------------------------------------------------- #
# Shelf / zone background models (shared between object rules)
# --------------------------------------------------------------------------- #
@dataclass
class ZoneState:
    zone: Zone
    bg: np.ndarray | None = None           # float32 crop of the zone
    changed_run: int = 0
    change_start: float | None = None
    last_person: float | None = None
    ever_person: bool = False
    occupancy: float = 0.0
    changed_ratio: float = 0.0
    largest_ratio: float = 0.0
    coherence: float = 0.0               # largest blob / all changed pixels
    observed: bool = True                  # enough unmasked pixels to judge
    consumed_by: str | None = None
    quiet: int = 0                         # frames without change evidence
    changed_mask: np.ndarray | None = None # pixel set of the current change
    person_hint: float | None = None       # a person is/was here (incl. footprint)
    last_alert_ts: float | None = None     # when we last reported this zone
    last_alert_mask: np.ndarray | None = None


class ZoneBackgrounds:
    """Slow background model per zone, used to notice objects appearing or
    vanishing while nobody (or somebody) is standing in front of them."""

    PARAMS = {
        "diff_threshold": 28,
        "change_ratio": 0.010,
        "largest_change_ratio": 0.015,
        # an object vanishing leaves ONE coherent hole; a person shuffling out
        # of frame, lighting and sensor noise leave scattered fragments. The
        # real removal measures ~1.0 here, a person-shaped ghost ~0.7.
        "min_blob_coherence": 0.80,
        "bg_alpha": 0.06,
        "freeze_ratio": 0.03,
        # how close (px) somebody's *feet* must be to the zone to count as
        # "at the shelf". Camera geometry decides this: a camera looking down
        # projects the feet of a standing shopper well below the shelf face,
        # so the number has to cover that gap (the demo needs ~150).
        "person_margin": 150,
        "settle_seconds": 20.0,
        "min_valid_ratio": 0.15,
        "clear_frames": 30,               # quiet frames that end an episode
        "lost_person_seconds": 60.0,      # how long a track we lost is remembered
        "shadow_margin": 15,              # px around a lost track box
        "person_fill_ratio": 0.5,         # box that differs that much = still there
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.p = dict(self.PARAMS)
        if params:
            for key in self.p:
                if key in params:
                    self.p[key] = params[key]
        self.states: dict[str, ZoneState] = {}
        self._last_frame = -1
        # track id -> (last time it was seen alive, its box); used to keep
        # covering the spot where a motion based detector just lost somebody
        self._last_alive: dict[int, tuple[float, tuple[int, int, int, int]]] = {}
        self._dead_boxes: list[tuple[int, float, tuple[int, int, int, int]]] = []
        self._person_like: set[int] = set()

    # ------------------------------------------------------------------ #
    def update(
        self,
        zones: list[Zone],
        gray: np.ndarray,
        person_mask: np.ndarray,
        tracks: list[Track],
        timestamp: float,
        frame_index: int,
    ) -> dict[str, ZoneState]:
        if frame_index == self._last_frame:
            return self.states
        self._last_frame = frame_index

        # Remember the last known position of tracks that just disappeared: a
        # motion detector loses people the moment they stand still, and their
        # body would otherwise be read as a sudden change in the scene.
        alive: dict[int, tuple[int, int, int, int]] = {}
        for t in tracks:
            tid = getattr(t, "track_id", None)
            if tid is not None:
                alive[tid] = t.box
                self._last_alive[tid] = (timestamp, t.box)
        lost = float(self.p["lost_person_seconds"])
        for tid, (seen, _box) in list(self._last_alive.items()):
            if tid not in alive and timestamp - seen > lost:
                del self._last_alive[tid]
        self._dead_boxes = [
            (tid, seen, box) for tid, (seen, box) in self._last_alive.items()
            if tid not in alive
        ]
        self._person_like = set()

        for zone in zones:
            state = self.states.get(zone.name)
            if state is None:
                state = ZoneState(zone=zone)
                self.states[zone.name] = state
            self._update_zone(state, gray, person_mask, tracks, timestamp)

        # a lost track whose footprint still looks like him is him - keep him
        # remembered for as long as he keeps standing there
        for tid in self._person_like:
            entry = self._last_alive.get(tid)
            if entry is not None:
                self._last_alive[tid] = (timestamp, entry[1])
        return self.states

    # ------------------------------------------------------------------ #
    def _update_zone(
        self,
        state: ZoneState,
        gray: np.ndarray,
        person_mask: np.ndarray,
        tracks: list[Track],
        timestamp: float,
    ) -> None:
        x1, y1, x2, y2 = state.zone.bbox()
        h, w = gray.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            state.observed = False
            return

        crop = gray[y1:y2, x1:x2]
        person = person_mask[y1:y2, x1:x2] > 0
        zh, zw = crop.shape[:2]

        # a person standing close to the zone counts as "present"
        margin = int(self.p["person_margin"])
        person_present = False
        for t in tracks:
            tx1, ty1, tx2, ty2 = t.box
            fx, fy = (tx1 + tx2) // 2, ty2          # feet = ground position
            dx = max(x1 - fx, 0, fx - x2)
            dy = max(y1 - fy, 0, fy - y2)
            if dx * dx + dy * dy <= margin * margin:
                person_present = True
                break
        if person_present:
            state.last_person = timestamp
            state.ever_person = True

        if state.bg is None:
            state.bg = crop.astype(np.float32)
            state.observed = int((~person).sum()) >= (
                float(self.p["min_valid_ratio"]) * (x2 - x1) * (y2 - y1)
            )
            return

        # Which footprints of tracks we lost still look like their owner? He is
        # still standing there when nearly all of his last box differs from the
        # background; anything smaller - the gap an item left behind, say - is a
        # real change and does not get that protection.
        raw_diff = (
            np.abs(crop.astype(np.float32) - state.bg)
            > float(self.p["diff_threshold"])
        )
        lost_box = np.zeros(crop.shape, dtype=bool)
        grow = int(self.p["shadow_margin"])
        fill_ratio = float(self.p["person_fill_ratio"])
        for tid, _seen, (bx1, by1, bx2, by2) in self._dead_boxes:
            sx1, sy1 = max(0, bx1 - x1), max(0, by1 - y1)
            sx2, sy2 = min(zw, bx2 - x1), min(zh, by2 - y1)
            if sx2 - sx1 < 4 or sy2 - sy1 < 4:
                continue
            area = (sx2 - sx1) * (sy2 - sy1)
            if int(raw_diff[sy1:sy2, sx1:sx2].sum()) < fill_ratio * area:
                continue                      # that is not him (any more)
            self._person_like.add(tid)
            mx1, my1 = max(0, sx1 - grow), max(0, sy1 - grow)
            mx2, my2 = min(zw, sx2 + grow), min(zh, sy2 + grow)
            lost_box[my1:my2, mx1:mx2] = True

        if person_present or lost_box.any():
            # a person is here, or stood here until a moment ago - enough for
            # the "somebody was at the shelf" test used by the removal rule
            state.person_hint = timestamp
            state.ever_person = True

        occupied = person | lost_box
        valid = ~occupied
        valid_area = int(valid.sum())
        zone_area = (x2 - x1) * (y2 - y1)

        if valid_area < self.p["min_valid_ratio"] * zone_area:
            # person is blocking the zone - keep the previous verdict
            state.observed = False
            state.occupancy = 1.0 - valid_area / max(1, zone_area)
            return

        state.observed = True
        state.occupancy = 1.0 - valid_area / zone_area

        # pixels right next to a person are their outline/limbs leaking around
        # the mask - exclude them so standing people never look like changes
        near_mask = np.zeros(valid.shape, dtype=bool)
        if occupied.any():
            near_mask = cv2.dilate(
                occupied.astype(np.uint8), np.ones((31, 31), np.uint8)
            ).astype(bool)

        changed = raw_diff & valid & ~near_mask
        state.changed_mask = changed
        state.changed_ratio = float(changed.sum()) / valid_area

        # largest coherent changed blob -> an object, not noise/lighting
        changed_px = int(changed.sum())
        largest = 0
        if changed_px:
            num, _, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
            if num > 1:
                largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        state.largest_ratio = largest / zone_area
        state.coherence = (largest / changed_px) if changed_px else 0.0

        is_change = (
            state.changed_ratio >= float(self.p["change_ratio"])
            and state.largest_ratio >= float(self.p["largest_change_ratio"])
            and state.coherence >= float(self.p["min_blob_coherence"])
        )

        if is_change:
            if state.changed_run == 0:
                state.change_start = timestamp
                state.consumed_by = None
            state.changed_run += 1
            state.quiet = 0
        else:
            # a single masked/ambiguous frame must not end the episode,
            # otherwise the same shelf change would alert over and over
            state.quiet += 1
            if state.quiet > int(self.p["clear_frames"]):
                state.changed_run = 0
                state.change_start = None
                state.consumed_by = None
                state.quiet = 0

        settled = (
            state.change_start is not None
            and (timestamp - state.change_start) >= float(self.p["settle_seconds"])
        )

        if (not is_change) or settled:
            crop_f = crop.astype(np.float32)
            # only pixels that are actually visible (no person in front) take
            # part in the update - everything else must stay untouched
            if settled and state.changed_run:
                # the new scene state becomes the baseline: adopt what we can
                # see, otherwise the very same change starts a new episode the
                # moment the timer expires
                state.bg[valid] = crop_f[valid]
                state.changed_run = 0
                state.change_start = None
                state.consumed_by = None
                state.quiet = 0
            else:
                alpha = float(self.p["bg_alpha"])
                state.bg[valid] += alpha * (crop_f[valid] - state.bg[valid])

    # ------------------------------------------------------------------ #
    def state(self, zone: Zone) -> ZoneState | None:
        return self.states.get(zone.name)


# --------------------------------------------------------------------------- #
# Rule base
# --------------------------------------------------------------------------- #
class Rule(ABC):
    name = "rule"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @property
    def enabled(self) -> bool:
        return bool(self.params.get("enabled", True))

    @abstractmethod
    def update(self, ctx: FrameContext) -> list[Event]:
        raise NotImplementedError

    def _severity(self, default: str = "medium") -> str:
        return str(self.params.get("severity", default))

    def _zones(self, ctx: FrameContext, default_kinds: list[str]) -> list[Zone]:
        kinds = self.params.get("zone_kinds")
        if kinds is None:
            kinds = default_kinds
        if not kinds:
            return list(ctx.zones)
        return [z for z in ctx.zones if z.kind in kinds]


# --------------------------------------------------------------------------- #
class RestrictedZoneRule(Rule):
    """A person walking into the counter / stock room area."""

    name = "restricted_zone"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.state: dict[tuple[int, str], dict[str, Any]] = {}

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        dwell_needed = float(self.params.get("min_dwell_seconds", 1.0))
        live: set[tuple[int, str]] = set()

        for track in ctx.tracks:
            fx, fy = track.feet
            for zone in self._zones(ctx, ["restricted", "register"]):
                key = (track.track_id, zone.name)
                if not zone.contains(fx, fy):
                    continue
                live.add(key)
                st = self.state.get(key)
                if st is None:
                    st = {"since": ctx.timestamp, "emitted": False}
                    self.state[key] = st
                dwell = ctx.timestamp - st["since"]
                if not st["emitted"] and dwell >= dwell_needed:
                    st["emitted"] = True
                    events.append(
                        Event(
                            rule=self.name,
                            severity=zone.severity,
                            zone=zone.name,
                            timestamp=ctx.timestamp,
                            track_ids=[track.track_id],
                            message=(
                                f"person #{track.track_id} is inside the "
                                f"{zone.kind} zone '{zone.name}' ({dwell:.1f}s)"
                            ),
                        )
                    )
        for key in list(self.state):
            if key not in live:
                del self.state[key]
        return events


# --------------------------------------------------------------------------- #
class LoiteringRule(Rule):
    """Someone staying in the same spot for too long.

    State is keyed by *place* rather than by track id: trackers churn (ids get
    recreated when a detector blips), but a spot keeps its clock as long as
    somebody stands around it for at most ``gap_seconds`` in a row.
    """

    name = "loitering"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.spots: list[dict[str, Any]] = []

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        max_seconds = float(self.params.get("max_seconds", 10.0))
        radius = float(self.params.get("max_radius", 60.0))
        gap = float(self.params.get("gap_seconds", 4.0))

        for track in ctx.tracks:
            fx, fy = track.feet
            spot = None
            for s in self.spots:
                if math.hypot(fx - s["x"], fy - s["y"]) <= radius:
                    spot = s
                    break
            if spot is None:
                spot = {"x": fx, "y": fy, "since": ctx.timestamp,
                        "last": ctx.timestamp, "emitted": False, "ids": set()}
                self.spots.append(spot)
            spot["last"] = ctx.timestamp
            spot["ids"].add(track.track_id)

            dwell = ctx.timestamp - spot["since"]
            if dwell >= max_seconds and not spot["emitted"]:
                spot["emitted"] = True
                zone = ctx.zone_of((int(spot["x"]), int(spot["y"])), None)
                events.append(
                    Event(
                        rule=self.name,
                        severity=self._severity("medium"),
                        zone=zone.name if zone else None,
                        timestamp=ctx.timestamp,
                        track_ids=sorted(spot["ids"]),
                        message=(
                            f"person #{track.track_id} has been loitering in the "
                            f"same spot for {dwell:.0f}s"
                        ),
                    )
                )

        self.spots = [s for s in self.spots if ctx.timestamp - s["last"] <= gap]
        return events


# --------------------------------------------------------------------------- #
class ItemRemovalRule(Rule):
    """Something changed inside a shelf zone while a person was next to it."""

    name = "item_removal"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.bg: ZoneBackgrounds = shared.get("zone_bg") or ZoneBackgrounds(
            {k: v for k, v in params.items() if k in ZoneBackgrounds.PARAMS}
        )

    @staticmethod
    def _is_repeat(st: ZoneState, overlap_needed: float, repeat_gap: float,
                   timestamp: float) -> bool:
        """True when this is a change we already raised an alert for.

        The pixels of a removed item stay "unknown" while somebody stands in
        front of them, so the very same change reappears once that person
        moves on - it must not produce a second alarm.
        """
        if st.last_alert_ts is None or st.last_alert_mask is None:
            return False
        if st.changed_mask is None or st.changed_mask.shape != st.last_alert_mask.shape:
            return False
        if timestamp - st.last_alert_ts > repeat_gap:
            return False
        total = int(st.changed_mask.sum())
        if total <= 0:
            return False
        overlap = int(np.logical_and(st.changed_mask, st.last_alert_mask).sum())
        return overlap >= overlap_needed * total

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        hold = int(self.params.get("hold_frames", 10))
        grace = float(self.params.get("person_grace_seconds", 8.0))
        settle = float(self.bg.p["settle_seconds"])
        repeat_gap = float(self.params.get("repeat_gap_seconds", 30.0))
        overlap_needed = float(self.params.get("duplicate_overlap", 0.4))

        if ctx.person_mask is None:
            return events

        states = self.bg.update(
            ctx.zones, ctx.gray, ctx.person_mask, ctx.tracks,
            ctx.timestamp, ctx.frame_index,
        )
        for zone in self._zones(ctx, ["shelf"]):
            st = states.get(zone.name)
            if st is None or not st.observed or st.change_start is None:
                continue
            if st.changed_run < hold or st.consumed_by is not None:
                continue
            hint = st.person_hint if st.person_hint is not None else st.last_person
            if hint is None:
                continue
            if (ctx.timestamp - hint) > grace:
                continue
            if (ctx.timestamp - st.change_start) > settle:
                continue

            # the same shelf change can show up again once the person who hid
            # it moves away - report each physical change only once
            if self._is_repeat(st, overlap_needed, repeat_gap, ctx.timestamp):
                st.consumed_by = self.name
                continue

            st.consumed_by = self.name
            st.last_alert_ts = ctx.timestamp
            st.last_alert_mask = (
                st.changed_mask.copy() if st.changed_mask is not None else None
            )
            events.append(
                Event(
                    rule=self.name,
                    severity=self._severity("high"),
                    zone=zone.name,
                    timestamp=ctx.timestamp,
                    confidence=min(1.0, 0.5 + st.largest_ratio * 4),
                    message=(
                        f"item likely taken from '{zone.name}' - shelf layout changed "
                        f"while a person was at the shelf"
                    ),
                )
            )
        return events


# --------------------------------------------------------------------------- #
class UnattendedObjectRule(Rule):
    """An object appeared/disappeared and nobody is around it any more."""

    name = "unattended_object"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.bg: ZoneBackgrounds = shared.get("zone_bg") or ZoneBackgrounds(
            {k: v for k, v in params.items() if k in ZoneBackgrounds.PARAMS}
        )

    def update(self, ctx: FrameContext) -> list[Event]:
        events: list[Event] = []
        hold = int(self.params.get("hold_frames", 30))
        idle = float(self.params.get("unattended_seconds", 10.0))

        if ctx.person_mask is None:
            return events

        states = self.bg.update(
            ctx.zones, ctx.gray, ctx.person_mask, ctx.tracks,
            ctx.timestamp, ctx.frame_index,
        )
        for zone in self._zones(ctx, ["shelf", "register"]):
            st = states.get(zone.name)
            if st is None or not st.observed or st.change_start is None:
                continue
            if st.changed_run < hold or st.consumed_by is not None:
                continue
            if not st.ever_person or st.last_person is None:
                continue
            if (ctx.timestamp - st.last_person) < idle:
                continue
            st.consumed_by = self.name
            events.append(
                Event(
                    rule=self.name,
                    severity=self._severity("medium"),
                    zone=zone.name,
                    timestamp=ctx.timestamp,
                    message=(
                        f"unattended object detected in '{zone.name}' - the scene "
                        f"changed and nobody is near it"
                    ),
                )
            )
        return events


# --------------------------------------------------------------------------- #
class RapidMotionRule(Rule):
    """Sudden fast movement near a shelf/counter (snatching, grabbing)."""

    name = "rapid_motion"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.runs: dict[str, int] = {}

    def update(self, ctx: FrameContext) -> list[Event]:
        if ctx.flow_mag is None:
            return []
        events: list[Event] = []
        threshold = float(self.params.get("threshold", 9.0))
        min_frames = int(self.params.get("min_frames", 3))
        zones = self._zones(ctx, ["shelf", "register", "restricted"])
        if not zones:
            zones = list(ctx.zones)

        for zone in zones:
            name = zone.name
            x1, y1, x2, y2 = zone.bbox()
            s = ctx.flow_scale
            fx1, fy1, fx2, fy2 = int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s)
            sub = ctx.flow_mag[fy1:fy2, fx1:fx2]
            if sub.size == 0:
                continue
            hi = sub[sub > 1.0]
            enough = hi.size >= max(4, 0.005 * sub.size)
            score = float(hi.mean()) if enough else 0.0
            margin = int(self.params.get("person_margin", 150))
            person = ctx.person_near(zone, margin=margin)

            if person and score >= threshold:
                self.runs[name] = self.runs.get(name, 0) + 1
            else:
                self.runs[name] = 0

            if self.runs[name] >= min_frames:
                self.runs[name] = 0
                ids = [t.track_id for t in ctx.tracks
                       if _near_zone(t.box, zone.bbox(), margin)]
                events.append(
                    Event(
                        rule=self.name,
                        severity=self._severity("medium"),
                        zone=zone.name,
                        timestamp=ctx.timestamp,
                        track_ids=ids,
                        confidence=min(1.0, score / (threshold * 2)),
                        message=(
                            f"sudden fast movement near '{zone.name}' "
                            f"(motion {score:.1f} px/frame)"
                        ),
                    )
                )
        return events


def _near_zone(box: tuple[int, int, int, int], bbox: tuple[int, int, int, int],
               margin: int) -> bool:
    m = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
    return _intersects(box, m)


# --------------------------------------------------------------------------- #
class CrowdRule(Rule):
    """Several people gathered together - a typical distraction-theft setup."""

    name = "crowd"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.since: float | None = None
        self.last_enough: float | None = None

    def update(self, ctx: FrameContext) -> list[Event]:
        need = int(self.params.get("min_persons", 3))
        seconds = float(self.params.get("seconds", 2.0))
        drop = float(self.params.get("drop_seconds", 1.5))
        count = len(ctx.tracks)

        if count >= need:
            if self.since is None:
                self.since = ctx.timestamp
            self.last_enough = ctx.timestamp
            if ctx.timestamp - self.since >= seconds:
                ids = [t.track_id for t in ctx.tracks]
                self.since = None
                self.last_enough = None
                return [
                    Event(
                        rule=self.name,
                        severity=self._severity("medium"),
                        timestamp=ctx.timestamp,
                        track_ids=ids,
                        message=f"{count} people gathered in view (possible distraction)",
                    )
                ]
        elif self.since is not None and self.last_enough is not None:
            # tolerate short detection drop-outs, reset only after `drop`
            if ctx.timestamp - self.last_enough > drop:
                self.since = None
                self.last_enough = None
        return []


# --------------------------------------------------------------------------- #
class CameraTamperRule(Rule):
    """Covered / frozen / out-of-focus / over-exposed camera."""

    name = "camera_tamper"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self.counts: dict[str, int] = {k: 0 for k in ("dark", "bright", "blur", "frozen")}

    def update(self, ctx: FrameContext) -> list[Event]:
        if ctx.frame_index < 10:
            return []
        need = int(self.params.get("min_frames", 8))
        problems: list[tuple[str, str]] = []

        if ctx.brightness < float(self.params.get("min_brightness", 22.0)):
            self.counts["dark"] += 1
            if self.counts["dark"] >= need:
                problems.append(("dark", "camera image is very dark - lens may be covered"))
        else:
            self.counts["dark"] = 0

        if ctx.brightness > float(self.params.get("max_brightness", 238.0)):
            self.counts["bright"] += 1
            if self.counts["bright"] >= need:
                problems.append(("bright", "camera image is blown out / whitewashed"))
        else:
            self.counts["bright"] = 0

        if ctx.sharpness < float(self.params.get("min_sharpness", 6.0)):
            self.counts["blur"] += 1
            if self.counts["blur"] >= need:
                problems.append(("blur", "camera image is blurred or out of focus"))
        else:
            self.counts["blur"] = 0

        static_seconds = float(self.params.get("static_seconds", 8.0))
        if ctx.global_diff < float(self.params.get("min_diff", 0.5)):
            self.counts["frozen"] += 1
            if self.counts["frozen"] * (1.0 / max(ctx.fps, 1e-6)) >= static_seconds:
                problems.append(("frozen", "video looks frozen - camera may be tampered with"))
        else:
            self.counts["frozen"] = 0

        events = []
        for kind, message in problems:
            self.counts[kind] = 0
            events.append(
                Event(
                    rule=self.name,
                    severity=self._severity("high"),
                    timestamp=ctx.timestamp,
                    message=message,
                )
            )
        return events


# --------------------------------------------------------------------------- #
class ObjectTakenRule(Rule):
    """An object-id left the shelf it was anchored to.

    This is the sharp end of the pipeline: *person id* + *object id* + *where
    the object came from*. It only fires when an object detector (``onnx`` /
    ``caffe``) is running - with ``hog``/``motion`` there are no object ids and
    the rule simply never sees anything.
    """

    name = "object_taken"

    def __init__(self, params: dict[str, Any], shared: dict[str, Any]) -> None:
        super().__init__(params)
        self._done: dict[int, float] = {}

    def update(self, ctx: FrameContext) -> list[Event]:
        if not ctx.objects:
            return []
        carry_min = float(self.params.get("min_carry_seconds", 2.0))
        leave_margin = int(self.params.get("leave_margin", 60))
        kinds = self.params.get("zone_kinds") or ["shelf", "register"]
        zones = {z.name: z for z in ctx.zones if z.kind in kinds}
        events: list[Event] = []

        for obj in ctx.objects:
            if obj.object_id in self._done:
                continue
            if not obj.anchor_zone or obj.anchor_zone not in zones:
                continue                       # it never started on a watched shelf
            zone = zones[obj.anchor_zone]

            held_since = obj.carried_since
            carried = held_since is not None and (
                ctx.timestamp - held_since
            ) >= carry_min and obj.carrier is not None
            vanished = (not obj.alive) and obj.state == CARRIED
            if not (carried and (vanished or not obj.in_zone(zone, leave_margin))):
                continue

            self._done[obj.object_id] = ctx.timestamp
            who = f" by person #{obj.carrier}" if obj.carrier else ""
            how = "disappeared while held" if vanished else "left"
            events.append(
                Event(
                    rule=self.name,
                    severity=zone.severity,
                    zone=zone.name,
                    timestamp=ctx.timestamp,
                    track_ids=[obj.carrier] if obj.carrier else [],
                    confidence=0.8,
                    message=(
                        f"{obj.label} '{obj.anchor_zone}' was {how}{who} "
                        f"(object #{obj.object_id})"
                    ),
                )
            )

        if len(self._done) > 400:
            for key in sorted(self._done, key=self._done.get)[:100]:
                del self._done[key]
        return events


# --------------------------------------------------------------------------- #
RULE_CLASSES: dict[str, type[Rule]] = {
    RestrictedZoneRule.name: RestrictedZoneRule,
    LoiteringRule.name: LoiteringRule,
    ItemRemovalRule.name: ItemRemovalRule,
    UnattendedObjectRule.name: UnattendedObjectRule,
    RapidMotionRule.name: RapidMotionRule,
    CrowdRule.name: CrowdRule,
    CameraTamperRule.name: CameraTamperRule,
    ObjectTakenRule.name: ObjectTakenRule,
}


def build_rules(
    rules_cfg: dict[str, dict[str, Any]],
    shared: dict[str, Any],
) -> list[Rule]:
    rules: list[Rule] = []
    for name, cls in RULE_CLASSES.items():
        if name not in rules_cfg:
            continue                       # not configured -> not active
        params = dict(rules_cfg[name])
        if not params.get("enabled", True):
            continue
        rules.append(cls(params, shared))
    return rules
