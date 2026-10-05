"""Drawing helpers: zones, tracks, HUD and the alert ticker."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .alerts import Event
from .tracker import Track
from .zones import Zone


def draw_zones(frame: np.ndarray, zones: list[Zone], busy: set[str]) -> None:
    for z in zones:
        z.draw(frame, active=z.name in busy)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    for t in tracks:
        x1, y1, x2, y2 = t.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"#{t.track_id}"
        cv2.putText(
            frame, label, (x1 + 4, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
        # feet marker - the point tested against zones
        fx, fy = t.feet
        cv2.circle(frame, (fx, fy), 4, (0, 255, 0), -1)


def draw_objects(frame: np.ndarray, objects: list) -> None:
    """Cyan/amber boxes for tracked objects; red once they are gone."""
    colors = {
        "on_shelf": (255, 255, 0),
        "carried": (0, 165, 255),
        "moved": (255, 0, 255),
        "idle": (200, 200, 200),
        "gone": (0, 0, 255),
    }
    for o in objects:
        x1, y1, x2, y2 = o.box
        state = o.state if o.alive else "gone"
        color = colors.get(state, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        who = f" <-#{o.carrier}" if o.carrier and state == "carried" else ""
        cv2.putText(
            frame, f"{o.label}#{o.object_id} {state}{who}",
            (x1, min(frame.shape[0] - 6, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    people: int,
    backend: str,
    alert_count: int,
    recent: deque,
    zones_busy: set[str],
    tracker: str = "iou",
    theft: float = 0.0,
    theft_level: str = "clear",
    objects: int = 0,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 34
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
    hud = (
        f"FPS {fps:4.1f}   persons {people}   objects {objects}   "
        f"{backend}/{tracker}   alerts {alert_count}"
    )
    cv2.putText(
        frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )

    # theft confidence: a live bar that turns amber, then red
    score = max(0.0, min(1.0, theft))
    x0, y0, bw, bh = w - 216, 12, 160, 10
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), -1)
    fill = int(bw * score)
    if fill > 0:
        color = (0, 0, 255) if theft_level == "high" else (
            (0, 165, 255) if theft_level == "elevated" else (0, 200, 200)
        )
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bh), color, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (150, 150, 150), 1)
    cv2.putText(
        frame, f"theft {score:.2f}", (x0 + bw + 8, y0 + bh + 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
    )

    if zones_busy:
        text = "IN ZONE: " + ", ".join(sorted(zones_busy))
        cv2.putText(
            frame, text, (w - 10 - int(0.55 * len(text) * 11), 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )

    if recent:
        y = h - 14 - 22 * len(recent)
        for text, severity in recent:
            color = {
                "high": (0, 0, 255),
                "medium": (0, 200, 255),
                "low": (0, 255, 255),
            }.get(severity, (255, 255, 255))
            cv2.rectangle(frame, (6, y - 17), (w - 6, y + 7), (0, 0, 0), -1)
            cv2.putText(
                frame, text[:110], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 2, cv2.LINE_AA,
            )
            y += 22


def event_line(ev: Event) -> str:
    where = f" @{ev.zone}" if ev.zone else ""
    return f"[{ev.severity}] {ev.rule}{where}: {ev.message}"
