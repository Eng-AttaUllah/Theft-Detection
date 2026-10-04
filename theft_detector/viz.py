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


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    people: int,
    backend: str,
    alert_count: int,
    recent: deque,
    zones_busy: set[str],
) -> None:
    h, w = frame.shape[:2]
    bar_h = 34
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
    hud = f"FPS {fps:4.1f}   persons {people}   detector {backend}   alerts {alert_count}"
    cv2.putText(
        frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    if zones_busy:
        text = "IN ZONE: " + ", ".join(sorted(zones_busy))
        cv2.putText(
            frame, text, (w - 10 - int(0.55 * len(text) * 11), 24),
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
