"""Shop layout zones (counter, shelves, stock room, ...).

Zones are defined with normalised coordinates (0..1) in ``config.json`` so the
same configuration works for any camera resolution.  They are resolved to
pixel contours once the frame size is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

# BGR colours per zone kind.
ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "restricted": (0, 0, 255),
    "register": (0, 140, 255),
    "shelf": (255, 170, 0),
    "entrance": (0, 200, 0),
    "default": (180, 180, 0),
}


@dataclass
class Zone:
    """A polygonal region of the shop floor."""

    name: str
    points: list[tuple[float, float]]
    kind: str = "restricted"
    severity: str = "high"
    _contour: np.ndarray | None = field(default=None, init=False, repr=False)
    _size: tuple[int, int] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zone":
        pts = [ (float(p[0]), float(p[1])) for p in data["points"] ]
        if len(pts) < 3:
            raise ValueError(f"zone {data.get('name')!r} needs at least 3 points")
        return cls(
            name=str(data.get("name", "zone")),
            points=pts,
            kind=str(data.get("kind", "restricted")),
            severity=str(data.get("severity", "high")),
        )

    # ------------------------------------------------------------------ #
    def resolve(self, width: int, height: int) -> None:
        """Convert normalised points to a pixel contour for this frame size."""
        if self._size == (width, height):
            return
        pts = np.array(
            [[int(round(x * width)), int(round(y * height))] for x, y in self.points],
            dtype=np.int32,
        )
        self._contour = pts.reshape((-1, 1, 2))
        self._size = (width, height)

    @property
    def contour(self) -> np.ndarray:
        if self._contour is None:
            raise RuntimeError(f"zone {self.name!r} has not been resolved yet")
        return self._contour

    @property
    def color(self) -> tuple[int, int, int]:
        return ZONE_COLORS.get(self.kind, ZONE_COLORS["default"])

    # ------------------------------------------------------------------ #
    def bbox(self) -> tuple[int, int, int, int]:
        """``(x1, y1, x2, y2)`` bounding box of the polygon."""
        x, y, w, h = cv2.boundingRect(self.contour)
        return x, y, x + w, y + h

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.contour, (float(x), float(y)), False) >= 0

    def mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Binary mask (uint8) of the polygon for a frame of ``shape``."""
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [self.contour], 1)
        return m

    def area(self) -> int:
        return int(cv2.contourArea(self.contour))

    def draw(self, frame: np.ndarray, active: bool = False, alpha: float = 0.16) -> None:
        color = (0, 255, 255) if active else self.color
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.contour], color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
        cv2.polylines(frame, [self.contour], True, color, 2, cv2.LINE_AA)
        x, y = self.bbox()[0], self.bbox()[1]
        cv2.putText(
            frame,
            f"{self.name} [{self.kind}]",
            (x + 6, max(18, y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def resolve_zones(zones: Iterable[Zone], width: int, height: int) -> None:
    for z in zones:
        z.resolve(width, height)
