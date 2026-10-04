#!/usr/bin/env python3
"""Generate a synthetic shop video used to demo/test the theft detector.

The scene is drawn from the zone layout in the config file, so the shelves in
the video match the ``shelf`` zones the rules watch.  The clip contains a
scripted shoplifting sequence:

* a customer walks behind the counter (restricted zone)
* the same person stands at a shelf and an item disappears from it
* the person loiters at the shelf, then makes a fast grab (rapid motion)
* three more people gather at another shelf (crowd / distraction)

Usage::

    python scripts/make_demo_video.py --config configs/demo.json --out samples/demo.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from theft_detector.config import Config  # noqa: E402
from theft_detector.zones import Zone  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 30
DURATION = 36.0

FLOOR = (196, 190, 184)          # warm grey floor (BGR)
WALL = (120, 116, 112)
RACK = (70, 62, 56)
BOARD = (150, 140, 130)
COUNTER = (90, 70, 50)


# --------------------------------------------------------------------------- #
@dataclass
class Actor:
    name: str
    color: tuple[int, int, int]
    appear: float
    vanish: float
    keys: list[tuple[float, float, float]]      # (time, x, feet_y) normalised
    arm_window: tuple[float, float] | None = None
    removed: list[tuple[float, str, int]] = field(default_factory=list)

    def position(self, t: float) -> tuple[float, float] | None:
        if t < self.appear or t > self.vanish:
            return None
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1], ks[0][2]
        for i in range(len(ks) - 1):
            t0, x0, y0 = ks[i]
            t1, x1, y1 = ks[i + 1]
            if t0 <= t <= t1:
                u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        return ks[-1][1], ks[-1][2]

    def speed(self, t: float) -> float:
        eps = 1.0 / FPS
        p0, p1 = self.position(t - eps), self.position(t + eps)
        if p0 is None or p1 is None:
            return 0.0
        dx = (p1[0] - p0[0]) * WIDTH
        dy = (p1[1] - p0[1]) * HEIGHT
        return math.hypot(dx, dy) / (2 * eps) / FPS      # px per frame


ACTORS: list[Actor] = [
    Actor(
        name="A",
        color=(255, 150, 40),
        appear=2.0,
        vanish=23.5,
        keys=[
            (2.0, 0.02, 0.86),
            (6.0, 0.70, 0.80),     # walks behind the counter
            (9.0, 0.72, 0.78),
            (11.5, 0.33, 0.68),    # moves to shelf A
            (13.0, 0.33, 0.68),
            (13.15, 0.45, 0.66),   # fast grab
            (13.35, 0.33, 0.68),
            (20.0, 0.33, 0.68),    # loiters at the shelf
            (23.5, -0.06, 0.86),   # leaves through the entrance
        ],
        arm_window=(12.9, 13.4),
        removed=[(13.05, "shelf_a", 9)],      # bottom board, right-most product
    ),
    Actor(
        name="B",
        color=(60, 60, 235),
        appear=16.0,
        vanish=36.0,
        keys=[(16.0, 0.02, 0.80), (20.0, 0.58, 0.59), (36.0, 0.58, 0.59)],
    ),
    Actor(
        name="C",
        color=(70, 205, 70),
        appear=18.0,
        vanish=36.0,
        keys=[(18.0, 0.02, 0.76), (22.5, 0.72, 0.60), (36.0, 0.72, 0.60)],
    ),
    Actor(
        name="D",
        color=(40, 190, 235),
        appear=21.0,
        vanish=36.0,
        keys=[(21.0, 0.02, 0.72), (25.0, 0.66, 0.58), (36.0, 0.66, 0.58)],
    ),
]


# --------------------------------------------------------------------------- #
def shelf_products(zone: Zone, board: int) -> list[tuple[int, int, int, int]]:
    """Product rectangles for one shelf board (board 0 = top, 1 = bottom)."""
    x1, y1, x2, y2 = _bbox(zone)
    zw, zh = x2 - x1, y2 - y1
    pw, ph = int(0.13 * zw), int(0.22 * zh)
    board_y = y1 + int((0.30, 0.74)[board] * zh)
    out = []
    for i in range(5):
        x = x1 + int(0.06 * zw) + i * int(0.19 * zw)
        out.append((x, board_y - ph, x + pw, board_y))
    return out


def _bbox(zone: Zone) -> tuple[int, int, int, int]:
    zone.resolve(WIDTH, HEIGHT)
    return zone.bbox()


def draw_shelf(frame: np.ndarray, zone: Zone, products: list[dict]) -> None:
    x1, y1, x2, y2 = _bbox(zone)
    cv2.rectangle(frame, (x1 + 8, y1 + 6), (x2 - 8, y2 - 6), RACK, -1)
    for board in (0, 1):
        by = y1 + int((0.30, 0.74)[board] * (y2 - y1))
        cv2.rectangle(frame, (x1 + 8, by), (x2 - 8, by + 7), BOARD, -1)
    for p in products:
        if p["gone"]:
            continue
        cv2.rectangle(frame, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]),
                      p["color"], -1)
        cv2.rectangle(frame, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]),
                      (30, 30, 30), 1)


def draw_background(zones: list[Zone]) -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), FLOOR, np.uint8)
    img[: int(HEIGHT * 0.18)] = WALL
    cv2.rectangle(img, (0, int(HEIGHT * 0.18)), (WIDTH, int(HEIGHT * 0.18) + 4),
                  (90, 86, 82), -1)

    by_name = {z.name: z for z in zones}

    for z in zones:
        if z.kind == "shelf":
            pass                                   # drawn per frame (products)
        elif z.kind == "register":
            x1, y1, x2, y2 = _bbox(z)
            cv2.rectangle(img, (x1 + 10, y1 + 60), (x2 - 10, y2 - 10), COUNTER, -1)
            cv2.rectangle(img, (x1 + 10, y1 + 60), (x2 - 10, y2 - 10),
                          (40, 30, 20), 3)
            cv2.rectangle(img, (x1 + 40, y1 + 20), (x1 + 130, y1 + 60),
                          (60, 60, 70), -1)          # till
        elif z.kind == "restricted":
            x1, y1, x2, y2 = _bbox(z)
            cv2.rectangle(img, (x1, y1), (x2, y2), (70, 66, 62), -1)
            for off in range(0, (x2 - x1) + (y2 - y1), 40):
                cv2.line(img, (x1 + off, y1), (x1, y1 + off), (110, 105, 100), 3)
        elif z.kind == "entrance":
            x1, y1, x2, y2 = _bbox(z)
            cv2.rectangle(img, (x1, y1), (x1 + 26, y2), (80, 76, 72), -1)
            cv2.rectangle(img, (x1, y1), (x2, y1 + 10), (80, 76, 72), -1)

    if "stock_room" in by_name:                     # sign
        x1, y1, _, _ = _bbox(by_name["stock_room"])
        cv2.putText(img, "STOCK ROOM", (x1 + 12, y1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    return img


def draw_person(frame: np.ndarray, cx: float, feet: float, color, phase: float,
                speed: float, arm_out: bool) -> None:
    cx, feet = int(cx), int(feet)
    swing = int(min(14, speed * 4) * math.sin(phase * 2 * math.pi))
    # legs
    cv2.rectangle(frame, (cx - 20 + swing // 3, feet - 58), (cx - 4, feet), (45, 45, 55), -1)
    cv2.rectangle(frame, (cx + 4, feet - 58), (cx + 20 - swing // 3, feet), (45, 45, 55), -1)
    # torso
    cv2.rectangle(frame, (cx - 24, feet - 122), (cx + 24, feet - 54), color, -1)
    # arms
    if arm_out:
        cv2.line(frame, (cx + 20, feet - 112), (cx + 74, feet - 96), color, 11)
        cv2.line(frame, (cx - 20, feet - 112), (cx - 26, feet - 70), color, 11)
    else:
        cv2.line(frame, (cx + 20, feet - 112), (cx + 26 + swing // 4, feet - 66), color, 11)
        cv2.line(frame, (cx - 20, feet - 112), (cx - 26 - swing // 4, feet - 66), color, 11)
    # head
    cv2.circle(frame, (cx, feet - 140), 17, (90, 140, 190), -1)
    cv2.circle(frame, (cx, feet - 140), 17, (40, 60, 80), 2)


# --------------------------------------------------------------------------- #
def build(config_path: Path, out_path: Path, duration: float = DURATION) -> Path:
    cfg = Config.load(config_path)
    for z in cfg.zones:
        z.resolve(WIDTH, HEIGHT)

    zones_by_kind: dict[str, list[Zone]] = {}
    for z in cfg.zones:
        zones_by_kind.setdefault(z.kind, []).append(z)

    # product state per shelf zone
    product_state: dict[str, list[dict]] = {}
    palette = [(60, 90, 230), (70, 190, 70), (200, 170, 50), (200, 80, 60),
               (170, 120, 210)]
    for z in zones_by_kind.get("shelf", []):
        items = []
        for board in (0, 1):
            for i, box in enumerate(shelf_products(z, board)):
                items.append({"box": box, "color": palette[(i + board) % len(palette)],
                              "gone": False})
        product_state[z.name] = items

    background = draw_background(cfg.zones)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    frames = int(duration * FPS)
    rng = np.random.default_rng(7)
    for i in range(frames):
        t = i / FPS
        frame = background.copy()

        for actor in ACTORS:
            for when, zone_name, idx in actor.removed:
                if t >= when and zone_name in product_state:
                    product_state[zone_name][idx]["gone"] = True

        for z in zones_by_kind.get("shelf", []):
            draw_shelf(frame, z, product_state[z.name])

        for actor in ACTORS:
            pos = actor.position(t)
            if pos is None:
                continue
            x, y = pos
            cx = x * WIDTH
            feet = y * HEIGHT
            arm = bool(actor.arm_window and actor.arm_window[0] <= t <= actor.arm_window[1])
            draw_person(frame, cx, feet, actor.color, t * 4.0, actor.speed(t), arm)

        # camera sensor noise, keeps "frozen camera" detection honest
        noise = rng.normal(0, 4.0, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return out_path


# --------------------------------------------------------------------------- #
def ensure(config_path: Path, out_path: Path) -> Path:
    """Build the demo clip unless it already exists."""
    if out_path.exists():
        return out_path
    print(f"[demo] generating {out_path} ...")
    return build(config_path, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "demo.json"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "samples" / "demo.mp4"))
    ap.add_argument("--duration", type=float, default=DURATION)
    args = ap.parse_args()
    path = build(Path(args.config), Path(args.out), args.duration)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
