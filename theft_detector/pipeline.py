"""Main detection loop: capture -> detect -> track -> rules -> alerts."""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alerts import AlertManager, Event
from .behaviors import FrameContext, Rule, ZoneBackgrounds, build_rules
from .config import Config
from .detectors import build_detector
from .tracker import Tracker
from .viz import draw_hud, draw_tracks, draw_zones, event_line
from .webpreview import PreviewServer
from .zones import resolve_zones

FLOW_WIDTH = 640          # optical-flow / global statistics resolution
PERSON_DILATE = 21        # px, grows the person mask over arms/shadows
PERSON_MEMORY = 2.5       # s, keep masking a person's last box after a blip


def gui_available() -> bool:
    """True when this OpenCV build can open a window at all.

    ``opencv-python-headless`` ships without GUI code: ``cv2.imshow`` exists
    but raises "The function is not implemented". Asking the build info keeps
    this side-effect free - no half-open window while probing.
    """
    try:
        info = cv2.getBuildInformation()
    except cv2.error:                      # pragma: no cover - exotic builds
        return False
    for line in info.splitlines():
        text = line.strip()
        if text.startswith("GUI:"):
            value = text.split(":", 1)[1].strip().upper()
            return bool(value) and value != "NONE"
    return False                           # pragma: no cover - very old builds


def parse_source(source: str | int) -> str | int:
    """``"0"`` -> webcam 0, anything else is a path / URL."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


class Pipeline:
    def __init__(
        self,
        config: Config,
        source: str | int,
        *,
        display: bool = False,
        record: str | Path | None = None,
        web: int | None = None,
        web_host: str = "127.0.0.1",
        max_frames: int | None = None,
        alert_manager: AlertManager | None = None,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.source = parse_source(source)
        self.display = display
        self.record_path = Path(record) if record else None
        self.web_port = web
        self.web_host = web_host
        self.preview: PreviewServer | None = None
        self.max_frames = max_frames
        self.verbose = verbose

        self.detector = build_detector(config.detector)
        self.tracker = Tracker(config.tracking)
        self.alerts = alert_manager or AlertManager(config.alerts)

        shared: dict[str, Any] = {
            "zone_bg": ZoneBackgrounds(config.rule("item_removal")),
        }
        self.rules: list[Rule] = build_rules(config.rules, shared)
        self.enabled_rules = {r.name for r in self.rules}

        self.cap: cv2.VideoCapture | None = None
        self.writer: cv2.VideoWriter | None = None
        self.frame_index = 0
        self.fps = 0.0
        self.prev_small: np.ndarray | None = None
        self.flow_prev: np.ndarray | None = None
        self.recent: deque[tuple[str, str, float]] = deque(maxlen=5)
        self.start_wall = 0.0
        self._prev_boxes: dict[int, tuple[int, int, int, int]] = {}
        self._person_memory: list[tuple[float, tuple[int, int, int, int]]] = []

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source: {self.source!r}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps and fps > 0 and np.isfinite(fps) else 30.0
        self.live = isinstance(self.source, int)
        self.start_wall = time.monotonic()

    # ------------------------------------------------------------------ #
    @property
    def needs_flow(self) -> bool:
        return "rapid_motion" in self.enabled_rules

    @property
    def url(self) -> str | None:
        """URL of the browser preview, when ``web`` was requested."""
        return self.preview.url if self.preview is not None else None

    def _timestamp(self) -> float:
        if getattr(self, "live", True):
            return time.monotonic() - self.start_wall
        return self.frame_index / max(self.fps, 1e-6)

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        if self.cap is None:
            self.open()
        assert self.cap is not None

        window_ok = self.display and self._can_display()
        if self.display and not window_ok:
            print("[warn] no display available - running headless "
                  "(use --web for a browser preview)", file=sys.stderr)
        elif self.display and not gui_available():
            window_ok = False
            print(
                "[warn] this OpenCV build cannot open a window "
                "(opencv-python-headless is installed).\n"
                "       fix: .venv/bin/pip uninstall -y opencv-python-headless &&\n"
                "            .venv/bin/pip install opencv-python\n"
                "       or watch the feed in a browser instead: --web",
                file=sys.stderr,
            )

        if window_ok and sys.platform.startswith("linux"):
            # this OpenCV wheel ships the X11 Qt plugin only; saying so up
            # front avoids Qt printing a scary "no wayland plugin" warning
            if "QT_QPA_PLATFORM" not in os.environ and os.environ.get("DISPLAY"):
                os.environ["QT_QPA_PLATFORM"] = "xcb"

        if self.web_port is not None:
            try:
                self.preview = PreviewServer(self.web_port, self.web_host)
                self.preview.start()
                if self.verbose:
                    print(f"live preview: {self.url}", flush=True)
            except RuntimeError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                self.preview = None

        fps_ema = 0.0
        last_tick = time.monotonic()

        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = None

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break
                self.frame_index += 1

                now = time.monotonic()
                inst = 1.0 / max(now - last_tick, 1e-6)
                last_tick = now
                fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst

                events = self._step(frame)

                out = self._annotate(frame, fps_ema, events)
                if self.preview is not None:
                    self.preview.publish(out)
                if self.writer is not None:
                    self.writer.write(out)
                elif self.record_path is not None:
                    self._start_writer(out)
                    self.writer.write(out)

                if window_ok:
                    try:
                        cv2.imshow("shop theft detector", out)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    except cv2.error as exc:
                        window_ok = False
                        print(f"[warn] cannot open a window ({str(exc).splitlines()[0]}) - "
                              "use --web for a browser preview", file=sys.stderr)

                if self.max_frames and self.frame_index >= self.max_frames:
                    break
        finally:
            self._close()
        return len(self.alerts.emitted)

    # ------------------------------------------------------------------ #
    def _step(self, frame: np.ndarray) -> list[Event]:
        h, w = frame.shape[:2]
        resolve_zones(self.config.zones, w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, FLOW_WIDTH / w)
        small = (
            gray
            if scale >= 1.0
            else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )

        brightness = float(small.mean())
        sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
        global_diff = 0.0
        if self.prev_small is not None and self.prev_small.shape == small.shape:
            global_diff = float(np.mean(cv2.absdiff(small, self.prev_small)))

        flow_mag = None
        if self.needs_flow:
            if self.flow_prev is not None and self.flow_prev.shape == small.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    self.flow_prev, small, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            self.flow_prev = small.copy()

        detections = self.detector(frame)
        timestamp = self._timestamp()
        tracks = self.tracker.update(detections, timestamp)
        confirmed = self.tracker.confirmed()

        # a track that just died marks "that spot is empty now" in the motion
        # background model, which prevents ghost people from lingering
        live_ids = {t.track_id for t in tracks}
        for tid, box in self._prev_boxes.items():
            if tid not in live_ids and hasattr(self.detector, "absorb"):
                self.detector.absorb(box, frame)
        self._prev_boxes = {t.track_id: t.box for t in tracks}

        # person mask, with a short memory so detector blips do not expose a
        # standing customer as a "shelf change"
        for t in tracks:
            if t.misses == 0:
                self._person_memory.append((timestamp, t.box))
        self._person_memory = [
            (ts, box) for ts, box in self._person_memory
            if timestamp - ts <= PERSON_MEMORY
        ]
        person_mask = np.zeros((h, w), dtype=np.uint8)
        for _, box in self._person_memory:
            x1, y1, x2, y2 = box
            person_mask[y1:y2, x1:x2] = 1
        if self._person_memory:
            person_mask = cv2.dilate(
                person_mask, np.ones((PERSON_DILATE, PERSON_DILATE), np.uint8)
            )

        ctx = FrameContext(
            frame=frame,
            gray=gray,
            timestamp=timestamp,
            frame_index=self.frame_index,
            fps=self.fps,
            zones=self.config.zones,
            tracks=confirmed,
            detections=detections,
            person_mask=person_mask,
            flow_mag=flow_mag,
            flow_scale=scale,
            brightness=brightness,
            sharpness=sharpness,
            global_diff=global_diff,
        )

        events: list[Event] = []
        for rule in self.rules:
            try:
                events.extend(rule.update(ctx))
            except Exception as exc:  # keep the camera running
                if self.verbose:
                    print(f"[error] rule {rule.name} failed: {exc}", file=sys.stderr)

        for ev in events:
            if self.alerts.emit(ev, frame, confirmed):
                self.recent.append((event_line(ev), ev.severity))

        self.prev_small = small
        self._last_ctx = ctx
        return events

    # ------------------------------------------------------------------ #
    def _annotate(self, frame: np.ndarray, fps: float, events: list[Event]) -> np.ndarray:
        ctx = getattr(self, "_last_ctx", None)
        busy = {z.name for z in self.config.zones if ctx and ctx.person_near(z, margin=10)}
        draw_zones(frame, self.config.zones, busy)
        draw_tracks(frame, ctx.tracks if ctx else [])
        draw_hud(
            frame,
            fps=fps,
            people=len(ctx.tracks) if ctx else 0,
            backend=self.config.detector.backend,
            alert_count=len(self.alerts.emitted),
            recent=self.recent,
            zones_busy=busy,
        )
        return frame

    # ------------------------------------------------------------------ #
    def _start_writer(self, sample: np.ndarray) -> None:
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.record_path), fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open output video {self.record_path}")

    # ------------------------------------------------------------------ #
    def _can_display(self) -> bool:
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            return False
        return True

    def _close(self) -> None:
        if self.preview is not None:
            self.preview.stop()
            self.preview = None
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.alerts.close()
