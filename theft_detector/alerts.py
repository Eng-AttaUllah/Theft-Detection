"""Alert events: console output, JSONL event log and snapshot images."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import AlertConfig
from .tracker import Track

SEVERITY_COLORS = {
    "high": "\033[1;31m",
    "medium": "\033[1;33m",
    "low": "\033[1;36m",
}
_RESET = "\033[0m"


@dataclass
class Event:
    rule: str
    severity: str
    message: str
    timestamp: float
    zone: str | None = None
    track_ids: list[int] = field(default_factory=list)
    confidence: float = 1.0

    def as_dict(self, snapshot: str | None = None) -> dict[str, Any]:
        return {
            "ts": round(self.timestamp, 3),
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rule": self.rule,
            "severity": self.severity,
            "zone": self.zone,
            "tracks": self.track_ids,
            "confidence": round(self.confidence, 3),
            "message": self.message,
            "snapshot": snapshot,
        }


class AlertManager:
    """Rates-limits events, logs them and stores evidence images."""

    def __init__(self, cfg: AlertConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.directory)
        self.snapshots = self.root / "snapshots"
        self.events_path = self.root / "events.jsonl"
        if cfg.save_snapshots:
            self.snapshots.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

        self._last_emit: dict[str, float] = {}
        self._fh = self.events_path.open("a", encoding="utf-8")
        self.emitted: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self._color = sys.stdout.isatty()

    # ------------------------------------------------------------------ #
    def _key(self, ev: Event) -> str:
        return f"{ev.rule}:{ev.zone or '-'}"

    def _cooldown_ok(self, key: str, now: float) -> bool:
        last = self._last_emit.get(key)
        return last is None or (now - last) >= self.cfg.cooldown_seconds

    # ------------------------------------------------------------------ #
    def emit(self, ev: Event, frame: np.ndarray | None = None,
             tracks: Iterable[Track] | None = None) -> bool:
        key = self._key(ev)
        if not self._cooldown_ok(key, ev.timestamp):
            return False

        self._last_emit[key] = ev.timestamp
        snapshot: str | None = None
        if frame is not None and self.cfg.save_snapshots:
            snapshot = self._save_snapshot(ev, frame, tracks or [])

        record = ev.as_dict(snapshot)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self.emitted.append(record)
        self.counts[ev.rule] = self.counts.get(ev.rule, 0) + 1

        if not self.cfg.quiet:
            self._print(ev, snapshot)
        if self.cfg.beep:
            print("\a", end="", flush=True)
        return True

    # ------------------------------------------------------------------ #
    def _print(self, ev: Event, snapshot: str | None) -> None:
        when = time.strftime("%H:%M:%S")
        if self._color:
            tone = SEVERITY_COLORS.get(ev.severity, "")
            head = f"{tone}[{ev.severity.upper()}]{_RESET}"
            body = f"{tone}{ev.message}{_RESET}"
        else:
            head = f"[{ev.severity.upper()}]"
            body = ev.message
        extra = f"  (snapshot: {snapshot})" if snapshot else ""
        print(f"{when} {head} {ev.rule}: {body}{extra}", flush=True)

    # ------------------------------------------------------------------ #
    def _save_snapshot(
        self, ev: Event, frame: np.ndarray, tracks: Iterable[Track]
    ) -> str:
        img = frame.copy()
        involved = set(ev.track_ids)
        for tr in tracks:
            if tr.track_id in involved:
                x1, y1, x2, y2 = tr.box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img, f"#{tr.track_id}", (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )
        label = f"{ev.rule}: {ev.message}"[:110]
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (0, 255, 255) if ev.severity != "high" else (0, 0, 255), 2, cv2.LINE_AA,
        )
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{ev.rule}_{ev.zone or 'global'}_{len(self.emitted):04d}.jpg"
        path = self.snapshots / name
        cv2.imwrite(str(path), img)
        return str(path)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        if not self.counts:
            return "no alerts raised"
        parts = [f"{rule}={n}" for rule, n in sorted(self.counts.items())]
        return f"{len(self.emitted)} alert(s): " + ", ".join(parts)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover - defensive
            pass
