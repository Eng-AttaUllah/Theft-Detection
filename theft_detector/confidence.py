"""Theft confidence: fuse every behaviour hit into one calibrated score.

The rule stage answers "did something suspicious happen *right now*?". This
stage answers the question a human actually asks - "how sure are we that this
is a theft?" - by pooling evidence over time into one number per camera:

    score(t) = min(1, score(t-1) + weight_rule * confidence * severity)
    score    decays with a time constant of ``decay_seconds``

so a shelf change on its own is a maybe (0.45), a shelf change plus somebody
loitering plus fast movement is a yes, and if nothing else happens the number
drifts back to zero instead of crying wolf forever.

Crossing ``report_threshold`` raises a ``theft_confidence`` alert that carries
the score, the evidence list and the zone the evidence came from; the alert
re-arms only after the score has fallen below ``threshold * reset_ratio``
(hysteresis, so one incident = one alert).

The score is deliberately global rather than per-zone: a shop camera sees one
incident, and "he stood at the counter, then the shelf changed, then he ran"
must add up instead of being filed in three separate columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .alerts import Event
from .config import ConfidenceConfig

# how much each behaviour contributes (overridable via confidence.weights)
DEFAULT_WEIGHTS: dict[str, float] = {
    "item_removal": 0.45,        # a shelf actually changed while he stood there
    "object_taken": 0.55,        # an object id left its shelf with a person
    "rapid_motion": 0.25,        # snatching-speed movement at a shelf
    "loitering": 0.20,           # waiting for an opportunity
    "crowd": 0.20,               # distraction theft pattern
    "restricted_zone": 0.15,     # behind the counter / stock room
    "unattended_object": 0.10,   # something left/moved with nobody around
    "camera_tamper": 0.12,       # somebody does not want to be recorded
}

SEVERITY_FACTOR = {"high": 1.0, "medium": 0.9, "low": 0.7}

CLEAR_AT = 0.01


@dataclass
class Assessment:
    """Result of one fusion step."""

    score: float
    reported: bool = False
    event: Event | None = None
    evidence: list[str] = field(default_factory=list)
    zone: str | None = None

    @property
    def level(self) -> str:
        if self.score >= 0.60:
            return "high"
        if self.score >= 0.35:
            return "elevated"
        return "clear" if self.score <= CLEAR_AT else "low"


class TheftConfidence:
    """One decaying suspicion score per camera, plus the flag that goes with it."""

    def __init__(self, cfg: ConfidenceConfig | None = None) -> None:
        self.cfg = cfg or ConfidenceConfig()
        self.weights = dict(DEFAULT_WEIGHTS)
        for key, value in (self.cfg.weights or {}).items():
            self.weights[str(key)] = float(value)
        self.score = 0.0
        self.evidence: list[str] = []
        self.zone: str | None = None
        self._last_ts: float | None = None
        self._armed = True

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def level(self) -> str:
        return Assessment(score=self.score, zone=self.zone).level

    # ------------------------------------------------------------------ #
    def update(self, events: list[Event], timestamp: float) -> Assessment:
        """Fold this frame's rule events in; return the assessment (maybe an alert)."""
        if not self.cfg.enabled:
            return Assessment(score=0.0)

        self._decay(timestamp)

        for ev in events:
            if ev.rule == "theft_confidence":
                continue                           # never eat our own alert
            weight = self.weights.get(ev.rule)
            if not weight:
                continue
            factor = SEVERITY_FACTOR.get(ev.severity, 1.0)
            self.score = min(
                1.0, self.score + weight * max(0.0, ev.confidence) * factor
            )
            if ev.rule not in self.evidence:
                self.evidence.append(ev.rule)
                del self.evidence[:-4]             # remember the last few reasons
            if ev.zone:
                self.zone = ev.zone

        return self._assess(timestamp)

    # ------------------------------------------------------------------ #
    def _decay(self, timestamp: float) -> None:
        last = self._last_ts
        self._last_ts = timestamp
        if last is None or timestamp <= last:
            return
        dt = timestamp - last
        tau = max(1.0, float(self.cfg.decay_seconds))
        value = self.score * math.exp(-dt / tau)
        self.score = value if value > CLEAR_AT else 0.0
        if self.score <= CLEAR_AT:
            self.evidence.clear()                # fully cleared: forget the
            self.zone = None                     # incident, incl. its zone

    def _assess(self, timestamp: float) -> Assessment:
        threshold = float(self.cfg.report_threshold)
        evidence = list(self.evidence)
        zone = self.zone

        if self.score >= threshold and self._armed:
            self._armed = False
            event = Event(
                rule="theft_confidence",
                severity="high",
                zone=zone,
                timestamp=timestamp,
                confidence=self.score,
                message=f"theft confidence {self.score:.2f} - {self._reason(evidence)}",
            )
            return Assessment(
                score=self.score, reported=True, event=event, evidence=evidence, zone=zone
            )

        if self.score < threshold * float(self.cfg.reset_ratio):
            self._armed = True
        return Assessment(score=self.score, evidence=evidence, zone=zone)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reason(evidence: list[str]) -> str:
        if not evidence:
            return "no signals"
        if len(evidence) == 1:
            return f"evidence: {evidence[0]}"
        return "evidence: " + ", ".join(evidence[:-1]) + f" + {evidence[-1]}"
