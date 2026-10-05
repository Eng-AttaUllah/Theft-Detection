"""Configuration objects and JSON loading for the theft detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .zones import Zone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "restricted_zone": {
        "enabled": True,
        "zone_kinds": ["restricted", "register"],
        "min_dwell_seconds": 1.0,
    },
    "loitering": {
        "enabled": True,
        "max_seconds": 10.0,
        "max_radius": 60.0,
        "gap_seconds": 4.0,
    },
    "item_removal": {
        "enabled": True,
        "zone_kinds": ["shelf"],
        # background model / change detection
        "diff_threshold": 28,
        "change_ratio": 0.010,
        "largest_change_ratio": 0.015,
        "bg_alpha": 0.06,
        "freeze_ratio": 0.03,
        # feet-to-zone distance (px) that counts as "at the shelf" - raise it
        # when the camera looks down and the floor sits far below the shelf face
        "person_margin": 150,
        # triggering
        "hold_frames": 10,
        "person_grace_seconds": 8.0,
        "settle_seconds": 20.0,
    },
    "object_taken": {
        # needs the object-ID stage (onnx/caffe detectors): a thing that sat
        # on a shelf is picked up and leaves its spot / disappears while held
        "enabled": True,
        "zone_kinds": ["shelf", "register"],
        "min_carry_seconds": 2.0,
        "leave_margin": 60,
        "carrier_margin": 80,
        "move_ratio": 0.35,
    },
    "unattended_object": {
        "enabled": True,
        "zone_kinds": ["shelf", "register"],
        "hold_frames": 30,
        "unattended_seconds": 10.0,
    },
    "rapid_motion": {
        "enabled": True,
        "threshold": 9.0,
        "min_frames": 3,
        "person_margin": 150,
        "zone_kinds": ["shelf", "register", "restricted"],
    },
    "crowd": {
        "enabled": True,
        "min_persons": 3,
        "seconds": 2.0,
        "drop_seconds": 1.5,
    },
    "camera_tamper": {
        "enabled": True,
        "min_brightness": 22.0,
        "max_brightness": 238.0,
        "min_sharpness": 6.0,
        "static_seconds": 8.0,
        "min_frames": 8,
        "min_diff": 0.5,
    },
}


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    backend: str = "hog"              # hog | motion | onnx | caffe
    model: str = ""                   # onnx path or MobileNet-SSD prototxt
    weights: str = ""                 # MobileNet-SSD caffemodel path
    input_size: int = 640             # onnx input size
    conf_threshold: float = 0.40
    nms_threshold: float = 0.45
    hit_threshold: float = 0.0        # hog
    scale: float = 1.05               # hog pyramid scale
    win_stride: list[int] = field(default_factory=lambda: [8, 8])
    padding: list[int] = field(default_factory=lambda: [16, 16])
    max_inference_width: int = 640    # downscale before detection for speed
    # motion backend (adaptive background model)
    motion_threshold: float = 26.0     # gray levels above background = foreground
    motion_bg_alpha: float = 0.08      # where nothing stands (fast adaptation)
    motion_fg_alpha: float = 0.003     # on foreground: keeps a standing person
                                       # visible for ~15 s before absorption
    motion_reset_ratio: float = 0.80   # global scene change -> rebuild model
    min_blob_area_ratio: float = 0.004
    max_blob_area_ratio: float = 0.30
    min_blob_height_ratio: float = 0.08
    min_aspect_ratio: float = 0.20     # width / height
    max_aspect_ratio: float = 1.60
    min_fill_ratio: float = 0.30
    warmup_frames: int = 3
    # YOLO object detection: which non-person classes are tracked as "things
    # people can take". Empty list = persons only.
    class_names: str = ""                 # optional file, one label per line
    object_classes: list[str] = field(
        default_factory=lambda: [
            "bottle", "cup", "handbag", "backpack", "book",
            "cell phone", "scissors", "teddy bear", "suitcase",
        ]
    )
    min_object_confidence: float = 0.35   # lower than conf_threshold: a bottle is
                                          # much smaller than a person to detect


@dataclass
class TrackingConfig:
    algorithm: str = "bytetrack"       # iou | bytetrack | botsort
    iou_threshold: float = 0.30        # legacy "iou"/SORT tracker only
    max_misses: int = 20               # frames a lost track is kept for
    min_hits: int = 3                  # matches before a track is reported
    min_iou_new_track: float = 0.0     # reserved
    # ByteTrack / BoT-SORT
    track_thresh: float = 0.50         # >= high score, < low score (still used!)
    reid_threshold: float = 0.55       # BoT-SORT appearance similarity 0..1
    reid_max_distance: float = 180.0   # px - how far a ReID match may jump


@dataclass
class AlertConfig:
    directory: str = "alerts"
    cooldown_seconds: float = 8.0
    save_snapshots: bool = True
    beep: bool = True
    quiet: bool = False


@dataclass
class ConfidenceConfig:
    """The "theft confidence" stage: fuse rule hits into one score per zone."""

    enabled: bool = True
    report_threshold: float = 0.60      # crossing this raises an alert
    warn_threshold: float = 0.35        # HUD turns amber from here
    decay_seconds: float = 45.0         # how fast suspicion fades away
    reset_ratio: float = 0.5            # re-arm after falling below threshold*ratio
    weights: dict = field(default_factory=dict)   # rule -> weight (overrides)


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    zones: list[Zone] = field(default_factory=list)
    rules: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load a config file; missing/absent file falls back to defaults."""
        cfg = cls()
        cfg.rules = {name: dict(params) for name, params in DEFAULT_RULES.items()}

        if path is None:
            path = DEFAULT_CONFIG_PATH
        path = Path(path)
        if not path.exists():
            if path != DEFAULT_CONFIG_PATH:
                raise FileNotFoundError(f"config file not found: {path}")
            return cfg

        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        _apply_section(cfg.detector, data.get("detector", {}), "detector")
        _apply_section(cfg.tracking, data.get("tracking", {}), "tracking")
        _apply_section(cfg.alerts, data.get("alerts", {}), "alerts")
        _apply_section(cfg.confidence, data.get("confidence", {}), "confidence")

        cfg.zones = [Zone.from_dict(z) for z in data.get("zones", [])]

        for name, params in data.get("rules", {}).items():
            merged = dict(DEFAULT_RULES.get(name, {}))
            merged.update(params)
            cfg.rules[name] = merged
        return cfg

    # ------------------------------------------------------------------ #
    def rule(self, name: str) -> dict[str, Any]:
        return self.rules.get(name, DEFAULT_RULES.get(name, {"enabled": False}))

    def enabled_rules(self) -> list[str]:
        return [n for n, p in self.rules.items() if p.get("enabled", True)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": asdict(self.detector),
            "tracking": asdict(self.tracking),
            "alerts": asdict(self.alerts),
            "confidence": asdict(self.confidence),
            "zones": [
                {"name": z.name, "kind": z.kind, "severity": z.severity, "points": z.points}
                for z in self.zones
            ],
            "rules": self.rules,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
def _apply_section(obj: Any, data: dict[str, Any], section: str) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown option {key!r} in config section {section!r}")
        current = getattr(obj, key)
        if isinstance(current, list) and isinstance(value, (list, tuple)):
            value = list(value)
        setattr(obj, key, value)
