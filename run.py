#!/usr/bin/env python3
"""Shop theft detector - command line entry point.

Examples
--------
Webcam::

    python run.py --source 0 --display

Recorded footage / RTSP camera::

    python run.py --source shop_cam.mp4 --record out.mp4

Bundled synthetic demo (downloads nothing, builds the clip on first run)::

    python run.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from theft_detector import __version__
from theft_detector.alerts import AlertManager
from theft_detector.config import DEFAULT_RULES, Config, PROJECT_ROOT
from theft_detector.pipeline import Pipeline

DEMO_CONFIG = PROJECT_ROOT / "configs" / "demo.json"
DEMO_VIDEO = PROJECT_ROOT / "samples" / "demo.mp4"


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Detect shoplifting behaviour from a camera or video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-s", "--source", default="0",
                   help="webcam index (0), video file or stream URL")
    p.add_argument("-c", "--config", default=None,
                   help="config JSON (default: config.json next to this script)")
    p.add_argument("--detector", choices=["hog", "motion", "onnx", "caffe"],
                   help="override the person detector backend")
    p.add_argument("--tracker", choices=["iou", "bytetrack", "botsort"],
                   help="override the multi-object tracker "
                        "(bytetrack default, botsort adds appearance re-ID)")
    p.add_argument("-d", "--display", action="store_true",
                   help="show an annotated window (needs a desktop session)")
    p.add_argument("--web", metavar="PORT", nargs="?", type=int, const=8080, default=None,
                   help="serve the live annotated feed in a browser "
                        "(http://127.0.0.1:PORT/, default 8080) - works without a GUI")
    p.add_argument("--record", metavar="PATH",
                   help="write the annotated video to PATH (.mp4)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after N frames")
    p.add_argument("--alert-dir", metavar="DIR", help="where to write alerts")
    p.add_argument("--no-snapshots", action="store_true",
                   help="do not save alert snapshot images")
    p.add_argument("--cooldown", type=float, metavar="SEC",
                   help="minimum seconds between two alerts of the same rule")
    p.add_argument("--demo", action="store_true",
                   help="run on generated demo footage with configs/demo.json")
    p.add_argument("--list-rules", action="store_true",
                   help="show the behaviour rules and exit")
    p.add_argument("--print-config", action="store_true",
                   help="print the effective configuration and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="no console alerts")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


# --------------------------------------------------------------------------- #
def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.detector:
        cfg.detector.backend = args.detector
    if getattr(args, "tracker", None):
        cfg.tracking.algorithm = args.tracker
    if args.alert_dir:
        cfg.alerts.directory = args.alert_dir
    if args.no_snapshots:
        cfg.alerts.save_snapshots = False
    if args.cooldown is not None:
        cfg.alerts.cooldown_seconds = args.cooldown
    if args.quiet:
        cfg.alerts.quiet = True
    return cfg


def print_rules(cfg: Config) -> None:
    print("behaviour rules (enabled: +, disabled: -):")
    for name, params in sorted({**DEFAULT_RULES, **cfg.rules}.items()):
        flag = "+" if params.get("enabled", True) else "-"
        rest = {k: v for k, v in params.items() if k != "enabled"}
        print(f"  {flag} {name:<20} {json.dumps(rest)}")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.demo and args.config is None:
        args.config = str(DEMO_CONFIG)

    try:
        cfg = Config.load(args.config)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    cfg = apply_overrides(cfg, args)

    if args.list_rules:
        print_rules(cfg)
        return 0
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    source = args.source
    if args.demo:
        from scripts import make_demo_video

        make_demo_video.ensure(DEMO_CONFIG, DEMO_VIDEO)
        source = str(DEMO_VIDEO)
        if args.detector is None:
            cfg.detector.backend = "motion"

    if not cfg.zones:
        print("[warn] no zones configured - restricted-zone / shelf rules are inert",
              file=sys.stderr)

    alerts = AlertManager(cfg.alerts)
    print(f"theft detector {__version__} | source={source} | "
          f"detector={cfg.detector.backend} | tracker={cfg.tracking.algorithm} | "
          f"rules={','.join(sorted(cfg.enabled_rules()))}")

    pipe = Pipeline(
        cfg,
        source,
        display=args.display,
        record=args.record,
        web=args.web,
        max_frames=args.max_frames,
        alert_manager=alerts,
    )
    try:
        pipe.run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped")

    print(f"processed {pipe.frame_index} frames | {alerts.summary()}")
    if alerts.emitted:
        print(f"event log: {alerts.events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
