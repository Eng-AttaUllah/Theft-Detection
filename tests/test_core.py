"""Unit / integration tests.

Run with::

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from theft_detector.alerts import Event                      # noqa: E402
from theft_detector.behaviors import ZoneBackgrounds         # noqa: E402
from theft_detector.config import Config, TrackingConfig     # noqa: E402
from theft_detector.detectors import Detection, decode_yolo  # noqa: E402
from theft_detector.tracker import Tracker                   # noqa: E402
from theft_detector.zones import Zone                        # noqa: E402


# --------------------------------------------------------------------------- #
class ZoneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = Zone(
            name="shelf",
            points=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
            kind="shelf",
        )
        self.zone.resolve(200, 100)

    def test_contains(self) -> None:
        self.assertTrue(self.zone.contains(100, 50))
        self.assertFalse(self.zone.contains(10, 10))

    def test_bbox(self) -> None:
        # bottom/right are exclusive: pixel column 150 is inside the zone
        self.assertEqual(self.zone.bbox(), (50, 25, 151, 76))

    def test_mask_area(self) -> None:
        m = self.zone.mask((100, 200))
        self.assertEqual(int(m.sum()), 101 * 51)

    def test_relative_zones_work_at_any_resolution(self) -> None:
        self.zone.resolve(1280, 720)
        self.assertEqual(self.zone.bbox(), (320, 180, 961, 541))


# --------------------------------------------------------------------------- #
class ConfigTest(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = Config.load(PROJECT_ROOT / "config.json")
        self.assertEqual(len(cfg.zones), 5)
        self.assertIn("item_removal", cfg.rules)
        self.assertTrue(cfg.rule("item_removal")["enabled"])
        self.assertIn("item_removal", cfg.enabled_rules())

    def test_demo_config(self) -> None:
        cfg = Config.load(PROJECT_ROOT / "configs" / "demo.json")
        self.assertEqual(cfg.detector.backend, "motion")
        self.assertGreater(cfg.rule("loitering")["max_seconds"], 0)

    def test_unknown_option_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text('{"detector": {"nope": 1}}', encoding="utf-8")
            with self.assertRaises(KeyError):
                Config.load(p)


# --------------------------------------------------------------------------- #
class TrackerTest(unittest.TestCase):
    def test_id_stability(self) -> None:
        tr = Tracker(TrackingConfig(min_hits=2))
        tr.update([Detection(box=(10, 10, 60, 160), confidence=0.9)], 0.0)
        tr.update([Detection(box=(14, 10, 64, 160), confidence=0.9)], 0.1)
        self.assertEqual(len(tr.tracks), 1)
        self.assertEqual(tr.tracks[0].track_id, 1)
        self.assertEqual(len(tr.confirmed()), 1)

    def test_new_track_and_decay(self) -> None:
        tr = Tracker(TrackingConfig(min_hits=1, max_misses=2))
        tr.update([Detection(box=(0, 0, 50, 150), confidence=0.9)], 0.0)
        tr.update([Detection(box=(500, 0, 550, 150), confidence=0.9)], 0.1)
        self.assertEqual(len(tr.tracks), 2)
        tr.update([], 0.2)
        self.assertEqual(len(tr.tracks), 2)      # 2nd miss / 1st miss
        tr.update([], 0.3)
        self.assertEqual(len(tr.tracks), 1)      # track 1 passed max_misses
        tr.update([], 0.4)
        self.assertEqual(len(tr.tracks), 0)

    def test_history_span(self) -> None:
        tr = Tracker(TrackingConfig(min_hits=1))
        tr.update([Detection(box=(0, 0, 50, 150), confidence=0.9)], 0.0)
        tr.update([Detection(box=(10, 0, 60, 150), confidence=0.9)], 5.0)
        self.assertAlmostEqual(tr.tracks[0].age, 5.0)


# --------------------------------------------------------------------------- #
class YoloDecodeTest(unittest.TestCase):
    def test_yolov8_layout(self) -> None:
        pred = np.zeros((1, 84, 100), np.float32)
        pred[0, 0:4, 42] = [50, 60, 20, 40]
        pred[0, 4 + 3, 42] = 0.91
        boxes, scores, ids = decode_yolo(pred, 0.25)
        self.assertEqual(boxes.shape, (1, 4))
        self.assertAlmostEqual(float(scores[0]), 0.91, places=5)
        self.assertEqual(int(ids[0]), 3)
        np.testing.assert_allclose(boxes[0], [50, 60, 20, 40])

    def test_yolov5_layout(self) -> None:
        pred = np.zeros((1, 300, 85), np.float32)
        pred[0, 2, 0:4] = [10, 20, 30, 40]
        pred[0, 2, 4] = 0.8           # objectness
        pred[0, 2, 5 + 6] = 0.5       # class score
        boxes, scores, ids = decode_yolo(pred, 0.25)
        self.assertEqual(boxes.shape, (1, 4))
        self.assertAlmostEqual(float(scores[0]), 0.4, places=5)
        self.assertEqual(int(ids[0]), 6)

    def test_below_threshold(self) -> None:
        pred = np.full((1, 84, 10), 0.01, np.float32)
        boxes, scores, _ = decode_yolo(pred, 0.25)
        self.assertEqual(boxes.shape[0], 0)


# --------------------------------------------------------------------------- #
class ZoneBackgroundTest(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = Zone(
            name="shelf",
            points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            kind="shelf",
        )
        self.zone.resolve(200, 200)
        self.bg = ZoneBackgrounds({})

    def _run(self, frames: int, gray, mask, tracks, start=0):
        state = None
        for i in range(start, start + frames):
            states = self.bg.update([self.zone], gray, mask, tracks,
                                    timestamp=i * 0.1, frame_index=i)
            state = states["shelf"]
        return state

    def test_builds_background(self) -> None:
        gray = np.full((200, 200), 100, np.uint8)
        mask = np.zeros((200, 200), np.uint8)
        state = self._run(5, gray, mask, [])
        self.assertIsNotNone(state.bg)
        self.assertEqual(state.changed_run, 0)

    def test_notices_removed_item(self) -> None:
        base = np.full((200, 200), 100, np.uint8)
        mask = np.zeros((200, 200), np.uint8)
        self._run(5, base, mask, [], start=0)

        # an item (dark block) vanishes from the shelf
        changed = base.copy()
        changed[70:110, 130:170] = 210

        person_mask = np.zeros((200, 200), np.uint8)
        person_mask[50:180, 40:90] = 1
        tracks = [type("T", (), {"box": (40, 50, 90, 180)})()]

        state = self._run(8, changed, person_mask, tracks, start=5)
        self.assertGreater(state.changed_run, 0)
        self.assertGreater(state.largest_ratio, 0.015)
        self.assertIsNotNone(state.last_person)
        self.assertTrue(state.ever_person)

    def test_person_pixels_do_not_trigger_change(self) -> None:
        base = np.full((200, 200), 100, np.uint8)
        mask = np.zeros((200, 200), np.uint8)
        self._run(5, base, mask, [], start=0)

        person_mask = np.zeros((200, 200), np.uint8)
        person_mask[50:180, 40:90] = 1
        frame = base.copy()
        frame[50:180, 40:90] = 30               # only the person changed
        tracks = [type("T", (), {"box": (40, 50, 90, 180)})()]

        state = self._run(8, frame, person_mask, tracks, start=5)
        self.assertEqual(state.changed_run, 0)

    def test_lost_track_footprint_is_not_a_change(self) -> None:
        """A motion detector drops people when they stand still - as long as
        their last box still looks like them they must not count as a change,
        but anything they left behind must be reported.
        """
        from theft_detector.tracker import Track

        self.bg = ZoneBackgrounds({"shadow_margin": 8, "person_fill_ratio": 0.5,
                                   "lost_person_seconds": 60.0})
        base = np.full((200, 200), 100, np.uint8)
        empty = np.zeros((200, 200), np.uint8)
        self._run(5, base, empty, [], start=0)          # learn the empty shelf

        box = (70, 70, 110, 150)
        standing = base.copy()
        standing[70:150, 70:110] = 40                   # "somebody" is standing here
        person_mask = empty.copy()
        person_mask[70:150, 70:110] = 1
        track = Track(track_id=3, box=box, hits=5, misses=0,
                      start_time=0.0, last_time=0.0)

        state = self._run(6, standing, person_mask, [track], start=5)
        self.assertEqual(state.changed_run, 0)          # masked while detected

        # ... the detector loses him, his body stays exactly where it was
        state = self._run(10, standing, empty, [], start=11)
        self.assertEqual(state.changed_run, 0)
        self.assertTrue(state.observed)
        self.assertIsNotNone(state.person_hint)         # we still believe he is there

        # ... he steps away and the item he took is all that is left behind:
        # it is far too small to be him, so it must be reported.
        stolen = base.copy()
        stolen[80:100, 75:95] = 210                     # 400 px inside a 3200 px box
        state = self._run(10, stolen, empty, [], start=30)
        self.assertGreater(state.changed_run, 0)
        self.assertGreater(state.largest_ratio, 0.015)


# --------------------------------------------------------------------------- #
class EventTest(unittest.TestCase):
    def test_serialisable(self) -> None:
        ev = Event(rule="loitering", severity="medium", message="x", timestamp=1.5)
        d = ev.as_dict("shot.jpg")
        self.assertEqual(d["rule"], "loitering")
        self.assertEqual(d["snapshot"], "shot.jpg")


# --------------------------------------------------------------------------- #
class DetectorFactoryTest(unittest.TestCase):
    def test_unknown_backend(self) -> None:
        from theft_detector.config import DetectorConfig
        from theft_detector.detectors import build_detector

        with self.assertRaises(ValueError):
            build_detector(DetectorConfig(backend="nope"))

    def test_hog_or_documented_fallback(self) -> None:
        from theft_detector.config import DetectorConfig
        from theft_detector.detectors import (
            HogPersonDetector, MotionDetector, build_detector, hog_available,
        )

        cfg = DetectorConfig(backend="hog")
        det = build_detector(cfg)
        if hog_available():
            self.assertIsInstance(det, HogPersonDetector)
        else:  # OpenCV 5.x - documented fallback
            self.assertIsInstance(det, MotionDetector)
            self.assertEqual(cfg.backend, "motion")

    def test_motion_backend_finds_a_moving_person_sized_blob(self) -> None:
        import cv2

        from theft_detector.config import DetectorConfig
        from theft_detector.detectors import MotionDetector

        det = MotionDetector(DetectorConfig(backend="motion", conf_threshold=0.3))
        found = []
        for i in range(15):
            frame = np.full((240, 320, 3), 180, np.uint8)
            x = 20 + i * 5
            cv2.rectangle(frame, (x, 60), (x + 60, 200), (30, 30, 200), -1)
            found = det(frame)
        self.assertGreaterEqual(len(found), 1)
        self.assertGreater(found[0].area, 1000)
        self.assertGreater(found[0].height, 100)


# --------------------------------------------------------------------------- #
class RuleTest(unittest.TestCase):
    @staticmethod
    def _ctx(tracks, ts: float, index: int):
        from theft_detector.behaviors import FrameContext

        gray = np.zeros((240, 320), np.uint8)
        return FrameContext(
            frame=np.zeros((240, 320, 3), np.uint8),
            gray=gray,
            timestamp=ts,
            frame_index=index,
            fps=10.0,
            zones=[],
            tracks=tracks,
            person_mask=np.zeros((240, 320), np.uint8),
        )

    @staticmethod
    def _track(tid: int, box):
        from theft_detector.tracker import Track

        return Track(track_id=tid, box=box, hits=5, misses=0,
                     start_time=0.0, last_time=0.0)

    def test_crowd_needs_several_people(self) -> None:
        from theft_detector.behaviors import build_rules

        rules = build_rules({"crowd": {"enabled": True, "min_persons": 3,
                                       "seconds": 1.0, "drop_seconds": 1.5}}, {})
        rule = next(r for r in rules if r.name == "crowd" if False) if False else rules[0]
        two = [self._track(1, (10, 10, 60, 200)), self._track(2, (100, 10, 150, 200))]
        three = two + [self._track(3, (200, 10, 250, 200))]

        events = []
        for i in range(30):
            ts = i / 10.0
            events += rule.update(self._ctx(three if i < 5 else two, ts, i))
        self.assertEqual(events, [])          # too few people, nothing raised

        for i in range(30):
            ts = 10 + i / 10.0
            events += rule.update(self._ctx(three, ts, 100 + i))
        self.assertTrue(any(e.rule == "crowd" for e in events))

    def test_loitering_fires_after_max_seconds(self) -> None:
        from theft_detector.behaviors import build_rules

        rules = build_rules({"loitering": {"enabled": True, "max_seconds": 5.0,
                                           "max_radius": 60.0, "gap_seconds": 4.0}}, {})
        rule = next(r for r in rules if r.name == "crowd" if False) if False else rules[0]
        events = []
        for i in range(120):                 # 12 s at 10 fps
            ts = i / 10.0
            box = (100 + (i % 3), 60, 160, 200)   # barely moving = loitering
            events += rule.update(self._ctx([self._track(1, box)], ts, i))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].rule, "loitering")

    def test_loitering_stays_quiet_for_a_moving_person(self) -> None:
        from theft_detector.behaviors import build_rules

        rules = build_rules({"loitering": {"enabled": True, "max_seconds": 5.0,
                                           "max_radius": 60.0, "gap_seconds": 4.0}}, {})
        rule = next(r for r in rules if r.name == "crowd" if False) if False else rules[0]
        events = []
        for i in range(120):
            ts = i / 10.0
            box = (i * 4, 60, i * 4 + 60, 200)   # walks steadily across frame
            events += rule.update(self._ctx([self._track(1, box)], ts, i))
        self.assertEqual(events, [])

    def test_item_removal_reports_one_physical_change_once(self) -> None:
        """A product that goes missing while a customer hides it must raise one
        alarm - and not another every time he steps aside.
        """
        from theft_detector.behaviors import FrameContext, build_rules

        rules = build_rules({"item_removal": {"enabled": True, "zone_kinds": ["shelf"],
                                              "hold_frames": 5, "clear_frames": 5,
                                              "repeat_gap_seconds": 30.0,
                                              "duplicate_overlap": 0.4}}, {})
        rule = next(r for r in rules if r.name == "item_removal")

        zone = Zone(name="shelf", kind="shelf",
                    points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])
        zone.resolve(320, 240)
        body = (180, 80, 280, 160)              # where the customer stands

        def scene(item_present: bool) -> np.ndarray:
            gray = np.full((240, 320), 100, np.uint8)
            if item_present:
                gray[100:140, 200:260] = 60     # the product on the shelf
            return gray

        def ctx(gray, mask, tracks, ts, i) -> FrameContext:
            return FrameContext(frame=np.zeros((240, 320, 3), np.uint8),
                                gray=gray, timestamp=ts, frame_index=i, fps=10.0,
                                zones=[zone], tracks=tracks, person_mask=mask)

        def at_shelf(present: bool):
            mask = np.zeros((240, 320), np.uint8)
            tracks = []
            if present:
                mask[80:160, 180:280] = 1
                tracks = [self._track(1, body)]
            return mask, tracks

        away_mask, _ = at_shelf(False)
        events = []

        for i in range(0, 10):                  # learn the shelf with the product
            events += rule.update(ctx(scene(True), away_mask, [], i / 10.0, i))
        for i in range(10, 20):                 # he blocks it while taking it
            mask, tracks = at_shelf(True)
            events += rule.update(ctx(scene(False), mask, tracks, i / 10.0, i))
        self.assertEqual(events, [])

        for i in range(20, 25):                 # he steps aside - report it
            events += rule.update(ctx(scene(False), away_mask, [], i / 10.0, i))
        self.assertEqual([e.rule for e in events], ["item_removal"])

        for i in range(25, 45):                 # he is back in front of it
            mask, tracks = at_shelf(True)
            events += rule.update(ctx(scene(False), mask, tracks, i / 10.0, i))
        self.assertEqual(len(events), 1)

        for i in range(45, 52):                 # same product, second look
            events += rule.update(ctx(scene(False), away_mask, [], i / 10.0, i))
        self.assertEqual(len(events), 1, "the same missing product alerted twice")
        st = rule.bg.states["shelf"]
        self.assertGreaterEqual(st.changed_run, 5)   # a second episode did open
        self.assertIsNotNone(st.consumed_by)

    def test_item_removal_allows_a_second_different_change(self) -> None:
        """Deduplication must not swallow another product going missing."""
        from theft_detector.behaviors import FrameContext, build_rules

        rules = build_rules({"item_removal": {"enabled": True, "zone_kinds": ["shelf"],
                                              "hold_frames": 5, "clear_frames": 5,
                                              "repeat_gap_seconds": 30.0,
                                              "duplicate_overlap": 0.4}}, {})
        rule = next(r for r in rules if r.name == "item_removal")

        zone = Zone(name="shelf", kind="shelf",
                    points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])
        zone.resolve(320, 240)
        # a customer stands in the middle of the shelf, away from both products
        bystander = (150, 150, 200, 200)
        mask = np.zeros((240, 320), np.uint8)
        mask[150:200, 150:200] = 1

        def ctx(gray, i) -> FrameContext:
            return FrameContext(frame=np.zeros((240, 320, 3), np.uint8),
                                gray=gray, timestamp=i / 10.0, frame_index=i, fps=10.0,
                                zones=[zone], tracks=[self._track(9, bystander)],
                                person_mask=mask)

        shelf = np.full((240, 320), 100, np.uint8)
        shelf[100:140, 40:100] = 60             # product A (left)
        shelf[100:140, 220:280] = 60            # product B (right)

        events = []
        for i in range(0, 10):                  # both products present
            events += rule.update(ctx(shelf, i))

        gone_a = shelf.copy()
        gone_a[100:140, 40:100] = 100           # product A is taken
        for i in range(10, 230):                # ... and the shelf settles down
            events += rule.update(ctx(gone_a, i))
        self.assertEqual(len(events), 1)

        gone_both = gone_a.copy()
        gone_both[100:140, 220:280] = 100       # now product B goes as well
        for i in range(230, 245):
            events += rule.update(ctx(gone_both, i))
        self.assertEqual([e.rule for e in events], ["item_removal"] * 2,
                         "a different missing product was swallowed as a repeat")


# --------------------------------------------------------------------------- #
class PreviewTest(unittest.TestCase):
    """The browser preview: no GUI needed, works with a headless OpenCV too."""

    def test_gui_probe_is_a_bool(self) -> None:
        from theft_detector.pipeline import gui_available

        self.assertIsInstance(gui_available(), bool)

    def test_preview_serves_page_snapshot_and_stream(self) -> None:
        import urllib.request

        from theft_detector.webpreview import PreviewServer

        server = PreviewServer(port=0)          # 0 -> any free port
        server.start()
        self.addCleanup(server.stop)
        server.publish(np.full((60, 80, 3), 30, np.uint8))

        with urllib.request.urlopen(server.url, timeout=5) as resp:
            html = resp.read().decode()
        self.assertIn("/stream.mjpg", html)

        with urllib.request.urlopen(server.url + "snapshot.jpg", timeout=5) as resp:
            jpeg = resp.read()
        self.assertEqual(jpeg[:2], b"\xff\xd8")          # JPEG magic

        with urllib.request.urlopen(server.url + "stream.mjpg", timeout=5) as resp:
            self.assertIn("multipart/x-mixed-replace", resp.headers["Content-Type"])
            chunk = resp.read(512)
        self.assertIn(b"\xff\xd8", chunk)                # a frame really flows

    def test_busy_port_reports_a_helpful_error(self) -> None:
        from theft_detector.webpreview import PreviewServer

        first = PreviewServer(port=0)
        first.start()
        self.addCleanup(first.stop)
        port = int(first.url.rsplit(":", 1)[1].rstrip("/"))
        with self.assertRaises(RuntimeError):
            PreviewServer(port=port)


# --------------------------------------------------------------------------- #
class EndToEndTest(unittest.TestCase):
    def test_pipeline_runs_on_generated_clip(self) -> None:
        from scripts.make_demo_video import build
        from theft_detector.pipeline import Pipeline

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "short.mp4"
            build(PROJECT_ROOT / "configs" / "demo.json", clip, duration=3.0)
            self.assertTrue(clip.exists())

            cfg = Config.load(PROJECT_ROOT / "configs" / "demo.json")
            cfg.alerts.directory = str(Path(tmp) / "alerts")
            cfg.alerts.beep = False
            pipe = Pipeline(cfg, str(clip), max_frames=90, verbose=False)
            count = pipe.run()
            self.assertGreater(pipe.frame_index, 80)
            self.assertGreaterEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
