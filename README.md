# Shop Theft Detection

A Python surveillance system that watches a shop camera (or a recorded video)
and raises **alerts when someone is stealing** - walking behind the counter,
taking an item off a shelf, loitering too long in front of merchandise, grabbing
things quickly, gathering a crowd as a distraction, or tampering with the
camera.

Everything runs offline with OpenCV + NumPy. No cloud, no GPU, no model
downloads required (optional deep-learning detectors are supported).

```
CCTV / video
   |
   v
YOLO person + object detection          detectors.py  (onnx / caffe / hog / motion)
   |
   v
ByteTrack / BoT-SORT tracking           bytetrack.py  (Kalman + 3-stage association + ReID)
   |
   v
person IDs  +  object IDs               tracker.py / objects.py
   |
   v
behaviour analysis                      behaviors.py   (restricted / loiter / remove / ...)
   |
   v
theft confidence                        confidence.py  (evidence fused into one score)
   |
   v
alert                                   alerts.py      (console + events.jsonl + snapshots)
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. built-in synthetic demo (generates samples/demo.mp4 on first run)
.venv/bin/python run.py --demo

# 2. your own footage
.venv/bin/python run.py --source shop_cam.mp4

# 3. a webcam with a live window
.venv/bin/python run.py --source 0 --display

# 4. no desktop session (SSH, Docker, headless OpenCV) -> browser preview
.venv/bin/python run.py --source 0 --web        # open http://127.0.0.1:8080/
```

Handy flags:

| flag | meaning |
|------|---------|
| `-s / --source` | webcam index (`0`), video file, or RTSP/HTTP URL |
| `-c / --config` | path to a JSON config (default `config.json`) |
| `--detector` | `hog`, `motion`, `onnx`, `caffe` |
| `--tracker` | `bytetrack` (default), `botsort` (adds appearance re-ID), `iou` (legacy SORT) |
| `-d / --display` | live annotated window (needs a desktop session) |
| `--web [PORT]` | live annotated feed in a browser (default `http://127.0.0.1:8080/`) - no GUI needed |
| `--record out.mp4` | save the annotated video |
| `--alert-dir DIR` | where alerts are written (default `alerts/`) |
| `--cooldown SEC` | minimum gap between two alerts of the same rule |
| `--demo` | run on the built-in synthetic clip with `configs/demo.json` |
| `--list-rules` | show every rule and its parameters |
| `--print-config` | print the effective configuration and exit |
| `--max-frames N` | stop after N frames (useful for testing) |
| `-q / --quiet` | write alerts to disk only, nothing on the console |

## Detected behaviours

| rule | what it catches | key parameters |
|------|-----------------|----------------|
| `restricted_zone` | person stands inside the counter / stock-room zone | `zone_kinds`, `min_dwell_seconds` |
| `loitering` | someone stays in the same spot too long | `max_seconds`, `max_radius`, `gap_seconds` |
| `item_removal` | shelf layout changes while a person is at the shelf | `zone_kinds`, `hold_frames`, `person_grace_seconds`, `person_margin`, `largest_change_ratio`, `min_blob_coherence`, `repeat_gap_seconds` |
| `unattended_object` | an object is left behind / scene changed with nobody near it | `unattended_seconds`, `hold_frames` |
| `rapid_motion` | sudden fast movement at a shelf/counter (snatching) | `threshold`, `min_frames`, `person_margin` |
| `crowd` | several people gathering (distraction theft) | `min_persons`, `seconds`, `drop_seconds` |
| `object_taken` | an object-id that sat on a shelf leaves it (or vanishes) while a person holds it - needs `onnx`/`caffe` | `zone_kinds`, `min_carry_seconds`, `leave_margin`, `carrier_margin` |
| `camera_tamper` | lens covered, frozen picture, out of focus, blown out | `min_brightness`, `min_sharpness`, `static_seconds` |

All of them can be turned on/off and tuned in the `rules` section of the config.

## Pipeline, stage by stage

| stage | module | what it does |
|-------|--------|--------------|
| 1. detection | `detectors.py` | finds **persons and objects**: class `person` uses `conf_threshold`, the classes listed in `detector.object_classes` (bottle, handbag, backpack, ...) use `min_object_confidence` and keep their label. `hog`/`motion` only ever produce people. |
| 2. tracking | `bytetrack.py` | ByteTrack: a Kalman filter predicts each box, then three association stages run - high-score detections, **low-score detections** (the ByteTrack trick: an occluded person keeps her id), and one-hit tracks. `botsort` adds appearance re-identification (HSV colour histogram, cosine similarity) so a lost id comes back when IoU is gone; `iou` keeps the original SORT-style tracker. |
| 3. IDs | `tracker.py`, `objects.py` | every person keeps a stable `track_id`; a second tracker over the object detections gives each thing an id plus a state machine (`on_shelf` -> `moved` -> `carried`, then gone), remembering which shelf it started on and which person is holding it. |
| 4. behaviour | `behaviors.py` | the rules above, fed with person ids, object ids, zones, optical flow and per-shelf backgrounds. |
| 5. theft confidence | `confidence.py` | rule hits are weighted, summed and decayed into **one score per camera** (0..1). Crossing `report_threshold` raises a `theft_confidence` alert carrying the score and the evidence that produced it; it re-arms only after the score falls below `threshold * reset_ratio`. |
| 6. alert | `alerts.py` | console, `events.jsonl`, snapshots, beep. |

Choosing the tracker:

```bash
run.py -s cam.mp4 --tracker bytetrack   # default
run.py -s cam.mp4 --tracker botsort     # + appearance re-ID (occlusion, crossings)
run.py -s cam.mp4 --tracker iou         # legacy SORT-style, cheapest
```

```json
"tracking": {"algorithm": "botsort", "track_thresh": 0.5, "reid_threshold": 0.55,
             "reid_max_distance": 180}
```

`track_thresh` is the high/low score split (detections below it are still used,
just in the second stage), `reid_threshold`/`reid_max_distance` bound how far a
lost id may reappear (similarity and pixels).

> A detection only *starts* a track when it clears `track_thresh`, so if people
> flicker in and out without ever getting an id, drop `track_thresh` to your
> `detector.conf_threshold`. The low band (between the two) is what lets an
> occluded or blurred person keep the id she already has.

Tuning the theft confidence:

```json
"confidence": {"enabled": true, "report_threshold": 0.6, "warn_threshold": 0.35,
               "decay_seconds": 45.0, "reset_ratio": 0.5, "weights": {"crowd": 0.1}}
```

Weights default to `item_removal 0.45`, `object_taken 0.55`, `rapid_motion 0.25`,
`loitering 0.20`, `crowd 0.20`, `restricted_zone 0.15`, `camera_tamper 0.12`,
`unattended_object 0.10`; a single weak signal never reaches the bar, while
`restricted_zone + rapid_motion + item_removal` does (the bundled demo reports
`theft confidence 0.63` exactly when the shelf item is taken).

## Alerts

Every alert is:

* printed to the console (severity coloured when on a TTY),
* appended to `alerts/events.jsonl` (timestamp, rule, severity, zone, people involved),
* saved as a snapshot image with the person outlined in `alerts/snapshots/`,
* accompanied by a terminal bell (`alerts.beep`).

Example lines:

```
16:42:07 [HIGH]   item_removal: item likely taken from 'shelf_a' - shelf layout changed while a person was at the shelf  (snapshot: alerts/snapshots/...)
16:42:07 [HIGH] theft_confidence: theft confidence 0.63 - evidence: restricted_zone, rapid_motion + item_removal  (snapshot: alerts/snapshots/...)
```

The `theft_confidence` line is the *summary* alert: one incident produces one
line with the fused score (its `confidence` field in `events.jsonl`), instead of
having to add up the individual rules by hand.

## Watching the feed

Two ways to see what the camera sees - both show the *annotated* picture
(zones, tracks, alert flashes):

| how | needs | command |
|-----|-------|---------|
| desktop window | a GUI OpenCV build (`opencv-python`, what `requirements.txt` installs) and a display | `run.py -s 0 -d` |
| browser preview | nothing - works with `opencv-python-headless`, in Docker and over SSH | `run.py -s 0 --web` |

* The window closes on `q` or `Esc`.
* `--web [PORT]` binds `127.0.0.1` by default and prints the URL
  (`live preview: http://127.0.0.1:8080/`). From another machine, tunnel it:
  `ssh -L 8080:127.0.0.1:8080 user@shop-pc`.
* If `--display` cannot open a window the program says exactly why and keeps
  running headless instead of failing silently (the usual cause is a
  `opencv-python-headless` install: swap it for `opencv-python`).

Nothing is written to disk unless you also pass `--record out.mp4`.

## Configuring your shop

Zones are polygons with **normalised coordinates** (`0.0 - 1.0`), so one config
works for any camera resolution. Edit `config.json`:

```json
"zones": [
  {"name": "counter",  "kind": "register",  "severity": "high",
   "points": [[0.55, 0.62], [0.97, 0.62], [0.97, 0.97], [0.55, 0.97]]},
  {"name": "shelf_a",  "kind": "shelf",     "severity": "medium",
   "points": [[0.20, 0.25], [0.50, 0.25], [0.50, 0.60], [0.20, 0.60]]}
]
```

Zone kinds and how they are used:

| kind | used by |
|------|---------|
| `register` | restricted-zone + item-removal + unattended-object + rapid-motion |
| `restricted` | restricted-zone + rapid-motion (stock room, office, back door) |
| `shelf` | item-removal + unattended-object + rapid-motion |
| `entrance` | informational only (drawn, no rule attached) |

Pick the layout straight from a still frame of your camera: display a frame, note
the pixel coordinates of the corners, divide by width/height.

## Person detectors

| backend | needs | notes |
|---------|-------|-------|
| `hog` (default) | nothing | OpenCV's HOG+SVM pedestrian detector; needs OpenCV 4.x (`opencv-python<5`) - on OpenCV 5 the code warns and falls back to `motion` |
| `motion` | nothing | adaptive background model for a **fixed** camera - moving *and* standing people become blobs; the default for `--demo`; works on every OpenCV build |
| `onnx` | a YOLO `.onnx` file | best accuracy. `detector.model = "yolov8n.onnx"` (YOLOv5/v7/v8 layouts are auto-detected). Detects **persons and objects** - see below |
| `caffe` | MobileNet-SSD prototxt + caffemodel | classic lightweight DNN detector, also reports objects |

### Object detection (person ID + object ID)

With `onnx`/`caffe` the detector also reports the classes listed in
`detector.object_classes` (default: bottle, cup, handbag, backpack, book, cell
phone, scissors, teddy bear, suitcase). Those detections get their own tracker
and state machine (`objects.py`), which is what the `object_taken` rule and the
object boxes in the preview use:

```json
"detector": {"backend": "onnx", "model": "models/yolov8n.onnx",
             "object_classes": ["bottle", "handbag", "backpack"],
             "min_object_confidence": 0.35}
```

Set `object_classes` to `[]` if you only care about people. Class names come
from COCO (YOLO) / VOC (MobileNet-SSD) by default; a custom model can override
them with `detector.class_names = "labels.txt"` (one label per line).
`hog` and `motion` never produce objects, so `object_taken` stays silent there.

`--demo` works with every backend: `motion` and `hog` both report the same core
timeline of the synthetic clip (restricted zone ~6 s, rapid motion ~13 s, item
removal ~16 s, loitering ~20 s, crowd ~21 s). Because `hog` keeps detecting
people who stand still - where `motion` loses them - it also fires extra
*crowd* / *restricted_zone* / *loitering* alerts for the same gathering; raise
`--cooldown` (or `alerts.cooldown_seconds`) to collapse those repeats.

Example for YOLOv8n (exported with `yolo export model=yolov8n.onnx format=onnx`):

```json
"detector": {"backend": "onnx", "model": "models/yolov8n.onnx", "conf_threshold": 0.45}
```

## Project layout

```
run.py                      CLI entry point
config.json                 default configuration (zones + rules)
configs/demo.json           configuration used by the built-in demo
theft_detector/
  config.py                 config dataclasses + JSON loading (incl. confidence)
  zones.py                  normalised polygon zones
  detectors.py              hog / motion / onnx / caffe detectors (person + object)
  bytetrack.py              Kalman + ByteTrack / BoT-SORT multi-object tracker
  tracker.py                legacy IoU tracker + the shared Track record
  objects.py                object IDs: what it is, where it sat, who holds it
  behaviors.py              the behaviour rules (the interesting part)
  confidence.py             theft-confidence fusion (rule hits -> one score)
  alerts.py                 alert manager: console, JSONL log, snapshots
  webpreview.py             --web: MJPEG browser preview of the live feed
  pipeline.py               detect -> track -> IDs -> rules -> confidence -> alerts
  viz.py                    on-frame drawing
scripts/make_demo_video.py  builds the synthetic shop clip
tests/test_core.py          unit + end-to-end tests
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## How the shelf rule works

A slow background model is kept **per shelf zone**:

1. pixels covered by a detected person are excluded (plus a margin around them
   for arms/limbs), so a customer standing in front of the shelf is not
   mistaken for a change; the mask also survives short detector drop-outs;
2. a *motion* based detector loses people as soon as they stand still, so the
   last box of a track that disappeared is kept (`lost_person_seconds`). It
   stays excluded **while it still looks like that person** - i.e. while almost
   all of the box differs from the background (`person_fill_ratio`). The moment
   something else is going on there (he left, or only the gap of a product he
   took differs) the footprint loses its protection;
3. the background keeps adapting while nothing happens;
4. if a *coherent* region of the zone differs from the background for
   `hold_frames` **and** a person was near the zone within `person_grace_seconds`,
   the rule reports an item removal - "near" means their **feet** are within
   `person_margin` pixels of the zone. *Coherent* is a hard test: the largest
   changed blob must account for `min_blob_coherence` (default 0.80) of **all**
   changed pixels, because a removed object is one solid hole while somebody
   shuffling out of frame leaves a tall, fragmented, person-shaped smear;
5. a change that is partly hidden and shows up again later is the **same**
   physical change: if it overlaps `duplicate_overlap` of the pixels we already
   alerted on within `repeat_gap_seconds` it is suppressed instead of raising a
   second alarm;
6. if the change survives long after everybody left, it reports an unattended
   object instead;
7. after `settle_seconds` the new state is copied into the background, so a
   restocked shelf neither alerts forever nor starts a fresh episode.

Tuning tips:

* `item_removal.person_fill_ratio` is the "is that still him?" test - raise it
  if a customer is mistaken for a shelf change, lower it if somebody can stand
  at a shelf without being noticed;
* `min_blob_coherence` is the "one object or scattered noise?" test - raise it
  (0.9) if customers stepping away from a shelf are reported as removals,
  lower it (0.6) if real removals are missed because the change is broken up
  by reflections or shelf dividers;
* `person_margin` (also used by `rapid_motion`) is the "is he *at* the shelf?"
  test in pixels, measured from his **feet** to the zone. It only has to bridge
  the gap your camera projects between the shelf face and the floor in front of
  it - the bundled demo needs about 150 because its shelves sit high in the
  frame. Too small and a theft is missed (the shopper's feet land outside the
  zone), too large and people merely walking past count as being at the shelf.

## Limitations (please read)

* This is a **behaviour** detector, not a court verdict - it flags *suspicious*
  events so a human can review the snapshot/clip. Expect false positives from
  lighting changes, reflections, cleaning carts and pets.
* Alerts come in two tiers, on purpose. A **rule** alert (`item_removal`,
  `loitering`, ...) is one piece of evidence; `theft_confidence` is the fused
  verdict and only fires once enough independent evidence has piled up within
  `confidence.decay_seconds` (`report_threshold`, default 0.60). A lone rule
  hit never reaches that bar - which is exactly how the detector rides out the
  false positives listed above. `--cooldown SEC` additionally collapses repeats
  of the same rule for busy feeds.
* Accuracy depends on the person detector: with `hog`/`motion` people are found
  by shape/movement only (no re-identification, no pose). Use an ONNX YOLO model
  for materially better results. The `motion` backend keeps a person visible for
  roughly 15 s after they stop moving (`detector.motion_fg_alpha`), then the
  background absorbs them until they move again - `hog`/`onnx` do not have this
  limitation. The shelf rule compensates by remembering the last box of a track
  it lost, but somebody who never moves for minutes on end can still end up
  counted as a change of the shelf.
* Occlusion (shelves in front of people, mirrors, crowded aisles) hides both
  people and items; place the camera high with a clear view of the shelf faces.
* Zones are 2-D polygons on the image; they approximate the floor plan but do
  not model perspective depth.
* Only use it where you are allowed to - surveillance of employees/customers is
  subject to local privacy law (notice, retention limits, etc.).
