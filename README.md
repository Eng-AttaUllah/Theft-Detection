# Shop Theft Detection

A Python surveillance system that watches a shop camera (or a recorded video)
and raises **alerts when someone is stealing** - walking behind the counter,
taking an item off a shelf, loitering too long in front of merchandise, grabbing
things quickly, gathering a crowd as a distraction, or tampering with the
camera.

Everything runs offline with OpenCV + NumPy. No cloud, no GPU, no model
downloads required (optional deep-learning detectors are supported).

```
camera / video  ->  person detector  ->  tracker  ->  behaviour rules  ->  alerts
                                                                              |
                                                    console + events.jsonl + snapshots
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
| `item_removal` | shelf layout changes while a person is at the shelf | `zone_kinds`, `hold_frames`, `person_grace_seconds`, `person_margin`, `largest_change_ratio`, `repeat_gap_seconds` |
| `unattended_object` | an object is left behind / scene changed with nobody near it | `unattended_seconds`, `hold_frames` |
| `rapid_motion` | sudden fast movement at a shelf/counter (snatching) | `threshold`, `min_frames`, `person_margin` |
| `crowd` | several people gathering (distraction theft) | `min_persons`, `seconds`, `drop_seconds` |
| `camera_tamper` | lens covered, frozen picture, out of focus, blown out | `min_brightness`, `min_sharpness`, `static_seconds` |

All of them can be turned on/off and tuned in the `rules` section of the config.

## Alerts

Every alert is:

* printed to the console (severity coloured when on a TTY),
* appended to `alerts/events.jsonl` (timestamp, rule, severity, zone, people involved),
* saved as a snapshot image with the person outlined in `alerts/snapshots/`,
* accompanied by a terminal bell (`alerts.beep`).

Example line:

```
16:42:07 [HIGH] item_removal: item likely taken from 'shelf_a' - shelf layout changed while a person was at the shelf  (snapshot: alerts/snapshots/...)
```

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
| `onnx` | a YOLO `.onnx` file | best accuracy. `detector.model = "yolov8n.onnx"` (YOLOv5/v7/v8 layouts are auto-detected) |
| `caffe` | MobileNet-SSD prototxt + caffemodel | classic lightweight DNN detector |

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
  config.py                 config dataclasses + JSON loading
  zones.py                  normalised polygon zones
  detectors.py              hog / motion / onnx / caffe person detectors
  tracker.py                IoU multi-object tracker
  behaviors.py              the behaviour rules (the interesting part)
  alerts.py                 alert manager: console, JSONL log, snapshots
  webpreview.py             --web: MJPEG browser preview of the live feed
  pipeline.py               capture -> detect -> track -> rules -> alerts
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
   `person_margin` pixels of the zone;
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
