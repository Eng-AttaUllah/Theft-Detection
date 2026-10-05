"""Person detectors.

Backends
--------
``hog``    OpenCV's built-in HOG + linear SVM people detector (no downloads).
``motion`` Background subtraction (MOG2) blob detector - good for a fixed shop
           camera where moving blobs are people.
``onnx``   YOLOv5/v7/v8 ONNX model loaded with ``cv2.dnn`` (best accuracy).
``caffe``  MobileNet-SSD Caffe model.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .config import DetectorConfig


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    """One detected person, ``box`` is ``(x1, y1, x2, y2)`` in pixels."""

    box: tuple[int, int, int, int]
    confidence: float
    label: str = "person"

    @property
    def x1(self) -> int:
        return self.box[0]

    @property
    def y1(self) -> int:
        return self.box[1]

    @property
    def x2(self) -> int:
        return self.box[2]

    @property
    def y2(self) -> int:
        return self.box[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def feet(self) -> tuple[int, int]:
        """Bottom-centre point - a good ground-plane estimate."""
        return (self.x1 + self.x2) // 2, self.y2


# --------------------------------------------------------------------------- #
class BaseDetector(ABC):
    name = "base"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.frame_index = 0

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        self.frame_index += 1
        return self.detect(frame)


# --------------------------------------------------------------------------- #
class HogPersonDetector(BaseDetector):
    """Classic HOG/SVM pedestrian detector bundled with OpenCV."""

    name = "hog"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        scale = 1.0
        small = frame
        max_w = self.cfg.max_inference_width
        if frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        rects, weights = self.hog.detectMultiScale(
            small,
            winStride=tuple(self.cfg.win_stride),
            padding=tuple(self.cfg.padding),
            scale=self.cfg.scale,
            hitThreshold=self.cfg.hit_threshold,
        )
        out: list[Detection] = []
        for (x, y, w, h), raw in zip(rects, np.asarray(weights).ravel()):
            conf = 1.0 / (1.0 + math.exp(-float(raw)))
            if conf < self.cfg.conf_threshold:
                continue
            box = (
                int(x / scale),
                int(y / scale),
                int((x + w) / scale),
                int((y + h) / scale),
            )
            out.append(Detection(box=_clip(box, frame), confidence=conf))
        return _nms(out, self.cfg.nms_threshold)


# --------------------------------------------------------------------------- #
class MotionDetector(BaseDetector):
    """Moving/still-blob detector for a fixed camera.

    Keeps its own adaptive background model: pixels where nothing stands are
    updated quickly (lighting drift), pixels belonging to a foreground object
    are updated very slowly, so a person who stops in front of a shelf keeps
    being detected for ~15 s instead of melting into the background.
    ``absorb()`` lets the pipeline erase the model inside a box once the
    tracker noticed the person is gone - that kills "ghost" people.
    """

    name = "motion"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self.bg: np.ndarray | None = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur = gray.astype(np.float32)

        if self.bg is None or self.bg.shape != cur.shape:
            self.bg = cur.copy()
            return []

        fg = np.abs(cur - self.bg) > float(self.cfg.motion_threshold)
        ratio = float(fg.mean())
        if ratio > float(self.cfg.motion_reset_ratio):
            # lights went out / camera moved - start over
            self.bg = cur.copy()
            return []

        alpha = np.where(fg, float(self.cfg.motion_fg_alpha),
                         float(self.cfg.motion_bg_alpha)).astype(np.float32)
        self.bg += alpha * (cur - self.bg)

        fg_u8 = fg.astype(np.uint8)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.cfg.min_blob_area_ratio * h * w
        max_area = self.cfg.max_blob_area_ratio * h * w
        min_h = self.cfg.min_blob_height_ratio * h
        warm = self.frame_index <= self.cfg.warmup_frames

        out: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < min_h or bw < 8:
                continue
            aspect = bw / max(1, bh)
            if not (self.cfg.min_aspect_ratio <= aspect <= self.cfg.max_aspect_ratio):
                continue
            fill = area / float(bw * bh)
            if fill < self.cfg.min_fill_ratio:
                continue
            conf = float(np.clip(0.45 + 0.40 * fill, 0.0, 0.95))
            if warm:
                conf *= 0.5
            if conf < self.cfg.conf_threshold:
                continue
            out.append(Detection(box=_clip((x, y, x + bw, y + bh), frame), confidence=conf))
        return _nms(out, self.cfg.nms_threshold)

    # ------------------------------------------------------------------ #
    def absorb(self, box: tuple[int, int, int, int], frame: np.ndarray) -> None:
        """Accept the current pixels of ``box`` as background (ghost removal)."""
        if self.bg is None:
            return
        h, w = self.bg.shape
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(w, box[2]), min(h, box[3])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg[y1:y2, x1:x2] = gray[y1:y2, x1:x2].astype(np.float32)


# --------------------------------------------------------------------------- #
def decode_yolo(
    pred: np.ndarray,
    conf_threshold: float = 0.25,
    layout: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLO output into ``(xywh, scores, class_ids)``.

    Accepts the common export layouts:

    * YOLOv8 : ``(1, 4 + nc, N)`` or ``(1, N, 4 + nc)`` - no objectness
    * YOLOv5/6/7 : ``(1, N, 5 + nc)`` - objectness * class scores
    """
    arr = np.squeeze(np.asarray(pred, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape {arr.shape}")

    transposed = False
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 512:
        arr = arr.T
        transposed = True

    cols = arr.shape[1]
    if layout == "auto":
        # (4 + nc) columns without objectness, or a rotated YOLOv8 export.
        use_objness = not transposed and cols not in (84,) and cols >= 6
    elif layout == "yolov8":
        use_objness = False
    elif layout in ("yolov5", "yolov7"):
        use_objness = True
    else:
        raise ValueError(f"unknown YOLO layout {layout!r}")

    boxes = arr[:, 0:4]
    if use_objness:
        obj = arr[:, 4:5]
        cls_scores = arr[:, 5:]
        scores = obj * cls_scores
    else:
        cls_scores = arr[:, 4:]
        scores = cls_scores

    class_ids = np.argmax(scores, axis=1)
    best = scores[np.arange(scores.shape[0]), class_ids]
    mask = best >= conf_threshold
    if not mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return boxes[mask], best[mask], class_ids[mask].astype(np.int32)


# --------------------------------------------------------------------------- #
# Class names - COCO (YOLO) and Pascal VOC (MobileNet-SSD) orders. A custom
# model can override them with detector.class_names (one label per line).
COCO_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

VOC_NAMES: tuple[str, ...] = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def load_class_names(path: str) -> list[str]:
    """One label per line; blank lines and ``#`` comments are ignored."""
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute():
        from .config import PROJECT_ROOT

        p = PROJECT_ROOT / p
    names: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            names.append(text)
    return names


def wanted_label(
    label: str,
    confidence: float,
    objects: set[str],
    conf_threshold: float,
    min_object_confidence: float,
) -> str | None:
    """Decide if a detection is one of the things this pipeline tracks.

    * ``person`` must clear the normal ``conf_threshold`` (people are what the
      behaviour rules are built on);
    * the classes listed in ``detector.object_classes`` only need
      ``min_object_confidence`` - a small bottle is much harder to see than a
      standing human, and the object-ID stage tolerates that;
    * everything else is dropped, so the tracker is not flooded with chairs.

    Returns the label to keep, or ``None`` to drop the detection.
    """
    if label == "person":
        return label if confidence >= conf_threshold else None
    if objects and label.lower() in objects:
        return label if confidence >= min_object_confidence else None
    return None


class OnnxDetector(BaseDetector):
    """YOLO-style ONNX model loaded through OpenCV's DNN module.

    Detects **persons and objects**: class ``person`` uses ``conf_threshold``,
    the classes listed in ``detector.object_classes`` (bottle, handbag, ...)
    use ``min_object_confidence`` and carry their label, which is what the
    object-ID stage tracks. Everything else is dropped.
    """

    name = "onnx"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        if not cfg.model:
            raise ValueError("detector.backend = 'onnx' requires detector.model (path to .onnx)")
        self.net = cv2.dnn.readNetFromONNX(cfg.model)
        self.layout = getattr(cfg, "layout", "auto")
        self.names: list[str] = (
            load_class_names(cfg.class_names) if cfg.class_names else list(COCO_NAMES)
        )
        self.objects = {n.lower() for n in (cfg.object_classes or [])}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        size = self.cfg.input_size
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(size, size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()
        # objects are allowed a lower bar than people, so decode from the lower one
        decode_at = min(self.cfg.conf_threshold, self.cfg.min_object_confidence)
        boxes, scores, class_ids = decode_yolo(out, decode_at, self.layout)
        if boxes.size == 0:
            return []

        # undo letterbox-style scaling (blobFromImage stretches, no crop)
        sx = w / size
        sy = h / size
        xywh = boxes.copy()
        xywh[:, 0] *= sx
        xywh[:, 1] *= sy
        xywh[:, 2] *= sx
        xywh[:, 3] *= sy

        idx = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.astype(float).tolist(),
            decode_at,
            self.cfg.nms_threshold,
        )
        out_dets: list[Detection] = []
        for i in np.array(idx).ravel() if len(idx) else []:
            cid = int(class_ids[i])
            label = self.names[cid] if 0 <= cid < len(self.names) else f"class_{cid}"
            conf = float(scores[i])
            keep = wanted_label(
                label, conf, self.objects,
                self.cfg.conf_threshold, self.cfg.min_object_confidence,
            )
            if keep is None:
                continue
            cx, cy, bw, bh = xywh[i]
            box = (
                int(cx - bw / 2),
                int(cy - bh / 2),
                int(cx + bw / 2),
                int(cy + bh / 2),
            )
            out_dets.append(
                Detection(box=_clip(box, frame), confidence=conf, label=keep)
            )
        return out_dets


class CaffeDetector(BaseDetector):
    """MobileNet-SSD (VOC classes: person is index 15).

    Emits persons plus any ``detector.object_classes`` entry that exists in
    VOC (``bottle`` is the useful one for a shop) so the object-ID stage has
    something to work with.
    """

    name = "caffe"
    PERSON_CLASS = 15

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        if not (cfg.model and cfg.weights):
            raise ValueError(
                "detector.backend = 'caffe' requires detector.model (prototxt) "
                "and detector.weights (caffemodel)"
            )
        self.net = cv2.dnn.readNetFromCaffe(cfg.model, cfg.weights)
        self.names: list[str] = (
            load_class_names(cfg.class_names) if cfg.class_names else list(VOC_NAMES)
        )
        self.objects = {n.lower() for n in (cfg.object_classes or [])}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True
        )
        self.net.setInput(blob)
        pred = self.net.forward()[0]

        out: list[Detection] = []
        for det in pred:
            cls_id = int(det[1])
            conf = float(det[2])
            label = self.names[cls_id] if 0 <= cls_id < len(self.names) else f"class_{cls_id}"
            keep = wanted_label(
                label, conf, self.objects,
                self.cfg.conf_threshold, self.cfg.min_object_confidence,
            )
            if keep is None:
                continue
            box = (
                int(det[3] * w),
                int(det[4] * h),
                int(det[5] * w),
                int(det[6] * h),
            )
            out.append(Detection(box=_clip(box, frame), confidence=conf, label=keep))
        return _nms(out, self.cfg.nms_threshold)


# --------------------------------------------------------------------------- #
def hog_available() -> bool:
    """OpenCV 5 removed the HOG people detector (and the Haar cascades)."""
    return hasattr(cv2, "HOGDescriptor") and hasattr(
        cv2, "HOGDescriptor_getDefaultPeopleDetector"
    )


def caffe_available() -> bool:
    return hasattr(cv2.dnn, "readNetFromCaffe")


def build_detector(cfg: DetectorConfig) -> BaseDetector:
    backend = cfg.backend.lower()
    if backend == "hog":
        if hog_available():
            return HogPersonDetector(cfg)
        print(
            "[warn] OpenCV was built without the HOG people detector "
            f"(OpenCV {cv2.__version__}) - falling back to the 'motion' backend.\n"
            "       Use --detector onnx (or install opencv-python<5) for the "
            "appearance-based detectors.",
            file=sys.stderr,
        )
        cfg.backend = "motion"
        return MotionDetector(cfg)
    if backend == "motion":
        return MotionDetector(cfg)
    if backend == "onnx":
        return OnnxDetector(cfg)
    if backend == "caffe":
        if not caffe_available():
            raise RuntimeError(
                "this OpenCV build has no Caffe loader - use detector.backend="
                "'onnx' or install a 4.x build of opencv-python"
            )
        return CaffeDetector(cfg)
    raise ValueError(
        f"unknown detector backend {cfg.backend!r} (expected hog|motion|onnx|caffe)"
    )


# --------------------------------------------------------------------------- #
def _clip(box: tuple[int, int, int, int], frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    return max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(0, min(x2, w)), max(0, min(y2, h))


def _nms(dets: Sequence[Detection], threshold: float) -> list[Detection]:
    if not dets:
        return []
    order = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    for cand in order:
        if all(_iou(cand.box, k.box) < threshold for k in keep):
            keep.append(cand)
    return keep


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter)
