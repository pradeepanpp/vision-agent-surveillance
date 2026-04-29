# src/detection/detector.py
"""
YOLO26n Object Detector.

Wraps the Ultralytics YOLO26n model into a clean interface.
Returns structured detection results per frame.

Model: yolo26n.pt
  - NMS-free end-to-end inference
  - STAL small-object improvements (ideal for drone footage)
  - ~30ms per frame on CPU at 640px input
"""

from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np

from src.utils.logger import logger


# COCO class IDs we care about for surveillance
SURVEILLANCE_CLASSES = {
    0:  "person",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    15: "cat",     # reduces false positives from animals
    16: "dog",
}


@dataclass
class Detection:
    """
    One detected object in a single frame.

    Attributes:
        class_id    : COCO class integer (0=person, 2=car, etc.)
        class_name  : human-readable class name
        confidence  : detection confidence 0.0 - 1.0
        bbox        : bounding box [x1, y1, x2, y2] in pixels
        center      : (cx, cy) center point of bounding box
    """
    class_id:   int
    class_name: str
    confidence: float
    bbox:       list[float]   # [x1, y1, x2, y2]
    center:     tuple[float, float]


class Detector:
    """
    YOLO26n detector for surveillance video frames.

    Usage:
        detector = Detector()
        detections = detector.detect(frame)
        for d in detections:
            print(d.class_name, d.confidence, d.bbox)
    """

    def __init__(
        self,
        model_path:   str   = "yolo26s.pt",
        confidence:   float = 0.25,
        iou:          float = 0.45,
        target_classes: list[int] | None = None,
    ):
        self.confidence      = confidence
        self.iou             = iou
        self.target_classes  = target_classes or list(SURVEILLANCE_CLASSES.keys())

        logger.info(f"Loading YOLO26s from {model_path}...")
        self._model = YOLO(model_path)
        logger.info(
            f"Detector ready | "
            f"conf={confidence} iou={iou} | "
            f"classes={self.target_classes}"
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLO26n detection on one frame.

        Args:
            frame : BGR numpy array (640x360 recommended)

        Returns:
            List of Detection objects, filtered to target classes.
            Empty list if no objects found.
        """
        results = self._model.predict(
            source  = frame,
            conf    = self.confidence,
            iou     = self.iou,
            classes = self.target_classes,
            verbose = False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            class_id   = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append(Detection(
                class_id   = class_id,
                class_name = SURVEILLANCE_CLASSES.get(class_id, f"class_{class_id}"),
                confidence = round(confidence, 3),
                bbox       = [round(x1, 1), round(y1, 1),
                              round(x2, 1), round(y2, 1)],
                center     = (round(cx, 1), round(cy, 1)),
            ))

        return detections
