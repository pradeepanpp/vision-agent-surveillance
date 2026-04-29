# src/detection/tracker.py
"""
ByteTrack Multi-Object Tracker.

Wraps supervision's ByteTrack implementation.
Assigns persistent track IDs across frames so the same
person or vehicle keeps the same ID throughout the video.

Without tracking:
    Frame 1: person at (120, 200) — unknown identity
    Frame 2: person at (122, 205) — unknown identity

With ByteTrack:
    Frame 1: person at (120, 200) → ID 3
    Frame 2: person at (122, 205) → ID 3  ← same person confirmed
    Frame 300: person at (300, 200) → ID 3  ← still the same person

This persistent identity is what makes dwell time calculation
possible — you know ID 3 has been in Zone B for exactly 47 seconds.
"""

from dataclasses import dataclass
import numpy as np
import supervision as sv

from src.detection.detector import Detection
from src.utils.logger import logger


@dataclass
class TrackedObject:
    """
    A detected object with a persistent track ID.

    Extends Detection with:
        track_id : stable integer ID across frames
        zone     : zone name if inside a zone, else None
    """
    track_id:   int
    class_id:   int
    class_name: str
    confidence: float
    bbox:       list[float]
    center:     tuple[float, float]
    zone:       str | None = None


class Tracker:
    """
    ByteTrack wrapper for multi-object tracking.

    Usage:
        tracker  = Tracker()
        # Each frame — pass detections in, get tracked objects back
        tracked  = tracker.update(detections, frame_shape)
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.35,
        lost_track_buffer:          int   = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate:                 int   = 30,
    ):
        self._tracker = sv.ByteTrack(
            track_activation_threshold = track_activation_threshold,
            lost_track_buffer          = lost_track_buffer,
            minimum_matching_threshold = minimum_matching_threshold,
            frame_rate                 = frame_rate,
        )
        logger.info(
            f"ByteTrack ready | "
            f"buffer={lost_track_buffer} frames | "
            f"match_thresh={minimum_matching_threshold}"
        )

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackedObject]:
        """
        Update tracker with new detections from one frame.

        Args:
            detections  : list of Detection from Detector.detect()
            frame_shape : (height, width) of the frame

        Returns:
            List of TrackedObject with persistent track_ids.
        """
        if not detections:
            return []

        # Convert Detection list → supervision Detections format
        xyxy       = np.array([d.bbox for d in detections], dtype=np.float32)
        confs      = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids  = np.array([d.class_id for d in detections], dtype=int)

        sv_dets = sv.Detections(
            xyxy      = xyxy,
            confidence= confs,
            class_id  = class_ids,
        )

        # Run ByteTrack
        tracked_sv = self._tracker.update_with_detections(sv_dets)

        if len(tracked_sv) == 0:
            return []

        # Convert back to TrackedObject list
        tracked_objects = []
        for i in range(len(tracked_sv)):
            x1, y1, x2, y2 = tracked_sv.xyxy[i].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cid = int(tracked_sv.class_id[i])

            from src.detection.detector import SURVEILLANCE_CLASSES
            tracked_objects.append(TrackedObject(
                track_id   = int(tracked_sv.tracker_id[i]),
                class_id   = cid,
                class_name = SURVEILLANCE_CLASSES.get(cid, f"class_{cid}"),
                confidence = round(float(tracked_sv.confidence[i]), 3),
                bbox       = [round(x1, 1), round(y1, 1),
                              round(x2, 1), round(y2, 1)],
                center     = (round(cx, 1), round(cy, 1)),
            ))

        return tracked_objects

    def reset(self):
        """Reset tracker state — call when switching videos."""
        self._tracker.reset()
        logger.info("Tracker reset")
