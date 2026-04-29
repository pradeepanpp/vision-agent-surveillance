# src/utils/video_utils.py
"""
Video utility functions.

Handles frame loading, resizing, and orientation correction.
All frames are normalised to landscape 640x360 before detection.
"""

import cv2
import numpy as np
from src.utils.logger import logger


TARGET_W = 640
TARGET_H = 360


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and return the capture object."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur    = total / fps if fps > 0 else 0

    logger.info(
        f"Video opened: {path} | "
        f"{w}x{h} | {fps:.1f}fps | "
        f"{total} frames | {dur:.1f}s"
    )
    return cap


def read_frame(
    cap: cv2.VideoCapture,
    loop: bool = True,
) -> tuple[bool, np.ndarray | None]:
    """
    Read one frame from the capture.

    If loop=True and the video ends, rewinds to frame 0.
    Returns (success, frame) where frame is BGR numpy array.
    """
    ret, frame = cap.read()

    if not ret and loop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    return ret, frame


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """
    Prepare a raw frame for YOLO26 detection.

    Steps:
      1. Rotate portrait (h > w) to landscape
      2. Resize to TARGET_W x TARGET_H (640x360)
      3. Return BGR frame ready for YOLO inference
    """
    h, w = frame.shape[:2]

    # Rotate portrait video to landscape
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w  = frame.shape[:2]

    # Resize to detection resolution
    if w != TARGET_W or h != TARGET_H:
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))

    return frame


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Return basic metadata about the opened video."""
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return {
        "width":        w,
        "height":       h,
        "fps":          fps,
        "total_frames": total,
        "duration_s":   total / fps if fps > 0 else 0,
        "orientation":  "portrait" if h > w else "landscape",
    }
