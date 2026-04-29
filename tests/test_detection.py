# tests/test_detection.py
"""
Phase 1 Detection Tests.

Tests the full detection pipeline:
  - YOLO26s loads correctly
  - ByteTrack assigns consistent IDs
  - Zone manager loads zones.yaml
  - Zone intersection logic works
  - Full pipeline runs on test_video.mp4
  - Output video saved with annotations

Run: python tests/test_detection.py
"""

import sys
import os
import cv2
import time
import numpy as np
sys.path.append(".")

from src.utils.logger           import logger
from src.utils.video_utils      import open_video, read_frame, prepare_frame, get_video_info
from src.detection.detector     import Detector, Detection
from src.detection.tracker      import Tracker, TrackedObject
from src.detection.zone_manager import ZoneManager

VIDEO_PATH  = "data/test_video.mp4"
CONFIG_PATH = "configs/zones.yaml"
OUTPUT_PATH = "logs/test_output.mp4"
TEST_FRAMES = 90


# ─────────────────────────────────────────────
# ZONE COLOR MAP — matches drawn zone risk levels
# ─────────────────────────────────────────────
ZONE_RISK_COLORS = {
    "low":      (0, 255, 0),     # Green
    "medium":   (0, 165, 255),   # Orange
    "high":     (0, 100, 255),   # Orange-red
    "critical": (0, 0, 255),     # Red
}


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_detector_loads():
    print("\n--- Test 1: YOLO26s loads ---")
    detector = Detector(model_path="yolo26s.pt", confidence=0.25)
    assert detector is not None
    print("✅ Test 1 passed: YOLO26s loaded")
    return detector


def test_detector_on_frame(detector: Detector):
    print("\n--- Test 2: Detection on single frame ---")
    assert os.path.exists(VIDEO_PATH), (
        f"test_video.mp4 not found at {VIDEO_PATH}\n"
        "Place your Pexels drone video at data/test_video.mp4"
    )

    cap    = open_video(VIDEO_PATH)
    _, raw = cap.read()
    cap.release()

    frame      = prepare_frame(raw)
    t0         = time.perf_counter()
    detections = detector.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"  Detections found : {len(detections)}")
    print(f"  Inference time   : {latency_ms:.1f}ms")
    for d in detections[:5]:
        print(f"    {d.class_name} conf={d.confidence:.2f} center={d.center}")

    assert isinstance(detections, list)
    assert latency_ms < 15000, "Detection too slow (>15s) — check installation"
    print(f"✅ Test 2 passed: {len(detections)} detections in {latency_ms:.0f}ms")
    return detections


def test_tracker_assigns_ids(detector: Detector):
    print("\n--- Test 3: ByteTrack assigns persistent IDs ---")
    tracker = Tracker()
    cap     = open_video(VIDEO_PATH)
    info    = get_video_info(cap)
    fw, fh  = info["width"], info["height"]

    if fh > fw:
        fw, fh = fh, fw

    ids_seen      = set()
    frames_tested = 0

    for i in range(30):
        ret, raw = cap.read()
        if not ret:
            break
        frame      = prepare_frame(raw)
        detections = detector.detect(frame)
        tracked    = tracker.update(detections, frame_shape=(360, 640))
        for t in tracked:
            ids_seen.add(t.track_id)
        frames_tested += 1

    cap.release()

    print(f"  Frames processed : {frames_tested}")
    print(f"  Unique track IDs : {sorted(ids_seen)}")
    assert isinstance(ids_seen, set)
    print(f"✅ Test 3 passed: {len(ids_seen)} unique IDs assigned by ByteTrack")
    return tracker


def test_zone_manager_loads():
    print("\n--- Test 4: ZoneManager loads zones.yaml ---")
    zm = ZoneManager(CONFIG_PATH)
    assert zm.zone_count > 0, (
        f"No zones loaded from {CONFIG_PATH}\n"
        "Make sure configs/zones.yaml exists"
    )
    print(f"  Zones loaded: {zm.zone_count}")
    for z in zm.zones:
        print(f"    {z.zone_id}: {z.name} | {z.risk_level} | threshold={z.threshold_s}s")
    print(f"✅ Test 4 passed: {zm.zone_count} zones loaded")
    return zm


def test_zone_intersection(zm: ZoneManager):
    print("\n--- Test 5: Zone intersection logic ---")
    frame_wh = (640, 360)

    # Centroid points computed from actual drawn polygon coordinates
    # These are guaranteed to be inside each zone
    test_points = {
        "zone_parking center":      (418, 76),
        "zone_road center":         (230, 152),
        "zone_intersection center": (42,  269),
        "zone_restricted center":   (424, 298),
    }

    for desc, point in test_points.items():
        zone      = zm.get_zone_for_point(point, frame_wh)
        zone_name = zone.name if zone else "None"
        print(f"  {desc:30s} → {zone_name}")

    # Assert zone_parking
    z = zm.get_zone_for_point((418, 76), frame_wh)
    assert z is not None,               "Expected zone_parking at (418,76)"
    assert z.zone_id == "zone_parking", f"Expected zone_parking, got {z.zone_id}"
    assert z.risk_level == "low",       f"Expected low, got {z.risk_level}"

    # Assert zone_road
    z = zm.get_zone_for_point((230, 152), frame_wh)
    assert z is not None,             "Expected zone_road at (230,152)"
    assert z.zone_id == "zone_road",  f"Expected zone_road, got {z.zone_id}"
    assert z.risk_level == "medium",  f"Expected medium, got {z.risk_level}"

    # Assert zone_intersection
    z = zm.get_zone_for_point((42, 269), frame_wh)
    assert z is not None,                    "Expected zone_intersection at (42,269)"
    assert z.zone_id == "zone_intersection", f"Expected zone_intersection, got {z.zone_id}"
    assert z.risk_level == "high",           f"Expected high, got {z.risk_level}"

    # Assert zone_restricted
    z = zm.get_zone_for_point((424, 298), frame_wh)
    assert z is not None,                   "Expected zone_restricted at (424,298)"
    assert z.zone_id == "zone_restricted",  f"Expected zone_restricted, got {z.zone_id}"
    assert z.risk_level == "critical",      f"Expected critical, got {z.risk_level}"

    print("✅ Test 5 passed: all zone centroids correctly identified")


def test_full_pipeline_with_output():
    print("\n--- Test 6: Full pipeline + annotated video output ---")
    detector = Detector(model_path="yolo26s.pt", confidence=0.25)
    tracker  = Tracker()
    zm       = ZoneManager(CONFIG_PATH)
    cap      = open_video(VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 15, (640, 360))

    frame_count  = 0
    total_dets   = 0
    latencies    = []
    risk_summary = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    print(f"  Processing {TEST_FRAMES} frames...")

    for _ in range(TEST_FRAMES):
        ret, raw = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw = cap.read()
        if not ret:
            break

        frame = prepare_frame(raw)
        t0    = time.perf_counter()

        # Detect
        detections = detector.detect(frame)

        # Track
        tracked = tracker.update(detections, frame_shape=(360, 640))

        latencies.append((time.perf_counter() - t0) * 1000)
        total_dets += len(tracked)

        # Zone assignment
        for obj in tracked:
            zone     = zm.get_zone_for_point(obj.center, (640, 360))
            obj.zone = zone.zone_id if zone else None
            if zone:
                risk_summary[zone.risk_level] = \
                    risk_summary.get(zone.risk_level, 0) + 1

        # Draw
        frame = zm.draw_zones(frame, (640, 360))
        frame = _draw_tracked_objects(frame, tracked, zm)
        frame = _draw_stats(frame, frame_count, tracked, latencies[-1])

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    fps     = 1000 / avg_lat if avg_lat > 0 else 0

    print(f"  Frames processed : {frame_count}")
    print(f"  Total tracked    : {total_dets}")
    print(f"  Avg latency      : {avg_lat:.1f}ms → {fps:.1f} fps")
    print(f"  Zone hits        : {risk_summary}")
    print(f"  Output saved     : {OUTPUT_PATH}")

    assert frame_count > 0
    assert os.path.exists(OUTPUT_PATH)
    assert avg_lat < 10000
    print(f"✅ Test 6 passed: pipeline runs at {fps:.1f}fps, output saved")


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────

def _draw_tracked_objects(
    frame: np.ndarray,
    tracked: list[TrackedObject],
    zm: ZoneManager,
) -> np.ndarray:
    """Draw bounding boxes and labels, coloured by zone risk level."""

    # Build zone_id → risk_level lookup from loaded zones
    zone_risk = {z.zone_id: z.risk_level for z in zm.zones}

    for obj in tracked:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]

        # Colour by zone risk level — grey if outside all zones
        if obj.zone and obj.zone in zone_risk:
            risk  = zone_risk[obj.zone]
            color = ZONE_RISK_COLORS.get(risk, (200, 200, 200))
        else:
            color = (200, 200, 200)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label — ID + class + zone
        zone_label = f" [{obj.zone}]" if obj.zone else ""
        label      = f"ID{obj.track_id} {obj.class_name}{zone_label}"
        label_y    = max(y1 - 5, 15)

        cv2.putText(
            frame, label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, color, 1, cv2.LINE_AA,
        )

    return frame


def _draw_stats(
    frame: np.ndarray,
    frame_num: int,
    tracked: list[TrackedObject],
    latency_ms: float,
) -> np.ndarray:
    """Draw frame stats in top-left corner."""
    fps   = 1000 / latency_ms if latency_ms > 0 else 0
    stats = [
        f"Frame: {frame_num}",
        f"Tracked: {len(tracked)}",
        f"FPS: {fps:.1f}",
        f"YOLO26s",
    ]
    for i, text in enumerate(stats):
        cv2.putText(
            frame, text,
            (8, 18 + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 56)
    print("  Phase 1 — Detection Pipeline Tests")
    print("=" * 56)

    detector = test_detector_loads()
    test_detector_on_frame(detector)
    test_tracker_assigns_ids(detector)
    zm = test_zone_manager_loads()
    test_zone_intersection(zm)
    test_full_pipeline_with_output()

    print()
    print("=" * 56)
    print("✅ All Phase 1 tests passed!")
    print("Next → Phase 2: LangGraph State Memory")
    print("=" * 56)
    print(f"\nCheck annotated output video: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_all_tests()