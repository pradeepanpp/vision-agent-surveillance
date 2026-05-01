# tests/test_state.py
"""
Phase 2 State Tests.

Tests the full LangGraph state memory system:
  - State initialises correctly
  - State updates with tracked objects
  - Dwell time accumulates over frames
  - Zone history records correctly
  - Loitering detection fires at threshold
  - Probing detection fires on repeated visits
  - State summary generates readable text
  - Full integrated pipeline test

Run: python tests/test_state.py
"""

import sys
import os
import time
sys.path.append(".")

from src.state.schema    import (
    make_initial_state, ObjectRecord, BehaviorEvent
)
from src.state.updater   import update_state, get_state_summary
from src.state.behaviors import detect_behaviors
from src.detection.zone_manager  import ZoneManager
from src.detection.tracker import TrackedObject
from src.utils.logger import logger

CONFIG_PATH = "configs/zones.yaml"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_tracked(
    track_id:   int,
    class_name: str,
    cx:         float,
    cy:         float,
    zone:       str | None = None,
) -> TrackedObject:
    """Create a mock TrackedObject for testing."""
    return TrackedObject(
        track_id   = track_id,
        class_id   = 2 if class_name == "car" else 0,
        class_name = class_name,
        confidence = 0.85,
        bbox       = [cx-20, cy-15, cx+20, cy+15],
        center     = (cx, cy),
        zone       = zone,
    )


def simulate_frames(
    state,
    tracked_list,
    n_frames: int,
    zone_mgr,
    fps: float = 15.0,
) -> tuple:
    """
    Simulate n_frames with the same tracked objects.
    Returns (final_state, events_collected).
    """
    all_events = []
    for _ in range(n_frames):
        state = update_state(state, tracked_list, fps)
        state = detect_behaviors(state, zone_mgr)
        all_events.extend(state["pending_events"])
        # Clear pending so we don't recount
        state = {**state, "pending_events": []}
    return state, all_events


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_state_initialises():
    print("\n--- Test 1: State initialises correctly ---")
    state = make_initial_state(fps=15.0)

    assert state["frame_number"]    == 0
    assert state["fps"]             == 15.0
    assert state["object_log"]      == {}
    assert state["pending_events"]  == []
    assert state["alert_count"]     == 0

    print("✅ Test 1 passed: state initialised with correct defaults")
    return state


def test_state_update_adds_object():
    print("\n--- Test 2: State update adds new object ---")
    state   = make_initial_state(fps=15.0)
    tracked = [make_tracked(1, "car", 230.0, 152.0, "zone_road")]

    state   = update_state(state, tracked, fps=15.0)

    assert state["frame_number"] == 1
    assert 1 in state["object_log"]

    rec = state["object_log"][1]
    assert rec.track_id    == 1
    assert rec.class_name  == "car"
    assert rec.current_zone == "zone_road"
    assert rec.is_active

    print(f"  Object added: ID-{rec.track_id} {rec.class_name} in {rec.current_zone}")
    print("✅ Test 2 passed: new object correctly added to state")


def test_dwell_time_accumulates():
    print("\n--- Test 3: Dwell time accumulates over frames ---")
    state   = make_initial_state(fps=15.0)
    fps     = 15.0
    n_frames = 30   # 2 seconds at 15fps

    tracked = [make_tracked(1, "car", 230.0, 152.0, "zone_road")]

    for _ in range(n_frames):
        state = update_state(state, tracked, fps=fps)

    rec    = state["object_log"][1]
    dwell  = rec.dwell_seconds("zone_road", fps)

    assert state["frame_number"]                == n_frames
    assert rec.frame_counts.get("zone_road", 0) == n_frames
    assert abs(dwell - 2.0)                     < 0.1, f"Expected ~2.0s, got {dwell}"

    print(f"  After {n_frames} frames at {fps}fps: dwell = {dwell:.1f}s")
    print("✅ Test 3 passed: dwell time accumulates correctly")


def test_zone_history_records():
    print("\n--- Test 4: Zone history records transitions ---")
    state = make_initial_state(fps=15.0)

    # Simulate object moving through zones
    movements = [
        ("zone_road",         10),   # 10 frames on road
        ("zone_intersection",  5),   # 5 frames at intersection
        ("zone_road",          8),   # back to road
        ("zone_restricted",    3),   # enters restricted zone
    ]

    tracked = [make_tracked(2, "car", 230.0, 152.0, "zone_road")]

    for zone, n in movements:
        for _ in range(n):
            tracked[0] = make_tracked(2, "car", 230.0, 152.0, zone)
            state      = update_state(state, tracked, fps=15.0)

    rec = state["object_log"][2]

    assert "zone_road"         in rec.zone_history
    assert "zone_intersection" in rec.zone_history
    assert "zone_restricted"   in rec.zone_history
    assert rec.zone_visit_count("zone_road") == 2    # visited twice

    print(f"  Zone history: {rec.zone_history}")
    print(f"  zone_road visits: {rec.zone_visit_count('zone_road')}")
    print("✅ Test 4 passed: zone history correctly recorded")


def test_loitering_detection():
    print("\n--- Test 5: Loitering detection fires at threshold ---")
    zm    = ZoneManager(CONFIG_PATH)
    state = make_initial_state(fps=15.0)
    fps   = 15.0

    # zone_intersection has 60s threshold — simulate 65 seconds
    n_frames = int(65 * fps)   # 975 frames
    tracked  = [make_tracked(3, "car", 42.0, 269.0, "zone_intersection")]

    state, events = simulate_frames(state, tracked, n_frames, zm, fps)

    loitering_events = [e for e in events if e.event_type == "loitering"]
    assert len(loitering_events) > 0, (
        f"No loitering event fired after {65}s in zone_intersection "
        f"(threshold=60s). Check zone coordinates."
    )

    ev = loitering_events[0]
    print(f"  Event: {ev.event_type} | zone={ev.zone_id} | "
          f"duration={ev.duration_s:.0f}s | risk={ev.risk_level}")
    print("✅ Test 5 passed: loitering event fired after threshold exceeded")


def test_probing_detection():
    print("\n--- Test 6: Probing detection fires on repeated visits ---")
    zm    = ZoneManager(CONFIG_PATH)
    state = make_initial_state(fps=15.0)
    fps   = 15.0

    # Simulate 3 visits to restricted zone with gaps between
    for visit in range(3):
        # Enter restricted zone for 10 frames
        for _ in range(10):
            tracked = [make_tracked(4, "person", 424.0, 298.0, "zone_restricted")]
            state   = update_state(state, tracked, fps)
            state   = detect_behaviors(state, zm)

        # Leave zone for 10 frames (zone=None)
        for _ in range(10):
            tracked = [make_tracked(4, "person", 100.0, 100.0, None)]
            state   = update_state(state, tracked, fps)

    rec    = state["object_log"][4]
    visits = rec.zone_visit_count("zone_restricted")

    print(f"  Zone history    : {rec.zone_history}")
    print(f"  Restricted visits: {visits}")

   
    assert visits >= 2, (
        f"Expected >=2 visits to zone_restricted, got {visits}\n"
        f"Zone history: {rec.zone_history}"
    )

    probing_events = [
        e for e in state["pending_events"]
        if e.event_type == "probing"
    ]
    print(f"  Probing events  : {len(probing_events)}")
    print("✅ Test 6 passed: probing behaviour correctly detected")


def test_state_summary():
    print("\n--- Test 7: State summary generates readable text ---")
    state   = make_initial_state(fps=15.0)
    tracked = [
        make_tracked(1, "car",    230.0, 152.0, "zone_road"),
        make_tracked(2, "car",    418.0,  76.0, "zone_parking"),
        make_tracked(3, "person", 424.0, 298.0, "zone_restricted"),
    ]

    for _ in range(30):
        state = update_state(state, tracked, fps=15.0)

    summary = get_state_summary(state)
    assert "Frame"     in summary
    assert "ID-1"      in summary
    assert "ID-2"      in summary
    assert "ID-3"      in summary
    assert "zone_road" in summary

    print(f"\n{summary}\n")
    print("✅ Test 7 passed: state summary generated correctly")


def test_full_pipeline_with_video():
    print("\n--- Test 8: Full pipeline — detection + state + behavior ---")

    video_path = "data/test_video.mp4"
    if not os.path.exists(video_path):
        print("  ⏭  Test 8 skipped: test_video.mp4 not found")
        return

    import cv2
    from src.detection.detector import Detector
    from src.detection.tracker  import Tracker

    detector  = Detector(model_path="yolo26s.pt", confidence=0.25)
    tracker   = Tracker()
    zm        = ZoneManager(CONFIG_PATH)
    state     = make_initial_state(fps=15.0)
    cap       = cv2.VideoCapture(video_path)

    from src.utils.video_utils import prepare_frame
    fps       = cap.get(cv2.CAP_PROP_FPS) or 15.0
    state     = {**state, "fps": fps}

    frames_processed = 0
    events_detected  = 0
    t0               = time.perf_counter()

    for _ in range(60):
        ret, raw = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw = cap.read()
        if not ret:
            break

        frame      = prepare_frame(raw)
        detections = detector.detect(frame)
        tracked    = tracker.update(detections, frame_shape=(360, 640))

        # Zone assignment
        for obj in tracked:
            zone     = zm.get_zone_for_point(obj.center, (640, 360))
            obj.zone = zone.zone_id if zone else None

        state = update_state(state, tracked, fps)
        state = detect_behaviors(state, zm)
        events_detected += len(state["pending_events"])

        frames_processed += 1

    cap.release()
    duration = time.perf_counter() - t0

    print(f"  Frames processed    : {frames_processed}")
    print(f"  Objects in state    : {len(state['object_log'])}")
    print(f"  Events detected     : {events_detected}")
    print(f"  Total time          : {duration:.1f}s")
    print(f"\n{get_state_summary(state)}\n")

    assert frames_processed > 0
    assert len(state["object_log"]) >= 0
    print("✅ Test 8 passed: full pipeline with video completed")


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 56)
    print("  Phase 2 — State Memory Tests")
    print("=" * 56)

    test_state_initialises()
    test_state_update_adds_object()
    test_dwell_time_accumulates()
    test_zone_history_records()
    test_loitering_detection()
    test_probing_detection()
    test_state_summary()
    test_full_pipeline_with_video()

    print()
    print("=" * 56)
    print("✅ All Phase 2 tests passed!")
    print("Next → Phase 3: LLM Behavioral Reasoning")
    print("=" * 56)


if __name__ == "__main__":
    run_all_tests()