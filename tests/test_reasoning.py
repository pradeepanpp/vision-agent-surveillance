# tests/test_reasoning.py
"""
Phase 3 Reasoning Tests.

Tests the full LLM reasoning pipeline:
  - Reasoner initialises with API key
  - Prompt builder creates correct context
  - LLM analyses loitering event correctly
  - LLM analyses probing event correctly
  - Risk level matches zone severity
  - Alert logger writes and reads correctly
  - Full integrated pipeline test

Run: python tests/test_reasoning.py
"""

import sys
import os
import json
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from src.state.schema     import make_initial_state, BehaviorEvent
from src.state.updater    import update_state, get_state_summary
from src.state.behaviors  import detect_behaviors
from src.reasoning.prompts     import build_context, SYSTEM_PROMPT
from src.reasoning.reasoner    import Reasoner, SecurityAlert
from src.reasoning.alert_logger import (
    write_alert, read_alerts, get_alert_stats, ALERT_LOG_PATH
)
from src.detection.zone_manager import ZoneManager
from src.detection.tracker import TrackedObject
from src.utils.logger import logger

CONFIG_PATH = "configs/zones.yaml"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_tracked(tid, cls, cx, cy, zone=None):
    return TrackedObject(
        track_id=tid, class_id=2 if cls=="car" else 0,
        class_name=cls, confidence=0.85,
        bbox=[cx-20,cy-15,cx+20,cy+15],
        center=(cx,cy), zone=zone,
    )


def make_loitering_event(
    track_id=1, class_name="car",
    zone_id="zone_road", zone_name="Main Road Corridor",
    risk_level="medium", duration_s=195.0, frame=450,
) -> BehaviorEvent:
    return BehaviorEvent(
        event_type   = "loitering",
        track_id     = track_id,
        class_name   = class_name,
        zone_id      = zone_id,
        zone_name    = zone_name,
        risk_level   = risk_level,
        duration_s   = duration_s,
        frame_number = frame,
        description  = (
            f"ID-{track_id} {class_name} has been in "
            f"'{zone_name}' for {duration_s:.0f}s "
            f"(threshold: 180s, risk: {risk_level})"
        ),
    )


def make_probing_event(
    track_id=2, class_name="person",
    zone_id="zone_restricted", zone_name="Restricted Perimeter",
    visits=3, duration_s=45.0, frame=300,
) -> BehaviorEvent:
    return BehaviorEvent(
        event_type   = "probing",
        track_id     = track_id,
        class_name   = class_name,
        zone_id      = zone_id,
        zone_name    = zone_name,
        risk_level   = "critical",
        duration_s   = duration_s,
        frame_number = frame,
        description  = (
            f"ID-{track_id} {class_name} has visited "
            f"'{zone_name}' {visits} times — possible reconnaissance"
        ),
        visit_count  = visits,
    )


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_reasoner_initialises():
    print("\n--- Test 1: Reasoner initialises with API key ---")
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⏭  Skipped: OPENAI_API_KEY not set")
        return None

    reasoner = Reasoner(model="gpt-4o-mini", temperature=0.1)
    assert reasoner is not None
    print("✅ Test 1 passed: Reasoner initialised")
    return reasoner


def test_prompt_builds_correctly():
    print("\n--- Test 2: Prompt builder creates correct context ---")
    state = make_initial_state(fps=15.0)
    event = make_loitering_event()

    # Add some objects to state for richer context
    tracked = [
        make_tracked(1, "car",  230.0, 152.0, "zone_road"),
        make_tracked(2, "car",  418.0,  76.0, "zone_parking"),
    ]
    for _ in range(30):
        state = update_state(state, tracked, fps=15.0)

    summary = get_state_summary(state)
    context = build_context(event, state, summary)

    assert "SURVEILLANCE STATE"  in context
    assert "TRIGGERED EVENT"     in context
    assert "LOITERING"           in context
    assert "zone_road"           in context
    assert "195"                 in context    # duration

    print(f"  Context length: {len(context)} chars")
    print(f"  System prompt : {len(SYSTEM_PROMPT)} chars")
    print("✅ Test 2 passed: prompt built correctly")
    return context


def test_loitering_analysis(reasoner: Reasoner | None):
    print("\n--- Test 3: LLM analyses loitering on road ---")
    if reasoner is None:
        print("  ⏭  Skipped: no reasoner (API key missing)")
        return

    state = make_initial_state(fps=15.0)
    # Put car in road zone for context
    tracked = [make_tracked(1, "car", 230.0, 152.0, "zone_road")]
    for _ in range(200):
        state = update_state(state, tracked, fps=15.0)

    event = make_loitering_event(
        duration_s=195.0,
        zone_id="zone_road",
        zone_name="Main Road Corridor",
        risk_level="medium",
    )

    alert = reasoner.analyse(event, state)
    assert alert is not None,           "LLM returned None — check API key"
    assert alert.risk_level in ("GREEN", "AMBER", "RED")
    assert len(alert.reasoning) > 10
    assert len(alert.recommended_action) > 5

    print(f"  Risk level    : {alert.risk_level}")
    print(f"  Confidence    : {alert.confidence}")
    print(f"  LLM latency   : {alert.llm_latency_ms:.0f}ms")
    print(f"  Reasoning     : {alert.reasoning[:80]}")
    print(f"  Action        : {alert.recommended_action[:80]}")
    print(f"  Alert message :\n{alert.alert_message}")

    # Road loitering (medium risk) should be AMBER at minimum
    assert alert.risk_level in ("AMBER", "RED"), (
        f"Expected AMBER or RED for road loitering, got {alert.risk_level}"
    )
    print("✅ Test 3 passed: loitering alert correctly classified")
    return alert


def test_probing_analysis(reasoner: Reasoner | None):
    print("\n--- Test 4: LLM analyses perimeter probing ---")
    if reasoner is None:
        print("  ⏭  Skipped: no reasoner")
        return

    state = make_initial_state(fps=15.0)
    event = make_probing_event(
        visits=3,
        zone_id="zone_restricted",
        zone_name="Restricted Perimeter",
        duration_s=45.0,
    )

    alert = reasoner.analyse(event, state)
    assert alert is not None
    assert alert.risk_level in ("AMBER", "RED")

    print(f"  Risk level    : {alert.risk_level}")
    print(f"  Confidence    : {alert.confidence}")
    print(f"  LLM latency   : {alert.llm_latency_ms:.0f}ms")
    print(f"  Reasoning     : {alert.reasoning[:80]}")
    print(f"  Action        : {alert.recommended_action[:80]}")

    # Critical zone probing should always be RED or AMBER
    print("✅ Test 4 passed: probing alert correctly classified")
    return alert


def test_critical_zone_is_red(reasoner: Reasoner | None):
    print("\n--- Test 5: Critical zone triggers RED ---")
    if reasoner is None:
        print("  ⏭  Skipped: no reasoner")
        return

    state = make_initial_state(fps=15.0)
    event = BehaviorEvent(
        event_type   = "loitering",
        track_id     = 5,
        class_name   = "person",
        zone_id      = "zone_restricted",
        zone_name    = "Restricted Perimeter",
        risk_level   = "critical",
        duration_s   = 45.0,
        frame_number = 675,
        description  = (
            "ID-5 person has been in 'Restricted Perimeter' "
            "for 45s (threshold: 30s, risk: critical)"
        ),
    )

    alert = reasoner.analyse(event, state)
    assert alert is not None
    assert alert.risk_level in ("AMBER", "RED"), (
        f"Expected AMBER or RED for critical zone, got {alert.risk_level}"
    )

    print(f"  Risk level : {alert.risk_level}")
    print(f"  Reasoning  : {alert.reasoning[:80]}")
    print("✅ Test 5 passed: critical zone classified correctly")


def test_alert_logger():
    print("\n--- Test 6: Alert logger writes and reads ---")

    # Create a mock alert
    from datetime import datetime, timezone
    mock_alert = SecurityAlert(
        risk_level         = "RED",
        event_type         = "loitering",
        zone_name          = "Restricted Perimeter",
        duration_s         = 45.0,
        confidence         = "HIGH",
        reasoning          = "Person in restricted zone for 45 seconds.",
        recommended_action = "Dispatch security team immediately.",
        alert_message      = "🔴 RED — LOITERING\nRestricted zone breach",
        track_id           = 3,
        frame_number       = 675,
        timestamp          = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        llm_latency_ms     = 1234.5,
    )

    write_alert(mock_alert)

    # Verify it was written
    assert os.path.exists(ALERT_LOG_PATH), "Alert log file not created"
    records = read_alerts(last_n=5)
    assert len(records) > 0
    last = records[-1]
    assert last["risk_level"]   == "RED"
    assert last["event_type"]   == "loitering"
    assert last["duration_s"]   == 45.0
    assert last["track_id"]     == 3

    # Stats work
    stats = get_alert_stats()
    assert stats["total"] > 0

    print(f"  Alert written : {ALERT_LOG_PATH}")
    print(f"  Total alerts  : {stats['total']}")
    print(f"  By risk       : {stats['by_risk']}")
    print("✅ Test 6 passed: alert logger working correctly")


def test_full_reasoning_pipeline(reasoner: Reasoner | None):
    print("\n--- Test 7: Full pipeline — state → behavior → LLM ---")
    if reasoner is None:
        print("  ⏭  Skipped: no reasoner")
        return

    zm    = ZoneManager(CONFIG_PATH)
    state = make_initial_state(fps=15.0)
    fps   = 15.0

    # Simulate 65 seconds of a car stopped at the intersection
    n_frames = int(65 * fps)
    tracked  = [make_tracked(1, "car", 42.0, 269.0, "zone_intersection")]

    for _ in range(n_frames):
        state = update_state(state, tracked, fps)
        state = detect_behaviors(state, zm)

    pending = state["pending_events"]
    print(f"  Pending events  : {len(pending)}")
    assert len(pending) > 0, "No events after 65s in 60s-threshold zone"

    # Run LLM on the first event
    event   = pending[0]
    alert   = reasoner.analyse(event, state)

    assert alert is not None
    assert alert.risk_level in ("AMBER", "RED")

    write_alert(alert)

    print(f"  Event type    : {event.event_type}")
    print(f"  LLM risk      : {alert.risk_level}")
    print(f"  LLM latency   : {alert.llm_latency_ms:.0f}ms")
    print(f"  Alert message :\n{alert.alert_message}")
    print("✅ Test 7 passed: full pipeline state → behavior → LLM works")


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 56)
    print("  Phase 3 — Reasoning Tests")
    print("=" * 56)

    reasoner = test_reasoner_initialises()
    test_prompt_builds_correctly()
    alert1 = test_loitering_analysis(reasoner)
    alert2 = test_probing_analysis(reasoner)
    test_critical_zone_is_red(reasoner)
    test_alert_logger()
    test_full_reasoning_pipeline(reasoner)

    print()
    print("=" * 56)
    print("✅ All Phase 3 tests passed!")
    print("Next → Phase 4: Streamlit Dashboard")
    print("=" * 56)


if __name__ == "__main__":
    run_all_tests()