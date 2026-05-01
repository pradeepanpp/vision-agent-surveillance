# src/state/behaviors.py
"""
Behavioral Pattern Detector — Phase 2.

Reads the LangGraph state after each frame update and checks
for three behavioral anomalies:

  1. Loitering      — object in same zone longer than threshold
  2. Perimeter probe — object visits restricted zone repeatedly
  3. Crowd formation — person count rises rapidly in a zone

When a pattern is detected, it creates a BehaviorEvent and
adds it to state["pending_events"] for the LLM to process.

This is pure Python — no LLM calls, no API cost.
Runs in <1ms per frame.
"""

from src.state.schema import (
    SurveillanceState,
    ObjectRecord,
    BehaviorEvent,
)
from src.detection.zone_manager import ZoneManager
from src.utils.logger import logger


# ─────────────────────────────────────────────
# DETECTION CONFIG
# ─────────────────────────────────────────────

# Minimum seconds in a zone before loitering check
LOITERING_MIN_SECONDS = 10.0

# Probing: how many zone visits trigger an alert
PROBING_MIN_VISITS = 2

# Crowd: minimum persons in one zone to be a crowd event
CROWD_MIN_PERSONS = 3

# Do not re-alert for the same behaviour on the same object
# more often than this many frames
ALERT_COOLDOWN_FRAMES = 150   # ~10 seconds at 15fps


# ─────────────────────────────────────────────
# BEHAVIORAL CHECKS
# ─────────────────────────────────────────────

def check_loitering(
    record:    ObjectRecord,
    zone_mgr:  ZoneManager,
    fps:       float,
    frame_num: int,
) -> BehaviorEvent | None:
    """
    Detect if an object has been in one zone longer than allowed.

    Returns a BehaviorEvent if loitering detected, else None.
    """
    zone_id = record.current_zone
    if not zone_id:
        return None

    # Find zone config
    zone = next((z for z in zone_mgr.zones if z.zone_id == zone_id), None)
    if not zone:
        return None

    dwell_s   = record.dwell_seconds(zone_id, fps)
    threshold = zone.threshold_s

    # Not long enough yet
    if dwell_s < LOITERING_MIN_SECONDS:
        return None

    # Only alert if dwell exceeds threshold
    if dwell_s < threshold:
        return None

    # Cooldown check — avoid spamming same alert
    alert_key = f"loitering_{zone_id}"
    if alert_key in record.alerts_fired:
        return None

    record.alerts_fired.append(alert_key)

    desc = (
        f"ID-{record.track_id} {record.class_name} has been in "
        f"'{zone.name}' for {dwell_s:.0f}s "
        f"(threshold: {threshold}s, risk: {zone.risk_level})"
    )
    logger.warning(f"[Behavior] LOITERING detected: {desc}")

    return BehaviorEvent(
        event_type   = "loitering",
        track_id     = record.track_id,
        class_name   = record.class_name,
        zone_id      = zone_id,
        zone_name    = zone.name,
        risk_level   = zone.risk_level,
        duration_s   = dwell_s,
        frame_number = frame_num,
        description  = desc,
    )


def check_perimeter_probing(
    record:    ObjectRecord,
    zone_mgr:  ZoneManager,
    fps:       float,
    frame_num: int,
) -> BehaviorEvent | None:
    """
    Detect if an object visits a high/critical zone repeatedly.

    Probing = entering a restricted zone, leaving, then re-entering.
    This is the reconnaissance pattern.
    """
    # Only check high and critical zones
    high_risk_zones = [
        z for z in zone_mgr.zones
        if z.risk_level in ("high", "critical")
    ]

    for zone in high_risk_zones:
        visits    = record.zone_visit_count(zone.zone_id)
        alert_key = f"probing_{zone.zone_id}"

        if visits >= PROBING_MIN_VISITS and alert_key not in record.alerts_fired:
            record.alerts_fired.append(alert_key)
            dwell_s = record.dwell_seconds(zone.zone_id, fps)

            desc = (
                f"ID-{record.track_id} {record.class_name} has visited "
                f"'{zone.name}' {visits} times "
                f"(total dwell: {dwell_s:.0f}s) — possible reconnaissance"
            )
            logger.warning(f"[Behavior] PROBING detected: {desc}")

            return BehaviorEvent(
                event_type   = "probing",
                track_id     = record.track_id,
                class_name   = record.class_name,
                zone_id      = zone.zone_id,
                zone_name    = zone.name,
                risk_level   = zone.risk_level,
                duration_s   = dwell_s,
                frame_number = frame_num,
                description  = desc,
                visit_count  = visits,
            )

    return None


def check_crowd_formation(
    state:     SurveillanceState,
    zone_mgr:  ZoneManager,
    frame_num: int,
) -> list[BehaviorEvent]:
    """
    Detect rapid accumulation of persons in any zone.

    Crowd formation = CROWD_MIN_PERSONS or more people
    in the same zone at the same time.
    """
    events = []

    for zone in zone_mgr.zones:
        # Count persons currently in this zone
        person_count = sum(
            1 for rec in state["object_log"].values()
            if rec.is_active
            and rec.current_zone == zone.zone_id
            and rec.class_name == "person"
        )

        if person_count < CROWD_MIN_PERSONS:
            continue

        # Check if we already alerted for this zone at this count
        alert_key = f"crowd_{zone.zone_id}_{person_count}"
        already_fired = any(
            alert_key in (e.description or "")
            for e in state["processed_events"]
        )
        if already_fired:
            continue

        desc = (
            f"{person_count} persons detected simultaneously "
            f"in '{zone.name}' ({zone.risk_level} zone) — "
            f"possible crowd formation"
        )
        logger.warning(f"[Behavior] CROWD FORMATION detected: {desc}")

        events.append(BehaviorEvent(
            event_type   = "crowd_formation",
            track_id     = None,
            class_name   = "person",
            zone_id      = zone.zone_id,
            zone_name    = zone.name,
            risk_level   = zone.risk_level,
            duration_s   = 0.0,
            frame_number = frame_num,
            description  = f"{alert_key}: {desc}",
            person_count = person_count,
        ))

    return events


# ─────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────

def detect_behaviors(
    state:    SurveillanceState,
    zone_mgr: ZoneManager,
) -> SurveillanceState:
    """
    Run all behavioral checks on the current state.

    Adds any detected BehaviorEvents to state["pending_events"].
    Returns updated state.

    Called once per frame, after state updater runs.
    """
    fps        = state["fps"]
    frame_num  = state["frame_number"]
    new_events = []

    # Check each active object
    for record in state["object_log"].values():
        if not record.is_active:
            continue

        # Loitering check
        event = check_loitering(record, zone_mgr, fps, frame_num)
        if event:
            new_events.append(event)

        # Probing check
        event = check_perimeter_probing(record, zone_mgr, fps, frame_num)
        if event:
            new_events.append(event)

    # Crowd formation check (scene-level, not per-object)
    crowd_events = check_crowd_formation(state, zone_mgr, frame_num)
    new_events.extend(crowd_events)

    if new_events:
        logger.info(
            f"[Behavior] Frame {frame_num}: "
            f"{len(new_events)} event(s) queued for LLM"
        )

    return SurveillanceState(
        frame_number     = state["frame_number"],
        fps              = state["fps"],
        object_log       = state["object_log"],
        pending_events   = state["pending_events"] + new_events,
        processed_events = state["processed_events"],
        alert_count      = state["alert_count"],
        session_start    = state["session_start"],
        zone_population  = state["zone_population"],
    )