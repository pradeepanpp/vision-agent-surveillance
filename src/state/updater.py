# src/state/updater.py
"""
State Updater — Phase 2.

Runs on every frame. Takes the list of tracked objects from
Phase 1 and merges them into the LangGraph state.

What it does each frame:
  1. Increments frame counter
  2. For each tracked object:
     - Creates a new ObjectRecord if first seen
     - Updates position, zone, dwell time if already known
     - Handles gaps (object temporarily not detected)
  3. Marks objects as lost if not seen for > MAX_LOST_FRAMES
  4. Updates zone population counts
  5. Returns updated state

The updater never calls the LLM — it is pure Python and runs
on every single frame at near-zero cost.
"""

from src.state.schema import (
    SurveillanceState,
    ObjectRecord,
    BehaviorEvent,
)
from src.detection.tracker import TrackedObject
from src.utils.logger import logger

# Object is considered lost after this many consecutive missed frames
MAX_LOST_FRAMES = 45   # ~3 seconds at 15fps


def update_state(
    state:   SurveillanceState,
    tracked: list[TrackedObject],
    fps:     float = 15.0,
) -> SurveillanceState:
    """
    Update surveillance state with new frame detections.

    Args:
        state   : current LangGraph state
        tracked : list of TrackedObject from Phase 1
        fps     : video frame rate for time calculations

    Returns:
        Updated state dictionary.
    """
    frame_num = state["frame_number"] + 1
    obj_log   = state["object_log"]

    # Build a set of track IDs seen in this frame
    seen_ids = {obj.track_id for obj in tracked}

    # ── Update seen objects ───────────────────
    for obj in tracked:
        tid = obj.track_id

        if tid not in obj_log:
            # First time we see this object — create record
            record = ObjectRecord(
                track_id         = tid,
                class_name       = obj.class_name,
                first_seen_frame = frame_num,
                last_seen_frame  = frame_num,
                current_zone     = obj.zone,
            )
            obj_log[tid] = record
            logger.debug(
                f"[State] New object ID-{tid} ({obj.class_name}) "
                f"in {obj.zone or 'no zone'}"
            )
        else:
            record = obj_log[tid]

        # Reset lost counter — object is visible
        record.lost_frames    = 0
        record.is_active      = True
        record.last_seen_frame = frame_num

        # Update trajectory (keep last 100 positions)
        record.trajectory.append(obj.center)
        if len(record.trajectory) > 100:
            record.trajectory = record.trajectory[-100:]

        # Update zone and dwell time
        if obj.zone:
            # Append to zone history only on zone change
            if not record.zone_history or record.zone_history[-1] != obj.zone:
                record.zone_history.append(obj.zone)

            # Increment frame count for this zone
            record.frame_counts[obj.zone] = \
                record.frame_counts.get(obj.zone, 0) + 1
            
        else:
            if record.zone_history and record.zone_history[-1] is not None:
               record.zone_history.append(None)

        record.current_zone = obj.zone

    # ── Handle missing objects ────────────────
    for tid, record in obj_log.items():
        if tid not in seen_ids and record.is_active:
            record.lost_frames += 1

            if record.lost_frames > MAX_LOST_FRAMES:
                record.is_active = True   # keep record but note it is lost
                logger.debug(
                    f"[State] ID-{tid} lost for "
                    f"{record.lost_frames} frames"
                )

    # ── Update zone population ────────────────
    zone_pop: dict[str, int] = {}
    for obj in tracked:
        if obj.zone:
            zone_pop[obj.zone] = zone_pop.get(obj.zone, 0) + 1

    # ── Return updated state ──────────────────
    return SurveillanceState(
        frame_number     = frame_num,
        fps              = fps,
        object_log       = obj_log,
        pending_events   = state["pending_events"],
        processed_events = state["processed_events"],
        alert_count      = state["alert_count"],
        session_start    = state["session_start"],
        zone_population  = zone_pop,
    )


def get_state_summary(state: SurveillanceState) -> str:
    """
    Build a concise text summary of the current state.
    This is what gets passed to the LLM in Phase 3.

    Returns a string like:
        Frame 450 | Active objects: 3
        ID-1 car  | zone_road      | 30.0s in zone | MEDIUM
        ID-2 car  | zone_parking   | 12.0s in zone | LOW
        ID-5 person | zone_restricted | 5.0s in zone | CRITICAL ⚠️
    """
    fps        = state["fps"]
    frame_num  = state["frame_number"]
    active     = [
        r for r in state["object_log"].values()
        if r.is_active and r.lost_frames < MAX_LOST_FRAMES
    ]

    lines = [
        f"Frame {frame_num} | "
        f"Active objects: {len(active)} | "
        f"Alerts fired: {state['alert_count']}"
    ]

    for rec in sorted(active, key=lambda r: r.track_id):
        zone     = rec.current_zone or "no zone"
        dwell_s  = rec.dwell_seconds(zone, fps) if rec.current_zone else 0.0
        total_s  = rec.total_seconds(fps)
        lines.append(
            f"  ID-{rec.track_id} {rec.class_name:<8} | "
            f"{zone:<20} | "
            f"dwell={dwell_s:.1f}s | "
            f"total={total_s:.1f}s"
        )

    return "\n".join(lines)