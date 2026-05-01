# src/state/schema.py
"""
LangGraph State Schema — Phase 2.

Defines the TypedDict that LangGraph uses as its persistent
state across every frame of the surveillance pipeline.

The state grows over time:
  - New objects are added when first detected
  - Existing objects are updated with position and dwell time
  - Objects that disappear are marked as lost (not deleted)
  - Events are appended when behavioral thresholds are crossed
"""

from typing import TypedDict, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# OBJECT RECORD
# ─────────────────────────────────────────────

@dataclass
class ObjectRecord:
    """
    Complete history of one tracked object across all frames.

    Attributes:
        track_id        : ByteTrack persistent integer ID
        class_name      : "car", "person", "truck", etc.
        first_seen_frame: frame number when first detected
        last_seen_frame : most recent frame number
        current_zone    : zone_id currently in (None if outside all)
        zone_history    : ordered list of zone_ids visited
        frame_counts    : {zone_id: frames spent there}
        trajectory      : list of (cx, cy) center positions (last 50)
        lost_frames     : consecutive frames without detection
        is_active       : False if lost for too long
        alerts_fired    : list of alert types already sent for this object
    """
    track_id:          int
    class_name:        str
    first_seen_frame:  int
    last_seen_frame:   int
    current_zone:      Optional[str]  = None
    zone_history:      list           = field(default_factory=list)
    frame_counts:      dict           = field(default_factory=dict)
    trajectory:        list           = field(default_factory=list)
    lost_frames:       int            = 0
    is_active:         bool           = True
    alerts_fired:      list           = field(default_factory=list)

    def dwell_seconds(self, zone_id: str, fps: float = 15.0) -> float:
        """Return seconds this object spent in a specific zone."""
        frames = self.frame_counts.get(zone_id, 0)
        return round(frames / fps, 1) if fps > 0 else 0.0

    def total_seconds(self, fps: float = 15.0) -> float:
        """Return total seconds tracked across all zones."""
        return round(sum(self.frame_counts.values()) / fps, 1)

    def zone_visit_count(self, zone_id: str) -> int:
        """Return number of separate visits to a specific zone."""
        visits  = 0
        in_zone = False
        for z in self.zone_history:
            if z == zone_id and not in_zone:
                visits  += 1
                in_zone  = True
            elif z != zone_id:
                in_zone  = False
        return visits


# ─────────────────────────────────────────────
# BEHAVIORAL EVENT
# ─────────────────────────────────────────────

@dataclass
class BehaviorEvent:
    """
    One detected behavioral anomaly ready for LLM reasoning.

    Created by behaviors.py when a threshold is crossed.
    Consumed by the LLM reasoner in Phase 3.
    """
    event_type:   str            # loitering | probing | crowd_formation
    track_id:     Optional[int]  # None for crowd events
    class_name:   str
    zone_id:      str
    zone_name:    str
    risk_level:   str
    duration_s:   float
    frame_number: int
    description:  str
    visit_count:  int  = 0       # for probing events
    person_count: int  = 0       # for crowd events


# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────

class SurveillanceState(TypedDict):
    """
    Complete LangGraph state for one surveillance session.

    LangGraph persists this across every node execution.
    Every frame updates this state. The LLM reads from it.
    """
    frame_number:     int
    fps:              float
    object_log:       dict   # track_id → ObjectRecord
    pending_events:   list   # BehaviorEvent items waiting for LLM
    processed_events: list   # BehaviorEvent items already handled
    alert_count:      int
    session_start:    int
    zone_population:  dict   # zone_id → current object count


def make_initial_state(fps: float = 15.0) -> SurveillanceState:
    """Create a fresh state for a new surveillance session."""
    return SurveillanceState(
        frame_number     = 0,
        fps              = fps,
        object_log       = {},
        pending_events   = [],
        processed_events = [],
        alert_count      = 0,
        session_start    = 0,
        zone_population  = {},
    )
