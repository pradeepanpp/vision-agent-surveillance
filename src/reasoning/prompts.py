# src/reasoning/prompts.py
"""
UAE Security Analyst Prompts — Phase 3.

Domain-specific prompts for the LLM behavioral reasoner.
Written to reflect UAE critical infrastructure security doctrine.

The LLM acts as a Tier-2 Security Analyst who:
  - Reads the surveillance state summary
  - Evaluates the triggered behavioral event
  - Classifies risk level (GREEN / AMBER / RED)
  - Generates a structured natural language report
  - Recommends a specific action

Why domain-specific prompts matter:
  A generic "you are a security guard" prompt produces generic output.
  A UAE-specific prompt produces output that matches real deployment
  context — EDGE Group, Dubai Police, smart city infrastructure.
  This specificity is what makes the system look production-ready.
"""

from src.state.schema import BehaviorEvent, SurveillanceState


# ─────────────────────────────────────────────
# SYSTEM PROMPT — fixed across all calls
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Tier-2 Security Analyst for UAE Critical Infrastructure.

You monitor drone surveillance feeds across facilities operated by
organizations such as EDGE Group, Dubai Police, and Derq Smart Cities.

Your risk framework:
  GREEN  → Normal activity. Log and continue monitoring.
  AMBER  → Suspicious pattern. Escalate for human review.
  RED    → Immediate security threat. Dispatch response team now.

Risk thresholds by zone:
  zone_parking    (LOW)      → vehicles expected, very long presence needed
  zone_road       (MEDIUM)   → moving traffic expected, stationary = suspicious
  zone_intersection (HIGH)   → brief stops only, loitering = threat
  zone_restricted  (CRITICAL)→ zero tolerance, any presence = immediate RED

Your response must always be valid JSON with exactly these fields:
{
  "risk_level": "GREEN" or "AMBER" or "RED",
  "event_type": the type of behavioral event,
  "subject": brief description of the object,
  "zone": zone name,
  "duration_seconds": numeric,
  "confidence": "LOW", "MEDIUM", or "HIGH",
  "reasoning": one clear sentence explaining your assessment,
  "recommended_action": specific action for the security team,
  "alert_message": the full alert text to display on the dashboard
}

Return ONLY the JSON object. No preamble. No markdown. No explanation."""


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────

def build_context(
    event:   BehaviorEvent,
    state:   SurveillanceState,
    summary: str,
) -> str:
    """
    Build the user message sent to the LLM for one event.

    Combines:
      - Current scene state (what the system has observed)
      - The specific behavioral event that triggered reasoning
      - Structured instruction for the expected output format
    """
    event_desc = _format_event(event)
    return f"""SURVEILLANCE STATE (last 60 seconds):
{summary}

TRIGGERED EVENT:
{event_desc}

Analyse this event in the context of the full scene state above.
Consider: Is this isolated or part of a pattern?
Respond with a JSON security assessment."""


def _format_event(event: BehaviorEvent) -> str:
    """Format a BehaviorEvent as a readable event description."""
    if event.event_type == "loitering":
        return (
            f"Type    : LOITERING\n"
            f"Object  : {event.class_name} (Track ID {event.track_id})\n"
            f"Zone    : {event.zone_name} [{event.zone_id}]\n"
            f"Risk    : {event.risk_level.upper()}\n"
            f"Duration: {event.duration_s:.0f} seconds\n"
            f"Details : {event.description}"
        )
    elif event.event_type == "probing":
        return (
            f"Type    : PERIMETER PROBING\n"
            f"Object  : {event.class_name} (Track ID {event.track_id})\n"
            f"Zone    : {event.zone_name} [{event.zone_id}]\n"
            f"Risk    : {event.risk_level.upper()}\n"
            f"Visits  : {event.visit_count} separate entries\n"
            f"Duration: {event.duration_s:.0f} seconds total\n"
            f"Details : {event.description}"
        )
    elif event.event_type == "crowd_formation":
        return (
            f"Type    : CROWD FORMATION\n"
            f"Zone    : {event.zone_name} [{event.zone_id}]\n"
            f"Risk    : {event.risk_level.upper()}\n"
            f"Count   : {event.person_count} persons\n"
            f"Details : {event.description}"
        )
    else:
        return f"Type: {event.event_type}\n{event.description}"


# ─────────────────────────────────────────────
# ALERT MESSAGE TEMPLATES
# ─────────────────────────────────────────────

def format_dashboard_alert(assessment: dict, event: BehaviorEvent) -> str:
    """
    Format the LLM assessment as a dashboard-ready alert string.
    Used when LLM response parsing fails — fallback template.
    """
    risk    = assessment.get("risk_level", "AMBER")
    zone    = assessment.get("zone", event.zone_name)
    action  = assessment.get("recommended_action", "Monitor situation")
    reason  = assessment.get("reasoning", event.description)
    dur     = assessment.get("duration_seconds", event.duration_s)

    icons = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}
    icon  = icons.get(risk, "🟡")

    return (
        f"{icon} {risk} — {event.event_type.upper().replace('_', ' ')}\n"
        f"Zone: {zone} | Duration: {dur:.0f}s\n"
        f"Assessment: {reason}\n"
        f"Action: {action}"
    )