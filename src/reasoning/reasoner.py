# src/reasoning/reasoner.py
"""
LLM Behavioral Reasoner — Phase 3.

Event-triggered GPT-4o-mini reasoning for detected anomalies.

Key architectural decision — tiered processing:
  Tier 1 (every frame)  : YOLO26s detection → ~30ms, no LLM
  Tier 2 (every frame)  : LangGraph state update → ~1ms, no LLM
  Tier 3 (events only)  : LLM reasoning → ~1-2s, only on threshold breach

This means the LLM runs 2-10 times per hour in a typical deployment,
not 30 times per second. Total cost: ~$0.01/hour.

The reasoner:
  1. Takes a BehaviorEvent from state["pending_events"]
  2. Builds context from the current state summary
  3. Calls GPT-4o-mini with the UAE security system prompt
  4. Parses the structured JSON response
  5. Returns a formatted alert for the dashboard
  6. Moves the event from pending to processed
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

from src.state.schema   import BehaviorEvent, SurveillanceState
from src.state.updater  import get_state_summary
from src.reasoning.prompts import (
    SYSTEM_PROMPT,
    build_context,
    format_dashboard_alert,
)
from src.utils.logger import logger

load_dotenv()


# ─────────────────────────────────────────────
# ALERT DATACLASS
# ─────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class SecurityAlert:
    """
    One processed security alert from the LLM.

    This is what gets displayed on the dashboard
    and written to the alert log.
    """
    risk_level:         str      # GREEN / AMBER / RED
    event_type:         str      # loitering / probing / crowd_formation
    zone_name:          str
    duration_s:         float
    confidence:         str      # LOW / MEDIUM / HIGH
    reasoning:          str
    recommended_action: str
    alert_message:      str      # formatted for dashboard display
    track_id:           int | None
    frame_number:       int
    timestamp:          str
    llm_latency_ms:     float


# ─────────────────────────────────────────────
# REASONER
# ─────────────────────────────────────────────

class Reasoner:
    """
    GPT-4o-mini security analyst for behavioral events.

    Usage:
        reasoner = Reasoner()
        alert = reasoner.analyse(event, state)
        if alert:
            print(alert.alert_message)
            print(alert.risk_level)
    """

    def __init__(
        self,
        model:       str   = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens:  int   = 500,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found.\n"
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )

        self._client     = OpenAI(api_key=api_key)
        self._model      = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info(
            f"Reasoner ready — model={model} "
            f"temp={temperature}"
        )

    def analyse(
        self,
        event: BehaviorEvent,
        state: SurveillanceState,
    ) -> SecurityAlert | None:
        """
        Analyse one behavioral event and return a SecurityAlert.

        Calls GPT-4o-mini with the full state context.
        Returns None if the API call fails.

        Args:
            event : BehaviorEvent from state["pending_events"]
            state : full current SurveillanceState

        Returns:
            SecurityAlert ready for dashboard display, or None on error.
        """
        summary = get_state_summary(state)
        context = build_context(event, state, summary)

        logger.info(
            f"[Reasoner] Analysing {event.event_type} — "
            f"ID-{event.track_id} in {event.zone_id} "
            f"({event.duration_s:.0f}s)"
        )

        t0 = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model       = self._model,
                temperature = self._temperature,
                max_tokens  = self._max_tokens,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": context},
                ],
            )

            llm_latency_ms = (time.perf_counter() - t0) * 1000
            raw_text       = response.choices[0].message.content.strip()

            assessment = _parse_response(raw_text)

            alert = SecurityAlert(
                risk_level         = assessment.get("risk_level", "AMBER"),
                event_type         = event.event_type,
                zone_name          = assessment.get("zone", event.zone_name),
                duration_s         = event.duration_s,
                confidence         = assessment.get("confidence", "MEDIUM"),
                reasoning          = assessment.get("reasoning", ""),
                recommended_action = assessment.get("recommended_action", ""),
                alert_message      = assessment.get(
                    "alert_message",
                    format_dashboard_alert(assessment, event)
                ),
                track_id           = event.track_id,
                frame_number       = event.frame_number,
                timestamp          = _timestamp(),
                llm_latency_ms     = round(llm_latency_ms, 1),
            )

            logger.info(
                f"[Reasoner] → {alert.risk_level} | "
                f"{alert.confidence} confidence | "
                f"{llm_latency_ms:.0f}ms"
            )

            return alert

        except Exception as e:
            logger.error(f"[Reasoner] LLM call failed: {e}")
            return None

    def process_pending_events(
        self,
        state: SurveillanceState,
    ) -> tuple[SurveillanceState, list[SecurityAlert]]:
        """
        Process all events in state["pending_events"].

        Moves each event from pending → processed.
        Returns updated state and list of new SecurityAlerts.

        Call this from the main video loop after detect_behaviors().
        """
        pending = state["pending_events"]
        if not pending:
            return state, []

        alerts = []
        for event in pending:
            alert = self.analyse(event, state)
            if alert:
                alerts.append(alert)

        # Move all pending → processed
        updated_state = {
            **state,
            "pending_events":   [],
            "processed_events": state["processed_events"] + pending,
            "alert_count":      state["alert_count"] + len(alerts),
        }

        return updated_state, alerts


def _timestamp() -> str:
    """Return current UTC time as ISO string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_response(raw: str) -> dict:
    """
    Parse LLM JSON response safely.

    Handles:
      - Clean JSON
      - JSON wrapped in ```json ... ``` fences
      - Partial JSON with missing fields
    """
    text = raw.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text  = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("[Reasoner] Could not parse JSON response")
        logger.debug(f"Raw response: {raw[:200]}")
        return {
            "risk_level":         "AMBER",
            "confidence":         "LOW",
            "reasoning":          "Assessment unavailable — manual review required",
            "recommended_action": "Review footage manually",
            "alert_message":      f"⚠ MANUAL REVIEW REQUIRED\n{raw[:100]}",
        }