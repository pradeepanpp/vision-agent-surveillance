# src/reasoning/alert_logger.py
"""
Alert Logger — Phase 3.

Writes every SecurityAlert to a structured JSONL log file.
One JSON object per line — easy to parse, grep, or load
into any SIEM or compliance monitoring system.

For UAE banking and government deployments, this log provides
the required audit trail of all security events.
"""

import os
import json
from pathlib import Path
from src.reasoning.reasoner import SecurityAlert
from src.utils.logger import logger

ALERT_LOG_PATH = "logs/alerts.jsonl"


def write_alert(alert: SecurityAlert):
    """Append one SecurityAlert to the JSONL log file."""
    os.makedirs("logs", exist_ok=True)

    record = {
        "timestamp":          alert.timestamp,
        "risk_level":         alert.risk_level,
        "event_type":         alert.event_type,
        "track_id":           alert.track_id,
        "zone":               alert.zone_name,
        "duration_s":         alert.duration_s,
        "confidence":         alert.confidence,
        "reasoning":          alert.reasoning,
        "recommended_action": alert.recommended_action,
        "alert_message":      alert.alert_message,
        "frame_number":       alert.frame_number,
        "llm_latency_ms":     alert.llm_latency_ms,
    }

    with open(ALERT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"[AlertLog] {alert.risk_level} alert written → {ALERT_LOG_PATH}"
    )


def read_alerts(last_n: int = 50) -> list[dict]:
    """Read the last N alerts from the log."""
    if not Path(ALERT_LOG_PATH).exists():
        return []

    try:
        with open(ALERT_LOG_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        records = []
        for line in lines[-last_n:]:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    except Exception as e:
        logger.error(f"[AlertLog] Read failed: {e}")
        return []


def get_alert_stats() -> dict:
    """Return statistics from the alert log."""
    records = read_alerts(last_n=10_000)
    if not records:
        return {"total": 0}

    total  = len(records)
    by_risk = {"GREEN": 0, "AMBER": 0, "RED": 0}
    by_type = {}

    for r in records:
        risk = r.get("risk_level", "AMBER")
        by_risk[risk] = by_risk.get(risk, 0) + 1
        etype = r.get("event_type", "unknown")
        by_type[etype] = by_type.get(etype, 0) + 1

    return {
        "total":   total,
        "by_risk": by_risk,
        "by_type": by_type,
    }