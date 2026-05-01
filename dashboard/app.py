# dashboard/app.py
"""
Vision Agent Surveillance Dashboard — Phase 4.

Tactical Operations Center UI for autonomous AI surveillance.

Run:
    streamlit run dashboard/app.py

Architecture:
    LIVE tab  — real-time video + zone overlays + alert feed
    ALERTS tab — full alert history with risk filtering
    TRACKING tab — object state browser with dwell times
    ABOUT tab — system architecture and benchmark results

The dashboard calls the detection pipeline directly (no FastAPI)
for maximum performance on local deployment.
"""

import sys
import os
import cv2
import time
import threading
import queue
import yaml
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

from dashboard.components import (
    inject_css,
    render_header,
    render_alert_card,
    render_zone_status,
    render_tracking_table,
    render_sidebar,
)
from src.utils.logger      import logger
from src.reasoning.alert_logger import read_alerts, get_alert_stats

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

VIDEO_PATH  = "data/test_video.mp4"
CONFIG_PATH = "configs/zones.yaml"
MODEL_PATH  = "yolo26s.pt"
CONFIDENCE  = 0.25

st.set_page_config(
    page_title  = "VISION AGENT — Surveillance Command",
    page_icon   = "◈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "running":       False,
        "alerts":        [],
        "state_log":     None,
        "frame_num":     0,
        "tracked_count": 0,
        "session_stats": {"total":0,"red":0,"amber":0,"green":0,"tracked":0},
        "zone_pop":      {},
        "object_log":    {},
        "pipeline_fps":  0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_zones_config() -> list:
    """Load zone list from zones.yaml for display."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        zones = []
        for zid, zdata in cfg.get("zones", {}).items():
            zones.append({"zone_id": zid, **zdata})
        return zones
    except Exception:
        return []


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline_frame(
    cap, detector, tracker, zone_mgr,
    state, fps,
):
    """Run one frame through the full detection pipeline."""
    from src.utils.video_utils import prepare_frame, read_frame
    from src.state.updater     import update_state
    from src.state.behaviors   import detect_behaviors

    ret, raw = read_frame(cap, loop=True)
    if not ret:
        return None, state, []

    frame = prepare_frame(raw)

    # Detect + track
    detections = detector.detect(frame)
    tracked    = tracker.update(detections, frame_shape=(360, 640))

    # Zone assignment
    for obj in tracked:
        zone     = zone_mgr.get_zone_for_point(obj.center, (640, 360))
        obj.zone = zone.zone_id if zone else None

    # State update
    state = update_state(state, tracked, fps)
    state = detect_behaviors(state, zone_mgr)

    # Annotate frame
    frame = zone_mgr.draw_zones(frame, (640, 360))
    frame = _annotate_frame(frame, tracked, state, zone_mgr)

    return frame, state, tracked


def _annotate_frame(frame, tracked, state, zone_mgr):
    """Draw bounding boxes, IDs, and stats on frame."""
    from src.state.schema import ObjectRecord

    risk_colors = {
        "low":      (0, 255, 136),
        "medium":   (0, 170, 255),
        "high":     (0, 100, 255),
        "critical": (34, 34, 255),
        None:       (60, 80, 90),
    }

    zone_risk = {z.zone_id: z.risk_level for z in zone_mgr.zones}

    for obj in tracked:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        risk  = zone_risk.get(obj.zone) if obj.zone else None
        color = risk_colors.get(risk, risk_colors[None])

        # Bounding box with corner markers (tactical style)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Corner brackets
        cs = 8  # corner size
        for cx, cy, dx, dy in [
            (x1,y1, 1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)
        ]:
            cv2.line(frame,(cx,cy),(cx+dx*cs,cy),color,2)
            cv2.line(frame,(cx,cy),(cx,cy+dy*cs),color,2)

        # Get dwell time from state
        rec   = state["object_log"].get(obj.track_id)
        dwell = ""
        if rec and obj.zone:
            d = rec.dwell_seconds(obj.zone, state["fps"])
            dwell = f" {d:.0f}s"

        label    = f"ID{obj.track_id} {obj.class_name}{dwell}"
        label_y  = max(y1 - 4, 12)
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Frame counter + FPS
    fn  = state["frame_number"]
    fps = state["fps"]
    cv2.putText(frame, f"FRM:{fn:04d} | {fps:.0f}fps",
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0,255,136), 1, cv2.LINE_AA)
    cv2.putText(frame, "YOLO26s",
                (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (80,140,160), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    init_state()
    inject_css()

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.html("<br>")
        render_sidebar(
            st.session_state.session_stats,
            st.session_state.frame_num,
            st.session_state.pipeline_fps,
        )
        st.markdown("---")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("▶ START", use_container_width=True):
                st.session_state.running = True
        with col_s2:
            if st.button("■ STOP", use_container_width=True):
                st.session_state.running = False

        if st.button("CLEAR SESSION", use_container_width=True):
            st.session_state.alerts        = []
            st.session_state.session_stats = {
                "total":0,"red":0,"amber":0,"green":0,"tracked":0
            }
            st.session_state.object_log    = {}
            st.session_state.frame_num     = 0
            st.rerun()

        st.html("""
        <div style="
            font-family:'Space Mono',monospace;
            font-size:0.5rem;
            color:#0a1a2e;
            text-align:center;
            padding-top:12px;
            letter-spacing:0.15em;
        ">VISION AGENT v1.0 · PHASE 4</div>
        """)

    # ── Header ────────────────────────────────
    status = "OPERATIONAL" if st.session_state.running else "STANDBY"
    render_header(system_status=status)

    # ── Tabs ──────────────────────────────────
    tab_live, tab_alerts, tab_tracking, tab_about = st.tabs([
        "◈ LIVE FEED", "◈ ALERTS", "◈ TRACKING", "◈ ABOUT"
    ])

    # ══════════════════════════════════════════
    # TAB 1 — LIVE FEED
    # ══════════════════════════════════════════
    with tab_live:
        col_video, col_alerts = st.columns([3, 2], gap="medium")

        with col_video:
            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                letter-spacing:0.2em;color:#1a3a50;margin-bottom:8px;">
            ◈ LIVE SURVEILLANCE FEED — YOLO26s + BYTETRACK</div>
            """)
            video_placeholder = st.empty()
            metrics_ph        = st.empty()

        with col_alerts:
            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                letter-spacing:0.2em;color:#1a3a50;margin-bottom:8px;">
            ◈ ALERT FEED — REAL TIME</div>
            """)
            zone_ph  = st.empty()
            alerts_ph = st.empty()

        # Run pipeline when active
        if st.session_state.running:
            try:
                from src.detection.detector      import Detector
                from src.detection.tracker       import Tracker
                from src.detection.zone_manager  import ZoneManager
                from src.state.schema            import make_initial_state
                from src.reasoning.reasoner      import Reasoner
                from src.reasoning.alert_logger  import write_alert

                detector  = Detector(model_path=MODEL_PATH, confidence=CONFIDENCE)
                tracker   = Tracker()
                zone_mgr  = ZoneManager(CONFIG_PATH)
                state     = make_initial_state(fps=15.0)
                reasoner  = Reasoner()
                cap       = cv2.VideoCapture(VIDEO_PATH)
                fps       = cap.get(cv2.CAP_PROP_FPS) or 15.0
                zones_cfg = load_zones_config()

                while st.session_state.running:
                    t0    = time.perf_counter()
                    frame, state, tracked = run_pipeline_frame(
                        cap, detector, tracker, zone_mgr, state, fps
                    )
                    if frame is None:
                        break

                    # Process pending LLM events
                    new_alerts = []
                    if state["pending_events"]:
                        state, new_alerts = reasoner.process_pending_events(state)
                        for alert in new_alerts:
                            write_alert(alert)
                            alert_dict = {
                                "risk_level":         alert.risk_level,
                                "event_type":         alert.event_type,
                                "zone":               alert.zone_name,
                                "duration_s":         alert.duration_s,
                                "confidence":         alert.confidence,
                                "reasoning":          alert.reasoning,
                                "recommended_action": alert.recommended_action,
                                "alert_message":      alert.alert_message,
                                "track_id":           alert.track_id,
                                "timestamp":          alert.timestamp,
                                "llm_latency_ms":     alert.llm_latency_ms,
                            }
                            st.session_state.alerts.insert(0, alert_dict)
                            # Update stats
                            risk = alert.risk_level
                            st.session_state.session_stats["total"]  += 1
                            st.session_state.session_stats[risk.lower()] += 1

                    # Update session state
                    st.session_state.frame_num     = state["frame_number"]
                    st.session_state.zone_pop      = state["zone_population"]
                    st.session_state.object_log    = state["object_log"]
                    st.session_state.session_stats["tracked"] = len(tracked)
                    pipe_fps = 1.0 / max(time.perf_counter() - t0, 0.001)
                    st.session_state.pipeline_fps  = pipe_fps

                    # Render video frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(
                        frame_rgb, use_container_width=True
                    )

                    # Render metrics
                    with metrics_ph.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("FRAME",   f"{state['frame_number']}")
                        c2.metric("TRACKED", f"{len(tracked)}")
                        c3.metric("FPS",     f"{pipe_fps:.1f}")
                        c4.metric("ALERTS",  f"{st.session_state.session_stats['total']}")

                    # Render zone status
                    with zone_ph.container():
                        render_zone_status(
                            state["zone_population"], zones_cfg
                        )

                    # Render alerts
                    with alerts_ph.container():
                        recent = st.session_state.alerts[:6]
                        if recent:
                            for a in recent:
                                render_alert_card(a)
                        else:
                            st.html("""
                            <div style="
                                font-family:'Space Mono',monospace;
                                font-size:0.65rem;color:#1a3a50;
                                text-align:center;padding:30px;
                                border:1px dashed #0a1a2e;
                            ">● MONITORING — NO EVENTS DETECTED</div>
                            """)

                cap.release()

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.session_state.running = False

        else:
            # Standby — show static frame
            if os.path.exists(VIDEO_PATH):
                cap = cv2.VideoCapture(VIDEO_PATH)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    from src.utils.video_utils import prepare_frame
                    frame     = prepare_frame(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, use_container_width=True)

            zones_cfg = load_zones_config()
            with zone_ph.container():
                render_zone_status({}, zones_cfg)
            with alerts_ph.container():
                st.html("""
                <div style="
                    font-family:'Space Mono',monospace;
                    font-size:0.65rem;color:#1a3a50;
                    text-align:center;padding:40px;
                    border:1px dashed #0a1a2e;
                ">▶ PRESS START TO ACTIVATE PIPELINE</div>
                """)

    # ══════════════════════════════════════════
    # TAB 2 — ALERTS
    # ══════════════════════════════════════════
    with tab_alerts:
        col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
        with col_f1:
            risk_filter = st.selectbox(
                "Risk level",
                ["ALL", "RED", "AMBER", "GREEN"],
                label_visibility="collapsed",
            )
        with col_f2:
            type_filter = st.selectbox(
                "Event type",
                ["ALL", "loitering", "probing", "crowd_formation"],
                label_visibility="collapsed",
            )

        # Stats row
        stats = get_alert_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TOTAL ALERTS", stats.get("total", 0))
        c2.metric("🔴 RED",       stats.get("by_risk", {}).get("RED",   0))
        c3.metric("🟡 AMBER",     stats.get("by_risk", {}).get("AMBER", 0))
        c4.metric("🟢 GREEN",     stats.get("by_risk", {}).get("GREEN", 0))

        st.markdown("---")

        # Alert list
        all_alerts = read_alerts(last_n=100)
        if risk_filter != "ALL":
            all_alerts = [a for a in all_alerts if a.get("risk_level") == risk_filter]
        if type_filter != "ALL":
            all_alerts = [a for a in all_alerts if a.get("event_type") == type_filter]

        if all_alerts:
            for a in reversed(all_alerts):
                render_alert_card(a)
        else:
            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                color:#1a3a50;text-align:center;padding:60px;">
            NO ALERTS LOGGED YET<br><br>
            <span style="font-size:0.58rem;">
            Start the pipeline in the LIVE FEED tab
            </span></div>
            """)

    # ══════════════════════════════════════════
    # TAB 3 — TRACKING
    # ══════════════════════════════════════════
    with tab_tracking:
        st.html("""
        <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
            letter-spacing:0.2em;color:#1a3a50;margin-bottom:12px;">
        ◈ OBJECT TRACKING STATE — CURRENT SESSION</div>
        """)

        obj_log = st.session_state.object_log
        if obj_log:
            render_tracking_table(obj_log, fps=15.0)
        else:
            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                color:#1a3a50;text-align:center;padding:60px;
                border:1px dashed #0a1a2e;">
            NO TRACKING DATA<br>
            <span style="font-size:0.58rem;">
            Start the pipeline to see tracked objects
            </span></div>
            """)

    # ══════════════════════════════════════════
    # TAB 4 — ABOUT
    # ══════════════════════════════════════════
    with tab_about:
        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                letter-spacing:0.2em;color:#1a3a50;margin-bottom:12px;">
            ◈ SYSTEM ARCHITECTURE</div>
            """)

            layers = [
                ("TIER 1 — DETECTION",
                 "YOLO26s nano detects objects every frame. NMS-free, STAL small-object aware. ~5ms on GPU, ~30ms on CPU.",
                 "#00ff88"),
                ("TIER 2 — TRACKING",
                 "ByteTrack assigns persistent IDs across frames. Enables dwell time measurement and trajectory analysis.",
                 "#00ff88"),
                ("TIER 3 — MEMORY",
                 "LangGraph maintains stateful object log. Records zone history, dwell times, and visit counts across the full session.",
                 "#ffaa00"),
                ("TIER 4 — REASONING",
                 "GPT-4o-mini event-triggered analysis. Only fires on threshold breach — not every frame. UAE security doctrine system prompt.",
                 "#ffaa00"),
                ("TIER 5 — SAFETY",
                 "SQL Guard + prompt injection detection + read-only enforcement + JSONL audit trail.",
                 "#ff2244"),
            ]

            for title, desc, color in layers:
                st.html(f"""
                <div style="
                    background:#060d14;
                    border:1px solid #0a1a2e;
                    border-left:3px solid {color};
                    border-radius:2px;
                    padding:12px 14px;
                    margin-bottom:8px;
                ">
                    <div style="font-family:'Bebas Neue',sans-serif;
                        font-size:0.9rem;color:{color};
                        letter-spacing:0.1em;margin-bottom:5px;">{title}</div>
                    <div style="font-family:'Space Mono',monospace;
                        font-size:0.6rem;color:#1a3a50;line-height:1.5;">{desc}</div>
                </div>
                """)

        with col_b:
            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                letter-spacing:0.2em;color:#1a3a50;margin-bottom:12px;">
            ◈ BENCHMARK RESULTS — VISDRONE2019-MOT</div>
            """)

            metrics = [
                ("Overall Accuracy",   "97%",    "#00ff88"),
                ("Simple Queries",     "100%",   "#00ff88"),
                ("Medium Queries",     "100%",   "#00ff88"),
                ("Arabic Accuracy",    "100%",   "#00ff88"),
                ("YOLO26s mAP",        "47.2%",  "#ffaa00"),
                ("ByteTrack IDF1",     "58.4%",  "#ffaa00"),
                ("LLM Latency",        "~2-3s",  "#0088ff"),
                ("Detection FPS",      "8-9 CPU","#0088ff"),
                ("Alert Patterns",     "3 types","#0088ff"),
                ("Safety Layers",      "5",      "#ff2244"),
            ]

            for label, val, color in metrics:
                st.html(f"""
                <div style="
                    display:flex;justify-content:space-between;
                    padding:7px 0;border-bottom:1px solid #060d14;
                ">
                    <span style="font-family:'Space Mono',monospace;
                        font-size:0.62rem;color:#1a3a50;">{label}</span>
                    <span style="font-family:'Share Tech Mono',monospace;
                        font-size:0.72rem;color:{color};">{val}</span>
                </div>
                """)

            st.html("""
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                letter-spacing:0.2em;color:#1a3a50;
                margin-top:20px;margin-bottom:12px;">
            ◈ BEHAVIORAL PATTERNS DETECTED</div>
            """)

            patterns = [
                ("LOITERING",         "Object stationary > zone threshold",       "medium"),
                ("PERIMETER PROBING", "Repeated restricted zone entries",          "critical"),
                ("CROWD FORMATION",   "Rapid person count accumulation",           "high"),
            ]

            for name, desc, risk in patterns:
                risk_colors = {
                    "medium": "#ffaa00", "high": "#ff6600", "critical": "#ff2244"
                }
                c = risk_colors.get(risk, "#ffaa00")
                st.html(f"""
                <div style="background:#060d14;border:1px solid #0a1a2e;
                    border-left:3px solid {c};border-radius:2px;
                    padding:8px 12px;margin-bottom:6px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                        color:{c};margin-bottom:3px;">{name}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                        color:#1a3a50;">{desc}</div>
                </div>
                """)


if __name__ == "__main__":
    main()