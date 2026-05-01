# dashboard/components.py
"""
Dashboard Visual Components — Phase 4.

Aesthetic: Tactical Operations Center — deep black background,
phosphor green primary accent, amber warnings, red alerts.
Military-grade surveillance UI inspired by real CCTV control rooms.

Color palette (updated — all text clearly readable):
  Background  : #020408  near-black
  Surface     : #060d14  dark navy cards
  Borders     : #0a1a2e  subtle separation (decorative only)
  Primary text: #e0e8f0  near-white
  Secondary   : #8fb5cc  steel blue — visible on dark background
  Muted       : #6a9ab5  softer blue — still readable
  Green       : #00ff88  phosphor green — active/safe
  Amber       : #ffaa00  warning state
  Red         : #ff2244  critical alert
  Blue        : #0088ff  data labels
"""

import streamlit as st


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=Share+Tech+Mono&display=swap');

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020408 !important;
        color: #e0e8f0 !important;
        font-family: 'Space Mono', monospace !important;
    }
    [data-testid="stSidebar"] {
        background-color: #030810 !important;
        border-right: 1px solid #0a1a2e !important;
    }
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    /* ── Scanline overlay ── */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,136,0.012) 2px,
            rgba(0,255,136,0.012) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #0a1a2e;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #6a9ab5 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        padding: 10px 24px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00ff88 !important;
        border-bottom: 2px solid #00ff88 !important;
        text-shadow: 0 0 8px rgba(0,255,136,0.6) !important;
    }

    /* ── Inputs ── */
    .stTextArea textarea, .stTextInput input {
        background: #060d14 !important;
        border: 1px solid #0a1a2e !important;
        color: #e0e8f0 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 2px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #00ff88 !important;
        box-shadow: 0 0 12px rgba(0,255,136,0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid #00ff88 !important;
        color: #00ff88 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        border-radius: 2px !important;
        padding: 8px 20px !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: rgba(0,255,136,0.08) !important;
        box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: #060d14 !important;
        border: 1px solid #0a1a2e !important;
        color: #e0e8f0 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #060d14;
        border: 1px solid #0a1a2e;
        border-radius: 2px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.58rem !important;
        letter-spacing: 0.22em !important;
        text-transform: uppercase !important;
        color: #8fb5cc !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Share Tech Mono', monospace !important;
        color: #00ff88 !important;
        font-size: 1.6rem !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: #00ff88 !important;
    }

    /* ── Divider ── */
    hr { border-color: #0a1a2e !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 3px; }
    ::-webkit-scrollbar-track { background: #020408; }
    ::-webkit-scrollbar-thumb { background: #1a3a50; }

    /* ── Animations ── */
    @keyframes glow-pulse {
        0%   { box-shadow: 0 0 5px rgba(255,34,68,0.4); }
        50%  { box-shadow: 0 0 20px rgba(255,34,68,0.8), 0 0 40px rgba(255,34,68,0.3); }
        100% { box-shadow: 0 0 5px rgba(255,34,68,0.4); }
    }
    @keyframes amber-pulse {
        0%   { box-shadow: 0 0 5px rgba(255,170,0,0.3); }
        50%  { box-shadow: 0 0 15px rgba(255,170,0,0.6); }
        100% { box-shadow: 0 0 5px rgba(255,170,0,0.3); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

def render_header(system_status: str = "OPERATIONAL"):
    status_color = "#00ff88" if system_status == "OPERATIONAL" else "#ff2244"
    st.html(f"""
    <div style="
        border-bottom: 1px solid #0a1a2e;
        padding-bottom: 16px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    ">
        <div>
            <div style="
                font-family:'Space Mono',monospace;
                font-size:0.62rem;
                letter-spacing:0.3em;
                color:#8fb5cc;
                margin-bottom:6px;
            ">◈ UAE CRITICAL INFRASTRUCTURE — SURVEILLANCE COMMAND</div>
            <div style="
                font-family:'Bebas Neue',sans-serif;
                font-size:2.8rem;
                color:#e0e8f0;
                letter-spacing:0.06em;
                line-height:1;
                text-shadow: 0 0 30px rgba(0,255,136,0.15);
            ">VISION AGENT</div>
            <div style="
                font-family:'Space Mono',monospace;
                font-size:0.65rem;
                color:#6a9ab5;
                letter-spacing:0.12em;
                margin-top:4px;
            ">YOLO26s · BYTETRACK · LANGGRAPH · GPT-4o-mini</div>
        </div>
        <div style="text-align:right;">
            <div style="
                font-family:'Space Mono',monospace;
                font-size:0.62rem;
                color:{status_color};
                letter-spacing:0.2em;
            ">● {system_status}</div>
            <div style="
                font-family:'Share Tech Mono',monospace;
                font-size:1.1rem;
                color:#e0e8f0;
                margin-top:4px;
            " id="live-clock">--:--:--</div>
            <div style="
                font-family:'Space Mono',monospace;
                font-size:0.58rem;
                color:#6a9ab5;
                margin-top:2px;
            ">UTC+4 GULF STANDARD TIME</div>
        </div>
    </div>
    <script>
    function updateClock() {{
        const now = new Date();
        const t = now.toUTCString().split(' ')[4];
        const el = document.getElementById('live-clock');
        if (el) el.textContent = t;
    }}
    setInterval(updateClock, 1000);
    updateClock();
    </script>
    """)


# ─────────────────────────────────────────────
# ALERT CARD
# ─────────────────────────────────────────────

def render_alert_card(alert: dict):
    risk   = alert.get("risk_level", "AMBER")
    etype  = alert.get("event_type", "unknown").replace("_", " ").upper()
    zone   = alert.get("zone", "—")
    dur    = alert.get("duration_s", 0)
    reason = alert.get("reasoning", "")
    action = alert.get("recommended_action", "")
    conf   = alert.get("confidence", "MEDIUM")
    ts     = alert.get("timestamp", "")
    tid    = alert.get("track_id", "—")
    msg    = alert.get("alert_message", "")

    colors = {
        "RED":   {"border":"#ff2244","bg":"#1a0008","icon":"🔴","anim":"glow-pulse 1.5s infinite"},
        "AMBER": {"border":"#ffaa00","bg":"#1a0e00","icon":"🟡","anim":"amber-pulse 2s infinite"},
        "GREEN": {"border":"#00ff88","bg":"#001a0e","icon":"🟢","anim":"none"},
    }
    c = colors.get(risk, colors["AMBER"])

    conf_colors = {"HIGH":"#00ff88","MEDIUM":"#ffaa00","LOW":"#ff2244"}
    conf_color  = conf_colors.get(conf, "#ffaa00")

    st.html(f"""
    <div style="
        background:{c['bg']};
        border:1px solid {c['border']};
        border-left:4px solid {c['border']};
        border-radius:2px;
        padding:16px 18px;
        margin-bottom:12px;
        animation:{c['anim']};
        position:relative;
        overflow:hidden;
    ">
        <div style="position:absolute;top:0;right:0;width:40px;height:40px;
            border-left:1px solid {c['border']}30;border-bottom:1px solid {c['border']}30;"></div>

        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div style="display:flex;gap:10px;align-items:center;">
                <span style="font-size:1rem;">{c['icon']}</span>
                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;
                    color:{c['border']};letter-spacing:0.1em;">{risk}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;
                    color:{c['border']}cc;letter-spacing:0.08em;">▸ {etype}</span>
            </div>
            <div style="display:flex;gap:12px;align-items:center;">
                <span style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    color:{conf_color};letter-spacing:0.12em;">CONF:{conf}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    color:#8fb5cc;">{ts[11:19] if ts else ''}</span>
            </div>
        </div>

        <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;
            color:#e0e8f0;margin-bottom:10px;line-height:1.5;">
            {msg if msg else reason}
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;
            font-family:'Space Mono',monospace;font-size:0.68rem;">
            <div><span style="color:#8fb5cc;">ZONE ▸ </span>
                 <span style="color:{c['border']};">{zone}</span></div>
            <div><span style="color:#8fb5cc;">TRACK ▸ </span>
                 <span style="color:{c['border']};">ID-{tid}</span></div>
            <div><span style="color:#8fb5cc;">DURATION ▸ </span>
                 <span style="color:{c['border']};">{dur:.0f}s</span></div>
            <div><span style="color:#8fb5cc;">LLM ▸ </span>
                 <span style="color:#8fb5cc;">{alert.get('llm_latency_ms',0):.0f}ms</span></div>
        </div>

        <div style="margin-top:10px;padding-top:10px;
            border-top:1px solid {c['border']}20;
            font-family:'Space Mono',monospace;font-size:0.68rem;">
            <span style="color:#8fb5cc;">ACTION ▸ </span>
            <span style="color:#e0e8f0;">{action}</span>
        </div>
    </div>
    """)


# ─────────────────────────────────────────────
# ZONE STATUS PANEL
# ─────────────────────────────────────────────

def render_zone_status(zone_population: dict, zones_config: list):
    risk_colors = {
        "low":      "#00ff88",
        "medium":   "#ffaa00",
        "high":     "#ff6600",
        "critical": "#ff2244",
    }
    st.html("""
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
        letter-spacing:0.2em;color:#8fb5cc;margin-bottom:10px;">
    ◈ ZONE OCCUPANCY — REAL TIME</div>
    """)

    for zone in zones_config:
        zid     = zone.get("zone_id", "")
        name    = zone.get("name", zid)
        risk    = zone.get("risk_level", "low")
        count   = zone_population.get(zid, 0)
        color   = risk_colors.get(risk, "#00ff88")
        bar_pct = min(count * 20, 100)
        obj_color = color if count > 0 else "#6a9ab5"

        st.html(f"""
        <div style="background:#060d14;border:1px solid #0a1a2e;
            border-left:3px solid {color};border-radius:2px;
            padding:8px 12px;margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-family:'Space Mono',monospace;font-size:0.68rem;
                    color:{color};">{name.upper()}</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                    color:{obj_color};">{count} OBJ</span>
            </div>
            <div style="background:#020408;height:3px;border-radius:1px;">
                <div style="width:{bar_pct}%;height:3px;background:{color};
                    border-radius:1px;transition:width 0.5s ease;
                    {'box-shadow:0 0 8px ' + color if count > 0 else ''}"></div>
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                color:#8fb5cc;margin-top:4px;">{risk.upper()} RISK</div>
        </div>
        """)


# ─────────────────────────────────────────────
# OBJECT TRACKING TABLE
# ─────────────────────────────────────────────

def render_tracking_table(object_log: dict, fps: float = 15.0):
    active = [r for r in object_log.values() if r.is_active and r.lost_frames < 45]
    if not active:
        st.html("""
        <div style="font-family:'Space Mono',monospace;font-size:0.78rem;
            color:#6a9ab5;text-align:center;padding:24px;
            border:1px dashed #0a1a2e;">NO ACTIVE TRACKS</div>
        """)
        return

    rows_html = ""
    for rec in sorted(active, key=lambda r: r.track_id):
        zone  = rec.current_zone or "—"
        dwell = rec.dwell_seconds(rec.current_zone, fps) if rec.current_zone else 0
        total = rec.total_seconds(fps)

        rows_html += f"""
        <tr style="border-bottom:1px solid #060d14;">
            <td style="padding:8px 10px;font-family:'Share Tech Mono',monospace;
                font-size:0.75rem;color:#00ff88;">ID-{rec.track_id}</td>
            <td style="padding:8px 10px;font-family:'Space Mono',monospace;
                font-size:0.68rem;color:#e0e8f0;">{rec.class_name.upper()}</td>
            <td style="padding:8px 10px;font-family:'Space Mono',monospace;
                font-size:0.65rem;color:#8fb5cc;">{zone}</td>
            <td style="padding:8px 10px;font-family:'Share Tech Mono',monospace;
                font-size:0.72rem;color:#ffaa00;">{dwell:.1f}s</td>
            <td style="padding:8px 10px;font-family:'Space Mono',monospace;
                font-size:0.68rem;color:#6a9ab5;">{total:.1f}s</td>
        </tr>
        """

    st.html(f"""
    <div style="background:#060d14;border:1px solid #0a1a2e;border-radius:2px;overflow:hidden;">
    <table style="width:100%;border-collapse:collapse;">
        <thead>
            <tr style="border-bottom:1px solid #0a1a2e;background:#020408;">
                <th style="padding:8px 10px;font-family:'Space Mono',monospace;
                    font-size:0.6rem;letter-spacing:0.18em;color:#8fb5cc;text-align:left;">ID</th>
                <th style="padding:8px 10px;font-family:'Space Mono',monospace;
                    font-size:0.6rem;letter-spacing:0.18em;color:#8fb5cc;text-align:left;">CLASS</th>
                <th style="padding:8px 10px;font-family:'Space Mono',monospace;
                    font-size:0.6rem;letter-spacing:0.18em;color:#8fb5cc;text-align:left;">ZONE</th>
                <th style="padding:8px 10px;font-family:'Space Mono',monospace;
                    font-size:0.6rem;letter-spacing:0.18em;color:#8fb5cc;text-align:left;">DWELL</th>
                <th style="padding:8px 10px;font-family:'Space Mono',monospace;
                    font-size:0.6rem;letter-spacing:0.18em;color:#8fb5cc;text-align:left;">TOTAL</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(session_stats: dict, frame_num: int, fps: float):
    st.html("""
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
        letter-spacing:0.22em;color:#8fb5cc;
        margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #0a1a2e;">
    ◈ SYSTEM STATUS</div>
    """)

    total   = session_stats.get("total",   0)
    red     = session_stats.get("red",     0)
    amber   = session_stats.get("amber",   0)
    green   = session_stats.get("green",   0)
    tracked = session_stats.get("tracked", 0)

    status_items = [
        ("YOLO26s",     "ONLINE", "#00ff88"),
        ("BYTETRACK",   "ONLINE", "#00ff88"),
        ("LANGGRAPH",   "ONLINE", "#00ff88"),
        ("GPT-4o-mini", "ONLINE", "#00ff88"),
        ("SQL GUARD",   "ACTIVE", "#00ff88"),
    ]

    rows = ""
    for label, val, color in status_items:
        rows += f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;
            border-bottom:1px solid #060d14;">
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;
                color:#8fb5cc;">{label}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                color:{color};">● {val}</span>
        </div>
        """

    st.html(f"""
    <div style="background:#060d14;border:1px solid #0a1a2e;
        border-radius:2px;padding:10px 12px;margin-bottom:14px;">
        {rows}
    </div>
    """)

    st.html("""
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
        letter-spacing:0.22em;color:#8fb5cc;
        margin-bottom:10px;margin-top:16px;
        padding-bottom:6px;border-bottom:1px solid #0a1a2e;">
    ◈ SESSION STATS</div>
    """)

    st.html(f"""
    <div style="background:#060d14;border:1px solid #0a1a2e;
        border-radius:2px;padding:10px 12px;margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#8fb5cc;">FRAME</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#00ff88;">{frame_num}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#8fb5cc;">TRACKED</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#00ff88;">{tracked}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#8fb5cc;">TOTAL ALERTS</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#e0e8f0;">{total}</span>
        </div>
        <div style="border-top:1px solid #0a1a2e;padding-top:8px;margin-top:4px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#ff2244;">⬤ RED</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#ff2244;">{red}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#ffaa00;">⬤ AMBER</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#ffaa00;">{amber}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#00ff88;">⬤ GREEN</span>
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#00ff88;">{green}</span>
            </div>
        </div>
    </div>
    """)