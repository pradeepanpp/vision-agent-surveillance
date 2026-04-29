# scripts/draw_zones.py
"""
Interactive Zone Drawing Tool.

Lets you click on a video frame to define surveillance zones.
Saves the result directly to configs/zones.yaml.

Controls:
    LEFT CLICK      → add a point to current zone polygon
    RIGHT CLICK     → undo last point
    ENTER / N       → finish current zone, start next
    Z               → undo entire last zone
    S               → save all zones to configs/zones.yaml
    Q               → quit without saving

Usage:
    python scripts/draw_zones.py
    python scripts/draw_zones.py --video data/test_video.mp4
    python scripts/draw_zones.py --frame 100
"""

import sys
import os
import cv2
import yaml
import argparse
import numpy as np
sys.path.append(".")

from src.utils.video_utils import open_video, prepare_frame

# ─────────────────────────────────────────────
# ZONE CONFIGURATION TEMPLATES
# ─────────────────────────────────────────────
ZONE_TEMPLATES = [
    {
        "zone_id":     "zone_parking",
        "name":        "Parking Area",
        "description": "Authorised vehicle storage — stationary vehicles expected",
        "risk_level":  "low",
        "alert_threshold_seconds": 1800,
        "color":       [0, 255, 0],
    },
    {
        "zone_id":     "zone_road",
        "name":        "Main Road Corridor",
        "description": "Active traffic lane — stationary vehicles are suspicious",
        "risk_level":  "medium",
        "alert_threshold_seconds": 180,
        "color":       [0, 165, 255],
    },
    {
        "zone_id":     "zone_intersection",
        "name":        "Intersection",
        "description": "Traffic junction — congestion or stopped vehicles flagged",
        "risk_level":  "high",
        "alert_threshold_seconds": 60,
        "color":       [0, 100, 255],
    },
    {
        "zone_id":     "zone_restricted",
        "name":        "Restricted Perimeter",
        "description": "No vehicle access permitted — immediate alert on entry",
        "risk_level":  "critical",
        "alert_threshold_seconds": 30,
        "color":       [0, 0, 255],
    },
]

RISK_COLORS = {
    "low":      (0, 255, 0),
    "medium":   (0, 165, 255),
    "high":     (0, 100, 255),
    "critical": (0, 0, 255),
}


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
state = {
    "current_points": [],   # points for the zone being drawn
    "finished_zones": [],   # list of completed zones
    "zone_index":     0,    # which template we are on
    "frame":          None, # base frame (never modified)
    "frame_wh":       (640, 360),
}


def get_current_template() -> dict | None:
    idx = state["zone_index"]
    if idx < len(ZONE_TEMPLATES):
        return ZONE_TEMPLATES[idx]
    return None


def get_current_color() -> tuple:
    tmpl = get_current_template()
    if tmpl:
        return tuple(tmpl["color"])
    return (255, 255, 255)


# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────

def render(window_name: str):
    """Redraw the full canvas with all zones and current polygon."""
    frame = state["frame"].copy()
    fw, fh = state["frame_wh"]

    # Draw completed zones
    for zone_data in state["finished_zones"]:
        pts   = zone_data["pixel_points"]
        color = tuple(zone_data["color"])
        poly  = np.array(pts, dtype=np.int32)

        # Fill with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Border
        cv2.polylines(frame, [poly], True, color, 2)

        # Label at centroid
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(frame, zone_data["zone_id"].upper(),
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    # Draw current in-progress polygon
    pts   = state["current_points"]
    color = get_current_color()

    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 2)

    # Draw points
    for pt in pts:
        cv2.circle(frame, pt, 5, color, -1)

    # Draw instruction overlay
    _draw_instructions(frame)

    cv2.imshow(window_name, frame)


def _draw_instructions(frame: np.ndarray):
    """Draw current instructions in top-right corner."""
    tmpl = get_current_template()
    zone_count = len(state["finished_zones"])

    lines = []
    if tmpl:
        lines = [
            f"Drawing zone {zone_count + 1}/{len(ZONE_TEMPLATES)}:",
            f"  {tmpl['zone_id'].upper()}",
            f"  Risk: {tmpl['risk_level'].upper()}",
            f"  Points placed: {len(state['current_points'])}",
            "",
            "LEFT CLICK  → add point",
            "RIGHT CLICK → undo point",
            "ENTER / N   → finish zone",
            "Z           → undo last zone",
            "S           → save + exit",
            "Q           → quit",
        ]
    else:
        lines = [
            f"All {zone_count} zones drawn!",
            "",
            "S → save to zones.yaml",
            "Q → quit without saving",
        ]

    x, y = frame.shape[1] - 230, 15
    for i, line in enumerate(lines):
        color = (255, 255, 255) if not line.startswith("  ") else (200, 200, 200)
        cv2.putText(frame, line, (x, y + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# MOUSE CALLBACK
# ─────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks."""
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point to current polygon
        if get_current_template():
            state["current_points"].append((x, y))
            render(param)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Undo last point
        if state["current_points"]:
            state["current_points"].pop()
            render(param)


# ─────────────────────────────────────────────
# ZONE COMPLETION
# ─────────────────────────────────────────────

def finish_current_zone():
    """
    Validate and save the current zone polygon.
    Converts pixel coordinates to normalised (0-1) coordinates.
    """
    pts = state["current_points"]
    if len(pts) < 3:
        print(f"  ⚠️  Need at least 3 points — only {len(pts)} placed. Keep drawing.")
        return False

    tmpl   = get_current_template()
    fw, fh = state["frame_wh"]

    # Normalise to 0.0-1.0
    norm_pts = [
        [round(px / fw, 4), round(py / fh, 4)]
        for px, py in pts
    ]

    zone_data = {
        "zone_id":      tmpl["zone_id"],
        "name":         tmpl["name"],
        "description":  tmpl["description"],
        "risk_level":   tmpl["risk_level"],
        "alert_threshold_seconds": tmpl["alert_threshold_seconds"],
        "color":        tmpl["color"],
        "pixel_points": pts,
        "norm_points":  norm_pts,
    }

    state["finished_zones"].append(zone_data)
    state["current_points"] = []
    state["zone_index"]    += 1

    print(
        f"  ✅ Zone '{tmpl['zone_id']}' saved "
        f"({len(pts)} points)"
    )
    return True


def undo_last_zone():
    """Remove the last completed zone."""
    if state["finished_zones"]:
        removed = state["finished_zones"].pop()
        state["zone_index"] -= 1
        state["current_points"] = []
        print(f"  ↩️  Undid zone '{removed['zone_id']}'")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_zones(output_path: str = "configs/zones.yaml"):
    """Save all drawn zones to zones.yaml."""
    if not state["finished_zones"]:
        print("No zones to save.")
        return

    # Build YAML structure
    zones_dict = {}
    for zd in state["finished_zones"]:
        zones_dict[zd["zone_id"]] = {
            "name":        zd["name"],
            "description": zd["description"],
            "risk_level":  zd["risk_level"],
            "alert_threshold_seconds": zd["alert_threshold_seconds"],
            "color":       zd["color"],
            "polygon":     zd["norm_points"],
        }

    # Load existing config to preserve non-zone sections
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    # Merge zones into existing config
    existing["zones"] = zones_dict

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)

    print(f"\n✅ Saved {len(zones_dict)} zones to {output_path}")
    for zid, zdata in zones_dict.items():
        print(f"   {zid}: {zdata['name']} ({zdata['risk_level']})")
        for pt in zdata["polygon"]:
            print(f"      [{pt[0]}, {pt[1]}]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive zone drawing tool")
    parser.add_argument("--video", default="data/test_video.mp4")
    parser.add_argument("--frame", type=int, default=30,
                        help="Which frame to use as background (default: 30)")
    parser.add_argument("--output", default="configs/zones.yaml")
    args = parser.parse_args()

    # Load frame
    cap = open_video(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, raw = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {args.frame} from {args.video}")
        sys.exit(1)

    frame = prepare_frame(raw)
    fw, fh = frame.shape[1], frame.shape[0]
    state["frame"]    = frame
    state["frame_wh"] = (fw, fh)

    print("=" * 56)
    print("  Zone Drawing Tool")
    print("=" * 56)
    print(f"  Video  : {args.video}")
    print(f"  Frame  : {args.frame}")
    print(f"  Size   : {fw}x{fh}")
    print(f"  Output : {args.output}")
    print()
    print("  You will draw these zones:")
    for i, tmpl in enumerate(ZONE_TEMPLATES):
        color_name = tmpl["risk_level"].upper()
        print(f"    {i+1}. {tmpl['zone_id']:20s} → {color_name}")
    print()
    print("  Instructions:")
    print("    LEFT CLICK  = add polygon point")
    print("    RIGHT CLICK = remove last point")
    print("    ENTER / N   = finish this zone")
    print("    Z           = undo last zone")
    print("    S           = save and exit")
    print("    Q           = quit without saving")
    print("=" * 56)

    # Create window
    win = "Zone Drawing Tool — ENTER to finish zone, S to save, Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, fw * 2, fh * 2)   # 2x zoom for easier clicking
    cv2.setMouseCallback(win, mouse_callback, win)

    render(win)

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            print("\nQuit without saving.")
            break

        elif key == ord("s"):
            save_zones(args.output)
            break

        elif key in (13, ord("n")):    # ENTER or N = finish zone
            if get_current_template():
                finish_current_zone()
                render(win)
                if not get_current_template():
                    print("\nAll zones drawn! Press S to save or Q to quit.")

        elif key == ord("z"):          # Z = undo last zone
            undo_last_zone()
            render(win)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()