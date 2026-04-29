# src/detection/zone_manager.py
"""
Zone Manager.

Loads zone definitions from configs/zones.yaml and determines
which zone (if any) each tracked object is currently inside.

Zones are defined as polygons in normalised coordinates (0.0 - 1.0).
At runtime they are scaled to pixel coordinates based on frame size.

Example zones.yaml entry:
    zone_d:
      name: "Restricted Perimeter"
      risk_level: critical
      alert_threshold_seconds: 30
      polygon:
        - [0.5, 0.5]
        - [1.0, 0.5]
        - [1.0, 1.0]
        - [0.5, 1.0]
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
import cv2

from src.utils.logger import logger


@dataclass
class Zone:
    """
    One surveillance zone loaded from zones.yaml.

    Attributes:
        zone_id    : config key (e.g. "zone_d")
        name       : human-readable name
        risk_level : low / medium / high / critical
        threshold_s: seconds before loitering alert fires
        color_bgr  : BGR tuple for OpenCV drawing
        polygon_norm: normalised polygon [(x,y), ...]
    """
    zone_id:      str
    name:         str
    risk_level:   str
    threshold_s:  int
    color_bgr:    tuple[int, int, int]
    polygon_norm: list[tuple[float, float]]


class ZoneManager:
    """
    Loads zones from YAML and checks object positions against them.

    Usage:
        zm = ZoneManager("configs/zones.yaml")
        zone = zm.get_zone_for_point((320.0, 180.0), frame_wh=(640, 360))
        if zone:
            print(f"Object is in {zone.name} ({zone.risk_level})")
    """

    def __init__(self, config_path: str = "configs/zones.yaml"):
        self.zones: list[Zone] = []
        self._load(config_path)

    def _load(self, path: str):
        """Parse zones.yaml and build Zone objects."""
        config_file = Path(path)
        if not config_file.exists():
            logger.warning(f"zones.yaml not found at {path} — no zones active")
            return

        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        zones_cfg = config.get("zones", {})
        for zone_id, cfg in zones_cfg.items():
            # Parse polygon — list of [x, y] → list of (x, y) tuples
            polygon = [
                (float(pt[0]), float(pt[1]))
                for pt in cfg.get("polygon", [])
            ]
            if len(polygon) < 3:
                logger.warning(f"Zone {zone_id} has fewer than 3 points — skipped")
                continue

            color = cfg.get("color", [0, 255, 0])

            zone = Zone(
                zone_id      = zone_id,
                name         = cfg.get("name", zone_id),
                risk_level   = cfg.get("risk_level", "low"),
                threshold_s  = cfg.get("alert_threshold_seconds", 300),
                color_bgr    = tuple(color),
                polygon_norm = polygon,
            )
            self.zones.append(zone)

        logger.info(
            f"ZoneManager loaded {len(self.zones)} zones: "
            f"{[z.zone_id for z in self.zones]}"
        )

    def get_zone_for_point(
        self,
        point: tuple[float, float],
        frame_wh: tuple[int, int],
    ) -> Zone | None:
        """
        Check which zone contains the given point.

        Args:
            point    : (cx, cy) pixel coordinates of object center
            frame_wh : (width, height) of the current frame

        Returns:
            The Zone object if point is inside, else None.
            If point is in multiple zones, returns the highest risk one.
        """
        fw, fh = frame_wh
        cx, cy = point

        # Risk priority for overlap cases
        risk_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        best_zone     = None
        best_priority = 0

        for zone in self.zones:
            # Scale normalised polygon to pixel coordinates
            pixel_poly = np.array([
                [int(x * fw), int(y * fh)]
                for x, y in zone.polygon_norm
            ], dtype=np.float32)

            # OpenCV pointPolygonTest — positive = inside
            result = cv2.pointPolygonTest(
                pixel_poly,
                (float(cx), float(cy)),
                measureDist=False,
            )

            if result >= 0:   # inside or on boundary
                priority = risk_priority.get(zone.risk_level, 1)
                if priority > best_priority:
                    best_zone     = zone
                    best_priority = priority

        return best_zone

    def draw_zones(
        self,
        frame: np.ndarray,
        frame_wh: tuple[int, int],
        alpha: float = 0.25,
    ) -> np.ndarray:
        """
        Draw all zone polygons onto the frame.

        Args:
            frame    : BGR frame to draw on
            frame_wh : (width, height) of the frame
            alpha    : transparency of zone fill (0=invisible, 1=opaque)

        Returns:
            Frame with zones drawn (original frame modified in-place copy).
        """
        fw, fh   = frame_wh
        overlay  = frame.copy()

        for zone in self.zones:
            pixel_poly = np.array([
                [int(x * fw), int(y * fh)]
                for x, y in zone.polygon_norm
            ], dtype=np.int32)

            # Fill zone with semi-transparent color
            cv2.fillPoly(overlay, [pixel_poly], zone.color_bgr)

            # Draw zone border
            cv2.polylines(frame, [pixel_poly], isClosed=True,
                         color=zone.color_bgr, thickness=2)

            # Zone label
            label_x = pixel_poly[:, 0].min() + 5
            label_y = pixel_poly[:, 1].min() + 18
            cv2.putText(
                frame,
                f"{zone.zone_id.upper()} | {zone.risk_level.upper()}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                zone.color_bgr,
                1,
                cv2.LINE_AA,
            )

        # Blend fill with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    @property
    def zone_count(self) -> int:
        return len(self.zones)
