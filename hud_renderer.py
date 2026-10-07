"""
Sentinel-Edge HUD Renderer
───────────────────────────
All OpenCV drawing helpers for the dark-tech heads-up display.
Keeps the main loop clean by encapsulating every overlay operation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

from config import TrackerConfig
from behavior_engine import (
    BehaviorVerdict,
    EntityState,
    breadcrumb_color,
    calculate_predicted_path,
)


# ═══════════════════════════════════════════════════════════════════════
#  Colour Constants
# ═══════════════════════════════════════════════════════════════════════

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_PLAIN
_WHITE      = (255, 255, 255)
_BLACK      = (0, 0, 0)
_DARK_BG    = (18, 18, 24)
_CYAN       = (200, 255, 0)        # BGR  → displayed as cyan-ish
_ALERT_RED  = (60, 60, 255)
_INTENSE_ORANGE = (0, 165, 255)
_INTERACTION_PURPLE = (255, 0, 255)
_FORECAST_BLUE = (255, 180, 0)


# ═══════════════════════════════════════════════════════════════════════
#  Utility – Semi-Transparent Rectangle
# ═══════════════════════════════════════════════════════════════════════

def _overlay_rect(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.55,
):
    """Draw a filled rectangle with alpha blending."""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ═══════════════════════════════════════════════════════════════════════
#  Zone Polygon Overlay
# ═══════════════════════════════════════════════════════════════════════

def draw_zone(frame: np.ndarray, zone_poly: np.ndarray, cfg: TrackerConfig):
    """Semi-transparent polygon + dashed border for restricted zone."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_poly], cfg.zone_overlay_color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Dashed border effect (draw short segments)
    pts = zone_poly.tolist() + [zone_poly[0].tolist()]
    for i in range(len(pts) - 1):
        p1, p2 = tuple(pts[i]), tuple(pts[i + 1])
        _draw_dashed_line(frame, p1, p2, cfg.zone_overlay_color, thickness=2, gap=12)

    # Label
    cx = int(np.mean(zone_poly[:, 0]))
    cy = int(np.mean(zone_poly[:, 1])) - 10
    cv2.putText(frame, "RESTRICTED ZONE", (cx - 80, cy),
                _FONT, 0.5, cfg.zone_overlay_color, 1, cv2.LINE_AA)


def _draw_dashed_line(img, p1, p2, color, thickness=1, gap=10):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist == 0:
        return
    dx, dy = (p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist
    i = 0
    draw = True
    while i < dist:
        end = min(i + gap, dist)
        sp = (int(p1[0] + dx * i), int(p1[1] + dy * i))
        ep = (int(p1[0] + dx * end), int(p1[1] + dy * end))
        if draw:
            cv2.line(img, sp, ep, color, thickness, cv2.LINE_AA)
        draw = not draw
        i = end


# ═══════════════════════════════════════════════════════════════════════
#  Kinematic Breadcrumbs
# ═══════════════════════════════════════════════════════════════════════

def draw_breadcrumbs(
    frame: np.ndarray,
    entity: EntityState,
    fps: float,
    cfg: TrackerConfig,
):
    """
    Draw a decaying trajectory polyline behind the entity.
    Older segments are more transparent (via decreasing thickness + alpha trick).
    """
    centroids = entity.centroids()
    n = min(len(centroids), cfg.breadcrumb_fade_length)
    if n < 2:
        return

    trail = centroids[-n:]
    color = breadcrumb_color(entity, fps, cfg)

    for i in range(1, len(trail)):
        # Alpha-fade via colour intensity scaling
        fade = i / len(trail)                       # 0 → old, 1 → newest
        c = tuple(int(ch * (0.25 + 0.75 * fade)) for ch in color)
        thick = max(1, int(cfg.breadcrumb_thickness * fade))
        pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        pt2 = (int(trail[i][0]), int(trail[i][1]))
        cv2.line(frame, pt1, pt2, c, thick, cv2.LINE_AA)

    # Glow dot on latest position
    latest = (int(trail[-1][0]), int(trail[-1][1]))
    cv2.circle(frame, latest, 4, color, -1, cv2.LINE_AA)
    cv2.circle(frame, latest, 7, color, 1, cv2.LINE_AA)


def draw_interaction_tethers(
    frame: np.ndarray,
    interacting_pairs: Set[Tuple[int, int]],
    entities: Dict[int, EntityState],
):
    """Draw live tethers for pairs that have sustained close proximity."""
    for first_id, second_id in interacting_pairs:
        first = entities[first_id].latest
        second = entities[second_id].latest
        if first is None or second is None:
            continue
        first_point = (int(first[1]), int(first[2]))
        second_point = (int(second[1]), int(second[2]))
        cv2.line(frame, first_point, second_point, _INTERACTION_PURPLE, 3, cv2.LINE_AA)


def draw_forklift_safety_zones(
    frame: np.ndarray,
    entities: Dict[int, EntityState],
    active_ids: Set[int],
    cfg: TrackerConfig,
):
    """Draw a translucent red safety radius around every active forklift."""
    overlay = frame.copy()
    radius = int(cfg.forklift_proximity_px)
    for entity_id in active_ids:
        entity = entities[entity_id]
        if entity.class_id != cfg.forklift_class_id or entity.latest is None:
            continue
        center = (int(entity.latest[1]), int(entity.latest[2]))
        cv2.circle(overlay, center, radius, _ALERT_RED, -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, _ALERT_RED, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def draw_predicted_paths(
    frame: np.ndarray,
    entities: Dict[int, EntityState],
    active_ids: Set[int],
    collision_points: List[Tuple[float, float]],
    cfg: TrackerConfig,
    frame_idx: int,
):
    """Draw projected entity paths and flash predicted collision points."""
    for entity_id in active_ids:
        path = calculate_predicted_path(entities[entity_id], cfg)
        if path is None:
            continue
        start = (int(path.start[0]), int(path.start[1]))
        end = (int(path.end[0]), int(path.end[1]))
        _draw_dashed_line(frame, start, end, _FORECAST_BLUE, thickness=2, gap=10)
        cv2.circle(frame, end, 6, _FORECAST_BLUE, 1, cv2.LINE_AA)

    if frame_idx % 20 < 10:
        for point in collision_points:
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, 18, _ALERT_RED, 3, cv2.LINE_AA)
            cv2.circle(frame, center, 28, _ALERT_RED, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  Entity Bounding Box + ID Label
# ═══════════════════════════════════════════════════════════════════════

def draw_entity_box(
    frame: np.ndarray,
    entity: EntityState,
    verdict: BehaviorVerdict,
    cfg: TrackerConfig,
):
    """Draw bounding box coloured by behaviour verdict and attach ID label."""
    latest = entity.latest
    if latest is None:
        return

    _, cx, cy, w, h = latest
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    color_map = {
        "NORMAL":           (0, 220, 80),
        "LOITERING":        (0, 0, 255),
        "POSTURE_COLLAPSE": (0, 80, 255),
        "ZONE_INTRUSION":   (255, 0, 80),
        "SUDDEN_ACCELERATION": _INTENSE_ORANGE,
        "SUSPICIOUS_INTERACTION": _INTERACTION_PURPLE,
        "CRITICAL: FORKLIFT_PROXIMITY": _ALERT_RED,
        "CRITICAL: COLLISION_PREDICTED": _ALERT_RED,
    }
    color = color_map.get(verdict.behavior, _WHITE)

    # Box with corner accents
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    corner_len = int(min(20, w // 4, h // 4))
    for (cx_, cy_), (dx, dy) in [
        ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
    ]:
        cv2.line(frame, (cx_, cy_), (int(cx_ + dx * corner_len), cy_), color, 3, cv2.LINE_AA)
        cv2.line(frame, (cx_, cy_), (cx_, int(cy_ + dy * corner_len)), color, 3, cv2.LINE_AA)

    # ID + behaviour tag
    label = f"ID:{entity.entity_id}"
    tag = verdict.behavior.replace("_", " ")
    label_full = f"{label}  {tag}" if verdict.behavior != "NORMAL" else label

    (tw, th), _ = cv2.getTextSize(label_full, _FONT, 0.48, 1)
    _overlay_rect(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, 0.65)
    cv2.putText(frame, label_full, (x1 + 4, y1 - 5),
                _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  Top HUD Bar
# ═══════════════════════════════════════════════════════════════════════

def draw_hud(
    frame: np.ndarray,
    fps_display: float,
    active_tracks: int,
    timestamp_str: str,
    anomalies: List[BehaviorVerdict],
    cfg: TrackerConfig,
):
    """
    Render the top-of-screen dark telemetry bar with FPS, track count,
    timestamp, and an anomaly warning banner if any alerts are active.
    """
    fh, fw = frame.shape[:2]
    bar_h = 44

    # ── Top bar background ───────────────────────────────────────────
    _overlay_rect(frame, (0, 0), (fw, bar_h), _DARK_BG, 0.75)

    # Left: System title
    cv2.putText(frame, "SENTINEL-EDGE", (12, 28),
                _FONT, 0.55, cfg.hud_color, 1, cv2.LINE_AA)

    # Centre: timestamp + FPS
    info = f"T {timestamp_str}   FPS {fps_display:.1f}   TRACKS {active_tracks}"
    (tw, _), _ = cv2.getTextSize(info, _FONT, 0.48, 1)
    cv2.putText(frame, info, (fw // 2 - tw // 2, 28),
                _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)

    # Right: status dot
    status_color = _ALERT_RED if anomalies else (0, 200, 0)
    cv2.circle(frame, (fw - 24, 22), 7, status_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (fw - 24, 22), 9, status_color, 1, cv2.LINE_AA)

    # ── Anomaly warning banner ───────────────────────────────────────
    if anomalies:
        banner_y = bar_h
        banner_h = 32
        _overlay_rect(frame, (0, banner_y), (fw, banner_y + banner_h), _ALERT_RED, 0.55)

        msgs = []
        for a in anomalies[:4]:  # cap at 4 to avoid overflow
            msgs.append(f"⚠ ID:{a.entity_id} {a.behavior.replace('_', ' ')} ({a.confidence:.0%})")
        banner_text = "   ".join(msgs)

        cv2.putText(frame, banner_text, (12, banner_y + 22),
                    _FONT, 0.44, _WHITE, 1, cv2.LINE_AA)

    # ── Bottom-left watermark ────────────────────────────────────────
    cv2.putText(frame, "sentinel-edge v1.0", (8, fh - 10),
                _FONT_SMALL, 0.9, (80, 80, 80), 1, cv2.LINE_AA)
