"""
Sentinel-Edge Configuration Module
───────────────────────────────────
Central dataclass-driven configuration for the entire vision pipeline.
All tunable thresholds, ROI definitions, and network settings live here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class TrackerConfig:
    """Immutable-ish configuration knob for the Sentinel-Edge pipeline."""

    # ── Video Source ──────────────────────────────────────────────────
    video_source: str = "sample_feed.mp4"
    webcam_fallback: int = 0
    output_dir: str = "output_videos"

    # ── YOLO / BoT-SORT ──────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"            # nano for speed; swap to yolov8s/m for accuracy
    tracker_yaml: str = "botsort_custom.yaml" # BoT-SORT: GMC + ReID for stable IDs
    detection_classes: List[int] = field(default_factory=lambda: [0, 7])  # person + truck
    confidence_threshold: float = 0.4         # reject faint / ghost detections
    track_buffer_frames: int = 120            # retain IDs through longer occlusions

    # ── Rolling History ──────────────────────────────────────────────
    history_length: int = 150                 # ~5s at 30fps for spatial variance analysis
    velocity_history_length: int = 30         # rolling velocity baseline (~1s at 30fps)

    # ── Kinematic Breadcrumbs ────────────────────────────────────────
    breadcrumb_fade_length: int = 40          # how many past points to draw
    breadcrumb_thickness: int = 2

    # ── Behavior Thresholds ──────────────────────────────────────────
    #   Rule A – Loitering (spatial variance approach)
    loiter_radius_px: float = 40.0            # maximum centroid radius over the history window
    centroid_smoothing_frames: int = 5        # SMA window for detection jitter

    #   Rule B – Posture Collapse / Fall
    fall_ratio_drop: float = 0.40             # 40 % drop in h/w ratio
    fall_window_sec: float = 1.0              # look-back window

    #   Rule D - Sudden acceleration
    sudden_acceleration_ratio: float = 3.0    # instantaneous / rolling average
    min_velocity_px_sec: float = 50.0         # reject tiny tracking jitter

    #   Rule E - Proximity interaction
    proximity_threshold_px: float = 60.0
    proximity_duration_sec: float = 4.0

    #   Rule F - Human-machine safety
    person_class_id: int = 0
    forklift_class_id: int = 7                # COCO truck class
    forklift_proximity_px: float = 150.0

    #   Rule G - Predictive collision forecasting
    forecast_history_frames: int = 10
    forecast_horizon_sec: float = 3.0
    forecast_fps: float = 30.0
    collision_time_window_sec: float = 1.0

    #   Rule C – Zone Intrusion
    #   Default polygon (normalised 0-1 coords; scaled at runtime to frame size)
    zone_polygon_norm: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.30, 0.30),
            (0.70, 0.30),
            (0.70, 0.80),
            (0.30, 0.80),
        ]
    )

    # ── Breadcrumb Colour Thresholds ─────────────────────────────────
    normal_displacement_px: float = 20.0      # green if net > this over 3 s
    normal_window_sec: float = 3.0
    erratic_ratio: float = 2.5               # total_path / net_displacement ratio for yellow

    # ── Telemetry ────────────────────────────────────────────────────
    websocket_uri: str = "ws://localhost:8080/events"
    telemetry_interval_sec: float = 1.0       # periodic heartbeat
    ws_reconnect_delay_sec: float = 2.0       # back-off on disconnect

    # ── Display ──────────────────────────────────────────────────────
    window_title: str = "Sentinel-Edge AI Vision Node"
    hud_font_scale: float = 0.55
    hud_color: Tuple[int, int, int] = (0, 255, 200)        # cyan-ish
    alert_color: Tuple[int, int, int] = (0, 0, 255)        # red
    zone_overlay_color: Tuple[int, int, int] = (255, 0, 80) # magenta

    # ── Derived helpers (not fields) ─────────────────────────────────
    def zone_polygon_px(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Return the zone polygon in absolute pixel coordinates."""
        return np.array(
            [(int(x * frame_w), int(y * frame_h)) for x, y in self.zone_polygon_norm],
            dtype=np.int32,
        )
