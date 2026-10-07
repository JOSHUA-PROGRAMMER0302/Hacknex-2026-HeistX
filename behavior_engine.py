"""
Sentinel-Edge Behavior Analysis Engine
───────────────────────────────────────
Stateless rule evaluators that operate on per-entity history buffers.
Each rule returns (is_triggered: bool, confidence: float, label: str).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from config import TrackerConfig


# ═══════════════════════════════════════════════════════════════════════
#  Per-Entity State Container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EntityState:
    """Rolling history buffer for a single tracked person."""

    entity_id: int
    first_seen_frame: int = 0
    class_id: int = 0

    # Deques hold (frame_idx, x, y, w, h) tuples
    history: Deque[Tuple[int, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )
    velocity_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    raw_centroids: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    instantaneous_velocity_px_sec: float = 0.0
    rolling_average_velocity_px_sec: float = 0.0

    def update(
        self,
        frame_idx: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        fps: float = 30.0,
    ):
        self.raw_centroids.append((cx, cy))
        smoothed_cx = sum(x for x, _ in self.raw_centroids) / len(self.raw_centroids)
        smoothed_cy = sum(y for _, y in self.raw_centroids) / len(self.raw_centroids)

        if self.history:
            previous = self.history[-1]
            frame_delta = frame_idx - previous[0]
            if frame_delta > 0 and fps > 0:
                distance = math.hypot(
                    smoothed_cx - previous[1],
                    smoothed_cy - previous[2],
                )
                self.instantaneous_velocity_px_sec = distance / frame_delta * fps
                if self.velocity_history:
                    self.rolling_average_velocity_px_sec = float(
                        np.mean(self.velocity_history)
                    )
                self.velocity_history.append(self.instantaneous_velocity_px_sec)
        self.history.append((frame_idx, smoothed_cx, smoothed_cy, w, h))

    @property
    def latest(self) -> Optional[Tuple[int, float, float, float, float]]:
        return self.history[-1] if self.history else None

    def centroids(self) -> List[Tuple[float, float]]:
        return [(x, y) for _, x, y, _, _ in self.history]

    def aspect_ratios(self) -> List[float]:
        """h / w for each stored frame."""
        return [h / w if w > 0 else 0.0 for _, _, _, w, h in self.history]


# ═══════════════════════════════════════════════════════════════════════
#  Breadcrumb Colour Logic
# ═══════════════════════════════════════════════════════════════════════

_GREEN  = (0, 220, 80)
_YELLOW = (0, 220, 255)
_RED    = (0, 0, 255)
_ORANGE = (0, 165, 255)


@dataclass(frozen=True)
class PredictedPath:
    """Linear future path for one tracked entity."""

    entity_id: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    velocity_per_frame: Tuple[float, float]
    speed_px_sec: float


def breadcrumb_color(
    entity: EntityState, fps: float, cfg: TrackerConfig
) -> Tuple[int, int, int]:
    """
    Determine the trajectory polyline colour for this entity.

    Green  → Normal displacement (> 20 px net over 3 s).
    Yellow → High total path but low net displacement (pacing / erratic).
    Red    → Very low net displacement over 5 s (loitering).
    """
    centroids = entity.centroids()
    if len(centroids) < 2:
        return _GREEN

    if is_sudden_acceleration(entity, cfg):
        return _ORANGE

    # --- Centroid radius check for red (stationary / loitering) ---
    if centroid_radius(entity) <= cfg.loiter_radius_px:
        return _RED

    # --- 3-second window for green / yellow ---
    frames_3s = max(1, int(fps * cfg.normal_window_sec))
    window_3s = centroids[-min(len(centroids), frames_3s):]
    net_3s = math.hypot(window_3s[-1][0] - window_3s[0][0],
                        window_3s[-1][1] - window_3s[0][1])

    if net_3s > cfg.normal_displacement_px:
        return _GREEN

    total_path = sum(
        math.hypot(window_3s[i + 1][0] - window_3s[i][0],
                   window_3s[i + 1][1] - window_3s[i][1])
        for i in range(len(window_3s) - 1)
    )

    if net_3s > 0 and total_path / net_3s > cfg.erratic_ratio:
        return _YELLOW

    return _GREEN


def is_sudden_acceleration(entity: EntityState, cfg: TrackerConfig) -> bool:
    """Return whether the latest velocity is a significant baseline spike."""
    baseline = entity.rolling_average_velocity_px_sec
    return (
        baseline > 0
        and entity.instantaneous_velocity_px_sec >= cfg.min_velocity_px_sec
        and entity.instantaneous_velocity_px_sec
        > cfg.sudden_acceleration_ratio * baseline
    )


def check_sudden_acceleration(
    entity: EntityState, cfg: TrackerConfig
) -> Tuple[bool, float]:
    """Detect a sudden velocity spike relative to this ID's rolling baseline."""
    if not is_sudden_acceleration(entity, cfg):
        return False, 0.0
    ratio = entity.instantaneous_velocity_px_sec / entity.rolling_average_velocity_px_sec
    confidence = min(1.0, 0.5 + 0.5 * (
        ratio - cfg.sudden_acceleration_ratio
    ) / cfg.sudden_acceleration_ratio)
    return True, round(confidence, 3)


def calculate_predicted_path(
    entity: EntityState,
    cfg: TrackerConfig,
) -> Optional[PredictedPath]:
    """Project an entity's smoothed centroid three seconds into the future."""
    if entity.class_id not in (cfg.person_class_id, cfg.forklift_class_id):
        return None
    if len(entity.history) < cfg.forecast_history_frames:
        return None

    previous = entity.history[-cfg.forecast_history_frames]
    latest = entity.history[-1]
    frame_delta = latest[0] - previous[0]
    if frame_delta <= 0:
        return None

    velocity_x = (latest[1] - previous[1]) / frame_delta
    velocity_y = (latest[2] - previous[2]) / frame_delta
    horizon_frames = cfg.forecast_horizon_sec * cfg.forecast_fps
    end = (
        latest[1] + velocity_x * horizon_frames,
        latest[2] + velocity_y * horizon_frames,
    )
    speed_px_sec = math.hypot(velocity_x, velocity_y) * cfg.forecast_fps
    return PredictedPath(
        entity.entity_id,
        (latest[1], latest[2]),
        end,
        (velocity_x, velocity_y),
        speed_px_sec,
    )


def _segment_intersection(
    first: PredictedPath,
    second: PredictedPath,
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """Return intersection point and normalized segment positions."""
    x1, y1 = first.start
    x2, y2 = first.end
    x3, y3 = second.start
    x4, y4 = second.end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None

    first_param = (
        (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    ) / denominator
    second_param = (
        (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    ) / denominator
    if not (0.0 <= first_param <= 1.0 and 0.0 <= second_param <= 1.0):
        return None

    point = (
        x1 + first_param * (x2 - x1),
        y1 + first_param * (y2 - y1),
    )
    return point, first_param, second_param


def predict_collisions(
    entities: Dict[int, EntityState],
    active_ids: Set[int],
    cfg: TrackerConfig,
) -> Tuple[Set[int], List[Tuple[float, float]]]:
    """Find forklift/person path intersections with near-simultaneous arrival."""
    paths = {
        entity_id: calculate_predicted_path(entities[entity_id], cfg)
        for entity_id in active_ids
    }
    paths = {entity_id: path for entity_id, path in paths.items() if path}
    collision_ids: Set[int] = set()
    collision_points: List[Tuple[float, float]] = []
    horizon = cfg.forecast_horizon_sec

    for forklift_id, forklift_path in paths.items():
        if entities[forklift_id].class_id != cfg.forklift_class_id:
            continue
        for person_id, person_path in paths.items():
            if entities[person_id].class_id != cfg.person_class_id:
                continue
            intersection = _segment_intersection(forklift_path, person_path)
            if intersection is None:
                continue
            point, forklift_position, person_position = intersection
            forklift_time = forklift_position * horizon
            person_time = person_position * horizon
            if abs(forklift_time - person_time) <= cfg.collision_time_window_sec:
                collision_ids.update((forklift_id, person_id))
                collision_points.append(point)

    return collision_ids, collision_points


def update_proximity_pairs(
    entities: Dict[int, EntityState],
    active_ids: Set[int],
    frame_idx: int,
    fps: float,
    cfg: TrackerConfig,
    proximity_starts: Dict[Tuple[int, int], int],
) -> Set[Tuple[int, int]]:
    """Update pair timers and return pairs that have persisted long enough."""
    active_pairs: Set[Tuple[int, int]] = set()
    ids = sorted(active_ids)
    for index, first_id in enumerate(ids):
        first = entities[first_id].latest
        if first is None:
            continue
        for second_id in ids[index + 1:]:
            second = entities[second_id].latest
            if second is None:
                continue
            distance = math.hypot(first[1] - second[1], first[2] - second[2])
            pair = (first_id, second_id)
            if distance < cfg.proximity_threshold_px:
                proximity_starts.setdefault(pair, frame_idx)
                elapsed = (
                    frame_idx - proximity_starts[pair]
                ) / fps if fps > 0 else 0.0
                if elapsed >= cfg.proximity_duration_sec:
                    active_pairs.add(pair)
            else:
                proximity_starts.pop(pair, None)

    valid_pairs = {
        pair for pair in proximity_starts
        if pair[0] in active_ids and pair[1] in active_ids
    }
    for pair in set(proximity_starts) - valid_pairs:
        proximity_starts.pop(pair, None)
    return active_pairs


# ═══════════════════════════════════════════════════════════════════════
#  Rule A – Loitering
# ═══════════════════════════════════════════════════════════════════════

def check_loitering(
    entity: EntityState, current_frame: int, fps: float, cfg: TrackerConfig
) -> Tuple[bool, float]:
    """
    Smoothed-centroid bounding-radius loitering detector.

    Two conditions must be met simultaneously:
      1. The configured history window contains 150 frames by default.
      2. No smoothed centroid has moved more than 40 pixels from the
         first smoothed centroid in that window.

    This mathematically guarantees the person is physically contained within
    a tight bounding radius (standing still or shifting slightly) rather
    than continuously advancing at any pace.

    Returns (triggered, confidence).
    """
    if len(entity.history) < cfg.history_length:
        return False, 0.0

    radius = centroid_radius(entity)
    if radius > cfg.loiter_radius_px:
        return False, 0.0

    confidence = min(
        1.0,
        0.5 + 0.5 * (1.0 - radius / max(cfg.loiter_radius_px, 0.1)),
    )
    return True, round(confidence, 3)


def centroid_radius(entity: EntityState) -> float:
    """Return the maximum distance from the first smoothed centroid."""
    centroids = entity.centroids()
    if len(centroids) < 2:
        return float("inf")
    anchor_x, anchor_y = centroids[0]
    return max(
        math.hypot(x - anchor_x, y - anchor_y)
        for x, y in centroids
    )


def filter_forklift_overlaps(
    xyxys: np.ndarray,
    class_ids: List[int],
    overlap_threshold: float = 0.60,
) -> Set[int]:
    """Return person detection indexes mostly covered by a truck box."""
    truck_boxes = [
        xyxys[index]
        for index, class_id in enumerate(class_ids)
        if class_id == 7
    ]
    rejected: Set[int] = set()
    for index, class_id in enumerate(class_ids):
        if class_id != 0:
            continue
        person = xyxys[index]
        person_area = max(0.0, person[2] - person[0]) * max(
            0.0, person[3] - person[1]
        )
        if person_area <= 0:
            continue
        for truck in truck_boxes:
            intersection_area = (
                max(0.0, min(person[2], truck[2]) - max(person[0], truck[0]))
                * max(0.0, min(person[3], truck[3]) - max(person[1], truck[1]))
            )
            if intersection_area / person_area > overlap_threshold:
                rejected.add(index)
                break
    return rejected


def forklift_proximity_ids(
    entities: Dict[int, EntityState],
    active_ids: Set[int],
    cfg: TrackerConfig,
) -> Set[int]:
    """Return person IDs currently within the safety radius of any forklift."""
    forklift_points = [
        (entity.latest[1], entity.latest[2])
        for entity_id, entity in entities.items()
        if entity_id in active_ids
        and entity.class_id == cfg.forklift_class_id
        and entity.latest is not None
    ]
    unsafe_people: Set[int] = set()
    for entity_id, entity in entities.items():
        if (
            entity_id not in active_ids
            or entity.class_id != cfg.person_class_id
            or entity.latest is None
        ):
            continue
        person_point = (entity.latest[1], entity.latest[2])
        if any(
            math.hypot(person_point[0] - forklift[0], person_point[1] - forklift[1])
            <= cfg.forklift_proximity_px
            for forklift in forklift_points
        ):
            unsafe_people.add(entity_id)
    return unsafe_people


# ═══════════════════════════════════════════════════════════════════════
#  Rule B – Posture Collapse / Fall Detection
# ═══════════════════════════════════════════════════════════════════════

def check_posture_collapse(
    entity: EntityState, fps: float, cfg: TrackerConfig
) -> Tuple[bool, float]:
    """
    Flag when the bounding-box aspect ratio (h/w) drops abruptly by
    > 40 % within a 1-second look-back window.

    Returns (triggered, confidence).
    """
    ratios = entity.aspect_ratios()
    window_len = max(2, int(fps * cfg.fall_window_sec))

    if len(ratios) < window_len:
        return False, 0.0

    window = ratios[-window_len:]
    peak = max(window)

    if peak == 0:
        return False, 0.0

    current = window[-1]
    drop_pct = (peak - current) / peak

    if drop_pct > cfg.fall_ratio_drop:
        conf = min(1.0, 0.6 + drop_pct)
        return True, round(conf, 3)

    return False, 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Rule C – Zone Intrusion
# ═══════════════════════════════════════════════════════════════════════

def check_zone_intrusion(
    entity: EntityState, zone_polygon: np.ndarray
) -> Tuple[bool, float]:
    """
    Flag if the entity's latest centroid lies inside the defined polygon ROI.

    Returns (triggered, confidence).
    """
    latest = entity.latest
    if latest is None:
        return False, 0.0

    _, cx, cy, _, _ = latest
    result = cv2.pointPolygonTest(zone_polygon.reshape(-1, 1, 2), (cx, cy), measureDist=True)

    if result >= 0:
        # Confidence increases the deeper inside the zone
        conf = min(1.0, 0.7 + 0.01 * result)
        return True, round(conf, 3)

    return False, 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Aggregate Evaluator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BehaviorVerdict:
    """Result object returned for each entity per frame."""

    entity_id: int
    behavior: str         # NORMAL | LOITERING | POSTURE_COLLAPSE | ZONE_INTRUSION
    confidence: float
    centroid: Tuple[float, float]
    aspect_ratio: float


def evaluate_entity(
    entity: EntityState,
    current_frame: int,
    fps: float,
    zone_polygon: np.ndarray,
    cfg: TrackerConfig,
    interacting: bool = False,
    forklift_proximity: bool = False,
    collision_predicted: bool = False,
) -> BehaviorVerdict:
    """
    Run all rules against a single entity and return the highest-priority
    verdict. Priority: POSTURE_COLLAPSE > SUSPICIOUS_INTERACTION >
    SUDDEN_ACCELERATION > ZONE_INTRUSION > LOITERING > NORMAL.
    """
    latest = entity.latest
    cx, cy = (latest[1], latest[2]) if latest else (0.0, 0.0)
    ar = (latest[4] / latest[3]) if latest and latest[3] > 0 else 0.0

    if collision_predicted:
        return BehaviorVerdict(
            entity.entity_id,
            "CRITICAL: COLLISION_PREDICTED",
            1.0,
            (cx, cy),
            round(ar, 3),
        )

    if entity.class_id != cfg.person_class_id:
        return BehaviorVerdict(entity.entity_id, "NORMAL", 1.0, (cx, cy), round(ar, 3))

    if forklift_proximity:
        return BehaviorVerdict(
            entity.entity_id,
            "CRITICAL: FORKLIFT_PROXIMITY",
            1.0,
            (cx, cy),
            round(ar, 3),
        )

    # Evaluate rules in priority order
    fall_flag, fall_conf = check_posture_collapse(entity, fps, cfg)
    if fall_flag:
        return BehaviorVerdict(entity.entity_id, "POSTURE_COLLAPSE", fall_conf, (cx, cy), round(ar, 3))

    if interacting:
        return BehaviorVerdict(entity.entity_id, "SUSPICIOUS_INTERACTION", 1.0, (cx, cy), round(ar, 3))

    acceleration_flag, acceleration_conf = check_sudden_acceleration(entity, cfg)
    if acceleration_flag:
        return BehaviorVerdict(entity.entity_id, "SUDDEN_ACCELERATION", acceleration_conf, (cx, cy), round(ar, 3))

    zone_flag, zone_conf = check_zone_intrusion(entity, zone_polygon)
    if zone_flag:
        return BehaviorVerdict(entity.entity_id, "ZONE_INTRUSION", zone_conf, (cx, cy), round(ar, 3))

    loiter_flag, loiter_conf = check_loitering(entity, current_frame, fps, cfg)
    if loiter_flag:
        return BehaviorVerdict(entity.entity_id, "LOITERING", loiter_conf, (cx, cy), round(ar, 3))

    return BehaviorVerdict(entity.entity_id, "NORMAL", 1.0, (cx, cy), round(ar, 3))
