"""
Sentinel-Edge — Main Vision Pipeline
═════════════════════════════════════
High-performance person tracking + behaviour analysis loop.

Run:
    python sentinel_edge.py                   # uses sample_feed.mp4 or webcam 0
    python sentinel_edge.py --source myvid.mp4
    python sentinel_edge.py --source 0        # force webcam
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from behavior_engine import (
    BehaviorVerdict,
    EntityState,
    evaluate_entity,
    filter_forklift_overlaps,
    forklift_proximity_ids,
    predict_collisions,
    update_proximity_pairs,
)
from config import TrackerConfig
from hud_renderer import (
    draw_breadcrumbs,
    draw_entity_box,
    draw_hud,
    draw_interaction_tethers,
    draw_forklift_safety_zones,
    draw_predicted_paths,
    draw_zone,
)
from telemetry import TelemetryClient

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-22s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.main")


# ═══════════════════════════════════════════════════════════════════════
#  Timestamp Formatter
# ═══════════════════════════════════════════════════════════════════════

def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """Convert a frame index to MM:SS.ms format."""
    if fps <= 0:
        return "00:00.000"
    total_ms = int((frame_idx / fps) * 1000)
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, ms = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


# ═══════════════════════════════════════════════════════════════════════
#  Video Source Resolver
# ═══════════════════════════════════════════════════════════════════════

def open_video_source(cfg: TrackerConfig, override: str | None = None) -> cv2.VideoCapture:
    """
    Try the explicit override first, then cfg.video_source, then webcam.
    Raises RuntimeError if nothing works.
    """
    candidates = []
    if override is not None:
        candidates.append(override)
    candidates.append(cfg.video_source)
    candidates.append(cfg.webcam_fallback)

    for src in candidates:
        # Attempt to open as int (webcam index) or string (file path)
        try:
            idx = int(src)
            cap = cv2.VideoCapture(idx)
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(str(src))

        if cap.isOpened():
            logger.info("Opened video source: %s", src)
            return cap
        cap.release()
        logger.warning("Could not open source: %s", src)

    raise RuntimeError("No valid video source found.")


def create_output_writer(
    cfg: TrackerConfig,
    frame_w: int,
    frame_h: int,
    fps: float,
) -> tuple[cv2.VideoWriter, str]:
    """Create the annotated-video folder and an MP4 writer for this run."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    filename = f"sentinel_edge_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    output_path = os.path.join(cfg.output_dir, filename)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    logger.info("Saving annotated video to: %s", output_path)
    return writer, output_path


def reject_forklift_overlaps(predictor):
    """Remove person boxes covered by forklifts before BoT-SORT receives them."""
    for result_index, result in enumerate(predictor.results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        class_ids = boxes.cls.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().numpy()
        rejected = filter_forklift_overlaps(xyxys, class_ids)
        if rejected:
            keep = [index for index in range(len(boxes)) if index not in rejected]
            predictor.results[result_index].boxes = boxes[keep]


# ═══════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: TrackerConfig, source_override: str | None = None):
    """Core frame loop."""

    # ── YOLO model ───────────────────────────────────────────────────
    logger.info("Loading YOLO model: %s", cfg.yolo_model)
    model = YOLO(cfg.yolo_model)
    model.add_callback("on_predict_postprocess_end", reject_forklift_overlaps)

    # ── Video capture ────────────────────────────────────────────────
    cap = open_video_source(cfg, source_override)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Frame geometry: %dx%d @ %.1f FPS", frame_w, frame_h, native_fps)
    output_writer, output_path = create_output_writer(
        cfg, frame_w, frame_h, native_fps
    )

    zone_polygon = cfg.zone_polygon_px(frame_w, frame_h)

    # ── Entity state registry ────────────────────────────────────────
    entities: Dict[int, EntityState] = {}
    proximity_starts: Dict[Tuple[int, int], int] = {}

    # ── Telemetry client ─────────────────────────────────────────────
    telem = TelemetryClient(cfg)
    telem.start()

    # ── FPS counter ──────────────────────────────────────────────────
    fps_counter_start = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    last_telemetry_time = time.perf_counter()
    frame_idx = 0

    logger.info("Pipeline running. Press 'q' in the display window to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            # ── YOLO + BoT-SORT ───────────────────────────────────
            results = model.track(
                source=frame,
                persist=True,
                tracker=cfg.tracker_yaml,
                classes=cfg.detection_classes,
                conf=0.4,
                verbose=False,
            )

            # ── Parse detections ─────────────────────────────────────
            current_ids: set = set()
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.int().cpu().tolist()
                xywhs = boxes.xywh.cpu().numpy()    # (cx, cy, w, h)
                class_ids = boxes.cls.int().cpu().tolist()
                xyxys = boxes.xyxy.cpu().numpy()
                rejected_indexes = filter_forklift_overlaps(xyxys, class_ids)

                for index, (track_id, xywh, class_id) in enumerate(
                    zip(ids, xywhs, class_ids)
                ):
                    if index in rejected_indexes:
                        continue
                    cx, cy, w, h = xywh
                    current_ids.add(track_id)

                    if track_id not in entities:
                        entities[track_id] = EntityState(
                            entity_id=track_id,
                            first_seen_frame=frame_idx,
                        )
                        # Adjust history maxlen from config
                        entities[track_id].history = deque(
                            maxlen=cfg.history_length
                        )
                        entities[track_id].velocity_history = deque(
                            maxlen=cfg.velocity_history_length
                        )
                        entities[track_id].raw_centroids = deque(
                            maxlen=cfg.centroid_smoothing_frames
                        )
                    entities[track_id].class_id = class_id

                    entities[track_id].update(
                        frame_idx, cx, cy, w, h, native_fps
                    )

            # ── Prune stale entities ─────────────────────────────────
            # Match the tracker's track_buffer (120 frames).
            # Use 4s to give a small grace period beyond the tracker.
            stale_threshold = int(native_fps * 4)
            stale_ids = [
                eid for eid, e in entities.items()
                if e.latest and (frame_idx - e.latest[0]) > stale_threshold
                and eid not in current_ids
            ]
            for eid in stale_ids:
                del entities[eid]  # frees the deque → no memory leak

            active_person_ids = {
                eid for eid in current_ids
                if entities[eid].class_id == cfg.person_class_id
            }
            interacting_pairs = update_proximity_pairs(
                entities, active_person_ids, frame_idx, native_fps, cfg, proximity_starts
            )
            interacting_ids: Set[int] = {
                entity_id
                for pair in interacting_pairs
                for entity_id in pair
            }
            forklift_proximity = forklift_proximity_ids(entities, current_ids, cfg)
            collision_ids, collision_points = predict_collisions(
                entities, current_ids, cfg
            )

            # ── Behaviour evaluation ─────────────────────────────────
            verdicts: List[BehaviorVerdict] = []
            anomalies: List[BehaviorVerdict] = []

            for eid in current_ids:
                entity = entities[eid]
                verdict = evaluate_entity(
                    entity, frame_idx, native_fps, zone_polygon, cfg,
                    interacting=eid in interacting_ids,
                    forklift_proximity=eid in forklift_proximity,
                    collision_predicted=eid in collision_ids,
                )
                verdicts.append(verdict)
                if verdict.behavior != "NORMAL":
                    anomalies.append(verdict)

            # ── Drawing ──────────────────────────────────────────────
            draw_zone(frame, zone_polygon, cfg)
            draw_forklift_safety_zones(frame, entities, current_ids, cfg)
            draw_predicted_paths(
                frame, entities, current_ids, collision_points, cfg, frame_idx
            )
            draw_interaction_tethers(frame, interacting_pairs, entities)

            for eid in current_ids:
                entity = entities[eid]
                draw_breadcrumbs(frame, entity, native_fps, cfg)

            for verdict in verdicts:
                entity = entities.get(verdict.entity_id)
                if entity:
                    draw_entity_box(frame, entity, verdict, cfg)

            # ── FPS calculation ──────────────────────────────────────
            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_counter_start
            if elapsed >= 0.5:
                display_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_counter_start = time.perf_counter()

            timestamp_str = frame_to_timestamp(frame_idx, native_fps)
            draw_hud(frame, display_fps, len(current_ids),
                     timestamp_str, anomalies, cfg)

            output_writer.write(frame)

            # ── Telemetry dispatch ───────────────────────────────────
            now = time.perf_counter()
            periodic_due = (now - last_telemetry_time) >= cfg.telemetry_interval_sec

            for v in verdicts:
                if v.behavior != "NORMAL" or periodic_due:
                    telem.send({
                        "timestamp": timestamp_str,
                        "entity_id": v.entity_id,
                        "behavior": v.behavior,
                        "confidence": v.confidence,
                        "centroid": list(v.centroid),
                        "aspect_ratio": v.aspect_ratio,
                    })

            if periodic_due:
                last_telemetry_time = now

            # ── Display ──────────────────────────────────────────────
            cv2.imshow(cfg.window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User pressed 'q'. Exiting.")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        output_writer.release()
        cv2.destroyAllWindows()
        telem.stop()
        logger.info("Pipeline shut down cleanly. Video saved to %s", output_path)


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-Edge AI Vision Node — Behaviour Detection Pipeline",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video file path or webcam index (default: sample_feed.mp4 → webcam 0)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLO model weight file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--ws", type=str, default="ws://localhost:8080/events",
        help="WebSocket telemetry endpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output_videos",
        help="Folder for annotated output videos (default: output_videos)",
    )
    args = parser.parse_args()

    cfg = TrackerConfig(
        yolo_model=args.model,
        websocket_uri=args.ws,
        output_dir=args.output_dir,
    )
    run_pipeline(cfg, source_override=args.source)


if __name__ == "__main__":
    main()
