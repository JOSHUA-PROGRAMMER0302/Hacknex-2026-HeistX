# Hacknex-2026-HeistX
Hacknext 2026 Hackathon 

## Vision

The Vision module provides YOLOv8 + BoT-SORT tracking for people and
forklifts, with:

- Five-frame centroid smoothing and radius-based loitering detection
- Occlusion-tolerant tracking with a 120-frame BoT-SORT buffer
- Forklift/person overlap rejection and proximity safety alerts
- Predictive three-second collision forecasting from ten-frame velocity vectors
- Annotated MP4 recording and WebSocket telemetry

Run the pipeline with:

```bash
pip install -r requirements.txt
python sentinel_edge.py
```

The annotated output video is written to `output_videos/`.
