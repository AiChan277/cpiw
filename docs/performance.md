# Performance

## Targets
- **Throughput**: 30+ FPS minimum on modern hardware.
- **Latency**: <50ms end-to-to processing time.

## Optimization Strategies
- **Hardware Inference**: Offload ML models to dedicated Neural Processing Units (NPUs) or integrated GPUs using OpenVINO.
- **Detection Interval**: Run full face detection only every N frames (e.g., every 3rd frame).
- **Tracking**: Use extremely fast CPU-bound tracking (e.g., Optical Flow or simple Kalman filters) for frames in-between full detections.
- **Asynchronous Pipeline**: Decouple capture, processing, and output into separate threads with bounded queues to prevent latency build-up.

## Metrics Collected
The internal telemetry tracks:
- Capture Latency
- Inference Time
- Tracking Time
- Anonymization Application Time
- Total Pipeline Latency
- Output Framerate
