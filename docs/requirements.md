# Requirements

## Functional Requirements
- Capture video from a local webcam.
- Detect faces in real-time.
- Track faces continuously across frames.
- Apply anonymization effects: Blur, Pixelate, Solid Color.
- Output anonymized video to a Virtual Camera.
- Provide a Graphical User Interface (GUI) for configuration and preview.
- Provide a Command Line Interface (CLI) for headless/daemon execution.
- Configurable settings via YAML files.

## Non-Functional Requirements
- Privacy: 100% local processing. No internet access needed. No data storage.
- Performance: Minimum 30 FPS targeting real-time latency.
- Hardware Acceleration: Leverage Intel NPU, GPU, and CPU via OpenVINO.
- Robustness: Graceful degradation (frame drops) when system is overloaded to maintain latency rather than accumulating lag.
