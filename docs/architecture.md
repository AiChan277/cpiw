# PrivacyCam Architecture

## Overview
PrivacyCam is a multi-threaded, asynchronous processing pipeline that performs real-time face anonymization. 
It captures frames from a camera, runs them through an AI model via Intel OpenVINO, tracks faces between frames to maintain high framerates and stable masking, applies the configured anonymization (e.g., blur), and outputs the resulting frames to a virtual camera and optionally a local UI.

## Subsystems

- **Capture**: Interfaces with hardware cameras (OpenCV) to extract raw frames continuously.
- **Detection**: Runs the OpenVINO IR model to find faces and return bounding boxes. This is typically executed on the NPU or GPU to offload the CPU.
- **Tracking**: An object tracker that predicts face positions between detection intervals. This ensures high FPS even if the detection model runs at a lower framerate.
- **Masking**: Creates bitmasks corresponding to tracked face regions, including features like expanding the region and feathering edges.
- **Anonymization**: Applies the blur, pixelation, or solid color algorithms to the regions defined by the mask.
- **Output**: Interfaces with `pyvirtualcam` to broadcast the final frame as a virtual webcam for use in other apps (Zoom, Discord).
- **Pipeline**: The orchestrator that handles threading, queues, and concurrency between these stages.

## Data Flow
`Camera -> Queue -> Detection (Async) / Tracker -> Queue -> Mask & Blur -> Queue -> Virtual Camera`

## Threading Model
PrivacyCam employs an asynchronous pipeline where stages are connected via limited-size thread-safe queues. This allows the capture thread to remain unblocked by the detection thread. Dropping frames gracefully under high load is handled by the pipeline coordinator.
