# PrivacyCam 🛡️🎥

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.0%2B-purple)
![Intel NPU](https://img.shields.io/badge/Intel-NPU_Accelerated-blue)
![Privacy](https://img.shields.io/badge/Privacy-First-brightgreen)

PrivacyCam is a real-time, privacy-preserving face anonymization camera application. It seamlessly captures video, detects and tracks faces, anonymizes them, and outputs the result as a virtual webcam. Everything runs entirely on-device, leveraging OpenVINO for hardware acceleration on Intel CPUs, GPUs, and NPUs. No cloud, no tracking, just privacy.

## Architecture

```text
+---------+    +-----------+    +----------+    +---------+    +---------------+    +--------+
| Camera  | -> | Detection | -> | Tracking | -> | Masking | -> | Anonymization | -> | Output |
+---------+    +-----------+    +----------+    +---------+    +---------------+    +--------+
```

## Features
- Real-time face detection & tracking
- Multiple anonymization modes (Blur, Pixelate, Solid)
- Seamless integration as a virtual camera (OBS, Zoom, Teams, Discord)
- Cross-platform support via PySide6
- Intel OpenVINO acceleration
- Dedicated NPU (Neural Processing Unit) support for extreme efficiency
- Advanced multi-stage asynchronous processing pipeline
- Zero-cloud, absolute privacy guarantees
- Real-time performance telemetry
- CLI and GUI modes
- Hot-swappable hardware backends (CPU, GPU, NPU)
- Pluggable tracker architecture
- Extensible mask generators
- Configuration via YAML profiles
- Low latency & minimal frame drop architecture
- Comprehensive logging and diagnostics
- Developer-friendly robust API
- Built-in metrics overlay
- Graceful degradation on high load
- Multi-camera input selection

## Engineering Highlights
PrivacyCam is built with modern Python 3.10+ leveraging type hints, asynchronous data flow, dataclasses, and strict OOP practices. The application's core processing loop operates on a multi-threaded asynchronous pipeline ensuring optimal throughput. The AI inference uses Intel OpenVINO for optimal inference speed, allowing the offloading of heavy ML tasks to dedicated hardware (NPU).

## Supported Hardware
PrivacyCam runs entirely on-device. With OpenVINO integration, inference can be executed on:
- **Intel CPU** (Broad support, general fallback)
- **Intel GPU** (Integrated Iris Xe, Arc Graphics)
- **Intel NPU** (Meteor Lake and newer processors for highly efficient AI processing)

## Benchmarks
| Device     | FPS | Latency (ms) | P95 Latency (ms) | Memory (MB) |
|------------|-----|--------------|------------------|-------------|
| Intel CPU  | ~30 | ~40          | ~55              | ~200        |
| Intel GPU  | ~45 | ~25          | ~35              | ~180        |
| Intel NPU  | ~60 | ~15          | ~20              | ~150        |

*(Note: Benchmarks are illustrative and heavily depend on specific hardware and model used)*

## Screenshots
*(UI and Output screenshots placeholder)*

## Installation

### Prerequisites
- Python 3.10+
- Virtual Camera Driver (e.g., OBS Virtual Camera on Windows/Mac, v4l2loopback on Linux)

### Setup
```bash
git clone https://github.com/yourorg/cpiw-main.git
cd cpiw-main
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -e .
```

### Download Model
You will need an OpenVINO IR model for face detection (e.g., `face-detection-adas-0001`). Place the `.xml` and `.bin` files in the `models/` directory or use a provided download script if available.

## Usage

### GUI
Run the visual application with configuration controls:
```bash
privacycam gui --config configs/default.yaml
```

### CLI
Run purely in the terminal as a background daemon:
```bash
privacycam cli --config configs/performance.yaml
```

## Configuration
PrivacyCam uses YAML configuration files to customize every stage of the pipeline.
- `configs/default.yaml`: Balanced settings for standard usage.
- `configs/development.yaml`: Debug mode with overlays and verbose logging.
- `configs/performance.yaml`: Minimized overhead for extreme framerates.

## Privacy Guarantees
- **No Cloud Required**: All processing happens on your local hardware.
- **No Video Recording**: Frames are processed and immediately discarded.
- **No Face Recognition**: The AI only detects the *presence* of a face; no identity vectors or embeddings are created, stored, or sent.
- **No Analytics**: We do not phone home usage telemetry or data.

## Limitations
- Virtual camera software required natively.
- Relies on standard face detection models (extreme angles or heavy occlusions may occasionally drop).

## Roadmap
- Custom lightweight model finetuning
- Additional hardware backend support (ONNX Runtime, TensorRT)
- More anonymization algorithms
- Full test coverage suite

## License
MIT License. See `LICENSE` for details.
