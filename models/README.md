# OpenVINO Models

This directory is intended to store the OpenVINO IR models (`.xml` and `.bin` files) required for Face Detection.

## Required Model
PrivacyCam is configured by default to use the `face-detection-adas-0001` model. 

## How to Download
You can download the models using the OpenVINO Model Downloader or fetch them manually from Intel's Open Model Zoo.

```bash
# If a download script is provided:
python tools/download_model.py
```

Expected files after download:
- `face-detection-adas-0001.xml`
- `face-detection-adas-0001.bin`
