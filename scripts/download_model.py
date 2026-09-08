#!/usr/bin/env python3
"""Download the OpenVINO face-detection model for PrivacyCam.

Downloads face-detection-adas-0001 (FP16) from Intel's Open Model Zoo.
This is a lightweight face detection model optimized for Intel NPU/GPU/CPU.

Usage:
    py scripts/download_model.py
    py scripts/download_model.py --output-dir models
    py scripts/download_model.py --precision FP32
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


# -- Model registry -----------------------------------------------------------

MODELS = {
    "face-detection-adas-0001": {
        "description": "Face Detection ADAS 0001 - lightweight face detector",
        "base_url": (
            "https://storage.openvinotoolkit.org/repositories/"
            "open_model_zoo/2023.0/models_bin/1/"
            "face-detection-adas-0001/{precision}/"
        ),
        "files": [
            "face-detection-adas-0001.xml",
            "face-detection-adas-0001.bin",
        ],
        "precisions": ["FP16", "FP32", "FP16-INT8"],
        "default_precision": "FP16",
    },
}


# -- Download helpers ----------------------------------------------------------

def _progress_hook(block_num, block_size, total_size):
    """Print a simple progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar_len = 40
        filled = bar_len * pct // 100
        bar = "#" * filled + "-" * (bar_len - filled)
        size_mb = total_size / (1024 * 1024)
        done_mb = min(downloaded, total_size) / (1024 * 1024)
        sys.stdout.write(
            f"\r  [{bar}] {pct:3d}%  {done_mb:.1f}/{size_mb:.1f} MB"
        )
    else:
        sys.stdout.write(f"\r  Downloaded {downloaded / 1024:.0f} KB")
    sys.stdout.flush()


def download_file(url, dest, retries=3):
    """Download a single file with retries and progress."""
    for attempt in range(1, retries + 1):
        try:
            print(f"  >> {url}")
            urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
            print()  # newline after progress bar
            return True
        except urllib.error.URLError as e:
            print(f"\n  [FAIL] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
        except Exception as e:
            print(f"\n  [FAIL] Unexpected error: {e}")
            if dest.exists():
                dest.unlink()
            return False

    return False


def validate_file(path, min_size=1024):
    """Basic validation - file exists and is not suspiciously small."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < min_size:
        print(f"  [!] {path.name} is only {size} bytes - may be corrupt")
        return False
    return True


# -- Main download logic -------------------------------------------------------

def download_model(model_name="face-detection-adas-0001", output_dir="models",
                   precision=None):
    """Download a model from the Open Model Zoo."""

    if model_name not in MODELS:
        print(f"[FAIL] Unknown model: {model_name}")
        print(f"  Available: {', '.join(MODELS.keys())}")
        return False

    spec = MODELS[model_name]
    prec = precision or spec["default_precision"]

    if prec not in spec["precisions"]:
        print(f"[FAIL] Precision {prec} not available for {model_name}")
        print(f"  Available: {', '.join(spec['precisions'])}")
        return False

    # Build output path:  models/face-detection-adas-0001/FP16/
    dest_dir = Path(output_dir) / model_name / prec
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_url = spec["base_url"].format(precision=prec)

    print("=" * 52)
    print("  PrivacyCam Model Downloader")
    print("=" * 52)
    print(f"  Model     : {model_name}")
    print(f"  Precision : {prec}")
    print(f"  Output    : {dest_dir}")
    print("=" * 52)
    print()

    all_ok = True
    for filename in spec["files"]:
        dest_path = dest_dir / filename
        url = base_url + filename

        # Skip if already downloaded and valid
        if dest_path.exists() and validate_file(dest_path):
            size_kb = dest_path.stat().st_size / 1024
            print(f"  [OK] {filename} already exists ({size_kb:.0f} KB)")
            continue

        print(f"  Downloading {filename}...")
        if download_file(url, dest_path):
            if validate_file(dest_path):
                size_kb = dest_path.stat().st_size / 1024
                print(f"  [OK] {filename} ({size_kb:.0f} KB)")
            else:
                print(f"  [FAIL] {filename} validation failed")
                all_ok = False
        else:
            print(f"  [FAIL] Failed to download {filename}")
            all_ok = False

    if all_ok:
        print()
        print(f"[OK] Model downloaded successfully to: {dest_dir}")
        print()
        print(f"  You can now run PrivacyCam:")
        print(f"    py main.py")
        print()
        print(f"  Or specify the model path explicitly:")
        print(f"    py -m privacycam run --config configs/default.yaml")
    else:
        print()
        print(f"[FAIL] Some files failed to download. Please try again.")

    return all_ok


# -- Also try omz_downloader as alternative ------------------------------------

def try_omz_downloader(model_name, output_dir):
    """Try using OpenVINO's omz_downloader if available."""
    import shutil
    import subprocess

    omz = shutil.which("omz_downloader")
    if omz is None:
        return False

    print(f"Found omz_downloader: {omz}")
    print(f"Downloading {model_name} via Model Zoo tools...")

    result = subprocess.run(
        [omz, "--name", model_name, "-o", output_dir],
        capture_output=False,
    )
    return result.returncode == 0


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download face detection model for PrivacyCam",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models/)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="face-detection-adas-0001",
        help="Model name (default: face-detection-adas-0001)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["FP16", "FP32", "FP16-INT8"],
        help="Model precision (default: FP16)",
    )
    parser.add_argument(
        "--use-omz",
        action="store_true",
        help="Try using omz_downloader first (requires openvino-dev)",
    )
    args = parser.parse_args()

    if args.use_omz:
        if try_omz_downloader(args.model_name, args.output_dir):
            return

    success = download_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        precision=args.precision,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
