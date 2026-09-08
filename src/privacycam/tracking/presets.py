"""Tracking presets configuration for PrivacyCam."""

from typing import Dict, Any

TRACKING_PRESETS: Dict[str, Dict[str, Any]] = {
    "smooth": {
        "name": "🛡️ Smooth & Anti-Jitter",
        "interval": 2,
        "smoothing": 0.65,
        "persistence": 25,
        "distance": 1.5,
        "iou": 0.20,
        "desc": "Best for webcams and calls. Eliminates trembling & jitter.",
    },
    "responsive": {
        "name": "⚡ Ultra Responsive",
        "interval": 1,
        "smoothing": 0.90,
        "persistence": 15,
        "distance": 1.3,
        "iou": 0.25,
        "desc": "Fastest lock. Snaps instantly to head movements.",
    },
    "motion": {
        "name": "🏃 High Fast-Motion",
        "interval": 1,
        "smoothing": 0.80,
        "persistence": 30,
        "distance": 2.2,
        "iou": 0.15,
        "desc": "Maintains lock during rapid head turns and quick motion.",
    },
    "stationary": {
        "name": "⚓ Stationary Anchor",
        "interval": 2,
        "smoothing": 0.45,
        "persistence": 35,
        "distance": 1.1,
        "iou": 0.25,
        "desc": "Heavy damping. Solid position hold with zero drift.",
    },
}
