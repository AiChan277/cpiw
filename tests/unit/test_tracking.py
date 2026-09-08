import pytest
from privacycam.detection.detection import BoundingBox, FaceDetection
from privacycam.tracking.tracker import FaceTracker
from privacycam.tracking.track import Track, TrackState
from privacycam.tracking.association import compute_iou, greedy_associate, compute_iou_matrix
from privacycam.tracking.presets import TRACKING_PRESETS


def test_new_track_creation():
    tracker = FaceTracker()
    bbox = BoundingBox(100.0, 100.0, 200.0, 200.0)
    det = FaceDetection(bounding_box=bbox, confidence=0.95)
    
    tracks = tracker.update([det], frame_id=0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert tracks[0].state == TrackState.ACTIVE
    assert tracks[0].hit_count == 1


def test_track_association_iou():
    tracker = FaceTracker(smoothing_alpha=1.0)
    b0 = BoundingBox(100.0, 100.0, 200.0, 200.0)
    det0 = FaceDetection(bounding_box=b0, confidence=0.95)
    tracks = tracker.update([det0], frame_id=0)
    assert len(tracks) == 1
    
    # Slight movement with high IoU
    b1 = BoundingBox(105.0, 102.0, 205.0, 202.0)
    det1 = FaceDetection(bounding_box=b1, confidence=0.94)
    tracks = tracker.update([det1], frame_id=1)
    
    assert len(tracks) == 1
    assert tracks[0].track_id == 0  # Same identity
    assert tracks[0].hit_count == 2


def test_distance_fallback_association():
    """Fast head movement where IoU drops below threshold but is within distance reach."""
    tracker = FaceTracker(iou_threshold=0.30, distance_threshold=2.0)
    b0 = BoundingBox(100.0, 100.0, 200.0, 200.0)
    tracks = tracker.update([FaceDetection(bounding_box=b0, confidence=0.9)], frame_id=0)
    
    # Fast move: moved 70px to the right. Face size = 100px.
    # IoU = 30*100 / (10000 + 10000 - 3000) = 3000 / 17000 = ~0.176 (below 0.30)
    # Center distance: (150+70) - 150 = 70px < 2.0 * 100px (within reach)
    b_fast = BoundingBox(170.0, 100.0, 270.0, 200.0)
    tracks = tracker.update([FaceDetection(bounding_box=b_fast, confidence=0.9)], frame_id=1)
    
    assert len(tracks) == 1
    assert tracks[0].track_id == 0  # Preserved identity via distance fallback!


def test_zero_drift_predict():
    """Zero-drift hold prevents bounding box from drifting away when detections are skipped."""
    tracker = FaceTracker()
    b0 = BoundingBox(100.0, 100.0, 200.0, 200.0)
    tracker.update([FaceDetection(bounding_box=b0, confidence=0.9)], frame_id=0)
    
    # Skipped detection frame (detections=None)
    tracks = tracker.update(None, frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].bounding_box.x1 == 100.0
    assert tracks[0].bounding_box.y1 == 100.0


def test_set_smoothing_alpha():
    """Setting smoothing alpha dynamically propagates to tracker and existing tracks."""
    tracker = FaceTracker(smoothing_alpha=0.70)
    b0 = BoundingBox(100.0, 100.0, 200.0, 200.0)
    tracker.update([FaceDetection(bounding_box=b0, confidence=0.9)], frame_id=0)
    
    tracker.set_smoothing_alpha(0.40)
    assert tracker.smoothing_alpha == 0.40
    assert tracker._tracks[0].smoothing_alpha == 0.40


def test_anti_ghost_deduplication():
    """Overlapping duplicate tracks (> 0.40 IoU) are deduplicated."""
    tracker = FaceTracker()
    b0 = BoundingBox(100.0, 100.0, 200.0, 200.0)
    tracker.update([FaceDetection(bounding_box=b0, confidence=0.9)], frame_id=0)
    
    # Artificially inject duplicate track with high overlap
    dup_track = tracker._create_track(FaceDetection(bounding_box=BoundingBox(105.0, 105.0, 205.0, 205.0), confidence=0.85))
    tracker._tracks.append(dup_track)
    assert len(tracker._tracks) == 2
    
    tracker._deduplicate_overlapping_tracks()
    tracker._prune_dead_tracks()
    assert len(tracker.get_active_tracks()) == 1


def test_presets_exist():
    """Verify all defined presets have the necessary keys and realistic ranges."""
    required_keys = {"name", "interval", "smoothing", "persistence", "distance", "iou", "desc"}
    assert "smooth" in TRACKING_PRESETS
    assert "responsive" in TRACKING_PRESETS
    assert "motion" in TRACKING_PRESETS
    assert "stationary" in TRACKING_PRESETS
    
    for key, p in TRACKING_PRESETS.items():
        assert required_keys.issubset(p.keys())
        assert 1 <= p["interval"] <= 6
        assert 0.10 <= p["smoothing"] <= 1.0
        assert 5 <= p["persistence"] <= 60
        assert 0.5 <= p["distance"] <= 3.0
        assert 0.05 <= p["iou"] <= 0.50


def test_tracker_reset():
    tracker = FaceTracker()
    b0 = BoundingBox(100.0, 100.0, 200.0, 200.0)
    tracker.update([FaceDetection(bounding_box=b0, confidence=0.9)], frame_id=0)
    assert len(tracker.get_active_tracks()) == 1
    
    tracker.reset()
    assert len(tracker.get_active_tracks()) == 0
