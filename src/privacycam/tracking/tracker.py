from typing import Optional, List
from privacycam.detection.detection import FaceDetection
from privacycam.tracking.track import Track, TrackState
from privacycam.tracking.association import compute_iou_matrix, greedy_associate

class FaceTracker:
    """Multi-face tracker maintaining identity and smoothing bounding boxes across frames."""

    def __init__(
        self,
        max_lost_frames: int = 25,
        iou_threshold: float = 0.20,
        min_hits: int = 1,
        smoothing_alpha: float = 0.70,
        distance_threshold: float = 1.50,
    ):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.smoothing_alpha = smoothing_alpha
        self.distance_threshold = distance_threshold
        self._next_id: int = 0
        self._tracks: List[Track] = []

    def update(self, detections: Optional[List[FaceDetection]], frame_id: int) -> List[Track]:
        """Updates tracks. If detections is None, advances existing tracks via prediction."""
        # Frame where detection was intentionally skipped or pending
        if detections is None:
            for track in self._tracks:
                track.predict()
            return self.get_active_tracks()

        # Step 1: Predict positions for existing tracks (once per cycle)
        for track in self._tracks:
            track.predict()

        # Step 2: Associate tracks with new detections using IoU + distance fallback
        iou_matrix = compute_iou_matrix(self._tracks, detections)
        matches, unmatched_tracks, unmatched_detections = greedy_associate(
            iou_matrix,
            threshold=self.iou_threshold,
            tracks=self._tracks,
            detections=detections,
            distance_threshold=self.distance_threshold,
        )

        # Step 3: Update matched tracks with EMA smoothing
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(detections[det_idx], self.min_hits)

        # Step 4: Handle unmatched tracks (graceful stationary hold)
        for track_idx in unmatched_tracks:
            track = self._tracks[track_idx]
            track.mark_lost()
            if track.frames_since_seen > self.max_lost_frames:
                track.mark_dead()

        # Step 5: Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._tracks.append(self._create_track(detections[det_idx]))

        # Step 5.5: Deduplicate overlapping tracks (Anti-Ghost NMS)
        self._deduplicate_overlapping_tracks()

        # Step 6: Prune dead tracks
        self._prune_dead_tracks()

        return self.get_active_tracks()

    def _deduplicate_overlapping_tracks(self) -> None:
        """Suppresses duplicate tracks tracking the same physical face."""
        from privacycam.tracking.association import compute_iou
        n = len(self._tracks)
        for i in range(n):
            if self._tracks[i].is_dead:
                continue
            for j in range(i + 1, n):
                if self._tracks[j].is_dead:
                    continue
                iou = compute_iou(self._tracks[i].bounding_box, self._tracks[j].bounding_box)
                if iou > 0.40:
                    # Keep track with more hits or active state
                    if self._tracks[i].hit_count >= self._tracks[j].hit_count:
                        self._tracks[j].mark_dead()
                    else:
                        self._tracks[i].mark_dead()

    def get_active_tracks(self) -> List[Track]:
        """Returns all currently active tracks."""
        return [t for t in self._tracks if t.state in (TrackState.ACTIVE, TrackState.NEW, TrackState.LOST)]

    def reset(self) -> None:
        """Clears all tracks."""
        self._tracks.clear()
        self._next_id = 0

    def set_smoothing_alpha(self, alpha: float) -> None:
        """Updates smoothing alpha on tracker and all existing tracks."""
        self.smoothing_alpha = alpha
        for t in self._tracks:
            t.smoothing_alpha = alpha

    def _create_track(self, detection: FaceDetection) -> Track:
        track = Track(
            track_id=self._next_id,
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            smoothing_alpha=self.smoothing_alpha,
        )
        self._next_id += 1
        track.hit_count = 1
        track.age = 1
        track.state = TrackState.ACTIVE
        return track

    def _prune_dead_tracks(self) -> None:
        self._tracks = [t for t in self._tracks if not t.is_dead]
