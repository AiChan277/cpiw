import numpy as np
from privacycam.detection.detection import BoundingBox, FaceDetection
from privacycam.tracking.track import Track

def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    x_left = max(box_a.x1, box_b.x1)
    y_top = max(box_a.y1, box_b.y1)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
    box_b_area = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)

    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
    return iou

def compute_iou_matrix(tracks: list[Track], detections: list[FaceDetection]) -> np.ndarray:
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        box = track.bounding_box
        for j, detection in enumerate(detections):
            matrix[i, j] = compute_iou(box, detection.bounding_box)
    return matrix

def compute_center_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    center_a_x = (box_a.x1 + box_a.x2) / 2
    center_a_y = (box_a.y1 + box_a.y2) / 2
    center_b_x = (box_b.x1 + box_b.x2) / 2
    center_b_y = (box_b.y1 + box_b.y2) / 2
    return float(np.sqrt((center_a_x - center_b_x)**2 + (center_a_y - center_b_y)**2))

def greedy_associate(
    iou_matrix: np.ndarray,
    threshold: float = 0.25,
    tracks: list[Track] = None,
    detections: list[FaceDetection] = None,
    distance_threshold: float = 1.50,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    matches = []
    num_tracks = iou_matrix.shape[0] if iou_matrix.ndim == 2 else 0
    num_dets = iou_matrix.shape[1] if iou_matrix.ndim == 2 else 0
    unmatched_tracks = list(range(num_tracks))
    unmatched_detections = list(range(num_dets))

    if num_tracks == 0 or num_dets == 0:
        return matches, unmatched_tracks, unmatched_detections

    # Stage 1: High-overlap IoU matching
    iou_copy = iou_matrix.copy()
    while True:
        max_idx = np.unravel_index(np.argmax(iou_copy, axis=None), iou_copy.shape)
        max_iou = iou_copy[max_idx]

        if max_iou < threshold:
            break

        track_idx, det_idx = int(max_idx[0]), int(max_idx[1])
        matches.append((track_idx, det_idx))
        if track_idx in unmatched_tracks:
            unmatched_tracks.remove(track_idx)
        if det_idx in unmatched_detections:
            unmatched_detections.remove(det_idx)

        iou_copy[track_idx, :] = -1.0
        iou_copy[:, det_idx] = -1.0

    # Stage 2: Distance-assisted matching for fast movement, head tilts, and nods
    if tracks and detections and unmatched_tracks and unmatched_detections:
        for t_idx in list(unmatched_tracks):
            t_box = tracks[t_idx].bounding_box
            best_d_idx = None
            best_dist = float("inf")
            # Face center distance threshold configurable
            max_allowed = max(t_box.width, t_box.height) * distance_threshold

            for d_idx in list(unmatched_detections):
                d_box = detections[d_idx].bounding_box
                dist = compute_center_distance(t_box, d_box)
                if dist <= max_allowed and dist < best_dist:
                    best_dist = dist
                    best_d_idx = d_idx

            if best_d_idx is not None:
                matches.append((t_idx, best_d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(best_d_idx)

    return matches, unmatched_tracks, unmatched_detections
