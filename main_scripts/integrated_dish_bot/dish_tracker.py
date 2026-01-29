"""
Dish Tracker Module
Extracted from dish_detector_groupme.py
Handles YOLO detection, dish tracking, and ID assignment
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional


class DishTracker:
    """Track dishes detected by YOLO model"""

    def __init__(self):
        """Initialize dish tracker"""
        # Configuration
        self.TRACK_EXPIRY_TIME = 15.0  # Remove track if not seen for 15 seconds
        self.POSITION_THRESHOLD = 150  # Pixels - max distance to match same dish
        self.STATIONARY_THRESHOLD = 20.0  # Seconds before considering stationary
        self.POSITION_STABILITY_THRESHOLD = 50  # Pixels - max movement to consider stationary
        self.IOU_THRESHOLD = 0.2  # Minimum IoU for matching

        # State
        self.tracked_dishes = {}  # {dish_id: {...tracking data...}}
        self.next_id = 1
        self.new_dishes_this_frame = []  # List of dish IDs created this frame

    def calculate_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centers"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def calculate_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def match_detections_to_tracks(self, detections, current_time: float) -> Dict[int, int]:
        """
        Match current detections to existing tracks or create new ones

        Args:
            detections: YOLO detection boxes
            current_time: Current timestamp

        Returns:
            Dictionary mapping detection index to track ID
        """
        # Clear new dishes list
        self.new_dishes_this_frame = []

        # Remove expired tracks
        expired_ids = [
            tid for tid, track in self.tracked_dishes.items()
            if current_time - track['last_seen'] > self.TRACK_EXPIRY_TIME
        ]
        for tid in expired_ids:
            del self.tracked_dishes[tid]

        # Get current detection centers and bboxes
        detection_centers = []
        detection_bboxes = []
        for box in detections:
            try:
                # Get bbox tensor and convert to numpy
                bbox_tensor = box.xyxy
                # Handle different tensor formats
                if hasattr(bbox_tensor, 'cpu'):
                    bbox = bbox_tensor.cpu().numpy()
                elif hasattr(bbox_tensor, 'numpy'):
                    bbox = bbox_tensor.numpy()
                else:
                    bbox = np.array(bbox_tensor)

                # Flatten and take first 4 values [x1, y1, x2, y2]
                bbox = bbox.flatten()[:4]
                if len(bbox) < 4:
                    continue
            except Exception as e:
                print(f"Error extracting bbox: {e}")
                continue

            center = self.calculate_center(bbox)
            detection_centers.append(center)
            detection_bboxes.append(bbox)

        # Match detections to existing tracks
        matched_track_ids = set()
        detection_to_track = {}

        for det_idx, det_center in enumerate(detection_centers):
            best_match_id = None
            best_score = -1

            det_bbox = detection_bboxes[det_idx]

            # Find best matching track
            for track_id, track in self.tracked_dishes.items():
                if track_id in matched_track_ids:
                    continue

                # Calculate IoU
                iou = self.calculate_iou(det_bbox, track['bbox'])

                # Calculate center distance
                distance = self.calculate_distance(det_center, track['center'])

                # Check against position history
                position_history = track.get('position_history', [])
                min_history_distance = distance
                if position_history:
                    for hist_pos in position_history[-3:]:
                        hist_dist = self.calculate_distance(det_center, hist_pos)
                        min_history_distance = min(min_history_distance, hist_dist)

                # Combined scoring
                score = iou * 0.7 + (1.0 - min(distance / self.POSITION_THRESHOLD, 1.0)) * 0.3

                # Match if IoU is good OR distance is close
                if (iou >= self.IOU_THRESHOLD or min_history_distance < self.POSITION_THRESHOLD) and score > best_score:
                    best_score = score
                    best_match_id = track_id

            if best_match_id is not None:
                # Update existing track
                track = self.tracked_dishes[best_match_id]
                last_position = track.get('last_position', track['center'])
                position_change = self.calculate_distance(det_center, last_position)

                # Check if dish is stationary
                if position_change <= self.POSITION_STABILITY_THRESHOLD:
                    if track.get('stationary_start') is None:
                        track['stationary_start'] = current_time
                        track['first_detected'] = current_time
                else:
                    # Dish moved - reset stationary tracking
                    track['stationary_start'] = None
                    track['first_detected'] = current_time

                # Update track info
                track['center'] = det_center
                track['last_seen'] = current_time
                track['last_position'] = det_center
                track['bbox'] = detection_bboxes[det_idx]

                # Update position history
                if 'position_history' not in track:
                    track['position_history'] = []
                track['position_history'].append(det_center)
                if len(track['position_history']) > 5:
                    track['position_history'].pop(0)

                matched_track_ids.add(best_match_id)
                detection_to_track[det_idx] = best_match_id
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.tracked_dishes[new_id] = {
                    'center': det_center,
                    'last_seen': current_time,
                    'bbox': detection_bboxes[det_idx],
                    'first_detected': current_time,
                    'last_position': det_center,
                    'stationary_start': current_time,
                    'position_history': [det_center],
                    'association_complete': False  # Flag for face association
                }
                detection_to_track[det_idx] = new_id
                self.new_dishes_this_frame.append(new_id)
                print(f"  🍽️  New dish detected: ID {new_id}")

        return detection_to_track

    def process_frame(self, frame: np.ndarray, model, current_time: float) -> Tuple:
        """
        Process a frame with YOLO model and update tracks

        Args:
            frame: Input frame (cropped)
            model: YOLO model
            current_time: Current timestamp

        Returns:
            Tuple of (results, detection_to_track mapping)
        """
        # Run inference
        results = model(frame, conf=0.4, verbose=False)

        # Match detections to tracks
        if results and len(results[0].boxes) > 0:
            detection_to_track = self.match_detections_to_tracks(
                results[0].boxes,
                current_time
            )
        else:
            detection_to_track = {}

        return results, detection_to_track

    def get_new_dishes(self) -> List[int]:
        """
        Get list of dish IDs that were created in the last frame

        Returns:
            List of new dish IDs
        """
        return self.new_dishes_this_frame.copy()

    def get_stationary_dishes(self, threshold: float = None) -> List[int]:
        """
        Get dishes that have been stationary for a threshold duration
        and haven't been associated yet

        Args:
            threshold: Seconds threshold (default: use STATIONARY_THRESHOLD)

        Returns:
            List of dish IDs ready for association
        """
        if threshold is None:
            threshold = self.STATIONARY_THRESHOLD

        current_time = time.time()
        ready_dishes = []

        for dish_id, track in self.tracked_dishes.items():
            # Skip if already associated
            if track.get('association_complete'):
                continue

            stationary_start = track.get('stationary_start')
            if stationary_start is None:
                continue

            stationary_duration = current_time - stationary_start
            if stationary_duration >= threshold:
                ready_dishes.append(dish_id)

        return ready_dishes

    def mark_dish_associated(self, dish_id: int):
        """Mark a dish as having completed face association"""
        if dish_id in self.tracked_dishes:
            self.tracked_dishes[dish_id]['association_complete'] = True

    def draw_annotations(self, frame: np.ndarray, results, detection_to_track: Dict[int, int], current_time: float) -> np.ndarray:
        """
        Draw bounding boxes and IDs on frame

        Args:
            frame: Input frame
            results: YOLO results
            detection_to_track: Mapping of detection index to track ID
            current_time: Current timestamp

        Returns:
            Annotated frame
        """
        # Use YOLO's built-in plot first
        try:
            annotated = results[0].plot(line_width=1, labels=False)
        except:
            annotated = frame.copy()

        # Overlay our IDs
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                if idx in detection_to_track:
                    track_id = detection_to_track[idx]
                    track = self.tracked_dishes.get(track_id, {})
                    try:
                        # Get bbox
                        bbox_tensor = box.xyxy
                        if hasattr(bbox_tensor, 'cpu'):
                            bbox = bbox_tensor.cpu().numpy()
                        elif hasattr(bbox_tensor, 'numpy'):
                            bbox = bbox_tensor.numpy()
                        else:
                            bbox = np.array(bbox_tensor)

                        bbox = bbox.flatten()[:4].astype(int)
                        if len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = bbox
                    except Exception:
                        continue

                    # Determine color based on association status
                    if track.get('association_complete'):
                        color = (0, 255, 0)  # Green - associated
                        status_text = "TRACKED"
                    else:
                        stationary_start = track.get('stationary_start')
                        if stationary_start:
                            duration = current_time - stationary_start
                            if duration >= self.STATIONARY_THRESHOLD:
                                color = (0, 165, 255)  # Orange - ready for association
                                status_text = f"READY ({duration:.0f}s)"
                            else:
                                color = (0, 255, 255)  # Yellow - waiting
                                status_text = f"{duration:.0f}s"
                        else:
                            color = (255, 0, 0)  # Blue - moving
                            status_text = "MOVING"

                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Draw ID label
                    label = f"ID: {track_id} ({status_text})"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(annotated, (x1, y1 - text_height - 10),
                                  (x1 + text_width, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated


if __name__ == '__main__':
    print("DishTracker module - use in main application")
