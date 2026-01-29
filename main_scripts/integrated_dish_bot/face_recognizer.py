"""
Face Recognizer Module
Extracted from face_recognition_web.py
Handles face detection and recognition using OpenCV LBPH
"""

import cv2
import pickle
import os
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
import time


class FaceRecognizer:
    """Face detection and recognition using OpenCV"""

    def __init__(self, model_dir: str = None):
        """
        Initialize face recognizer

        Args:
            model_dir: Directory containing face_recognizer.xml and known_faces_opencv.pkl
                      If None, uses facial_recognition_scripts directory
        """
        # Configuration
        self.CONFIDENCE_THRESHOLD = 85  # Lower = more confident (LBPH scale)
        self.FACE_BUFFER_SIZE = 30  # Keep last 30 frames (~3s at 10fps)

        # Determine model directory
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            model_dir = os.path.join(project_root, "facial_recognition_scripts")

        self.model_file = os.path.join(model_dir, "face_recognizer.xml")
        self.label_file = os.path.join(model_dir, "known_faces_opencv.pkl")

        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load model and labels
        self.recognizer = None
        self.label_map = {}
        self.load_model()

        # Frame buffer for capturing footage around detection events
        self.frame_buffer = deque(maxlen=self.FACE_BUFFER_SIZE)

        # Current frame state
        self.current_faces = []  # List of detected faces in current frame

    def load_model(self):
        """Load face recognition model and label map"""
        if not os.path.exists(self.model_file) or not os.path.exists(self.label_file):
            print(f"⚠️  Face recognition model not found!")
            print(f"   Missing: {self.model_file} or {self.label_file}")
            print(f"   Face recognition will be disabled")
            return

        try:
            # Load the recognizer
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.model_file)

            # Load label map
            with open(self.label_file, "rb") as f:
                self.label_map = pickle.load(f)

            print(f"✓ Loaded face recognition model for {len(self.label_map)} people")
            print(f"  People: {', '.join(sorted(self.label_map.values()))}")
        except Exception as e:
            print(f"✗ Error loading face recognition model: {e}")
            self.recognizer = None

    def process_frame(self, frame: np.ndarray, current_time: float = None) -> List[Dict]:
        """
        Process a frame for face detection and recognition

        Args:
            frame: Input frame (BGR format)
            current_time: Current timestamp (optional)

        Returns:
            List of detected faces with information:
            [{'name': str, 'confidence': float, 'bbox': (x,y,w,h), 'frame': np.ndarray}, ...]
        """
        if current_time is None:
            current_time = time.time()

        # Add frame to buffer
        self.frame_buffer.append({
            'frame': frame.copy(),
            'timestamp': current_time
        })

        # If no model loaded, return empty
        if self.recognizer is None:
            self.current_faces = []
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Recognize each face
        faces_info = []
        for (x, y, w, h) in detected_faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Resize to match training size
            face_roi = cv2.resize(face_roi, (200, 200))

            # Predict
            label_id, confidence = self.recognizer.predict(face_roi)

            # Lower confidence value = better match in LBPH
            if confidence < self.CONFIDENCE_THRESHOLD:
                name = self.label_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            faces_info.append({
                'name': name,
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'frame': frame.copy(),  # Save frame where this face was detected
                'timestamp': current_time
            })

        self.current_faces = faces_info
        return faces_info

    def get_current_faces(self) -> List[Dict]:
        """Get faces detected in the most recent frame"""
        return self.current_faces.copy()

    def get_buffered_frames(self, count: int = 3) -> List[Dict]:
        """
        Get last N frames from buffer

        Args:
            count: Number of frames to return

        Returns:
            List of frame dictionaries: [{'frame': np.ndarray, 'timestamp': float}, ...]
        """
        buffer_list = list(self.frame_buffer)
        return buffer_list[-count:] if len(buffer_list) >= count else buffer_list

    def find_closest_face(self, faces: List[Dict]) -> Optional[Dict]:
        """
        Find the face closest to camera (largest bounding box)

        Args:
            faces: List of face dictionaries

        Returns:
            Face dictionary with largest bbox, or None if no faces
        """
        if not faces:
            return None

        # Calculate area for each face and find max
        def bbox_area(face):
            x, y, w, h = face['bbox']
            return w * h

        return max(faces, key=bbox_area)

    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw face detection boxes and labels on frame

        Args:
            frame: Input frame

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for face_info in self.current_faces:
            x, y, w, h = face_info['bbox']
            name = face_info['name']
            confidence = face_info['confidence']

            # Determine color
            if name == "Unknown":
                color = (0, 0, 255)  # Red
                label_bg_color = (0, 0, 200)
            else:
                color = (0, 255, 0)  # Green
                label_bg_color = (0, 200, 0)

            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Draw label
            label = f"{name} ({confidence:.0f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x, y + h),
                (x + text_width + 10, y + h + text_height + 10),
                label_bg_color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x + 5, y + h + text_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Add FPS text
        face_count_text = f"Faces: {len(self.current_faces)}"
        cv2.putText(
            annotated,
            face_count_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return annotated


if __name__ == '__main__':
    print("FaceRecognizer module - use in main application")
