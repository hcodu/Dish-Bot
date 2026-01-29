#!/usr/bin/env python3
"""
Face Recognition Test Script (OpenCV LBPH version)
Uses OpenCV's built-in face recognizer - NO extra packages needed!
"""

import cv2
import pickle
import os
import time
import sys

# Configuration
CAMERA_INDEX = 4  # Your 1080p webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LABEL_FILE = "known_faces_opencv.pkl"
MODEL_FILE = "face_recognizer.xml"

# Performance tuning for RPi 4
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
CONFIDENCE_THRESHOLD = 70  # Lower = more confident (0-100, lower is better for LBPH)

# Try V4L2 on Linux/Raspberry Pi
try:
    USE_V4L2 = sys.platform.startswith('linux')
except:
    USE_V4L2 = False

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_model():
    """Load the trained face recognizer and label map"""

    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: Model file '{MODEL_FILE}' not found!")
        print(f"\nPlease run training first:")
        print(f"  python train_faces_opencv.py")
        return None, None

    if not os.path.exists(LABEL_FILE):
        print(f"❌ Error: Label file '{LABEL_FILE}' not found!")
        print(f"\nPlease run training first:")
        print(f"  python train_faces_opencv.py")
        return None, None

    try:
        # Load the recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)

        # Load label map
        with open(LABEL_FILE, "rb") as f:
            label_map = pickle.load(f)

        # Count unique people
        print(f"✓ Loaded model for {len(label_map)} people")
        print(f"  People: {', '.join(sorted(label_map.values()))}")

        return recognizer, label_map

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def recognize_faces_main():
    """Main face recognition loop"""

    print("=" * 60)
    print("Face Recognition Test Script (OpenCV LBPH)")
    print("=" * 60)

    # Load model
    print("\nLoading face recognition model...")
    recognizer, label_map = load_model()

    if recognizer is None:
        return False

    # Initialize camera
    print(f"\nInitializing camera {CAMERA_INDEX}...")

    if USE_V4L2:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        print("  Using V4L2 backend")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print("  Using default backend")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {CAMERA_INDEX}")
        print("Try changing CAMERA_INDEX in the script")
        return False

    print("✓ Camera initialized")

    print("\n" + "=" * 60)
    print("Settings:")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Process every: {PROCESS_EVERY_N_FRAMES} frames")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Using: OpenCV LBPH + Haar Cascades")
    print("=" * 60)
    print("\nPress 'Q' to quit\n")

    # FPS tracking
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    # Recognition state (persistent across frames)
    face_boxes = []
    face_names = []
    face_confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading from camera")
            break

        frame_count += 1

        # Process every Nth frame for face recognition
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            face_boxes = []
            face_names = []
            face_confidences = []

            for (x, y, w, h) in detected_faces:
                face_boxes.append((x, y, w, h))

                # Extract face region
                face_roi = gray[y:y+h, x:x+w]

                # Resize to match training size
                face_roi = cv2.resize(face_roi, (200, 200))

                # Predict
                label_id, confidence = recognizer.predict(face_roi)

                # Lower confidence value = better match in LBPH
                if confidence < CONFIDENCE_THRESHOLD:
                    name = label_map.get(label_id, "Unknown")
                else:
                    name = "Unknown"

                face_names.append(name)
                face_confidences.append(confidence)

        # Draw results on frame
        for (x, y, w, h), name, confidence in zip(face_boxes, face_names, face_confidences):
            # Determine color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label_bg_color = (0, 0, 200)
            else:
                color = (0, 255, 0)  # Green for known
                label_bg_color = (0, 200, 0)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw label background
            label = f"{name} ({confidence:.0f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x, y + h),
                (x + text_width + 10, y + h + text_height + 10),
                label_bg_color,
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x + 5, y + h + text_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Calculate and display FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = 30 / elapsed if elapsed > 0 else 0
            start_time = time.time()

        # Draw FPS on frame
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Draw face count
        face_count_text = f"Faces: {len(face_boxes)}"
        cv2.putText(
            frame,
            face_count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display frame
        cv2.imshow('Face Recognition Test (OpenCV LBPH)', frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    try:
        success = recognize_faces_main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
        exit(0)
