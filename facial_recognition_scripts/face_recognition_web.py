#!/usr/bin/env python3
"""
Face Recognition Web Test (OpenCV LBPH version)
Flask web interface for face recognition - no display needed!
View in browser instead of OpenCV window
"""

import cv2
import pickle
import os
import time
from flask import Flask, Response

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
CAMERA_INDEX = 4  # Your 1080p webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LABEL_FILE = os.path.join(SCRIPT_DIR, "known_faces_opencv.pkl")
MODEL_FILE = os.path.join(SCRIPT_DIR, "face_recognizer.xml")

# Performance tuning for RPi 4
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
CONFIDENCE_THRESHOLD = 85  # Lower = more confident (0-100) - Increased from 70

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model
print("=" * 60)
print("Face Recognition Web Test (OpenCV LBPH)")
print("=" * 60)

if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_FILE):
    print(f"\n❌ Error: Model files not found!")
    print(f"Missing: {MODEL_FILE} or {LABEL_FILE}")
    print(f"\nPlease run training first:")
    print(f"  cd face_recognition")
    print(f"  python train_faces_opencv.py")
    exit(1)

try:
    # Load the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    # Load label map
    with open(LABEL_FILE, "rb") as f:
        label_map = pickle.load(f)

    print(f"\n✓ Loaded model for {len(label_map)} people")
    print(f"  People: {', '.join(sorted(label_map.values()))}")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    exit(1)

# Initialize camera
print(f"\nInitializing camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"❌ Error: Could not open camera {CAMERA_INDEX}")
    exit(1)

print("✓ Camera initialized")

# Flask app
app = Flask(__name__)

def generate_frames():
    """Generate video frames with face recognition"""

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
            print("Error reading from camera")
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
            (0, 255, 0),
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
            (0, 255, 0),
            2
        )

        # Draw settings info
        settings_text = f"Threshold: {CONFIDENCE_THRESHOLD}"
        cv2.putText(
            frame,
            settings_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home page with video feed"""
    people_list = ', '.join(sorted(label_map.values()))

    return f'''
    <html>
    <head>
        <title>Face Recognition Test</title>
        <style>
            body {{
                background-color: #1a1a1a;
                text-align: center;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                color: #00ff00;
                margin-bottom: 10px;
            }}
            .info {{
                color: #888;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            .people {{
                color: #00ff00;
                margin-bottom: 20px;
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 5px;
                display: inline-block;
            }}
            img {{
                max-width: 90%;
                border: 3px solid #00ff00;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
            }}
            .legend {{
                color: #888;
                margin-top: 20px;
                font-size: 14px;
            }}
            .green {{
                color: #00ff00;
            }}
            .red {{
                color: #ff0000;
            }}
        </style>
    </head>
    <body>
        <h1>Face Recognition Test</h1>
        <div class="info">
            OpenCV LBPH | Camera {CAMERA_INDEX} | Confidence Threshold: {CONFIDENCE_THRESHOLD}
        </div>
        <div class="people">
            <strong>Trained People:</strong> {people_list}
        </div>
        <br>
        <img src="/video_feed">
        <div class="legend">
            <span class="green">● Green</span> = Recognized |
            <span class="red">● Red</span> = Unknown |
            Lower confidence value = better match
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Settings:")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Process every: {PROCESS_EVERY_N_FRAMES} frames")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Using: OpenCV LBPH + Haar Cascades")
    print("=" * 60)
    print("\nStarting web server...")
    print("View at: http://dish-bot.local:5002")
    print("Or: http://<your-pi-ip>:5002")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        app.run(host='0.0.0.0', port=5002, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping server...")
        cap.release()
