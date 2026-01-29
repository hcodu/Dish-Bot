#!/usr/bin/env python3
"""
Integrated Dish Bot - Main Flask Application
Combines dish detection and face recognition
"""

import cv2
import numpy as np
import time
import os
import sys
import json
from datetime import datetime, timedelta
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO

# Import our modules
from database import Database
from dish_tracker import DishTracker
from face_recognizer import FaceRecognizer

# Configuration
DISH_CAMERA_INDEX = 0
FACE_CAMERA_INDEX = 4

DISH_PROCESS_EVERY_N_FRAMES = 5
FACE_PROCESS_EVERY_N_FRAMES = 3

STATIONARY_THRESHOLD = 15.0  # Seconds before associating dish with person
ASSOCIATION_WINDOW_START = -1.0  # Capture faces from 1s before dish detected
ASSOCIATION_WINDOW_END = 2.0  # Capture faces until 2s after dish detected

# Media cleanup
MEDIA_RETENTION_DAYS = 7

# Flask app
app = Flask(__name__)

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Initialize components
print("=" * 60)
print("INTEGRATED DISH BOT - INITIALIZING")
print("=" * 60)

# Database
db = Database('dish_bot.db')

# Dish tracker
dish_tracker = DishTracker()

# Face recognizer
face_recognizer = FaceRecognizer()

# Load YOLO model
print("\nLoading YOLO model...")
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_ncnn_model")
dish_model = YOLO(MODEL_PATH, task="detect")
print("✓ YOLO model loaded")

# Initialize cameras
print("\nInitializing cameras...")
dish_cam = cv2.VideoCapture(DISH_CAMERA_INDEX, cv2.CAP_V4L2)
dish_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
dish_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
dish_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
dish_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

face_cam = cv2.VideoCapture(FACE_CAMERA_INDEX, cv2.CAP_V4L2)
face_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
face_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
face_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
face_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not dish_cam.isOpened():
    print(f"✗ Error: Cannot open dish camera (index {DISH_CAMERA_INDEX})")
    sys.exit(1)

if not face_cam.isOpened():
    print(f"✗ Error: Cannot open face camera (index {FACE_CAMERA_INDEX})")
    sys.exit(1)

print("✓ Both cameras initialized")

# Pending associations
# {dish_id: {'first_detected': timestamp, 'dish_bbox': ..., 'dish_frame': ..., 'face_data': [...]}}
pending_associations = {}

print("\n" + "=" * 60)
print("INITIALIZATION COMPLETE")
print("=" * 60 + "\n")


def crop_center(frame: np.ndarray, crop_percent: float) -> np.ndarray:
    """Crop frame to center region"""
    fh, fw = frame.shape[:2]
    nw = int(fw * crop_percent)
    nh = int(fh * crop_percent)
    xs = (fw - nw) // 2
    ys = (fh - nh) // 2
    return frame[ys:ys+nh, xs:xs+nw]


def associate_dish_with_person(dish_id: int, association_data: dict):
    """
    Associate a stationary dish with the person who placed it

    Args:
        dish_id: Dish tracking ID
        association_data: Dictionary with face data collected around detection
    """
    print(f"\n{'='*60}")
    print(f"ASSOCIATING DISH ID {dish_id}")
    print(f"{'='*60}")

    # Analyze collected face data
    all_faces = association_data.get('face_data', [])

    if not all_faces:
        # No faces detected - mark as Unknown
        person_name = "Unknown"
        confidence = 0.0
        face_bbox = None
        face_frame = None
        print("  ⚠️  No faces detected - marking as Unknown")
    else:
        # Find closest (largest) face across all collected data
        closest_face = face_recognizer.find_closest_face(all_faces)
        person_name = closest_face['name']
        confidence = closest_face['confidence']
        face_bbox = closest_face['bbox']
        face_frame = closest_face['frame']
        print(f"  👤 Identified: {person_name} (confidence: {confidence:.1f})")

    # Save snapshots
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dish_snapshot_path = f"static/snapshots/dish_{dish_id}_{timestamp_str}.jpg"
    face_snapshot_path = f"static/snapshots/face_{dish_id}_{timestamp_str}.jpg" if face_frame is not None else None

    os.makedirs("static/snapshots", exist_ok=True)

    cv2.imwrite(dish_snapshot_path, association_data['dish_frame'])
    print(f"  💾 Dish snapshot saved: {dish_snapshot_path}")

    if face_frame is not None:
        cv2.imwrite(face_snapshot_path, face_frame)
        print(f"  💾 Face snapshot saved: {face_snapshot_path}")

    # Store in database
    event_id = db.insert_dish_event(
        dish_id=dish_id,
        timestamp=datetime.fromtimestamp(association_data['first_detected']),
        person_name=person_name,
        confidence=confidence,
        dish_snapshot_path=dish_snapshot_path,
        face_snapshot_path=face_snapshot_path,
        dish_bbox=json.dumps([float(v) for v in association_data['dish_bbox']]),
        face_bbox=json.dumps(face_bbox) if face_bbox else None
    )

    # Mark dish as associated in tracker
    dish_tracker.mark_dish_associated(dish_id)

    print(f"  ✅ Dish ID {dish_id} associated with {person_name}")
    print(f"  📊 Event ID {event_id} saved to database")
    print(f"{'='*60}\n")


def generate_dual_feed():
    """Generate side-by-side video feed with both cameras"""
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    # Pre-calculate crop
    fh, fw = 480, 640
    nw, nh = int(fw * 0.4), int(fh * 0.4)
    xs, ys = (fw - nw) // 2, (fh - nh) // 2

    print("Starting dual camera feed...")

    while True:
        # Read from both cameras
        ret1, dish_frame = dish_cam.read()
        ret2, face_frame = face_cam.read()

        if not ret1 or not ret2:
            print("Error reading from cameras")
            continue

        current_time = time.time()

        # Process dish detection (every 5th frame)
        dish_results = None
        detection_to_track = {}

        if frame_count % DISH_PROCESS_EVERY_N_FRAMES == 0:
            # Crop dish frame
            dish_frame_cropped = dish_frame[ys:ys+nh, xs:xs+nw]

            # Process with YOLO and tracker
            dish_results, detection_to_track = dish_tracker.process_frame(
                dish_frame_cropped,
                dish_model,
                current_time
            )

            # Check for new dishes
            new_dishes = dish_tracker.get_new_dishes()
            for dish_id in new_dishes:
                # Start tracking this dish for face association
                pending_associations[dish_id] = {
                    'first_detected': current_time,
                    'dish_bbox': dish_tracker.tracked_dishes[dish_id]['bbox'],
                    'dish_frame': dish_frame_cropped.copy(),
                    'face_data': []  # Will collect face detections
                }
                print(f"  📝 Started tracking Dish ID {dish_id} for face association")

        # Process face recognition (every 3rd frame)
        if frame_count % FACE_PROCESS_EVERY_N_FRAMES == 0:
            faces = face_recognizer.process_frame(face_frame, current_time)

            # Collect face data for pending dishes within the association window
            for dish_id, data in list(pending_associations.items()):
                elapsed = current_time - data['first_detected']

                # Collect faces from -1s to +2s window
                if ASSOCIATION_WINDOW_START <= elapsed <= ASSOCIATION_WINDOW_END:
                    if faces:
                        data['face_data'].extend(faces)

        # Check for dishes ready to be associated (stationary for 15s)
        stationary_dishes = dish_tracker.get_stationary_dishes(threshold=STATIONARY_THRESHOLD)
        for dish_id in stationary_dishes:
            if dish_id in pending_associations:
                # This dish is ready - associate with person
                associate_dish_with_person(dish_id, pending_associations[dish_id])
                del pending_associations[dish_id]

        # Draw annotations
        if dish_results is not None:
            annotated_dish = dish_tracker.draw_annotations(
                dish_frame[ys:ys+nh, xs:xs+nw],
                dish_results,
                detection_to_track,
                current_time
            )
        else:
            annotated_dish = dish_frame[ys:ys+nh, xs:xs+nw].copy()

        annotated_face = face_recognizer.draw_annotations(face_frame)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time)
            active_tracks = len([t for t in dish_tracker.tracked_dishes.values()
                                 if current_time - t['last_seen'] <= dish_tracker.TRACK_EXPIRY_TIME])
            pending_count = len(pending_associations)
            print(f"FPS: {fps:.1f} | Active Dishes: {active_tracks} | Pending Associations: {pending_count}")
            start_time = time.time()

        # Add title labels to each feed
        cv2.putText(annotated_dish, "DISH DETECTION", (10, annotated_dish.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_face, "FACE RECOGNITION", (10, annotated_face.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize frames to same height before stacking
        target_height = 480
        dish_height = annotated_dish.shape[0]
        dish_width = annotated_dish.shape[1]
        face_height = annotated_face.shape[0]
        face_width = annotated_face.shape[1]

        # Resize dish frame to target height
        dish_scale = target_height / dish_height
        dish_resized = cv2.resize(annotated_dish,
                                   (int(dish_width * dish_scale), target_height))

        # Resize face frame to target height
        face_scale = target_height / face_height
        face_resized = cv2.resize(annotated_face,
                                   (int(face_width * face_scale), target_height))

        # Stack side-by-side
        combined = np.hstack([dish_resized, face_resized])

        # Add FPS to combined frame
        cv2.putText(combined, f"FPS: {fps:.1f}", (combined.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main dashboard with dual live feeds"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_dual_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def history():
    """Detection events history page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20

    # Get paginated events
    data = db.get_events_paginated(page=page, per_page=per_page)

    return render_template('history.html', **data)


@app.route('/api/daily_stats/<date_str>')
def api_daily_stats(date_str):
    """Get statistics for a specific day (format: YYYY-MM-DD)"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        stats = db.get_daily_stats(date)
        return jsonify(stats)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400


@app.route('/api/recent_events')
def api_recent_events():
    """Get recent events (last 10)"""
    events = db.get_all_events(limit=10)
    return jsonify({'events': events})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("INTEGRATED DISH BOT - STARTING")
    print("=" * 60)
    print(f"Dish Camera: {DISH_CAMERA_INDEX}")
    print(f"Face Camera: {FACE_CAMERA_INDEX}")
    print(f"Stationary Threshold: {STATIONARY_THRESHOLD}s")
    print(f"Association Window: {ASSOCIATION_WINDOW_START}s to {ASSOCIATION_WINDOW_END}s")
    print("\nWeb Interface:")
    print("  http://dish-bot.local:5000")
    print("  http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        dish_cam.release()
        face_cam.release()
        cv2.destroyAllWindows()
        print("✓ Cameras released")
        print("Goodbye!")
