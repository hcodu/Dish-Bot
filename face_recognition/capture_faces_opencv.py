#!/usr/bin/env python3
"""
Face Capture Helper Script (OpenCV version)
Uses OpenCV's built-in Haar Cascades - NO extra packages needed!
"""

import cv2
import os
import sys
from pathlib import Path

# Configuration
KNOWN_FACES_DIR = "known_faces"
CAMERA_INDEX = 4  # Your 1080p webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Try V4L2 on Linux/Raspberry Pi
try:
    USE_V4L2 = sys.platform.startswith('linux')
except:
    USE_V4L2 = False

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_faces(person_name):
    """
    Capture face photos for a specific person.

    Args:
        person_name: Name of the person (used for directory name)
    """

    print("=" * 60)
    print(f"Face Capture for: {person_name}")
    print("=" * 60)

    # Create person's directory
    person_dir = Path(KNOWN_FACES_DIR) / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Find next available image number
    existing_images = list(person_dir.glob(f"{person_name}_*.jpg"))
    if existing_images:
        numbers = []
        for img in existing_images:
            try:
                num = int(img.stem.split('_')[-1])
                numbers.append(num)
            except:
                pass
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    print(f"\nSaving to: {person_dir}/")
    print(f"Starting at: {person_name}_{next_num:03d}.jpg")

    # Initialize camera
    print("\nInitializing camera...")

    if USE_V4L2:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {CAMERA_INDEX}")
        print("Try changing CAMERA_INDEX in the script")
        return False

    print("✓ Camera initialized")
    print("\n" + "=" * 60)
    print("Controls:")
    print("  [S] - Save current frame")
    print("  [Q] - Quit")
    print("=" * 60)
    print("\nPosition your face in the frame and press 'S' to capture")
    print("Try to capture from different angles and expressions\n")

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading from camera")
            break

        # Create a copy for display
        display_frame = frame.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw green rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add label
            label = "Face detected - Press 'S'"
            cv2.putText(display_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show face count
        face_count_text = f"Faces: {len(faces)}"
        cv2.putText(display_frame, face_count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show saved count
        saved_text = f"Saved: {saved_count}"
        cv2.putText(display_frame, saved_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow(f"Capture Faces - {person_name}", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break

        elif key == ord('s') or key == ord('S'):
            # Save the frame
            if len(faces) == 0:
                print("⚠️  No face detected - not saving")
            elif len(faces) > 1:
                print("⚠️  Multiple faces detected - not saving")
            else:
                filename = f"{person_name}_{next_num:03d}.jpg"
                filepath = person_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"✓ Saved: {filename}")
                saved_count += 1
                next_num += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Capture complete! Saved {saved_count} image(s)")
    print("=" * 60)

    if saved_count > 0:
        print(f"\nNext steps:")
        print(f"1. Review images in: {person_dir}/")
        print(f"2. Run training: python train_faces_opencv.py")

    return True

if __name__ == "__main__":
    # Get person name from command line or prompt
    if len(sys.argv) > 1:
        person_name = " ".join(sys.argv[1:])
    else:
        person_name = input("Enter person's name: ").strip()

    if not person_name:
        print("❌ Error: Person name is required")
        print("Usage: python capture_faces_opencv.py <PersonName>")
        print("Example: python capture_faces_opencv.py John")
        exit(1)

    # Clean up name (remove special characters, use for directory)
    person_name = "".join(c for c in person_name if c.isalnum() or c in " -_")
    person_name = person_name.strip()

    if not person_name:
        print("❌ Error: Invalid person name")
        exit(1)

    success = capture_faces(person_name)
    exit(0 if success else 1)
