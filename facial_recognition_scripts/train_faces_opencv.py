#!/usr/bin/env python3
"""
Train Face Recognition Model (OpenCV LBPH version)
Uses OpenCV's built-in face recognizer - NO extra packages needed!
"""

import cv2
import os
import pickle
import numpy as np
from pathlib import Path

# Configuration
KNOWN_FACES_DIR = "known_faces"
OUTPUT_FILE = "known_faces_opencv.pkl"
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def train_faces():
    """
    Load images from known_faces/ directory, detect faces, and train LBPH recognizer
    """

    print("=" * 60)
    print("Face Recognition Training Script (OpenCV LBPH)")
    print("=" * 60)

    # Check if directory exists
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"\n❌ Error: Directory '{KNOWN_FACES_DIR}' not found!")
        print(f"\nPlease create the directory structure:")
        print(f"  {KNOWN_FACES_DIR}/")
        print(f"    PersonName1/")
        print(f"      photo1.jpg")
        print(f"      photo2.jpg")
        print(f"    PersonName2/")
        print(f"      photo1.jpg")
        print(f"      ...")
        return False

    faces = []
    labels = []
    label_map = {}  # Map label IDs to names
    current_label = 0

    # Statistics
    total_images = 0
    successful_encodings = 0
    failed_images = 0
    people_count = 0

    print(f"\nScanning directory: {KNOWN_FACES_DIR}/")
    print("-" * 60)

    # Iterate through each person's directory
    known_faces_path = Path(KNOWN_FACES_DIR)
    person_dirs = [d for d in known_faces_path.iterdir() if d.is_dir()]

    if not person_dirs:
        print(f"\n❌ Error: No person directories found in '{KNOWN_FACES_DIR}'!")
        print(f"Please create subdirectories for each person.")
        return False

    for person_dir in sorted(person_dirs):
        person_name = person_dir.name
        print(f"\n📁 Processing: {person_name}")

        person_encodings = 0
        person_images = 0

        # Assign a label ID to this person
        label_map[current_label] = person_name

        # Get all image files in person's directory
        image_files = [
            f for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ]

        if not image_files:
            print(f"   ⚠️  No images found for {person_name}")
            current_label += 1
            continue

        people_count += 1

        for image_path in sorted(image_files):
            person_images += 1
            total_images += 1

            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"   ❌ Could not load: {image_path.name}")
                    failed_images += 1
                    continue

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces
                detected_faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(detected_faces) == 0:
                    print(f"   ⚠️  No face found in: {image_path.name}")
                    failed_images += 1
                    continue

                if len(detected_faces) > 1:
                    print(f"   ⚠️  Multiple faces found in: {image_path.name} (using largest face)")

                # Use the largest face detected
                largest_face = max(detected_faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face

                # Extract face region
                face_roi = gray[y:y+h, x:x+w]

                # Resize to standard size for consistency
                face_roi = cv2.resize(face_roi, (200, 200))

                # Add to training data
                faces.append(face_roi)
                labels.append(current_label)

                person_encodings += 1
                successful_encodings += 1
                print(f"   ✓ Encoded: {image_path.name}")

            except Exception as e:
                print(f"   ❌ Error processing {image_path.name}: {e}")
                failed_images += 1

        print(f"   → Processed {person_encodings}/{person_images} images successfully")
        current_label += 1

    # Train the recognizer
    if successful_encodings == 0:
        print("\n❌ No face encodings created. Nothing to save.")
        return False

    print("\n" + "=" * 60)
    print("Training LBPH recognizer...")

    # Create and train the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # Save the model and label map
    print("Saving model...")

    data = {
        "label_map": label_map,
        "model": recognizer
    }

    try:
        # Save recognizer to XML file
        recognizer.write("face_recognizer.xml")

        # Save label map separately
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(label_map, f)

        print(f"✓ Saved model to: face_recognizer.xml")
        print(f"✓ Saved labels to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return False

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"People recognized: {people_count}")
    print(f"Total images processed: {total_images}")
    print(f"Successful encodings: {successful_encodings}")
    print(f"Failed images: {failed_images}")
    print("=" * 60)

    # Show breakdown by person
    print("\nPeople trained:")
    for label_id, name in sorted(label_map.items()):
        count = labels.count(label_id)
        print(f"  {name}: {count} image(s)")

    print("\n✅ Training complete!")
    print(f"You can now run: python face_recognition_test_opencv.py")

    return True

if __name__ == "__main__":
    success = train_faces()
    exit(0 if success else 1)
