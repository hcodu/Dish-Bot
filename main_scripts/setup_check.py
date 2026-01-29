#!/usr/bin/env python3
"""
Setup Checker
Verifies that all required files and models are present before launching
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def check_file(path, name):
    """Check if a file exists and print status"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {name}")
    if not exists:
        print(f"      Missing: {path}")
    return exists

def main():
    print("=" * 60)
    print("SETUP CHECKER")
    print("=" * 60)

    all_good = True

    # Check dish detection model
    print("\n1. DISH DETECTION MODEL")
    print("-" * 60)
    model_dir = os.path.join(PROJECT_ROOT, "best_ncnn_model")
    model_param = os.path.join(model_dir, "model.ncnn.param")
    model_bin = os.path.join(model_dir, "model.ncnn.bin")

    if not check_file(model_dir, "Model directory"):
        all_good = False
    else:
        if not check_file(model_param, "model.ncnn.param"):
            all_good = False
        if not check_file(model_bin, "model.ncnn.bin"):
            all_good = False

    # Check face recognition model
    print("\n2. FACE RECOGNITION MODEL")
    print("-" * 60)
    face_dir = os.path.join(PROJECT_ROOT, "facial_recognition_scripts")
    face_model = os.path.join(face_dir, "face_recognizer.xml")
    face_labels = os.path.join(face_dir, "known_faces_opencv.pkl")

    has_face_model = check_file(face_model, "face_recognizer.xml")
    has_face_labels = check_file(face_labels, "known_faces_opencv.pkl")

    if not has_face_model or not has_face_labels:
        all_good = False
        print("\n  ⚠️  Face recognition model not trained!")
        print("  To train:")
        print(f"    cd {face_dir}")
        print("    python3 train_faces_opencv.py")
        print("\n  See facial_recognition_scripts/README.md for details")

    # Check for known faces directory
    print("\n3. TRAINING DATA")
    print("-" * 60)
    known_faces_dir = os.path.join(face_dir, "known_faces")
    if check_file(known_faces_dir, "known_faces directory"):
        # Count subdirectories (one per person)
        try:
            people = [d for d in os.listdir(known_faces_dir)
                     if os.path.isdir(os.path.join(known_faces_dir, d))]
            if people:
                print(f"  Found {len(people)} people: {', '.join(people)}")
                for person in people:
                    person_dir = os.path.join(known_faces_dir, person)
                    images = [f for f in os.listdir(person_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"    {person}: {len(images)} images")
            else:
                print("  ⚠️  No people found in known_faces/")
                print("  Add training images to known_faces/<PersonName>/")
        except Exception as e:
            print(f"  ⚠️  Error reading known_faces: {e}")
    else:
        print("  ⚠️  No training data directory found")
        print("  Create: mkdir -p facial_recognition_scripts/known_faces/<PersonName>")
        print("  Add photos to each person's directory")

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✓ ALL CHECKS PASSED - Ready to launch!")
        print("=" * 60)
        print("\nRun: python3 launch_dual_system_simple.py")
    else:
        print("✗ SETUP INCOMPLETE")
        print("=" * 60)
        print("\nPlease fix the issues above before launching")
    print("=" * 60)

if __name__ == '__main__':
    main()
