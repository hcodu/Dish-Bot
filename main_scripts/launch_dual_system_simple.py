#!/usr/bin/env python3
"""
Simple Dual System Launcher
Launches both dish detection and face recognition in separate windows
"""

import subprocess
import sys
import os
import time

# Paths to the scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DISH_DETECTOR_SCRIPT = os.path.join(PROJECT_ROOT, "dish_detection_scripts", "dish_detector_groupme.py")
FACE_RECOGNITION_SCRIPT = os.path.join(PROJECT_ROOT, "facial_recognition_scripts", "face_recognition_web.py")

def main():
    print("=" * 60)
    print("SIMPLE DUAL SYSTEM LAUNCHER")
    print("=" * 60)

    # Verify scripts exist
    if not os.path.exists(DISH_DETECTOR_SCRIPT):
        print(f"❌ Error: Dish detector script not found")
        print(f"   Looking for: {DISH_DETECTOR_SCRIPT}")
        sys.exit(1)

    if not os.path.exists(FACE_RECOGNITION_SCRIPT):
        print(f"❌ Error: Face recognition script not found")
        print(f"   Looking for: {FACE_RECOGNITION_SCRIPT}")
        sys.exit(1)

    print("✓ Both scripts found\n")

    # Launch dish detector
    print("Launching Dish Detection (port 5000, camera 0)...")
    if os.name == 'nt':  # Windows
        subprocess.Popen(
            ['start', 'cmd', '/k', sys.executable, DISH_DETECTOR_SCRIPT],
            shell=True
        )
    else:  # Linux/Mac
        subprocess.Popen(
            [sys.executable, DISH_DETECTOR_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    time.sleep(2)

    # Launch face recognition
    print("Launching Face Recognition (port 5002, camera 4)...")
    if os.name == 'nt':  # Windows
        subprocess.Popen(
            ['start', 'cmd', '/k', sys.executable, FACE_RECOGNITION_SCRIPT],
            shell=True
        )
    else:  # Linux/Mac
        subprocess.Popen(
            [sys.executable, FACE_RECOGNITION_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    time.sleep(1)

    print("\n" + "=" * 60)
    print("BOTH SYSTEMS LAUNCHED")
    print("=" * 60)
    print("Dish Detection:      http://localhost:5000")
    print("Face Recognition:    http://localhost:5002")
    print("=" * 60)
    print("\nBoth systems are running in separate windows/processes")
    print("Close each window individually to stop the systems")
    print("=" * 60)

if __name__ == '__main__':
    main()
