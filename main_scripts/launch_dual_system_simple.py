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
    else:  # Linux/Mac - use nohup to detach properly
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)

        dish_log = os.path.join(log_dir, "dish_detector.log")
        with open(dish_log, 'w') as f:
            subprocess.Popen(
                ['nohup', sys.executable, DISH_DETECTOR_SCRIPT],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        print(f"  Log file: {dish_log}")

    time.sleep(2)

    # Launch face recognition
    print("Launching Face Recognition (port 5002, camera 4)...")
    if os.name == 'nt':  # Windows
        subprocess.Popen(
            ['start', 'cmd', '/k', sys.executable, FACE_RECOGNITION_SCRIPT],
            shell=True
        )
    else:  # Linux/Mac - use nohup to detach properly
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)

        face_log = os.path.join(log_dir, "face_recognition.log")
        with open(face_log, 'w') as f:
            subprocess.Popen(
                ['nohup', sys.executable, FACE_RECOGNITION_SCRIPT],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        print(f"  Log file: {face_log}")

    time.sleep(1)

    print("\n" + "=" * 60)
    print("BOTH SYSTEMS LAUNCHED")
    print("=" * 60)
    print("Dish Detection:      http://localhost:5000")
    print("Face Recognition:    http://localhost:5002")
    print("=" * 60)

    if os.name != 'nt':
        print("\nBoth systems are running in background (detached)")
        print(f"View logs:")
        print(f"  tail -f {os.path.join(PROJECT_ROOT, 'logs/dish_detector.log')}")
        print(f"  tail -f {os.path.join(PROJECT_ROOT, 'logs/face_recognition.log')}")
        print("\nTo stop the systems:")
        print("  pkill -f dish_detector_groupme.py")
        print("  pkill -f face_recognition_web.py")
    else:
        print("\nBoth systems are running in separate windows")
        print("Close each window individually to stop the systems")
    print("=" * 60)

if __name__ == '__main__':
    main()
