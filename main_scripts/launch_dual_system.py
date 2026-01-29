#!/usr/bin/env python3
"""
Dual System Launcher
Launches both dish detection and face recognition systems simultaneously
"""

import subprocess
import sys
import os
import time
import signal

# Paths to the scripts (relative to this file's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DISH_DETECTOR_SCRIPT = os.path.join(PROJECT_ROOT, "dish_detection_scripts", "dish_detector_groupme.py")
FACE_RECOGNITION_SCRIPT = os.path.join(PROJECT_ROOT, "facial_recognition_scripts", "face_recognition_web.py")

# Process handles
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n" + "=" * 60)
    print("Shutting down both systems...")
    print("=" * 60)

    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()

    print("✓ All processes stopped")
    sys.exit(0)

def main():
    global processes

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("DUAL SYSTEM LAUNCHER")
    print("=" * 60)
    print(f"Dish Detection: {DISH_DETECTOR_SCRIPT}")
    print(f"Face Recognition: {FACE_RECOGNITION_SCRIPT}")
    print("=" * 60)

    # Verify scripts exist
    if not os.path.exists(DISH_DETECTOR_SCRIPT):
        print(f"❌ Error: Dish detector script not found at {DISH_DETECTOR_SCRIPT}")
        sys.exit(1)

    if not os.path.exists(FACE_RECOGNITION_SCRIPT):
        print(f"❌ Error: Face recognition script not found at {FACE_RECOGNITION_SCRIPT}")
        sys.exit(1)

    print("\n✓ Both scripts found")
    print("\nStarting systems...\n")

    # Launch dish detector (port 5000, camera 0)
    print("=" * 60)
    print("Starting Dish Detection System (port 5000, camera 0)...")
    print("=" * 60)
    dish_proc = subprocess.Popen(
        [sys.executable, DISH_DETECTOR_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(dish_proc)
    time.sleep(2)  # Give it time to start

    # Launch face recognition (port 5002, camera 4)
    print("\n" + "=" * 60)
    print("Starting Face Recognition System (port 5002, camera 4)...")
    print("=" * 60)
    face_proc = subprocess.Popen(
        [sys.executable, FACE_RECOGNITION_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    processes.append(face_proc)
    time.sleep(2)

    print("\n" + "=" * 60)
    print("BOTH SYSTEMS RUNNING")
    print("=" * 60)
    print("Dish Detection:      http://dish-bot.local:5000")
    print("Face Recognition:    http://dish-bot.local:5002")
    print("=" * 60)
    print("\nPress Ctrl+C to stop both systems")
    print("=" * 60 + "\n")

    # Monitor both processes and display their output
    try:
        while True:
            # Check if either process has died
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    name = "Dish Detector" if i == 0 else "Face Recognition"
                    print(f"\n⚠️  {name} process has stopped!")
                    print("Shutting down all systems...")
                    signal_handler(None, None)

            # Read output from dish detector
            if dish_proc.stdout:
                line = dish_proc.stdout.readline()
                if line:
                    print(f"[DISH] {line.rstrip()}")

            # Read output from face recognition
            if face_proc.stdout:
                line = face_proc.stdout.readline()
                if line:
                    print(f"[FACE] {line.rstrip()}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()
