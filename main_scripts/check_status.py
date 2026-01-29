#!/usr/bin/env python3
"""
Check Status of Dual System
Shows which systems are running and their recent logs
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

def check_process(pattern):
    """Check if a process matching the pattern is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', pattern],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return True, pids
        return False, []
    except:
        return False, []

def tail_log(log_file, lines=10):
    """Get last N lines from log file"""
    if not os.path.exists(log_file):
        return None
    try:
        result = subprocess.run(
            ['tail', '-n', str(lines), log_file],
            capture_output=True,
            text=True
        )
        return result.stdout
    except:
        return None

def main():
    print("=" * 60)
    print("DUAL SYSTEM STATUS")
    print("=" * 60)

    if os.name == 'nt':
        print("\nStatus checking only available on Linux")
        print("Check Task Manager on Windows")
        sys.exit(0)

    # Check dish detector
    print("\n1. DISH DETECTION (port 5000)")
    print("-" * 60)
    running, pids = check_process('dish_detector_groupme.py')
    if running:
        print(f"  Status: ✓ RUNNING (PID: {', '.join(pids)})")

        dish_log = os.path.join(LOG_DIR, "dish_detector.log")
        log_output = tail_log(dish_log, 5)
        if log_output:
            print(f"\n  Recent log (last 5 lines):")
            for line in log_output.split('\n'):
                if line.strip():
                    print(f"    {line}")
    else:
        print("  Status: ✗ NOT RUNNING")

    # Check face recognition
    print("\n2. FACE RECOGNITION (port 5002)")
    print("-" * 60)
    running, pids = check_process('face_recognition_web.py')
    if running:
        print(f"  Status: ✓ RUNNING (PID: {', '.join(pids)})")

        face_log = os.path.join(LOG_DIR, "face_recognition.log")
        log_output = tail_log(face_log, 5)
        if log_output:
            print(f"\n  Recent log (last 5 lines):")
            for line in log_output.split('\n'):
                if line.strip():
                    print(f"    {line}")
    else:
        print("  Status: ✗ NOT RUNNING")

    # Show access URLs
    print("\n" + "=" * 60)
    print("ACCESS URLs")
    print("=" * 60)
    print("Dish Detection:      http://dish-bot.local:5000")
    print("Face Recognition:    http://dish-bot.local:5002")

    print("\n" + "=" * 60)
    print("COMMANDS")
    print("=" * 60)
    print("View full logs:")
    print(f"  tail -f {os.path.join(LOG_DIR, 'dish_detector.log')}")
    print(f"  tail -f {os.path.join(LOG_DIR, 'face_recognition.log')}")
    print("\nStop systems:")
    print("  python3 stop_dual_system.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
