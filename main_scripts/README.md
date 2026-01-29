# Main Scripts - Combined Dish Detection & Face Recognition

This folder contains launcher scripts that run both the dish detection and face recognition systems simultaneously.

## Scripts Overview

### System Components

1. **Dish Detector** ([dish_detector_groupme.py](../dish_detection_scripts/dish_detector_groupme.py))
   - Port: 5000
   - Camera: 0 (main dish detection camera)
   - Features: YOLO dish detection, GroupMe alerts, tracking

2. **Face Recognition** ([face_recognition_web.py](../facial_recognition_scripts/face_recognition_web.py))
   - Port: 5002
   - Camera: 4 (1080p webcam for face recognition)
   - Features: OpenCV LBPH face recognition

### Launcher Scripts

#### Option 1: Simple Launcher (Recommended for SSH/Raspberry Pi)
```bash
python3 launch_dual_system_simple.py
```

**Features:**
- Launches both systems as background processes (Linux) or separate windows (Windows)
- Uses `nohup` and session detachment on Linux - processes survive SSH disconnection
- Logs output to `logs/` directory
- Simple and reliable

**On Linux:**
- Processes run in background and won't stop when SSH session ends
- Logs saved to `logs/dish_detector.log` and `logs/face_recognition.log`
- Use `python3 check_status.py` to verify systems are running
- Use `python3 stop_dual_system.py` to stop both systems

**Best for:** SSH deployment, Raspberry Pi, production use

#### Option 2: Advanced Launcher (Interactive Monitoring)
```bash
python launch_dual_system.py
```

**Features:**
- Launches both systems with live combined output monitoring
- Single Ctrl+C stops both systems gracefully
- Shows tagged output: `[DISH]` and `[FACE]`
- Automatic cleanup if one system fails
- **Warning:** Processes will stop if SSH session ends

**Best for:** Development, debugging, when you want to see live output

### Helper Scripts

#### Setup Checker (Run This First!)
```bash
python3 setup_check.py
```

Verifies before launching:
- Dish detection model files present
- Face recognition model trained
- Training data available
- Shows what's missing and how to fix it

**Run this first** before launching the dual system!

#### Check Status
```bash
python3 check_status.py
```

Shows:
- Which systems are running (with PIDs)
- Recent log output (last 5 lines)
- Access URLs
- Command hints

#### Stop Systems
```bash
python3 stop_dual_system.py
```

Gracefully stops both dish detection and face recognition processes.

## Usage

### Prerequisites

Make sure you have:
- Both cameras connected (camera 0 and camera 4)
- Face recognition model trained (see [facial_recognition_scripts/README.md](../facial_recognition_scripts/README.md))
- YOLO dish detection model available
- All dependencies installed (see [requirements.txt](../requirements.txt))

### Running on Raspberry Pi (via SSH)

**First time setup (make scripts executable):**
```bash
cd main_scripts
chmod +x *.py
```

**Recommended method for SSH:**

```bash
cd main_scripts
python3 launch_dual_system_simple.py
# Or: ./launch_dual_system_simple.py
```

This will:
- Launch both systems in background
- Create log files in `logs/` directory
- Continue running even if you disconnect from SSH
- Can be accessed from any device on your network

**Check if systems are running:**
```bash
python3 check_status.py
```

**View live logs:**
```bash
# Dish detection
tail -f ../logs/dish_detector.log

# Face recognition
tail -f ../logs/face_recognition.log
```

**Stop both systems:**
```bash
python3 stop_dual_system.py
```

Access the interfaces:
- Dish Detection: http://dish-bot.local:5000
- Face Recognition: http://dish-bot.local:5002

### Running on Windows (Development)

```bash
cd main_scripts
python launch_dual_system_simple.py
```

Access the interfaces:
- Dish Detection: http://localhost:5000
- Face Recognition: http://localhost:5002

### Stopping the Systems

**Simple Launcher:**
- Close each terminal window individually

**Advanced Launcher:**
- Press Ctrl+C once to stop both systems gracefully

## Configuration

Edit the individual scripts to configure:

**Dish Detector** ([dish_detector_groupme.py](../dish_detection_scripts/dish_detector_groupme.py:18)):
- `ENABLE_GROUPME`: Enable/disable GroupMe notifications
- `STATIONARY_THRESHOLD`: Time before first alert (default: 15s)
- Camera index, resolution, etc.

**Face Recognition** ([face_recognition_web.py](../facial_recognition_scripts/face_recognition_web.py:15)):
- `CAMERA_INDEX`: Camera device (default: 4)
- `CONFIDENCE_THRESHOLD`: Recognition confidence (default: 85)
- Processing frequency, resolution, etc.

## Next Steps

After verifying both systems work independently:
1. Create integrated script that combines dish detection + face ID
2. Send person's name with dish alerts to GroupMe
3. Add logic for "unknown person" alerts
4. Implement dish-to-person association tracking

## Troubleshooting

### "Cannot open camera"
- Check camera indices with: `ls /dev/video*` (Linux) or Device Manager (Windows)
- Make sure cameras aren't already in use by another process
- Verify camera permissions

### Port already in use
- Make sure no other Flask apps are running on ports 5000 or 5002
- Kill existing processes: `lsof -i :5000` and `kill <PID>` (Linux)

### Face recognition model not found
- Train the model first:
  ```bash
  cd facial_recognition_scripts
  python train_faces_opencv.py
  ```

### YOLO model not found
- Check that `best_ncnn_model/` exists in project root
- Verify model files are present
