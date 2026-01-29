# Integrated Dish Bot

Unified Flask application combining dish detection and face recognition to automatically identify who places dishes in the sink.

## Features

- **Dual Camera Live Feeds**: Side-by-side display of dish detection (camera 0) and face recognition (camera 4)
- **Automatic Association**: Links detected dishes with the person who placed them
- **Smart Tracking**: Waits for dishes to be stationary for 15 seconds before capturing face data
- **Database Storage**: SQLite database stores all detection events with timestamps and snapshots
- **Web Interface**: View live feeds and browse detection history
- **Auto Cleanup**: Automatically deletes snapshot files older than 7 days
- **REST API**: Query statistics and recent events programmatically

## File Structure

```
integrated_dish_bot/
├── app.py                      # Main Flask application
├── database.py                 # SQLite database operations
├── dish_tracker.py             # Dish detection and tracking logic
├── face_recognizer.py          # Face detection and recognition
├── media_cleanup.py            # Automatic media file cleanup
├── templates/
│   ├── index.html             # Live feed dashboard
│   └── history.html           # Detection events history
├── static/
│   └── snapshots/             # Stored dish and face snapshots
└── dish_bot.db                # SQLite database (created on first run)
```

## Requirements

- Python 3.7+
- Two cameras connected (camera 0 for dishes, camera 4 for faces)
- Face recognition model trained (see facial_recognition_scripts/README.md)
- YOLO dish detection model (best_ncnn_model/)

### Python Dependencies

All dependencies are already in the main requirements.txt:
- Flask
- OpenCV (opencv-python)
- Ultralytics (YOLO)
- NumPy

## Setup

### 1. Verify Prerequisites

```bash
# Check cameras are available
ls /dev/video*

# Should see video0 and video4 (or your camera indices)
```

### 2. Train Face Recognition Model (if not done)

```bash
cd ../../facial_recognition_scripts
python3 train_faces_opencv.py
# This creates face_recognizer.xml and known_faces_opencv.pkl
```

### 3. Make Scripts Executable

```bash
chmod +x *.py
```

## Running the Application

### Start the Server

```bash
python3 app.py
# Or: ./app.py
```

The application will:
1. Initialize the database (creates dish_bot.db)
2. Load YOLO model for dish detection
3. Load OpenCV LBPH model for face recognition
4. Start both cameras
5. Begin Flask server on port 5000

### Access the Web Interface

Open a browser and navigate to:
- **Live Feed**: http://dish-bot.local:5000 or http://localhost:5000
- **History**: http://dish-bot.local:5000/history

## Usage

### How It Works

1. **Dish Detection**: Camera 0 watches the sink for dishes
2. **New Dish Appears**: System assigns a unique ID and starts tracking
3. **Face Capture**: Camera 4 captures faces in a 3-second window (-1s to +2s from detection)
4. **Wait Period**: Dish must remain stationary for 15 seconds
5. **Association**: After 15s, system identifies the closest face and links it to the dish
6. **Storage**: Event is saved to database with dish ID, person name, timestamp, and snapshots

### Testing

1. **Stand in front of camera 4** (face camera)
2. **Place a dish in the sink** (visible to camera 0)
3. **Wait 15 seconds** for the association to complete
4. **Check console logs** for confirmation
5. **View history page** to see the recorded event

### Understanding the Display

**Dish Detection Feed (Left)**:
- 🟦 Blue box: Dish is moving
- 🟨 Yellow box: Dish is stationary (waiting)
- 🟧 Orange box: Dish ready for association (15s+)
- 🟩 Green box: Dish associated with person

**Face Recognition Feed (Right)**:
- 🟩 Green box: Person recognized
- 🟥 Red box: Unknown person

## API Endpoints

### GET /api/daily_stats/<date>

Get statistics for a specific day.

**Example**:
```bash
curl http://localhost:5000/api/daily_stats/2026-01-29
```

**Response**:
```json
{
  "date": "2026-01-29",
  "total_dishes": 12,
  "by_person": {
    "John": 5,
    "Sarah": 3,
    "Unknown": 4
  }
}
```

### GET /api/recent_events

Get the 10 most recent detection events.

**Example**:
```bash
curl http://localhost:5000/api/recent_events
```

## Database Schema

```sql
CREATE TABLE dish_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dish_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    date DATE NOT NULL,
    person_name TEXT NOT NULL,
    confidence_score REAL,
    dish_snapshot_path TEXT,
    face_snapshot_path TEXT,
    dish_bbox TEXT,
    face_bbox TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

Edit these constants in `app.py`:

```python
# Camera indices
DISH_CAMERA_INDEX = 0
FACE_CAMERA_INDEX = 4

# Processing rates
DISH_PROCESS_EVERY_N_FRAMES = 5  # Process every 5th frame
FACE_PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame

# Timing
STATIONARY_THRESHOLD = 15.0  # Seconds before association
ASSOCIATION_WINDOW_START = -1.0  # Capture from 1s before
ASSOCIATION_WINDOW_END = 2.0  # Capture until 2s after

# Media retention
MEDIA_RETENTION_DAYS = 7  # Auto-delete after 7 days
```

## Troubleshooting

### "Cannot open camera"
- Check camera indices: `ls /dev/video*`
- Make sure cameras aren't in use by another process
- Update `DISH_CAMERA_INDEX` or `FACE_CAMERA_INDEX` in app.py

### "Face recognition model not found"
- Train the model first:
  ```bash
  cd ../../facial_recognition_scripts
  python3 train_faces_opencv.py
  ```

### "YOLO model not found"
- Verify best_ncnn_model/ exists in project root
- Check model files: `ls ../../best_ncnn_model/`

### Database locked
- Close any other programs accessing dish_bot.db
- Delete dish_bot.db to start fresh (all history will be lost)

### Poor face recognition accuracy
- Retrain with more photos per person (8-10 recommended)
- Adjust CONFIDENCE_THRESHOLD in face_recognizer.py
- Ensure good lighting on face camera

### Dishes not being associated
- Check console logs for dish detection
- Verify dish stays stationary for full 15 seconds
- Make sure someone is in front of face camera when dish is placed

## Media Cleanup

Snapshot files (dish and face photos) are automatically deleted after 7 days to save disk space. Database records are kept permanently.

To manually trigger cleanup:
```python
from media_cleanup import MediaCleaner
from database import Database

db = Database('dish_bot.db')
cleaner = MediaCleaner(db, retention_days=7)
cleaner.manual_cleanup()
```

## Future Enhancements

- Daily leaderboard sent to GroupMe at 10pm
- Manual override commands via GroupMe
- Dish removal tracking (when person comes back to wash)
- Achievement system and streaks
- Real-time statistics dashboard
- Export data to CSV

## Background Deployment

To run as a background service (survives SSH disconnection):

### Option 1: Using nohup

```bash
nohup python3 app.py > output.log 2>&1 &
```

### Option 2: Update launch script

Modify `../launch_dual_system_simple.py` to launch this integrated app instead of separate scripts.

### Stop Background Process

```bash
pkill -f "app.py"
# Or find PID:
ps aux | grep app.py
kill <PID>
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review console logs for error messages
3. Verify all prerequisites are met
4. Check that both cameras are working independently

---

**Version**: 1.0
**Last Updated**: January 2026
**Platform**: Raspberry Pi 4 (works on any Linux system with cameras)
