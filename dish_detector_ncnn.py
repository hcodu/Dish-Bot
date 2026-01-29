from ultralytics import YOLO
import cv2
import time
from flask import Flask, Response
import numpy as np

app = Flask(__name__)

# Load NCNN model (much faster on Pi 3!)
model = YOLO("best_ncnn_model", task="detect")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Tracking configuration
TRACK_EXPIRY_TIME = 3.0  # Remove track if not seen for 3 seconds
POSITION_THRESHOLD = 50  # Pixels - max distance to match same dish
next_id = 1

# Tracked dishes: {id: {'center': (x, y), 'last_seen': timestamp, 'bbox': (x1, y1, x2, y2)}}
tracked_dishes = {}

def calculate_center(bbox):
    """Calculate center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(center1, center2):
    """Calculate Euclidean distance between two centers"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def match_detections_to_tracks(detections, current_time):
    """Match current detections to existing tracks or create new ones"""
    global next_id, tracked_dishes
    
    # Remove expired tracks
    expired_ids = [tid for tid, track in tracked_dishes.items() 
                   if current_time - track['last_seen'] > TRACK_EXPIRY_TIME]
    for tid in expired_ids:
        del tracked_dishes[tid]
    
    # Get current detection centers
    detection_centers = []
    detection_bboxes = []
    for box in detections:
        # Handle tensor conversion more safely
        try:
            # Get bbox tensor and convert to numpy
            bbox_tensor = box.xyxy
            # Handle different tensor formats
            if hasattr(bbox_tensor, 'cpu'):
                bbox = bbox_tensor.cpu().numpy()
            elif hasattr(bbox_tensor, 'numpy'):
                bbox = bbox_tensor.numpy()
            else:
                bbox = np.array(bbox_tensor)
            
            # Flatten and take first 4 values [x1, y1, x2, y2]
            bbox = bbox.flatten()[:4]
            if len(bbox) < 4:
                continue
        except Exception as e:
            print(f"Error extracting bbox: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        center = calculate_center(bbox)
        detection_centers.append(center)
        detection_bboxes.append(bbox)
    
    # Match detections to existing tracks
    matched_track_ids = set()
    detection_to_track = {}  # Maps detection index to track ID
    
    for det_idx, det_center in enumerate(detection_centers):
        best_match_id = None
        best_distance = POSITION_THRESHOLD
        
        # Find closest track within threshold
        for track_id, track in tracked_dishes.items():
            if track_id in matched_track_ids:
                continue  # Already matched
            
            distance = calculate_distance(det_center, track['center'])
            if distance < best_distance:
                best_distance = distance
                best_match_id = track_id
        
        if best_match_id is not None:
            # Update existing track
            tracked_dishes[best_match_id]['center'] = det_center
            tracked_dishes[best_match_id]['last_seen'] = current_time
            tracked_dishes[best_match_id]['bbox'] = detection_bboxes[det_idx]
            matched_track_ids.add(best_match_id)
            detection_to_track[det_idx] = best_match_id
        else:
            # Create new track
            new_id = next_id
            next_id += 1
            tracked_dishes[new_id] = {
                'center': det_center,
                'last_seen': current_time,
                'bbox': detection_bboxes[det_idx]
            }
            detection_to_track[det_idx] = new_id
    
    return detection_to_track

def draw_ids_on_frame(frame, results, detection_to_track):
    """Draw bounding boxes and IDs on frame"""
    # Use the original plot method first (handles NCNN format correctly)
    try:
        annotated = results[0].plot(line_width=1, labels=False)
    except:
        annotated = frame.copy()
    
    # Now overlay our IDs on top
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            if idx in detection_to_track:
                track_id = detection_to_track[idx]
                try:
                    # Get bbox tensor and convert to numpy
                    bbox_tensor = box.xyxy
                    # Handle different tensor formats
                    if hasattr(bbox_tensor, 'cpu'):
                        bbox = bbox_tensor.cpu().numpy()
                    elif hasattr(bbox_tensor, 'numpy'):
                        bbox = bbox_tensor.numpy()
                    else:
                        bbox = np.array(bbox_tensor)
                    
                    # Flatten and take first 4 values [x1, y1, x2, y2]
                    bbox = bbox.flatten()[:4].astype(int)
                    if len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = bbox
                except Exception as e:
                    print(f"Error extracting bbox for drawing: {e}")
                    continue
                
                # Draw ID label
                label = f"ID: {track_id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                # Draw background rectangle for text
                cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated

def generate_frames():
    global tracked_dishes
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    last_annotated = None
    last_results = None
    last_detection_to_track = {}
    
    # Pre-calculate crop
    fh, fw = 480, 640
    nw, nh = int(fw * 0.4), int(fh * 0.4)
    xs, ys = (fw - nw) // 2, (fh - nh) // 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Crop
        cropped = frame[ys:ys+nh, xs:xs+nw]
        
        # Run inference every 5th frame
        if frame_count % 5 == 0:
            try:
                results = model(cropped, conf=0.4, verbose=False)
                last_results = results
                
                # Match detections to tracks and assign IDs
                if results and len(results[0].boxes) > 0:
                    last_detection_to_track = match_detections_to_tracks(results[0].boxes, current_time)
                    # Draw IDs on frame
                    last_annotated = draw_ids_on_frame(cropped, results, last_detection_to_track)
                else:
                    last_annotated = cropped.copy()
                    last_detection_to_track = {}
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                last_annotated = cropped.copy()
                last_detection_to_track = {}
        
        # Use last annotated frame if available
        display_frame = last_annotated if last_annotated is not None else cropped
        
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time)
            dishes = len(last_results[0].boxes) if last_results and len(last_results[0].boxes) > 0 else 0
            active_tracks = len([t for t in tracked_dishes.values() 
                               if current_time - t['last_seen'] <= TRACK_EXPIRY_TIME])
            print(f"FPS: {fps:.1f} | Dishes: {dishes} | Active Tracks: {active_tracks}")
            start_time = time.time()
        
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Dish Detection (NCNN)</title></head>
    <body style="background-color: #000; text-align: center;">
        <h1 style="color: #fff;">Dish Detection - NCNN Optimized</h1>
        <img src="/video_feed" style="max-width: 90%; border: 2px solid #0f0;">
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\nNCNN web stream starting...")
    print("View at: http://dish-bot.local:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
