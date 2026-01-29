from ultralytics import YOLO
import cv2
import time
from flask import Flask, Response, request
import numpy as np
import requests
from datetime import datetime
import os
import tempfile
import threading
import sys
import base64
from urllib.parse import quote

app = Flask(__name__)

# Load NCNN model (much faster on Pi 3!)
model = YOLO("../best_ncnn_model", task="detect")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# GroupMe Bot Configuration
# Get these from: https://dev.groupme.com/bots
BOT_ID = "fcb2048f4132da6ac6f1ea97c6"  # Your GroupMe bot ID
ACCESS_TOKEN = "nvgRcnb0i5KFWYAChTTeMX68gvtYUHnSkJTMa6xW"  # Your GroupMe access token (required for image service)
GROUP_ID = "112821791"  # Your GroupMe group ID
AUTHORIZED_USER_ID = None  # Set to GroupMe user ID to restrict /stop command (get from webhook)
# Leave as None to allow anyone in the group to use /stop

# Webhook configuration (OPTIONAL - only needed for commands like /stop, /status)
# If you don't set up a webhook, alerts will still work, but commands won't
ENABLE_WEBHOOK = False  # Set to True if you have a public URL for webhooks
# If False, you can still send alerts, but can't receive commands

# Image hosting configuration
# GroupMe requires image URLs. We'll use GroupMe Image Service (requires access token)
USE_GROUPME_IMAGE_SERVICE = True  # Must be True - uses GroupMe Image Service via API

# Tracking configuration
TRACK_EXPIRY_TIME = 15.0  # Remove track if not seen for 15 seconds (increased for fixed camera)
POSITION_THRESHOLD = 150  # Pixels - max distance to match same dish (increased for better matching)
STATIONARY_THRESHOLD = 20.0  # Seconds - time before first alerting
FOLLOWUP_ALERT_INTERVAL = 60.0  # Seconds - time between follow-up alerts
POSITION_STABILITY_THRESHOLD = 50  # Pixels - max movement to consider stationary (increased)
IOU_THRESHOLD = 0.2  # Minimum IoU (Intersection over Union) for matching (0.0 to 1.0)
next_id = 1

# Tracked dishes: {id: {
#   'center': (x, y), 
#   'last_seen': timestamp, 
#   'bbox': (x1, y1, x2, y2),
#   'first_detected': timestamp,  # When first detected at this position
#   'last_position': (x, y),  # Last known position
#   'stationary_start': timestamp,  # When dish became stationary
#   'last_alert_time': timestamp or None,  # When last alert was sent (None if never sent)
#   'position_history': [(x, y), ...]  # History of positions for better matching
# }}
tracked_dishes = {}

# GroupMe command handling
shutdown_requested = False
script_start_time = time.time()  # Track when script started

def calculate_center(bbox):
    """Calculate center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(center1, center2):
    """Calculate Euclidean distance between two centers"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def upload_image_to_groupme(image_path):
    """Upload image to GroupMe Image Service and return URL"""
    if not USE_GROUPME_IMAGE_SERVICE or not ACCESS_TOKEN:
        return None
    
    try:
        url = "https://image.groupme.com/pictures"
        headers = {
            "X-Access-Token": ACCESS_TOKEN,
            "Content-Type": "image/jpeg"
        }
        
        with open(image_path, 'rb') as img_file:
            response = requests.post(url, headers=headers, data=img_file, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result.get("payload", {}).get("url")
    except Exception as e:
        print(f"Error uploading to GroupMe Image Service: {e}")
        return None

def send_groupme_message(text, attachments=None):
    """Send a text message to GroupMe"""
    url = "https://api.groupme.com/v3/bots/post"
    
    payload = {
        "bot_id": BOT_ID,
        "text": text
    }
    
    if attachments:
        payload["attachments"] = attachments
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.status_code == 202  # GroupMe returns 202 for success
    except Exception as e:
        print(f"Error sending GroupMe message: {e}")
        return False

def send_groupme_photo(photo_path, caption=""):
    """Send a photo to GroupMe using GroupMe Image Service (API only, no webhook needed)"""
    # Upload image to GroupMe Image Service
    if not ACCESS_TOKEN:
        print("✗ ACCESS_TOKEN required for sending images")
        return False
    
    image_url = upload_image_to_groupme(photo_path)
    if not image_url:
        print("✗ Failed to upload image to GroupMe Image Service")
        return False
    
    # Create attachment
    attachment = {
        "type": "image",
        "url": image_url
    }
    
    # Send message with image
    return send_groupme_message(caption, [attachment])

def check_and_alert_stationary_dishes(current_time, annotated_frame):
    """Check for dishes that have been stationary for 15+ seconds and send alerts"""
    global tracked_dishes
    
    for dish_id, track in tracked_dishes.items():
        # Check if dish has been stationary long enough
        stationary_start = track.get('stationary_start')
        if stationary_start is None:
            continue
        
        stationary_duration = current_time - stationary_start
        last_alert_time = track.get('last_alert_time')
        
        # Determine if we should send an alert
        should_alert = False
        is_followup = False
        
        if last_alert_time is None:
            # First alert: send after 15 seconds
            if stationary_duration >= STATIONARY_THRESHOLD:
                should_alert = True
        else:
            # Follow-up alerts: send every 30 seconds after the last alert
            time_since_last_alert = current_time - last_alert_time
            if time_since_last_alert >= FOLLOWUP_ALERT_INTERVAL:
                should_alert = True
                is_followup = True
        
        if should_alert:
            # Time to send alert!
            alert_type = "FOLLOW-UP ALERT" if is_followup else "ALERT"
            print(f"🚨 {alert_type}: Dish ID {dish_id} has been stationary for {stationary_duration:.1f} seconds!")
            
            # Save snapshot
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"alert_dish_{dish_id}_{timestamp_str}.jpg"
            snapshot_path = os.path.join(tempfile.gettempdir(), snapshot_filename)
            
            try:
                # Save the annotated frame
                cv2.imwrite(snapshot_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"  Snapshot saved: {snapshot_path}")
                
                # Create caption
                if is_followup:
                    caption = (
                        f"🚨 Dish Still There!\n\n"
                        f"Dish ID: {dish_id}\n"
                        f"Stationary for: {stationary_duration:.1f} seconds\n"
                        f"Time since last alert: {time_since_last_alert:.1f} seconds\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    caption = (
                        f"🚨 Dish Alert!\n\n"
                        f"Dish ID: {dish_id}\n"
                        f"Stationary for: {stationary_duration:.1f} seconds\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                
                # Send to GroupMe
                if send_groupme_photo(snapshot_path, caption):
                    # Update last alert time
                    track['last_alert_time'] = current_time
                    print(f"  ✓ Alert sent for Dish ID {dish_id}")
                else:
                    print(f"  ✗ Failed to send alert for Dish ID {dish_id}")
                
                # Clean up snapshot file after a delay (optional)
                # os.remove(snapshot_path)  # Uncomment if you want to delete after sending
                
            except Exception as e:
                print(f"  ✗ Error saving/sending snapshot: {e}")
                import traceback
                traceback.print_exc()

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
            continue
        
        center = calculate_center(bbox)
        detection_centers.append(center)
        detection_bboxes.append(bbox)
    
    # Match detections to existing tracks using IoU and position
    matched_track_ids = set()
    detection_to_track = {}  # Maps detection index to track ID
    
    for det_idx, det_center in enumerate(detection_centers):
        best_match_id = None
        best_score = -1  # Combined score for matching
        
        det_bbox = detection_bboxes[det_idx]
        
        # Find best matching track
        for track_id, track in tracked_dishes.items():
            if track_id in matched_track_ids:
                continue  # Already matched
            
            # Calculate IoU
            iou = calculate_iou(det_bbox, track['bbox'])
            
            # Calculate center distance
            distance = calculate_distance(det_center, track['center'])
            
            # Also check against position history for better matching
            position_history = track.get('position_history', [])
            min_history_distance = distance
            if position_history:
                for hist_pos in position_history[-3:]:  # Check last 3 positions
                    hist_dist = calculate_distance(det_center, hist_pos)
                    min_history_distance = min(min_history_distance, hist_dist)
            
            # Combined scoring: IoU is more important, but also consider distance
            # Higher IoU = better match, lower distance = better match
            score = iou * 0.7 + (1.0 - min(distance / POSITION_THRESHOLD, 1.0)) * 0.3
            
            # Match if IoU is good OR distance is close
            if (iou >= IOU_THRESHOLD or min_history_distance < POSITION_THRESHOLD) and score > best_score:
                best_score = score
                best_match_id = track_id
        
        if best_match_id is not None:
            # Update existing track
            track = tracked_dishes[best_match_id]
            last_position = track.get('last_position', track['center'])
            position_change = calculate_distance(det_center, last_position)
            
            # Check if dish is stationary (hasn't moved much)
            if position_change <= POSITION_STABILITY_THRESHOLD:
                # Dish is in same position
                if track.get('stationary_start') is None:
                    # Just became stationary
                    track['stationary_start'] = current_time
                    track['first_detected'] = current_time
                    print(f"  Dish ID {best_match_id} is now stationary at position")
            else:
                # Dish has moved significantly - reset stationary tracking
                track['stationary_start'] = None
                track['first_detected'] = current_time
                track['last_alert_time'] = None  # Reset alert time if dish moved
                print(f"  Dish ID {best_match_id} moved - resetting stationary timer")
            
            # Update track info
            track['center'] = det_center
            track['last_seen'] = current_time
            track['last_position'] = det_center
            track['bbox'] = detection_bboxes[det_idx]
            
            # Update position history (keep last 5 positions)
            if 'position_history' not in track:
                track['position_history'] = []
            track['position_history'].append(det_center)
            if len(track['position_history']) > 5:
                track['position_history'].pop(0)
            
            matched_track_ids.add(best_match_id)
            detection_to_track[det_idx] = best_match_id
        else:
            # Create new track
            new_id = next_id
            next_id += 1
            tracked_dishes[new_id] = {
                'center': det_center,
                'last_seen': current_time,
                'bbox': detection_bboxes[det_idx],
                'first_detected': current_time,
                'last_position': det_center,
                'stationary_start': current_time,  # Start tracking stationary time
                'last_alert_time': None,  # No alerts sent yet
                'position_history': [det_center]  # Initialize position history
            }
            detection_to_track[det_idx] = new_id
            print(f"  New dish detected: ID {new_id}")
    
    return detection_to_track

def draw_ids_on_frame(frame, results, detection_to_track, current_time):
    """Draw bounding boxes and IDs on frame, with alert status"""
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
                track = tracked_dishes.get(track_id, {})
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
                    continue
                
                # Determine color based on alert status
                last_alert_time = track.get('last_alert_time')
                stationary_start = track.get('stationary_start')
                
                if last_alert_time is not None:
                    # Alert has been sent - show red
                    time_since_alert = current_time - last_alert_time
                    color = (0, 0, 255)  # Red - alert sent
                    if time_since_alert < 60:
                        status_text = f"ALERTED ({time_since_alert:.0f}s ago)"
                    else:
                        status_text = "ALERTED"
                elif stationary_start:
                    stationary_duration = current_time - stationary_start
                    if stationary_duration >= STATIONARY_THRESHOLD:
                        color = (0, 165, 255)  # Orange - about to alert or waiting for follow-up
                        status_text = f"ALERT ({stationary_duration:.0f}s)"
                    else:
                        color = (0, 255, 255)  # Yellow - stationary
                        status_text = f"{stationary_duration:.0f}s"
                else:
                    color = (0, 255, 0)  # Green - moving
                    status_text = ""
                
                # Draw bounding box with status color
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID label
                label = f"ID: {track_id}"
                if status_text:
                    label += f" ({status_text})"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                # Draw background rectangle for text
                cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated

def generate_frames():
    global tracked_dishes, shutdown_requested
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
    
    while not shutdown_requested:
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
                    # Draw IDs on frame with status
                    last_annotated = draw_ids_on_frame(cropped, results, last_detection_to_track, current_time)
                    
                    # Check for stationary dishes and send alerts
                    check_and_alert_stationary_dishes(current_time, last_annotated)
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
            
            # Count stationary dishes (that haven't been alerted yet, or waiting for follow-up)
            stationary_count = sum(1 for t in tracked_dishes.values() 
                                 if t.get('stationary_start') is not None)
            
            print(f"FPS: {fps:.1f} | Dishes: {dishes} | Active Tracks: {active_tracks} | Stationary: {stationary_count}")
            start_time = time.time()
        
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webhook', methods=['POST'])
def groupme_webhook():
    """Handle GroupMe webhook for commands (optional - only if ENABLE_WEBHOOK is True)"""
    if not ENABLE_WEBHOOK:
        return '', 200  # Webhook disabled, just return OK
    
    global shutdown_requested, script_start_time
    
    try:
        data = request.get_json()
        
        # Ignore messages from the bot itself
        if data.get('sender_type') == 'bot':
            return '', 200
        
        # Check if message is from the configured group
        group_id = data.get('group_id')
        if str(group_id) != str(GROUP_ID):
            return '', 200
        
        # Get message info
        text = data.get('text', '').strip().lower()
        user_id = data.get('user_id')
        name = data.get('name', 'Unknown')
        
        # Ignore messages older than when script started
        created_at = data.get('created_at', 0)
        if created_at < (script_start_time - 5):
            return '', 200  # Ignore old messages
        
        # Process commands
        if text in ["/stop", "/shutdown", "/exit", "stop", "shutdown"]:
            # Check if user is authorized to stop the system
            if AUTHORIZED_USER_ID is not None:
                if str(user_id) != str(AUTHORIZED_USER_ID):
                    # Unauthorized user tried to stop
                    print(f"⚠️  Unauthorized stop attempt from user {user_id} ({name})")
                    send_groupme_message(
                        f"❌ Access Denied\n\n"
                        f"Only authorized users can stop the system.\n"
                        f"Your user ID: {user_id}"
                    )
                    return '', 200
            
            # Authorized - proceed with shutdown
            print(f"\n🛑 STOP command received from GroupMe!")
            print(f"   User: {user_id} ({name})")
            send_groupme_message("🛑 Stopping dish detection system...")
            shutdown_requested = True
            return '', 200
        
        elif text in ["/status", "/info", "status"]:
            # Send status information
            current_time = time.time()
            active_tracks = len([t for t in tracked_dishes.values() 
                               if current_time - t['last_seen'] <= TRACK_EXPIRY_TIME])
            stationary_count = sum(1 for t in tracked_dishes.values() 
                                 if t.get('stationary_start') is not None)
            
            status_msg = (
                f"📊 System Status\n\n"
                f"Active tracks: {active_tracks}\n"
                f"Stationary dishes: {stationary_count}\n"
                f"System: Running"
            )
            send_groupme_message(status_msg)
            return '', 200
        
        elif text in ["/myid", "/id", "myid"]:
            # Show user's ID
            id_msg = (
                f"🆔 Your GroupMe Info\n\n"
                f"User ID: {user_id}\n"
                f"Name: {name}\n\n"
                f"Use this ID in AUTHORIZED_USER_ID to restrict /stop command."
            )
            send_groupme_message(id_msg)
            return '', 200
        
        elif text in ["/help", "help"]:
            stop_note = ""
            if AUTHORIZED_USER_ID is not None:
                stop_note = f"\n⚠️ /stop is restricted to authorized users only"
            
            help_msg = (
                f"🤖 Dish Detection Bot Commands\n\n"
                f"/stop - Stop the detection system{stop_note}\n"
                f"/status - Get system status\n"
                f"/myid - Show your user ID\n"
                f"/help - Show this help message"
            )
            send_groupme_message(help_msg)
            return '', 200
        
    except Exception as e:
        print(f"Error processing GroupMe webhook: {e}")
        import traceback
        traceback.print_exc()
    
    return '', 200

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Dish Detector!</title></head>
    <body style="background-color: #000; text-align: center;">
        <h1 style="color: #fff;">Dish Detection - NCNN with GroupMe Alerts</h1>
        <p style="color: #0f0;">First alert at 15s, then every 30s while dish remains</p>
        <img src="/video_feed" style="max-width: 90%; border: 2px solid #0f0;">
    </body>
    </html>
    '''

def shutdown_handler():
    """Handle graceful shutdown"""
    global shutdown_requested
    
    print("\n" + "=" * 50)
    print("Shutting down dish detection system...")
    print("=" * 50)
    
    try:
        # Release camera
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released")
    except:
        pass
    
    try:
        # Send final message
        send_groupme_message("✅ Dish detection system stopped")
    except:
        pass
    
    print("System stopped.")
    os._exit(0)

def shutdown_monitor():
    """Monitor for shutdown request and exit when needed"""
    global shutdown_requested
    while True:
        if shutdown_requested:
            time.sleep(0.5)  # Brief delay to allow final operations
            shutdown_handler()
        time.sleep(0.5)  # Check every 0.5 seconds

if __name__ == '__main__':
    # Set script start time (used to ignore old messages)
    script_start_time = time.time()
    
    print("\n" + "=" * 50)
    print("Dish Detection with GroupMe Alerts")
    print("=" * 50)
    print(f"First alert threshold: {STATIONARY_THRESHOLD} seconds")
    print(f"Follow-up alert interval: {FOLLOWUP_ALERT_INTERVAL} seconds")
    print(f"Position stability: {POSITION_STABILITY_THRESHOLD} pixels")
    print("\n" + "=" * 50)
    print("ALERT SYSTEM: Works via API only (no webhook needed!)")
    print("=" * 50)
    print("✓ Alerts will be sent automatically when dishes are detected")
    print("✓ Images uploaded via GroupMe Image Service API")
    print("✓ No public URL or ngrok required for alerts")
    
    if ENABLE_WEBHOOK:
        print("\n" + "=" * 50)
        print("COMMANDS: Enabled (requires webhook)")
        print("=" * 50)
        print("  /stop - Stop the system")
        print("  /status - Get system status")
        print("  /myid - Show your user ID")
        print("  /help - Show help")
        if AUTHORIZED_USER_ID is not None:
            print(f"\n⚠️  /stop command restricted to user ID: {AUTHORIZED_USER_ID}")
        else:
            print("\n⚠️  /stop command available to anyone in the group")
        print("\n⚠️  Configure webhook URL in bot settings:")
        print("   https://dev.groupme.com/bots")
        print("   Webhook URL: http://YOUR_PUBLIC_IP:5000/webhook")
    else:
        print("\n" + "=" * 50)
        print("COMMANDS: Disabled (webhook not configured)")
        print("=" * 50)
        print("⚠️  Commands like /stop, /status require a webhook")
        print("   Set ENABLE_WEBHOOK = True and configure webhook to enable")
        print("   Alerts will still work without webhook!")
    
    print("\nNCNN web stream starting...")
    print("View at: http://dish-bot.local:5000")
    print("=" * 50 + "\n")
    
    # Start shutdown monitor thread
    monitor_thread = threading.Thread(target=shutdown_monitor, daemon=False)
    monitor_thread.start()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        shutdown_requested = True
        shutdown_handler()
