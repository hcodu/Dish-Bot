import cv2
import os
from datetime import datetime
import time


# Create directory for captured images
output_dir = "dish_dataset"
os.makedirs(output_dir, exist_ok=True)


# Initialize camera with V4L2 backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)


# Set MJPEG codec for better performance
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# CRITICAL: Reduce buffer size to minimize lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()


print("Camera initialized successfully!")
print("Resolution: 640x480 -> 256x192 (60% zoom)")
print("\nCommands: c=Capture | a=Auto | s=Stop | q=Quit\n")


image_count = 0
auto_capture = False
last_capture_time = 0


def capture_image():
    global image_count
    
    # Clear buffer by reading and discarding 2-3 frames
    for _ in range(3):
        cap.grab()
    
    # Now get the fresh frame
    ret, frame = cap.read()
    
    if ret:
        height, width = frame.shape[:2]
        
        # Calculate crop (60% zoom)
        crop_percent = 0.4
        new_width = int(width * crop_percent)
        new_height = int(height * crop_percent)
        
        x_start = (width - new_width) // 2
        y_start = (height - new_height) // 2
        x_end = x_start + new_width
        y_end = y_start + new_height
        
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/dish_{timestamp}_{image_count:04d}.jpg"
        cv2.imwrite(filename, cropped_frame)
        image_count += 1
        print(f"✓ #{image_count}: {filename}")
        return True
    else:
        print("✗ Failed")
        return False


print("Ready!")

seconds_interval = 120

try:
    while True:
        if auto_capture:
            current_time = time.time()
            if current_time - last_capture_time >= seconds_interval:
                capture_image()
                last_capture_time = current_time
            
            import sys
            import select
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 's':
                    auto_capture = False
                    print("Auto-capture STOPPED")
                elif cmd == 'q':
                    break
            time.sleep(0.1)
        else:
            cmd = input("> ").strip().lower()
            
            if cmd == 'c':
                capture_image()
            elif cmd == 'a':
                auto_capture = True
                last_capture_time = time.time()
                print("Auto-capture STARTED (2 min interval)")
            elif cmd == 'q':
                break

except KeyboardInterrupt:
    print("\n\nInterrupted")

cap.release()
print(f"\nTotal: {image_count} images in {output_dir}/")
