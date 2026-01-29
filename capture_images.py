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

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera initialized successfully!")
print("\n=== CONTROLS ===")
print("1. Click on the camera window")
print("2. Press SPACE to capture")
print("3. Press Q to quit")
print("=================\n")

image_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading frame")
        break
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Calculate 60% zoom (keep center 40% of image)
    crop_percent = 0.4
    new_width = int(width * crop_percent)
    new_height = int(height * crop_percent)
    
    # Calculate crop coordinates to center the crop
    x_start = (width - new_width) // 2
    y_start = (height - new_height) // 2
    x_end = x_start + new_width
    y_end = y_start + new_height
    
    # Crop to center 40% (60% zoom)
    cropped_frame = frame[y_start:y_end, x_start:x_end]
    
    # Display live feed with counter and crop indicator
    display_frame = frame.copy()
    
    # Draw rectangle showing crop area
    cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    cv2.putText(display_frame, f"Images: {image_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "SPACE=Capture | Q=Quit", 
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(display_frame, "60% zoom - Green box = capture area", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Dish Detection - Data Capture', display_frame)
    
    # Wait for
