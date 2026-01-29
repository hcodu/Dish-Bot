from ultralytics import YOLO
import cv2
import time

# Load your trained model
model = YOLO("best_91.pt")

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Model loaded. Press 'q' to quit.\n")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply same crop as training (60% zoom)
    fh, fw = frame.shape[:2]
    crop_percent = 0.4
    nw = int(fw * crop_percent)
    nh = int(fh * crop_percent)
    xs = (fw - nw) // 2
    ys = (fh - nh) // 2
    cropped = frame[ys:ys+nh, xs:xs+nw]
    
    # Run inference (conf threshold and hide verbose output)
    results = model(cropped, conf=0.5, verbose=False)
    
    # Draw detections
    annotated_frame = results[0].plot()
    
    # Calculate FPS every 30 frames
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = 30 / elapsed
        detections = len(results[0].boxes)
        print(f"FPS: {fps:.2f} | Dishes detected: {detections}")
        start_time = time.time()
    
    # Display result
    cv2.imshow('Dish Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nInference stopped")
