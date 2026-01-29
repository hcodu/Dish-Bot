from ultralytics import YOLO
import cv2
import time
from flask import Flask, Response

app = Flask(__name__)

model = YOLO("best_91.pt")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Keep original resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def generate_frames():
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    last_annotated = None
    
    # Pre-calculate crop values (optimization #16)
    fh, fw = 480, 640
    nw, nh = int(fw * 0.4), int(fh * 0.4)
    xs, ys = (fw - nw) // 2, (fh - nh) // 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop using pre-calculated values
        cropped = frame[ys:ys+nh, xs:xs+nw]
        
        # Run inference every 3rd frame (optimization #7)
        if frame_count % 3 == 0:
            # Smaller input size + higher confidence + limit detections (optimizations #3, #10, #13)
            results = model(cropped, conf=0.4, verbose=False, imgsz=320, max_det=10, iou=0.7)
            # Simplified annotation - no labels (optimization #15)
            last_annotated = results[0].plot(line_width=1, labels=False)
        
        # Use last annotated frame
        display_frame = last_annotated if last_annotated is not None else cropped
        
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time)
            dishes = len(results[0].boxes) if 'results' in locals() else 0
            print(f"FPS: {fps:.1f} | Dishes: {dishes}")
            start_time = time.time()
        
        # Lower JPEG quality for faster encoding (optimization #8)
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
    <head><title>Dish Detection</title></head>
    <body style="background-color: #000; text-align: center;">
        <h1 style="color: #fff;">Dish Detection Live Feed</h1>
        <img src="/video_feed" style="max-width: 90%; border: 2px solid #0f0;">
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\nWeb stream starting...")
    print("View at: http://dish-bot.local:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
