#!/usr/bin/env python3
"""
Simple Flask Camera Test
Streams camera 1 video feed to webpage for testing
"""

from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

# Camera 1 (second camera)
cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def generate_frames():
    """Generate video frames from camera"""
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break

        frame_count += 1

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time)
            print(f"FPS: {fps:.1f}")
            start_time = time.time()

        # Add FPS counter to frame
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Home page with video feed"""
    return '''
    <html>
    <head>
        <title>Camera 1 Test</title>
        <style>
            body {
                background-color: #1a1a1a;
                text-align: center;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }
            h1 {
                color: #00ff00;
                margin-bottom: 20px;
            }
            .info {
                color: #888;
                margin-bottom: 20px;
            }
            img {
                max-width: 90%;
                border: 3px solid #00ff00;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
            }
        </style>
    </head>
    <body>
        <h1>Camera 1 Test Feed</h1>
        <div class="info">
            Testing second camera (index 1) - 640x480
        </div>
        <img src="/video_feed">
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Camera 1 Test Web Stream")
    print("="*60)
    print("Camera index: 1 (second camera)")
    print("Resolution: 640x480")
    print("Backend: V4L2")
    print("\nStarting server...")
    print("View at: http://dish-bot.local:5001")
    print("Or: http://<your-pi-ip>:5001")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping server...")
        cap.release()
