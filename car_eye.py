from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load pre-trained classifiers
car_tracker_file = 'cas4.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Placeholder for tracking cars across frames
previous_positions = {}

# Function to estimate speed
def estimate_speed(car_id, x, y):
    global previous_positions
    speed_text = "Calculating..."
    
    if car_id in previous_positions:
        prev_x, prev_y, prev_time = previous_positions[car_id]
        distance_pixels = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

        # Time elapsed in seconds
        time_elapsed = time.time() - prev_time

        # Conversion factor - real calibration needed
        pixel_to_meter = 0.05
        distance_meters = distance_pixels * pixel_to_meter

        # Speed = distance / time (m/s and then converted to km/h)
        speed_m_per_sec = distance_meters / time_elapsed
        speed_km_per_hr = speed_m_per_sec * 3.6

        speed_text = f"{speed_km_per_hr:.2f} km/h"
    
    previous_positions[car_id] = (x, y, time.time())
    return speed_text

def generate_frames():
    video = cv2.VideoCapture(0)
    
    while True:
        success, frame = video.read()  # Read the camera frame
        if not success:
            break
        else:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_tracker.detectMultiScale(grayscaled_frame)
            pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                car_id = f"{x}_{y}"
                speed_text = estimate_speed(car_id, x, y)
                cv2.putText(frame, speed_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            for (x, y, w, h) in pedestrians:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Ensure 'index.html' exists in the 'templates' directory and properly references '/video_feed'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
