import cv2
import numpy as np
import time

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

# Video capture (0 for the first webcam)
video = cv2.VideoCapture(0)

while True:
    # Read current frame
    read_successful, frame = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around cars and display estimated speed
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # ID based on position
        car_id = f"{x}_{y}"  
        speed_text = estimate_speed(car_id, x, y)
        cv2.putText(frame, speed_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw rectangles around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('Car and Pedestrian Detector with Speed Estimation', frame)

    # Break loop with 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
