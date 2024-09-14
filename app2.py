import cv2
import numpy as np
import time
import math
from collections import deque
from PIL import Image

def get_limits(color):
    # Placeholder function to get color limits
    if color == [0, 255, 255]:  # Yellow
        lower_limit = np.array([20, 100, 100])
        upper_limit = np.array([30, 255, 255])
    return lower_limit, upper_limit

def generate_frames():
    yellow = [0, 255, 255]  # Enemy characteristic
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Load and resize the radar background image
    background = cv2.imread('gid_image.png')
    if background is None:
        print("Error: Could not load radar background image.")
        return
    background = cv2.resize(background, (640, 480))  # Resize to match webcam resolution

    prev_position = None
    prev_time = time.time()
    speed_history = deque(maxlen=10)  # Store last 10 speeds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frames from BGR to HSV
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_limit, upper_limit = get_limits(color=yellow)

        # Create a binary mask that highlights the regions of an image that fall within a specified color range
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        # Creates an image object from array
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        # Overlay detection results on the background
        detection_layer = np.zeros_like(background)  # Create a blank layer for detections

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            detection_layer = cv2.rectangle(detection_layer, (x1, y1), (x2, y2), (0, 0, 255), 5)
            
            # Find the midpoint of the object detected
            midpoint = [(x1 + x2) // 2, (y1 + y2) // 2]
            
            # Calculate the distance between prev position and this curr position of enemy
            curr_time = time.time()
            if prev_position is not None:
                dist = np.sqrt((midpoint[0] - prev_position[0]) ** 2 + (midpoint[1] - prev_position[1]) ** 2)
                time_diff = curr_time - prev_time
                if time_diff > 0:
                    speed = dist / time_diff
                    speed_history.append(speed)
                prev_position = midpoint
                prev_time = curr_time

                # Calculate the angle between enemy and aircraft
                w, h = 320, 240  # Update to match half of background dimensions
                aircraft = [w, h]
                enemy = midpoint
                theta_rad = math.atan2((enemy[0] - aircraft[0]), (aircraft[1] - enemy[1]))
                angle_deg = math.degrees(theta_rad)
                if angle_deg < 0:
                    angle_deg += 360

                # Calculate the moving average speed
                avg_speed = np.mean(speed_history) if speed_history else 0

                # Print for debugging
                print(f"Enemy at angle: {angle_deg:.2f} degrees, Speed: {avg_speed:.2f} pixels/sec")

                # Create a circle around the enemy
                detection_layer = cv2.circle(detection_layer, (midpoint[0], midpoint[1]), 5, (0, 0, 255), -1)
                
                # Overlay speed and angle on the detection layer
                cv2.putText(detection_layer, f"Speed: {avg_speed:.2f} px/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(detection_layer, f"Angle: {angle_deg:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine background with detection overlay
        combined_frame = cv2.addWeighted(background, 1.0, detection_layer, 1.0, 0)

        ret, buffer = cv2.imencode('.jpg', combined_frame)
        if not ret:
            print("Error: Frame encoding failed.")
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    cv2.destroyAllWindows()
