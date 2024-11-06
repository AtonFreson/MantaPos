# Add parent directory to the path to import mantaPosLib
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
from collections import deque
import time
import mantaPosLib as manta

CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.12:554/"
FRAME_HISTORY_SIZE = 30  # Store the last 30 frames (2 seconds at 15 FPS)
NUM_DISTINCT_COLOURS = 32  # Number of distinct colours to cycle through
DISPLAY_FPS = 5  # Target FPS for colour display, slightly slower than camera
FRAME_TIME_MS = 1000 / DISPLAY_FPS  # Time per frame in milliseconds
MAX_COLOR_DISTANCE_PCT = 0.15  # Maximum allowed color difference as percentage of full range

roi_size = 900  # Increased ROI size for better colour averaging

def generate_distinct_colours(n):
    """Generate n maximally distinct colours using golden ratio method"""
    colours = []
    golden_ratio = 1/(NUM_DISTINCT_COLOURS+1)
    #golden_ratio = 0.618033988749895
    hue = 0
    
    for i in range(n):
        # Use golden ratio to spread hues evenly
        hue = (hue + golden_ratio) % 1.0
        
        # Use maximum saturation and value for best distinction
        hsv = np.array([[[hue * 360, 1.0, 1.0]]], dtype=np.float32)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Convert to 8-bit RGB values
        colour = (
            int(rgb[0][0][0] * 255),
            int(rgb[0][0][1] * 255),
            int(rgb[0][0][2] * 255)
        )
        colours.append(colour)
    
    return colours

def calculate_colour_distance(c1, c2):
    """Calculate normalized colour distance between two RGB colours"""
    # Calculate per-channel differences and normalize to [0,1]
    diff_r = abs(c1[0] - c2[0]) / 255.0
    diff_g = abs(c1[1] - c2[1]) / 255.0
    diff_b = abs(c1[2] - c2[2]) / 255.0
    
    # Return average channel difference as percentage
    return (diff_r + diff_g + diff_b) / 3.0

def colour_to_timestamp(colour, colour_history):
    """Find the closest matching colour in the history and return its timestamp"""
    best_match_time = None
    min_distance = float('inf')
    
    for timestamp, hist_colour in colour_history:
        distance = calculate_colour_distance(colour, hist_colour)
        if distance < min_distance:
            min_distance = distance
            best_match_time = timestamp
    
    return best_match_time, min_distance

# Generate distinct colours
distinct_colours = generate_distinct_colours(NUM_DISTINCT_COLOURS)

# Initialize time reference
start_time = time.time()
last_frame_time = start_time

# Initialize video capture
cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Initialize colour history as a deque of (timestamp, colour) tuples
colour_history = deque(maxlen=FRAME_HISTORY_SIZE)
rolling_delays = deque(maxlen=50)  # Keep last 10 delay measurements for averaging

# Create named windows with fixed size
cv2.namedWindow('Colour Display', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Colour Display', 1200, 800)
cv2.namedWindow('Camera ROI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera ROI', 300, 300)

frame_counter = 0

while True:
    current_time = time.time()
    elapsed_since_last_frame = (current_time - last_frame_time) * 1000
    
    # Only update display if enough time has passed
    if elapsed_since_last_frame >= FRAME_TIME_MS:
        # Calculate current timestamp
        current_time_ms = int((current_time - start_time) * 1000)
        
        # Select current colour based on frame counter
        current_colour = distinct_colours[frame_counter % NUM_DISTINCT_COLOURS]
        frame_counter += 1
        
        # Add to colour history
        colour_history.append((current_time_ms, current_colour))
        
        # Create and display solid colour frame
        display_frame = np.full((600, 800, 3), current_colour, dtype=np.uint8)
        
        # Add frame counter and colour index overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, f'Frame: {frame_counter}', (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(display_frame, f'Colour: {frame_counter % NUM_DISTINCT_COLOURS}', (10, 60), font, 1, (0, 0, 0), 2)
        actual_fps = 1000 / elapsed_since_last_frame
        cv2.putText(display_frame, f'FPS: {actual_fps:.1f}', (10, 90), font, 1, (0, 0, 0), 2)
        cv2.putText(display_frame, f'History: {len(colour_history)}/{FRAME_HISTORY_SIZE}', (10, 120), font, 1, (0, 0, 0), 2)
        
        cv2.imshow('Colour Display', display_frame)
        last_frame_time = current_time
    
        # Capture frame from camera
        ret, captured_frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        
        # Record capture time
        capture_time_ms = int((time.time() - start_time) * 1000)
        
        # Calculate average colour in the center region of the captured frame
        center_x = captured_frame.shape[1] // 2
        center_y = captured_frame.shape[0] // 2

        roi = captured_frame[
            center_y - roi_size//2:center_y + roi_size//2,
            center_x - roi_size//2:center_x + roi_size//2
        ]
        average_colour = tuple(map(int, cv2.mean(roi)[:3]))
        
        # Find matching timestamp from colour history
        detected_time_ms, colour_distance = colour_to_timestamp(average_colour, colour_history)
        
        # Display ROI with debug information
        roi_display = roi.copy()
        cv2.putText(roi_display, f'Avg Colour: {average_colour}', (10, 30), font, 0.5, (255, 255, 255), 1)
        cv2.putText(roi_display, f'Dist: {colour_distance:.1%}', (10, 50), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(captured_frame, 
                    (center_x - roi_size//2, center_y - roi_size//2),
                    (center_x + roi_size//2, center_y + roi_size//2),
                    (0, 255, 0), 2)
        
        if detected_time_ms is not None and colour_distance < MAX_COLOR_DISTANCE_PCT:
            delay_ms = capture_time_ms - detected_time_ms
            if delay_ms >= 0:  # Only record non-negative delays
                rolling_delays.append(delay_ms)
                rolling_avg = sum(rolling_delays) / len(rolling_delays)
                print(f"Delay: {delay_ms}ms\t(avg: {int(rolling_avg)}ms)\tDistance: {colour_distance:.1%}")
                
                # Add delay information to ROI display
                cv2.putText(roi_display, f'Delay: {delay_ms}ms', (10, 120), font, 3, (255, 255, 255), 7)
        
        # Display the ROI and full frame
        cv2.imshow('Camera ROI', roi_display)
        #cv2.imshow('Camera View', captured_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()