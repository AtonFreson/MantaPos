# Add parent directory to the path to import mantaPosLib
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

# Import necessary libraries
import cv2
import numpy as np
import pytesseract
from collections import deque
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\A242937\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
import time
import mantaPosLib as manta

CAMERA_RTSP_ADDR = "rtsp://admin:@169.254.178.12:554/"

# Initialize time reference
start_time = time.time()

# Initialize video capture (camera pointed at the screen)
cap = manta.RealtimeCapture(CAMERA_RTSP_ADDR)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Font settings for timestamp overlay
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 10
text_color = (255, 255, 255)  # White color
bg_color = (0, 0, 0)  # Black background

# Initialize a deque to store the last 10 delay values
rolling_delays = deque(maxlen=10)

while True:
    # Calculate current timestamp in milliseconds since the script started
    current_time_ms = int((time.time() - start_time) * 1000)
    timestamp_text = f"{current_time_ms}"

    # Create a black image to display the timestamp
    display_frame = np.zeros((800, 1200, 3), dtype=np.uint8)

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)

    # Calculate text position to center it
    text_x = (display_frame.shape[1] - text_width) // 2
    text_y = (display_frame.shape[0] + text_height) // 2

    # Put the timestamp text on the display frame
    cv2.putText(display_frame, timestamp_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Display the timestamp window
    cv2.imshow('Timestamp Display', display_frame)

    # Capture frame from the camera (pointed at the screen)
    ret, captured_frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    # Record capture time immediately after capturing the frame
    capture_time_ms = int((time.time() - start_time) * 1000)

    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

    # Crop the center region where the timestamp is expected
    crop_width = 1600  # Adjust based on the size of the timestamp on the captured frame
    crop_height = 800
    center_x = captured_frame.shape[1] // 2
    center_y = captured_frame.shape[0] // 2
    crop_x1 = max(center_x - crop_width // 2, 0)
    crop_y1 = max(center_y - crop_height // 2, 0)
    crop_x2 = min(center_x + crop_width // 2, captured_frame.shape[1])
    crop_y2 = min(center_y + crop_height // 2, captured_frame.shape[0])
    cropped = gray_frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Preprocess the cropped image for OCR
    _, thresh = cv2.threshold(cropped, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)  # Invert colors if necessary

    # Use pytesseract to extract the timestamp from the captured frame
    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    extracted_text = pytesseract.image_to_string(thresh, config=config).strip()

    try:
        detected_time_ms = int(extracted_text)
        # Calculate the delay using capture time
        delay_ms = capture_time_ms - detected_time_ms
        if delay_ms >= 0:
            # Add the delay to the rolling list
            rolling_delays.append(delay_ms)
            # Calculate the rolling average
            rolling_avg = sum(rolling_delays) / len(rolling_delays)
            print(f"\nDetected Delay: {delay_ms} ms ({len(rolling_delays)}pt average: {int(rolling_avg)} ms)", end='')
        else:
            print("\nDetected future timestamp, possible OCR error.", end='')
    except ValueError:
        print(".", end='', flush=True)

    # Display the cropped frame (optional)
    resized_cropped = cv2.resize(cropped, (400, 200))
    cv2.imshow('Cropped Frame', resized_cropped)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
