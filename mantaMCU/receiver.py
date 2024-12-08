import socket
import threading
import json
import sys
import cv2
import numpy as np
from datetime import datetime

# UDP socket parameters
UDP_IP = ''           # Listen on all available network interfaces
UDP_PORT = 13233      # Must match the udpPort in your Arduino code

COMMAND_PORT = 13234  # Port for sending commands

# Create a UDP socket for receiving data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)  # Set a timeout of 1 second

# Create a UDP socket for sending commands
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# List of unit names (can be edited)
unit_names = ["X-axis Encoder", "Z-axis Encoder - Left", "Z-axis Encoder - Right", "Surface Pressure Sensor"]
units_selected = [False] * len(unit_names)

# Shared variables and lock
data_lock = threading.Lock()
received_data = None
stop_threads = False
button_pressed = False
button_press_time = None
wait_for_zero_position_selected = True  # New variable for the checkbox state

# Constants for image dimensions
IMG_WIDTH = 800
IMG_HEIGHT = 900  # Adjusted height for the new elements

# Positions for data display (left side) and unit list (right side)
DATA_START_X = 20
DATA_START_Y = 20

CHECKBOX_START_X = 420
CHECKBOX_START_Y = 20
CHECKBOX_SIZE = 20
CHECKBOX_GAP = 30

# Calculate button coordinates based on unit list
BUTTON_X1 = CHECKBOX_START_X
BUTTON_Y1 = CHECKBOX_START_Y + len(unit_names) * CHECKBOX_GAP + 20
BUTTON_X2 = BUTTON_X1 + 300
BUTTON_Y2 = BUTTON_Y1 + 40

# Positions for the "Wait for Zero Position Signal" checkbox
WAIT_CHECKBOX_X = BUTTON_X1
WAIT_CHECKBOX_Y = BUTTON_Y2 + 30
WAIT_CHECKBOX_SIZE = 20

# Adjust ZERO_BUTTON_Y1 and ZERO_BUTTON_Y2 for the zero button
ZERO_BUTTON_Y1 = WAIT_CHECKBOX_Y + WAIT_CHECKBOX_SIZE + 20
ZERO_BUTTON_Y2 = ZERO_BUTTON_Y1 + 40

def create_data_image(data_dict):
    # Create a black image
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    
    # Left side data display
    lines = []
    lines.append(f"Data Update: {datetime.now().strftime('%H:%M:%S')}")
    lines.append("")
    
    if "encoder" in data_dict:
        enc = data_dict["encoder"]
        lines.append("Encoder Data:")
        lines.append(f"Timestamp * {enc['timestamp']}")
        lines.append(f" Rev:     {enc['revolutions']: .5f}")
        lines.append(f" RPM:    {enc['rpm']: .5f} rpm")
        lines.append(f" Speed:  {enc['speed']: .5f} m/s")
        lines.append(f" Dist:    {enc['distance']: .5f} m")
        lines.append("")
    
    if "imu" in data_dict:
        imu = data_dict["imu"]
        lines.append("IMU Data:")
        lines.append(f"Timestamp * {imu['timestamp']}")
        acc = imu["acceleration"]
        lines.append(f" Accel X:    {acc['x']: .5f} m/s^2")
        lines.append(f" Accel Y:    {acc['y']: .5f} m/s^2")
        lines.append(f" Accel Z:    {acc['z']: .5f} m/s^2")
        gyro = imu["gyroscope"]
        lines.append(f" Gyro X:     {gyro['x']: .5f} rad/s")
        lines.append(f" Gyro Y:     {gyro['y']: .5f} rad/s")
        lines.append(f" Gyro Z:     {gyro['z']: .5f} rad/s")
        lines.append("")
    
    if "temperature" in data_dict:
        temp = data_dict["temperature"]
        lines.append("Temperature Data:")
        lines.append(f"Timestamp * {temp['timestamp']}")
        lines.append(f" Temperature:   {temp['value']: .2f} C")
        lines.append("")
    
    if "pressure" in data_dict:
        press = data_dict["pressure"]
        lines.append("Pressure Data:")
        lines.append(f"Timestamp * {press['timestamp']}")
        lines.append(f" Pressure ADC:   {press['adc_value']}")
    
    # Add text to left side of image
    y = DATA_START_Y
    for line in lines:
        cv2.putText(img, line, (DATA_START_X, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        y += 30
    
    # Right side unit list and checkboxes
    for idx, unit_name in enumerate(unit_names):
        # Draw checkbox
        top_left = (CHECKBOX_START_X, CHECKBOX_START_Y + idx * CHECKBOX_GAP)
        bottom_right = (top_left[0] + CHECKBOX_SIZE, top_left[1] + CHECKBOX_SIZE)
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)
        
        # If selected, draw a tick
        if units_selected[idx]:
            cv2.line(img, 
                     (top_left[0], top_left[1]),
                     (bottom_right[0], bottom_right[1]),
                     (0, 255, 0), 2)
            cv2.line(img,
                     (top_left[0], bottom_right[1]),
                     (bottom_right[0], top_left[1]),
                     (0, 255, 0), 2)
        
        # Draw unit name with index
        text_position = (bottom_right[0] + 10, top_left[1] + CHECKBOX_SIZE - 5)
        cv2.putText(img, f"{idx}: {unit_name}", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    
    # Draw "Reboot Selected" button
    button_color = (70, 70, 70) if not button_pressed else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), button_color, -1)
    cv2.putText(img, "Reboot Selected", (BUTTON_X1 + 20, BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Draw "Wait for Zero Position Signal" checkbox
    checkbox_top_left = (WAIT_CHECKBOX_X, WAIT_CHECKBOX_Y)
    checkbox_bottom_right = (checkbox_top_left[0] + WAIT_CHECKBOX_SIZE, checkbox_top_left[1] + WAIT_CHECKBOX_SIZE)
    cv2.rectangle(img, checkbox_top_left, checkbox_bottom_right, (255, 255, 255), 2)
    
    # If selected, draw a tick
    if wait_for_zero_position_selected:
        cv2.line(img,
                 (checkbox_top_left[0], checkbox_top_left[1]),
                 (checkbox_bottom_right[0], checkbox_bottom_right[1]),
                 (0, 255, 0), 2)
        cv2.line(img,
                 (checkbox_top_left[0], checkbox_bottom_right[1]),
                 (checkbox_bottom_right[0], checkbox_top_left[1]),
                 (0, 255, 0), 2)
    
    # Draw the label next to the checkbox
    text_position = (checkbox_bottom_right[0] + 10, checkbox_top_left[1] + WAIT_CHECKBOX_SIZE - 5)
    cv2.putText(img, "Wait for Zero Position Signal", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    # Draw "Zero Selected" button
    zero_button_color = (70, 70, 70) if not button_pressed else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, ZERO_BUTTON_Y1), (BUTTON_X2, ZERO_BUTTON_Y2), zero_button_color, -1)
    cv2.putText(img, "Zero Selected", (BUTTON_X1 + 20, ZERO_BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    return img

def udp_listener():
    global received_data
    while not stop_threads:
        try:
            data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
            with data_lock:
                received_data = (addr, data.decode())
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error in UDP listener: {e}")
            break

def mouse_callback(event, x, y, flags, param):
    global button_pressed, button_press_time, units_selected, wait_for_zero_position_selected
    if event == cv2.EVENT_LBUTTONUP:
        # Check if click is inside any checkbox
        for idx in range(len(unit_names)):
            top_left = (CHECKBOX_START_X, CHECKBOX_START_Y + idx * CHECKBOX_GAP)
            bottom_right = (top_left[0] + CHECKBOX_SIZE, top_left[1] + CHECKBOX_SIZE)
            if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                units_selected[idx] = not units_selected[idx]
                print(f"Unit {idx} {'selected' if units_selected[idx] else 'deselected'}: {unit_names[idx]}")
                return  # Exit after handling checkbox click
        
        if (WAIT_CHECKBOX_X <= x <= WAIT_CHECKBOX_X + WAIT_CHECKBOX_SIZE and
              WAIT_CHECKBOX_Y <= y <= WAIT_CHECKBOX_Y + WAIT_CHECKBOX_SIZE):

            wait_for_zero_position_selected = not wait_for_zero_position_selected
            print(f"Wait for Zero Position Signal {'selected' if wait_for_zero_position_selected else 'deselected'}")
        
        selected_units = [idx for idx, selected in enumerate(units_selected) if selected]
        command_dict = None

        # Check if click is inside "Reboot Selected" button
        if BUTTON_X1 <= x <= BUTTON_X2 and BUTTON_Y1 <= y <= BUTTON_Y2:
            command_dict = {"units": selected_units, "command": "reboot"}
        # Check if click is inside "Zero Selected" button
        elif BUTTON_X1 <= x <= BUTTON_X2 and ZERO_BUTTON_Y1 <= y <= ZERO_BUTTON_Y2:
            if wait_for_zero_position_selected:
                command_dict = {"units": selected_units, "command": "zero wait"}
            else:
                command_dict = {"units": selected_units, "command": "zero now"}
        
        # Send command if button was pressed and units are selected
        if command_dict and any(units_selected):
            button_pressed = True
            button_press_time = datetime.now()
            
            # Prepare JSON command
            command_dict["units"] = selected_units
            command_json = json.dumps(command_dict)
            
            try:
                cmd_sock.sendto(command_json.encode(), ('<broadcast>', COMMAND_PORT))
                print(f"Sent command: {command_json}")
            except Exception as e:
                print(f"Error sending command: {command_json}, error-code: {e}")
        elif command_dict:
            print("No units selected")
            

# Start UDP listener thread
udp_thread = threading.Thread(target=udp_listener)
udp_thread.start()

# Create OpenCV window
cv2.namedWindow("Sensor Data", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sensor Data", IMG_WIDTH, IMG_HEIGHT)

# Set mouse callback function
cv2.setMouseCallback("Sensor Data", mouse_callback)

# Main loop
try:
    while True:
        with data_lock:
            if received_data:
                addr, data = received_data
                try:
                    data_dict = json.loads(data)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON data: {e}")
                    print(f"Received data:\n{data}")
                    data_dict = {}
                received_data = None
            else:
                data_dict = {}
        
        # Create and show image
        img = create_data_image(data_dict)
        cv2.imshow("Sensor Data", img)
        
        # Check if button was pressed and change color back after 1 second
        if button_pressed and (datetime.now() - button_press_time).total_seconds() > 0.5:
            button_pressed = False
        
        # Check for window close
        if cv2.waitKey(1) & 0xFF == 27:  # 27 = ASCII code for the ESC key
            break

except KeyboardInterrupt:
    print("Main thread exiting...")
finally:
    stop_threads = True
    udp_thread.join()
    cv2.destroyAllWindows()
    sys.exit()
