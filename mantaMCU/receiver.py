import socket
import threading
import json
import sys
import cv2
import numpy as np
from datetime import datetime
import os
import traceback
import ctypes

# Add parent directory to the path to import mantaPosLib
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
import mantaPosLib as manta

# Prevent Windows from entering sleep mode
if os.name == 'nt':
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    print("Windows execution state set to keep screen active.\n")

# UDP socket parameters
UDP_IP = ''             # Listen on all available network interfaces
UDP_PORT = 13233        # Must match the udpPort in your Arduino code
COMMAND_PORT = 13234    # Port for sending commands

# Create a UDP socket for receiving data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.5)

# Create a UDP socket for sending commands
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# List of unit names (can be edited)
unit_names = ["X-axis Encoder", "Z-axis Encoder - Main", "Z-axis Encoder - Second", "Surface Pressure Sensor", "mantaPos Positioning"]
units_selected = [False] * len(unit_names)
# list of all data dicts, one for each unit
data_dicts = [{} for _ in range(len(unit_names))]
last_received_times = [0] * len(unit_names)
time_diff_history = [[] for _ in range(len(unit_names))]

# Data folders
recording_folder = "recordings"
pressure_calib_folder = "calibrations/pressure_calibrations/"
pressure_sensors = ["pressure1", "pressure2", "pressure3", "pressure4"]

# Shared variables and lock
data_lock = threading.Lock()
received_data = None
received_data_local = None
stop_threads = False
button_pressed = False
button_press_time = None
wait_for_zero_position_selected = True
ref_unit_time = None
ref_unit = 0
print_range = 60 * 60 * 1000  # Maximum range for printing RTC difference, 60 minutes
recording_timestamp = ""
recording_last_filename = ""

# Recording-related globals
recording = False
default_filename = "data"
save_filepath = os.path.join(recording_folder, f"{default_filename}.json")
input_box_focused = False
input_box_text = ""
recording_start_time = None
last_recording_save_time = None
recording_units = []
recording_data_buffer = []
recording_interval = 10.0  # append every 10 seconds

# Positions for data displays (left side)
DATA_START_X = 10
DATA_START_Y = 20
DATA_SPACING_X = 370
DATA_ROW_Y = 30
SPACER_WIDTH = 5 # Width of the white vertical spacer lines

# Position for unit list and user input options (right side)
TIME_START_X = 20

CHECKBOX_START_X = DATA_START_X + DATA_SPACING_X * len(unit_names) + 5
CHECKBOX_START_Y = TIME_START_X + 30
CHECKBOX_SIZE = 20
CHECKBOX_GAP = 30

# Calculate button coordinates based on unit list
BUTTON_X1 = CHECKBOX_START_X
BUTTON_Y1 = CHECKBOX_START_Y + len(unit_names) * CHECKBOX_GAP + 20
BUTTON_X2 = BUTTON_X1 + 300
BUTTON_Y2 = BUTTON_Y1 + 40

# Adjust coordinates for the "Sync All Clocks" button
SYNC_BUTTON_Y1 = BUTTON_Y2 + 20
SYNC_BUTTON_Y2 = SYNC_BUTTON_Y1 + 40

# Define new button coordinates below "Sync All Clocks" button
CALIBRATE_BUTTON_Y1 = SYNC_BUTTON_Y2 + 20
CALIBRATE_BUTTON_Y2 = CALIBRATE_BUTTON_Y1 + 40

# Adjust coordinates for the "Encoders Section"
ZERO_ENCODER_TITLE_Y = CALIBRATE_BUTTON_Y2 + 65

WAIT_CHECKBOX_X = BUTTON_X1
WAIT_CHECKBOX_Y = ZERO_ENCODER_TITLE_Y + 20
WAIT_CHECKBOX_SIZE = 20

ZERO_BUTTON_Y1 = WAIT_CHECKBOX_Y + WAIT_CHECKBOX_SIZE + 15
ZERO_BUTTON_Y2 = ZERO_BUTTON_Y1 + 40

# "Data Recording Section" coordinates
DATA_RECORDING_TITLE_Y = ZERO_BUTTON_Y2 + 60
FILENAME_BOX_X1 = BUTTON_X1
FILENAME_BOX_Y1 = DATA_RECORDING_TITLE_Y + 20
FILENAME_BOX_X2 = FILENAME_BOX_X1 + 355
FILENAME_BOX_Y2 = FILENAME_BOX_Y1 + 40

RECORD_BUTTON_Y1 = FILENAME_BOX_Y2 + 20
RECORD_BUTTON_Y2 = RECORD_BUTTON_Y1 + 40

INFO_BOX_X1 = BUTTON_X1
INFO_BOX_Y1 = RECORD_BUTTON_Y2 + 20
INFO_BOX_X2 = FILENAME_BOX_X1 + 355
INFO_BOX_Y2 = INFO_BOX_Y1 + 70 + 27*len(unit_names)

# Constants for image dimensions
IMG_WIDTH = CHECKBOX_START_X + 370
IMG_HEIGHT = 990


def update_time_diff(timestamp, time_diff, current_unit):
    global ref_unit_time, ref_unit
    if ref_unit_time == None:
        ref_unit_time = int(timestamp)
        ref_unit = current_unit
    elif time_diff == None and ref_unit != current_unit:
        recording_time_diff = int((last_received_times[current_unit] - last_received_times[ref_unit])*1000)
        time_diff = ref_unit_time - int(timestamp) + recording_time_diff
    return time_diff

def create_unit_lines(data_dict, unit_number):
    global ref_unit
    lines = []
    time_diff = None
    
    # Add unit name at the top
    lines.append(unit_names[unit_number].center(30))

    if "encoder" in data_dict:
        enc = data_dict["encoder"]
        lines.append("Encoder Data:")
        lines.append(f"-- Time: {int(enc['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(enc['timestamp'], time_diff, unit_number)
        lines.append(f" Rev:     {enc['revolutions']: .5f}")
        lines.append(f" RPM:    {enc['rpm']: .5f} rpm")
        lines.append(f" Speed:  {enc['speed']: .5f} m/s")
        lines.append(f" Dist:    {enc['distance']: .5f} m")
        lines.append("")
    elif unit_number != 4:
        for i in range(7): lines.append("")
    
    if "temperature" in data_dict:
        temp = data_dict["temperature"]
        lines.append("Temperature Data:")
        lines.append(f"-- Time: {int(temp['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(temp['timestamp'], time_diff, unit_number)
        lines.append(f" Temperature:   {temp['value']: .2f} C")
        lines.append("")
    elif unit_number != 4:
        for i in range(4): lines.append("")
    
    if "pressure" in data_dict:
        press = data_dict["pressure"]
        lines.append("Pressure Data:")
        lines.append(f"-- Time: {int(press['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(press['timestamp'], time_diff, unit_number)
        
        for i in range(0+2*unit_number//3, 2+2*unit_number//3):                    
            adc_value = int(press['adc_value' + str(i % 2)])
            if unit_number == 0:
                depth = press['depth' + str(i)]
                if depth is None:
                    lines.append(f" {pressure_sensors[i]}: {adc_value} - Err(Calc)")
                else:
                    lines.append(f" {pressure_sensors[i]}: {adc_value} - {depth: .5f}m")
            else:
                lines.append(f" {pressure_sensors[i]}: {adc_value}")
        lines.append("")
    elif unit_number != 4:
        for i in range(5): lines.append("")

    if "imu" in data_dict:
        imu = data_dict["imu"]
        if "info" in imu:
            lines.append("IMU Info:")
            lines.append(f" {imu['info']}")
            lines.append("")
            lines.append(f" {imu['info_1']} {imu['info_2']}")
            lines.append(f" {imu['info_3']} {imu['info_4']}")
            last_received_times[unit_number] = datetime.now().timestamp() + 600  # 10 minutes in the future
            for i in range(3): lines.append("")
        else:
            lines.append("IMU Data:")
            lines.append(f"-- Time: {int(imu['timestamp'])/1000:.3f} --")
            time_diff = update_time_diff(imu['timestamp'], time_diff, unit_number)
            acc = imu["acceleration"]
            lines.append(f" Accel X:    {acc['x']: .5f} m/s^2")
            lines.append(f" Accel Y:    {acc['y']: .5f} m/s^2")
            lines.append(f" Accel Z:    {acc['z']: .5f} m/s^2")
            gyro = imu["gyroscope"]
            lines.append(f" Gyro X:     {gyro['x']: .5f} rad/s")
            lines.append(f" Gyro Y:     {gyro['y']: .5f} rad/s")
            lines.append(f" Gyro Z:     {gyro['z']: .5f} rad/s")
            
        lines.append("")
    elif unit_number != 4:
        for i in range(9): lines.append("")
    
    if "ptp" in data_dict:
        press = data_dict["ptp"]
        lines.append("PTP Clock Info:")
        lines.append(f" Syncing: {press['syncing']}")
        lines.append(f" Status:  {press['status']}")
        lines.append(f" {press['details']}")
        
        time_since_sync = int(press['time_since_sync'])
        if time_since_sync < 0:
            lines.append(" Time Since Sync: Err(Neg. Vl.)")
        else:
            hours = time_since_sync // 3600
            minutes = (time_since_sync % 3600) // 60
            seconds = time_since_sync % 60
            lines.append(f" Time Since Sync: {hours:02d}:{minutes:02d}:{seconds:02d}")

        if abs(int(press['difference'])) < print_range:
            lines.append(f" RTC Diff:   {press['difference']}ms")
        else:
            lines.append(" RTC Diff:   Err(Out of Range)")
    elif unit_number != 4:
        for i in range(6): lines.append("")

    if "camera" in data_dict:
        cam = data_dict["camera"]
        lines.append("Camera Data:")
        lines.append(f"-- Time: {int(cam['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(cam['timestamp'], time_diff, unit_number)
        position = cam["position"]
        position_std = cam["position_std"]
        rotation = cam["rotation"]
        rotation_std = cam["rotation_std"]
        
        lines.append(" Position:")
        lines.append(f"  X={position[0]: >+6.3f}m ±{position_std[0]: >6.3f}")
        lines.append(f"  Y={position[1]: >+6.3f}m ±{position_std[1]: >6.3f}")
        lines.append(f"  Z={position[2]: >+6.3f}m ±{position_std[2]: >6.3f}")
        lines.append(" Rotation:")
        lines.append(f"  Roll={rotation[0]: >+6.3f}° ±{rotation_std[0]: >6.3f}")
        lines.append(f"  Pitch={rotation[1]: >+6.3f}° ±{rotation_std[1]: >6.3f}")
        lines.append(f"  Yaw={rotation[2]: >+6.3f}° ±{rotation_std[2]: >6.3f}")

    if data_dict != {}:
        if time_diff is not None:
            # Store time_diff in the history for this unit
            time_diff_history[unit_number].append(time_diff)

            # Keep only the last N time_diffs (e.g., N=10)
            if len(time_diff_history[unit_number]) > 200:
                time_diff_history[unit_number].pop(0)

            # Compute the moving average
            avg_time_diff = sum(time_diff_history[unit_number]) / len(time_diff_history[unit_number])
        else:
            avg_time_diff = None

        if time_diff is not None:
            if abs(time_diff) > print_range:
                lines.append(f" Unit {ref_unit} Diff: Err(Out of Range)")
            elif avg_time_diff is not None:
                lines.append(f" Unit {ref_unit} Diff: {time_diff}ms ({avg_time_diff:.0f}ms)")
            else:
                lines.append(f" Unit {ref_unit} Diff: {time_diff}ms")
        else:
            lines.append(f" Unit {ref_unit} Diff: N/A")
    
    return lines

def multiline_putText(img, text, position, font, font_scale, color, thickness, line_spacing=25):
    x, y = position
    for line in text.split('\n'):
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
        y += line_spacing

def create_data_image(data_dicts):
    global ref_unit_time, input_box_text, recording_filename, recording, recording_start_time, recording_units

    # Create a black image
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    
    # Left side data displays from units
    ref_unit_time = None
    for i in range(len(unit_names)):
        lines = create_unit_lines(data_dicts[i], i)

        # Add data text for unit
        y = DATA_START_Y
        for line in lines:
            cv2.putText(img, line, (DATA_START_X + DATA_SPACING_X*i, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            y += DATA_ROW_Y
        
        # Draw vertical spacer line
        spacer_x = DATA_START_X + DATA_SPACING_X * (i + 1) - SPACER_WIDTH // 2 - 10
        cv2.line(img, (spacer_x, 0), (spacer_x, IMG_HEIGHT), (255, 255, 255), SPACER_WIDTH)
    

    ### Right side unit list and user input options ###
    # Display "Desktop Time" above the unit boxes
    cv2.putText(img, f"Desktop Time: {datetime.now().timestamp():.3f}", (CHECKBOX_START_X, TIME_START_X),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Unit selection checkboxes
    for idx, unit_name in enumerate(unit_names):
        # Draw checkbox
        top_left = (CHECKBOX_START_X, CHECKBOX_START_Y + idx * CHECKBOX_GAP)
        bottom_right = (top_left[0] + CHECKBOX_SIZE, top_left[1] + CHECKBOX_SIZE)
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)
        
        # If selected, draw a tick
        if units_selected[idx]:
            cv2.line(img, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0, 255, 0), 2)
            cv2.line(img, (top_left[0], bottom_right[1]), (bottom_right[0], top_left[1]), (0, 255, 0), 2)
        
        # Draw unit name with index
        text_position = (bottom_right[0] + 10, top_left[1] + CHECKBOX_SIZE - 5)
        cv2.putText(img, f"{idx}: {unit_name}", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    
    # Draw "Reboot Selected" button
    button_color = (70, 70, 70) if not button_pressed else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), button_color, -1)
    cv2.putText(img, "Reboot Selected", (BUTTON_X1 + 20, BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Draw "Sync All Clocks" button
    sync_button_color = (70, 70, 70) if not button_pressed else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, SYNC_BUTTON_Y1), (BUTTON_X2, SYNC_BUTTON_Y2), sync_button_color, -1)
    cv2.putText(img, "Sync All Clocks", (BUTTON_X1 + 20, SYNC_BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Draw "Calibrate IMU" button
    calibrate_button_color = (70, 70, 70) if not button_pressed else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, CALIBRATE_BUTTON_Y1), (BUTTON_X2, CALIBRATE_BUTTON_Y2), calibrate_button_color, -1)
    cv2.putText(img, "Calibrate IMU", (BUTTON_X1 + 20, CALIBRATE_BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Draw "Encoders" title
    cv2.putText(img, "Encoders", (BUTTON_X1, ZERO_ENCODER_TITLE_Y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.line(img, (BUTTON_X1, ZERO_ENCODER_TITLE_Y + 3), (BUTTON_X1 + 142, ZERO_ENCODER_TITLE_Y + 3), (255, 255, 255), 1)

    # Draw "Wait for Zero Position Signal" checkbox
    checkbox_top_left = (WAIT_CHECKBOX_X, WAIT_CHECKBOX_Y)
    checkbox_bottom_right = (checkbox_top_left[0] + WAIT_CHECKBOX_SIZE, checkbox_top_left[1] + WAIT_CHECKBOX_SIZE)
    cv2.rectangle(img, checkbox_top_left, checkbox_bottom_right, (255, 255, 255), 2)
    
    if wait_for_zero_position_selected:
        cv2.line(img, (checkbox_top_left[0], checkbox_top_left[1]), (checkbox_bottom_right[0], checkbox_bottom_right[1]), (0, 255, 0), 2)
        cv2.line(img, (checkbox_top_left[0], checkbox_bottom_right[1]), (checkbox_bottom_right[0], checkbox_top_left[1]), (0, 255, 0), 2)
    
    # Draw the label next to the checkbox
    text_position = (checkbox_bottom_right[0] + 10, checkbox_top_left[1] + WAIT_CHECKBOX_SIZE - 5)
    cv2.putText(img, "Wait for Index Position Signal", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    # Draw "Zero Selected" button
    zero_button_color = (70, 70, 70) if not button_pressed else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, ZERO_BUTTON_Y1), (BUTTON_X2, ZERO_BUTTON_Y2), zero_button_color, -1)
    cv2.putText(img, "Zero Selected", (BUTTON_X1 + 20, ZERO_BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    
    # Data Recording Section
    cv2.putText(img, "Data Recording", (BUTTON_X1, DATA_RECORDING_TITLE_Y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.line(img, (BUTTON_X1, DATA_RECORDING_TITLE_Y + 3), (BUTTON_X1 + 240, DATA_RECORDING_TITLE_Y + 3), (255, 255, 255), 1)
    
    # Filename input box
    filename_box_color = (0, 255, 0) if input_box_focused else (255, 255, 255)
    cv2.rectangle(img, (FILENAME_BOX_X1, FILENAME_BOX_Y1), (FILENAME_BOX_X2, FILENAME_BOX_Y2), filename_box_color, 2)
    display_text = input_box_text if input_box_text else "recording name..."
    text_color = (255, 255, 255) if input_box_text else (80, 80, 80)  # White if text exists, gray otherwise
    cv2.putText(img, display_text, (FILENAME_BOX_X1 + 10, FILENAME_BOX_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, text_color, 2)
    
    # Start/Stop Recording button
    record_button_color = (70, 70, 70) if not recording else (0, 0, 255)
    cv2.rectangle(img, (BUTTON_X1, RECORD_BUTTON_Y1), (BUTTON_X2, RECORD_BUTTON_Y2), record_button_color, -1)
    record_button_text = "Stop Recording..." if recording else "Record Selected"
    cv2.putText(img, record_button_text, (BUTTON_X1 + 20, RECORD_BUTTON_Y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # Info box if recording
    cv2.rectangle(img, (INFO_BOX_X1, INFO_BOX_Y1), (INFO_BOX_X2, INFO_BOX_Y2), (255, 255, 255), 2)
    if recording and recording_start_time:
        elapsed = datetime.now() - recording_start_time
        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        seconds = (elapsed.seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        cv2.putText(img, f"Recording Time: {time_str}", (INFO_BOX_X1 + 10, INFO_BOX_Y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        units_str = "\n".join([unit_names[u] for u in recording_units])
        multiline_putText(img, f"Units:\n{units_str}", (INFO_BOX_X1 + 10, INFO_BOX_Y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    else:
        cv2.putText(img, f"Not Recording", (INFO_BOX_X1 + 10, INFO_BOX_Y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        multiline_putText(img, recording_last_filename, (INFO_BOX_X1 + 10, INFO_BOX_Y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return img

def udp_listener():
    global received_data
    while not stop_threads:
        try:
            data, addr = sock.recvfrom(4096)
            with data_lock:
                received_data = (addr, data.decode())
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error in UDP listener: {e}")
            break

def mouse_callback(event, x, y, flags, param):
    global button_pressed, button_press_time, units_selected, wait_for_zero_position_selected
    global input_box_focused, recording, recording_filename, input_box_text, recording_units
    global recording_start_time, last_recording_save_time, recording_data_buffer, recording_timestamp, recording_last_filename, save_filepath

    if event == cv2.EVENT_LBUTTONUP:
        # If clicked inside filename box
        if (FILENAME_BOX_X1 <= x <= FILENAME_BOX_X2 and FILENAME_BOX_Y1 <= y <= FILENAME_BOX_Y2):
            if not input_box_text:
                input_box_text = ""
            input_box_focused = True
            return

        # If clicked outside filename box, unfocus if currently focused
        if not (FILENAME_BOX_X1 <= x <= FILENAME_BOX_X2 and FILENAME_BOX_Y1 <= y <= FILENAME_BOX_Y2):
            input_box_focused = False

        # Check if click is inside any checkbox
        for idx in range(len(unit_names)):
            top_left = (CHECKBOX_START_X, CHECKBOX_START_Y + idx * CHECKBOX_GAP)
            bottom_right = (top_left[0] + CHECKBOX_SIZE, top_left[1] + CHECKBOX_SIZE)
            if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                units_selected[idx] = not units_selected[idx]
                print(f"Unit {idx} {'selected' if units_selected[idx] else 'deselected'}: {unit_names[idx]}")
                return  # Exit after handling checkbox click
        
        # Check if click is inside "Wait for Index Position Signal" checkbox
        if (WAIT_CHECKBOX_X <= x <= WAIT_CHECKBOX_X + WAIT_CHECKBOX_SIZE and
            WAIT_CHECKBOX_Y <= y <= WAIT_CHECKBOX_Y + WAIT_CHECKBOX_SIZE):

            wait_for_zero_position_selected = not wait_for_zero_position_selected
            print(f"Wait for Index Position Signal {'selected' if wait_for_zero_position_selected else 'deselected'}")
        
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
        # Check if click is inside "Sync All Clocks" button
        elif BUTTON_X1 <= x <= BUTTON_X2 and SYNC_BUTTON_Y1 <= y <= SYNC_BUTTON_Y2:
            selected_units = list(range(len(unit_names))) # Automatically select all units
            command_dict = {"units": selected_units, "command": "sync"}
        # Check if click is inside "Calibrate IMU" button
        elif BUTTON_X1 <= x <= BUTTON_X2 and CALIBRATE_BUTTON_Y1 <= y <= CALIBRATE_BUTTON_Y2:
            command_dict = {"units": selected_units, "command": "calibrate imu"}

        # Check if click is inside "Start/Stop Recording" button
        elif BUTTON_X1 <= x <= BUTTON_X2 and RECORD_BUTTON_Y1 <= y <= RECORD_BUTTON_Y2:
            # If currently recording, stop
            if recording:
                # finalize recording and flush buffer
                if recording_data_buffer:
                    try:
                        with open(save_filepath, 'a') as f:
                            for entry in recording_data_buffer:
                                f.write(json.dumps(entry) + "\n")
                        recording_data_buffer = []
                    except Exception as e:
                        print("Error writing final data to file:", e)
                recording = False
                print("Stopped recording.")
            else:
                # start recording if units selected
                if selected_units:
                    recording = True
                    now = datetime.now()
                    recording_timestamp = now.strftime('%m-%d@%H-%M')
                    recording_units = selected_units.copy()
                    recording_start_time = datetime.now()
                    last_recording_save_time = datetime.now()
                    recording_data_buffer = []
                    recording_filename = input_box_text if input_box_text else default_filename
                    save_filepath = os.path.join(recording_folder, f"{recording_filename} - {recording_timestamp}.json")
                    print(f"Started recording to '{recording_filename} - {recording_timestamp}.json', units: {recording_units}")
                    
                    # Check if there is a file with the given name, if not create it
                    try:
                        with open(save_filepath, 'x') as f:
                            pass
                    except FileExistsError:
                        pass

                    # Split the temp_text into multiple lines if it exceeds the char_turnover limit
                    temp_text = f"'{recording_filename} - {recording_timestamp}.json'"
                    char_turnover = 25 # Number of characters before splitting text
                    loops = 0
                    while len(temp_text[char_turnover*loops:]) > char_turnover:
                        loops += 1
                        temp_text = temp_text[:char_turnover*loops] + "\n" + temp_text[char_turnover*loops:]
                    recording_last_filename = "Saved recording to:\n" + temp_text
                else:
                    print("No units selected, cannot start recording.")

        # Send command if button was pressed and units are selected
        if command_dict and selected_units != []:
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

# Start pressure sensor data translation system
pressure_system = [manta.PressureSensorSystem(), manta.PressureSensorSystem()]
for i in range(0,4):
    pressure_calib = pressure_calib_folder + pressure_sensors[i] + "_calibration.pkl"
    ret = pressure_system[i // 2].load_calibration("surface" if i % 2 else "bottom", pressure_calib)
    if not ret:
        print(f"Error loading calibration for sensor {pressure_sensors[i]}")
    else:
        print(f"Loaded calibration for sensor {pressure_sensors[i]} as {'surface' if i % 2 else 'bottom'}{i//2}")

# Initialize shared memory for depth values (as creator)
depth_shared = manta.DepthSharedMemory(create=True)

# Main loop
try:
    while True:
        # Handle keyboard input for filename if focused
        key = cv2.waitKey(1) & 0xFF
        if input_box_focused:
            if key == 13:  # ENTER key
                input_box_focused = False
            elif key == 8:  # BACKSPACE
                input_box_text = input_box_text[:-1]
            elif key != 255 and key not in [27, 9, 10, 13, 8]:
                input_box_text += chr(key)

        new_datas = [None, None]
        with data_lock:
            if received_data:
                addr, new_datas[0] = received_data
                received_data = None
            if received_data_local:
                addr, new_datas[1] = received_data_local
                received_data_local = None
        
        for new_data in new_datas:
            if new_data is None:
                continue                
            try:
                data_dict = json.loads(new_data)
                unit_num_str = data_dict.get("mpu_unit")
                if unit_num_str is None:
                    print("Warning: 'mpu_unit' key missing in data")
                else:
                    unit_num = int(unit_num_str)
                    
                    # Check if unit number is valid
                    if 0 <= unit_num < len(data_dicts):
                        last_received_times[unit_num] = datetime.now().timestamp()
                        data_dict["mpu_unit"] = unit_num

                        if unit_num in [0, 3]:
                            sensor_name = "bottom" if unit_num == 0 else "surface"
                            for i in range(2):
                                pressure_system[i].set_sensor_value(sensor_name, data_dict.get("pressure", {}).get("adc_value" + str(i), 0))

                                # Add depth fields to bottom sensor data_dict, one for each system
                                if unit_num == 0:
                                    data_dict["pressure"]["depth"+str(i)] = pressure_system[i].get_depth()

                        data_dicts[unit_num] = data_dict
                        
                        # If recording and this unit is being recorded, append data directly to buffer
                        if recording and (unit_num in recording_units):
                            # Add a timestamp field for local recording time
                            data_dict = {"mpu_unit": unit_num, "recv_time": datetime.now().isoformat(), **data_dict}
                            recording_data_buffer.append(data_dict)
                    else:
                        print(f"Warning: Invalid unit number received: {unit_num}")
            except Exception as e:
                print(f"\nException occurred during data processing: {e}")
                print(f"Received data:\n{new_data}")

        # Clear data for units that haven't sent data in the last few seconds
        for unit_num in range(len(unit_names)):
            if datetime.now().timestamp() - last_received_times[unit_num] > 5.0:
                data_dicts[unit_num] = {}
                time_diff_history[unit_num] = []

        # Every 10 seconds, if recording, flush buffer to file
        if recording and (datetime.now() - last_recording_save_time).total_seconds() >= recording_interval:
            if recording_data_buffer:
                try:
                    with open(save_filepath, 'a') as f:
                        for entry in recording_data_buffer:
                            f.write(json.dumps(entry) + "\n")
                    recording_data_buffer = []
                except Exception as e:
                    print("Error writing to file:", e)
            
            last_recording_save_time = datetime.now()

        # Get depth values from data_dicts if they exist
        depth_main = None
        depth_sec = None
        
        if data_dicts[1].get("encoder"):  # Unit 1: Main encoder
            depth_main = data_dicts[1]["encoder"].get("distance")
        if data_dicts[2].get("encoder"):  # Unit 2: Secondary encoder
            depth_sec = data_dicts[2]["encoder"].get("distance")
            
        # Write depth values to shared memory
        depth_shared.write_depths(depth_main, depth_sec)

        # Create and show image
        img = create_data_image(data_dicts)
        cv2.imshow("Sensor Data", img)
        
        # Check if button was pressed and reset color after 0.5 second
        if button_pressed and (datetime.now() - button_press_time).total_seconds() > 0.5:
            button_pressed = False
        
        # Check for window close (ESC)
        if key == 27:
            break

except KeyboardInterrupt:
    print("User exiting...")

except Exception as e:
    traceback.print_exc()

finally:
    if recording_data_buffer:
        print("Flushing remaining recording data to file...")
        try:
            with open(save_filepath, 'a') as f:
                for entry in recording_data_buffer:
                    f.write(json.dumps(entry) + "\n")
            recording_data_buffer = []
        except Exception as e:
            print("Error writing to file:", e)

    print("Cleaning up shared memory...")
    depth_shared.close()
    depth_shared.unlink()  # Only the creator should unlink

    print("Exiting...")
    stop_threads = True
    udp_thread.join()
    cv2.destroyAllWindows()
    sys.exit()