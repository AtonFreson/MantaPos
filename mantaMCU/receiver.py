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
import time

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

# Initialize shared memory for depth values (as creator) and position values (as creator)
depth_shared = manta.DepthSharedMemory(create=True)
position_shared = manta.PositionSharedMemory(create=True)

# List of unit names (can be edited)
unit_names = ["X-axis Encoder", "Z-axis Encoder - Main", "Z-axis Encoder - Second", "Surface Pressure Sensor", "mantaPos Positioning"]
units_selected = [False] * len(unit_names)
# list of all data dicts, one for each unit
data_dicts = [{} for _ in range(len(unit_names))]
last_received_times = [0] * len(unit_names)
update_frequencies = [None] * len(unit_names)
FREQ_HISTORY_LENGTH = 10
frequency_history = [[] for _ in range(len(unit_names))]
time_diff_history = [[] for _ in range(len(unit_names))]
dropped_packets = [0] * len(unit_names)
last_packet = [None] * len(unit_names)

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

# Average depth values for the pressure system
depth_history = []
depth_offset_history = []
avg_depth = [None, None]
AVG_DEPTH_LENGTH = 30  # Number of depth values to average per list


# Positions for data displays (left side)
DATA_START_X = 10
DATA_START_Y = 20
DATA_SPACING_X = 380
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
IMG_WIDTH = CHECKBOX_START_X + DATA_SPACING_X
IMG_HEIGHT = 990

# Initialize shared data lists for UDP and position data
received_data_list = []
received_data_local_list = []

def update_depth_average(value, history_list):
    if value is not None:
        history_list.append(value)
        if len(history_list) > AVG_DEPTH_LENGTH:
            history_list.pop(0)
        return sum(history_list) / len(history_list)
    return None

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
    
    if "global_pos" in data_dict:
        pos = data_dict["global_pos"]
        lines.append("Global Position Data:")
        lines.append(f"-- Time: {int(pos['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(pos['timestamp'], time_diff, unit_number)
        position = pos["position"]
        rotation = pos["rotation"]

        lines.append(" Pos:   X        Y       Z")
        lines.append(f"     {position[0]: .4f} {position[1]: .4f} {position[2]: .4f}")
        lines.append(" Rot:   X        Y       Z")
        lines.append(f"     {rotation[0]: .3f}  {rotation[1]: .3f}  {rotation[2]: .3f}")
        lines.append("")

    if "camera" in data_dict:
        cam = data_dict["camera"]            
        lines.append("Camera Data:")
        lines.append(f"-- Time: {int(cam['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(cam['timestamp'], time_diff, unit_number)
        if cam['fps'] is not None:
            lines.append(f" Avg. FPS: {cam['fps']:.2f}")
        else:
            lines.append(" Avg. FPS: Err(Camera disabled)")

        if any(key in data_dict for key in ["camera_pos_0", "camera_pos_1", "camera_pos_2", "camera_pos_3"]):
            lines.append(" Pos:    X        Y       Z")
            for i in range(4):
                if f"camera_pos_{i}" in data_dict:
                    position = data_dict[f"camera_pos_{i}"]["position"]
                    lines.append(f"  {i}: {position[0]: .4f} {position[1]: .4f} {position[2]: .4f}")
                else:
                    lines.append(f"  {i}:")
            lines.append(" Rot:    X        Y       Z")
            for i in range(4):
                if f"camera_pos_{i}" in data_dict:
                    rotation = data_dict[f"camera_pos_{i}"]["rotation"]
                    lines.append(f"  {i}: {rotation[0]: .3f}  {rotation[1]: .3f}  {rotation[2]: .3f}")
                else:
                    lines.append(f"  {i}:")
        else:
            for i in range(10): lines.append("")
        lines.append("")
    
    if "global_pos" in data_dict:
        for i in range(7): lines.append("")

    if "encoder" in data_dict:
        enc = data_dict["encoder"]
        lines.append("Encoder Data:")
        lines.append(f"-- Time: {int(enc['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(enc['timestamp'], time_diff, unit_number)
        lines.append(f" Counts:  {enc['counts']}")
        lines.append(f" Speed:  {enc['speed']: .5f} m/s")
        lines.append(f" Dist:    {enc['distance']: .5f} m")
        lines.append("")
    else:
        for i in range(6): lines.append("")
    
    if "temperature" in data_dict:
        temp = data_dict["temperature"]
        lines.append("Temperature Data:")
        lines.append(f"-- Time: {int(temp['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(temp['timestamp'], time_diff, unit_number)
        lines.append(f" Temperature:   {temp['value']: .2f} C")
        lines.append("")
    else:
        for i in range(4): lines.append("")
    
    if "pressure" in data_dict:
        press = data_dict["pressure"]
        lines.append("Pressure Data:")
        lines.append(f"-- Time: {int(press['timestamp'])/1000:.3f} --")
        time_diff = update_time_diff(press['timestamp'], time_diff, unit_number)
        
        for i in range(0+2*unit_number//3, 2+2*unit_number//3):                    
            adc_value = int(press['adc_value' + str(i % 2)])
            depth = press['depth' + ('_offset' if unit_number != 0 else '') + str(i % 2)]
            if depth is None:
                lines.append(f" {pressure_sensors[i]}: {adc_value} (Err(Calc))")
            else:
                lines.append(f" {pressure_sensors[i]}: {adc_value} ({depth: .5f}m)")
        
        running_avg = avg_depth[unit_number//3]
        if running_avg is not None:
            lines.append(f" Average ({AVG_DEPTH_LENGTH} vals): {avg_depth[unit_number//3]: .5f}m")
        else:
            lines.append(f" Average ({AVG_DEPTH_LENGTH} vals): Err(Calc)")
        lines.append("")
    else:
        for i in range(6): lines.append("")

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
    else:
        for i in range(9): lines.append("")
    
    if "ptp" in data_dict:
        press = data_dict["ptp"]
        lines.append("PTP Clock Info:")
        if press['syncing'] == "IN PROGRESS...":
            lines.append(f" Status:  {press['status']} (SYNCING)")
        else:
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
    else:
        for i in range(4): lines.append("")

    if data_dict != {}:
        # Fit data for unit 4
        if unit_number == 4:
            lines = lines[:-14*(2 if "global_pos" in data_dict else 1)]

        if "ptp" not in data_dict:
            lines.append("Clock Info:")
        if update_frequencies[unit_number] is not None:
            lines.append(f" Pckt Freq: {update_frequencies[unit_number]:.2f}Hz ({dropped_packets[unit_number]} lost)")
        else:
            lines.append(" Pckt Freq: Err(N/A)")

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
                lines.append(f" Unit {ref_unit} Diff: {time_diff:>4}ms ({avg_time_diff:>4.0f}ms)")
            else:
                lines.append(f" Unit {ref_unit} Diff: {time_diff:>4}ms")
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

    # Fill right side background with light gray
    cv2.rectangle(img, (DATA_START_X + DATA_SPACING_X * len(unit_names) + SPACER_WIDTH // 2 - 10, 0), (IMG_WIDTH, IMG_HEIGHT), (50, 50, 50), -1)
    
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
    global received_data_list
    while not stop_threads:
        try:
            data, _ = sock.recvfrom(4096)
            data = data.decode()
            recv_time = datetime.now().timestamp()
            if data is not None:
                with data_lock:
                    received_data_list.append((data, recv_time))
                    if len(received_data_list) > 10:
                        print("Warning: UDP data backlog exceeded 10 items, dropping oldest.")
                        received_data_list.pop(0)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error in UDP listener: {e}")
            break

def position_listener():
    global received_data_local_list
    while not stop_threads:
        try:
            data = position_shared.get_position()
            if data is not None:
                recv_time = datetime.now().timestamp()
                with data_lock:
                    received_data_local_list.append((data, recv_time))
                    if len(received_data_local_list) > 10:
                        print("Warning: Position data backlog exceeded 10 items, dropping oldest.")
                        received_data_local_list.pop(0)
            time.sleep(0.01) # Short sleep to prevent high CPU usage
        except Exception as e:
            print(f"Error in position listener: {e}")
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
            for unit_number in selected_units:
                dropped_packets[unit_number] = 0
                last_packet[unit_number] = None
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

# Start position memory listener thread
position_thread = threading.Thread(target=position_listener)
position_thread.start()

# Create OpenCV window
cv2.namedWindow("Sensor Data", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sensor Data", IMG_WIDTH, IMG_HEIGHT)
cv2.startWindowThread()  # Add this so window events run in a dedicated thread

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
print("")


def process_incoming_data():
    global latest_data_dict, last_recording_save_time, recording, recording_data_buffer
    while not stop_threads:
        new_datas = []
        with data_lock:
            while received_data_list:
                new_datas.append(received_data_list.pop(0))
            while received_data_local_list:
                new_datas.append(received_data_local_list.pop(0))
        
        for new_item in new_datas:              
            try:
                data_str, recv_time = new_item
                data_dict = json.loads(data_str)
                unit_num_str = data_dict.get("mpu_unit")

                if unit_num_str is None:
                    print("Warning: 'mpu_unit' key missing in data")
                else:
                    unit_num = int(unit_num_str)
    
                    if 0 <= unit_num < len(data_dicts):
                        if recording and (unit_num in recording_units): 
                            rec_time_str = datetime.fromtimestamp(recv_time).isoformat()
    
                        current_time = recv_time
                        if last_received_times[unit_num] is not None:
                            inst_freq = 1.0 / (current_time - last_received_times[unit_num])
                            frequency_history[unit_num].append(inst_freq)
                            if len(frequency_history[unit_num]) > FREQ_HISTORY_LENGTH:
                                frequency_history[unit_num].pop(0)
                            update_frequencies[unit_num] = sum(frequency_history[unit_num]) / len(frequency_history[unit_num])
                        else:
                            update_frequencies[unit_num] = None
                        last_received_times[unit_num] = current_time
                        data_dict["mpu_unit"] = unit_num
    
                        current_packet = data_dict["packet_number"]
                        if last_packet[unit_num] is not None:
                            dropped_packets[unit_num] += current_packet - last_packet[unit_num] - 1
                        last_packet[unit_num] = current_packet
    
                        if unit_num in [0, 3]:
                            sensor_name = "bottom" if unit_num == 0 else "surface"
                            for i in range(2):
                                pressure_system[i].set_sensor_value(sensor_name, data_dict.get("pressure", {}).get("adc_value" + str(i), 0))
    
                                if unit_num == 0:
                                    depth = pressure_system[i].get_depth()
                                    data_dict["pressure"]["depth" + str(i)] = depth
                                    avg_depth[0] = update_depth_average(depth, depth_history)
                                else:
                                    offset = pressure_system[i].sensor_values[sensor_name]
                                    data_dict["pressure"]["depth_offset" + str(i)] = offset
                                    avg_depth[1] = update_depth_average(offset, depth_offset_history)
    
                        if unit_num == 0:
                            new_data_dict = {k: v for k, v in data_dict.items() if k not in ('encoder', 'imu')}
                            if 'encoder' in data_dict:
                                new_data_dict['encoder'] = data_dict['encoder']
                            if 'imu' in data_dict:
                                new_data_dict['imu'] = data_dict['imu']
                            data_dict = new_data_dict
    
                        data_dicts[unit_num] = data_dict
                        
                        if recording and (unit_num in recording_units):
                            data_dict = {"mpu_unit": unit_num, "recv_time": rec_time_str, **data_dict}
                            recording_data_buffer.append(data_dict)
                    else:
                        print(f"Warning: Invalid unit number received: {unit_num}")
            except Exception as e:
                print(f"\nException occurred during data processing: {e}")
                print(f"Received data:\n{data_str}")
        
        for unit_num in range(len(unit_names)):
            if datetime.now().timestamp() - last_received_times[unit_num] > 5.0:
                data_dicts[unit_num] = {}
                time_diff_history[unit_num] = []
                dropped_packets[unit_num] = 0
                last_packet[unit_num] = None
                update_frequencies[unit_num] = None
    
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
    
        depth_main = None
        depth_sec = None
        frame_pos = None
        timestamps = []

        if data_dicts[0].get("encoder"):
            frame_pos = data_dicts[0]["encoder"].get("distance")
            timestamps.append(int(data_dicts[0]["encoder"].get("timestamp", 0)))
        if data_dicts[1].get("encoder"):
            depth_main = data_dicts[1]["encoder"].get("distance")
            timestamps.append(int(data_dicts[1]["encoder"].get("timestamp", 0)))
        if data_dicts[2].get("encoder"):
            depth_sec = data_dicts[2]["encoder"].get("distance")
            timestamps.append(int(data_dicts[2]["encoder"].get("timestamp", 0)))
        
        # Find the oldest timestamp (smallest value)
        timestamp_oldest = min(timestamps) if len(timestamps) == 3 else None
        
        depth_shared.write_depths(depth_main, depth_sec, frame_pos, timestamp_oldest)
    
        with data_lock:
            latest_data_dict = data_dicts.copy()
        
        time.sleep(0.001)

# Start the data processing thread
data_processing_thread = threading.Thread(target=process_incoming_data, daemon=True)
data_processing_thread.start()

# Create a global variable to hold the latest data_dict and a lock for thread-safety
latest_data_dict = None
data_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
def process_data_image():
    global latest_data_dict, data_image
    while not stop_threads:
        # get a snapshot of latest_data_dict if available.
        with data_lock:
            data = latest_data_dict
            latest_data_dict = None  # reset after getting the latest data
        if data is not None:
            data_image = create_data_image(data)  # assuming create_data_image accepts one argument
        # sleep briefly so as to not hog the CPU
        time.sleep(0.001)

# Start the background image processing thread
process_data_image_thread = threading.Thread(target=process_data_image, daemon=True)
process_data_image_thread.start()


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
        
        
        # Create and show image
        #loop_start = datetime.now()
        #img = process_data_image(latest_data_dict)
        #print(f"Main loop iteration took {((datetime.now() - loop_start).total_seconds())*1000:.3f} milliseconds")
        cv2.imshow("Sensor Data", data_image)

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

    print("Stopping threads...")
    stop_threads = True
    udp_thread.join()
    position_thread.join()
    data_processing_thread.join()
    process_data_image_thread.join()

    print("Cleaning up shared memory...")
    position_shared.close()
    depth_shared.close()
    depth_shared.unlink()  # Only the creator should unlink
    position_shared.unlink() # Only the creator should unlink

    print("Exiting...")
    cv2.destroyAllWindows()
    sys.exit()