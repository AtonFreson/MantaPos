import socket
import threading
import json
import sys
import cv2
import numpy as np
from datetime import datetime

# UDP socket parameters
UDP_IP = ''           # Empty string means to listen on all available network interfaces
UDP_PORT = 13233      # Must match the udpPort in your Arduino code

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)  # Set a timeout of 1 second

# Shared variables and lock
data_lock = threading.Lock()
received_data = None
stop_threads = False

def create_data_image(data_dict):
    # Create a black image
    img = np.zeros((800, 400, 3), dtype=np.uint8)
    
    # Format strings for display
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

    # Add text to image
    y = 20
    for line in lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        y += 30
    
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

# Start UDP listener thread
udp_thread = threading.Thread(target=udp_listener)
udp_thread.start()

# Create OpenCV window
cv2.namedWindow("Sensor Data", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sensor Data", 400, 800)

# Main loop
try:
    while True:
        with data_lock:
            if received_data:
                addr, data = received_data
                try:
                    data_dict = json.loads(data)
                    # Create and show image
                    img = create_data_image(data_dict)
                    cv2.imshow("Sensor Data", img)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON data: {e}")
                received_data = None
        
        # Check for window close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Main thread exiting...")
finally:
    stop_threads = True
    udp_thread.join()
    cv2.destroyAllWindows()
    sys.exit()
