import socket
import threading
import json
import sys

# UDP socket parameters
UDP_IP = ''           # Empty string means to listen on all available network interfaces
UDP_PORT = 13233      # Must match the udpPort in your Arduino code

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)  # Set a timeout of 1 second

# Shared variable and lock
data_lock = threading.Lock()
received_data = None

# Flag to signal threads to stop
stop_threads = False

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

# Start the UDP listener thread
udp_thread = threading.Thread(target=udp_listener)
udp_thread.start()

# Main thread can perform other tasks
try:
    while True:
        with data_lock:
            if received_data:
                addr, data = received_data
                try:
                    # Parse the JSON data
                    data_dict = json.loads(data)
                    print(f"\nReceived data from {addr}:")
                    print(json.dumps(data_dict, indent=4))  # Pretty-print the dictionary
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON data from {addr}: {e}")
                    print(f"Raw data: {data}")
                received_data = None  # Reset after processing
except KeyboardInterrupt:
    print("Main thread exiting...")
    stop_threads = True
finally:
    udp_thread.join()
    print("UDP listener thread has been terminated.")
    sys.exit()
