import socket
import threading

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

def listen_for_udp():
    global received_data
    print(f"Listening for incoming UDP packets on port {UDP_PORT}...")
    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
                with data_lock:
                    received_data = (addr, data.decode('utf-8'))
            except socket.timeout:
                pass  # Just pass if no data is received within the timeout period
    except Exception as e:
        print(f"An error occurred in the listener thread: {e}")
    finally:
        sock.close()

# Start the UDP listener in a separate thread
udp_thread = threading.Thread(target=listen_for_udp)
udp_thread.daemon = True  # Ensure the thread exits when the main program exits
udp_thread.start()

# Main thread can perform other tasks
try:
    while True:
        with data_lock:
            if received_data:
                addr, data = received_data
                print(f"Received message from {addr}: {data}")
                received_data = None  # Reset after printing
except KeyboardInterrupt:
    print("Main thread exiting...")
finally:
    udp_thread.join()
