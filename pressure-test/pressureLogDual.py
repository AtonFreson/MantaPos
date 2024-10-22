import time
import datetime
import Adafruit_ADS1x15  # pip install adafruit-ads1x15
import csv
import os

# Initialize the ADS1115 ADCs with their respective I2C addresses
adc1 = Adafruit_ADS1x15.ADS1115(busnum=1, address=0x48)
adc2 = Adafruit_ADS1x15.ADS1115(busnum=1, address=0x49)

# Set the gain
#GAIN = 1

# Data storage lists
data_stream1 = []
data_stream2 = []

# Write interval in seconds (10 minutes)
WRITE_INTERVAL = 600
last_write_time = time.time()

# File paths for CSV files
file1 = 'pressure_top.csv'
file2 = 'pressure_bottom.csv'

# Ensure that the CSV file has headers only if it doesn't exist
def initialize_csv_file(file):
    if not os.path.exists(file):  # Check if the file already exists
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'ADC_Value'])  # Write CSV header

# Initialize both CSV files only if they do not exist
initialize_csv_file(file1)
initialize_csv_file(file2)

try:
    while True:
        # Read data from both ADCs
        value1 = adc1.read_adc(0)#, gain=GAIN)
        value2 = adc2.read_adc(0)#, gain=GAIN)
        
        # Get the current timestamp
        timestamp = time.time()
        timestamp_str = datetime.datetime.fromtimestamp(timestamp).isoformat()
        
        # Append data to the lists (each row will be [timestamp, value])
        data_stream1.append([timestamp_str, value1])
        data_stream2.append([timestamp_str, value2])
        
        # Check if it's time to write to the files
        current_time = time.time()
        if current_time - last_write_time >= WRITE_INTERVAL:
            # Write data to CSV files
            with open(file1, 'a', newline='') as f1:
                writer1 = csv.writer(f1)
                writer1.writerows(data_stream1)  # Write all rows for data_stream1
            with open(file2, 'a', newline='') as f2:
                writer2 = csv.writer(f2)
                writer2.writerows(data_stream2)  # Write all rows for data_stream2

            # Clear the data lists after writing
            data_stream1.clear()
            data_stream2.clear()
            
            # Update the last write time
            last_write_time = current_time
        
        # Small delay to prevent excessive CPU usage
        time.sleep(1)

except KeyboardInterrupt:
    print("Data logging stopped by user.")
