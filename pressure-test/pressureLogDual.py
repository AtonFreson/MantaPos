import time
import datetime
import Adafruit_ADS1x15  # pip install adafruit-ads1x15
import csv
import os
import statistics

# Initialize the ADS1115 ADCs with their respective I2C addresses
adc1 = Adafruit_ADS1x15.ADS1115(busnum=1, address=0x49)
adc2 = Adafruit_ADS1x15.ADS1115(busnum=1, address=0x48)

# Set the gain
#GAIN = 1

# Data storage lists
data_stream1 = []
data_stream2 = []

# Write interval in seconds (10 minutes)
WRITE_INTERVAL = 600
last_write_time = time.time()
last_minute_time = time.time()

# Directory for CSV files
log_dir = 'pressure-test/pressure_logs'
os.makedirs(log_dir, exist_ok=True)

# File paths for CSV files
file1 = os.path.join(log_dir, 'pressure_top.csv')
file2 = os.path.join(log_dir, 'pressure_bottom.csv')

# Ensure that the CSV file has headers only if it doesn't exist
def initialize_csv_file(file):
    if not os.path.exists(file):  # Check if the file already exists
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            #writer.writerow(['Timestamp', 'Average', 'Standard Deviation', 'Minimum', 'Maximum'])
            writer.writerow(['Timestamp', 'Average', ' ', 'StdRange', 'Minimum', 'Maximum'])
            
# Write data to CSV files and clear the data lists
def write_to_csv():
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

# Initialize both CSV files only if they do not exist
initialize_csv_file(file1)
initialize_csv_file(file2)

# Initialize variables for minute data collection
minute_data1 = []
minute_data2 = []

try:
    while True:
        # Read data from both ADCs
        value1 = adc1.read_adc(0)#, gain=GAIN)
        value2 = adc2.read_adc(0)#, gain=GAIN)
        
        # Get the current timestamp
        timestamp = time.time()
        timestamp_str = datetime.datetime.fromtimestamp(timestamp).isoformat()
        
        # Append data to the minute lists
        minute_data1.append(value1)
        minute_data2.append(value2)
        
        # Check if 60 seconds have passed
        current_time = time.time()
        if current_time - last_minute_time >= 60:
            # Calculate average, standard deviation, min and max for the minute data
            avg1 = statistics.mean(minute_data1)
            stddev1 = statistics.stdev(minute_data1)
            min1 = min(minute_data1)
            max1 = max(minute_data1)
            avg2 = statistics.mean(minute_data2)
            stddev2 = statistics.stdev(minute_data2)
            min2 = min(minute_data2)
            max2 = max(minute_data2)

            # Append statistics to the data streams and print them, formatted for excel
            data_stream1.append([timestamp_str, avg1, avg1-stddev1, stddev1*2, min1, max1-min1])
            #data_stream1.append([timestamp_str, avg1, stddev1, min1, max1])
            data_stream2.append([timestamp_str, avg2, avg2-stddev2, stddev2*2, min2, max2-min2])
            #data_stream2.append([timestamp_str, avg2, stddev2, min2, max2])

            # Print the statistics
            print(f"Timestamp: {timestamp_str}")
            print(f"    Top Sensor: {avg1:.2f} ± {stddev1:.2f} (min: {min1}, max: {max1})")
            print(f"    Bot Sensor: {avg2:.2f} ± {stddev2:.2f} (min: {min2}, max: {max2})")

            # Clear the minute data lists
            minute_data1.clear()
            minute_data2.clear()
            
            # Update the last minute time
            last_minute_time = current_time
        
        # Check if it's time to write to the files
        if current_time - last_write_time >= WRITE_INTERVAL:
            # Write data to CSV files
            print("Writing data to CSV files... ", end="")
            write_to_csv()
            print("Saved.")
            
            # Update the last write time
            last_write_time = current_time
        
        # Small delay to prevent excessive CPU usage
        time.sleep(1)

except KeyboardInterrupt:
    print("\nData logging stopped by user, writing data to CSV files... ", end="")
    write_to_csv()  # Write any remaining data before exiting
    print("Saved.")
