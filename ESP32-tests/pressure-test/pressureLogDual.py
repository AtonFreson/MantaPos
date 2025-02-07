import time
import datetime
import Adafruit_ADS1x15  # pip install adafruit-ads1x15
import csv
import os
import statistics

# Initialize the ADS1115 ADCs with their respective I2C addresses
adc1 = Adafruit_ADS1x15.ADS1115(busnum=1, address=0x48)
adc2 = Adafruit_ADS1x15.ADS1115(busnum=1, address=0x49)

# Set the gain for the ADCs (1 for ±4.096V, 2 for ±2.048V, 4 for ±1.024V, 8 for ±0.512V, 16 for ±0.256V)
#GAIN = 1

# Data storage lists
data_stream1 = []
data_stream2 = []
data_stream_raw = []

# Write interval in seconds (10 minutes)
WRITE_INTERVAL = 600

# Data interval in milliseconds (e.g., 100 ms)
DATA_INTERVAL = 100  # in milliseconds

# Initialize the last time intervals
last_write_time = time.time()
last_minute_time = time.time()
last_second_time = time.time()

# Directory for CSV files
log_dir = 'pressure-test/pressure_logs'
os.makedirs(log_dir, exist_ok=True)

# File paths for CSV files
file1 = os.path.join(log_dir, 'pressure_top.csv')
file2 = os.path.join(log_dir, 'pressure_bottom.csv')
file_raw = os.path.join(log_dir, 'pressure_raw.csv')

# Create a CSV file if one is not existent and ensure that it has headers
def initialize_csv_file(file, type='formatted'):
    if not os.path.exists(file):  # Check if the file already exists
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            if type == 'raw':
                writer.writerow(['Timestamp', 'Top Sensor', 'Bottom Sensor'])
            elif type == 'formatted':
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
    with open(file_raw, 'a', newline='') as f_raw:
        writer_raw = csv.writer(f_raw)
        writer_raw.writerows(data_stream_raw)  # Write all rows for data_stream_raw

    # Clear the data lists after writing
    data_stream1.clear()
    data_stream2.clear()
    data_stream_raw.clear()

# Initialize CSV files only if they do not exist
initialize_csv_file(file1, 'formatted')
initialize_csv_file(file2, 'formatted')
initialize_csv_file(file_raw, 'raw')

# Initialize variables for minute data collection
minute_data1 = []
minute_data2 = []

# Initialize the warning flag
warning_printed = False

# Main loop for data logging
try:
    # Start time for synchronizing the loop
    next_loop_time = time.time()

    while True:
        # Wait until it's time for the next reading
        current_time = time.time()
        while current_time < next_loop_time:
            # Sleep for a short duration (e.g., 5 ms)
            time.sleep(0.005)
            current_time = time.time()

        # Record the loop start time
        loop_start = current_time  # Use current_time instead of calling time.time() again

        # Read data from both ADCs
        value1 = adc1.read_adc(0)#, gain=GAIN)
        value2 = adc2.read_adc(0)#, gain=GAIN)

        # Get the current timestamp
        timestamp_str = datetime.datetime.fromtimestamp(loop_start).isoformat()
        data_stream_raw.append([timestamp_str, value1, value2])

        # Append data to the minute lists
        minute_data1.append(value1)
        minute_data2.append(value2)

        # Every second, update last_second_time
        if loop_start - last_second_time >= 1:
            last_second_time = loop_start

            # Check if a minute has passed to calculate statistics
            if loop_start - last_minute_time >= 60:
                last_minute_time = loop_start

                # Calculate statistics for the minute data
                avg1 = statistics.mean(minute_data1)
                stddev1 = statistics.stdev(minute_data1) if len(minute_data1) > 1 else 0
                min1 = min(minute_data1)
                max1 = max(minute_data1)
                avg2 = statistics.mean(minute_data2)
                stddev2 = statistics.stdev(minute_data2) if len(minute_data2) > 1 else 0
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

        # Check if it's time to write to the files
        if loop_start - last_write_time >= WRITE_INTERVAL:
            # Write data to CSV files
            print("Writing data to CSV files... ", end="")
            write_to_csv()
            print("Saved.")

            # Update the last write time
            last_write_time = loop_start

        # Schedule the next loop
        next_loop_time += DATA_INTERVAL / 1000.0  # Convert milliseconds to seconds

        # Calculate the time taken for this iteration
        iteration_time = (time.time() - loop_start) * 1000  # in milliseconds

        # Check if the iteration took longer than DATA_INTERVAL
        if iteration_time > DATA_INTERVAL:
            delay = iteration_time - DATA_INTERVAL
            if not warning_printed:
                print(f"Warning: Iteration took longer than DATA_INTERVAL ({DATA_INTERVAL}) by {delay:.2f} ms")
                print("Please lower the data interval or optimize the code.")
                warning_printed = True
        else:
            warning_printed = False

except KeyboardInterrupt:
    print("\nData logging stopped by user, writing data to CSV files... ", end="")
    write_to_csv()  # Write any remaining data before exiting
    print("Saved.")
