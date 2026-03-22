import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
import mantaPosLib as manta
import os
import datetime
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    message="No artists with labels found to put in legend.*",
    category=UserWarning,
)


timestamp_lines = 0

def compute_pos_from_accel(accelerations: np.ndarray, timestamps: np.ndarray, start_pos: float) -> np.ndarray:
    """
    Compute positions by integrating acceleration twice.

    Parameters:
    - accelerations: numpy array of acceleration values in m/s².
    - timestamps: numpy array of unixtime timestamps in milliseconds.
    - start_pos: starting position in meters.

    Returns:
    - positions: numpy array of position values in meters corresponding to the timestamps.
    """
    # Ensure the input arrays have the same length.
    if accelerations.shape[0] != timestamps.shape[0]:
        raise ValueError("The acceleration and timestamp arrays must have the same length.")

    n = len(accelerations)
    velocities = np.empty(n)
    positions = np.empty(n)
    
    # Assume initial velocity is 0 m/s.
    velocities[0] = 0.0
    positions[0] = start_pos
    
    # Loop through each time interval, using the trapezoidal rule.
    for i in range(1, n):
        # Convert time difference from milliseconds to seconds.
        dt = (timestamps[i] - timestamps[i-1]) / 1000.0
        if dt > 10: # Ignore time differences larger than 10 seconds
            print(f"Warning: Large time difference detected: {dt}s. Assuming velocity, position reset.")
            velocities[i-1] = 0.0
            positions[i-1] = start_pos
            velocities[i] = 0.0
            positions[i] = start_pos
            continue
        
        # Integrate acceleration to update velocity.
        velocities[i] = velocities[i-1] + 0.5 * (accelerations[i-1] + accelerations[i]) * dt
        
        # Integrate velocity to update position.
        positions[i] = positions[i-1] + 0.5 * (velocities[i-1] + velocities[i]) * dt
    
    return positions

class DataProcessor:
    """
    A class to process and align various sensor data from JSON recordings.
    Supports data extraction, alignment, visualization, and export.
    """

    def __init__(self, file_paths=None):
        """Initialize the DataProcessor with optional file paths."""
        self.file_paths = file_paths
        self.data = None
        self.extracted_data = {}
        self.aligned_data = None

    def load_data(self, file_paths=None):
        """Load and combine JSON data from multiple files."""
        if file_paths:
            self.file_paths = file_paths

        if not self.file_paths:
            raise ValueError("File paths not provided.")
        
        files_with_timestamp = []
        for file in self.file_paths:
            try:
                # Check if file exists before trying to open
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Error: File not found at {file}")
                with open(file, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    ts_str = first_line[30:30+26]
                    ts =  datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
                files_with_timestamp.append((file, ts))
            except Exception as e:
                print(f"Error reading timestamp from {file}: {e}")

        print("Sorting files by timestamp...")
        files_with_timestamp.sort(key=lambda x: x[1])

        combined_data = []
        try:  
            for file, _ in files_with_timestamp:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            combined_data.append(json.loads(line))
            self.data = combined_data

            # Print the number of records loaded and from which files
            print(f"Loaded {len(self.data)} records from {len(files_with_timestamp)} file{'s' if len(files_with_timestamp) > 1 else ''}:", end=" ")
            files_str = ""
            for file, _ in files_with_timestamp[:min(4, len(files_with_timestamp))]: # Limit to 5 files for printing]
                files_str += f"'{file[11:]}', "
            if len(files_with_timestamp) < 5:
                print(files_str[:-2])
            else:
                print(f"{files_str[:-2]}...")

            return True
        except json.JSONDecodeError as e:
            # Try to find the line number
            error_line = -1
            try:
                with open(file, 'r') as f_err:
                    for i, _ in enumerate(f_err):
                        if i + 1 == e.lineno:
                            break
                    error_line = e.lineno
            except Exception:
                pass # Ignore errors trying to find the line
            if error_line > 0:
                raise ValueError(f"Error decoding JSON in file '{file}' at line {error_line}: {e}")
            else:
                raise ValueError(f"Error decoding JSON in file '{file}': {e}")
        except Exception as e: # Catch other potential file reading errors
            raise IOError(f"Error reading file '{self.file_path}': {e}")

    def process_data_timings(self, debug_print=""):
        """
        Analyze the timing of the data and print debug information.
        Returns the averaged timestamps that tries to assume a camera timestamp.
        Parameters:
        - debug_print: If "all", prints detailed debug information. If "result", prints summary statistics.
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")

        data_types = defaultdict(int)
        cam_fpss = []
        recv_times = []
        cam_timestamps = []
        time_step = []
        last_camera_timestamp = 1741266898664

        # Analyze the structure of the JSON data
        for item in self.data:
            mpu_unit = item.get('mpu_unit', 'N/A')
            prefix = f"mpu{mpu_unit}." if mpu_unit != 'N/A' else ""
            
            # Printing the unit 4 time differences for debugging
            if mpu_unit == 4:
                cam_fpss.append(item['camera']['fps'])
                recv_times.append(int(datetime.datetime.fromisoformat(item.get('recv_time')).timestamp() * 1000))
                cam_timestamps.append(item['camera']['timestamp'])

                time_step.append(cam_timestamps[-1] - last_camera_timestamp)
                if time_step[-1] < 0:
                    cam_timestamps[-1] += 1000
                    last_camera_timestamp = cam_timestamps[-1]//1000 * 1000
                else:
                    last_camera_timestamp = cam_timestamps[-1]

            for key, value in item.items():
                if key in ['mpu_unit', 'recv_time', 'packet_number']:
                    data_types[key] += 1
                elif isinstance(value, dict):
                    if key == 'imu': # Special handling for nested IMU
                        if 'acceleration' in value and isinstance(value['acceleration'], dict):
                            for subkey in value['acceleration'].keys():
                                data_types[f"{prefix}imu.acceleration.{subkey}"] += 1
                        if 'gyroscope' in value and isinstance(value['gyroscope'], dict):
                            for subkey in value['gyroscope'].keys():
                                data_types[f"{prefix}imu.gyroscope.{subkey}"] += 1
                         # Add timestamp if present
                        if 'timestamp' in value:
                            data_types[f"{prefix}imu.timestamp"] += 1
                    else: # Generic dictionary handling
                        for subkey in value.keys():
                            # Handle nested position/rotation in camera data
                            if isinstance(value[subkey], dict) and ('position' in value[subkey] or 'rotation' in value[subkey]):
                                 for subsubkey in value[subkey].keys():
                                    data_types[f"{prefix}{key}.{subkey}.{subsubkey}"] += 1
                            # Handle lists like position/rotation directly under camera_pos_X
                            elif isinstance(value[subkey], list) and subkey in ['position', 'rotation']:
                                data_types[f"{prefix}{key}.{subkey} (list)"] += 1
                            else:
                                data_types[f"{prefix}{key}.{subkey}"] += 1
                else:
                    data_types[f"{prefix}{key}"] += 1

        # Print timestep debug data
        mean_recv_camera = int(np.mean(np.array(recv_times)-np.array(cam_timestamps)))
        recv_times = np.array(recv_times) - mean_recv_camera

        #ref_timestamps = np.copy(cam_timestamps)
        ref_timestamps = np.copy(recv_times) # referencing receiver timestamps seems more reliable
        avg_timestep = 1000/np.mean(cam_fpss)
        avg_timestamps = [ref_timestamps[0]-1]
        for i in range(1, len(ref_timestamps)):
            avg_timestamps.append(avg_timestamps[i-1] + avg_timestep)
            while avg_timestamps[i] + avg_timestep < ref_timestamps[i]:
                avg_timestamps[i] += avg_timestep

        avg_timestamps = np.array(avg_timestamps)
        avg_timestamps = np.round(avg_timestamps)
        avg_timestamps = avg_timestamps.astype(int)
        
        for i in range(len(cam_fpss)):
            if debug_print == "all":
                print(f"cam_fps-mean: {(cam_fpss[i]-np.mean(cam_fpss)):+3.4f}Hz, time_step_cam: {time_step[i]:4d}ms, time_step_recv: {recv_times[i]-recv_times[i-1]:4d}ms, time_step_avg: {avg_timestamps[i]-avg_timestamps[i-1]:4d}, recv-camera: {(recv_times[i]-cam_timestamps[i]):4d}ms, avg-camera: {(avg_timestamps[i]-cam_timestamps[i]):4d}ms, avg-recv: {(avg_timestamps[i]-recv_times[i]):4d}ms", end="")
                
                # Check for positive difference
                if avg_timestamps[i] > ref_timestamps[i]:
                    print("     <- Warning: camera time after PC")
                else:
                    print("")
            elif avg_timestamps[i] > ref_timestamps[i]:
                print(f"Warning: camera time after PC at index {i} & time {ref_timestamps[i]}")

        if len(cam_fpss) > 0 and debug_print == "result":
            print(f"\nMean fps: {np.mean(cam_fpss):3.3f} +- {np.std(cam_fpss):3.3f}Hz\
                  \n -> Avg cam timestep: {avg_timestep:.4f}ms\
                  \n\nMean recv-camera: {mean_recv_camera} +- {np.std(np.array(recv_times)-np.array(cam_timestamps)):.2f}ms\
                  \nMean avg-camera: {np.mean(avg_timestamps-np.array(cam_timestamps)):.4f} +- {np.std(avg_timestamps-np.array(cam_timestamps)):.4f}ms\
                  \nMean avg-recv: {np.mean(avg_timestamps-np.array(recv_times)):.4f} +- {np.std(avg_timestamps-np.array(recv_times)):.4f}ms")
        
        # Sort for readability
        return avg_timestamps

    def extract_all_data(self):
        """Extract all available sensor data and organize by type and MPU unit."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")

        # Reset extracted data
        self.extracted_data = {}

        # Process all items
        for item in self.data:
            mpu_unit = item.get('mpu_unit') # Can be None if mpu_unit is missing

            # Skip items without mpu_unit if we decide to require it
            # For now, allow None mpu_unit to be grouped separately if needed
            # MPU key will be 'mpuNone' if mpu_unit is None, handled by _ensure_dict_in_extracted_data

            # Iterate through all top-level keys in the JSON object
            for top_key, top_value in item.items():
                # Skip metadata fields we don't want to treat as sensors
                if top_key in ['mpu_unit', 'recv_time', 'packet_number']:
                    continue

                # Process dictionary values (potential sensor groups)
                if isinstance(top_value, dict):
                    # --- Special Handling for IMU ---
                    if top_key == 'imu':
                        imu_timestamp = top_value.get('timestamp')
                        if imu_timestamp is None:
                            # print(f"Warning: IMU data missing timestamp in record: {item}")
                            continue # Skip IMU record if no timestamp

                        # Process acceleration
                        if 'acceleration' in top_value and isinstance(top_value['acceleration'], dict):
                            accel_data = self._ensure_dict_in_extracted_data('acceleration', mpu_unit)
                            # Append the shared IMU timestamp
                            if 'timestamp' not in accel_data: accel_data['timestamp'] = []
                            accel_data['timestamp'].append(imu_timestamp)
                            # Append axis values
                            for axis, value in top_value['acceleration'].items():
                                if axis not in accel_data: accel_data[axis] = []
                                accel_data[axis].append(value)

                        # Process gyroscope
                        if 'gyroscope' in top_value and isinstance(top_value['gyroscope'], dict):
                            gyro_data = self._ensure_dict_in_extracted_data('gyroscope', mpu_unit)
                            # Append the shared IMU timestamp
                            if 'timestamp' not in gyro_data: gyro_data['timestamp'] = []
                            gyro_data['timestamp'].append(imu_timestamp)
                            # Append axis values
                            for axis, value in top_value['gyroscope'].items():
                                if axis not in gyro_data: gyro_data[axis] = []
                                gyro_data[axis].append(value)
                        # Add any other potential direct children of 'imu' here if needed

                    # --- Generic Handling for Other Dictionary Data ---
                    # Covers: encoder, temperature, pressure, ptp, camera, global_pos, camera_pos_*
                    else:
                        sensor_type = top_key
                        sensor_data = self._ensure_dict_in_extracted_data(sensor_type, mpu_unit)

                        # if 'sensor_type' is camera_pos_*, get time from the camera
                        if sensor_type.startswith('camera_pos_'):
                            # Use the camera's timestamp for this sensor group
                            if 'camera' in item and 'timestamp' in item['camera']:
                                if 'timestamp' not in sensor_data: sensor_data['timestamp'] = []
                                sensor_data['timestamp'].append(item['camera']['timestamp'])
                            else:
                                print(f"Warning: Camera position data missing timestamp in record: {item}")
                                continue

                        # Append all key-value pairs from the dictionary
                        for subkey, subvalue in top_value.items():
                            if subkey not in sensor_data:
                                sensor_data[subkey] = []
                            sensor_data[subkey].append(subvalue)

        # Convert all collected data lists to numpy arrays for easier processing
        self._convert_lists_to_arrays()
        
        # Compute integ_pos.x for IMU using acceleration.x from the 'acceleration' sensor group.
        for mpu_key, mpu_data in self.extracted_data.items():
            if 'acceleration' in mpu_data:
                accel_data = mpu_data['acceleration']
                for direction in ['x', 'y', 'z']:
                    if 'timestamp' in accel_data and direction in accel_data:
                        ts = accel_data['timestamp']
                        a_dir = accel_data[direction].copy() # Copy to avoid modifying original data
                        
                        # Center the acceleration data around 0 for integration
                        #a_dir -= np.mean(a_dir)
                        a_dir -= np.median(a_dir)
                        
                        if isinstance(ts, np.ndarray) and len(ts) > 1 and isinstance(a_dir, np.ndarray) and len(a_dir) == len(ts):
                            try:
                                pos_dir = compute_pos_from_accel(a_dir.astype(float), ts.astype(float), start_pos=0.0)
                                # Add computed integ_pos.x to new 'imu' key
                                if 'integ_pos' not in mpu_data:
                                    mpu_data['integ_pos'] = {}
                                mpu_data['integ_pos'][direction] = pos_dir
                                if 'timestamp' not in mpu_data['integ_pos']:
                                    mpu_data['integ_pos']['timestamp'] = ts
                            except Exception as e:
                                print(f"Error computing position for {mpu_key}: {e}")

        return self.extracted_data

    def _ensure_dict_in_extracted_data(self, sensor_type, mpu_unit=None):
        """Ensure that the dictionary structure exists in extracted_data."""
        # Use 'mpuNone' if mpu_unit is actually None
        mpu_key = f"mpu{mpu_unit}"

        if mpu_key not in self.extracted_data:
            self.extracted_data[mpu_key] = {}
        if sensor_type not in self.extracted_data[mpu_key]:
            self.extracted_data[mpu_key][sensor_type] = {}
        return self.extracted_data[mpu_key][sensor_type]

    def _convert_lists_to_arrays(self):
        """Convert all data lists within extracted_data to numpy arrays. Handles nested dicts."""
        def convert_dict_values(d):
            for key, value in d.items():
                if isinstance(value, list):
                    try:
                        # Attempt to convert to numpy array
                        # Allow object dtype for lists containing non-numerics or varying lengths (like rotation lists)
                        d[key] = np.array(value, dtype=object if not all(isinstance(i, (int, float)) for i in value) else None)
                        # Try converting object arrays of numbers/lists-of-numbers to numeric if possible
                        if d[key].dtype == object:
                            try:
                                first_elem = d[key][0]
                                if isinstance(first_elem, (list, np.ndarray)):
                                    # Attempt conversion to a 2D numeric array if elements are lists/arrays
                                    d[key] = np.array(value, dtype=float)
                                elif isinstance(first_elem, (int, float)):
                                    # Attempt conversion to a 1D numeric array if elements are numbers
                                    d[key] = np.array(value, dtype=float)
                            except (IndexError, TypeError, ValueError):
                                pass # Keep as object array if conversion fails or list is empty
                    except Exception as e:
                        print(f"Warning: Could not convert list for key '{key}' to numpy array: {e}")
                        # Keep as list if conversion fails entirely
                        pass
                elif isinstance(value, dict):
                    convert_dict_values(value) # Recurse for nested dictionaries

        convert_dict_values(self.extracted_data)


    def align_data(self, reference_sensor='encoder', reference_field='timestamp',
                  reference_mpu=1, target_fields=None):
        """
        Align sensor data to a reference timestamp using linear interpolation.

        Parameters:
        - reference_sensor: The sensor type containing the reference timestamps (default: 'encoder')
        - reference_field: The field name of the timestamps within the reference sensor (default: 'timestamp')
        - reference_mpu: The MPU unit number of the reference sensor (default: 1)
        - target_fields: Dict specifying {sensor_type: [field_name1, field_name2]} to align.
                         If None, attempts to align all numeric fields that have a 'timestamp'
                         field in the same sensor group (default: None).

        Returns:
        - Dictionary containing the aligned data. Timestamps are under ['reference']['timestamps'].
          Other data is nested under [mpu_number][sensor_type][field_name].
        """
        if not self.extracted_data:
            print("Warning: No data extracted. Running extract_all_data().")
            self.extract_all_data()
            if not self.extracted_data:
                raise ValueError("Data extraction failed or produced no data.")


        # Create aligned data structure
        self.aligned_data = {}

        # --- Get Reference Timestamps ---
        ref_mpu_key = f"mpu{reference_mpu}"
        if ref_mpu_key not in self.extracted_data:
            raise ValueError(f"Reference MPU key '{ref_mpu_key}' not found in extracted data. Available keys: {list(self.extracted_data.keys())}")
        if reference_sensor not in self.extracted_data[ref_mpu_key]:
            raise ValueError(f"Reference sensor '{reference_sensor}' not found in MPU {reference_mpu}. Available sensors: {list(self.extracted_data[ref_mpu_key].keys())}")

        ref_sensor_data = self.extracted_data[ref_mpu_key][reference_sensor]
        if reference_field not in ref_sensor_data:
            raise ValueError(f"Reference field '{reference_field}' not found in {ref_mpu_key}.{reference_sensor}. Available fields: {list(ref_sensor_data.keys())}")

        ref_timestamps = ref_sensor_data[reference_field]

        # Ensure reference timestamps are numeric and sorted
        if not isinstance(ref_timestamps, np.ndarray) or not np.issubdtype(ref_timestamps.dtype, np.number):
            raise TypeError(f"Reference timestamps ({ref_mpu_key}.{reference_sensor}.{reference_field}) are not a numeric numpy array.")
        if not np.all(np.diff(ref_timestamps) >= 0):
            print("Warning: Reference timestamps are not monotonically increasing. Sorting them.")
            sort_indices = np.argsort(ref_timestamps)
            ref_timestamps = ref_timestamps[sort_indices]
            # Apply sorting to other fields in the reference sensor if needed? For now, just sort timestamps.

        if len(ref_timestamps) < 2:
            raise ValueError("Need at least two reference timestamps for interpolation.")

        self.aligned_data['reference'] = {
            'sensor': reference_sensor,
            'mpu': reference_mpu,
            'timestamps': ref_timestamps
        }
        print(f"Using reference timestamps from {ref_mpu_key}.{reference_sensor}.{reference_field} (count: {len(ref_timestamps)}, range: {ref_timestamps.min()} to {ref_timestamps.max()})")

        # --- Determine Target Fields ---
        # If target_fields is None, automatically find all numeric fields
        # associated with a 'timestamp' field within the same sensor group.
        auto_target_fields = {}
        if target_fields is None:
            for mpu_key, mpu_data in self.extracted_data.items():
                if not isinstance(mpu_data, dict): continue
                mpu_number = int(mpu_key.replace('mpu', '')) if mpu_key.startswith('mpu') and mpu_key[3:].isdigit() else mpu_key

                for sensor_type, sensor_data in mpu_data.items():
                    # Check if this sensor group has its own timestamp series
                    if 'timestamp' in sensor_data and isinstance(sensor_data['timestamp'], np.ndarray) and len(sensor_data['timestamp']) > 1:
                        current_target_fields = []
                        for field_name, field_data in sensor_data.items():
                            # Align numeric numpy arrays, excluding the timestamp itself
                            if field_name != 'timestamp' and isinstance(field_data, np.ndarray) and np.issubdtype(field_data.dtype, np.number):
                                # Ensure data length matches timestamp length for this sensor
                                if len(field_data) == len(sensor_data['timestamp']):
                                    current_target_fields.append(field_name)
                                else:
                                    print(f"Warning: Skipping {mpu_key}.{sensor_type}.{field_name}. Length mismatch with its timestamp (Data: {len(field_data)}, Timestamp: {len(sensor_data['timestamp'])}).")

                        if current_target_fields:
                            # Store per MPU to handle cases where same sensor type exists on multiple MPUs
                            if mpu_number not in auto_target_fields: auto_target_fields[mpu_number] = {}
                            auto_target_fields[mpu_number][sensor_type] = current_target_fields

            target_fields_to_use = auto_target_fields
        else:
             # Validate user-provided target_fields structure if needed
             target_fields_to_use = target_fields # Assume user provided dict like {mpu_num: {sensor:[fields]}} or similar structure expected below
             print(f"Using provided target fields: {target_fields_to_use}")


        # --- Align Each Target Field ---
        for mpu_key, mpu_data in self.extracted_data.items():
            if not isinstance(mpu_data, dict): continue

            try:
                # Extract MPU number reliably, handle 'mpuNone' etc.
                if mpu_key.startswith('mpu') and mpu_key[3:].isdigit():
                    mpu_number = int(mpu_key.replace('mpu', ''))
                else:
                    # Use the key itself if it doesn't fit the pattern, or skip
                    # mpu_number = mpu_key
                    print(f"Skipping alignment for non-standard MPU key: {mpu_key}")
                    continue
            except ValueError:
                print(f"Skipping alignment for MPU key with invalid number: {mpu_key}")
                continue

            # Check if this MPU has any fields targeted for alignment
            if mpu_number not in target_fields_to_use:
                continue

            mpu_target_sensors = target_fields_to_use[mpu_number]

            for sensor_type, sensor_data in mpu_data.items():
                # Check if this sensor type is targeted for this MPU
                if sensor_type not in mpu_target_sensors:
                    continue

                # Check for timestamp again, essential for interpolation
                if 'timestamp' not in sensor_data or not isinstance(sensor_data['timestamp'], np.ndarray):
                    print(f"Warning: Skipping {mpu_key}.{sensor_type}. No valid timestamp array found.")
                    continue

                sensor_timestamps = sensor_data['timestamp']

                # Ensure sensor timestamps are numeric and have sufficient length
                if not np.issubdtype(sensor_timestamps.dtype, np.number) or len(sensor_timestamps) < 2:
                    print(f"Warning: Skipping {mpu_key}.{sensor_type}. Timestamps are not numeric or insufficient count ({len(sensor_timestamps)}).")
                    continue

                # Ensure sensor timestamps are sorted
                if not np.all(np.diff(sensor_timestamps) >= 0):
                    print(f"Warning: Timestamps for {mpu_key}.{sensor_type} are not sorted. Sorting them along with data.")
                    sort_indices = np.argsort(sensor_timestamps)
                    sensor_timestamps = sensor_timestamps[sort_indices]
                    # Also sort the data fields that will be interpolated
                    fields_to_align = mpu_target_sensors[sensor_type]
                    for field_name in fields_to_align:
                        if field_name in sensor_data and isinstance(sensor_data[field_name], np.ndarray) and len(sensor_data[field_name]) == len(sort_indices):
                            sensor_data[field_name] = sensor_data[field_name][sort_indices]
                        elif field_name in sensor_data:
                            print(f"Warning: Could not sort data for {mpu_key}.{sensor_type}.{field_name} due to length mismatch or type.")


                # Get fields to align for this sensor from the target list
                fields_to_align = mpu_target_sensors[sensor_type]
                if not isinstance(fields_to_align, list): # Ensure it's a list
                    fields_to_align = [fields_to_align]

                for field_name in fields_to_align:
                    if field_name not in sensor_data:
                        print(f"Warning: Target field '{field_name}' not found in {mpu_key}.{sensor_type}.")
                        continue

                    field_data = sensor_data[field_name]

                    # Final checks before interpolation
                    if not isinstance(field_data, np.ndarray) or not np.issubdtype(field_data.dtype, np.number):
                        print(f"Warning: Skipping interpolation for {mpu_key}.{sensor_type}.{field_name}. Data is not a numeric numpy array.")
                        continue
                    if len(field_data) != len(sensor_timestamps):
                        print(f"Warning: Skipping interpolation for {mpu_key}.{sensor_type}.{field_name}. Data length ({len(field_data)}) doesn't match timestamp length ({len(sensor_timestamps)}).")
                        continue

                    # --- Perform Interpolation ---
                    try:
                        # Use fill_value="extrapolate" if you want to extrapolate beyond the sensor's time range
                        # Use bounds_error=False, fill_value=np.nan to put NaN for points outside the range
                        interp_func = interp1d(
                            sensor_timestamps,
                            field_data,
                            kind='linear', # Common choice, others: 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
                            bounds_error=False, # Don't raise error if ref_timestamps are outside sensor_timestamps range
                            fill_value=np.nan   # Fill values outside the range with NaN
                        )

                        # Interpolate the field at reference timestamps
                        interpolated_values = interp_func(ref_timestamps)

                        # --- Store the Aligned Data ---
                        if mpu_number not in self.aligned_data:
                            self.aligned_data[mpu_number] = {}
                        if sensor_type not in self.aligned_data[mpu_number]:
                            self.aligned_data[mpu_number][sensor_type] = {}

                        self.aligned_data[mpu_number][sensor_type][field_name] = interpolated_values
                        # print(f"Aligned {mpu_key}.{sensor_type}.{field_name}") # Verbose logging

                    except ValueError as ve:
                        # Catches issues like non-unique timestamp values if sorting didn't fix it
                        print(f"Interpolation ValueError for {mpu_key}.{sensor_type}.{field_name}: {str(ve)}. Ensure timestamps are unique and sorted.")
                        print(f"Timestamps: {sensor_timestamps}")
                    except Exception as e:
                        # Catch any other unexpected interpolation errors
                        print(f"Error interpolating {mpu_key}.{sensor_type}.{field_name}: {type(e).__name__} - {str(e)}")

        # Check how much data was actually aligned
        aligned_count = 0
        for mpu, mpu_d in self.aligned_data.items():
            if mpu == 'reference': continue
            for sensor, sensor_d in mpu_d.items():
                aligned_count += len(sensor_d)

        if aligned_count == 0 and target_fields is None:
            print("\nWarning: No data fields were aligned. Possible reasons:")
            print("- Reference sensor/MPU/field incorrect?")
            print("- No other sensors had matching 'timestamp' fields and numeric data?")
            print("- Timestamps ranges do not overlap significantly?")
            print("- Data length mismatches within sensor groups?")


        return self.aligned_data

    def export_to_pandas(self):
        """Export data to pandas DataFrame."""
        if not self.aligned_data or 'reference' not in self.aligned_data or 'timestamps' not in self.aligned_data['reference']:
            #print("Warning: Aligned data not available, exporting raw extracted data if available.")
            data_source = self.extracted_data
        else:
            data_source = self.aligned_data

        df_dict = {}
        if 'reference' in data_source and 'timestamps' in data_source['reference']:
            df_dict['timestamp'] = data_source['reference']['timestamps']
        else:
            # Try to find timestamp(s) from any sensor group
            found_ts = False
            for mpu_key, mpu_data in data_source.items():
                if mpu_key == 'reference':
                    continue
                for sensor_type, sensor_data in mpu_data.items():
                    if 'timestamp' in sensor_data:
                        df_dict['timestamp'] = sensor_data['timestamp']
                        found_ts = True
                        break
                if found_ts:
                    break
            if 'timestamp' not in df_dict:
                raise ValueError("No timestamp data available for export.")

        for key, value in data_source.items():
            if key == 'reference':
                continue
            for sensor_type, sensor_data in value.items():
                for field_name, field_values in sensor_data.items():
                    if field_name == 'timestamp':
                        continue
                    column_name = f"{key}_{sensor_type}_{field_name}" if key.startswith("mpu") else f"{sensor_type}_{field_name}"
                    if column_name in df_dict:
                        print(f"Warning: Duplicate column name generated: {column_name}. Overwriting.")
                    df_dict[column_name] = field_values

        try:
            # Flatten multi-dimensional arrays to avoid DataFrame errors
            for name, value in df_dict.items():
                if isinstance(value, np.ndarray) and (value.ndim > 1):
                    df_dict[name] = list(value)
            df = pd.DataFrame(df_dict)
            return df
        except ValueError as e:
            print(f"\nError creating DataFrame. Error: {e} \nCheck for unequal array lengths:")
            for name, data in df_dict.items():
                print(f"  Column '{name}': Length {len(data)}")
            raise ValueError(f"Could not create DataFrame: {e}")

    def print_data(self, sensor_types=None, mpu_units=None, num_rows=10):
        # Print a preview of the data in a DataFrame format.
 
        try:
            df = self.export_to_pandas()
        except Exception as e:
            print(f"Error exporting data to DataFrame for printing: {e}")
            return

        # Filter columns based on sensor_types and mpu_units if specified
        cols_to_show = ['timestamp']
        available_mpus = [mpu for mpu in self.aligned_data.keys() if mpu != 'reference'] if self.aligned_data else [mpu for mpu in self.extracted_data.keys()]
        show_mpus = mpu_units if mpu_units is not None else available_mpus

        all_sensors = set()
        for mpu in show_mpus:
            if (self.aligned_data and mpu in self.aligned_data) or (not self.aligned_data and mpu in self.extracted_data):
                if self.aligned_data:
                    all_sensors.update(self.aligned_data[mpu].keys())
                else:
                    all_sensors.update(self.extracted_data[mpu].keys())
        show_sensors = sensor_types if sensor_types is not None else list(all_sensors)

        for mpu in show_mpus:
            for sensor in show_sensors:
                data_source = self.aligned_data if self.aligned_data else self.extracted_data
                if mpu in data_source and sensor in data_source[mpu]:
                    for field in data_source[mpu][sensor].keys():
                         col_name = f"mpu{mpu}_{sensor}_{field}"
                         if col_name in df.columns:
                            cols_to_show.append(col_name)

        if len(cols_to_show) > 1:
            with pd.option_context('display.max_rows', num_rows,
                                    'display.max_columns', None,
                                    'display.width', 1000):
                                    print(df[cols_to_show].head(num_rows))
        else:
            print("No columns match the specified filters.")

        print(f"\nShowing first {min(num_rows, len(df))} rows.")
        print(f"Total rows: {len(df)}, Total columns in full exported data: {len(df.columns)}")

    def visualize(self, sensor_types=None, mpu_units=None, fields=None, figsize=(8, 3), sharex=True,
                  ref_timestamps=None, ref_data=None, all_camera_timestamps = None, all_camera_data = None):
        """
        Visualize the aligned sensor data, plotting each field on its own subplot.

        Parameters:
        - sensor_types: List of sensor types to plot (e.g., ['pressure', 'acceleration']). Default: all aligned.
        - mpu_units: List of MPU units to plot (e.g., [0, 1]). Default: all aligned.
        - fields: Optional: List of specific field names to plot (e.g., ['depth0', 'x']). Default: all aligned fields for the selected sensors/MPUs.
        - figsize: Figure size (width, height) per subplot in inches.
        - sharex: Whether subplots should share the x-axis (default: True).
        """
        # If aligned data exists, use it; otherwise, fallback to extracted data.
        if self.aligned_data and 'reference' in self.aligned_data and 'timestamps' in self.aligned_data['reference']:
            # Reference timestamps (use copy to avoid modifying original)
            timestamps = self.aligned_data['reference']['timestamps'].copy()
            # Optional: Convert timestamps for plotting if they are large numbers (e.g., ms to s)
            # timestamps_plot = (timestamps - timestamps[0]) / 1000.0 # Example: relative time in seconds if timestamps are ms
            timestamps_plot = timestamps # Use original for now

            # Determine which MPUs, sensors, and fields to plot
            target_mpus = set()
            target_sensors = set()
            all_plot_items = [] # List of (mpu, sensor, field, data) tuples

            available_mpus = [mpu for mpu in self.aligned_data.keys() if mpu != 'reference']
            plot_mpus = mpu_units if mpu_units is not None else available_mpus

            for mpu in plot_mpus:
                if mpu not in self.aligned_data:
                    print(f"Warning: MPU {mpu} not found in aligned data.")
                    continue
                target_mpus.add(mpu)
                available_sensors = self.aligned_data[mpu].keys()
                plot_sensors = sensor_types if sensor_types is not None else available_sensors

                for sensor in plot_sensors:
                    if sensor not in self.aligned_data[mpu]:
                        # print(f"Warning: Sensor {sensor} not found for MPU {mpu} in aligned data.")
                        continue
                    target_sensors.add(sensor)
                    available_fields = self.aligned_data[mpu][sensor].keys()
                    plot_fields = fields if fields is not None else available_fields

                    for field in plot_fields:
                        negative = False
                        if field[0] == '-':
                            field = field[1:]
                            negative = True


                        if field not in self.aligned_data[mpu][sensor]:
                            # print(f"Warning: Field {field} not found for MPU {mpu}, Sensor {sensor}.")
                            continue

                        field_data = self.aligned_data[mpu][sensor][field]
                        # Ensure data has same length as timestamps
                        if len(field_data) == len(timestamps_plot):
                            all_plot_items.append((mpu, sensor, field, field_data, negative))
                        else:
                            print(f"Warning: Skipping plot for mpu{mpu}_{sensor}_{field}. Length mismatch (Data: {len(field_data)}, Timestamps: {len(timestamps_plot)}).")


            if not all_plot_items:
                print("No data found to plot based on the specified filters.")
                return

            # Group items by field name for plotting on separate axes
            plots_by_field = defaultdict(list)
            for mpu, sensor, field, data, negative in all_plot_items:
                plots_by_field[field].append({'mpu': mpu, 'sensor': sensor, 'data': data, 'negative': negative})

            num_plots = len(plots_by_field)
            if num_plots == 0:
                print("No valid fields to plot.")
                return

            # Create figure and axes
            fig, axes = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots), sharex=sharex)
            if num_plots == 1: # Make axes subscriptable even if only one plot
                axes = [axes]

            # Plot data
            ax_idx = 0
            for field_name, plot_list in plots_by_field.items():
                ax = axes[ax_idx]
                for plot_item in plot_list:
                    mpu = plot_item['mpu']
                    sensor = plot_item['sensor']
                    data = plot_item['data']
                    if plot_item['negative']:
                        data = -data # Negate data if specified
                    label = f"MPU {mpu} ({sensor})"
                    ax.plot(timestamps_plot, data, label=label, marker='.', markersize=2, linestyle='-') # Small markers + line

                ax.set_title(f"Field: {field_name}")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
                ax_idx += 1

            # Common X label
            if sharex:
                axes[-1].set_xlabel("Timestamp") # Or "Time (s)" if converted
            else:
                for ax in axes:
                    ax.set_xlabel("Timestamp")


            plt.suptitle(f"Aligned Sensor Data (Ref: MPU {self.aligned_data['reference']['mpu']} {self.aligned_data['reference']['sensor']})", fontsize=16)#, y=1.02) # Adjust y position
            plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
            plt.show()

        else:
            print("Aligned data not available. Using extracted data for visualization. Data might not be synchronized.")
            all_plot_items = []  # Each item: (mpu_key, sensor, field, timestamps, y_data)
            available_mpus = list(self.extracted_data.keys())
            # Use provided mpu_units if given, adapting number to key format if needed.
            plot_mpus = []
            if mpu_units is not None:
                for m in mpu_units:
                    key = f"mpu{m}" if f"mpu{m}" in self.extracted_data else m
                    plot_mpus.append(key)
            else:
                plot_mpus = available_mpus

            for mpu_key in plot_mpus:
                if mpu_key not in self.extracted_data:
                    print(f"Warning: MPU {mpu_key} not found in extracted data.")
                    continue
                mpu_data = self.extracted_data[mpu_key]
                sensors = sensor_types if sensor_types is not None else mpu_data.keys()
                for sensor in sensors:
                    if sensor not in mpu_data:
                        continue
                    sensor_data = mpu_data[sensor]
                    if 'timestamp' not in sensor_data:
                        print(f"Warning: Sensor {sensor} in {mpu_key} has no timestamp.")
                        continue
                    timestamps = sensor_data['timestamp']
                    for field in (fields if fields is not None else sensor_data.keys()):
                        negative = False
                        if field[0] == '-':
                            field = field[1:]
                            negative = True

                        if field == 'timestamp' or field not in sensor_data:
                            continue
                        y_data = sensor_data[field]
                        if len(y_data) != len(timestamps):
                            print(f"Warning: Length mismatch in {mpu_key}.{sensor}.{field}. Skipping.")
                            continue
                        all_plot_items.append((mpu_key, sensor, field, timestamps, y_data, negative))
            if not all_plot_items:
                print("No data to plot based on the specified filters in extracted data.")
                return

            num_plots = len(all_plot_items)
            if num_plots > 8:
                ncols = 3
                nrows = (num_plots + 2) // ncols
            elif num_plots > 3:
                ncols = 2
                nrows = (num_plots + 1) // ncols
            else:
                ncols = 1
                nrows = num_plots
            
            _, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows), sharex=sharex)

            # Flatten the axes array so that each element is a matplotlib Axes object
            axes = np.array(axes).flatten()
            # Remove the single plot check (not needed after flattening)
            for ax, (mpu_key, sensor, field, timestamps, y_data, negative) in zip(axes, all_plot_items):
                if y_data.ndim > 1:
                    vals = ["x", "y", "z", "xr", "yr", "zr"]
                    for i in range(y_data.shape[1]):
                        if negative:
                            y_data[:, i] = -y_data[:, i]

                        # Add aditional data for camera and reference if requested
                        if i == 0 and ref_timestamps is not None and ref_data is not None:
                            ax.plot(ref_timestamps, ref_data, marker='.', markersize=2, linestyle='-',
                                    label=f"reference - position_y")
                        if i == 0 and all_camera_timestamps is not None and all_camera_data is not None:
                            for k in range(5):
                                if len(all_camera_data[k]) > 0 and int(sensor[-1]) != k:
                                    ax.plot(all_camera_timestamps[k], all_camera_data[k], linestyle='-', alpha=0.4,
                                            label=f"{mpu_key} (camera_pos_{'avg' if k==4 else k}) - {field}_y")

                        if field == 'timestamps_shifted':
                            ax.plot(y_data[:, 1], y_data[:, 0], marker='.', markersize=2, linestyle='-')
                            for j in range(2, y_data.shape[1]):
                                ax.plot(y_data[:, j], y_data[:, 0], marker='.', markersize=2, linestyle='-', alpha=0.03)
                        else:
                            if i < len(vals):
                                if ref_timestamps is not None and ref_data is not None:
                                    continue
                                ax.plot(timestamps, y_data[:, i], marker='.', markersize=2, linestyle='-',
                                        label=f"{mpu_key} ({'camera_pos_avg' if sensor[-1]=='4' else sensor}) - {field}_{vals[i] if y_data.shape[1]<=len(vals) else i}")
                            else:
                                ax.plot(timestamps, y_data[:, i], marker='.', markersize=2, linestyle='-')
                else:
                    if negative:
                        y_data = -y_data
                    ax.plot(timestamps, y_data, marker='.', markersize=2, linestyle='-',
                            label=f"{mpu_key} ({sensor}) - {field}")
                ax.set_title(f"{mpu_key}.{sensor[:-1]+'avg' if sensor[-1]=='4' else sensor}.{field}")
                ax.set_ylabel("Value")
                ax.legend()
                ax.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax.yaxis.set_major_locator(plt.MaxNLocator(12))
                ax.minorticks_on()
                ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
                ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.tick_params(which='both', top=True, right=True)

                # Set x-axis indicators to scientific notation at full seconds level
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((3, 3))
                ax.xaxis.set_major_formatter(formatter)
                
            axes[-1].set_xlabel("Timestamp (display: s, data: ms)")
            plt.tight_layout()
            plt.show(block=False)

# --- Example Usage ---

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for data processing."""
    # Marker configuration
    MARKERS_Z_LEVEL = -0.3 + 0.1124
    QUAD_MARKER_POS = [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]]
    SQUARES_VERTICALLY = 3
    SQUARE_LENGTH = 0.500 / SQUARES_VERTICALLY
    CORNER_OFFSET = SQUARE_LENGTH * SQUARES_VERTICALLY / 2
    QUAD_ORDER = [3, 0, 1, 2]
    MARKER_ORDER = [[0, 2], [1, 2], [2, 2], [3, 2]]
    
    # Processing settings
    marker_unit = 4  # 0-3 for individual markers, 4 for average
    display_all = True
    rising_edge = 1
    offset_encoder = False
    replace_erronious_pos = False
    #time_correction = 628.49  # Time correction in ms for camera data, good for ChArUco
    time_correction = None  # Set to None for auto-correction, or specify manual value in ms
    save_altered_data = False
    filename = "ArUco Quad 7m run1" # Set to None if using multi-file processing, selected in main
    auto_correction_range = [-2000, 4000] # Good for ChArUco
    #auto_correction_range = [-6000, 0] # Good for ArUco
    #auto_correction_range = [-3000, 3000] # Good for ArUco 2m
    camera_pos_offset_ref = 1.3895  # Expected camera y-position offset


def get_marker_config() -> tuple[list, list]:
    """
    Get quad marker positions and rotations.
    
    Returns:
        Tuple of (quad_marker_pos, quad_marker_rot)
    """
    cfg = Config
    quad_marker_pos = [
        [cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[0]][0] + cfg.CORNER_OFFSET, 
         cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[0]][1] - cfg.CORNER_OFFSET, cfg.MARKERS_Z_LEVEL],
        [cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[1]][0] + cfg.CORNER_OFFSET, 
         cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[1]][1] + cfg.CORNER_OFFSET, cfg.MARKERS_Z_LEVEL],
        [cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[2]][0] - cfg.CORNER_OFFSET, 
         cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[2]][1] - cfg.CORNER_OFFSET, cfg.MARKERS_Z_LEVEL],
        [cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[3]][0] + cfg.CORNER_OFFSET, 
         cfg.QUAD_MARKER_POS[cfg.QUAD_ORDER[3]][1] - cfg.CORNER_OFFSET, cfg.MARKERS_Z_LEVEL]
    ]
    quad_marker_rot = [[0, 0, 180], [0, 0, 180], [0, 0, 180], [0, 0, 180]]
    return quad_marker_pos, quad_marker_rot


def apply_pose_corrections(processor: DataProcessor, quad_marker_pos: list, quad_marker_rot: list,
                           time_correction: float) -> list:
    """
    Apply pose corrections to fix solvePnP planar marker ambiguity.
    
    Also creates camera_pos_4 as average of all corrected markers.
    
    Args:
        processor: DataProcessor with loaded data
        quad_marker_pos: Marker world positions
        quad_marker_rot: Marker rotations
        time_correction: Time offset to apply
        
    Returns:
        List of modified camera data for saving
    """
    marker_order = Config.MARKER_ORDER
    modified_camera_data = []
    
    # Handle None time_correction (will be updated later with auto value)
    tc = time_correction if time_correction is not None else 0
    
    for data in processor.data:
        if data['mpu_unit'] != 4:
            continue
            
        average_pos = []
        average_rot = []
        modified_camera_data.append([
            data['recv_time'], 
            int(data['camera']['timestamp'] + round(tc)),
            [None, None, None, None], 
            [None, None, None, None]
        ])
        
        for marker_idx in range(4):
            key = f'camera_pos_{marker_idx}'
            if key not in data:
                continue
            
            if Config.replace_erronious_pos:
                (camera_pos, camera_rot), error_scores, corrected_pose = manta.alter_to_correct_pose(
                    data[key]['position'],
                    data[key]['rotation'],
                    [quad_marker_pos[marker_order[marker_idx][0]], 
                     quad_marker_rot[marker_order[marker_idx][1]]]
                )
                
                if corrected_pose:
                    # Apply rotation wraparound fix
                    for i in range(3):
                        if camera_rot[i] > 90:
                            camera_rot[i] -= 180
                        elif camera_rot[i] < -90:
                            camera_rot[i] += 180
                    data[key]['position'] = camera_pos
                    data[key]['rotation'] = camera_rot
                    modified_camera_data[-1][2][marker_idx] = camera_pos.tolist()
                    modified_camera_data[-1][3][marker_idx] = camera_rot.tolist()
                else:
                    print(f"Warning: did not correct pose for marker {marker_idx} at timestamp {data['camera']['timestamp']}")
                
                if abs(error_scores[0] - error_scores[1]) < 0.1:
                    print(f"Warning: Error values for marker {marker_idx} at timestamp {data['camera']['timestamp']} are too similar: {error_scores[0]} and {error_scores[1]}")
            
            average_pos.append(data[key]['position'])
            average_rot.append(data[key]['rotation'])
        
        # Create camera_pos_4 as average
        if len(average_pos) > 0:
            average_pos = np.mean(np.stack(average_pos, axis=0), axis=0)
            average_rot = np.mean(np.stack(average_rot, axis=0), axis=0)
            data['camera_pos_4'] = {
                'position': average_pos,
                'rotation': average_rot
            }
    
    return modified_camera_data


def extract_camera_encoder_data(processor: DataProcessor, marker_unit: int) -> dict:
    """
    Extract camera and encoder data for offset calculations.
    
    Args:
        processor: DataProcessor with loaded data
        marker_unit: Which marker to use (0-3, or 4 for average)
        
    Returns:
        Dictionary with camera_data, camera_timestamps, ref_data, ref_timestamps, etc.
    """
    camera_data = []
    camera_data_ext = []
    camera_timestamps = []
    ref_data = []
    ref_timestamps = []
    initial_timestamp = None
    
    for data in processor.data:
        if data['mpu_unit'] == 4:
            key = f'camera_pos_{marker_unit}'
            if key not in data:
                continue
            if initial_timestamp is None:
                initial_timestamp = data['camera']['timestamp']
            
            pos_y = data[key]['position'][1]
            if data['camera']['timestamp'] - initial_timestamp < 1000:
                camera_data.append(pos_y)
            else:
                camera_data_ext.append(pos_y)
            camera_timestamps.append(data['camera']['timestamp'])
            
        if data['mpu_unit'] == 0:
            if 'encoder' not in data:
                continue
            ref_data.append(data['encoder']['distance'])
            ref_timestamps.append(data['encoder']['timestamp'])
    
    return {
        'camera_data': np.array(camera_data) if camera_data else np.array([]),
        'camera_data_ext': np.array(camera_data_ext),
        'camera_timestamps': camera_timestamps,
        'ref_data': ref_data,
        'ref_timestamps': np.array(ref_timestamps),
        'initial_timestamp': initial_timestamp,
    }


def calculate_position_offset(camera_data: np.ndarray, expected_offset: float) -> float:
    """
    Calculate camera y-position offset from first 1000ms of data.
    
    Args:
        camera_data: Array of y-positions from first 1000ms
        expected_offset: Expected offset value for validation
        
    Returns:
        Position offset value
    """
    if len(camera_data) == 0:
        return expected_offset
    
    camera_pos_offset = np.mean(camera_data)
    print(f"\nCamera position offset: {camera_pos_offset:2.4f} +- {np.std(camera_data):.5f} (# vals: {len(camera_data)})")
    
    if abs(camera_pos_offset - expected_offset) > 0.1:
        print(f"Warning: Camera position offset is significantly different from expected value ({expected_offset}m). Using the expected value instead.")
        return expected_offset
    
    return camera_pos_offset


def find_optimal_time_offset(camera_data: np.ndarray, camera_timestamps: list,
                             ref_data: list, ref_timestamps: np.ndarray,
                             search_range: list, current_time_correction: float) -> dict:
    """
    Find optimal time offset by minimizing position difference std.
    
    Uses 8th order polynomial fit and detailed search for precision.
    
    Args:
        camera_data: All camera y-positions
        camera_timestamps: Camera timestamps
        ref_data: Encoder distances
        ref_timestamps: Encoder timestamps
        search_range: [min, max] search range in ms
        current_time_correction: Current time correction value
        
    Returns:
        Dictionary with best_offset, std_vals, y_curve, camera_timestamps_shifted, etc.
    """
    std_vals = []
    camera_timestamps_shifted = [(camera_data).tolist(), []]
    
    # First pass: find the best offset
    for i in range(search_range[0], search_range[1] + 10, 10):
        pos_diff = []
        for j in range(len(camera_data)):
            closest_index = np.argmin(np.abs(camera_timestamps[j] - ref_timestamps + i))
            closest_index = min(closest_index, len(ref_data) - 1)
            
            if abs(camera_timestamps[j] - ref_timestamps[closest_index] + i) > 500:
                continue
            pos_diff.append(-ref_data[closest_index] - camera_data[j])
        
        if len(pos_diff) > 10:
            pos_diff = np.array(pos_diff)
            pos_diff = pos_diff[np.abs(pos_diff - np.mean(pos_diff)) < 2 * np.std(pos_diff)]
            std_vals.append([i, np.std(pos_diff), np.mean(pos_diff)])
    
    std_vals = np.array(std_vals)
    
    # Find minimum and apply polynomial smoothing
    lowest_offset = np.argmin(std_vals[:, 1])
    if lowest_offset < 100:
        lowest_offset = 100
    elif lowest_offset > len(std_vals) - 100:
        lowest_offset = len(std_vals) - 100
    
    # 8th order polynomial fit
    std_range = std_vals[lowest_offset - 100:lowest_offset + 100]
    y_curve = np.poly1d(np.polyfit(std_range[:, 0], std_range[:, 1], 8))
    y_fit = y_curve(std_vals[:, 0])
    y_fit += 0.02 * (np.max(std_vals[:, 1]) - np.min(std_vals[:, 1]))
    y_fit[:lowest_offset - 100] = np.max(std_vals[:, 1])
    y_fit[lowest_offset + 100:] = np.max(std_vals[:, 1])
    
    best_offset_idx = np.argmin(y_fit)
    y_fit[best_offset_idx] = np.max(std_vals[:, 1])
    
    # Detailed search around best offset
    detailed_range = np.arange(
        std_vals[max(best_offset_idx - 10, 0)][0],
        std_vals[min(best_offset_idx + 10, len(std_vals) - 1)][0],
        0.01
    )
    best_offset_improved = detailed_range[np.argmin(y_curve(detailed_range))]
    
    print(f"\nBest offset for camera data: {best_offset_improved:.2f} ms (closest: {std_vals[best_offset_idx][0]} & std: {std_vals[best_offset_idx][1]:.4f})")
    
    # Build shifted timestamps for visualization (use best offset if manual is None)
    viz_time_correction = current_time_correction if current_time_correction is not None else best_offset_improved
    for i in range(search_range[0], search_range[1] + 10, 10):
        if i in range(int(viz_time_correction) // 10 * 10 - 400, 
                     int(viz_time_correction) // 10 * 10 + 450, 50):
            if i < int(viz_time_correction) - 15 or i > int(viz_time_correction) + 15:
                camera_timestamps_shifted.append([ts + i for ts in camera_timestamps])
    
    return {
        'best_offset': best_offset_improved,
        'best_offset_idx': best_offset_idx,
        'std_vals': std_vals,
        'y_fit': y_fit,
        'y_curve': y_curve,
        'camera_timestamps_shifted': np.array(camera_timestamps_shifted, dtype=object),
    }


def apply_time_corrections(processor: DataProcessor, camera_data: np.ndarray,
                           camera_timestamps: list, ref_data: list, ref_timestamps: np.ndarray,
                           time_correction: float, camera_pos_offset: float,
                           offset_result: dict, marker_unit: int) -> np.ndarray:
    """
    Apply time corrections and calculate position differences.
    
    Args:
        processor: DataProcessor with loaded data
        camera_data: All camera y-positions
        camera_timestamps: Camera timestamps  
        ref_data: Encoder distances
        ref_timestamps: Encoder timestamps
        time_correction: Time offset to apply
        camera_pos_offset: Position offset
        offset_result: Result from find_optimal_time_offset
        marker_unit: Which marker to use
        
    Returns:
        Array of position differences
    """
    # Calculate position differences
    pos_difference = []
    for i in range(len(camera_data)):
        closest_index = np.argmin(np.abs(camera_timestamps[i] - ref_timestamps + time_correction))
        closest_index = min(closest_index, len(ref_data) - 1)
        if abs(camera_timestamps[i] - ref_timestamps[closest_index] + time_correction) > 500 and len(pos_difference):
            pos_difference.append(pos_difference[-1])
        else:
            pos_difference.append(-ref_data[closest_index] - camera_data[i] + camera_pos_offset)
    
    pos_difference = np.array(pos_difference)
    
    # Add auto-diff entries to processor data
    for i, (offset_val, std_val, mean_val) in enumerate(offset_result['std_vals']):
        processor.data.append({
            'mpu_unit': 6,
            'auto-diff_vs_offset': {
                'timestamp': offset_val,
                'mean': mean_val,
                'std': [std_val, offset_result['y_fit'][i]]
            }
        })
    
    # Apply corrections to camera entries
    idx = 0
    ref_pos_diff = 0
    camera_timestamps_shifted = offset_result['camera_timestamps_shifted']
    
    for data in processor.data:
        if data['mpu_unit'] == 4:
            key = f'camera_pos_{marker_unit}'
            if key not in data:
                continue
            # Extract column idx from the shifted timestamps array
            col_data = [row[idx] if idx < len(row) else None for row in camera_timestamps_shifted]
            data[key]['timestamps_shifted'] = col_data
            if idx >= len(pos_difference):
                break
            if np.abs(pos_difference[idx] - np.mean(pos_difference)) > 2 * np.std(pos_difference) and idx > 0:
                data[key]['enc-cam_diff'] = ref_pos_diff
            else:
                data[key]['enc-cam_diff'] = pos_difference[idx]
                ref_pos_diff = pos_difference[idx]
            idx += 1
    
    return pos_difference


def calculate_time_offsets_by_position(processor: DataProcessor, marker_unit: int,
                                       camera_pos_offset: float, time_correction: float,
                                       rising_edge: int, offset_encoder: bool):
    """
    Calculate time offset for each camera entry based on position matching.
    
    Args:
        processor: DataProcessor with loaded data
        marker_unit: Which marker to use
        camera_pos_offset: Position offset
        time_correction: Time correction
        rising_edge: Direction flag
        offset_encoder: Whether encoder is offset
    """
    for data in processor.data:
        if data['mpu_unit'] != 4:
            continue
        
        key = f'camera_pos_{marker_unit}'
        if key not in data:
            continue
        
        position_val = data[key]['position'][1] - camera_pos_offset
        position_timestamp = data['camera']['timestamp'] + time_correction
        
        for data_ref in processor.data:
            if data_ref['mpu_unit'] != 0:
                continue
            if data_ref['encoder']['timestamp'] < position_timestamp - 2000:
                continue
            
            enc_dist = -data_ref['encoder']['distance']
            condition = enc_dist > position_val if rising_edge else enc_dist < position_val
            
            if condition:
                if abs(position_timestamp - data_ref['encoder']['timestamp']) > 1000:
                    # Search for crossing point
                    for data_ref_inner in processor.data:
                        if data_ref_inner['mpu_unit'] != 0:
                            continue
                        if data_ref_inner['encoder']['timestamp'] < position_timestamp - 2000:
                            continue
                        enc_dist_inner = -data_ref_inner['encoder']['distance']
                        inner_condition = enc_dist_inner < position_val if rising_edge else enc_dist_inner > position_val
                        if inner_condition:
                            data[key]['time_offset'] = min(max(-1000, position_timestamp - data_ref_inner['encoder']['timestamp']), 1000)
                            break
                        else:
                            data[key]['time_offset'] = 0
                else:
                    data[key]['time_offset'] = position_timestamp - data_ref['encoder']['timestamp']
                break
            else:
                data[key]['time_offset'] = 0
        
        if not offset_encoder:
            data[key]['position'][1] = data[key]['position'][1] - camera_pos_offset


def collect_all_camera_data(processor: DataProcessor, camera_pos_offset: float,
                            time_correction: float) -> tuple[list, list]:
    """
    Collect all camera positions and corrected timestamps for visualization.
    
    Returns:
        Tuple of (all_camera_data, all_camera_timestamps_corrected)
    """
    all_camera_data = [[], [], [], [], []]
    all_camera_timestamps_corrected = [[], [], [], [], []]
    
    for data in processor.data:
        if data['mpu_unit'] == 4:
            for i in range(5):
                key = f'camera_pos_{i}'
                if key not in data:
                    continue
                offset = 0 if i == 4 else camera_pos_offset
                all_camera_data[i].append(data[key]['position'][1] - offset)
                all_camera_timestamps_corrected[i].append(data['camera']['timestamp'] + time_correction)
    
    return all_camera_data, all_camera_timestamps_corrected


def compare_pressure_to_encoder(processor: DataProcessor):
    """Compare pressure depths to z-axis encoder values."""
    encoder_entries = [d for d in processor.data if d.get('mpu_unit') == 1]
    
    for data in processor.data:
        if data['mpu_unit'] == 0:
            closest = min(encoder_entries, key=lambda x: abs(x['encoder']['timestamp'] - data['pressure']['timestamp']))
            data['pressure']['depth0_diff'] = data['pressure']['depth0'] - closest['encoder']['distance']
            data['pressure']['depth1_diff'] = data['pressure']['depth1'] - closest['encoder']['distance']


def save_altered_recording(filepath: str, modified_camera_data: list):
    """
    Save altered camera data to a new file.
    
    Args:
        filepath: Original file path
        modified_camera_data: List of modified camera entries
    """
    altered_filepath = filepath.split('/')[0] + '/altered/' + filepath.split('/')[1]
    
    with open(altered_filepath, 'w', encoding='utf-8') as altered_file:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                altered_file.write(line)
                if '"mpu_unit": 4' in line:
                    line_json = json.loads(line)
                    line_json['mpu_unit'] = 5
                    
                    try:
                        idx = next(i for i, row in enumerate(modified_camera_data) 
                                  if row[0] == line_json['recv_time'])
                    except StopIteration:
                        print("Warning: No matching timestamp found for the camera data.")
                        return
                    
                    line_json['camera']['timestamp'] = modified_camera_data[idx][1]
                    for marker_idx in range(4):
                        key = f'camera_pos_{marker_idx}'
                        if key not in line_json:
                            if modified_camera_data[idx][2][marker_idx] is not None:
                                print(f"Warning: {key} not found in original but found in modified data.")
                                return
                        else:
                            line_json[key]['position'] = modified_camera_data[idx][2][marker_idx]
                            line_json[key]['rotation'] = modified_camera_data[idx][3][marker_idx]
                    
                    altered_file.write(json.dumps(line_json) + '\n')
    
    print(f"Altered data saved to {altered_filepath}")


def visualize_results(processor: DataProcessor, marker_unit: int, display_all: int,
                      ref_timestamps: np.ndarray, ref_data: list,
                      all_camera_timestamps: list, all_camera_data: list):
    """Run all visualizations."""
    print("\n--- Visualizing Data ---")
    
    if display_all:
        processor.visualize(mpu_units=[4], 
                           sensor_types=['camera_pos_0', 'camera_pos_1', 'camera_pos_2', 'camera_pos_3', 'camera_pos_4'], 
                           fields=['position'])
        processor.visualize(mpu_units=[4], 
                           sensor_types=['camera_pos_0', 'camera_pos_1', 'camera_pos_2', 'camera_pos_3', 'camera_pos_4'], 
                           fields=['rotation'])
        processor.visualize(mpu_units=[6], sensor_types=['auto-diff_vs_offset'], fields=['std'])
        processor.visualize(mpu_units=[0, 4], 
                           sensor_types=[f'camera_pos_{marker_unit}', 'encoder', 'camera', 'pressure'],
                           fields=['position', '-distance', 'time_offset', 'enc-cam_diff'])
        processor.visualize(mpu_units=[0], sensor_types=['pressure'], 
                           fields=['depth0_diff', 'depth1_diff', 'depth0', 'depth1'])
        processor.visualize(mpu_units=[0], sensor_types=['acceleration', 'gyroscope', 'integ_pos'], 
                           fields=['x', 'y', 'z'])
    
    processor.visualize(mpu_units=[4], sensor_types=[f'camera_pos_{marker_unit}'], 
                       fields=['timestamps_shifted'],
                       ref_timestamps=ref_timestamps, ref_data=-1 * np.array(ref_data),
                       all_camera_timestamps=all_camera_timestamps, all_camera_data=all_camera_data)
    
    plt.show(block=True)


if __name__ == "__main__":
    # File selection
    filenames = []
    #filenames.extend(["all_runs_ordered"])

    '''filenames.extend(["ChArUco Quad 2m run1", "ChArUco Quad 2m run2", "ChArUco Quad 2m run3"])
    filenames.extend(["ChArUco Quad 4.5m run1", "ChArUco Quad 4.5m run2", "ChArUco Quad 4.5m run3"])
    filenames.extend(["ChArUco Quad 7m run3", "ChArUco Quad 7m run2", "ChArUco Quad 7m run1"])

    filenames.extend(["ArUco Quad 2m run1", "ArUco Quad 2m run2", "ArUco Quad 2m run3"])
    filenames.extend(["ArUco Quad 4.5m run1", "ArUco Quad 4.5m run2", "ArUco Quad 4.5m run3"])
    filenames.extend(["ArUco Quad 7m run1", "ArUco Quad 7m run2", "ArUco Quad 7m run3"])

    filenames.extend(["ChArUco Single 2m run1", "ChArUco Single 2m run2", "ChArUco Single 2m run3"])
    filenames.extend(["ChArUco Single 4.5m run1", "ChArUco Single 4.5m run2", "ChArUco Single 4.5m run3"])
    filenames.extend(["ChArUco Single 7m run1", "ChArUco Single 7m run2", "ChArUco Single 7m run3"])

    filenames.extend(["ArUco Single 2m run1", "ArUco Single 2m run2", "ArUco Single 2m run3"])
    filenames.extend(["ArUco Single 4.5m run1", "ArUco Single 4.5m run2", "ArUco Single 4.5m run3"])
    filenames.extend(["ArUco Single 7m run1", "ArUco Single 7m run2", "ArUco Single 7m run3"])

    filenames.extend(["ChArUco Single 4.5-2m", "ChArUco Single 4.5-7m"])
    filenames.extend(["ArUco Single 2-4.5m", "ArUco Single 7-4.5m"])
    filenames.extend(["ChArUco Quad 3.8-4.5m", "ChArUco Quad 7-4.5m"])
    filenames.extend(["ArUco Quad 4.5-2m", "ArUco Quad 4.5-7m"])'''

    #filenames.extend(["ArUco Quad 4.5-7m"])
    filenames.extend(["ChArUco Quad 2m run2"])
    #filenames.extend(["altered/ChArUco Quad 7m run1"])

    # Load configuration
    cfg = Config
    
    # Load and process data
    if cfg.filename is not None:
        filepaths = [f"recordings/{cfg.filename}.json"]
    else:
        filepaths = [f"recordings/{name}.json" for name in filenames]
    processor = DataProcessor(filepaths)
    processor.load_data()

    # Process camera timestamps
    timestamp_lines = 0
    averaged_timestamps = processor.process_data_timings("result")
    for data in processor.data:
        if data['mpu_unit'] == 4:
            data['camera']['timestamp'] = averaged_timestamps[timestamp_lines]
            timestamp_lines += 1
    if timestamp_lines != len(averaged_timestamps):
        print("Warning: Mismatch in camera timestamps and averaged timestamps count. Stopping.")
        exit(1)

    # Get marker configuration
    quad_marker_pos, quad_marker_rot = get_marker_config()

    # Apply pose corrections
    modified_camera_data = apply_pose_corrections(
        processor, quad_marker_pos, quad_marker_rot, cfg.time_correction
    )

    # Extract camera and encoder data
    extracted = extract_camera_encoder_data(processor, cfg.marker_unit)
    
    if len(extracted['camera_data']) > 0:
        # Calculate position offset
        camera_pos_offset = calculate_position_offset(
            extracted['camera_data'], cfg.camera_pos_offset_ref
        )
        
        # Combine all camera data
        all_cam_data = np.append(extracted['camera_data'], extracted['camera_data_ext'])
        
        # Handle encoder offset if needed
        if cfg.offset_encoder:
            for data in processor.data:
                if data['mpu_unit'] == 0:
                    data['encoder']['distance'] -= camera_pos_offset
            all_cam_data -= camera_pos_offset
            camera_pos_offset = 0
        
        # Find optimal time offset
        offset_result = find_optimal_time_offset(
            all_cam_data, extracted['camera_timestamps'],
            extracted['ref_data'], extracted['ref_timestamps'],
            cfg.auto_correction_range, cfg.time_correction
        )
        
        # Use auto-corrected time offset if manual is None, otherwise use manual
        if cfg.time_correction is None:
            time_correction = offset_result['best_offset']
            print(f"Using auto-corrected time offset: {time_correction:.2f} ms")
        else:
            time_correction = cfg.time_correction
            print(f"Using manual time correction: {time_correction:.2f} ms (auto found: {offset_result['best_offset']:.2f} ms)")
        
        # Apply time corrections
        pos_difference = apply_time_corrections(
            processor, all_cam_data, extracted['camera_timestamps'],
            extracted['ref_data'], extracted['ref_timestamps'],
            time_correction, camera_pos_offset, offset_result, cfg.marker_unit
        )
        
        # Calculate time offsets by position matching
        calculate_time_offsets_by_position(
            processor, cfg.marker_unit, camera_pos_offset,
            time_correction, cfg.rising_edge, cfg.offset_encoder
        )
        
        # Collect all camera data for visualization
        all_camera_data, all_camera_timestamps_corrected = collect_all_camera_data(
            processor, camera_pos_offset, time_correction
        )
        
        # Save altered data if requested
        if len(filepaths) == 1 and 'altered' not in filepaths[0] and cfg.save_altered_data:
            save_altered_recording(filepaths[0], modified_camera_data)
    
    # Compare pressure to encoder
    compare_pressure_to_encoder(processor)
    
    # Extract all data
    extracted_data = processor.extract_all_data()
    
    # Visualize
    visualize_results(
        processor, cfg.marker_unit, cfg.display_all,
        extracted['ref_timestamps'], extracted['ref_data'],
        all_camera_timestamps_corrected, all_camera_data
    )