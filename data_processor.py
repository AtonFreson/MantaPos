import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class DataProcessor:
    """
    A class to process and align various sensor data from JSON recordings.
    Supports data extraction, alignment, visualization, and export.
    """

    def __init__(self, file_path=None):
        """Initialize the DataProcessor with an optional file path."""
        self.file_path = file_path
        self.data = None
        self.extracted_data = {}
        self.aligned_data = None

    def load_data(self, file_path=None):
        """Load JSON data from file."""
        if file_path:
            self.file_path = file_path

        if not self.file_path:
            raise ValueError("File path not provided.")

        # Check if file exists before trying to open
        if not os.path.exists(self.file_path):
             raise FileNotFoundError(f"Error: File not found at {self.file_path}")

        try:
            with open(self.file_path, 'r') as f:
                self.data = [json.loads(line) for line in f]
            print(f"Loaded {len(self.data)} records from {self.file_path}")
            return True
        except json.JSONDecodeError as e:
            # Try to find the line number
            error_line = -1
            try:
                with open(self.file_path, 'r') as f_err:
                    for i, _ in enumerate(f_err):
                        if i + 1 == e.lineno:
                            break
                    error_line = e.lineno
            except Exception:
                pass # Ignore errors trying to find the line
            if error_line > 0:
                 raise ValueError(f"Error decoding JSON in file '{self.file_path}' at line {error_line}: {e}")
            else:
                 raise ValueError(f"Error decoding JSON in file '{self.file_path}': {e}")
        except Exception as e: # Catch other potential file reading errors
             raise IOError(f"Error reading file '{self.file_path}': {e}")


    def extract_data_types(self):
        """
        Extract and categorize all available data types from the loaded JSON.
        Returns a dictionary of data types and their occurrences.
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")

        data_types = defaultdict(int)

        # Analyze the structure of the JSON data
        for item in self.data:
            mpu_unit = item.get('mpu_unit', 'N/A')
            prefix = f"mpu{mpu_unit}." if mpu_unit != 'N/A' else ""

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
                    data_types[f"{prefix}{key}"] += 1 # Should not happen based on sample, but good practice

        # Sort for readability
        return dict(sorted(data_types.items()))

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

                        # Append all key-value pairs from the dictionary
                        for subkey, subvalue in top_value.items():
                            if subkey not in sensor_data:
                                sensor_data[subkey] = []
                            sensor_data[subkey].append(subvalue)

                # Handle potential non-dictionary top-level values if necessary
                # else:
                #     # Example: if a sensor value was directly at the top level
                #     sensor_type = top_key
                #     # Decide how to handle timestamping if it's not in a dict
                #     # This part depends on the exact data format if it differs from sample
                #     pass

        # Convert all collected data lists to numpy arrays for easier processing
        self._convert_lists_to_arrays()

        # Optional: Print structure of extracted data for verification
        # import pprint
        # pprint.pprint(self.extracted_data)

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
        """Export aligned data to pandas DataFrame."""
        if not self.aligned_data:
            raise ValueError("No aligned data available. Call align_data() first.")
        if 'reference' not in self.aligned_data or 'timestamps' not in self.aligned_data['reference']:
             raise ValueError("Aligned data is missing reference timestamps.")


        # Create a dictionary for the DataFrame
        df_dict = {'timestamp': self.aligned_data['reference']['timestamps']}

        # Add all aligned data fields
        for mpu_number, mpu_data in self.aligned_data.items():
            if mpu_number == 'reference':
                continue

            for sensor_type, sensor_data in mpu_data.items():
                for field_name, field_values in sensor_data.items():
                    # Create a descriptive column name
                    column_name = f"mpu{mpu_number}_{sensor_type}_{field_name}"
                    # Handle potential duplicate column names if needed, though unlikely with this naming
                    if column_name in df_dict:
                         print(f"Warning: Duplicate column name generated: {column_name}. Overwriting.")
                    df_dict[column_name] = field_values

        # Create DataFrame
        try:
             df = pd.DataFrame(df_dict)
             return df
        except ValueError as e:
             print("\nError creating DataFrame. Check for unequal array lengths:")
             for name, data in df_dict.items():
                  print(f"  Column '{name}': Length {len(data)}")
             raise ValueError(f"Could not create DataFrame: {e}")


    def visualize(self, sensor_types=None, mpu_units=None, fields=None, figsize=(15, 5), sharex=True):
        """
        Visualize the aligned sensor data, plotting each field on its own subplot.

        Parameters:
        - sensor_types: List of sensor types to plot (e.g., ['pressure', 'acceleration']). Default: all aligned.
        - mpu_units: List of MPU units to plot (e.g., [0, 1]). Default: all aligned.
        - fields: Optional: List of specific field names to plot (e.g., ['depth0', 'x']). Default: all aligned fields for the selected sensors/MPUs.
        - figsize: Figure size (width, height) per subplot in inches.
        - sharex: Whether subplots should share the x-axis (default: True).
        """
        if not self.aligned_data:
            raise ValueError("No aligned data available. Call align_data() first.")

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
                    if field not in self.aligned_data[mpu][sensor]:
                         # print(f"Warning: Field {field} not found for MPU {mpu}, Sensor {sensor}.")
                         continue

                    field_data = self.aligned_data[mpu][sensor][field]
                    # Ensure data has same length as timestamps
                    if len(field_data) == len(timestamps_plot):
                         all_plot_items.append((mpu, sensor, field, field_data))
                    else:
                         print(f"Warning: Skipping plot for mpu{mpu}_{sensor}_{field}. Length mismatch (Data: {len(field_data)}, Timestamps: {len(timestamps_plot)}).")


        if not all_plot_items:
            print("No data found to plot based on the specified filters.")
            return

        # Group items by field name for plotting on separate axes
        plots_by_field = defaultdict(list)
        for mpu, sensor, field, data in all_plot_items:
             plots_by_field[field].append({'mpu': mpu, 'sensor': sensor, 'data': data})

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

    def print_aligned_data(self, sensor_types=None, mpu_units=None, num_rows=10):
        """Print a preview of the aligned data DataFrame to the console."""
        if not self.aligned_data:
            print("No aligned data available. Call align_data() first.")
            return

        try:
            df = self.export_to_pandas()
        except Exception as e:
            print(f"Error exporting data to DataFrame for printing: {e}")
            return

        # Filter columns based on sensor_types and mpu_units if specified
        cols_to_show = ['timestamp']
        available_mpus = [mpu for mpu in self.aligned_data.keys() if mpu != 'reference']
        show_mpus = mpu_units if mpu_units is not None else available_mpus

        all_sensors = set()
        for mpu in show_mpus:
             if mpu in self.aligned_data:
                 all_sensors.update(self.aligned_data[mpu].keys())
        show_sensors = sensor_types if sensor_types is not None else list(all_sensors)

        for mpu in show_mpus:
            for sensor in show_sensors:
                if mpu in self.aligned_data and sensor in self.aligned_data[mpu]:
                    for field in self.aligned_data[mpu][sensor].keys():
                         col_name = f"mpu{mpu}_{sensor}_{field}"
                         if col_name in df.columns:
                              cols_to_show.append(col_name)

        # Print the head of the filtered DataFrame
        if len(cols_to_show) > 1:
            with pd.option_context('display.max_rows', num_rows,
                                    'display.max_columns', None, # Show all selected columns
                                    'display.width', 1000): # Adjust width for wider output
                                    print(df[cols_to_show].head(num_rows))
        else:
            print("No columns match the specified filters.")

        print(f"\nShowing first {min(num_rows, len(df))} rows.")
        print(f"Total rows: {len(df)}, Total columns in full aligned data: {len(df.columns)}")


# --- Example Usage ---
if __name__ == "__main__":
    dummy_file_path = "recordings/ArUco Quad 4.5-7m.json"
    # Copy the content from the user prompt into the dummy file
    
    # --- Run the Processor ---
    processor = DataProcessor(dummy_file_path)
    processor.load_data()

    # Print available data types (more detailed now)
    print("\n--- Available Data Types ---")
    data_types = processor.extract_data_types()
    # Use pprint for better readability of nested structures
    import pprint
    pprint.pprint(data_types)

    extracted = processor.extract_all_data()

    print("\n--- Aligning Data ---")
    aligned_data = processor.align_data(reference_sensor='encoder', reference_mpu=1)

    # Print preview of aligned data
    print("\n--- Aligned Data Preview ---")
    processor.print_aligned_data(sensor_types=['pressure', 'acceleration'], mpu_units=[0, 1], num_rows=5)

    # Visualize specific aligned data
    # Plot acceleration (x, y, z) from MPU 0 and pressure depth0 from MPU 0 & 3
    print("\n--- Visualizing Data ---")
    processor.visualize(mpu_units=[0], sensor_types=['acceleration'], fields=['x', 'y', 'z'])
    processor.visualize(mpu_units=[0], sensor_types=['gyroscope'], fields=['x', 'y', 'z'])
    processor.visualize(mpu_units=[0, 3], sensor_types=['pressure'], fields=['depth0'])
    # Visualize temperature from all MPUs that have it
    processor.visualize(sensor_types=['temperature'])