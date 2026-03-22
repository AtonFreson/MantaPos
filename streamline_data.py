import os
import json
import glob

def process_file(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            unit = data.get("mpu_unit")
            if unit == 0:
                if "pressure" in data:
                    ts = data["pressure"].get("timestamp")
                    if ts is not None:
                        val = {"depth0": data["pressure"].get("depth0"), "depth1": data["pressure"].get("depth1")}
                        records.append((ts, 'pressure_bottom', val))
                if "encoder" in data:
                    ts = data["encoder"].get("timestamp")
                    if ts is not None:
                        records.append((ts, 'ref_pos_track', data["encoder"].get("distance")))
                if "imu" in data:
                    ts = data["imu"].get("timestamp")
                    if ts is not None:
                        val = {
                            "acceleration": data["imu"].get("acceleration"),
                            "gyroscope": data["imu"].get("gyroscope")
                        }
                        records.append((ts, 'imu', val))
            elif unit == 1:
                if "encoder" in data:
                    ts = data["encoder"].get("timestamp")
                    if ts is not None:
                        records.append((ts, 'ref_pos_z1', data["encoder"].get("distance")))
            elif unit == 2:
                if "encoder" in data:
                    ts = data["encoder"].get("timestamp")
                    if ts is not None:
                        records.append((ts, 'ref_pos_z2', data["encoder"].get("distance")))
            elif unit == 3:
                if "pressure" in data:
                    ts = data["pressure"].get("timestamp")
                    if ts is not None:
                        val = {"depth_offset0": data["pressure"].get("depth_offset0"), "depth_offset1": data["pressure"].get("depth_offset1")}
                        records.append((ts, 'pressure_top', val))
            elif unit == 4:
                if "camera" in data:
                    ts = data["camera"].get("timestamp")
                    if ts is not None:
                        for i in range(0, 4):
                            cam_key = f"camera_pos_{i}"
                            if cam_key in data:
                                records.append((ts, cam_key, data[cam_key]))
                                
    records.sort(key=lambda x: x[0])
    return records

def streamline_data(input_dir, output_file):
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    all_files_records = []
    for filepath in json_files:
        records = process_file(filepath)
        if records:
            all_files_records.append((filepath, records))
            
    # Sort files by their first timestamp
    all_files_records.sort(key=lambda x: x[1][0][0])
    
    last_global_ts = -1
    
    with open(output_file, 'w') as out_f:
        for filepath, records in all_files_records:
            filename = os.path.basename(filepath)
            filename_no_ext = os.path.splitext(filename)[0]
            
            first_ts = records[0][0]
            if last_global_ts != -1 and first_ts <= last_global_ts:
                raise ValueError(f"Overlap detected! File {filename} starts at {first_ts} which is <= previous end {last_global_ts}")
                
            out_f.write(f"### '{filename_no_ext}' ###\n")
            for ts, dtype, dval in records:
                # Format: {timestamp} - '{data_type}': {data_value}
                dtype_str = f"'{dtype}'"
                out_f.write(f"{ts} - {dtype_str:<17}: {dval}\n")
                
            last_global_ts = records[-1][0]
            print(f"Processed {filename}: {len(records)} records")

if __name__ == "__main__":
    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
    output_path = os.path.join(os.path.dirname(__file__), "MantaPos_data.json")
    streamline_data(recordings_dir, output_path)
    print(f"Data successfully streamlined into {output_path}")