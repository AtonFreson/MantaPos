import os
import json
import glob

def process_file(filepath):
    records = []
    
    # First pass: gather offsets
    offsets = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("mpu_unit") == 3 and "pressure" in data:
                ts = data["pressure"].get("timestamp")
                if ts is not None:
                    o0 = data["pressure"].get("depth_offset0")
                    o1 = data["pressure"].get("depth_offset1")
                    if o0 is not None and o1 is not None:
                        offsets.append((ts, o0, o1))
    
    offsets.sort(key=lambda x: x[0])

    def get_offset_for_ts(target_ts):
        if not offsets:
            return 0, 0
        # Find most recent before target_ts
        valid_offsets = [o for o in offsets if o[0] <= target_ts]
        if valid_offsets:
            return valid_offsets[-1][1], valid_offsets[-1][2]
        # If none before, use the very first one
        return offsets[0][1], offsets[0][2]

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
                        o0, o1 = get_offset_for_ts(ts)
                        d0 = data["pressure"].get("depth0")
                        d1 = data["pressure"].get("depth1")
                        
                        val = {}
                        if d0 is not None: val["pressure_z0"] = -(d0 - o0)
                        if d1 is not None: val["pressure_z1"] = -(d1 - o1)
                        
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
                        records.append((ts, 'ref_pos_z1', -data["encoder"].get("distance")))
            elif unit == 2:
                if "encoder" in data:
                    ts = data["encoder"].get("timestamp")
                    if ts is not None:
                        records.append((ts, 'ref_pos_z2', -data["encoder"].get("distance")))
            elif unit == 4:
                if "camera" in data:
                    ts = data["camera"].get("timestamp")
                    if ts is not None:
                        for i in range(0, 4):
                            cam_key = f"camera_pos_{i}"
                            if cam_key in data:
                                # X seems to be flipped in the camera data, can possibly flip it back.
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