import json
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

from mantaPosLib import MantaUKF, global_reference_pos
from visualise_ukf import visualize_ukf_results

def load_data_stream(filepath, target_runs=None):
    """
    Loads the streamlined MantaPos_data.json file, optionally filtered by specific run names.
    """
    all_data = []
    current_run = None
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('###'):
                # Extract run name from: ### 'ChArUco Single 2m run1' ###
                current_run = line.strip('#').strip().strip("'")
                continue
            
            if target_runs is not None and current_run not in target_runs:
                continue
            
            # Line format: timestamp - 'type' : value
            parts = line.split(' - ', 1)
            if len(parts) != 2:
                continue
                
            timestamp = int(parts[0])
            type_val_split = parts[1].split(':', 1)
            
            if len(type_val_split) != 2:
                continue
                
            data_type = type_val_split[0].strip().strip("'")
            data_val_str = type_val_split[1].strip()
            
            try:
                data_val = ast.literal_eval(data_val_str)
            except (ValueError, SyntaxError):
                continue
                
            all_data.append({
                'timestamp': timestamp,
                'type': data_type,
                'data': data_val
            })
            
    return all_data

def run_kalman_filter(filepath, target_runs=None):
    print(f"Loading data stream... Target runs: {target_runs if target_runs else 'All'}")
    # 4. Practical Implementation Details: Continuous Timeline
    data_stream = load_data_stream(filepath, target_runs)
    print(f"Loaded {len(data_stream)} datapoints.")
    
    # Initialize the UKF
    ukf = MantaUKF()
    
    last_time = None
    last_accel_vec = np.array([0.0, 0.0, -9.81])
    
    ukf_data = []
    ref_data = []
    camera_points_by_time = {}
    
    latest_ref_z1 = None
    latest_ref_z2 = None
    
    # Process each asynchronous data point
    for dp in data_stream:
        current_time = dp['timestamp']
        data_type = dp['type']
        data_val = dp['data']
        
        if last_time is None:
            last_time = current_time
            continue
            
        dt = (current_time - last_time) / 1000.0  # timestamp is ms
        
        if dt > 0:
            # 1. Kinematic Prediction step
            # Predict kinematic state forward in time using the prior acceleration estimates
            ukf.predict(dt, last_accel_vec)
            
        if data_type == 'imu':
            accel_data = data_val.get('acceleration', {})
            accel_vec = np.array([accel_data.get('x', 0), accel_data.get('y', 0), accel_data.get('z', -9.81)])
            last_accel_vec = accel_vec  # Update last known acceleration
            ukf.update_accelerometer(accel_vec)
            
        elif data_type == 'pressure_bottom':
            pressure_z0 = data_val.get('pressure_z0')
            pressure_z1 = data_val.get('pressure_z1')
            if pressure_z0 is not None and pressure_z1 is not None:
                ukf.update_pressure([pressure_z0, pressure_z1])
                
        elif data_type.startswith('camera_pos_'):
            camera_pos = data_val.get('position')
            if camera_pos is not None:
                ukf.update_camera(np.array(camera_pos))
                # MantaPos outputs X, Y, Z. Force X=-1.5406 for 2d-on-3d alignment
                if current_time not in camera_points_by_time:
                    camera_points_by_time[current_time] = []
                camera_points_by_time[current_time].append([-1.5406, camera_pos[1], camera_pos[2]])
        
        elif data_type == 'ref_pos_z1':
            latest_ref_z1 = data_val
        elif data_type == 'ref_pos_z2':
            latest_ref_z2 = data_val
        elif data_type == 'ref_pos_track':
            if latest_ref_z1 is not None and latest_ref_z2 is not None:
                # Custom interpolation for reference position based on track distance and depths
                track_width = 3.1305 # Horizontal distance between the two rigid rail rails
                frame_y_pos_offset = 0.196 # Minimum frame offset from the main depth sensor to the camera.
                
                # Use the latest depth readings to determine the Z position of the track, and interpolate Y based on the encoder distance along the track.
                z0 = latest_ref_z1
                z1 = latest_ref_z2
                
                # Constants for the frame pool setup, in meters.
                adj = track_width # Horizontal distance between the depth sensors.
                #frame_z_pos_offset = 0.069 # Vertical frame offset from the main depth sensor to the camera.

                camera_x_offset = -1.5406 # Offset of the camera from the center of the pool in the x-direction.
                camera_y_offset = 1.3757 # Maximum zeroing offset of the camera from the center of the pool in the y-direction.
                camera_z_offset = -0.1186 + 0.225 - 0.187 # Offset of the camera at the zeroing position at the top of the pool in the z-direction.
                
                frame_pos = data_val + frame_y_pos_offset
                # Determine y position based on the frame position, where frame_pose makes up the hypotenuse of a right triangle.
                opp = z0 - z1
                hyp = np.sqrt((opp**2) + (adj**2))

                x = camera_x_offset
                y = -frame_pos/hyp * adj + camera_y_offset
                z = z0 - opp * frame_pos/hyp + camera_z_offset

                # Camera rotation around y-axis based on right triangle. Assume the camera is level otherwise.
                camera_rot_x = np.arctan(opp/adj)

                ref_pos = np.array([x, y, z])
                ref_rot = np.array([camera_rot_x, 0, 0])
                

                ref_data.append({
                    'timestamp': current_time,
                    'position': ref_pos.tolist(),
                    'rotation': ref_rot.tolist()
                })

        # Record actual estimated position from the Kalman Filter
        if dt > 0:
            ukf_data.append({
                'timestamp': current_time,
                'position': [-1.5406, ukf.x[0], ukf.x[1]]
            })
            
        last_time = current_time
        
        if len(ukf_data) % 1000 == 0 and dt > 0:
            print(f"Processed {len(ukf_data)} points...")
        
    print("Filter execution complete!")
    print(f"Processed {len(ukf_data)} state updates.")
    
    camera_avg_data = []
    for ts, pos_list in camera_points_by_time.items():
        if pos_list:
            avg_pos = np.mean(pos_list, axis=0).tolist()
            camera_avg_data.append({'timestamp': ts, 'position': avg_pos})
    
    return ukf_data, ref_data, camera_avg_data

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), 'MantaPos_data.json')
    target_runs = [
        'ChArUco Quad 4.5m run2',
        'ChArUco Quad 4.5m run3'
    ]
    ukf_data, ref_data, camera_data = run_kalman_filter(filepath, target_runs)
    
    print("Visualizing results...")
    # Optional parameter: use_2d=True to ignore X-axis when there is no variability
    visualize_ukf_results(ukf_data, ref_data, camera_data, use_2d=True)

