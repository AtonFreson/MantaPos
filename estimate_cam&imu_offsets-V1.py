import os
import ast
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.stats as stats

def load_data_stream(filepath):
    """
    Loads the streamlined MantaPos_data.json file.
    Returns a dictionary of runs, each containing its datapoints.
    """
    runs = {}
    current_run = "Unknown"
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('###'):
                current_run = line.strip('#').strip().strip("'")
                runs[current_run] = []
                continue
            
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
                
            runs[current_run].append({
                'timestamp': timestamp,
                'type': data_type,
                'data': data_val
            })
            
    return runs

def process_runs(runs_dict):
    results = {'ArUco': {'imu': [], 'cam': [], 'ref': []}, 
               'ChArUco': {'imu': [], 'cam': [], 'ref': []}}

    track_width = 3.1305 # horizontal distance between depth sensors
    
    for run_name, data_stream in runs_dict.items():
        if "ChArUco" in run_name:
            tag_type = "ChArUco"
        elif "ArUco" in run_name:
            tag_type = "ArUco"
        else:
            continue
            
        latest_ref_z1 = None
        latest_ref_z2 = None
        
        run_imu = []
        run_cam = []
        run_ref_pitch = []
        
        for dp in data_stream:
            current_time = dp['timestamp']
            data_type = dp['type']
            data_val = dp['data']
            
            if data_type == 'imu':
                accel_data = data_val.get('acceleration', {})
                accel_vec = [accel_data.get('x', 0), accel_data.get('y', 0), accel_data.get('z', -9.81)]
                run_imu.append(accel_vec)
                
            elif data_type == 'ref_pos_z1':
                latest_ref_z1 = data_val
            elif data_type == 'ref_pos_z2':
                latest_ref_z2 = data_val
            elif data_type == 'ref_pos_track':
                if latest_ref_z1 is not None and latest_ref_z2 is not None:
                    opp = latest_ref_z1 - latest_ref_z2
                    # The angle formula from kalman filter script
                    camera_rot_x = np.arctan(opp / track_width)
                    run_ref_pitch.append(camera_rot_x)
                    
            elif data_type.startswith('camera_pos_'):
                cam_rot = data_val.get('rotation')
                if cam_rot is not None:
                    run_cam.append(cam_rot)
                    
        # Process this run's IMU to find the average
        if len(run_imu) > 0:
            mean_accel = np.mean(run_imu, axis=0)
            results[tag_type]['imu'].append(mean_accel)
            
        if len(run_ref_pitch) > 0:
            mean_ref_pitch = np.mean(run_ref_pitch)
            results[tag_type]['ref'].append(mean_ref_pitch)
            
        if len(run_cam) > 0:
            mean_cam_rot = np.mean(run_cam, axis=0)
            results[tag_type]['cam'].append(mean_cam_rot)
            
    return results

def mean_conf_interval(data, confidence=0.95):
    """Calculates the mean and the margin of error (half-width of CI)."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def calculate_offsets():
    filepath = os.path.join(os.path.dirname(__file__), 'MantaPos_data.json')
    print("Loading data stream...")
    runs_dict = load_data_stream(filepath)
    print(f"Loaded {len(runs_dict)} runs.")
    
    results = process_runs(runs_dict)
    
    for tag_type in ['ChArUco', 'ArUco']:
        print(f"\n{'='*40}")
        print(f"       ANALYSIS FOR {tag_type.upper()}")
        print(f"{'='*40}")
        
        # Calculate true pitch from the reference track data
        ref_pitches = results[tag_type]['ref']
        if not ref_pitches:
            print("No reference data available.")
            continue
            
        true_pitch_rad, true_pitch_h_rad = mean_conf_interval(ref_pitches)
        true_pitch_deg = np.rad2deg(true_pitch_rad)
        true_pitch_h_deg = np.rad2deg(true_pitch_h_rad)
        print(f"True Track Pitch (from pressure): {true_pitch_deg:+.3f}° ± {true_pitch_h_deg:.3f}°\n")

        # CAMERA OFFSET
        cam_rots = results[tag_type]['cam']
        if cam_rots:
            mean_cam, cam_h = mean_conf_interval(cam_rots)
            
            print(f"--- CAMERA MOUNTING OFFSET ---")
            print(f"Average Raw Camera Rotation : X={mean_cam[0]:+.3f}° ± {cam_h[0]:.3f}°, Y={mean_cam[1]:+.3f}° ± {cam_h[1]:.3f}°, Z={mean_cam[2]:+.3f}° ± {cam_h[2]:.3f}°")
            
            offset_x = mean_cam[0] - true_pitch_deg
            offset_y = mean_cam[1] - 0.0 
            offset_z = mean_cam[2] - 0.0 
            
            # Error propagation: sqrt(e1^2 + e2^2)
            offset_x_h = np.sqrt(cam_h[0]**2 + true_pitch_h_deg**2)
            
            print(f"Mounting Offset (Cam - True): X={offset_x:+.3f}° ± {offset_x_h:.3f}°, Y={offset_y:+.3f}° ± {cam_h[1]:.3f}°, Z={offset_z:+.3f}° ± {cam_h[2]:.3f}°\n")
            
        # IMU OFFSET
        imums = results[tag_type]['imu']
        if imums:
            run_imu_pitches = []
            run_imu_rolls = []
            
            # Compute pitch/roll offset independently per run to get proper CI
            for mean_accel in imums:
                g_vec = mean_accel / np.linalg.norm(mean_accel) * 9.81
                R_track_pitch_inv = R.from_euler('x', -true_pitch_deg, degrees=True).as_matrix()
                flat_accel = R_track_pitch_inv.dot(g_vec)
                
                flat_accel_norm = flat_accel / np.linalg.norm(flat_accel)
                
                imu_pitch_offset = np.arcsin(flat_accel_norm[0]) 
                imu_roll_offset = -np.arcsin(flat_accel_norm[1] / np.cos(imu_pitch_offset))
                
                run_imu_pitches.append(np.rad2deg(imu_pitch_offset))
                run_imu_rolls.append(np.rad2deg(imu_roll_offset))
                
            mean_accel, accel_h = mean_conf_interval(imums)
                
            m_pitch, h_pitch = mean_conf_interval(run_imu_pitches)
            m_roll, h_roll = mean_conf_interval(run_imu_rolls)
            
            print(f"--- IMU MOUNTING OFFSET ---")
            print(f"Average Raw Accelerometer : X={mean_accel[0]:+.3f} ± {accel_h[0]:.3f}, Y={mean_accel[1]:+.3f} ± {accel_h[1]:.3f}, Z={mean_accel[2]:+.3f} ± {accel_h[2]:.3f}")
            print(f"Mounting Offset  : Pitch(Rot Y)= {m_pitch:+.3f}° ± {h_pitch:.3f}°, Roll(Rot X)= {m_roll:+.3f}° ± {h_roll:.3f}°, Yaw(Rot Z)=Unobservable")
            
        print("\n")

if __name__ == "__main__":
    calculate_offsets()
