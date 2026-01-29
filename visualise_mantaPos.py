import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

# Configuration
DATA_DIRECTORY = 'recordings'
DATA_FILE = 'ChArUco Quad 7m run2'
display_only_avg = False
display_gradient = False
PLAYBACK_SPEED = 4  # 4x speed
FRAME_INTERVAL_MS = 50  # Base frame interval in milliseconds
GLOBAL_POS_Y_OFFSET = 0.25  # Y offset for global position in meters, if it's offset. to help with visualisation

def load_and_extract_data(file_path):
    """Load data from JSON and extract the required position data."""
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in file '{file_path}': {e}")

    # Initialize lists to store different types of position data
    pressure_depth_data = []  # mpu_unit 0 - pressure - depth0 and depth1
    global_pos_data = []      # mpu_unit 4 - global_pos - position
    camera_pos_data = [[], [], [], []]  # mpu_unit 4 - camera_pos_X - position
    camera_avg_data = []      # Average of all available camera positions 
    
    # Extract data points
    for item in data:
        # Extract pressure depth data
        if item.get('mpu_unit') == 0 and 'pressure' in item and 'depth0' in item['pressure'] and 'depth1' in item['pressure']:
            depth0 = item['pressure']['depth0']
            depth1 = item['pressure']['depth1']
            # We'll use depth0 as x, depth1 as y, and 0 as z for pressure data
            pressure_depth_data.append({
                'timestamp': item['pressure'].get('timestamp', 0),
                'position': [1.5406,0,-(depth0+depth1)/2-0.33]  # z is set to 0 for visualization
            })
        
        # Extract position data from mpu_unit 4
        if item.get('mpu_unit') == 4:
            if 'global_pos' in item and 'position' in item['global_pos']:
                pos = item['global_pos']['position']
                global_pos_data.append({
                    'timestamp': item['global_pos'].get('timestamp', 0),
                    'position': [pos[0], pos[1] + GLOBAL_POS_Y_OFFSET, pos[2]]
                })
            avg_value = []
            timestamp = None
            for camera_name in ['camera_pos_0', 'camera_pos_1', 'camera_pos_2', 'camera_pos_3']:
                if camera_name in item and 'position' in item[camera_name]:
                    if item[camera_name]['position'][0] < 1 or item[camera_name]['position'][0] > 2:
                        continue
                    camera_pos_data[int(camera_name[-1])].append({
                        'timestamp': item['camera'].get('timestamp', 0),
                        'position': item[camera_name]['position']
                    })
                    avg_value.append(item[camera_name]['position'])
                    timestamp = item['camera'].get('timestamp', 0)
                
            if avg_value and timestamp:
                camera_avg_data.append({
                    'timestamp': timestamp,
                    'position': np.mean(avg_value, axis=0).tolist()
                })
    
    # Check if data was found
    if not any([pressure_depth_data, global_pos_data, *camera_pos_data, camera_avg_data]):
        raise ValueError("No valid position data found in the JSON file")
        
    return (pressure_depth_data, global_pos_data, *camera_pos_data, camera_avg_data)

def extract_position_arrays(data_list):
    """Extract x, y, z arrays from a list of position data."""
    if not data_list:
        return [], [], []
    
    x = []
    y = []
    z = []
    
    for item in data_list:
        position = item['position']
        if len(position) >= 3:
            x.append(position[0])
            y.append(position[1])
            z.append(position[2])
    
    return x, y, z

def set_axes_equal(ax):
    """Set equal scaling for a 3D plot while minimizing whitespace."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # Use set_box_aspect to set aspect ratio proportional to data ranges
    # This avoids large whitespace while maintaining proper proportions
    ax.set_box_aspect([x_range, y_range, z_range])

def extract_position_arrays_with_timestamps(data_list):
    """Extract x, y, z arrays and timestamps from a list of position data."""
    if not data_list:
        return [], [], [], []
    
    x = []
    y = []
    z = []
    timestamps = []
    
    for item in data_list:
        position = item['position']
        if len(position) >= 3:
            x.append(position[0])
            y.append(position[1])
            z.append(position[2])
            timestamps.append(item['timestamp'])
    
    return x, y, z, timestamps


def find_position_at_time(x_arr, y_arr, z_arr, timestamps, target_time):
    """
    Find the position at a given time using the closest earlier timestamp.
    Returns (x, y, z) or (None, None, None) if no data available.
    """
    if not timestamps:
        return None, None, None
    
    # Find the closest timestamp that is <= target_time
    best_idx = None
    for i, ts in enumerate(timestamps):
        if ts <= target_time:
            best_idx = i
        else:
            break
    
    if best_idx is None:
        # Target time is before all data, use first point
        if timestamps[0] - target_time < 500:  # Within 500ms tolerance
            return x_arr[0], y_arr[0], z_arr[0]
        return None, None, None
    
    return x_arr[best_idx], y_arr[best_idx], z_arr[best_idx]


def find_trail_at_time(x_arr, y_arr, z_arr, timestamps, target_time, trail_duration_ms=2000):
    """
    Find trail points within trail_duration_ms before target_time.
    """
    if not timestamps:
        return [], [], []
    
    trail_x, trail_y, trail_z = [], [], []
    start_time = target_time - trail_duration_ms
    
    for i, ts in enumerate(timestamps):
        if start_time <= ts <= target_time:
            trail_x.append(x_arr[i])
            trail_y.append(y_arr[i])
            trail_z.append(z_arr[i])
    
    return trail_x, trail_y, trail_z


def visualize_3d_comparison(original_data: tuple, corrected_data: tuple, title_prefix: str = ""):
    """
    Create an animated side-by-side 3D comparison visualization.
    Left panel shows original data, right panel shows corrected data.
    Both panels are synchronized and animate together.
    
    Args:
        original_data: Tuple from load_and_extract_data for original recording
        corrected_data: Tuple from load_and_extract_data (or extract_data_from_memory) for corrected data
        title_prefix: Optional prefix for window title
    """
    # Unpack data tuples
    (orig_pressure, orig_global, orig_cam0, orig_cam1, orig_cam2, orig_cam3, orig_avg) = original_data
    (corr_pressure, corr_global, corr_cam0, corr_cam1, corr_cam2, corr_cam3, corr_avg) = corrected_data
    
    # Extract position arrays with timestamps for all data
    def extract_with_ts(data_list):
        if not data_list:
            return [], [], [], []
        x, y, z, ts = [], [], [], []
        for item in data_list:
            pos = item['position']
            if len(pos) >= 3:
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                ts.append(item.get('timestamp', 0))
        return x, y, z, ts
    
    # Extract all data with timestamps
    orig_data_arrays = {
        'pressure': extract_with_ts(orig_pressure),
        'global': extract_with_ts(orig_global),
        'cam0': extract_with_ts(orig_cam0),
        'cam1': extract_with_ts(orig_cam1),
        'cam2': extract_with_ts(orig_cam2),
        'cam3': extract_with_ts(orig_cam3),
        'avg': extract_with_ts(orig_avg),
    }
    corr_data_arrays = {
        'pressure': extract_with_ts(corr_pressure),
        'global': extract_with_ts(corr_global),
        'cam0': extract_with_ts(corr_cam0),
        'cam1': extract_with_ts(corr_cam1),
        'cam2': extract_with_ts(corr_cam2),
        'cam3': extract_with_ts(corr_cam3),
        'avg': extract_with_ts(corr_avg),
    }
    
    # Get global time range from all data sources (use original timestamps for sync)
    all_timestamps = []
    for key in orig_data_arrays:
        all_timestamps.extend(orig_data_arrays[key][3])
    
    if not all_timestamps:
        print("No timestamp data found!")
        return None
    
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    duration_ms = max_time - min_time
    
    # Calculate frames
    real_time_per_frame = FRAME_INTERVAL_MS * PLAYBACK_SPEED
    n_frames = max(int(duration_ms / real_time_per_frame), 1)
    
    # Create figure with two 3D subplots side by side
    fig = plt.figure(figsize=(18, 9))
    ax_orig = fig.add_subplot(121, projection='3d')
    ax_corr = fig.add_subplot(122, projection='3d')
    
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.15, wspace=0.05)
    
    # Helper to setup static trails on an axis
    def setup_static_trails(ax, data_arrays):
        px, py, pz, _ = data_arrays['pressure']
        gx, gy, gz, _ = data_arrays['global']
        
        if px:
            ax.plot(px, py, pz, c='green', alpha=0.2, linewidth=1)
        if gx:
            ax.plot(gx, gy, gz, c='red', alpha=0.2, linewidth=1)
        
        for key, color in [('cam0', 'blue'), ('cam1', 'cyan'), ('cam2', 'navy'), ('cam3', 'dodgerblue')]:
            cx, cy, cz, _ = data_arrays[key]
            if cx:
                ax.plot(cx, cy, cz, color=color, alpha=0.15, linewidth=1)
        
        ax_x, ax_y, ax_z, _ = data_arrays['avg']
        if ax_x:
            ax.plot(ax_x, ax_y, ax_z, color='purple', alpha=0.3, linewidth=1)
    
    # Setup static trails
    setup_static_trails(ax_orig, orig_data_arrays)
    setup_static_trails(ax_corr, corr_data_arrays)
    
    # Create animated scatter points for both panels
    def create_scatter_points(ax):
        pressure_point = ax.scatter([], [], [], c='green', marker='o', s=80, label='Pressure', zorder=5)
        global_point = ax.scatter([], [], [], c='red', marker='^', s=80, label='Global Pos', zorder=5)
        cam_points = [
            ax.scatter([], [], [], c='blue', marker='s', s=60, zorder=5),
            ax.scatter([], [], [], c='cyan', marker='s', s=60, zorder=5),
            ax.scatter([], [], [], c='navy', marker='s', s=60, zorder=5),
            ax.scatter([], [], [], c='dodgerblue', marker='s', s=60, zorder=5),
        ]
        avg_point = ax.scatter([], [], [], c='purple', marker='*', s=150, label='Avg Cam', zorder=5)
        trail_line, = ax.plot([], [], [], c='purple', linewidth=2, alpha=0.8, zorder=4)
        return pressure_point, global_point, cam_points, avg_point, trail_line
    
    orig_points = create_scatter_points(ax_orig)
    corr_points = create_scatter_points(ax_corr)
    
    # Collect all points for axis limits
    all_x, all_y, all_z = [], [], []
    for data_arrays in [orig_data_arrays, corr_data_arrays]:
        for key in data_arrays:
            x, y, z, _ = data_arrays[key]
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
    
    # Set axis limits and labels
    if all_x:
        margin = 0.1
        for ax in [ax_orig, ax_corr]:
            ax.set_xlim([min(all_x) - margin, max(all_x) + margin])
            ax.set_ylim([min(all_y) - margin, max(all_y) + margin])
            ax.set_zlim([min(all_z) - margin, max(all_z) + margin])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(loc='upper left', fontsize=8)
            
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            z_range = max(all_z) - min(all_z)
            ax.set_box_aspect([x_range, y_range, z_range])
            ax.view_init(elev=35, azim=-20)
    
    title_orig = ax_orig.set_title('ORIGINAL')
    title_corr = ax_corr.set_title('CORRECTED')
    main_title = fig.suptitle(f'{title_prefix}3D Comparison - t=0ms', fontsize=14)
    
    trail_duration_ms = 3000
    
    # Animation state
    state = {'playing': True, 'current_frame': 0, 'anim': None, 'slider': None}
    
    def update_panel(data_arrays, points, current_time):
        pressure_point, global_point, cam_points, avg_point, trail_line = points
        
        # Update pressure
        px, py, pz, pts = data_arrays['pressure']
        x, y, z = find_position_at_time(px, py, pz, pts, current_time)
        if x is not None:
            pressure_point._offsets3d = ([x], [y], [z])
        else:
            pressure_point._offsets3d = ([], [], [])
        
        # Update global
        gx, gy, gz, gts = data_arrays['global']
        x, y, z = find_position_at_time(gx, gy, gz, gts, current_time)
        if x is not None:
            global_point._offsets3d = ([x], [y], [z])
        else:
            global_point._offsets3d = ([], [], [])
        
        # Update cameras
        for i, key in enumerate(['cam0', 'cam1', 'cam2', 'cam3']):
            cx, cy, cz, cts = data_arrays[key]
            x, y, z = find_position_at_time(cx, cy, cz, cts, current_time)
            if x is not None:
                cam_points[i]._offsets3d = ([x], [y], [z])
            else:
                cam_points[i]._offsets3d = ([], [], [])
        
        # Update avg
        ax_x, ax_y, ax_z, ax_ts = data_arrays['avg']
        x, y, z = find_position_at_time(ax_x, ax_y, ax_z, ax_ts, current_time)
        if x is not None:
            avg_point._offsets3d = ([x], [y], [z])
        else:
            avg_point._offsets3d = ([], [], [])
        
        # Update trail
        trail_x, trail_y, trail_z = find_trail_at_time(ax_x, ax_y, ax_z, ax_ts, current_time, trail_duration_ms)
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)
    
    def update_frame(frame_idx):
        frame = int(frame_idx) % n_frames
        state['current_frame'] = frame
        
        current_time = min_time + (frame / n_frames) * duration_ms
        
        update_panel(orig_data_arrays, orig_points, current_time)
        update_panel(corr_data_arrays, corr_points, current_time)
        
        elapsed = current_time - min_time
        main_title.set_text(f'{title_prefix}3D Comparison - t={elapsed:.0f}ms / {duration_ms:.0f}ms ({PLAYBACK_SPEED}x)')
        
        if state.get('slider'):
            state['slider'].eventson = False
            state['slider'].set_val(elapsed / 1000.0)
            state['slider'].eventson = True
        
        return []
    
    def animate(frame):
        return update_frame(frame)
    
    # Create animation
    from matplotlib.animation import FuncAnimation
    state['anim'] = FuncAnimation(fig, animate, frames=n_frames, interval=FRAME_INTERVAL_MS, blit=False, repeat=True)
    
    # Add controls
    from matplotlib.widgets import Slider, Button
    
    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Time (s)', 0, duration_ms / 1000.0, valinit=0)
    state['slider'] = slider
    
    def on_slider_change(val):
        if not state['playing']:
            time_ms = val * 1000.0
            frame = int((time_ms / duration_ms) * n_frames)
            frame = max(0, min(n_frames - 1, frame))
            update_frame(frame)
            fig.canvas.draw_idle()
    
    slider.on_changed(on_slider_change)
    
    ax_play = plt.axes([0.75, 0.05, 0.08, 0.04])
    button_play = Button(ax_play, 'Pause')
    
    def toggle_play(event):
        state['playing'] = not state['playing']
        if state['playing']:
            state['anim'].resume()
            button_play.label.set_text('Pause')
        else:
            state['anim'].pause()
            button_play.label.set_text('Play')
        fig.canvas.draw_idle()
    
    button_play.on_clicked(toggle_play)
    
    ax_reset = plt.axes([0.85, 0.05, 0.08, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    def reset_view(event):
        ax_orig.view_init(elev=35, azim=-20)
        ax_corr.view_init(elev=35, azim=-20)
        fig.canvas.draw_idle()
    
    button_reset.on_clicked(reset_view)
    
    # Speed indicator
    plt.figtext(0.05, 0.05, f'Speed: {PLAYBACK_SPEED}x', fontsize=10, 
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 3})
    
    # Keyboard controls
    def on_key(event):
        if event.key == ' ':
            toggle_play(None)
        elif event.key == 'home':
            state['anim'].pause()
            state['playing'] = False
            button_play.label.set_text('Play')
            update_frame(0)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    return fig


def visualize_3d_positions(pressure_depth_data, global_pos_data, camera_pos_0_data, 
                          camera_pos_1_data, camera_pos_2_data, camera_pos_3_data, avg_camera_pos_data):
    """Create an animated 3D visualization of all position data with autoplay."""    
    # Extract position arrays with timestamps
    pressure_x, pressure_y, pressure_z, pressure_ts = extract_position_arrays_with_timestamps(pressure_depth_data)
    global_x, global_y, global_z, global_ts = extract_position_arrays_with_timestamps(global_pos_data)
    
    camera_0_x, camera_0_y, camera_0_z, camera_0_ts = extract_position_arrays_with_timestamps(camera_pos_0_data)
    camera_1_x, camera_1_y, camera_1_z, camera_1_ts = extract_position_arrays_with_timestamps(camera_pos_1_data)
    camera_2_x, camera_2_y, camera_2_z, camera_2_ts = extract_position_arrays_with_timestamps(camera_pos_2_data)
    camera_3_x, camera_3_y, camera_3_z, camera_3_ts = extract_position_arrays_with_timestamps(camera_pos_3_data)
    
    avg_camera_x, avg_camera_y, avg_camera_z, avg_camera_ts = extract_position_arrays_with_timestamps(avg_camera_pos_data)
    
    # Get global time range from all data sources
    all_timestamps = []
    for ts_list in [pressure_ts, global_ts, camera_0_ts, camera_1_ts, camera_2_ts, camera_3_ts, avg_camera_ts]:
        all_timestamps.extend(ts_list)
    
    if not all_timestamps:
        print("No timestamp data found!")
        return
    
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    duration_ms = max_time - min_time
    total_duration_ms = duration_ms  # Store for slider
    
    print(f"Time range: {min_time:.0f}ms to {max_time:.0f}ms (duration: {duration_ms:.0f}ms)")
    
    # Calculate number of frames based on duration and desired frame rate
    # At 4x speed with 50ms interval, we cover 200ms of real time per frame
    real_time_per_frame = FRAME_INTERVAL_MS * PLAYBACK_SPEED
    time_step_ms = real_time_per_frame  # Store for slider
    n_frames = max(int(duration_ms / real_time_per_frame), 1)
    
    print(f"Animation: {n_frames} frames at {PLAYBACK_SPEED}x speed")
    
    # Create 3D plot with tight layout
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Adjust subplot to use more space
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)
    
    # Plot static trail data (faded)
    if pressure_x:
        ax.plot(pressure_x, pressure_y, pressure_z, c='green', alpha=0.2, linewidth=1)
    if global_x:
        ax.plot(global_x, global_y, global_z, c='red', alpha=0.2, linewidth=1)
    for cam_x, cam_y, cam_z in [
        (camera_0_x, camera_0_y, camera_0_z),
        (camera_1_x, camera_1_y, camera_1_z),
        (camera_2_x, camera_2_y, camera_2_z),
        (camera_3_x, camera_3_y, camera_3_z)
    ]:
        if cam_x:
            ax.plot(cam_x, cam_y, cam_z, color='blue', alpha=0.15, linewidth=1)
    if avg_camera_x:
        ax.plot(avg_camera_x, avg_camera_y, avg_camera_z, color='purple', alpha=0.3, linewidth=1)
    
    # Create animated scatter points (current position markers)
    pressure_point = ax.scatter([], [], [], c='green', marker='o', s=100, label='Pressure', zorder=5)
    global_point = ax.scatter([], [], [], c='red', marker='^', s=100, label='Global Position', zorder=5)
    camera_points = [
        ax.scatter([], [], [], c='blue', marker='s', s=80, label='Camera 0', zorder=5),
        ax.scatter([], [], [], c='cyan', marker='s', s=80, label='Camera 1', zorder=5),
        ax.scatter([], [], [], c='navy', marker='s', s=80, label='Camera 2', zorder=5),
        ax.scatter([], [], [], c='dodgerblue', marker='s', s=80, label='Camera 3', zorder=5),
    ]
    avg_point = ax.scatter([], [], [], c='purple', marker='*', s=200, label='Avg Camera', zorder=5)
    
    # Trail line for avg camera
    trail_line, = ax.plot([], [], [], c='purple', linewidth=2, alpha=0.8, zorder=4)
    trail_duration_ms = 3000  # Show 3 seconds of trail
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = ax.set_title('MantaPos 3D Visualization - t=0ms')
    
    # Set axis limits based on all data
    all_x = pressure_x + global_x + camera_0_x + camera_1_x + camera_2_x + camera_3_x + avg_camera_x
    all_y = pressure_y + global_y + camera_0_y + camera_1_y + camera_2_y + camera_3_y + avg_camera_y
    all_z = pressure_z + global_z + camera_0_z + camera_1_z + camera_2_z + camera_3_z + avg_camera_z
    
    if all_x:
        margin = 0.1
        ax.set_xlim([min(all_x) - margin, max(all_x) + margin])
        ax.set_ylim([min(all_y) - margin, max(all_y) + margin])
        ax.set_zlim([min(all_z) - margin, max(all_z) + margin])
    
    ax.legend(loc='upper left')
    
    # Animation state
    state = {
        'playing': True,
        'current_frame': 0,
        'anim': None
    }
    
    # Store all data sources for time-based lookup
    data_sources = {
        'pressure': (pressure_x, pressure_y, pressure_z, pressure_ts),
        'global': (global_x, global_y, global_z, global_ts),
        'camera_0': (camera_0_x, camera_0_y, camera_0_z, camera_0_ts),
        'camera_1': (camera_1_x, camera_1_y, camera_1_z, camera_1_ts),
        'camera_2': (camera_2_x, camera_2_y, camera_2_z, camera_2_ts),
        'camera_3': (camera_3_x, camera_3_y, camera_3_z, camera_3_ts),
        'avg_camera': (avg_camera_x, avg_camera_y, avg_camera_z, avg_camera_ts),
    }
    
    def update_frame(frame_idx):
        """Update the plot for the given frame based on time."""
        frame = int(frame_idx) % n_frames
        state['current_frame'] = frame
        
        # Calculate current time based on frame
        current_time = min_time + (frame / n_frames) * duration_ms
        
        # Update pressure point
        px, py, pz = find_position_at_time(pressure_x, pressure_y, pressure_z, pressure_ts, current_time)
        if px is not None:
            pressure_point._offsets3d = ([px], [py], [pz])
        else:
            pressure_point._offsets3d = ([], [], [])
        
        # Update global point
        gx, gy, gz = find_position_at_time(global_x, global_y, global_z, global_ts, current_time)
        if gx is not None:
            global_point._offsets3d = ([gx], [gy], [gz])
        else:
            global_point._offsets3d = ([], [], [])
        
        # Update camera points
        cam_data = [
            (camera_0_x, camera_0_y, camera_0_z, camera_0_ts),
            (camera_1_x, camera_1_y, camera_1_z, camera_1_ts),
            (camera_2_x, camera_2_y, camera_2_z, camera_2_ts),
            (camera_3_x, camera_3_y, camera_3_z, camera_3_ts)
        ]
        for i, (cx, cy, cz, cts) in enumerate(cam_data):
            x, y, z = find_position_at_time(cx, cy, cz, cts, current_time)
            if x is not None:
                camera_points[i]._offsets3d = ([x], [y], [z])
            else:
                camera_points[i]._offsets3d = ([], [], [])
        
        # Update average point
        ax_val, ay_val, az_val = find_position_at_time(avg_camera_x, avg_camera_y, avg_camera_z, avg_camera_ts, current_time)
        if ax_val is not None:
            avg_point._offsets3d = ([ax_val], [ay_val], [az_val])
        else:
            avg_point._offsets3d = ([], [], [])
        
        # Update trail line
        trail_x, trail_y, trail_z = find_trail_at_time(
            avg_camera_x, avg_camera_y, avg_camera_z, avg_camera_ts, 
            current_time, trail_duration_ms
        )
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)
        
        # Update title with time info
        elapsed = current_time - min_time
        title.set_text(f'MantaPos 3D Visualization - t={elapsed:.0f}ms / {duration_ms:.0f}ms ({PLAYBACK_SPEED}x)')
        
        # Update slider without triggering callback (convert to seconds)
        if state.get('slider'):
            state['slider'].eventson = False
            state['slider'].set_val(elapsed / 1000.0)
            state['slider'].eventson = True
        
        return [pressure_point, global_point, avg_point, trail_line, title] + camera_points
    
    def animate(frame):
        """Animation function called by FuncAnimation."""
        return update_frame(frame)
    
    # Create animation
    interval = FRAME_INTERVAL_MS
    state['anim'] = FuncAnimation(
        fig, animate, frames=n_frames, interval=interval, 
        blit=False, repeat=True
    )
    
    # Add playback controls
    # Slider for time selection (in seconds)
    ax_slider = plt.axes([0.2, 0.08, 0.5, 0.03])
    total_duration_s = total_duration_ms / 1000.0
    slider = Slider(ax_slider, 'Time (s)', 0, total_duration_s, valinit=0)
    state['slider'] = slider
    
    def on_slider_change(val):
        if not state['playing']:
            # Convert slider value (time in seconds) to frame number
            time_ms = val * 1000.0
            frame = int(time_ms / time_step_ms)
            frame = max(0, min(n_frames - 1, frame))
            update_frame(frame)
            fig.canvas.draw_idle()
    
    slider.on_changed(on_slider_change)
    
    # Play/Pause button
    ax_play = plt.axes([0.75, 0.08, 0.1, 0.04])
    button_play = Button(ax_play, 'Pause')
    
    def toggle_play(event):
        state['playing'] = not state['playing']
        if state['playing']:
            state['anim'].resume()
            button_play.label.set_text('Pause')
        else:
            state['anim'].pause()
            button_play.label.set_text('Play')
        fig.canvas.draw_idle()
    
    button_play.on_clicked(toggle_play)
    
    # Reset view button
    ax_home = plt.axes([0.87, 0.08, 0.1, 0.04])
    button_home = Button(ax_home, 'Reset View')
    
    def reset_view(event):
        ax.view_init(elev=35, azim=-20)
        fig.canvas.draw_idle()
        
    button_home.on_clicked(reset_view)
    
    # Speed indicator
    plt.figtext(0.05, 0.08, f'Speed: {PLAYBACK_SPEED}x', fontsize=10, 
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 3})
    
    # Navigation instructions
    plt.figtext(0.5, 0.01, 
                "Controls: Rotate=Left-drag | Zoom=Scroll | Pan=Right-drag | "
                "Space=Play/Pause",
                ha="center", fontsize=9, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 3})
    
    # Set the default view
    ax.view_init(elev=35, azim=-20)
    set_axes_equal(ax)
    
    # Keyboard controls
    def on_key(event):
        if event.key == ' ':
            toggle_play(None)
        elif event.key == 'left':
            state['anim'].pause()
            state['playing'] = False
            button_play.label.set_text('Play')
            new_frame = max(0, state['current_frame'] - 1)
            update_frame(new_frame)
            fig.canvas.draw_idle()
        elif event.key == 'right':
            state['anim'].pause()
            state['playing'] = False
            button_play.label.set_text('Play')
            new_frame = min(n_frames - 1, state['current_frame'] + 1)
            update_frame(new_frame)
            fig.canvas.draw_idle()
        elif event.key == 'home':
            state['anim'].pause()
            state['playing'] = False
            button_play.label.set_text('Play')
            update_frame(0)
            fig.canvas.draw_idle()
        elif event.key == 'end':
            state['anim'].pause()
            state['playing'] = False
            button_play.label.set_text('Play')
            update_frame(n_frames - 1)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()

if __name__ == "__main__":
    try:
        data_file = f"{DATA_DIRECTORY}/{DATA_FILE}.json"
        print(f"Loading data from {data_file}...")
        
        # Load and extract the data
        (pressure_depth_data, global_pos_data, camera_pos_0_data, 
         camera_pos_1_data, camera_pos_2_data, camera_pos_3_data, camera_avg_data) = load_and_extract_data(data_file)
        
        # Print summary of loaded data
        print(f"Loaded {len(pressure_depth_data)} pressure depth points")
        print(f"Loaded {len(global_pos_data)} global position points")
        print(f"Loaded {len(camera_pos_0_data)} camera 0 position points")
        print(f"Loaded {len(camera_pos_1_data)} camera 1 position points")
        print(f"Loaded {len(camera_pos_2_data)} camera 2 position points")
        print(f"Loaded {len(camera_pos_3_data)} camera 3 position points")
        print(f"Loaded {len(camera_avg_data)} average camera position points")

        if display_only_avg:
            # Clear all other camera position data
            camera_pos_0_data = []
            camera_pos_1_data = []
            camera_pos_2_data = []
            camera_pos_3_data = []

        # If only 1 out of 4 camera positions has values, remove the average
        if sum([1 for data in [camera_pos_0_data, camera_pos_1_data, camera_pos_2_data, camera_pos_3_data] if data]) == 1:
            camera_avg_data = []
        
        # Visualize the data
        print("Creating 3D visualization...")
        visualize_3d_positions(pressure_depth_data, global_pos_data, camera_pos_0_data, 
                              camera_pos_1_data, camera_pos_2_data, camera_pos_3_data, camera_avg_data)
        
    except Exception as e:
        print(f"Error: {str(e)}")
