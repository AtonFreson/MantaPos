import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

from visualise_mantaPos import (
    extract_position_arrays_with_timestamps, 
    find_position_at_time, 
    find_trail_at_time, 
    set_axes_equal, 
    PLAYBACK_SPEED, 
    FRAME_INTERVAL_MS
)

def plot_differences(ukf_data, ref_data, camera_data=None):
    import numpy as np
    if not ukf_data or not ref_data:
        print("Missing data for difference plot")
        return
        
    ukf_ts = np.array([d['timestamp'] for d in ukf_data])
    ukf_y = np.array([d['position'][1] for d in ukf_data])
    ukf_z = np.array([d['position'][2] for d in ukf_data])
    
    ref_ts = np.array([d['timestamp'] for d in ref_data])
    ref_y = np.array([d['position'][1] for d in ref_data])
    ref_z = np.array([d['position'][2] for d in ref_data])
    
    # Interpolate reference data precisely at UKF timestamps
    ref_y_interp = np.interp(ukf_ts, ref_ts, ref_y)
    ref_z_interp = np.interp(ukf_ts, ref_ts, ref_z)
    
    diff_y = ukf_y - ref_y_interp
    diff_z = ukf_z - ref_z_interp
    
    start_ts = ukf_ts[0]
    
    # Time in seconds from start
    t_sec = (ukf_ts - start_ts) / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.canvas.manager.set_window_title('UKF vs Reference Differences')
    
    ax1.plot(t_sec, diff_y, label='Y Difference (UKF - Ref)', color='green')
    
    if camera_data:
        cam_ts = np.array([d['timestamp'] for d in camera_data])
        cam_y = np.array([d['position'][1] for d in camera_data])
        cam_z = np.array([d['position'][2] for d in camera_data])
        
        # Interpolate reference data at Camera timestamps
        ref_cam_y_interp = np.interp(cam_ts, ref_ts, ref_y)
        ref_cam_z_interp = np.interp(cam_ts, ref_ts, ref_z)
        
        diff_cam_y = cam_y - ref_cam_y_interp
        diff_cam_z = cam_z - ref_cam_z_interp
        
        t_sec_cam = (cam_ts - start_ts) / 1000.0
        
        ax1.plot(t_sec_cam, diff_cam_y, label='Y Difference (Camera - Ref)', color='blue', alpha=0.5, linestyle='--')
        ax2.plot(t_sec_cam, diff_cam_z, label='Z Difference (Camera - Ref)', color='blue', alpha=0.5, linestyle='--')

    ax1.set_ylabel('Y Difference (m)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t_sec, diff_z, label='Z Difference (UKF - Ref)', color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z Difference (m)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()


def visualize_ukf_results(ukf_data, ref_data, camera_data, use_2d=False):
    """Create an animated visualization of UKF vs Reference position data."""
    import numpy as np
    # Extract position arrays with timestamps
    ukf_x, ukf_y, ukf_z, ukf_ts = extract_position_arrays_with_timestamps(ukf_data)
    ref_x, ref_y, ref_z, ref_ts = extract_position_arrays_with_timestamps(ref_data)
    cam_x, cam_y, cam_z, cam_ts = extract_position_arrays_with_timestamps(camera_data)
    
    all_x = ukf_x + ref_x + cam_x
    if use_2d and all_x:
        x_min, x_max = min(all_x), max(all_x)
        if x_max - x_min > 1e-3:
            print(f"WARNING: 2D view selected, but X-axis has variability! Min: {x_min:.4f}, Max: {x_max:.4f}")
    
    
    # Get global time range from all data sources
    all_timestamps = []
    for ts_list in [ukf_ts, ref_ts, cam_ts]:
        all_timestamps.extend(ts_list)
    
    if not all_timestamps:
        print("No timestamp data found for visualization!")
        return
    
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    duration_ms = max_time - min_time
    total_duration_ms = duration_ms
    
    real_time_per_frame = FRAME_INTERVAL_MS * PLAYBACK_SPEED
    time_step_ms = real_time_per_frame
    n_frames = max(int(duration_ms / real_time_per_frame), 1)
    
    fig = plt.figure(figsize=(14, 10))
    if use_2d:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)
    
    # Static trails (faded)
    if use_2d:
        if ukf_x: ax.plot(ukf_y, ukf_z, c='purple', alpha=0.3, linewidth=2)
        if ref_x: ax.plot(ref_y, ref_z, c='red', alpha=0.3, linewidth=1)
        if cam_x: ax.plot(cam_y, cam_z, c='blue', alpha=0.15, linewidth=1)
    else:
        if ukf_x: ax.plot(ukf_x, ukf_y, ukf_z, c='purple', alpha=0.3, linewidth=2)
        if ref_x: ax.plot(ref_x, ref_y, ref_z, c='red', alpha=0.3, linewidth=1)
        if cam_x: ax.plot(cam_x, cam_y, cam_z, c='blue', alpha=0.15, linewidth=1)

    # Animated scatter points
    if use_2d:
        ukf_point = ax.scatter([], [], c='purple', marker='*', s=200, label='UKF Estimation', zorder=6)
        ref_point = ax.scatter([], [], c='red', marker='^', s=100, label='Reference POS', zorder=5)
        cam_point = ax.scatter([], [], c='blue', marker='s', s=80, label='Avg Camera', zorder=5)
        trail_line, = ax.plot([], [], c='purple', linewidth=2, alpha=0.8, zorder=4)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        title = ax.set_title('UKF vs Reference 2D Visualization - t=0ms')
    else:
        ukf_point = ax.scatter([], [], [], c='purple', marker='*', s=200, label='UKF Estimation', zorder=6)
        ref_point = ax.scatter([], [], [], c='red', marker='^', s=100, label='Reference POS', zorder=5)
        cam_point = ax.scatter([], [], [], c='blue', marker='s', s=80, label='Avg Camera', zorder=5)
        trail_line, = ax.plot([], [], [], c='purple', linewidth=2, alpha=0.8, zorder=4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = ax.set_title('UKF vs Reference 3D Visualization - t=0ms')
    
    trail_duration_ms = 3000
    
    # Calculate axis limits
    all_x = ukf_x + ref_x + cam_x
    all_y = ukf_y + ref_y + cam_y
    all_z = ukf_z + ref_z + cam_z
    
    if all_x:
        margin = 0.5
        if use_2d:
            ax.set_xlim([min(all_y) - margin, max(all_y) + margin])
            ax.set_ylim([min(all_z) - margin, max(all_z) + margin])
            ax.set_aspect('equal', adjustable='datalim')
        else:
            ax.set_xlim([min(all_x) - margin, max(all_x) + margin])
            ax.set_ylim([min(all_y) - margin, max(all_y) + margin])
            ax.set_zlim([min(all_z) - margin, max(all_z) + margin])
    
    ax.legend(loc='upper left')
    
    state = {'playing': True, 'current_frame': 0, 'anim': None}
    
    def update_frame(frame_idx):
        frame = int(frame_idx) % n_frames
        state['current_frame'] = frame
        current_time = min_time + (frame / n_frames) * duration_ms
        
        # update ukf
        ux, uy, uz = find_position_at_time(ukf_x, ukf_y, ukf_z, ukf_ts, current_time)
        if use_2d:
            if ux is not None: ukf_point.set_offsets(np.c_[[uy], [uz]])
            else: ukf_point.set_offsets(np.empty((0, 2)))
        else:
            if ux is not None: ukf_point._offsets3d = ([ux], [uy], [uz])
            else: ukf_point._offsets3d = ([], [], [])
            
        # update ref
        rx, ry, rz = find_position_at_time(ref_x, ref_y, ref_z, ref_ts, current_time)
        if use_2d:
            if rx is not None: ref_point.set_offsets(np.c_[[ry], [rz]])
            else: ref_point.set_offsets(np.empty((0, 2)))
        else:
            if rx is not None: ref_point._offsets3d = ([rx], [ry], [rz])
            else: ref_point._offsets3d = ([], [], [])
            
        # update cameras
        cx_val, cy_val, cz_val = find_position_at_time(cam_x, cam_y, cam_z, cam_ts, current_time)
        if use_2d:
            if cx_val is not None: cam_point.set_offsets(np.c_[[cy_val], [cz_val]])
            else: cam_point.set_offsets(np.empty((0, 2)))
        else:
            if cx_val is not None: cam_point._offsets3d = ([cx_val], [cy_val], [cz_val])
            else: cam_point._offsets3d = ([], [], [])
                
        # ukf trail
        trail_x, trail_y, trail_z = find_trail_at_time(ukf_x, ukf_y, ukf_z, ukf_ts, current_time, trail_duration_ms)
        if use_2d:
            trail_line.set_data(trail_y, trail_z)
        else:
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)
        
        elapsed = current_time - min_time
        title.set_text(f"UKF vs Reference - t={elapsed:.0f}ms / {duration_ms:.0f}ms ({PLAYBACK_SPEED}x)")
        
        if state.get('slider'):
            state['slider'].eventson = False
            state['slider'].set_val(elapsed / 1000.0)
            state['slider'].eventson = True
            
        return [ukf_point, ref_point, trail_line, title, cam_point]

    state['anim'] = FuncAnimation(fig, update_frame, frames=n_frames, interval=FRAME_INTERVAL_MS, blit=False, repeat=True)
    
    # Playback Controls
    ax_slider = plt.axes([0.2, 0.08, 0.5, 0.03])
    slider = Slider(ax_slider, 'Time (s)', 0, total_duration_ms / 1000.0, valinit=0)
    state['slider'] = slider
    def on_slider_change(val):
        if not state['playing']:
            time_ms = val * 1000.0
            frame = int(time_ms / time_step_ms)
            frame = max(0, min(n_frames - 1, frame))
            update_frame(frame)
            fig.canvas.draw_idle()
    slider.on_changed(on_slider_change)
    
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
    
    ax_home = plt.axes([0.87, 0.08, 0.1, 0.04])
    button_home = Button(ax_home, 'Reset View')
    def reset_view(event):
        if not use_2d:
            ax.view_init(elev=35, azim=-20)
        fig.canvas.draw_idle()
    button_home.on_clicked(reset_view)
    
    plt.figtext(0.05, 0.08, f'Speed: {PLAYBACK_SPEED}x', fontsize=10, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 3})
    
    if not use_2d:
        ax.view_init(elev=35, azim=-20)
        set_axes_equal(ax)
    
    def on_key(event):
        if event.key == ' ': toggle_play(None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()