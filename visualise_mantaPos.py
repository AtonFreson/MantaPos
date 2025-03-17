import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Configuration
DATA_DIRECTORY = 'recordings'
DATA_FILE = 'ArUco Single 7-4.5m'
display_only_avg = False
display_gradient = False

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
                global_pos_data.append({
                    'timestamp': item['global_pos'].get('timestamp', 0),
                    'position': item['global_pos']['position']
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
    """Set equal scaling for a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

def visualize_3d_positions(pressure_depth_data, global_pos_data, camera_pos_0_data, 
                          camera_pos_1_data, camera_pos_2_data, camera_pos_3_data, avg_camera_pos_data):
    """Create a 3D visualization of all position data."""    
    # Extract position arrays
    pressure_x, pressure_y, pressure_z = extract_position_arrays(pressure_depth_data)
    global_x, global_y, global_z = extract_position_arrays(global_pos_data)
    
    camera_0_x, camera_0_y, camera_0_z = extract_position_arrays(camera_pos_0_data)
    camera_1_x, camera_1_y, camera_1_z = extract_position_arrays(camera_pos_1_data)
    camera_2_x, camera_2_y, camera_2_z = extract_position_arrays(camera_pos_2_data)
    camera_3_x, camera_3_y, camera_3_z = extract_position_arrays(camera_pos_3_data)
    
    avg_camera_x, avg_camera_y, avg_camera_z = extract_position_arrays(avg_camera_pos_data)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot pressure data in green
    if pressure_x:
        ax.scatter(pressure_x, pressure_y, pressure_z, c='green', marker='o', label='Pressure', s=30)
        ax.plot(pressure_x, pressure_y, pressure_z, c='green', alpha=0.5)
    
    # Plot global position data in red
    if global_x:
        ax.scatter(global_x, global_y, global_z, c='red', marker='^', label='Global Position', s=30)
        ax.plot(global_x, global_y, global_z, c='red', alpha=0.5)
    
    # Plot camera position data with blue-to-green gradient
    for cam_x, cam_y, cam_z, cam_num in [
        (camera_0_x, camera_0_y, camera_0_z, 0),
        (camera_1_x, camera_1_y, camera_1_z, 1),
        (camera_2_x, camera_2_y, camera_2_z, 2),
        (camera_3_x, camera_3_y, camera_3_z, 3)
    ]:
        if cam_x:
            # Create color gradient from blue to green
            n_points = len(cam_x)
            if n_points > 0:
                if display_gradient:    
                    # Generate colors from blue to green
                    colors = plt.cm.winter(np.linspace(0, 1, n_points))
                    # Plot each point with its corresponding color in the gradient
                    for i in range(n_points):
                        ax.scatter(cam_x[i], cam_y[i], cam_z[i], c=[colors[i]], marker='s', s=20)
                else:
                    ax.scatter(cam_x, cam_y, cam_z, c='blue', marker='s', label=f'Camera {cam_num}', s=20)

                # Add a single representative point for the legend
                ax.scatter([], [], [], c='blue', marker='s', label=f'Camera {cam_num}' if cam_num == 0 else "", s=20)
                
                # Plot the connecting line
                ax.plot(cam_x, cam_y, cam_z, color='gray', alpha=0.3)
    
    # Plot average camera positions in purple
    if avg_camera_x:
        # Create color gradient for average camera positions
        n_points = len(avg_camera_x)
        if n_points > 0:
            if display_gradient:
                colors = plt.cm.winter(np.linspace(0, 1, n_points))
                for i in range(n_points):
                    ax.scatter(avg_camera_x[i], avg_camera_y[i], avg_camera_z[i], c=[colors[i]], marker='*', s=50)
            else:
                ax.scatter(avg_camera_x, avg_camera_y, avg_camera_z, c='purple', marker='*', label='Avg Camera', s=50)

            # Add a single representative point for the legend
            ax.scatter([], [], [], c='purple', marker='*', label='Avg Camera', s=50)
            
            # Plot the connecting line
            ax.plot(avg_camera_x, avg_camera_y, avg_camera_z, color='gray', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('MantaPos 3D Visualization')
    
    # Add legend with only one entry per color
    ax.legend()
    
    # Add navigation instructions text at bottom of figure
    plt.figtext(0.5, 0.01, 
                "Navigation Controls:\n"
                "• Rotate: Click and drag with left mouse button\n"
                "• Zoom: Scroll mouse wheel\n"
                "• Pan: Click and drag with right mouse button",
                ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Add home button to reset view
    ax_home = plt.axes([0.8, 0.05, 0.1, 0.05])  # Moved up to accommodate instruction text
    button_home = Button(ax_home, 'Reset View')
    
    def reset_view(event):
        ax.view_init(elev=30, azim=-60)
        plt.draw()
        
    button_home.on_clicked(reset_view)
    
    # Set the default view
    ax.view_init(elev=30, azim=-60)
    set_axes_equal(ax)
    
    # Show the plot (interactive)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to accommodate the text
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
