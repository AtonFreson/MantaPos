import os
import numpy as np

calibration_dir = './camera_calibrations'

# List all files in the calibration directory
all_files = os.listdir(calibration_dir)

# Filter the list to include only .npz files
npz_files = [file for file in all_files if file.endswith('.npz')]

# Print out the contents of each .npz camera calibration file
for file in npz_files:
    file_path = os.path.join(calibration_dir, file)
    with np.load(file_path) as data:
        print(f"Contents of {file}:")
        for key, value in data.items():
            print(f"{key}: {value}")
        print()