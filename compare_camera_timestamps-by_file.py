import os
import re
import json

def get_first_camera_timestamp(filepath):
    """
    Find the first line containing '"camera": {"timestamp": XXXXX' and extract the timestamp.
    """
    pattern = re.compile(r'"camera":\s*\{\s*"timestamp":\s*(\d+)')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None

def main():
    recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
    original_dir = os.path.join(recordings_dir, 'original')
    output_file = os.path.join(os.path.dirname(__file__), 'camera_timestamp_corrections.txt')
    
    results = []
    
    # Get all json files in the original folder
    original_files = [f for f in os.listdir(original_dir) if f.endswith('.json')]
    original_files.sort()
    
    for filename in original_files:
        current_file = os.path.join(recordings_dir, filename)
        original_file = os.path.join(original_dir, filename)
        
        # Check if both files exist
        if not os.path.exists(current_file):
            print(f"Skipping {filename}: current file not found")
            continue
        
        # Get timestamps
        current_ts = get_first_camera_timestamp(current_file)
        original_ts = get_first_camera_timestamp(original_file)
        
        if current_ts is None:
            print(f"Skipping {filename}: no camera timestamp in current file")
            continue
        if original_ts is None:
            print(f"Skipping {filename}: no camera timestamp in original file")
            continue
        
        # Calculate difference (new - original)
        difference = current_ts - original_ts
        
        # Get the name without .json extension
        name = filename.replace('.json', '')
        
        results.append((name, difference))
        print(f'"{name}" {difference}')
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, diff in results:
            f.write(f'"{name}" {diff}\n')
    
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
