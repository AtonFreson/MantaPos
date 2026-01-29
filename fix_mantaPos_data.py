#!/usr/bin/env python3
"""
Fix camera pose ambiguity and timestamp offset in recording files.

This script permanently corrects the source JSON data so all downstream scripts
work correctly without needing per-script patching.

Corrections applied:
1. Camera timestamp offset (shift camera.timestamp to align with encoder data)
2. Camera pose ambiguity (resolve solvePnP planar marker flip using alter_to_correct_pose)
3. Adds metadata to track that corrections were applied

Uses data_processor.py functions for calculations to ensure consistency.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Import functions from data_processor
from data_processor import (
    DataProcessor,
    Config,
    get_marker_config,
    apply_pose_corrections,
    extract_camera_encoder_data,
    calculate_position_offset,
    find_optimal_time_offset,
    apply_time_corrections,
    collect_all_camera_data,
)
from fix_recording_datapoint import edit_recording

# Import visualization functions from visualise_mantaPos
from visualise_mantaPos import (
    load_and_extract_data,
    visualize_3d_positions,
    visualize_3d_comparison,
    GLOBAL_POS_Y_OFFSET,
)

ROOT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT_DIR / "recordings"


def load_recording(filepath: Path) -> list[dict]:
    """Load NDJSON recording file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_recording(filepath: Path, data: list[dict]) -> None:
    """Save data as NDJSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            # Convert numpy arrays to lists for JSON serialization
            entry = convert_numpy_to_list(entry)
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj


def analyze_recording(filepath: Path, marker_unit: int = 4, quiet: bool = False) -> dict:
    """
    Analyze a recording file using data_processor functions.
    
    Args:
        filepath: Path to the recording JSON
        marker_unit: Which marker to use (0-3, or 4 for average)
        quiet: Suppress output from data_processor functions
        
    Returns:
        Analysis results dictionary
    """
    import io
    import sys
    
    # Optionally suppress output from data_processor
    if quiet:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        # Use DataProcessor to load and process data
        processor = DataProcessor([str(filepath)])
        processor.load_data()
        
        # Process camera timestamps (same as data_processor main)
        averaged_timestamps = processor.process_data_timings("result")
        timestamp_idx = 0
        for data in processor.data:
            if data['mpu_unit'] == 4:
                data['camera']['timestamp'] = averaged_timestamps[timestamp_idx]
                timestamp_idx += 1
        
        if timestamp_idx != len(averaged_timestamps):
            return {"error": "Mismatch in camera timestamps count"}
        
        # Get marker configuration and apply pose corrections
        quad_marker_pos, quad_marker_rot = get_marker_config()
        
        # Apply pose corrections (modifies processor.data in place)
        apply_pose_corrections(processor, quad_marker_pos, quad_marker_rot, 0)  # 0 time correction for analysis
        
        # Extract camera and encoder data
        extracted = extract_camera_encoder_data(processor, marker_unit)
        
        if len(extracted['camera_data']) == 0:
            return {"error": "No camera data found"}
        
        # Calculate position offset (suppress its print)
        old_stdout_inner = sys.stdout
        sys.stdout = io.StringIO()
        camera_pos_offset = calculate_position_offset(
            extracted['camera_data'], Config.camera_pos_offset_ref
        )
        sys.stdout = old_stdout_inner
        
        # Combine all camera data
        all_cam_data = np.append(extracted['camera_data'], extracted['camera_data_ext'])
        
        # Find optimal time offset (suppress its print)
        sys.stdout = io.StringIO()
        offset_result = find_optimal_time_offset(
            all_cam_data, extracted['camera_timestamps'],
            extracted['ref_data'], extracted['ref_timestamps'],
            Config.auto_correction_range, 0  # 0 as current time correction
        )
        sys.stdout = old_stdout_inner
        
        return {
            "best_time_offset_ms": offset_result['best_offset'],
            "best_offset_std": offset_result['std_vals'][offset_result['best_offset_idx']][1],
            "closest_offset_ms": offset_result['std_vals'][offset_result['best_offset_idx']][0],
            "position_offset_y": camera_pos_offset,
            "position_offset_std": np.std(extracted['camera_data']),
            "num_offset_vals": len(extracted['camera_data']),
            "num_camera_entries": len(extracted['camera_timestamps']),
            "num_encoder_entries": len(extracted['ref_data']),
        }
    finally:
        if quiet:
            sys.stdout = old_stdout


def show_visualizations(filepath: Path, marker_unit: int = 4) -> dict:
    """
    Show visualizations for a recording file:
    1. 3D visualization from visualise_mantaPos
    2. timestamps_shifted graph from data_processor
    
    Args:
        filepath: Path to the recording JSON
        marker_unit: Which marker to use (0-3, or 4 for average)
        
    Returns:
        Analysis results dictionary (same as analyze_recording)
    """
    import io
    import sys
    
    print(f"\nLoading and analyzing: {filepath.name}")
    
    # Use DataProcessor to load and process data
    processor = DataProcessor([str(filepath)])
    processor.load_data()
    
    # Process camera timestamps
    averaged_timestamps = processor.process_data_timings("result")
    timestamp_idx = 0
    for data in processor.data:
        if data['mpu_unit'] == 4:
            data['camera']['timestamp'] = averaged_timestamps[timestamp_idx]
            timestamp_idx += 1
    
    if timestamp_idx != len(averaged_timestamps):
        print("Error: Mismatch in camera timestamps count")
        return {"error": "Mismatch in camera timestamps count"}
    
    # Get marker configuration and apply pose corrections
    quad_marker_pos, quad_marker_rot = get_marker_config()
    
    # Apply pose corrections (modifies processor.data in place)
    apply_pose_corrections(processor, quad_marker_pos, quad_marker_rot, 0)
    
    # Extract camera and encoder data
    extracted = extract_camera_encoder_data(processor, marker_unit)
    
    if len(extracted['camera_data']) == 0:
        print("Error: No camera data found")
        return {"error": "No camera data found"}
    
    # Calculate position offset
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    camera_pos_offset = calculate_position_offset(
        extracted['camera_data'], Config.camera_pos_offset_ref
    )
    sys.stdout = old_stdout
    
    # Combine all camera data
    all_cam_data = np.append(extracted['camera_data'], extracted['camera_data_ext'])
    
    # Find optimal time offset
    sys.stdout = io.StringIO()
    offset_result = find_optimal_time_offset(
        all_cam_data, extracted['camera_timestamps'],
        extracted['ref_data'], extracted['ref_timestamps'],
        Config.auto_correction_range, 0
    )
    sys.stdout = old_stdout
    
    # Apply time corrections to get timestamps_shifted data
    pos_difference = apply_time_corrections(
        processor, all_cam_data, extracted['camera_timestamps'],
        extracted['ref_data'], extracted['ref_timestamps'],
        offset_result['best_offset'], camera_pos_offset, offset_result, marker_unit
    )
    
    # Collect all camera data for visualization
    all_camera_data, all_camera_timestamps_corrected = collect_all_camera_data(
        processor, camera_pos_offset, offset_result['best_offset']
    )
    
    # Extract data for visualization
    processor.extract_all_data()
    
    # Show timestamps_shifted graph
    print(f"\nShowing timestamps_shifted graph...")
    processor.visualize(
        mpu_units=[4], 
        sensor_types=[f'camera_pos_{marker_unit}'], 
        fields=['timestamps_shifted'],
        ref_timestamps=extracted['ref_timestamps'], 
        ref_data=-1 * np.array(extracted['ref_data']),
        all_camera_timestamps=all_camera_timestamps_corrected, 
        all_camera_data=all_camera_data
    )
    
    return {
        "best_time_offset_ms": offset_result['best_offset'],
        "best_offset_std": offset_result['std_vals'][offset_result['best_offset_idx']][1],
        "closest_offset_ms": offset_result['std_vals'][offset_result['best_offset_idx']][0],
        "position_offset_y": camera_pos_offset,
        "position_offset_std": np.std(extracted['camera_data']),
        "num_offset_vals": len(extracted['camera_data']),
        "num_camera_entries": len(extracted['camera_timestamps']),
        "num_encoder_entries": len(extracted['ref_data']),
    }


def show_3d_comparison(original_path: Path, modified_path: Path):
    """
    Show side-by-side 3D visualization of original and modified recordings.
    
    Args:
        original_path: Path to original recording
        modified_path: Path to modified recording
    """
    print(f"\nLoading original: {original_path.name}")
    try:
        orig_data = load_and_extract_data(str(original_path))
    except Exception as e:
        print(f"Error loading original: {e}")
        return
    
    print(f"Loading modified: {modified_path.name}")
    try:
        mod_data = load_and_extract_data(str(modified_path))
    except Exception as e:
        print(f"Error loading modified: {e}")
        return
    
    # Show original
    print("\n=== Original Recording ===")
    visualize_3d_positions(*orig_data)
    
    # Show modified
    print("\n=== Modified Recording ===")
    visualize_3d_positions(*mod_data)


def apply_corrections_in_memory(filepath: Path, time_offset_ms: float, fix_poses: bool = True) -> list[dict]:
    """
    Apply all corrections to recording data in memory without saving.
    
    Args:
        filepath: Path to the original recording JSON
        time_offset_ms: Milliseconds to add to camera.timestamp
        fix_poses: Whether to fix pose ambiguity
        
    Returns:
        List of corrected data entries
    """
    import mantaPosLib as manta
    
    # Load original data
    data = load_recording(filepath)
    
    # Get marker config
    quad_marker_pos, quad_marker_rot = get_marker_config()
    marker_order = Config.MARKER_ORDER
    
    # Apply corrections to each entry
    for entry in data:
        if entry.get('mpu_unit') != 4:
            continue
        
        # Apply timestamp offset
        if 'camera' in entry and 'timestamp' in entry['camera']:
            entry['camera']['timestamp'] = int(entry['camera']['timestamp'] + round(time_offset_ms))
        
        # Apply pose corrections
        if fix_poses:
            for marker_idx in range(4):
                key = f'camera_pos_{marker_idx}'
                if key not in entry:
                    continue
                
                position = entry[key]['position']
                rotation = entry[key]['rotation']
                
                # Apply alter_to_correct_pose
                (camera_pos, camera_rot), error_scores, corrected = manta.alter_to_correct_pose(
                    position,
                    rotation,
                    [quad_marker_pos[marker_order[marker_idx][0]], 
                     quad_marker_rot[marker_order[marker_idx][1]]]
                )
                
                if corrected and camera_pos is not None and camera_rot is not None:
                    # Apply rotation wraparound fix
                    camera_rot = list(camera_rot)
                    for i in range(3):
                        if camera_rot[i] > 90:
                            camera_rot[i] -= 180
                        elif camera_rot[i] < -90:
                            camera_rot[i] += 180
                    
                    entry[key]['position'] = camera_pos.tolist() if hasattr(camera_pos, 'tolist') else list(camera_pos)
                    entry[key]['rotation'] = camera_rot
    
    return data


def extract_data_from_memory(data: list[dict]) -> tuple:
    """
    Extract visualization data from in-memory data (same format as load_and_extract_data).
    
    Returns tuple matching load_and_extract_data output.
    """
    from visualise_mantaPos import GLOBAL_POS_Y_OFFSET
    
    pressure_depth_data = []
    global_pos_data = []
    camera_pos_data = [[], [], [], []]
    camera_avg_data = []
    
    for item in data:
        # Extract pressure depth data
        if item.get('mpu_unit') == 0 and 'pressure' in item and 'depth0' in item['pressure'] and 'depth1' in item['pressure']:
            depth0 = item['pressure']['depth0']
            depth1 = item['pressure']['depth1']
            pressure_depth_data.append({
                'timestamp': item['pressure'].get('timestamp', 0),
                'position': [1.5406, 0, -(depth0+depth1)/2 - 0.33]
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
    
    return (pressure_depth_data, global_pos_data, *camera_pos_data, camera_avg_data)


def fix_recording(
    filepath: Path,
    time_offset_ms: float | None = None,
    fix_poses: bool = True,
    backup: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Apply all corrections to a recording file.
    
    Args:
        filepath: Path to the recording JSON
        time_offset_ms: Milliseconds to add to camera.timestamp. If None, auto-calculated.
        fix_poses: Whether to fix pose ambiguity
        backup: Create backup before modifying
        dry_run: Print what would be done without modifying
        
    Returns:
        Summary dict with statistics
    """
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Processing: {filepath.name}")
    
    # Load data to check for existing metadata
    data = load_recording(filepath)
    print(f"  Loaded {len(data)} records")
    
    # Check if already has fix metadata
    existing_metadata = [d for d in data if d.get('mpu_unit') == -1 and '_fix_metadata' in d]
    if existing_metadata and not dry_run:
        prev_fix = existing_metadata[0].get('_fix_metadata', {})
        print(f"  WARNING: File already has fix metadata!")
        print(f"  Previous fix at: {prev_fix.get('fixed_at', 'unknown')}")
        print("  Skipping to avoid double-fixing.")
        return {"error": "Already fixed", "previous_metadata": existing_metadata[0]}
    
    # Analyze if time_offset not provided
    if time_offset_ms is None:
        print("  Analyzing timing using data_processor...")
        analysis = analyze_recording(filepath)
        if "error" in analysis:
            print(f"  Error: {analysis['error']}")
            return {"error": analysis['error']}
        
        time_offset_ms = analysis['best_time_offset_ms']
        print(f"  Auto-detected time offset: {time_offset_ms:.2f}ms")
        print(f"  Position offset (y): {analysis.get('position_offset_y', 'N/A'):.4f}m")
    
    # Use DataProcessor to apply pose corrections and count them
    processor = DataProcessor([str(filepath)])
    processor.load_data()
    
    # Process camera timestamps
    averaged_timestamps = processor.process_data_timings("result")
    timestamp_idx = 0
    for entry in processor.data:
        if entry['mpu_unit'] == 4:
            entry['camera']['timestamp'] = averaged_timestamps[timestamp_idx]
            timestamp_idx += 1
    
    # Get marker config
    quad_marker_pos, quad_marker_rot = get_marker_config()
    
    # Count corrections needed (before applying)
    pose_corrections = 0
    timestamp_corrections = 0
    
    for entry in processor.data:
        if entry.get('mpu_unit') == 4:
            if 'camera' in entry:
                timestamp_corrections += 1
    
    # Apply pose corrections to count them
    if fix_poses:
        modified = apply_pose_corrections(processor, quad_marker_pos, quad_marker_rot, time_offset_ms)
        pose_corrections = sum(1 for m in modified if any(m[2]))  # Count entries with corrections
    
    summary = {
        "file": str(filepath),
        "time_offset_ms": time_offset_ms,
        "pose_corrections": pose_corrections,
        "timestamp_corrections": timestamp_corrections,
        "total_records": len(processor.data),
    }
    
    print(f"  Timestamp corrections needed: {timestamp_corrections}")
    print(f"  Pose corrections needed: {pose_corrections}")
    
    if dry_run:
        print("  [DRY-RUN] No changes written")
        return summary
    
    # Apply corrections to file
    backup_mode = "backup" if backup else "no-backup"
    file_arg = filepath.stem
    
    # 1. Fix camera timestamps using edit_recording
    print(f"\n  Applying timestamp offset: +{time_offset_ms:.2f}ms")
    edit_recording(
        backup_mode, "addition", file_arg,
        "camera", "timestamp",
        mpu_unit=4,
        value=round(time_offset_ms),
    )
    
    # 2. Apply pose corrections and add metadata
    if fix_poses:
        print(f"  Applying pose corrections...")
        apply_pose_corrections_to_file(filepath, quad_marker_pos, quad_marker_rot)
    
    print(f"\n  Corrections applied to: {filepath.name}")
    
    return summary


def apply_pose_corrections_to_file(filepath: Path, quad_marker_pos: list, quad_marker_rot: list) -> int:
    """
    Apply pose corrections to all camera entries in the file.
    Uses data_processor's apply_pose_corrections logic.
    """
    import mantaPosLib as manta
    
    # Load data
    data = load_recording(filepath)
    corrections = 0
    
    marker_order = Config.MARKER_ORDER
    
    for entry in data:
        if entry.get('mpu_unit') != 4:
            continue
        
        for marker_idx in range(4):
            key = f'camera_pos_{marker_idx}'
            if key not in entry:
                continue
            
            position = entry[key]['position']
            rotation = entry[key]['rotation']
            
            # Apply alter_to_correct_pose
            (camera_pos, camera_rot), error_scores, corrected = manta.alter_to_correct_pose(
                position,
                rotation,
                [quad_marker_pos[marker_order[marker_idx][0]], 
                 quad_marker_rot[marker_order[marker_idx][1]]]
            )
            
            if corrected and camera_pos is not None and camera_rot is not None:
                # Apply rotation wraparound fix (same as data_processor)
                camera_rot = list(camera_rot)
                for i in range(3):
                    if camera_rot[i] > 90:
                        camera_rot[i] -= 180
                    elif camera_rot[i] < -90:
                        camera_rot[i] += 180
                
                entry[key]['position'] = camera_pos.tolist() if hasattr(camera_pos, 'tolist') else list(camera_pos)
                entry[key]['rotation'] = camera_rot
                corrections += 1
    
    # Save
    save_recording(filepath, data)
    print(f"    Pose corrections applied: {corrections}")
    
    return corrections


def main():
    # ============================================================
    # CONFIGURATION - Edit these values directly
    # ============================================================
    
    filename = "ChArUco Single 4.5-2m"
    
    mode = "fix"  # Options: "analyze", "fix", "dry-run", "visualize"
    
    time_offset_ms = None  # Set to None for auto-detection, or specify manually (e.g., 628.49)
    fix_poses = True       # Fix solvePnP pose ambiguity
    create_backup = False  # Create backup before modifying
    marker_unit = 4        # Which marker to use (0-3, or 4 for average)
    show_vis = True        # Show visualizations before fixing (3D + timestamps_shifted)
    
    # ============================================================
    
    # Resolve file path
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1
    
    if mode == "analyze":
        print(f"\nAnalyzing: {filepath.name}")
        
        analysis = analyze_recording(filepath, marker_unit, quiet=True)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return 1
        
        # Output format matches data_processor.py exactly
        print(f"\nCamera position offset: {analysis['position_offset_y']:.4f} +- {analysis['position_offset_std']:.5f} (# vals: {analysis['num_offset_vals']})")
        print(f"\nBest offset for camera data: {analysis['best_time_offset_ms']:.2f} ms (closest: {analysis['closest_offset_ms']:.1f} & std: {analysis['best_offset_std']:.4f})")
        print(f"\nCamera entries: {analysis['num_camera_entries']}")
        print(f"Encoder entries: {analysis['num_encoder_entries']}")
        
        return 0
    
    elif mode == "visualize":
        # Just show visualizations without fixing
        print(f"\n=== Visualization Mode ===")
        
        # Show 3D visualization
        print(f"\nLoading 3D visualization for: {filepath.name}")
        try:
            data = load_and_extract_data(str(filepath))
            visualize_3d_positions(*data)
        except Exception as e:
            print(f"Error loading 3D visualization: {e}")
        
        # Show timestamps_shifted graph
        show_visualizations(filepath, marker_unit)
        plt.show(block=True)
        
        return 0
    
    elif mode in ("fix", "dry-run"):
        dry_run = (mode == "dry-run")
        
        # First, analyze to get time offset
        print(f"\n=== Analyzing Recording ===")
        analysis = analyze_recording(filepath, marker_unit, quiet=True)
        
        if "error" in analysis:
            print(f"Error during analysis: {analysis['error']}")
            return 1
        
        # Use analyzed time offset if not manually specified
        if time_offset_ms is None:
            time_offset_ms = analysis['best_time_offset_ms']
        
        # Print analysis summary
        print(f"Best time offset: {time_offset_ms:.2f} ms")
        print(f"Position offset (y): {analysis['position_offset_y']:.4f} m")
        print(f"Camera entries: {analysis['num_camera_entries']}")
        print(f"Encoder entries: {analysis['num_encoder_entries']}")
        
        # Apply corrections in memory
        print("\nApplying corrections in memory...")
        corrected_data = apply_corrections_in_memory(filepath, time_offset_ms, fix_poses)
        
        # Show visualizations if enabled
        if show_vis:
            print(f"\n=== Comparison Visualization ===")
            print("Loading original data for comparison...")
            
            # Load original data for 3D visualization
            try:
                orig_vis_data = load_and_extract_data(str(filepath))
            except Exception as e:
                print(f"Error loading original 3D visualization: {e}")
                orig_vis_data = None
            
            # Extract visualization data from corrected memory data
            try:
                corrected_vis_data = extract_data_from_memory(corrected_data)
            except Exception as e:
                print(f"Error extracting corrected data: {e}")
                corrected_vis_data = None
            
            # Show timestamps_shifted graph (from original analysis)
            print("\nShowing timestamps_shifted analysis graph...")
            show_visualizations(filepath, marker_unit)
            
            # Show side-by-side 3D comparison (original left, corrected right)
            if orig_vis_data and corrected_vis_data:
                print("\nShowing side-by-side 3D comparison (Original left, Corrected right)...")
                visualize_3d_comparison(orig_vis_data, corrected_vis_data, f"{filepath.stem} - ")
            
            # Show all plots simultaneously
            print("\nDisplaying all visualizations (close windows to continue)...")
            plt.show(block=True)
            
            # Ask for user confirmation
            print("\n" + "="*50)
            print(f"Time offset to apply: {time_offset_ms:.2f} ms")
            print(f"Pose corrections: {'Yes' if fix_poses else 'No'}")
            print("="*50)
            
            while True:
                response = input("Save changes to file? (y/n): ").strip().lower()
                if response in ('y', 'yes',''):
                    print("Saving changes...")
                    break
                elif response in ('n', 'no'):
                    print("Changes discarded.")
                    return 0
                else:
                    print("Please enter 'y' or 'n'")
        
        if dry_run:
            print("\n[DRY-RUN] No changes written")
            return 0
        
        # Save the corrected data to file
        print(f"\nSaving corrected data to: {filepath.name}")
        save_recording(filepath, corrected_data)
        
        # Count corrections for summary
        pose_corrections = 0
        timestamp_corrections = 0
        for entry in corrected_data:
            if entry.get('mpu_unit') == 4:
                timestamp_corrections += 1
        
        print("\n=== Summary ===")
        print(f"Time offset applied: {time_offset_ms:.2f} ms")
        print(f"Timestamp corrections: {timestamp_corrections}")
        print(f"File saved: {filepath.name}")
        
        return 0
    
    else:
        print(f"Unknown mode: {mode}. Use 'analyze', 'fix', 'dry-run', or 'visualize'")
        return 1


if __name__ == "__main__":
    exit(main())
