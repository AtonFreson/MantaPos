#!/usr/bin/env python3
"""
Fix camera pose ambiguity and timestamp offset in recording files.

This script permanently corrects the source JSON data so all downstream scripts
work correctly without needing per-script patching.

Corrections applied:
1. Camera timestamp offset (shift camera.timestamp to align with encoder data)
2. Camera pose ambiguity (resolve solvePnP planar marker flip using alter_to_correct_pose)
3. Adds metadata to track that corrections were applied

Usage:
    python fix_recording_data.py "ChArUco Quad 7-4.5m" --analyze
    python fix_recording_data.py "ChArUco Quad 7-4.5m" --fix --time-offset 671.41
    python fix_recording_data.py "ChArUco Quad 7-4.5m" --fix --auto
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

import mantaPosLib as manta
from fix_recording_datapoint import edit_recording

ROOT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT_DIR / "recordings"

# Marker configuration (must match mantaPos.py)
MARKERS_Z_LEVEL = -0.3 + 0.1124
QUAD_MARKER_POS = [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]]


def get_marker_config(marker_type: str) -> tuple[list, list]:
    """Get marker positions and rotations based on marker type.
    
    Note: Uses same configuration as data_processor.py for consistency.
    data_processor.py uses ChArUco Quad config for ALL quad marker files.
    """
    
    # Configuration matches data_processor.py exactly
    squares_vertically = 3
    square_length = 0.500 / squares_vertically
    corner_offset = square_length * squares_vertically / 2
    quad_order = [3, 0, 1, 2]
    
    quad_marker_pos = [
        [QUAD_MARKER_POS[quad_order[0]][0] + corner_offset, QUAD_MARKER_POS[quad_order[0]][1] - corner_offset, MARKERS_Z_LEVEL],
        [QUAD_MARKER_POS[quad_order[1]][0] + corner_offset, QUAD_MARKER_POS[quad_order[1]][1] + corner_offset, MARKERS_Z_LEVEL],
        [QUAD_MARKER_POS[quad_order[2]][0] - corner_offset, QUAD_MARKER_POS[quad_order[2]][1] - corner_offset, MARKERS_Z_LEVEL],
        [QUAD_MARKER_POS[quad_order[3]][0] + corner_offset, QUAD_MARKER_POS[quad_order[3]][1] - corner_offset, MARKERS_Z_LEVEL]
    ]
    quad_marker_rot = [[0, 0, 180], [0, 0, 180], [0, 0, 180], [0, 0, 180]]
    
    return quad_marker_pos, quad_marker_rot


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
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def analyze_timing(data: list[dict]) -> dict:
    """Analyze camera timing issues and find optimal offset."""
    
    # Extract camera and encoder data
    camera_entries = [d for d in data if d.get('mpu_unit') == 4 and 'camera' in d]
    encoder_entries = [d for d in data if d.get('mpu_unit') == 0 and 'encoder' in d]
    
    if not camera_entries or not encoder_entries:
        return {"error": "Missing camera or encoder data"}
    
    # Get camera timestamps and fps
    cam_timestamps = [e['camera']['timestamp'] for e in camera_entries]
    cam_fps = [e['camera']['fps'] for e in camera_entries]
    recv_times = [int(datetime.fromisoformat(e['recv_time']).timestamp() * 1000) for e in camera_entries]
    
    # Calculate basic stats
    mean_fps = np.mean(cam_fps)
    avg_timestep = 1000 / mean_fps
    
    # The key insight: recv_time is when PC received the data, camera.timestamp is from camera
    # The offset between them (after accounting for transmission delay) tells us the clock difference
    mean_recv_camera_diff = int(np.mean(np.array(recv_times) - np.array(cam_timestamps)))
    
    # Get camera position data for correlation with encoder
    camera_positions = []
    camera_ts_for_pos = []
    for e in camera_entries:
        # Find any available camera_pos marker
        for marker_idx in range(4):
            key = f'camera_pos_{marker_idx}'
            if key in e:
                camera_positions.append(e[key]['position'][1])  # y-position
                camera_ts_for_pos.append(e['camera']['timestamp'])
                break
    
    if not camera_positions:
        return {"error": "No camera position data found"}
    
    # Get encoder data
    encoder_positions = [-e['encoder']['distance'] for e in encoder_entries]
    encoder_timestamps = [e['encoder']['timestamp'] for e in encoder_entries]
    
    # Search for best time offset by minimizing position difference std
    search_range = range(-2000, 4000, 10)
    best_offset = 0
    best_std = float('inf')
    std_curve = []
    
    for offset in search_range:
        pos_diffs = []
        for i, cam_ts in enumerate(camera_ts_for_pos):
            adjusted_ts = cam_ts + offset
            # Find closest encoder timestamp
            closest_idx = np.argmin(np.abs(np.array(encoder_timestamps) - adjusted_ts))
            if abs(encoder_timestamps[closest_idx] - adjusted_ts) < 500:  # Within 500ms
                pos_diffs.append(encoder_positions[closest_idx] - camera_positions[i])
        
        if len(pos_diffs) > 10:
            # Remove outliers
            pos_diffs = np.array(pos_diffs)
            mask = np.abs(pos_diffs - np.mean(pos_diffs)) < 2 * np.std(pos_diffs)
            std_val = np.std(pos_diffs[mask])
            std_curve.append((offset, std_val))
            
            if std_val < best_std:
                best_std = std_val
                best_offset = offset
    
    # Calculate position offset (y-axis bias when stationary)
    initial_ts = cam_timestamps[0]
    stationary_positions = [
        camera_positions[i] for i, ts in enumerate(camera_ts_for_pos)
        if ts - initial_ts < 1000  # First 1 second
    ]
    position_offset = np.mean(stationary_positions) if stationary_positions else None
    
    return {
        "mean_fps": mean_fps,
        "fps_std": np.std(cam_fps),
        "avg_timestep_ms": avg_timestep,
        "mean_recv_camera_diff_ms": mean_recv_camera_diff,
        "best_time_offset_ms": best_offset,
        "best_offset_std": best_std,
        "position_offset_y": position_offset,
        "num_camera_entries": len(camera_entries),
        "num_encoder_entries": len(encoder_entries),
    }


def fix_pose_ambiguity(entry: dict, quad_marker_pos: list, quad_marker_rot: list) -> tuple[dict, int]:
    """
    Fix solvePnP planar marker ambiguity for a single entry.
    
    Returns:
        Tuple of (modified_entry, num_corrections)
    """
    corrections = 0
    
    if entry.get('mpu_unit') != 4:
        return entry, 0
    
    marker_order = [[0, 2], [1, 2], [2, 2], [3, 2]]  # From data_processor.py
    
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
            [quad_marker_pos[marker_order[marker_idx][0]], quad_marker_rot[marker_order[marker_idx][1]]]
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
            corrections += 1
    
    return entry, corrections


def fix_recording(
    filepath: Path,
    time_offset_ms: float | None = None,
    fix_poses: bool = True,
    backup: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Apply all corrections to a recording file using fix_recording_datapoint infrastructure.
    
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
    
    # Load data for analysis
    data = load_recording(filepath)
    print(f"  Loaded {len(data)} records")
    
    # Check if already has metadata (avoid double-fixing)
    existing_metadata = [d for d in data if d.get('mpu_unit') == -1]
    if existing_metadata and not dry_run:
        print(f"  WARNING: File already has fix metadata! Previous fix at: {existing_metadata[0].get('_fix_metadata', {}).get('fixed_at', 'unknown')}")
        print("  Skipping to avoid double-fixing.")
        return {"error": "Already fixed", "previous_metadata": existing_metadata[0]}
    
    # Determine marker type from filename
    filename = filepath.stem
    quad_marker_pos, quad_marker_rot = get_marker_config(filename)
    
    # Analyze if time_offset not provided
    if time_offset_ms is None:
        print("  Analyzing timing...")
        analysis = analyze_timing(data)
        if "error" in analysis:
            print(f"  Error: {analysis['error']}")
            return {"error": analysis['error']}
        
        time_offset_ms = analysis['best_time_offset_ms']
        print(f"  Auto-detected time offset: {time_offset_ms:.2f}ms (std: {analysis['best_offset_std']:.4f})")
        print(f"  Position offset (y): {analysis.get('position_offset_y', 'N/A')}")
    
    # Count what needs to be fixed
    pose_corrections = 0
    timestamp_corrections = 0
    
    # Calculate pose corrections needed
    if fix_poses:
        for entry in data:
            if entry.get('mpu_unit') == 4:
                _, num_fixes = fix_pose_ambiguity(entry.copy(), quad_marker_pos, quad_marker_rot)
                pose_corrections += num_fixes
                if 'camera' in entry:
                    timestamp_corrections += 1
    
    summary = {
        "file": str(filepath),
        "time_offset_ms": time_offset_ms,
        "pose_corrections": pose_corrections,
        "timestamp_corrections": timestamp_corrections,
        "total_records": len(data),
    }
    
    print(f"  Timestamp corrections needed: {timestamp_corrections}")
    print(f"  Pose corrections needed: {pose_corrections}")
    
    if dry_run:
        print("  [DRY-RUN] No changes written")
        return summary
    
    # Use edit_recording for actual modifications
    backup_mode = "backup" if backup else "no-backup"
    file_arg = filepath.stem  # filename without .json
    
    # 1. Fix camera timestamps
    print(f"\n  Applying timestamp offset: +{time_offset_ms:.2f}ms")
    edit_recording(
        backup_mode, "addition", file_arg,
        "camera", "timestamp",
        mpu_unit=4,
        value=round(time_offset_ms),
    )
    
    # 2. Fix pose ambiguity - this requires custom logic per entry
    if fix_poses:
        print(f"  Applying pose corrections...")
        apply_pose_corrections(filepath, quad_marker_pos, quad_marker_rot)
    
    print(f"\n  Corrections applied to: {filepath.name}")
    
    return summary


def apply_pose_corrections(filepath: Path, quad_marker_pos: list, quad_marker_rot: list) -> int:
    """
    Apply pose corrections to all camera entries in the file.
    Uses direct file rewrite since each entry needs individual calculation.
    """
    import shutil
    from datetime import datetime
    
    # Load data
    data = load_recording(filepath)
    corrections = 0
    
    marker_order = [[0, 2], [1, 2], [2, 2], [3, 2]]
    
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
                [quad_marker_pos[marker_order[marker_idx][0]], quad_marker_rot[marker_order[marker_idx][1]]]
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
                corrections += 1
    
    # Add metadata
    metadata = {
        "mpu_unit": -1,
        "_fix_metadata": {
            "fixed_at": datetime.now().isoformat(),
            "script": "fix_mantaPos_data.py",
            "pose_corrections": corrections,
        }
    }
    data.insert(0, metadata)
    
    # Save
    save_recording(filepath, data)
    print(f"    Pose corrections applied: {corrections}")
    
    return corrections



def main():
    # ============================================================
    # CONFIGURATION - Edit these values directly
    # ============================================================
    
    filename = "ChArUco Quad 7-4.5m"  # Recording filename (without .json)
    
    mode = "analyze"  # Options: "analyze", "fix", "dry-run"
    
    time_offset_ms = None  # Set to None for auto-detection, or specify manually (e.g., 671.41)
    fix_poses = True       # Fix solvePnP pose ambiguity
    create_backup = True   # Create backup before modifying
    
    # ============================================================
    
    # Resolve file path
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1
    
    if mode == "analyze":
        data = load_recording(filepath)
        print(f"\nAnalyzing: {filepath.name}")
        print(f"Loaded {len(data)} records")
        
        analysis = analyze_timing(data)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return 1
        
        print(f"\n=== Timing Analysis ===")
        print(f"Mean FPS: {analysis['mean_fps']:.3f} ± {analysis['fps_std']:.3f} Hz")
        print(f"Avg timestep: {analysis['avg_timestep_ms']:.2f} ms")
        print(f"Mean recv-camera diff: {analysis['mean_recv_camera_diff_ms']} ms")
        print(f"\nBest time offset: {analysis['best_time_offset_ms']:.2f} ms (std: {analysis['best_offset_std']:.4f})")
        if analysis['position_offset_y'] is not None:
            print(f"Position offset (y): {analysis['position_offset_y']:.4f} m")
        else:
            print("Position offset: N/A")
        print(f"\nCamera entries: {analysis['num_camera_entries']}")
        print(f"Encoder entries: {analysis['num_encoder_entries']}")
        
        return 0
    
    elif mode in ("fix", "dry-run"):
        dry_run = (mode == "dry-run")
        
        result = fix_recording(
            filepath,
            time_offset_ms=time_offset_ms,
            fix_poses=fix_poses,
            backup=create_backup,
            dry_run=dry_run,
        )
        
        if "error" in result:
            return 1
        
        print("\n=== Summary ===")
        print(f"Time offset applied: {result['time_offset_ms']:.2f} ms")
        print(f"Pose corrections: {result['pose_corrections']}")
        print(f"Timestamp corrections: {result['timestamp_corrections']}")
        
        return 0
    
    else:
        print(f"Unknown mode: {mode}. Use 'analyze', 'fix', or 'dry-run'")
        return 1


if __name__ == "__main__":
    exit(main())
