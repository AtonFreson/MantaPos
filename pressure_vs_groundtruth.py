import os
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
from bisect import bisect_left, bisect_right
# Use the reference implementation from mantaPosLib
from mantaPosLib import global_reference_pos
from typing import Any
import matplotlib.pyplot as plt

def _maybe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def _interp_at(t: float, xs: List[float], ys: List[float]) -> Optional[float]:
    """Linear interpolate y at time t from (xs, ys). xs must be sorted ascending.
    Returns None if t is out of bounds or arrays too small."""
    if not xs or len(xs) != len(ys) or len(xs) < 2:
        return None
    if t < xs[0] or t > xs[-1]:
        return None
    i = bisect_left(xs, t)
    if i == 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    x0, x1 = xs[i-1], xs[i]
    y0, y1 = ys[i-1], ys[i]
    if x1 == x0:
        return y0
    alpha = (t - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0)


def process_recording(file_path: Path) -> Dict[str, Tuple[int, Optional[float], Optional[float]]]:
    """
    Build time-aligned ground truth using global_reference_pos with encoder distances:
      - z0: encoder.distance from mpu_unit 1
      - z1: encoder.distance from mpu_unit 2
      - frame_pos: encoder.distance from mpu_unit 0
    Align to pressure samples (mpu_unit 0) by linear interpolation in time (pressure.timestamp).
    Compare corrected pressure depths (depthN - depth_offsetN) to ground-truth depth per sample.

    Returns per-channel stats: (count, mean, std) of diffs.
    """
    enc_ts: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    enc_val: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    pressure_samples: List[Dict[str, float]] = []  # {t, d0, d1}
    offset_ts: Dict[int, List[float]] = {0: [], 1: []}
    offset_val: Dict[int, List[float]] = {0: [], 1: []}

    # First pass: collect timelines
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(entry, dict):
                    continue
                mpu = entry.get('mpu_unit')

                # Encoder timelines from all MPU units 0,1,2
                enc = entry.get('encoder')
                if isinstance(enc, dict) and mpu in (0, 1, 2):
                    t = enc.get('timestamp')
                    v = enc.get('distance')
                    t = None if t is None else float(t)
                    v = _maybe_float(v)
                    if t is not None and v is not None:
                        enc_ts[mpu].append(t)
                        enc_val[mpu].append(v)

                # Pressure samples on MPU 0 (reference timeline)
                pr = entry.get('pressure')
                if isinstance(pr, dict) and mpu == 0:
                    tpr = pr.get('timestamp')
                    d0 = _maybe_float(pr.get('depth0'))
                    d1 = _maybe_float(pr.get('depth1'))
                    if tpr is not None and d0 is not None and d1 is not None:
                        pressure_samples.append({'t': float(tpr), 'd0': d0, 'd1': d1})

                # Offsets are emitted from MPU 3 pressure packets; capture per-channel timelines
                if isinstance(pr, dict):
                    toff = pr.get('timestamp')
                    if toff is not None:
                        t_off = float(toff)
                        for ch in (0, 1):
                            key = f'depth_offset{ch}'
                            val = _maybe_float(pr.get(key))
                            if val is not None:
                                offset_ts[ch].append(t_off)
                                offset_val[ch].append(val)
    except FileNotFoundError:
        return {"ch0": (0, None, None), "ch1": (0, None, None)}

    # Ensure timelines are sorted
    for m in (0, 1, 2):
        if enc_ts[m] and enc_ts[m] != sorted(enc_ts[m]):
            # sort both by time
            pairs = sorted(zip(enc_ts[m], enc_val[m]), key=lambda p: p[0])
            enc_ts[m] = [p[0] for p in pairs]
            enc_val[m] = [p[1] for p in pairs]
    pressure_samples.sort(key=lambda s: s['t'])
    for ch in (0, 1):
        if offset_ts[ch] and offset_ts[ch] != sorted(offset_ts[ch]):
            pairs = sorted(zip(offset_ts[ch], offset_val[ch]), key=lambda p: p[0])
            offset_ts[ch] = [p[0] for p in pairs]
            offset_val[ch] = [p[1] for p in pairs]

    # Helper to get last offsets at time t
    def _offset_at_channel(t: float, ch: int) -> Optional[float]:
        ts = offset_ts[ch]
        vs = offset_val[ch]
        if not ts:
            return None
        if len(ts) == 1:
            return vs[0] if t >= ts[0] else None
        if t < ts[0]:
            return None
        if t >= ts[-1]:
            return vs[-1]
        return _interp_at(t, ts, vs)

    def offsets_at(t: float) -> Tuple[Optional[float], Optional[float]]:
        return (_offset_at_channel(t, 0), _offset_at_channel(t, 1))

    # Second pass: evaluate differences at each pressure sample time
    diffs: Dict[int, List[float]] = {0: [], 1: []}
    gt_depths: List[float] = []
    for s in pressure_samples:
        t = s['t']
        z0_enc = _interp_at(t, enc_ts[1], enc_val[1])  # MPU 1
        z1_enc = _interp_at(t, enc_ts[2], enc_val[2])  # MPU 2
        fpos   = _interp_at(t, enc_ts[0], enc_val[0])  # MPU 0

        if z0_enc is None or z1_enc is None or fpos is None:
            continue

        try:
            cam_pos, _ = global_reference_pos(z0_enc, z1_enc, fpos)
        except Exception:
            cam_pos = None
        if cam_pos is None:
            continue
        gt_depth = -float(cam_pos[2])

        off0, off1 = offsets_at(t)
        if off0 is None or off1 is None:
            continue

        z0_press = s['d0'] - off0
        z1_press = s['d1'] - off1
        diffs[0].append(z0_press - gt_depth)
        diffs[1].append(z1_press - gt_depth)
        gt_depths.append(gt_depth)

    # Summarize
    result: Dict[str, Tuple[int, Optional[float], Optional[float]]] = {}
    for ch in (0, 1):
        arr = diffs[ch]
        if len(arr) == 0:
            result[f"ch{ch}"] = (0, None, None)
        elif len(arr) == 1:
            result[f"ch{ch}"] = (1, arr[0], None)
        else:
            result[f"ch{ch}"] = (len(arr), mean(arr), stdev(arr))
    # Attach mean ground-truth depth for plotting
    result["mean_gt_depth"] = (len(gt_depths), (mean(gt_depths) if gt_depths else None), None)
    # Attach per-sample arrays for detailed plotting
    result["samples"] = {"gt": gt_depths, "diff0": diffs[0], "diff1": diffs[1]}
    return result


def main():
    root = Path(__file__).resolve().parent
    recordings_dir = root / 'recordings'
    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return

    files = sorted([p for p in recordings_dir.glob('*.json') if p.is_file()])
    # Ignore listed files/patterns
    ignore_substrings = [
        'data_handling_test',
        'depth 1 2 calib',
        'depth 3 4 calib',
    ]
    files = [p for p in files if not any(sub.lower() in p.name.lower() for sub in ignore_substrings)]
    if not files:
        print(f"No .json files found in {recordings_dir}")
        return

    print("file,                   count_ch0,    mean_diff_ch0_m, std_diff_ch0_m,    mean_diff_ch1_m, std_diff_ch1_m")
    # Collect plotting data: depth (x) vs mean diff (y) with std bars by marker type
    plot_data: Dict[str, Dict[str, List[float]]] = {
        'ArUco': {'x': [], 'y': [], 'yerr': []},
        'ChArUco': {'x': [], 'y': [], 'yerr': []},
    }

    # Collect per-sample data for requested plots
    single_target_labels = [
        'ArUco Quad 4.5-2m',
        'ArUco Single 2-4.5m',
        #'ArUco Single 2m Run2',
        #'ArUco Quad 2m Run2',
        #'ChArUco Single 2m Run2',
        #'ChArUco Quad 2m Run2',
    ]

    # Skip these from combined graph
    combined_graph_skip_labels = [
        'ChArUco Quad 7-4.5m',
        'ChArUco Quad 3.8-4.5m',
        'ChArUco Single 4.5-2m',
        'ChArUco Single 4.5-7m',
        'Aruco Quad 4.5-7m',
        'Aruco Quad 4.5-2m',
        'Aruco Single 2-4.5m',
        'Aruco Single 7-4.5m'
    ]

    sample_plots: Dict[str, Dict[str, List[float]]] = {}
    last_name_c0 = None
    for fp in files:
        stats = process_recording(fp)
        c0, m0, s0 = stats['ch0']
        c1, m1, s1 = stats['ch1']
        n_gt, mean_gt_depth, _ = stats.get('mean_gt_depth', (0, None, None))
        def fmt(v: Optional[float]) -> str:
            return f"{v:.6f}" if isinstance(v, (int, float)) else ""
        
        name_c0 = f"{fp.name}   @ {c0}:"
        if last_name_c0 and name_c0[0] != last_name_c0[0]:
            print()
        line = (
            f"{name_c0:<39}"
            f"{fmt(m0*100):<12}"
            f"+-{fmt(s0*100):<12}"
            f"{fmt(m1*100):<12}"
            f"+-{fmt(s1*100):<12}"
        )
        print(line)
        last_name_c0 = name_c0

        # Prepare plotting points (in meters) if we have a mean ground-truth depth
        label = 'ChArUco' if 'charuco' in fp.name.lower() else ('ArUco' if 'aruco' in fp.name.lower() else None)
        if (
            label
            and isinstance(mean_gt_depth, (int, float))
            and mean_gt_depth is not None
            and not any(skip.lower() in fp.name.lower() for skip in combined_graph_skip_labels)
        ):
            # Add both channels as separate points if they have stats
            for mean_val, std_val, count_val in ((m0, s0, c0), (m1, s1, c1)):
                if count_val and count_val > 0 and isinstance(mean_val, (int, float)):
                    plot_data[label]['x'].append(float(mean_gt_depth))
                    plot_data[label]['y'].append(float(mean_val))
                    plot_data[label]['yerr'].append(float(std_val) if isinstance(std_val, (int, float)) else 0.0)

        # Capture per-sample data for specific recordings
        lowname = fp.name.lower()
        for pretty_key in single_target_labels:
            key = pretty_key.lower()
            key_is_charuco = 'charuco' in key
            file_is_charuco = 'charuco' in lowname
            if key_is_charuco != file_is_charuco:
                continue
            if key in lowname:
                samples = stats.get('samples') or {}
                gt = samples.get('gt') or []
                d0 = samples.get('diff0') or []
                d1 = samples.get('diff1') or []
                sample_plots[pretty_key] = {'gt': list(gt), 'diff0': list(d0), 'diff1': list(d1)}

    # Plot and save
    if plt is None:
        print("matplotlib not available; skipping plot generation.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'ArUco': '#1f77b4', 'ChArUco': '#ff7f0e'}
    for label in ('ArUco', 'ChArUco'):
        xs = plot_data[label]['x']
        ys = plot_data[label]['y']
        es = plot_data[label]['yerr']
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt='o', linestyle='none', label=label, color=colors[label], alpha=0.85)
    ax.axhline(0.0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Mean diff (pressure - GT) (m)')
    ax.set_title('Pressure vs encoder ground truth by depth')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    # Don't show yet; more figures follow


    # Additional plots per user request: per-sample diff vs reference position for two specific recordings
    for pretty_name, data in sample_plots.items():
        gt = data.get('gt') or []
        d0 = data.get('diff0') or []
        d1 = data.get('diff1') or []
        if not gt or (not d0 and not d1):
            continue
        fig_i, ax_i = plt.subplots(figsize=(8, 5))
        if d0:
            ax_i.plot(gt, d0, '.', label='ch0', alpha=0.7, color='#2ca02c')
        if d1:
            ax_i.plot(gt, d1, '.', label='ch1', alpha=0.7, color='#d62728')
        ax_i.axhline(0.0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
        ax_i.set_xlabel('Reference position (m)')
        ax_i.set_ylabel('Pressure - reference (m)')
        ax_i.set_title(f'{pretty_name}: diff vs reference position')
        ax_i.grid(True, alpha=0.3)
        ax_i.legend()
        fig_i.tight_layout()

    # Render all figures
    plt.show(block=True)

if __name__ == '__main__':
    main()

# the reason for the difference between aruco and charuco was that they were recorded on different days, so lets handle them as seperate cases.
# the realistic implementation for this system is to start off by 'zeroing' the sensors to the ground truth