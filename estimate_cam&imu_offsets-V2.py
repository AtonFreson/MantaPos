"""
Estimate fixed IMU and camera rotation offsets from the recorded MantaPos data.

The estimates are printed separately for ArUco and ChArUco recordings because the
frame was remounted between those days.

Outputs use the same phi/theta/psi state convention as MantaUKF:

    R = scipy.spatial.transform.Rotation.from_euler("ZYX", [psi, theta, phi])

That is, the printed values can be read as the UKF state order
[phi, theta, psi] in radians, with degrees beside them for humans.

Methods:
  * IMU offset:
      Fit one fixed rotation from IMU local acceleration to the moving reference
      frame. The target vector is gravity plus smoothed reference acceleration,
      transformed into the instantaneous reference frame using ref_rot.

  * Camera position-frame offset:
      Fit one fixed rotation from reference-frame motion deltas to camera
      position deltas. Each run gets its own translation removed by centering,
      so the fit is about axis leakage/skew, not absolute origin.

  * Camera reported-pose orientation offset:
      Diagnostic only. Averages the stored camera rotation fields relative to
      ref_rot using rotation averaging. This can be noisier than the
      position-frame fit if the solvePnP orientation data is unstable.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as R
except ModuleNotFoundError as exc:
    print(
        "This script requires numpy and scipy. Run it with the same Python "
        "environment used for the existing MantaPos scripts.",
        file=sys.stderr,
    )
    raise exc


ROOT = Path(__file__).resolve().parent
DEFAULT_STREAM = ROOT / "MantaPos_data.json"
DEFAULT_TIME_CORRECTIONS = ROOT / "camera_time_corrections.txt"

GRAVITY_GLOBAL = np.array([0.0, 0.0, -9.81])

# Match mantaPos-runKalmanFilter.py reference reconstruction. Translation
# constants do not affect the centered rotation fits, but keeping the same model
# avoids subtle sign/convention mismatches.
TRACK_WIDTH_M = 3.1305
FRAME_Y_POS_OFFSET_M = 0.196
CAMERA_X_OFFSET_M = -1.5406
CAMERA_Y_OFFSET_M = 1.4477259577554185
CAMERA_Z_OFFSET_M = -0.1186 + 0.225 - 0.187


@dataclass
class ReferenceSample:
    timestamp_ms: int
    position: np.ndarray
    rotation: np.ndarray  # [phi, theta, psi] in radians


@dataclass
class ImuSample:
    timestamp_ms: int
    acceleration: np.ndarray


@dataclass
class CameraPose:
    position: np.ndarray
    rotation_deg: np.ndarray | None


@dataclass
class CameraSample:
    timestamp_ms: int
    position: np.ndarray
    rotations_deg: list[np.ndarray]
    sigma_m: float
    marker_count: int


@dataclass
class RunData:
    name: str
    marker_type: str
    reference: list[ReferenceSample] = field(default_factory=list)
    imu: list[ImuSample] = field(default_factory=list)
    camera_by_ts: dict[int, list[CameraPose]] = field(default_factory=lambda: defaultdict(list))
    camera: list[CameraSample] = field(default_factory=list)


@dataclass
class CameraPair:
    run_name: str
    ref_position: np.ndarray
    ref_rotation: np.ndarray
    camera_position: np.ndarray
    rotations_deg: list[np.ndarray]
    sigma_m: float


@dataclass
class FitSummary:
    name: str
    rotation: R
    euler_rad: np.ndarray
    euler_std_rad: np.ndarray
    rms_raw: float
    median_raw: float
    n_samples: int
    n_runs: int
    observability: np.ndarray
    condition: float
    confidence: str
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate IMU and camera rotation offsets for MantaUKF."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_STREAM,
        help="Path to streamlined MantaPos_data.json.",
    )
    parser.add_argument(
        "--time-corrections",
        type=Path,
        default=DEFAULT_TIME_CORRECTIONS,
        help="Per-run camera timestamp corrections to add in milliseconds.",
    )
    parser.add_argument(
        "--no-time-corrections",
        action="store_true",
        help="Do not apply camera_time_corrections.txt to camera samples.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["ArUco", "ChArUco"],
        default=["ArUco", "ChArUco"],
        help="Marker families to estimate.",
    )
    parser.add_argument(
        "--max-ref-gap-ms",
        type=float,
        default=350.0,
        help="Maximum allowed nearest reference timestamp gap for interpolation.",
    )
    parser.add_argument(
        "--camera-sigma-m",
        type=float,
        default=0.035,
        help="Base 1-sigma camera position noise used for confidence estimates.",
    )
    parser.add_argument(
        "--imu-sigma-ms2",
        type=float,
        default=0.45,
        help="Base 1-sigma IMU/reference acceleration mismatch for confidence estimates.",
    )
    parser.add_argument(
        "--imu-deriv-window-s",
        type=float,
        default=1.25,
        help="Savitzky-Golay window, in seconds, for reference acceleration.",
    )
    parser.add_argument(
        "--imu-deriv-polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order for reference acceleration.",
    )
    parser.add_argument(
        "--imu-dynamic-gain",
        type=float,
        default=2.0,
        help="Extra weight for samples with reference acceleration, for yaw observability.",
    )
    parser.add_argument(
        "--max-imu-accel-norm",
        type=float,
        default=13.0,
        help="Drop IMU samples with acceleration norm above this value.",
    )
    parser.add_argument(
        "--min-run-camera-samples",
        type=int,
        default=12,
        help="Minimum paired camera samples in a run before it contributes.",
    )
    parser.add_argument(
        "--min-run-imu-samples",
        type=int,
        default=30,
        help="Minimum paired IMU samples in a run before it contributes.",
    )
    parser.add_argument(
        "--skip-camera-orientation",
        action="store_true",
        help="Skip diagnostic averaging of stored camera rotation fields.",
    )
    return parser.parse_args()


def marker_type_from_run(name: str) -> str:
    if name.startswith("ChArUco"):
        return "ChArUco"
    if name.startswith("ArUco"):
        return "ArUco"
    return "Unknown"


def reference_pose_from_stream(z0: float, z1: float, frame_track: float) -> tuple[np.ndarray, np.ndarray]:
    opp = z0 - z1
    hyp = math.sqrt(opp * opp + TRACK_WIDTH_M * TRACK_WIDTH_M)

    x = CAMERA_X_OFFSET_M
    y = -frame_track / hyp * TRACK_WIDTH_M + CAMERA_Y_OFFSET_M

    frame_pos = frame_track + FRAME_Y_POS_OFFSET_M
    z = z0 - opp * frame_pos / hyp + CAMERA_Z_OFFSET_M

    phi = math.atan2(opp, TRACK_WIDTH_M)
    return np.array([x, y, z], dtype=float), np.array([phi, 0.0, 0.0], dtype=float)


def ukf_rotation(phi_theta_psi: np.ndarray) -> R:
    phi, theta, psi = np.asarray(phi_theta_psi, dtype=float)
    return R.from_euler("ZYX", [psi, theta, phi], degrees=False)


def euler_ukf(rot: R) -> np.ndarray:
    psi, theta, phi = rot.as_euler("ZYX", degrees=False)
    return np.array([phi, theta, psi], dtype=float)


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def robust_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    med = np.median(values)
    return 1.4826 * np.median(np.abs(values - med))


def load_time_corrections(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}

    corrections: dict[str, float] = {}
    pattern = re.compile(r'"([^"]+)"\s+(-?\d+(?:\.\d+)?)')
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                corrections[match.group(1)] = float(match.group(2))
    return corrections


def parse_streamlined_data(path: Path) -> list[RunData]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    runs: list[RunData] = []
    current: RunData | None = None
    latest_z1: float | None = None
    latest_z2: float | None = None

    with path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("###"):
                name = line.strip("#").strip().strip("'")
                current = RunData(name=name, marker_type=marker_type_from_run(name))
                runs.append(current)
                latest_z1 = None
                latest_z2 = None
                continue

            if current is None:
                continue

            try:
                ts_str, rest = line.split(" - ", 1)
                timestamp_ms = int(ts_str)
                dtype_str, value_str = rest.split(":", 1)
                dtype = dtype_str.strip().strip("'")
                value = ast.literal_eval(value_str.strip())
            except (ValueError, SyntaxError) as exc:
                print(f"Skipping malformed line {line_no}: {exc}", file=sys.stderr)
                continue

            if dtype == "ref_pos_z1":
                latest_z1 = float(value)
            elif dtype == "ref_pos_z2":
                latest_z2 = float(value)
            elif dtype == "ref_pos_track":
                if latest_z1 is None or latest_z2 is None or value is None:
                    continue
                pos, rot = reference_pose_from_stream(latest_z1, latest_z2, float(value))
                current.reference.append(ReferenceSample(timestamp_ms, pos, rot))
            elif dtype == "imu":
                accel = value.get("acceleration", {})
                vec = np.array(
                    [
                        accel.get("x", np.nan),
                        accel.get("y", np.nan),
                        accel.get("z", np.nan),
                    ],
                    dtype=float,
                )
                if np.all(np.isfinite(vec)):
                    current.imu.append(ImuSample(timestamp_ms, vec))
            elif dtype.startswith("camera_pos_"):
                pos = value.get("position")
                if pos is None:
                    continue
                position = np.asarray(pos, dtype=float)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    continue
                rotation_value = value.get("rotation")
                rotation = None
                if rotation_value is not None:
                    rotation = np.asarray(rotation_value, dtype=float)
                    if rotation.shape != (3,) or not np.all(np.isfinite(rotation)):
                        rotation = None
                current.camera_by_ts[timestamp_ms].append(CameraPose(position, rotation))

    for run in runs:
        finalize_camera_samples(run)

    return runs


def finalize_camera_samples(run: RunData) -> None:
    samples: list[CameraSample] = []
    for timestamp_ms, poses in sorted(run.camera_by_ts.items()):
        valid = [
            pose
            for pose in poses
            if np.all(np.isfinite(pose.position))
            and abs(pose.position[0]) < 5.0
            and abs(pose.position[1]) < 8.0
            and abs(pose.position[2]) < 12.0
        ]
        if not valid:
            continue

        positions = np.vstack([pose.position for pose in valid])
        position = np.mean(positions, axis=0)
        if len(valid) > 1:
            marker_spread = float(np.sqrt(np.mean(np.sum((positions - position) ** 2, axis=1))))
        else:
            marker_spread = 0.0

        rotations = [pose.rotation_deg for pose in valid if pose.rotation_deg is not None]
        sigma = max(0.01, marker_spread / math.sqrt(max(1, len(valid))))
        samples.append(CameraSample(timestamp_ms, position, rotations, sigma, len(valid)))

    run.camera = samples
    run.camera_by_ts.clear()


def unique_reference_arrays(run: RunData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not run.reference:
        return np.array([]), np.empty((0, 3)), np.empty((0, 3))

    grouped: dict[int, list[ReferenceSample]] = defaultdict(list)
    for sample in run.reference:
        grouped[sample.timestamp_ms].append(sample)

    times = []
    positions = []
    rotations = []
    for timestamp_ms in sorted(grouped):
        group = grouped[timestamp_ms]
        times.append(timestamp_ms)
        positions.append(np.mean([s.position for s in group], axis=0))
        rotations.append(np.mean([s.rotation for s in group], axis=0))

    return np.asarray(times, dtype=float), np.vstack(positions), np.vstack(rotations)


def interpolate_reference(
    run: RunData,
    timestamp_ms: float,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    times, positions, rotations = unique_reference_arrays(run)
    if len(times) < 2 or timestamp_ms < times[0] or timestamp_ms > times[-1]:
        return None

    idx = int(np.searchsorted(times, timestamp_ms))
    candidates = []
    if 0 <= idx < len(times):
        candidates.append(abs(times[idx] - timestamp_ms))
    if 0 <= idx - 1 < len(times):
        candidates.append(abs(times[idx - 1] - timestamp_ms))
    if not candidates or min(candidates) > max_gap_ms:
        return None

    pos = np.array([np.interp(timestamp_ms, times, positions[:, i]) for i in range(3)])
    rot = np.array([np.interp(timestamp_ms, times, rotations[:, i]) for i in range(3)])
    return pos, rot


def build_camera_pairs(
    runs: list[RunData],
    marker_type: str,
    time_corrections: dict[str, float],
    apply_time_corrections: bool,
    max_gap_ms: float,
) -> dict[str, list[CameraPair]]:
    pairs_by_run: dict[str, list[CameraPair]] = {}

    for run in runs:
        if run.marker_type != marker_type or not run.camera:
            continue

        correction = time_corrections.get(run.name, 0.0) if apply_time_corrections else 0.0
        pairs: list[CameraPair] = []
        for camera_sample in run.camera:
            ref = interpolate_reference(run, camera_sample.timestamp_ms + correction, max_gap_ms)
            if ref is None:
                continue
            ref_pos, ref_rot = ref
            pairs.append(
                CameraPair(
                    run.name,
                    ref_pos,
                    ref_rot,
                    camera_sample.position,
                    camera_sample.rotations_deg,
                    camera_sample.sigma_m,
                )
            )
        if pairs:
            pairs_by_run[run.name] = pairs

    return pairs_by_run


def build_camera_fit_arrays(
    pairs_by_run: dict[str, list[CameraPair]],
    min_run_samples: int,
    camera_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    src = []
    dst = []
    sigma = []
    sample_runs = []
    used_runs = []

    for run_name, pairs in sorted(pairs_by_run.items()):
        if len(pairs) < min_run_samples:
            continue

        ref_positions = np.vstack([p.ref_position for p in pairs])
        cam_positions = np.vstack([p.camera_position for p in pairs])
        ref_center = np.median(ref_positions, axis=0)
        cam_center = np.median(cam_positions, axis=0)

        run_src = []
        run_dst = []
        run_sigma = []
        for pair in pairs:
            ref_delta_global = pair.ref_position - ref_center
            ref_delta = ukf_rotation(pair.ref_rotation).inv().apply(ref_delta_global)
            cam_delta = pair.camera_position - cam_center
            if np.linalg.norm(ref_delta) < 0.02 or np.linalg.norm(cam_delta) < 0.02:
                continue
            run_src.append(ref_delta)
            run_dst.append(cam_delta)
            run_sigma.append(math.sqrt(camera_sigma_m * camera_sigma_m + pair.sigma_m * pair.sigma_m))

        if len(run_src) >= min_run_samples:
            src.extend(run_src)
            dst.extend(run_dst)
            sigma.extend(run_sigma)
            sample_runs.extend([run_name] * len(run_src))
            used_runs.append(run_name)

    if not src:
        return np.empty((0, 3)), np.empty((0, 3)), np.array([]), [], []

    return np.vstack(src), np.vstack(dst), np.asarray(sigma), sample_runs, used_runs


def interpolate_uniform(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty((len(grid), values.shape[1]), dtype=float)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(grid, times, values[:, i])
    return out


def odd_window_length(window_s: float, dt_s: float, n: int, polyorder: int) -> int:
    if n <= polyorder + 2:
        return 0

    estimated = max(polyorder + 2, int(round(window_s / dt_s)))
    if estimated % 2 == 0:
        estimated += 1
    if estimated > n:
        estimated = n if n % 2 == 1 else n - 1
    if estimated <= polyorder:
        estimated = polyorder + 2
        if estimated % 2 == 0:
            estimated += 1
    return estimated if estimated <= n else 0


def build_imu_fit_arrays(
    runs: list[RunData],
    marker_type: str,
    max_gap_ms: float,
    deriv_window_s: float,
    polyorder: int,
    max_accel_norm: float,
    base_sigma_ms2: float,
    dynamic_gain: float,
    min_run_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    local_vectors = []
    target_vectors = []
    sigma = []
    sample_runs = []
    used_runs = []
    linear_accel_mag = []

    for run in runs:
        if run.marker_type != marker_type or not run.imu:
            continue

        ref_times, ref_positions, ref_rotations = unique_reference_arrays(run)
        if len(ref_times) < 8:
            continue

        dt_ms = float(np.median(np.diff(ref_times)))
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            continue

        grid = np.arange(ref_times[0], ref_times[-1] + 0.5 * dt_ms, dt_ms)
        if len(grid) < 8:
            continue

        dt_s = dt_ms / 1000.0
        window = odd_window_length(deriv_window_s, dt_s, len(grid), polyorder)
        if window == 0:
            continue

        ref_pos_grid = interpolate_uniform(ref_times, ref_positions, grid)
        ref_rot_grid = interpolate_uniform(ref_times, ref_rotations, grid)

        ref_acc_grid = np.zeros_like(ref_pos_grid)
        for axis in (1, 2):
            ref_acc_grid[:, axis] = savgol_filter(
                ref_pos_grid[:, axis],
                window_length=window,
                polyorder=polyorder,
                deriv=2,
                delta=dt_s,
                mode="interp",
            )

        run_local = []
        run_target = []
        run_sigma = []
        run_dynamic = []

        for imu_sample in run.imu:
            accel = imu_sample.acceleration
            accel_norm = float(np.linalg.norm(accel))
            if not np.isfinite(accel_norm) or accel_norm > max_accel_norm or accel_norm < 7.0:
                continue

            timestamp = float(imu_sample.timestamp_ms)
            if timestamp < grid[0] or timestamp > grid[-1]:
                continue

            ref_idx = int(np.searchsorted(ref_times, timestamp))
            gaps = []
            if 0 <= ref_idx < len(ref_times):
                gaps.append(abs(ref_times[ref_idx] - timestamp))
            if 0 <= ref_idx - 1 < len(ref_times):
                gaps.append(abs(ref_times[ref_idx - 1] - timestamp))
            if not gaps or min(gaps) > max_gap_ms:
                continue

            ref_acc = np.array([np.interp(timestamp, grid, ref_acc_grid[:, i]) for i in range(3)])
            ref_rot = np.array([np.interp(timestamp, grid, ref_rot_grid[:, i]) for i in range(3)])
            ref_frame_rot = ukf_rotation(ref_rot)

            target_global = GRAVITY_GLOBAL + ref_acc
            target_ref = ref_frame_rot.inv().apply(target_global)

            dynamic_mag = float(np.linalg.norm(ref_acc))
            dynamic_weight = 1.0 + dynamic_gain * min(1.0, dynamic_mag / 0.8)
            run_local.append(accel)
            run_target.append(target_ref)
            run_sigma.append(base_sigma_ms2 / math.sqrt(dynamic_weight))
            run_dynamic.append(dynamic_mag)

        if len(run_local) < min_run_samples:
            continue

        used_runs.append(run.name)
        local_vectors.extend(run_local)
        target_vectors.extend(run_target)
        sigma.extend(run_sigma)
        sample_runs.extend([run.name] * len(run_local))
        linear_accel_mag.extend(run_dynamic)

    if not local_vectors:
        return np.empty((0, 3)), np.empty((0, 3)), np.array([]), [], [], np.array([])

    return (
        np.vstack(local_vectors),
        np.vstack(target_vectors),
        np.asarray(sigma),
        sample_runs,
        used_runs,
        np.asarray(linear_accel_mag),
    )


def initial_alignment(src: np.ndarray, dst: np.ndarray, sigma: np.ndarray) -> R:
    if len(src) < 2:
        return R.identity()
    weights = 1.0 / np.maximum(sigma, 1e-9) ** 2
    try:
        rotation, _ = R.align_vectors(dst, src, weights=weights)
        return rotation
    except Exception:
        return R.identity()


def residuals_for_rotation(rotvec: np.ndarray, src: np.ndarray, dst: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    pred = R.from_rotvec(rotvec).apply(src)
    return ((pred - dst) / sigma[:, None]).ravel()


def fit_rotation(
    name: str,
    src: np.ndarray,
    dst: np.ndarray,
    sigma: np.ndarray,
    sample_runs: list[str],
    used_runs: list[str],
    raw_unit_scale: float,
    trim_iterations: int = 2,
) -> FitSummary | None:
    if len(src) < 6:
        return None

    keep = np.ones(len(src), dtype=bool)
    rot0 = initial_alignment(src, dst, sigma).as_rotvec()

    for _ in range(trim_iterations):
        result = least_squares(
            residuals_for_rotation,
            rot0,
            args=(src[keep], dst[keep], sigma[keep]),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=300,
        )
        rot0 = result.x

        raw_res = R.from_rotvec(result.x).apply(src) - dst
        per_sample = np.linalg.norm(raw_res, axis=1)
        med = np.median(per_sample[keep])
        mad = robust_mad(per_sample[keep])
        if not np.isfinite(mad) or mad <= 1e-12:
            break
        threshold = med + 6.0 * mad
        new_keep = per_sample <= threshold
        if np.array_equal(new_keep, keep) or np.count_nonzero(new_keep) < 6:
            break
        keep = new_keep

    result = least_squares(
        residuals_for_rotation,
        rot0,
        args=(src[keep], dst[keep], sigma[keep]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )
    final_rot = R.from_rotvec(result.x)
    raw_res = final_rot.apply(src[keep]) - dst[keep]
    per_sample_raw = np.linalg.norm(raw_res, axis=1)
    residual_norm = residuals_for_rotation(result.x, src[keep], dst[keep], sigma[keep])

    euler = euler_ukf(final_rot)
    cov_rotvec = covariance_from_result(result, residual_norm)
    euler_std = euler_std_from_cov(result.x, cov_rotvec)

    obs = observability_ratios(src[keep], 1.0 / sigma[keep] ** 2)
    condition = float(obs[0] / max(obs[-1], 1e-12)) if len(obs) else float("inf")
    confidence, notes = classify_confidence(euler_std, obs, residual_norm)

    kept_run_names = sorted(set(np.asarray(sample_runs, dtype=object)[keep].tolist()))
    dropped = len(src) - np.count_nonzero(keep)
    if dropped:
        notes.append(f"trimmed {dropped} outlier samples")

    if raw_unit_scale != 1.0:
        raw_res_for_print = per_sample_raw * raw_unit_scale
    else:
        raw_res_for_print = per_sample_raw

    return FitSummary(
        name=name,
        rotation=final_rot,
        euler_rad=euler,
        euler_std_rad=euler_std,
        rms_raw=float(np.sqrt(np.mean(raw_res_for_print**2))),
        median_raw=float(np.median(raw_res_for_print)),
        n_samples=int(np.count_nonzero(keep)),
        n_runs=len(kept_run_names) if kept_run_names else len(used_runs),
        observability=obs,
        condition=condition,
        confidence=confidence,
        notes=notes,
    )


def covariance_from_result(result, residual_norm: np.ndarray) -> np.ndarray:
    jac = result.jac
    if jac.size == 0 or len(residual_norm) <= jac.shape[1]:
        return np.full((3, 3), np.nan)

    _, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
    if len(singular_values) == 0:
        return np.full((3, 3), np.nan)

    tol = np.finfo(float).eps * max(jac.shape) * singular_values[0]
    inv_s = np.array([1.0 / s if s > tol else 0.0 for s in singular_values])
    cov = vt.T @ np.diag(inv_s**2) @ vt
    dof = max(1, len(residual_norm) - jac.shape[1])
    variance = float(np.sum(residual_norm**2) / dof)
    return cov * variance


def euler_std_from_cov(rotvec: np.ndarray, cov_rotvec: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(cov_rotvec)):
        return np.full(3, np.nan)

    base = euler_ukf(R.from_rotvec(rotvec))
    jac = np.zeros((3, 3), dtype=float)
    eps = 1e-6
    for i in range(3):
        perturbed = rotvec.copy()
        perturbed[i] += eps
        diff = wrap_pi(euler_ukf(R.from_rotvec(perturbed)) - base)
        jac[:, i] = diff / eps

    cov_euler = jac @ cov_rotvec @ jac.T
    diag = np.diag(cov_euler)
    diag = np.where(diag >= 0.0, diag, np.nan)
    return np.sqrt(diag)


def observability_ratios(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(vectors) < 3:
        return np.array([1.0, 0.0, 0.0])

    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-9
    if np.count_nonzero(valid) < 3:
        return np.array([1.0, 0.0, 0.0])

    weighted = vectors[valid] * np.sqrt(weights[valid])[:, None]
    _, singular_values, _ = np.linalg.svd(weighted, full_matrices=False)
    if singular_values[0] <= 0:
        return np.array([1.0, 0.0, 0.0])
    ratios = singular_values / singular_values[0]
    if len(ratios) < 3:
        ratios = np.pad(ratios, (0, 3 - len(ratios)))
    return ratios[:3]


def classify_confidence(
    euler_std_rad: np.ndarray,
    observability: np.ndarray,
    residual_norm: np.ndarray,
) -> tuple[str, list[str]]:
    notes = []
    std_deg = np.rad2deg(euler_std_rad)
    max_ci95_deg = float(np.nanmax(1.96 * std_deg)) if np.any(np.isfinite(std_deg)) else float("inf")
    rms_norm = float(np.sqrt(np.mean(residual_norm**2))) if len(residual_norm) else float("inf")
    min_obs = float(observability[-1]) if len(observability) else 0.0

    if min_obs < 0.01:
        notes.append("weak 3-axis excitation; at least one angle is poorly observable")
    if rms_norm > 3.0:
        notes.append("fit residual is large relative to configured sensor noise")
    if max_ci95_deg > 10.0:
        notes.append("wide 95% interval on at least one angle")

    if min_obs < 0.01 or rms_norm > 4.0 or max_ci95_deg > 10.0:
        return "LOW", notes
    if min_obs < 0.05 or rms_norm > 2.0 or max_ci95_deg > 3.0:
        return "MEDIUM", notes
    return "HIGH", notes


def fit_camera_position_offset(
    pairs_by_run: dict[str, list[CameraPair]],
    min_run_samples: int,
    camera_sigma_m: float,
) -> FitSummary | None:
    src, dst, sigma, sample_runs, used_runs = build_camera_fit_arrays(
        pairs_by_run,
        min_run_samples,
        camera_sigma_m,
    )
    return fit_rotation(
        "camera position-frame offset",
        src,
        dst,
        sigma,
        sample_runs,
        used_runs,
        raw_unit_scale=1.0,
    )


def fit_imu_offset(runs: list[RunData], marker_type: str, args: argparse.Namespace) -> tuple[FitSummary | None, np.ndarray]:
    src, dst, sigma, sample_runs, used_runs, dynamic_mag = build_imu_fit_arrays(
        runs,
        marker_type,
        args.max_ref_gap_ms,
        args.imu_deriv_window_s,
        args.imu_deriv_polyorder,
        args.max_imu_accel_norm,
        args.imu_sigma_ms2,
        args.imu_dynamic_gain,
        args.min_run_imu_samples,
    )
    summary = fit_rotation(
        "IMU local-to-reference offset",
        src,
        dst,
        sigma,
        sample_runs,
        used_runs,
        raw_unit_scale=1.0,
    )
    if summary is not None:
        summary.notes.insert(
            0,
            "IMU psi/yaw is diagnostic: it depends on clean reference-frame "
            "horizontal acceleration, while phi/theta are primarily gravity anchored",
        )
        summary.notes.insert(
            1,
            "endpoint impacts or rig twist can bias psi/yaw even when the least-squares fit converges",
        )
        if summary.confidence == "HIGH":
            summary.confidence = "MEDIUM"
            summary.notes.append("overall IMU confidence capped because yaw is not statically observable")
    return summary, dynamic_mag


def robust_rotation_mean(rotations: list[R], max_angle_deg: float = 20.0) -> tuple[R, np.ndarray, np.ndarray]:
    if not rotations:
        return R.identity(), np.array([], dtype=bool), np.array([])

    stack = R.from_quat([rot.as_quat() for rot in rotations])
    keep = np.ones(len(rotations), dtype=bool)
    mean = stack.mean()

    for _ in range(4):
        kept_stack = R.from_quat([rotations[i].as_quat() for i in range(len(rotations)) if keep[i]])
        mean = kept_stack.mean()
        errors = np.array([(mean.inv() * rotations[i]).magnitude() for i in range(len(rotations))])
        threshold = math.radians(max_angle_deg)
        mad = robust_mad(errors[keep])
        if np.isfinite(mad) and mad > 1e-12:
            threshold = max(threshold, float(np.median(errors[keep]) + 6.0 * mad))
        new_keep = errors <= threshold
        if np.array_equal(new_keep, keep) or np.count_nonzero(new_keep) < 3:
            break
        keep = new_keep

    errors = np.array([(mean.inv() * rotations[i]).magnitude() for i in range(len(rotations))])
    return mean, keep, errors


def fit_camera_orientation_diagnostic(
    pairs_by_run: dict[str, list[CameraPair]],
) -> FitSummary | None:
    offsets: list[R] = []
    run_names: list[str] = []

    for run_name, pairs in sorted(pairs_by_run.items()):
        for pair in pairs:
            if not pair.rotations_deg:
                continue
            try:
                cam_rots = R.from_euler("xyz", np.vstack(pair.rotations_deg), degrees=True)
                cam_global = cam_rots.mean()
            except Exception:
                continue

            ref_global = ukf_rotation(pair.ref_rotation)
            # Stored camera rotation is camera-local to global. Remove ref_rot so
            # this diagnostic reports camera-local to moving-reference frame.
            offsets.append(ref_global.inv() * cam_global)
            run_names.append(run_name)

    if len(offsets) < 3:
        return None

    mean, keep, errors = robust_rotation_mean(offsets)
    kept_errors = errors[keep]
    euler = euler_ukf(mean)
    std_scalar = float(np.std(kept_errors) / math.sqrt(max(1, len(kept_errors))))
    euler_std = np.full(3, std_scalar)
    obs = np.array([1.0, 1.0, 1.0])
    confidence = "HIGH"
    notes = []
    ci95_deg = math.degrees(1.96 * std_scalar)
    rms_deg = math.degrees(float(np.sqrt(np.mean(kept_errors**2))))
    if ci95_deg > 10.0 or rms_deg > 15.0:
        confidence = "LOW"
        notes.append("stored camera rotations are widely scattered")
    elif ci95_deg > 3.0 or rms_deg > 7.0:
        confidence = "MEDIUM"
        notes.append("stored camera rotations are noticeably scattered")
    dropped = len(offsets) - int(np.count_nonzero(keep))
    if dropped:
        notes.append(f"trimmed {dropped} orientation outliers")

    kept_runs = sorted(set(np.asarray(run_names, dtype=object)[keep].tolist()))
    return FitSummary(
        name="camera reported-pose orientation offset (diagnostic)",
        rotation=mean,
        euler_rad=euler,
        euler_std_rad=euler_std,
        rms_raw=rms_deg,
        median_raw=math.degrees(float(np.median(kept_errors))),
        n_samples=int(np.count_nonzero(keep)),
        n_runs=len(kept_runs),
        observability=obs,
        condition=1.0,
        confidence=confidence,
        notes=notes,
    )


def fmt_vec(values: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in values) + "]"


def fmt_angle_ci_deg(values_deg: np.ndarray, ci95_deg: np.ndarray, precision: int = 3) -> str:
    labels = ("phi", "theta", "psi")
    parts = []
    for label, value, ci in zip(labels, values_deg, ci95_deg):
        parts.append(f"{label}={value:.{precision}f} +/- {ci:.{precision}f}")
    return ", ".join(parts)


def print_summary(summary: FitSummary | None, raw_unit: str) -> None:
    if summary is None:
        print("  Not enough valid data to estimate this offset.")
        return

    euler_deg = np.rad2deg(summary.euler_rad)
    std_deg = np.rad2deg(summary.euler_std_rad)
    ci95_deg = 1.96 * std_deg

    print(f"  {summary.name}")
    print(f"    confidence : {summary.confidence}")
    print(f"    angles deg : {fmt_angle_ci_deg(euler_deg, ci95_deg)}")
    print(
        f"    residual   : RMS {summary.rms_raw:.5f} {raw_unit}, "
        f"median {summary.median_raw:.5f} {raw_unit}"
    )
    print(
        f"    data       : {summary.n_samples} samples across {summary.n_runs} runs, "
        f"observability ratios {fmt_vec(summary.observability, 4)}, "
        f"condition {summary.condition:.1f}"
    )
    for note in summary.notes:
        print(f"    note       : {note}")


def print_dataset_summary(runs: list[RunData]) -> None:
    print("Dataset summary")
    for marker_type in ("ChArUco", "ArUco"):
        selected = [run for run in runs if run.marker_type == marker_type]
        if not selected:
            continue
        n_ref = sum(len(run.reference) for run in selected)
        n_imu = sum(len(run.imu) for run in selected)
        n_cam = sum(len(run.camera) for run in selected)
        print(
            f"  {marker_type}: {len(selected)} runs, "
            f"{n_ref} reference samples, {n_imu} IMU samples, {n_cam} camera frames"
        )


def main() -> int:
    args = parse_args()
    runs = parse_streamlined_data(args.data)
    time_corrections = load_time_corrections(args.time_corrections)
    apply_time_corrections = not args.no_time_corrections

    print(f"Loaded {len(runs)} runs from {args.data}")
    if apply_time_corrections:
        print(f"Loaded {len(time_corrections)} camera time corrections from {args.time_corrections}")
    else:
        print("Camera time corrections disabled")
    print_dataset_summary(runs)
    print()

    for marker_type in args.types:
        print("=" * 72)
        print(f"{marker_type} fixed rotation offsets")

        imu_summary, dynamic_mag = fit_imu_offset(runs, marker_type, args)
        print_summary(imu_summary, "m/s^2")
        if len(dynamic_mag):
            print(
                f"    IMU dynamic reference acceleration: median {np.median(dynamic_mag):.4f} m/s^2, "
                f"95th percentile {np.percentile(dynamic_mag, 95):.4f} m/s^2"
            )
        print()

        pairs_by_run = build_camera_pairs(
            runs,
            marker_type,
            time_corrections,
            apply_time_corrections,
            args.max_ref_gap_ms,
        )
        usable_camera_pairs = sum(len(v) for v in pairs_by_run.values())
        print(f"  Camera/reference pairs: {usable_camera_pairs} across {len(pairs_by_run)} runs")

        camera_summary = fit_camera_position_offset(
            pairs_by_run,
            args.min_run_camera_samples,
            args.camera_sigma_m,
        )
        print_summary(camera_summary, "m")

        if not args.skip_camera_orientation:
            print()
            orientation_summary = fit_camera_orientation_diagnostic(pairs_by_run)
            print_summary(orientation_summary, "deg")
        print()

    print("Interpretation notes")
    print("  * IMU result maps IMU-local acceleration into the moving reference frame.")
    print("  * Camera position-frame result maps reference-frame motion deltas into")
    print("    camera-measured position deltas; it is the best match to the current")
    print("    MantaUKF camera coupling state.")
    print("  * Camera reported-pose orientation is diagnostic because stored tag")
    print("    rotations may contain solvePnP/planar ambiguity noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
