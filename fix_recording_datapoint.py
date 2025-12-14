#!/usr/bin/env python3
"""Utility for editing nested datapoints inside NDJSON recording files.

Example usage:
    See main() function at the bottom of this file.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, MutableSequence, Sequence, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "recordings"
PathLike = Union[str, Path]


class PathResolutionError(RuntimeError):
    """Raised when a provided path cannot be resolved."""


def _normalize_file_argument(file_arg: str) -> str:
    """Ensure the provided filename includes the .json suffix."""

    file_arg = file_arg.strip()
    if not file_arg:
        raise ValueError("File argument may not be empty")

    if file_arg.lower().endswith(".json"):
        return file_arg

    return f"{file_arg}.json"


def resolve_data_file(file_arg: str, data_dir: Path | None) -> Path:
    """Resolve the recordings data file path.

    The function accepts bare file names (with or without extension) and full
    paths. If the file is not found relative to the current working directory it
    is searched for inside ``data_dir`` (when provided) and finally inside the
    repository ``recordings`` folder.
    """

    normalized = Path(_normalize_file_argument(file_arg))

    candidates: List[Path] = []
    if normalized.is_absolute():
        candidates.append(normalized)
    else:
        candidates.append(Path.cwd() / normalized)
        if data_dir:
            candidates.append(Path(data_dir) / normalized)
        candidates.append(DEFAULT_DATA_DIR / normalized.name)
        candidates.append(DEFAULT_DATA_DIR / normalized)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not locate '{file_arg}'. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _parse_segment(segment: str) -> Union[str, int]:
    """Interpret a single path segment as either a dict key or list index."""

    segment = segment.strip()
    if not segment:
        raise ValueError("Path segments may not be empty")

    if segment.startswith("[") and segment.endswith("]"):
        segment = segment[1:-1]

    try:
        return int(segment)
    except ValueError:
        return segment


def _ensure_list_capacity(target: MutableSequence[Any], index: int) -> None:
    if index < 0:
        raise IndexError("Negative indexes are not supported for JSON arrays")

    while len(target) <= index:
        target.append(None)


def _traverse_to_parent(
    entry: Any,
    path_segments: Sequence[str],
    *,
    create_missing: bool,
) -> Tuple[Any, Union[str, int]]:
    """Return the immediate parent container and the final key/index."""

    if not path_segments:
        raise ValueError("Path must contain at least one segment")

    current = entry
    parsed_segments = [_parse_segment(seg) for seg in path_segments]

    for index, segment in enumerate(parsed_segments[:-1]):
        next_segment = parsed_segments[index + 1]

        if isinstance(segment, int):
            if not isinstance(current, list):
                raise TypeError(
                    f"Encountered list index '{segment}' but current object is not a list"
                )
            if segment < 0:
                raise IndexError("Negative indexes are not supported")

            if segment >= len(current):
                if not create_missing:
                    raise IndexError(
                        f"Index {segment} out of range while traversing path"
                    )
                _ensure_list_capacity(current, segment)
                # Choose sensible defaults for deeper levels
                placeholder = [] if isinstance(next_segment, int) else {}
                current[segment] = placeholder
            elif current[segment] is None and create_missing:
                current[segment] = [] if isinstance(next_segment, int) else {}

            current = current[segment]
        else:
            if not isinstance(current, dict):
                raise TypeError(
                    f"Encountered key '{segment}' but current object is not a dict"
                )
            if segment not in current:
                if not create_missing:
                    raise KeyError(f"Missing key '{segment}' while traversing path")
                current[segment] = [] if isinstance(next_segment, int) else {}

            current = current[segment]

    return current, parsed_segments[-1]


def update_nested_value(
    entry: Any,
    path_segments: Sequence[str],
    value_operand: Any,
    *,
    mode: str,
    create_missing: bool,
) -> Tuple[bool, Any, Any]:
    """Update a nested value and report whether it changed.

    Returns a tuple ``(changed, previous_value, new_value)``.
    """

    parent, final_segment = _traverse_to_parent(
        entry, path_segments, create_missing=create_missing
    )

    if isinstance(final_segment, int):
        if not isinstance(parent, list):
            raise TypeError(
                f"Encountered list index '{final_segment}' but parent is not a list"
            )
        if final_segment < 0:
            raise IndexError("Negative indexes are not supported")
        if final_segment >= len(parent):
            if not create_missing:
                raise IndexError(
                    f"Index {final_segment} out of range while setting nested value"
                )
            _ensure_list_capacity(parent, final_segment)

        previous = parent[final_segment]
        
        if mode == "addition":
            if isinstance(previous, (int, float)) and isinstance(value_operand, (int, float)):
                new_value = round(previous + value_operand, 16)
            else:
                new_value = previous + value_operand
        else:
            new_value = value_operand

        parent[final_segment] = new_value
    else:
        if not isinstance(parent, dict):
            raise TypeError(
                f"Encountered key '{final_segment}' but parent is not a dict"
            )
        previous = parent.get(final_segment)
        if previous is None and not create_missing and final_segment not in parent:
            raise KeyError(
                f"Missing key '{final_segment}' while setting nested value (use --create-missing)."
            )
        
        if mode == "addition":
            if isinstance(previous, (int, float)) and isinstance(value_operand, (int, float)):
                new_value = round(previous + value_operand, 16)
            else:
                new_value = previous + value_operand
        else:
            new_value = value_operand

        parent[final_segment] = new_value

    return previous != new_value, previous, new_value


def _apply_edit(
    *,
    file_path: Path,
    target_path: Sequence[str],
    new_value: Any,
    mode: str,
    backup: bool,
    mpu_unit: int | None,
    create_missing: bool,
    strict: bool,
    dry_run: bool,
    output_path: Path | None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Edit the specified datapoint across all matching entries."""

    total_rows = 0
    matching_rows = 0
    updated_rows = 0
    skipped_missing = 0
    output_lines: list[str] = []

    try:
        with file_path.open("r", encoding=encoding) as src:
            for line_number, line in enumerate(src, 1):
                stripped = line.strip()
                if not stripped:
                    output_lines.append(line)
                    continue

                total_rows += 1
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Line {line_number} in '{file_path}' is not valid JSON: {exc}"
                    ) from exc

                if mpu_unit is not None and entry.get("mpu_unit") != mpu_unit:
                    output_lines.append(line)
                    continue

                matching_rows += 1

                try:
                    changed, previous, _ = update_nested_value(
                        entry,
                        target_path,
                        new_value,
                        mode=mode,
                        create_missing=create_missing,
                    )
                except (KeyError, TypeError, IndexError) as exc:
                    if strict:
                        raise
                    skipped_missing += 1
                    output_lines.append(line)
                    continue

                if changed:
                    updated_rows += 1
                    serialized = json.dumps(entry, ensure_ascii=False)
                    output_lines.append(serialized + ("\n" if not serialized.endswith("\n") else ""))
                else:
                    output_lines.append(line)
    except FileNotFoundError as exc:
        raise exc

    destination = output_path or file_path
    summary = {
        "total_rows": total_rows,
        "matching_rows": matching_rows,
        "updated_rows": updated_rows,
        "skipped_missing": skipped_missing,
        "output_path": destination,
        "dry_run": dry_run,
    }

    if dry_run:
        return summary

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination == file_path and backup:
        backup_name = file_path.with_suffix(
            file_path.suffix + f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copyfile(file_path, backup_name)
        summary["backup"] = backup_name

    with tempfile.NamedTemporaryFile("w", encoding=encoding, delete=False, dir=str(destination.parent)) as tmp:
        tmp_path = Path(tmp.name)
        tmp.writelines(output_lines)

    shutil.move(tmp_path, destination)
    return summary


def edit_recording(
    backup_mode: str,
    mode: str,
    file: str,
    *path: str | int,
    value: Any,
    mpu_unit: int | None = None,
    create_missing: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    output: Path | None = None,
    encoding: str = "utf-8",
) -> None:
    """
    User-facing wrapper to edit a recording.
    
    Args:
        backup_mode: 'backup' or 'no-backup'
        mode: 'overwrite' or 'addition'
        file: Filename or path
        *path: Path segments to the datapoint (e.g. "pressure", "depth_offset0")
        value: The value to set or add
        mpu_unit: Optional MPU unit filter
        create_missing: Create missing keys/indices
        strict: Raise error on missing keys
        dry_run: Don't write changes
        output: Output path
        encoding: File encoding
    """
    try:
        file_path = resolve_data_file(file, None)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    # Convert path segments to strings as expected by internal logic
    target_path = [str(p) for p in path]

    try:
        summary = _apply_edit(
            file_path=file_path,
            target_path=target_path,
            new_value=value,
            mode=mode,
            backup=backup_mode == "backup",
            mpu_unit=mpu_unit,
            create_missing=create_missing,
            strict=strict,
            dry_run=dry_run,
            output_path=output,
            encoding=encoding,
        )
    except Exception as exc:
        print(f"Error processing {file}: {exc}")
        return

    destination = summary["output_path"]
    dry_run_prefix = "[DRY-RUN] " if summary["dry_run"] else ""
    print(
        f"{dry_run_prefix}Processed {summary['total_rows']} rows, "
        f"matched {summary['matching_rows']}, updated {summary['updated_rows']} "
        f"(skipped {summary['skipped_missing']} missing). Output: {destination}"
    )
    if summary.get("backup"):
        print(f"Backup written to: {summary['backup']}")


def main() -> int:
    
    #edit_recording("no-backup", "addition", "ArUco Quad 2m run1", "pressure", "depth_offset0", mpu_unit=3, value=0.023)

    edit_recording("no-backup", "addition", f"ArUco Single 2-4.5m", "pressure", "depth_offset0", mpu_unit=3, value=-0.236)
    edit_recording("no-backup", "addition", f"ArUco Single 7-4.5m", "pressure", "depth_offset1", mpu_unit=3, value=-0.239)
    edit_recording("no-backup", "addition", f"ChArUco Single 4.5-2m", "pressure", "depth_offset0", mpu_unit=3, value=-0.300)
    edit_recording("no-backup", "addition", f"ChArUco Single 4.5-7m", "pressure", "depth_offset1", mpu_unit=3, value=-0.313)
    
    return 0


if __name__ == "__main__":
    main()
