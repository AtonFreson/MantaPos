#!/usr/bin/env python3
"""Utility for editing nested datapoints inside NDJSON recording files.

Example usage:
    python edit_recording_datapoint.py "ArUco Quad 2m run1" \
        pressure depth_offset0 --mpu-unit 3 --value -0.02
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, MutableSequence, Sequence, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "recordings"
PathLike = Union[str, Path]

# Set this to a tuple/list of CLI arguments (e.g. ("ArUco Quad 2m run1", "pressure", ...))
# when you want to run the tool without typing them on the command line. Leave it as
# ``None`` to keep using real CLI inputs.
PRESET_ARGUMENTS: Sequence[str] | None = ("ArUco Quad 2m run1", "pressure", "depth_offset0", "--mpu-unit", "3", "--value", "-0.023")


class PathResolutionError(RuntimeError):
    """Raised when a provided path cannot be resolved."""


def _coerce_new_value(raw: str, *, force_string: bool) -> Any:
    """Turn the CLI string into a Python value.

    If ``force_string`` is True the string is returned unchanged. Otherwise the
    function attempts ``ast.literal_eval`` to interpret numbers/containers and
    falls back to the raw string.
    """

    if force_string:
        return raw

    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


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


def set_nested_value(
    entry: Any,
    path_segments: Sequence[str],
    new_value: Any,
    *,
    create_missing: bool,
) -> Tuple[bool, Any, Any]:
    """Set a nested value and report whether it changed.

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
        parent[final_segment] = new_value

    return previous != new_value, previous, new_value


def edit_recording(
    *,
    file_path: Path,
    target_path: Sequence[str],
    new_value: Any,
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
                    changed, previous, _ = set_nested_value(
                        entry,
                        target_path,
                        new_value,
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

    if destination == file_path:
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Edit a nested datapoint for every matching entry in a recordings NDJSON file. "
            "Provide the path to the datapoint as a sequence of keys and/or indexes."
        )
    )

    parser.add_argument(
        "file",
        help="Recording file to edit (with or without .json extension).",
    )
    parser.add_argument(
        "path",
        nargs="+",
        help=(
            "Nested keys or list indexes identifying the target datapoint. "
            "Use numbers or [index] for arrays, e.g. imu acceleration x -> imu acceleration x."
        ),
    )
    parser.add_argument(
        "--value",
        required=True,
        help="New value to assign. Parsed with ast.literal_eval unless --raw-string is used.",
    )
    parser.add_argument(
        "--mpu-unit",
        type=int,
        help="Only edit entries matching this mpu_unit. Omit to edit all entries.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the default recordings directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the edited content to a new file instead of modifying in place.",
    )
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create intermediate objects/indexes when the path does not exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if the target path is missing or incompatible (default skips those rows).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the edit and print a summary without writing any files.",
    )
    parser.add_argument(
        "--raw-string",
        action="store_true",
        help="Store the new value as a literal string instead of parsing it.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding to use (default: utf-8).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors (suppress the summary line).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()

    effective_argv = argv
    if effective_argv is None and PRESET_ARGUMENTS is not None:
        effective_argv = PRESET_ARGUMENTS

    args = parser.parse_args(effective_argv)

    try:
        target_value = _coerce_new_value(args.value, force_string=args.raw_string)
    except ValueError as exc:  # pragma: no cover - defensive
        parser.error(str(exc))

    try:
        file_path = resolve_data_file(str(args.file), args.data_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    try:
        summary = edit_recording(
            file_path=file_path,
            target_path=args.path,
            new_value=target_value,
            mpu_unit=args.mpu_unit,
            create_missing=args.create_missing,
            strict=args.strict,
            dry_run=args.dry_run,
            output_path=args.output,
            encoding=args.encoding,
        )
    except Exception as exc:  # pragma: no cover - CLI surface area
        parser.error(str(exc))

    if not args.quiet:
        destination = summary["output_path"]
        dry_run_prefix = "[DRY-RUN] " if summary["dry_run"] else ""
        print(
            f"{dry_run_prefix}Processed {summary['total_rows']} rows, "
            f"matched {summary['matching_rows']}, updated {summary['updated_rows']} "
            f"(skipped {summary['skipped_missing']} missing). Output: {destination}"
        )
        if summary.get("backup"):
            print(f"Backup written to: {summary['backup']}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
