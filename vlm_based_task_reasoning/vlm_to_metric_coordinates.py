# -*- coding: utf-8 -*-
"""
VLM-to-Metric Coordinate Converter
Python 3.10

Purpose
-------
Convert polygon coordinates produced by the VLM task-reasoning program
(normalized image coordinates in [0, 999]) into the metric XYZ coordinates
required by the hierarchical UGV path planner.

Input
-----
1. One UAV orthographic / bird's-eye image.
2. The VLM result CSV. Expected core columns:
       image_filename
       zone_type
       zone_index
       normalized_coordinates
   Optional columns such as target_description and action_sequence are preserved.
3. XYZ coordinates of the four IMAGE CORNERS:
       top_left, top_right, bottom_right, bottom_left
4. Optional start/goal points in normalized image coordinates [0, 999].

Output
------
A planner-ready CSV with:
       type, number, X/E, Y/N, Z/U
plus preserved optional task/action columns when available.

Coordinate mapping
------------------
For a normalized point (x_norm, y_norm):
    u = x_norm / 999
    v = y_norm / 999

The metric XYZ point is obtained by bilinear interpolation of the four
corner coordinates:

    P(u,v) =
        (1-u)(1-v) * P_TL
      + u(1-v)     * P_TR
      + uv         * P_BR
      + (1-u)v     * P_BL

This is appropriate for orthographic / georeferenced bird's-eye imagery
covering approximately planar ground. The interpolated Z is a smooth surface
defined by the four corner elevations.

IMPORTANT
---------
The four supplied coordinates must correspond to the image's actual corner
pixels in this exact order:
    top-left, top-right, bottom-right, bottom-left.

For a strongly perspective image or terrain with large relief, four corners
alone are not sufficient for high-accuracy 3-D georeferencing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


NORMALIZED_MAX = 999.0
ZONE_TYPES = ("target_zones", "obstacle_zones", "risk_zones")
BASE_OUTPUT_COLUMNS = ["type", "number", "X/E", "Y/N", "Z/U"]
PRESERVED_OPTIONAL_COLUMNS = [
    "target_description",
    "action_sequence",
]

XYZ = Tuple[float, float, float]
XYN = Tuple[float, float]


@dataclass(frozen=True)
class CornerCoordinates:
    top_left: XYZ
    top_right: XYZ
    bottom_right: XYZ
    bottom_left: XYZ

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        def item(point: XYZ) -> Dict[str, float]:
            return {"X": point[0], "Y": point[1], "Z": point[2]}

        return {
            "top_left": item(self.top_left),
            "top_right": item(self.top_right),
            "bottom_right": item(self.bottom_right),
            "bottom_left": item(self.bottom_left),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VLM normalized polygon coordinates [0,999] to metric "
            "XYZ coordinates for the hierarchical UGV path planner."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="UAV image corresponding to the VLM result.",
    )
    parser.add_argument(
        "--vlm-csv",
        type=Path,
        default=None,
        help="CSV produced by the VLM task-reasoning program.",
    )
    parser.add_argument(
        "--corners",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing top_left, top_right, "
            "bottom_right, bottom_left XYZ coordinates. "
            "If omitted, coordinates are requested interactively."
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help='Optional normalized start point as "x,y", each in [0,999].',
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help='Optional normalized goal point as "x,y", each in [0,999].',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output planner-ready CSV path.",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="",
        help='Optional CRS label stored in metadata, e.g. "EPSG:32650".',
    )
    return parser.parse_args()


def ask_open_file(title: str, filetypes: Sequence[Tuple[str, str]]) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return Path(selected) if selected else None
    except Exception as exc:
        print(f"Unable to open file-selection dialog: {exc}")
        return None


def ask_save_file(title: str, default_name: str) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.asksaveasfilename(
            title=title,
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(selected) if selected else None
    except Exception as exc:
        print(f"Unable to open save-file dialog: {exc}")
        return None


def resolve_image(path: Optional[Path]) -> Path:
    if path is None:
        path = ask_open_file(
            "Select the UAV image",
            [
                ("Image files", "*.png *.jpg *.jpeg *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
    if path is None:
        raise RuntimeError("No image was selected.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def resolve_vlm_csv(path: Optional[Path]) -> Path:
    if path is None:
        path = ask_open_file(
            "Select the VLM result CSV",
            [("CSV files", "*.csv"), ("All files", "*.*")],
        )
    if path is None:
        raise RuntimeError("No VLM result CSV was selected.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VLM CSV not found: {path}")
    return path


def resolve_output(path: Optional[Path], image_path: Path) -> Path:
    if path is None:
        default_name = f"{image_path.stem}_planner_annotations.csv"
        path = ask_save_file(
            "Save planner-ready coordinate CSV",
            default_name,
        )
        if path is None:
            path = image_path.parent / default_name

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_xyz_text(text: str, label: str) -> XYZ:
    cleaned = text.strip().replace("，", ",")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"{label}: enter exactly three values as X,Y,Z."
        )
    try:
        point = tuple(float(value) for value in parts)
    except ValueError as exc:
        raise ValueError(f"{label}: X, Y and Z must be numeric.") from exc
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label}: all coordinates must be finite.")
    return point  # type: ignore[return-value]


def prompt_xyz(label: str) -> XYZ:
    while True:
        raw = input(f"{label} X,Y,Z: ").strip()
        try:
            return parse_xyz_text(raw, label)
        except ValueError as exc:
            print(f"Invalid input: {exc}")


def read_corner_json(path: Path) -> Tuple[CornerCoordinates, str]:
    data = json.loads(path.read_text(encoding="utf-8"))

    def point(name: str) -> XYZ:
        if name not in data:
            raise ValueError(f"Corner JSON is missing '{name}'.")
        value = data[name]

        if isinstance(value, dict):
            keys = {str(key).lower(): key for key in value}
            try:
                xyz = (
                    float(value[keys["x"]]),
                    float(value[keys["y"]]),
                    float(value[keys["z"]]),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Corner '{name}' must contain X, Y and Z."
                ) from exc
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            xyz = tuple(float(item) for item in value)
        else:
            raise ValueError(
                f"Corner '{name}' must be an XYZ object or a 3-value list."
            )

        if not all(math.isfinite(v) for v in xyz):
            raise ValueError(f"Corner '{name}' contains non-finite values.")
        return xyz  # type: ignore[return-value]

    corners = CornerCoordinates(
        top_left=point("top_left"),
        top_right=point("top_right"),
        bottom_right=point("bottom_right"),
        bottom_left=point("bottom_left"),
    )
    crs = str(data.get("coordinate_reference_system", data.get("crs", ""))).strip()
    return corners, crs


def get_corners(
    corners_path: Optional[Path],
    crs_from_args: str,
) -> Tuple[CornerCoordinates, str]:
    if corners_path is not None:
        corners_path = corners_path.expanduser().resolve()
        corners, crs_from_file = read_corner_json(corners_path)
        return corners, crs_from_args.strip() or crs_from_file

    print()
    print("=" * 78)
    print("Enter the actual XYZ coordinates of the FOUR IMAGE CORNERS.")
    print("Order is important:")
    print("  1. top-left")
    print("  2. top-right")
    print("  3. bottom-right")
    print("  4. bottom-left")
    print("Example input format: 234380.123, 2502115.456, 1.025")
    print("=" * 78)

    corners = CornerCoordinates(
        top_left=prompt_xyz("Top-left"),
        top_right=prompt_xyz("Top-right"),
        bottom_right=prompt_xyz("Bottom-right"),
        bottom_left=prompt_xyz("Bottom-left"),
    )
    return corners, crs_from_args.strip()


def validate_corner_geometry(corners: CornerCoordinates) -> None:
    xy = [
        corners.top_left[:2],
        corners.top_right[:2],
        corners.bottom_right[:2],
        corners.bottom_left[:2],
    ]

    # Shoelace area. A near-zero value indicates degenerate or badly ordered corners.
    twice_area = 0.0
    for first, second in zip(xy, xy[1:] + xy[:1]):
        twice_area += first[0] * second[1] - second[0] * first[1]

    if abs(twice_area) < 1e-10:
        raise ValueError(
            "The four corner XY coordinates form a degenerate quadrilateral. "
            "Check corner order and coordinate values."
        )


def bilinear_xyz(
    x_norm: float,
    y_norm: float,
    corners: CornerCoordinates,
) -> XYZ:
    if not (0.0 <= x_norm <= NORMALIZED_MAX):
        raise ValueError(f"x_norm={x_norm} is outside [0, 999].")
    if not (0.0 <= y_norm <= NORMALIZED_MAX):
        raise ValueError(f"y_norm={y_norm} is outside [0, 999].")

    u = x_norm / NORMALIZED_MAX
    v = y_norm / NORMALIZED_MAX

    tl = corners.top_left
    tr = corners.top_right
    br = corners.bottom_right
    bl = corners.bottom_left

    weights = (
        (1.0 - u) * (1.0 - v),  # TL
        u * (1.0 - v),          # TR
        u * v,                  # BR
        (1.0 - u) * v,          # BL
    )
    points = (tl, tr, br, bl)

    x = sum(weight * point[0] for weight, point in zip(weights, points))
    y = sum(weight * point[1] for weight, point in zip(weights, points))
    z = sum(weight * point[2] for weight, point in zip(weights, points))
    return float(x), float(y), float(z)


def parse_normalized_point(text: Optional[str], label: str) -> Optional[XYN]:
    if text is None:
        return None

    cleaned = text.strip().replace("，", ",")
    if not cleaned:
        return None

    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{label} must be entered as x,y.")

    x = float(parts[0])
    y = float(parts[1])
    if not (0.0 <= x <= NORMALIZED_MAX and 0.0 <= y <= NORMALIZED_MAX):
        raise ValueError(f"{label} must lie within [0,999] x [0,999].")
    return x, y


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("The VLM CSV has no header.")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def choose_rows_for_image(
    rows: Sequence[Dict[str, str]],
    image_path: Path,
) -> Tuple[List[Dict[str, str]], str]:
    if not rows:
        raise ValueError("The VLM CSV contains no data rows.")

    if "image_filename" not in rows[0]:
        raise ValueError("The VLM CSV is missing 'image_filename'.")

    image_name = image_path.name
    image_names = [str(row.get("image_filename", "")).strip() for row in rows]

    # 1) Prefer exact basename match.
    selected = [
        row
        for row in rows
        if Path(str(row.get("image_filename", "")).replace("\\", "/")).name
        == image_name
    ]
    selected_names = sorted(
        {
            str(row.get("image_filename", "")).strip()
            for row in selected
        }
    )

    if selected:
        if len(selected_names) > 1:
            raise ValueError(
                f"Multiple CSV image paths have the basename '{image_name}': "
                + ", ".join(selected_names)
                + ". Use unique image filenames."
            )
        return selected, selected_names[0]

    # 2) If the CSV contains results for exactly one image, allow it.
    unique_names = sorted({name for name in image_names if name})
    if len(unique_names) == 1:
        print(
            f"Warning: image filename '{image_name}' does not exactly match "
            f"CSV image_filename '{unique_names[0]}'. The CSV contains only "
            "one image, so those rows will be used."
        )
        return list(rows), unique_names[0]

    raise ValueError(
        f"No VLM rows match image '{image_name}'. "
        f"CSV contains {len(unique_names)} distinct image_filename values."
    )


def parse_polygon(raw_value: str, row_number: int) -> List[XYN]:
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"CSV row {row_number}: invalid normalized_coordinates JSON."
        ) from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"CSV row {row_number}: normalized_coordinates must be a list."
        )

    points: List[XYN] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                raise ValueError(
                    f"CSV row {row_number}, point {index}: dict must contain x and y."
                )
            x = float(item["x"])
            y = float(item["y"])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = float(item[0])
            y = float(item[1])
        else:
            raise ValueError(
                f"CSV row {row_number}, point {index}: unsupported point format."
            )

        if not (0.0 <= x <= NORMALIZED_MAX and 0.0 <= y <= NORMALIZED_MAX):
            raise ValueError(
                f"CSV row {row_number}, point {index}: "
                f"({x}, {y}) is outside [0,999]."
            )
        points.append((x, y))

    if len(points) < 3:
        raise ValueError(
            f"CSV row {row_number}: polygon has fewer than three points."
        )
    return points


def convert_zone_rows(
    vlm_rows: Sequence[Dict[str, str]],
    corners: CornerCoordinates,
    optional_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []

    for row_number, row in enumerate(vlm_rows, start=2):
        zone_type = str(row.get("zone_type", "")).strip()
        if zone_type not in ZONE_TYPES:
            raise ValueError(
                f"CSV row {row_number}: unsupported zone_type '{zone_type}'."
            )

        try:
            zone_index = int(str(row.get("zone_index", "")).strip())
        except ValueError as exc:
            raise ValueError(
                f"CSV row {row_number}: invalid zone_index."
            ) from exc

        polygon = parse_polygon(
            str(row.get("normalized_coordinates", "")),
            row_number,
        )

        preserved = {
            name: row.get(name, "")
            for name in optional_columns
        }

        for x_norm, y_norm in polygon:
            x, y, z = bilinear_xyz(x_norm, y_norm, corners)
            output_rows.append(
                {
                    "type": zone_type,
                    "number": zone_index,
                    "X/E": x,
                    "Y/N": y,
                    "Z/U": z,
                    **preserved,
                }
            )

    return output_rows


def make_point_row(
    point_type: str,
    normalized_point: XYN,
    corners: CornerCoordinates,
) -> Dict[str, Any]:
    x, y, z = bilinear_xyz(
        normalized_point[0],
        normalized_point[1],
        corners,
    )
    return {
        "type": point_type,
        "number": 1,
        "X/E": x,
        "Y/N": y,
        "Z/U": z,
    }


def write_planner_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    optional_columns: Sequence[str],
) -> None:
    fieldnames = BASE_OUTPUT_COLUMNS + list(optional_columns)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["X/E"] = f"{float(row['X/E']):.9f}"
            formatted["Y/N"] = f"{float(row['Y/N']):.9f}"
            formatted["Z/U"] = f"{float(row['Z/U']):.9f}"
            writer.writerow(formatted)


def write_metadata(
    output_csv: Path,
    image_path: Path,
    image_width: int,
    image_height: int,
    vlm_csv: Path,
    matched_image_filename: str,
    corners: CornerCoordinates,
    crs: str,
    start: Optional[XYN],
    goal: Optional[XYN],
    zone_count: int,
    vertex_count: int,
) -> Path:
    metadata_path = output_csv.with_name(
        output_csv.stem + "_georeferencing.json"
    )

    payload = {
        "format_version": "V1",
        "mapping_method": "bilinear_3d_from_four_image_corners",
        "normalization": {
            "range": [0, 999],
            "origin": "top-left",
            "x_direction": "right",
            "y_direction": "down",
            "x_pixel_formula": (
                "x_pixel = x_norm / 999 * (image_width_px - 1)"
            ),
            "y_pixel_formula": (
                "y_pixel = y_norm / 999 * (image_height_px - 1)"
            ),
        },
        "image": {
            "path": str(image_path),
            "filename": image_path.name,
            "width_px": image_width,
            "height_px": image_height,
        },
        "source_vlm_csv": str(vlm_csv),
        "matched_vlm_image_filename": matched_image_filename,
        "coordinate_reference_system": crs,
        "corner_coordinates": corners.to_dict(),
        "start_normalized": (
            {"x": start[0], "y": start[1]} if start is not None else None
        ),
        "goal_normalized": (
            {"x": goal[0], "y": goal[1]} if goal is not None else None
        ),
        "output": {
            "planner_csv": str(output_csv),
            "zone_count": zone_count,
            "zone_vertex_count": vertex_count,
            "contains_start": start is not None,
            "contains_goal": goal is not None,
        },
        "assumptions": [
            "The input image is an orthographic/georeferenced bird's-eye image or is sufficiently close to one.",
            "The four XYZ values correspond exactly to the top-left, top-right, bottom-right and bottom-left image corners.",
            "Ground elevation inside the image is approximated by bilinear interpolation of the four corner elevations.",
        ],
    }

    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def count_unique_zones(rows: Sequence[Dict[str, str]]) -> int:
    return len(
        {
            (
                str(row.get("zone_type", "")).strip(),
                str(row.get("zone_index", "")).strip(),
            )
            for row in rows
        }
    )


def main() -> int:
    args = parse_args()

    try:
        image_path = resolve_image(args.image)
        vlm_csv = resolve_vlm_csv(args.vlm_csv)
        output_csv = resolve_output(args.output, image_path)

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        if image_width <= 1 or image_height <= 1:
            raise ValueError(
                f"Invalid image size: {image_width} x {image_height}"
            )

        corners, crs = get_corners(args.corners, args.crs)
        validate_corner_geometry(corners)

        start = parse_normalized_point(args.start, "start")
        goal = parse_normalized_point(args.goal, "goal")

        # Friendly interactive mode for users who run the script directly.
        # If --start/--goal are not supplied, the user may optionally enter
        # normalized start/goal points so the output can be consumed directly
        # by the current hierarchical path planner.
        if args.start is None and args.goal is None and sys.stdin.isatty():
            answer = input(
                "\nAdd start and goal points to the planner CSV? "
                "[y/N]: "
            ).strip().lower()
            if answer in {"y", "yes"}:
                while start is None:
                    try:
                        start = parse_normalized_point(
                            input("Start x,y in [0,999]: ").strip(),
                            "start",
                        )
                    except ValueError as exc:
                        print(f"Invalid input: {exc}")
                while goal is None:
                    try:
                        goal = parse_normalized_point(
                            input("Goal x,y in [0,999]: ").strip(),
                            "goal",
                        )
                    except ValueError as exc:
                        print(f"Invalid input: {exc}")

        rows, fieldnames = read_csv_rows(vlm_csv)
        required = {
            "image_filename",
            "zone_type",
            "zone_index",
            "normalized_coordinates",
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                "VLM CSV is missing required column(s): "
                + ", ".join(missing)
            )

        selected_rows, matched_image_filename = choose_rows_for_image(
            rows,
            image_path,
        )

        optional_columns = [
            name
            for name in PRESERVED_OPTIONAL_COLUMNS
            if name in fieldnames
        ]

        zone_rows = convert_zone_rows(
            selected_rows,
            corners,
            optional_columns,
        )

        output_rows: List[Dict[str, Any]] = []

        # The path planner expects start and goal as single rows.
        if start is not None:
            output_rows.append(
                make_point_row("start", start, corners)
            )
        if goal is not None:
            output_rows.append(
                make_point_row("goal", goal, corners)
            )

        output_rows.extend(zone_rows)

        write_planner_csv(
            output_csv,
            output_rows,
            optional_columns,
        )

        metadata_path = write_metadata(
            output_csv=output_csv,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            vlm_csv=vlm_csv,
            matched_image_filename=matched_image_filename,
            corners=corners,
            crs=crs,
            start=start,
            goal=goal,
            zone_count=count_unique_zones(selected_rows),
            vertex_count=len(zone_rows),
        )

        print()
        print("=" * 78)
        print("VLM -> METRIC COORDINATE CONVERSION COMPLETED")
        print("=" * 78)
        print(f"Image                  : {image_path}")
        print(f"Image size             : {image_width} x {image_height} px")
        print(f"Matched VLM image      : {matched_image_filename}")
        print(f"Coordinate system      : {crs or '(not specified)'}")
        print(f"Zone count             : {count_unique_zones(selected_rows)}")
        print(f"Zone vertex count      : {len(zone_rows)}")
        print(f"Start included         : {start is not None}")
        print(f"Goal included          : {goal is not None}")
        print(f"Planner-ready CSV      : {output_csv}")
        print(f"Georeferencing metadata: {metadata_path}")
        print("=" * 78)

        if start is None or goal is None:
            print()
            print(
                "NOTE: The hierarchical path planner requires exactly one "
                "start row and one goal row. Re-run with --start and --goal, "
                "or add start/goal coordinates before using this CSV directly "
                "with the planner."
            )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
