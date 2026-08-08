from __future__ import annotations

import argparse
import heapq
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep
from pyproj import Transformer



# =============================================================================
# V4 — USER INPUT / OUTPUT PATHS
# 通常只需要修改下面四行，然后直接运行本程序。
#
# ACTION_CSV_PATH 可以为 None；此时程序从地理标注 CSV 的可选列
# target_description 和 action_sequence 读取目标动作。
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV_PATH = SCRIPT_DIR / "example_annotations.csv"
ACTION_CSV_PATH: Optional[Path] = None
CONFIG_JSON_PATH = SCRIPT_DIR / "planner_config.json"
OUTPUT_ROOT_DIR = SCRIPT_DIR / "planner_outputs"

XY = Tuple[float, float]

# MAVLink command and coordinate-frame identifiers used by ArduPilot missions.
MAV_FRAME_GLOBAL = 0
MAV_FRAME_MISSION = 2
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_DELAY = 93
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_DO_SET_RELAY = 181
MAV_CMD_DO_SET_SERVO = 183

SUPPORTED_ACTION_IDS = {
    "lower_collection_scoop",
    "raise_collection_scoop",
    "dump_collected_waste",
}
SUPPORTED_ACTION_TRIGGERS = {
    "on_zone_entry",
    "within_zone",
    "on_zone_exit",
}


def default_action_bindings() -> Dict[str, Dict[str, Any]]:
    """
    Hardware bindings are examples only. The program refuses to emit a mission
    containing an action until hardware_verified is set to true for that action.
    Calibrate channels and PWM values on the actual Pixhawk/ArduRover vehicle.
    """
    return {
        "lower_collection_scoop": {
            "command": "MAV_CMD_DO_SET_SERVO",
            "servo_channel": 9,
            "pwm": 1100,
            "duration_s": 2.0,
            "restore_pwm": None,
            "restore_duration_s": 0.0,
            "hardware_verified": False,
        },
        "raise_collection_scoop": {
            "command": "MAV_CMD_DO_SET_SERVO",
            "servo_channel": 9,
            "pwm": 1900,
            "duration_s": 2.0,
            "restore_pwm": None,
            "restore_duration_s": 0.0,
            "hardware_verified": False,
        },
        "dump_collected_waste": {
            "command": "MAV_CMD_DO_SET_SERVO",
            "servo_channel": 10,
            "pwm": 1900,
            "duration_s": 4.0,
            "restore_pwm": 1100,
            "restore_duration_s": 2.0,
            "hardware_verified": False,
        },
    }


@dataclass
class PlannerConfig:
    # Vehicle and operating parameters
    vehicle_length_m: float = 1.20
    vehicle_width_m: float = 0.80
    safety_margin_m: float = 0.15
    minimum_turning_radius_m: float = 1.00
    normal_speed_mps: float = 0.50
    risk_speed_ratio: float = 0.50

    # Target coverage model. null in JSON means vehicle_width_m / 2.
    target_coverage_half_width_m: Optional[float] = None

    # Coarse 2-D grid used to optimize the target visiting order
    order_grid_resolution_m: float = 0.10
    exact_order_max_targets: int = 14

    # Forward-only Hybrid A* parameters
    state_xy_resolution_m: float = 0.20
    heading_bins: int = 36
    motion_step_m: float = 0.40
    collision_check_step_m: float = 0.08
    curvature_ratios: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    goal_tolerance_m: float = 0.25
    max_expansions_per_leg: int = 300000

    # ArduRover mission export
    input_crs: str = "EPSG:32650"
    output_crs: str = "EPSG:4326"
    mission_frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT
    mission_altitude_m: float = 0.0
    mission_waypoint_spacing_m: float = 0.75
    mission_acceptance_radius_m: float = 0.25
    mission_command_anchor_delay_s: int = 1
    action_exit_clearance_m: float = 0.25
    max_mission_items: int = 700
    require_target_actions: bool = True
    emit_ardurover_mission: bool = True
    action_bindings: Dict[str, Dict[str, Any]] = field(
        default_factory=default_action_bindings
    )

    # Map and visualization
    map_margin_m: float = 3.0
    footprint_count: int = 12
    figure_dpi: int = 180
    show_plot: bool = True

    def validate(self) -> None:
        positive = {
            "vehicle_length_m": self.vehicle_length_m,
            "vehicle_width_m": self.vehicle_width_m,
            "minimum_turning_radius_m": self.minimum_turning_radius_m,
            "normal_speed_mps": self.normal_speed_mps,
            "order_grid_resolution_m": self.order_grid_resolution_m,
            "state_xy_resolution_m": self.state_xy_resolution_m,
            "motion_step_m": self.motion_step_m,
            "collision_check_step_m": self.collision_check_step_m,
            "goal_tolerance_m": self.goal_tolerance_m,
            "mission_waypoint_spacing_m": self.mission_waypoint_spacing_m,
            "mission_acceptance_radius_m": self.mission_acceptance_radius_m,
            "map_margin_m": self.map_margin_m,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if self.safety_margin_m < 0:
            raise ValueError("safety_margin_m cannot be negative.")
        if not 0 < self.risk_speed_ratio <= 1:
            raise ValueError("risk_speed_ratio must be in the interval (0, 1].")
        if self.heading_bins < 8:
            raise ValueError("heading_bins must be at least 8.")
        if self.max_expansions_per_leg <= 0:
            raise ValueError("max_expansions_per_leg must be positive.")
        if not self.curvature_ratios:
            raise ValueError("curvature_ratios cannot be empty.")
        if any(abs(value) > 1.0 + 1e-12 for value in self.curvature_ratios):
            raise ValueError(
                "Every curvature ratio must be between -1 and 1. "
                "This guarantees R >= minimum_turning_radius_m."
            )
        if 0.0 not in self.curvature_ratios:
            raise ValueError("curvature_ratios must contain 0.0 for straight motion.")
        if self.mission_frame not in {MAV_FRAME_GLOBAL, MAV_FRAME_GLOBAL_RELATIVE_ALT}:
            raise ValueError("mission_frame must be 0 (GLOBAL) or 3 (GLOBAL_RELATIVE_ALT).")
        if self.mission_command_anchor_delay_s < 0:
            raise ValueError("mission_command_anchor_delay_s cannot be negative.")
        if self.action_exit_clearance_m < 0:
            raise ValueError("action_exit_clearance_m cannot be negative.")
        if self.max_mission_items <= 0:
            raise ValueError("max_mission_items must be positive.")
        if not isinstance(self.action_bindings, dict):
            raise ValueError("action_bindings must be a JSON object/dictionary.")
        for action_id, binding in self.action_bindings.items():
            if action_id not in SUPPORTED_ACTION_IDS:
                raise ValueError(f"Unsupported action binding: {action_id}")
            if not isinstance(binding, dict):
                raise ValueError(f"Action binding for {action_id} must be an object.")
            command = str(binding.get("command", ""))
            if command not in {"MAV_CMD_DO_SET_SERVO", "MAV_CMD_DO_SET_RELAY"}:
                raise ValueError(
                    f"Action {action_id} must use MAV_CMD_DO_SET_SERVO or "
                    "MAV_CMD_DO_SET_RELAY."
                )
            duration_s = float(binding.get("duration_s", 0.0))
            restore_duration_s = float(binding.get("restore_duration_s", 0.0))
            if duration_s < 0 or restore_duration_s < 0:
                raise ValueError(f"Action durations cannot be negative: {action_id}")

    @property
    def vehicle_envelope_radius_m(self) -> float:
        """Conservative circle containing the rectangular vehicle."""
        return (
            0.5 * math.hypot(self.vehicle_length_m, self.vehicle_width_m)
            + self.safety_margin_m
        )

    @property
    def target_half_width_m(self) -> float:
        if self.target_coverage_half_width_m is None:
            return 0.5 * self.vehicle_width_m
        return float(self.target_coverage_half_width_m)


@dataclass(frozen=True)
class TargetActionCommand:
    sequence_order: int
    action_id: str
    trigger: str


@dataclass
class TargetTask:
    target_number: int
    target_description: str
    action_sequence: List[TargetActionCommand]


@dataclass
class AnnotationMap:
    origin_e: float
    origin_n: float
    start: XY
    goal: XY
    target_zones: Dict[int, Polygon]
    obstacle_zones: Dict[int, Polygon]
    risk_zones: Dict[int, Polygon]
    target_tasks: Dict[int, TargetTask]
    z_xy_global: np.ndarray
    z_values: np.ndarray
    source_csv: str


@dataclass
class SearchSample:
    x: float
    y: float
    heading: float
    curvature: float
    segment_length: float
    in_risk: bool
    phase: str = ""
    target_reached: str = ""


@dataclass
class SearchRecord:
    x: float
    y: float
    heading: float
    g_cost: float
    parent_id: Optional[int]
    primitive_samples: List[SearchSample]
    state_key: Tuple[int, int, int]


@dataclass
class LegResult:
    samples: List[SearchSample]
    weighted_length_m: float
    expansions: int
    terminal_heading_rad: float


@dataclass
class MissionItem:
    sequence: int
    current: int
    frame: int
    command: int
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    autocontinue: int = 1
    label: str = ""
    phase: str = ""
    target_number: Optional[int] = None
    logical_action_id: str = ""
    trigger: str = ""

    def to_wpl_line(self) -> str:
        values = [
            str(self.sequence),
            str(self.current),
            str(self.frame),
            str(self.command),
            f"{self.param1:.9f}",
            f"{self.param2:.9f}",
            f"{self.param3:.9f}",
            f"{self.param4:.9f}",
            f"{self.x:.10f}",
            f"{self.y:.10f}",
            f"{self.z:.6f}",
            str(self.autocontinue),
        ]
        return "\t".join(values)



def resolve_user_path(path: Path) -> Path:
    """Resolve relative paths against the V4 script directory."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = SCRIPT_DIR / resolved
    return resolved.resolve()


def create_timestamped_output_dir(output_root: Path) -> Path:
    """
    Create one new result folder for every run.

    Examples:
        planner_outputs_V4/20260723_153045
        planner_outputs_V4/20260723_153045_01
    """
    root = resolve_user_path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1

    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def load_config(path: Path) -> PlannerConfig:
    config = PlannerConfig()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    valid_names = {field.name for field in fields(PlannerConfig)}
    unknown = sorted(set(data) - valid_names)
    if unknown:
        raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")

    for name, value in data.items():
        if name == "curvature_ratios":
            value = tuple(float(item) for item in value)
        setattr(config, name, value)

    config.validate()
    return config


def _make_polygon(group: pd.DataFrame, label: str) -> Polygon:
    coordinates = group[["X/E", "Y/N"]].to_numpy(dtype=float)
    if len(coordinates) < 3:
        raise ValueError(f"{label} has fewer than three vertices.")

    polygon = Polygon(coordinates)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError(f"{label} is empty or invalid.")
    if polygon.geom_type != "Polygon":
        raise ValueError(f"{label} must form one Polygon.")
    return polygon


def _first_nonempty_text(values: Iterable[Any]) -> str:
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _parse_action_sequence(value: Any, target_number: int) -> List[TargetActionCommand]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Target {target_number} has invalid action_sequence JSON: {exc}"
            ) from exc
    else:
        payload = value

    if not isinstance(payload, list):
        raise ValueError(f"Target {target_number} action_sequence must be a JSON list.")

    commands: List[TargetActionCommand] = []
    seen_orders = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Target {target_number} action #{index} must be a JSON object."
            )
        order = int(item.get("sequence_order", index))
        action_id = str(item.get("action_id", "")).strip()
        trigger = str(item.get("trigger", "")).strip()
        if order <= 0 or order in seen_orders:
            raise ValueError(
                f"Target {target_number} has duplicate or invalid sequence_order {order}."
            )
        if action_id not in SUPPORTED_ACTION_IDS:
            raise ValueError(
                f"Target {target_number} uses unsupported action_id '{action_id}'."
            )
        if trigger not in SUPPORTED_ACTION_TRIGGERS:
            raise ValueError(
                f"Target {target_number} uses unsupported trigger '{trigger}'."
            )
        seen_orders.add(order)
        commands.append(TargetActionCommand(order, action_id, trigger))

    return sorted(commands, key=lambda command: command.sequence_order)


def _extract_tasks_from_frame(
    frame: pd.DataFrame,
    valid_target_numbers: Sequence[int],
    source_label: str,
) -> Dict[int, TargetTask]:
    if frame.empty:
        return {}
    if "zone_type" in frame.columns:
        frame = frame[frame["zone_type"].astype(str).str.strip() == "target_zones"]
    elif "type" in frame.columns:
        frame = frame[frame["type"].astype(str).str.strip() == "target_zones"]

    number_column = next(
        (name for name in ("target_number", "number", "zone_index") if name in frame.columns),
        None,
    )
    if number_column is None or "action_sequence" not in frame.columns:
        return {}

    working = frame.copy()
    working[number_column] = pd.to_numeric(working[number_column], errors="coerce")
    working = working.dropna(subset=[number_column])
    working[number_column] = working[number_column].astype(int)

    tasks: Dict[int, TargetTask] = {}
    for target_number, group in working.groupby(number_column, sort=True):
        target_number = int(target_number)
        if target_number not in valid_target_numbers:
            continue
        descriptions = (
            group["target_description"].tolist()
            if "target_description" in group.columns
            else []
        )
        description = _first_nonempty_text(descriptions)

        candidate_sequences: List[List[TargetActionCommand]] = []
        for raw_value in group["action_sequence"].tolist():
            parsed = _parse_action_sequence(raw_value, target_number)
            if parsed:
                candidate_sequences.append(parsed)

        if not candidate_sequences:
            continue
        canonical = candidate_sequences[0]
        for candidate in candidate_sequences[1:]:
            if candidate != canonical:
                raise ValueError(
                    f"Conflicting action sequences for target {target_number} in {source_label}. "
                    "If the VLM CSV contains multiple images, merge/georeference the targets and "
                    "assign globally unique target numbers before planning."
                )
        tasks[target_number] = TargetTask(
            target_number=target_number,
            target_description=description or f"target_{target_number}",
            action_sequence=canonical,
        )
    return tasks


def load_target_tasks(
    annotation_frame: pd.DataFrame,
    target_numbers: Sequence[int],
    action_csv_path: Optional[Path],
) -> Dict[int, TargetTask]:
    tasks = _extract_tasks_from_frame(
        annotation_frame,
        target_numbers,
        "annotation CSV",
    )
    if action_csv_path is not None:
        if not action_csv_path.exists():
            raise FileNotFoundError(f"Action CSV not found: {action_csv_path}")
        action_frame = pd.read_csv(action_csv_path)
        external_tasks = _extract_tasks_from_frame(
            action_frame,
            target_numbers,
            str(action_csv_path),
        )
        tasks.update(external_tasks)
    return tasks


def load_annotations(csv_path: Path, action_csv_path: Optional[Path] = None) -> AnnotationMap:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = ["type", "number", "X/E", "Y/N", "Z/U"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["type"] = frame["type"].astype(str).str.strip()
    frame["number"] = pd.to_numeric(frame["number"], errors="raise").astype(int)
    for column in ("X/E", "Y/N", "Z/U"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    allowed = {
        "start",
        "goal",
        "target_zones",
        "obstacle_zones",
        "risk_zones",
    }
    invalid_types = sorted(set(frame["type"]) - allowed)
    if invalid_types:
        raise ValueError(f"Unsupported type values: {', '.join(invalid_types)}")

    start_rows = frame[frame["type"] == "start"]
    goal_rows = frame[frame["type"] == "goal"]
    if len(start_rows) != 1:
        raise ValueError("The CSV must contain exactly one start row.")
    if len(goal_rows) != 1:
        raise ValueError("The CSV must contain exactly one goal row.")

    origin_e = float(frame["X/E"].min())
    origin_n = float(frame["Y/N"].min())

    def local_point(row: pd.Series) -> XY:
        return (
            float(row["X/E"] - origin_e),
            float(row["Y/N"] - origin_n),
        )

    def local_polygon(global_polygon: Polygon) -> Polygon:
        coordinates = np.asarray(global_polygon.exterior.coords, dtype=float)
        coordinates[:, 0] -= origin_e
        coordinates[:, 1] -= origin_n
        return Polygon(coordinates)

    polygons_global: Dict[str, Dict[int, Polygon]] = {}
    for zone_type in ("target_zones", "obstacle_zones", "risk_zones"):
        polygons_global[zone_type] = {}
        subset = frame[frame["type"] == zone_type]
        for number, group in subset.groupby("number", sort=True):
            polygons_global[zone_type][int(number)] = _make_polygon(
                group,
                f"{zone_type} #{number}",
            )

    start = local_point(start_rows.iloc[0])
    goal = local_point(goal_rows.iloc[0])

    z_xy_global = frame[["X/E", "Y/N"]].to_numpy(dtype=float)
    z_values = frame["Z/U"].to_numpy(dtype=float)
    target_numbers = sorted(polygons_global["target_zones"])
    target_tasks = load_target_tasks(
        annotation_frame=frame,
        target_numbers=target_numbers,
        action_csv_path=action_csv_path,
    )

    return AnnotationMap(
        origin_e=origin_e,
        origin_n=origin_n,
        start=start,
        goal=goal,
        target_zones={
            number: local_polygon(polygon)
            for number, polygon in polygons_global["target_zones"].items()
        },
        obstacle_zones={
            number: local_polygon(polygon)
            for number, polygon in polygons_global["obstacle_zones"].items()
        },
        risk_zones={
            number: local_polygon(polygon)
            for number, polygon in polygons_global["risk_zones"].items()
        },
        target_tasks=target_tasks,
        z_xy_global=z_xy_global,
        z_values=z_values,
        source_csv=str(csv_path.resolve()),
    )


def geometry_bounds(annotation_map: AnnotationMap, margin: float) -> Tuple[float, ...]:
    geometries: List[object] = [
        Point(annotation_map.start),
        Point(annotation_map.goal),
    ]
    geometries.extend(annotation_map.target_zones.values())
    geometries.extend(annotation_map.obstacle_zones.values())
    geometries.extend(annotation_map.risk_zones.values())

    xmin, ymin, xmax, ymax = unary_union(geometries).bounds
    return xmin - margin, ymin - margin, xmax + margin, ymax + margin


class GridCostMap:
    def __init__(
        self,
        annotation_map: AnnotationMap,
        config: PlannerConfig,
        obstacle_inflated,
        risk_inflated,
    ) -> None:
        self.resolution = config.order_grid_resolution_m
        extra_margin = config.map_margin_m + config.vehicle_envelope_radius_m
        self.bounds = geometry_bounds(annotation_map, extra_margin)
        xmin, ymin, xmax, ymax = self.bounds

        self.xs = np.arange(
            xmin,
            xmax + 0.5 * self.resolution,
            self.resolution,
            dtype=float,
        )
        self.ys = np.arange(
            ymin,
            ymax + 0.5 * self.resolution,
            self.resolution,
            dtype=float,
        )

        self.blocked = np.zeros((len(self.ys), len(self.xs)), dtype=bool)
        self.risk = np.zeros_like(self.blocked)

        obstacle_prepared = prep(obstacle_inflated)
        risk_prepared = prep(risk_inflated) if not risk_inflated.is_empty else None

        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                point = Point(float(x), float(y))
                self.blocked[iy, ix] = obstacle_prepared.intersects(point)
                self.risk[iy, ix] = (
                    risk_prepared.intersects(point)
                    if risk_prepared is not None
                    else False
                )

        self.obstacle_inflated = obstacle_inflated

    def point_to_index(self, point: XY) -> Tuple[int, int]:
        ix = int(round((point[0] - self.xs[0]) / self.resolution))
        iy = int(round((point[1] - self.ys[0]) / self.resolution))
        return iy, ix

    def index_to_point(self, index: Tuple[int, int]) -> XY:
        iy, ix = index
        return float(self.xs[ix]), float(self.ys[iy])

    def nearest_visible_free_cell(
        self,
        point: XY,
        maximum_radius_m: float = 4.0,
    ) -> Tuple[int, int]:
        iy0, ix0 = self.point_to_index(point)
        maximum_ring = int(math.ceil(maximum_radius_m / self.resolution))

        for ring in range(maximum_ring + 1):
            candidates: List[Tuple[float, Tuple[int, int]]] = []
            for dy in range(-ring, ring + 1):
                for dx in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    iy, ix = iy0 + dy, ix0 + dx
                    if not (
                        0 <= iy < self.blocked.shape[0]
                        and 0 <= ix < self.blocked.shape[1]
                    ):
                        continue
                    if self.blocked[iy, ix]:
                        continue

                    candidate = self.index_to_point((iy, ix))
                    connecting_line = LineString([point, candidate])
                    if connecting_line.intersects(self.obstacle_inflated):
                        continue

                    squared_distance = (
                        (candidate[0] - point[0]) ** 2
                        + (candidate[1] - point[1]) ** 2
                    )
                    candidates.append((squared_distance, (iy, ix)))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]

        raise RuntimeError(
            f"No visible free grid cell found near point {point}. "
            "Check the vehicle size, safety margin, or map annotations."
        )

    def shortest_weighted_cost(
        self,
        start: XY,
        goal: XY,
        risk_speed_ratio: float,
    ) -> float:
        start_index = self.nearest_visible_free_cell(start)
        goal_index = self.nearest_visible_free_cell(goal)

        directions = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_index))
        g_score = {start_index: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            current_g = g_score[current]
            if current == goal_index:
                return current_g

            iy, ix = current
            for dy, dx in directions:
                ny, nx = iy + dy, ix + dx
                if not (
                    0 <= ny < self.blocked.shape[0]
                    and 0 <= nx < self.blocked.shape[1]
                ):
                    continue
                if self.blocked[ny, nx]:
                    continue

                step_length = self.resolution * math.hypot(dx, dy)
                current_factor = (
                    1.0 / risk_speed_ratio if self.risk[iy, ix] else 1.0
                )
                next_factor = (
                    1.0 / risk_speed_ratio if self.risk[ny, nx] else 1.0
                )
                weighted_step = step_length * 0.5 * (
                    current_factor + next_factor
                )

                neighbor = (ny, nx)
                tentative = current_g + weighted_step
                if tentative + 1e-12 >= g_score.get(neighbor, math.inf):
                    continue

                g_score[neighbor] = tentative
                heuristic = self.resolution * math.hypot(
                    nx - goal_index[1],
                    ny - goal_index[0],
                )
                heapq.heappush(
                    open_heap,
                    (tentative + heuristic, neighbor),
                )

        raise RuntimeError(
            f"No 2-D route exists between {start} and {goal}. "
            "The inflated obstacle map may disconnect the workspace."
        )


def calculate_pairwise_costs(
    annotation_map: AnnotationMap,
    grid_map: GridCostMap,
    config: PlannerConfig,
) -> Tuple[List[XY], np.ndarray]:
    target_numbers = sorted(annotation_map.target_zones)
    target_points = [
        (
            float(annotation_map.target_zones[number].centroid.x),
            float(annotation_map.target_zones[number].centroid.y),
        )
        for number in target_numbers
    ]

    task_points = [annotation_map.start] + target_points + [annotation_map.goal]
    size = len(task_points)
    matrix = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(i + 1, size):
            cost = grid_map.shortest_weighted_cost(
                task_points[i],
                task_points[j],
                config.risk_speed_ratio,
            )
            matrix[i, j] = cost
            matrix[j, i] = cost

    return task_points, matrix


def held_karp_order(cost_matrix: np.ndarray, target_count: int) -> List[int]:
    """Exact start -> all targets -> goal order. Matrix nodes are 0, targets, goal."""
    if target_count == 0:
        return [0, 1]

    full_mask = (1 << target_count) - 1
    dp: Dict[Tuple[int, int], float] = {}
    parent: Dict[Tuple[int, int], int] = {}

    for target_index in range(target_count):
        node_index = target_index + 1
        dp[(1 << target_index, target_index)] = cost_matrix[0, node_index]
        parent[(1 << target_index, target_index)] = -1

    for mask in range(1, full_mask + 1):
        for last in range(target_count):
            key = (mask, last)
            if key not in dp:
                continue
            current_cost = dp[key]

            for next_target in range(target_count):
                bit = 1 << next_target
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_key = (next_mask, next_target)
                candidate = (
                    current_cost
                    + cost_matrix[last + 1, next_target + 1]
                )
                if candidate < dp.get(next_key, math.inf):
                    dp[next_key] = candidate
                    parent[next_key] = last

    goal_node = target_count + 1
    best_last = min(
        range(target_count),
        key=lambda last: dp[(full_mask, last)]
        + cost_matrix[last + 1, goal_node],
    )

    target_order_reversed: List[int] = []
    mask = full_mask
    last = best_last
    while last >= 0:
        target_order_reversed.append(last + 1)
        previous = parent[(mask, last)]
        mask ^= 1 << last
        last = previous

    target_order = list(reversed(target_order_reversed))
    return [0] + target_order + [goal_node]


def approximate_order(cost_matrix: np.ndarray, target_count: int) -> List[int]:
    """Nearest-neighbor initialization followed by repeated 2-opt."""
    if target_count == 0:
        return [0, 1]

    unvisited = set(range(1, target_count + 1))
    route = [0]
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda node: cost_matrix[current, node])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    route.append(target_count + 1)

    def route_cost(candidate: Sequence[int]) -> float:
        return sum(
            cost_matrix[first, second]
            for first, second in zip(candidate[:-1], candidate[1:])
        )

    improved = True
    while improved:
        improved = False
        current_cost = route_cost(route)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                candidate = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                candidate_cost = route_cost(candidate)
                if candidate_cost + 1e-9 < current_cost:
                    route = candidate
                    improved = True
                    current_cost = candidate_cost
        # Repeat until no improving reversal remains.
    return route


def choose_target_order(
    cost_matrix: np.ndarray,
    target_count: int,
    exact_order_max_targets: int,
) -> Tuple[List[int], str]:
    if target_count <= exact_order_max_targets:
        return held_karp_order(cost_matrix, target_count), "Held-Karp exact"
    return approximate_order(cost_matrix, target_count), "Nearest-neighbor + 2-opt"


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def propagate_pose(
    x: float,
    y: float,
    heading: float,
    curvature: float,
    distance: float,
) -> Tuple[float, float, float]:
    if abs(curvature) < 1e-12:
        return (
            x + distance * math.cos(heading),
            y + distance * math.sin(heading),
            heading,
        )

    new_heading = heading + curvature * distance
    new_x = x + (math.sin(new_heading) - math.sin(heading)) / curvature
    new_y = y - (math.cos(new_heading) - math.cos(heading)) / curvature
    return new_x, new_y, wrap_angle(new_heading)


class ForwardHybridAStar:
    def __init__(
        self,
        annotation_map: AnnotationMap,
        config: PlannerConfig,
        obstacle_inflated,
        risk_inflated,
    ) -> None:
        self.map = annotation_map
        self.config = config
        self.obstacle_inflated = obstacle_inflated
        self.risk_inflated = risk_inflated
        self.obstacle_prepared = prep(obstacle_inflated)
        self.risk_prepared = (
            prep(risk_inflated) if not risk_inflated.is_empty else None
        )

        margin = config.map_margin_m + config.vehicle_envelope_radius_m
        self.bounds = geometry_bounds(annotation_map, margin)

        maximum_curvature = 1.0 / config.minimum_turning_radius_m
        self.curvatures = tuple(
            ratio * maximum_curvature
            for ratio in config.curvature_ratios
        )
        self.heading_resolution = 2.0 * math.pi / config.heading_bins

    def _state_key(
        self,
        x: float,
        y: float,
        heading: float,
    ) -> Tuple[int, int, int]:
        ix = int(
            round((x - self.bounds[0]) / self.config.state_xy_resolution_m)
        )
        iy = int(
            round((y - self.bounds[1]) / self.config.state_xy_resolution_m)
        )
        heading_index = int(
            round((wrap_angle(heading) + math.pi) / self.heading_resolution)
        ) % self.config.heading_bins
        return ix, iy, heading_index

    def _inside_bounds(self, x: float, y: float) -> bool:
        xmin, ymin, xmax, ymax = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def _collides(self, x: float, y: float) -> bool:
        return self.obstacle_prepared.intersects(Point(x, y))

    def _in_risk(self, x: float, y: float) -> bool:
        return (
            self.risk_prepared.intersects(Point(x, y))
            if self.risk_prepared is not None
            else False
        )

    def search(
        self,
        start_xy: XY,
        start_heading_rad: Optional[float],
        goal_polygon: Optional[Polygon] = None,
        goal_point: Optional[XY] = None,
        phase: str = "",
    ) -> LegResult:
        if (goal_polygon is None) == (goal_point is None):
            raise ValueError("Provide exactly one of goal_polygon or goal_point.")

        coverage_half_width = self.config.target_half_width_m

        def goal_reached(x: float, y: float) -> bool:
            point = Point(x, y)
            if goal_polygon is not None:
                return point.distance(goal_polygon) <= coverage_half_width
            return point.distance(Point(goal_point)) <= self.config.goal_tolerance_m

        def heuristic(x: float, y: float) -> float:
            point = Point(x, y)
            if goal_polygon is not None:
                return max(
                    0.0,
                    point.distance(goal_polygon) - coverage_half_width,
                )
            return point.distance(Point(goal_point))

        if self._collides(*start_xy):
            raise RuntimeError(
                f"Leg '{phase}' starts inside an inflated obstacle."
            )

        records: List[SearchRecord] = []
        best_g: Dict[Tuple[int, int, int], float] = {}
        open_heap: List[Tuple[float, int, int]] = []
        counter = 0

        if start_heading_rad is None:
            initial_headings = [
                -math.pi + index * self.heading_resolution
                for index in range(self.config.heading_bins)
            ]
        else:
            initial_headings = [wrap_angle(start_heading_rad)]

        for heading in initial_headings:
            key = self._state_key(start_xy[0], start_xy[1], heading)
            if 0.0 >= best_g.get(key, math.inf):
                continue

            sample = SearchSample(
                x=start_xy[0],
                y=start_xy[1],
                heading=heading,
                curvature=0.0,
                segment_length=0.0,
                in_risk=self._in_risk(*start_xy),
                phase=phase,
            )
            record = SearchRecord(
                x=start_xy[0],
                y=start_xy[1],
                heading=heading,
                g_cost=0.0,
                parent_id=None,
                primitive_samples=[sample],
                state_key=key,
            )
            record_id = len(records)
            records.append(record)
            best_g[key] = 0.0
            heapq.heappush(
                open_heap,
                (heuristic(record.x, record.y), counter, record_id),
            )
            counter += 1

        expansions = 0
        goal_record_id: Optional[int] = None

        while open_heap:
            _, _, record_id = heapq.heappop(open_heap)
            current = records[record_id]

            if current.g_cost > best_g.get(current.state_key, math.inf) + 1e-12:
                continue

            expansions += 1
            if expansions > self.config.max_expansions_per_leg:
                break

            if goal_reached(current.x, current.y):
                goal_record_id = record_id
                break

            for curvature in self.curvatures:
                substep_count = max(
                    2,
                    int(
                        math.ceil(
                            self.config.motion_step_m
                            / self.config.collision_check_step_m
                        )
                    ),
                )
                substep_distance = self.config.motion_step_m / substep_count

                x = current.x
                y = current.y
                heading = current.heading
                primitive_samples: List[SearchSample] = []
                weighted_increment = 0.0
                valid = True

                for _ in range(substep_count):
                    new_x, new_y, new_heading = propagate_pose(
                        x,
                        y,
                        heading,
                        curvature,
                        substep_distance,
                    )
                    if (
                        not self._inside_bounds(new_x, new_y)
                        or self._collides(new_x, new_y)
                    ):
                        valid = False
                        break

                    midpoint_x = 0.5 * (x + new_x)
                    midpoint_y = 0.5 * (y + new_y)
                    in_risk = self._in_risk(midpoint_x, midpoint_y)
                    cost_factor = (
                        1.0 / self.config.risk_speed_ratio
                        if in_risk
                        else 1.0
                    )
                    weighted_increment += substep_distance * cost_factor

                    primitive_samples.append(
                        SearchSample(
                            x=new_x,
                            y=new_y,
                            heading=new_heading,
                            curvature=curvature,
                            segment_length=substep_distance,
                            in_risk=in_risk,
                            phase=phase,
                        )
                    )
                    x, y, heading = new_x, new_y, new_heading

                if not valid:
                    continue

                new_cost = current.g_cost + weighted_increment
                new_key = self._state_key(x, y, heading)
                if new_cost + 1e-12 >= best_g.get(new_key, math.inf):
                    continue

                new_record = SearchRecord(
                    x=x,
                    y=y,
                    heading=heading,
                    g_cost=new_cost,
                    parent_id=record_id,
                    primitive_samples=primitive_samples,
                    state_key=new_key,
                )
                new_record_id = len(records)
                records.append(new_record)
                best_g[new_key] = new_cost
                heapq.heappush(
                    open_heap,
                    (
                        new_cost + heuristic(x, y),
                        counter,
                        new_record_id,
                    ),
                )
                counter += 1

        if goal_record_id is None:
            raise RuntimeError(
                f"No forward-only curvature-constrained path found for '{phase}' "
                f"after {expansions} node expansions. Try reducing the vehicle "
                "dimensions/safety margin, decreasing state_xy_resolution_m and "
                "motion_step_m, increasing heading_bins, or increasing "
                "map_margin_m."
            )

        record_ids: List[int] = []
        current_id: Optional[int] = goal_record_id
        while current_id is not None:
            record_ids.append(current_id)
            current_id = records[current_id].parent_id
        record_ids.reverse()

        samples: List[SearchSample] = []
        for index, item_id in enumerate(record_ids):
            record_samples = records[item_id].primitive_samples
            if index == 0:
                samples.extend(record_samples)
            else:
                samples.extend(record_samples)

        terminal_record = records[goal_record_id]
        return LegResult(
            samples=samples,
            weighted_length_m=terminal_record.g_cost,
            expansions=expansions,
            terminal_heading_rad=terminal_record.heading,
        )


def make_vehicle_polygon(
    x: float,
    y: float,
    heading_rad: float,
    config: PlannerConfig,
) -> Polygon:
    rectangle = box(
        -0.5 * config.vehicle_length_m,
        -0.5 * config.vehicle_width_m,
        0.5 * config.vehicle_length_m,
        0.5 * config.vehicle_width_m,
    )
    rectangle = rotate(
        rectangle,
        heading_rad,
        origin=(0.0, 0.0),
        use_radians=True,
    )
    rectangle = translate(rectangle, xoff=x, yoff=y)
    return rectangle


def interpolate_z(
    query_xy_global: np.ndarray,
    known_xy_global: np.ndarray,
    known_z: np.ndarray,
    neighbor_count: int = 4,
) -> np.ndarray:
    output = np.zeros(len(query_xy_global), dtype=float)
    neighbor_count = max(1, min(neighbor_count, len(known_xy_global)))

    for index, point in enumerate(query_xy_global):
        distances = np.linalg.norm(known_xy_global - point, axis=1)
        nearest = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        nearest_distances = distances[nearest]

        exact = nearest_distances < 1e-10
        if np.any(exact):
            output[index] = float(known_z[nearest[exact][0]])
            continue

        weights = 1.0 / np.maximum(nearest_distances, 1e-9) ** 2
        output[index] = float(
            np.sum(weights * known_z[nearest]) / np.sum(weights)
        )
    return output


def validate_and_measure(
    samples: Sequence[SearchSample],
    annotation_map: AnnotationMap,
    config: PlannerConfig,
) -> Dict[str, object]:
    if len(samples) < 2:
        raise RuntimeError("The final path contains fewer than two samples.")

    path_xy = [(sample.x, sample.y) for sample in samples]
    centerline = LineString(path_xy)
    swept_corridor = centerline.buffer(
        config.target_half_width_m,
        cap_style=2,
        join_style=1,
    )

    covered_targets = [
        number
        for number, polygon in sorted(annotation_map.target_zones.items())
        if swept_corridor.intersects(polygon)
    ]
    all_target_numbers = sorted(annotation_map.target_zones)
    missed_targets = [
        number for number in all_target_numbers if number not in covered_targets
    ]

    obstacle_union = (
        unary_union(list(annotation_map.obstacle_zones.values()))
        if annotation_map.obstacle_zones
        else Polygon()
    )
    exact_collision = False
    collision_sample_index: Optional[int] = None
    for index, sample in enumerate(samples):
        vehicle = make_vehicle_polygon(
            sample.x,
            sample.y,
            sample.heading,
            config,
        ).buffer(config.safety_margin_m)
        if not obstacle_union.is_empty and vehicle.intersects(obstacle_union):
            exact_collision = True
            collision_sample_index = index
            break

    total_length = sum(sample.segment_length for sample in samples[1:])
    risk_length = sum(
        sample.segment_length
        for sample in samples[1:]
        if sample.in_risk
    )
    normal_length = total_length - risk_length
    weighted_length = (
        normal_length + risk_length / config.risk_speed_ratio
    )
    estimated_time = weighted_length / config.normal_speed_mps

    used_curvatures = [
        abs(sample.curvature)
        for sample in samples[1:]
        if abs(sample.curvature) > 1e-12
    ]
    minimum_observed_radius = (
        1.0 / max(used_curvatures)
        if used_curvatures
        else math.inf
    )

    final_error = Point(samples[-1].x, samples[-1].y).distance(
        Point(annotation_map.goal)
    )

    return {
        "total_path_length_m": total_length,
        "normal_zone_length_m": normal_length,
        "risk_zone_length_m": risk_length,
        "weighted_equivalent_length_m": weighted_length,
        "estimated_travel_time_s": estimated_time,
        "estimated_travel_time_min": estimated_time / 60.0,
        "minimum_observed_turning_radius_m": (
            minimum_observed_radius
            if math.isfinite(minimum_observed_radius)
            else None
        ),
        "commanded_minimum_turning_radius_m": (
            config.minimum_turning_radius_m
        ),
        "final_goal_position_error_m": final_error,
        "covered_target_numbers": covered_targets,
        "missed_target_numbers": missed_targets,
        "all_targets_covered": not missed_targets,
        "exact_vehicle_collision_free": not exact_collision,
        "collision_sample_index": collision_sample_index,
        "path_sample_count": len(samples),
        "initial_heading_deg": math.degrees(samples[0].heading) % 360.0,
        "final_heading_deg": math.degrees(samples[-1].heading) % 360.0,
    }


def samples_to_dataframe(
    samples: Sequence[SearchSample],
    annotation_map: AnnotationMap,
    config: PlannerConfig,
) -> pd.DataFrame:
    local_xy = np.array([(sample.x, sample.y) for sample in samples])
    global_xy = local_xy + np.array(
        [annotation_map.origin_e, annotation_map.origin_n]
    )
    z_values = interpolate_z(
        global_xy,
        annotation_map.z_xy_global,
        annotation_map.z_values,
    )

    cumulative_length = 0.0
    cumulative_time = 0.0
    rows: List[Dict[str, object]] = []

    for index, (sample, xy, z_value) in enumerate(
        zip(samples, global_xy, z_values)
    ):
        if index == 0:
            segment_time = 0.0
        else:
            speed = (
                config.normal_speed_mps * config.risk_speed_ratio
                if sample.in_risk
                else config.normal_speed_mps
            )
            segment_time = sample.segment_length / speed
            cumulative_length += sample.segment_length
            cumulative_time += segment_time

        turning_radius = (
            1.0 / abs(sample.curvature)
            if abs(sample.curvature) > 1e-12
            else math.inf
        )

        rows.append(
            {
                "sequence": index,
                "X/E": xy[0],
                "Y/N": xy[1],
                "Z/U": z_value,
                "heading_deg": math.degrees(sample.heading) % 360.0,
                "curvature_1_per_m": sample.curvature,
                "turning_radius_m": (
                    turning_radius if math.isfinite(turning_radius) else ""
                ),
                "in_risk_zone": int(sample.in_risk),
                "segment_length_m": sample.segment_length,
                "cumulative_length_m": cumulative_length,
                "segment_time_s": segment_time,
                "cumulative_time_s": cumulative_time,
                "phase": sample.phase,
                "target_reached": sample.target_reached,
            }
        )

    return pd.DataFrame(rows)



def select_mission_sample_indices(
    samples: Sequence[SearchSample],
    spacing_m: float,
) -> List[int]:
    """Down-sample the dense path while preserving task and risk boundaries."""
    if not samples:
        return []

    mandatory = {0, len(samples) - 1}
    for index in range(1, len(samples)):
        if samples[index].target_reached:
            mandatory.add(index)
        if samples[index].in_risk != samples[index - 1].in_risk:
            # Preserve both sides of the boundary. Speed commands are inserted
            # after reaching index-1 and before driving toward index.
            mandatory.add(index - 1)
            mandatory.add(index)

    selected = [0]
    distance_since_last = 0.0
    for index in range(1, len(samples)):
        distance_since_last += samples[index].segment_length
        if index in mandatory or distance_since_last >= spacing_m:
            if index != selected[-1]:
                selected.append(index)
            distance_since_last = 0.0
    if selected[-1] != len(samples) - 1:
        selected.append(len(samples) - 1)
    return selected


def _append_mission_item(
    items: List[MissionItem],
    **kwargs: Any,
) -> MissionItem:
    item = MissionItem(sequence=len(items), **kwargs)
    items.append(item)
    return item


def _append_command_anchor(
    items: List[MissionItem],
    config: PlannerConfig,
    label: str,
    phase: str,
    target_number: Optional[int] = None,
) -> MissionItem:
    """
    Insert a NAV_DELAY item so subsequent DO commands begin only after the
    preceding waypoint has been reached. Rover holds position during NAV_DELAY.
    """
    return _append_mission_item(
        items,
        current=0,
        frame=MAV_FRAME_MISSION,
        command=MAV_CMD_NAV_DELAY,
        param1=float(int(config.mission_command_anchor_delay_s)),
        label=label,
        phase=phase,
        target_number=target_number,
    )


def _validate_action_binding(action_id: str, config: PlannerConfig) -> Dict[str, Any]:
    binding = config.action_bindings.get(action_id)
    if binding is None:
        raise ValueError(f"No hardware binding is configured for action '{action_id}'.")
    if not bool(binding.get("hardware_verified", False)):
        raise ValueError(
            f"Action '{action_id}' is used by the mission, but hardware_verified is false. "
            "Calibrate the Pixhawk output channel/PWM on the actual mechanism, then set "
            "hardware_verified=true in the config before exporting an executable mission."
        )
    return binding


def _append_action_command(
    items: List[MissionItem],
    action_records: List[Dict[str, Any]],
    action: TargetActionCommand,
    target_number: int,
    phase: str,
    config: PlannerConfig,
) -> None:
    binding = _validate_action_binding(action.action_id, config)
    command_name = str(binding["command"])

    if command_name == "MAV_CMD_DO_SET_SERVO":
        channel = int(binding["servo_channel"])
        pwm = int(binding["pwm"])
        if channel <= 0:
            raise ValueError(f"Invalid servo_channel for {action.action_id}: {channel}")
        if not 500 <= pwm <= 2500:
            raise ValueError(f"Unsafe/unusual PWM for {action.action_id}: {pwm}")
        item = _append_mission_item(
            items,
            current=0,
            frame=MAV_FRAME_MISSION,
            command=MAV_CMD_DO_SET_SERVO,
            param1=float(channel),
            param2=float(pwm),
            label=f"ACTION {action.action_id}",
            phase=phase,
            target_number=target_number,
            logical_action_id=action.action_id,
            trigger=action.trigger,
        )
    elif command_name == "MAV_CMD_DO_SET_RELAY":
        relay_number = int(binding["relay_number"])
        relay_state = int(binding["relay_state"])
        item = _append_mission_item(
            items,
            current=0,
            frame=MAV_FRAME_MISSION,
            command=MAV_CMD_DO_SET_RELAY,
            param1=float(relay_number),
            param2=float(relay_state),
            label=f"ACTION {action.action_id}",
            phase=phase,
            target_number=target_number,
            logical_action_id=action.action_id,
            trigger=action.trigger,
        )
    else:
        raise ValueError(f"Unsupported action command mapping: {command_name}")

    action_records.append(
        {
            "target_number": target_number,
            "sequence_order": action.sequence_order,
            "action_id": action.action_id,
            "trigger": action.trigger,
            "mission_item_sequence": item.sequence,
            "mavlink_command": command_name,
            "binding": json.dumps(binding, ensure_ascii=False, separators=(",", ":")),
        }
    )

    duration_s = float(binding.get("duration_s", 0.0))
    if duration_s > 0:
        _append_mission_item(
            items,
            current=0,
            frame=MAV_FRAME_MISSION,
            command=MAV_CMD_NAV_DELAY,
            param1=float(math.ceil(duration_s)),
            label=f"WAIT after {action.action_id}",
            phase=phase,
            target_number=target_number,
            logical_action_id=action.action_id,
            trigger=action.trigger,
        )

    restore_pwm = binding.get("restore_pwm")
    if command_name == "MAV_CMD_DO_SET_SERVO" and restore_pwm is not None:
        restore_pwm = int(restore_pwm)
        _append_mission_item(
            items,
            current=0,
            frame=MAV_FRAME_MISSION,
            command=MAV_CMD_DO_SET_SERVO,
            param1=float(int(binding["servo_channel"])),
            param2=float(restore_pwm),
            label=f"RESTORE after {action.action_id}",
            phase=phase,
            target_number=target_number,
            logical_action_id=action.action_id,
            trigger=action.trigger,
        )
        restore_duration_s = float(binding.get("restore_duration_s", 0.0))
        if restore_duration_s > 0:
            _append_mission_item(
                items,
                current=0,
                frame=MAV_FRAME_MISSION,
                command=MAV_CMD_NAV_DELAY,
                param1=float(math.ceil(restore_duration_s)),
                label=f"WAIT restore {action.action_id}",
                phase=phase,
                target_number=target_number,
                logical_action_id=action.action_id,
                trigger=action.trigger,
            )


def build_ardurover_mission(
    samples: Sequence[SearchSample],
    annotation_map: AnnotationMap,
    config: PlannerConfig,
) -> Tuple[List[MissionItem], List[Dict[str, Any]], List[int]]:
    if config.require_target_actions:
        missing = sorted(set(annotation_map.target_zones) - set(annotation_map.target_tasks))
        if missing:
            raise ValueError(
                "No executable action sequence was provided for target(s): "
                + ", ".join(map(str, missing))
                + ". Add target_description/action_sequence to the annotation CSV or "
                "provide --actions with a V6 action-assignment CSV."
            )

    selected_indices = select_mission_sample_indices(
        samples,
        config.mission_waypoint_spacing_m,
    )
    if not selected_indices:
        raise RuntimeError("No mission waypoints were selected from the planned path.")

    global_xy = np.array(
        [
            (
                samples[index].x + annotation_map.origin_e,
                samples[index].y + annotation_map.origin_n,
            )
            for index in selected_indices
        ],
        dtype=float,
    )
    transformer = Transformer.from_crs(
        config.input_crs,
        config.output_crs,
        always_xy=True,
    )
    longitudes, latitudes = transformer.transform(global_xy[:, 0], global_xy[:, 1])

    items: List[MissionItem] = []
    action_records: List[Dict[str, Any]] = []

    # Mission Planner/ArduPilot convention: item 0 is the home-position row.
    _append_mission_item(
        items,
        current=1,
        frame=MAV_FRAME_GLOBAL,
        command=MAV_CMD_NAV_WAYPOINT,
        x=float(latitudes[0]),
        y=float(longitudes[0]),
        z=float(config.mission_altitude_m),
        label="HOME",
        phase="home",
    )

    _append_mission_item(
        items,
        current=0,
        frame=MAV_FRAME_MISSION,
        command=MAV_CMD_DO_CHANGE_SPEED,
        param2=float(config.normal_speed_mps),
        label="SET NORMAL SPEED",
        phase="mission_start",
    )
    current_risk_state = False
    pending_exit_actions: List[Tuple[int, TargetActionCommand]] = []

    for point_order, sample_index in enumerate(selected_indices):
        sample = samples[sample_index]
        target_number: Optional[int] = None
        if sample.target_reached and sample.target_reached.isdigit():
            target_number = int(sample.target_reached)

        _append_mission_item(
            items,
            current=0,
            frame=config.mission_frame,
            command=MAV_CMD_NAV_WAYPOINT,
            param1=0.0,
            param2=float(config.mission_acceptance_radius_m),
            param3=0.0,
            param4=0.0,
            x=float(latitudes[point_order]),
            y=float(longitudes[point_order]),
            z=float(config.mission_altitude_m),
            label=(
                f"TARGET {target_number}"
                if target_number is not None
                else ("GOAL" if sample.target_reached == "goal" else "PATH WAYPOINT")
            ),
            phase=sample.phase,
            target_number=target_number,
        )

        # Execute deferred on_zone_exit actions after this waypoint is reached
        # and the vehicle is clear of the corresponding target region.
        ready_by_target: Dict[int, List[TargetActionCommand]] = {}
        still_pending: List[Tuple[int, TargetActionCommand]] = []
        current_point = Point(sample.x, sample.y)
        for pending_target, pending_action in pending_exit_actions:
            clearance = current_point.distance(annotation_map.target_zones[pending_target])
            required_clearance = config.target_half_width_m + config.action_exit_clearance_m
            if clearance >= required_clearance:
                ready_by_target.setdefault(pending_target, []).append(pending_action)
            else:
                still_pending.append((pending_target, pending_action))
        pending_exit_actions = still_pending

        for exited_target, exit_actions in sorted(ready_by_target.items()):
            _append_command_anchor(
                items,
                config,
                label=f"ANCHOR EXIT TARGET {exited_target}",
                phase=f"exit_target_{exited_target}",
                target_number=exited_target,
            )
            for action in sorted(exit_actions, key=lambda value: value.sequence_order):
                _append_action_command(
                    items=items,
                    action_records=action_records,
                    action=action,
                    target_number=exited_target,
                    phase=f"exit_target_{exited_target}",
                    config=config,
                )

        if target_number is not None:
            task = annotation_map.target_tasks.get(target_number)
            if task is not None:
                immediate_actions = [
                    action
                    for action in task.action_sequence
                    if action.trigger != "on_zone_exit"
                ]
                exit_actions = [
                    action
                    for action in task.action_sequence
                    if action.trigger == "on_zone_exit"
                ]
                if immediate_actions:
                    _append_command_anchor(
                        items,
                        config,
                        label=f"ANCHOR TARGET {target_number}",
                        phase=sample.phase,
                        target_number=target_number,
                    )
                    for action in immediate_actions:
                        _append_action_command(
                            items=items,
                            action_records=action_records,
                            action=action,
                            target_number=target_number,
                            phase=sample.phase,
                            config=config,
                        )
                pending_exit_actions.extend(
                    (target_number, action) for action in exit_actions
                )

        # Set the speed for the NEXT path segment only after this waypoint has
        # been reached. Risk-boundary samples are preserved on both sides.
        if point_order + 1 < len(selected_indices):
            next_sample = samples[selected_indices[point_order + 1]]
            next_segment_risk = bool(next_sample.in_risk)
            if next_segment_risk != current_risk_state:
                _append_command_anchor(
                    items,
                    config,
                    label="ANCHOR SPEED CHANGE",
                    phase=sample.phase,
                )
                requested_speed = (
                    config.normal_speed_mps * config.risk_speed_ratio
                    if next_segment_risk
                    else config.normal_speed_mps
                )
                _append_mission_item(
                    items,
                    current=0,
                    frame=MAV_FRAME_MISSION,
                    command=MAV_CMD_DO_CHANGE_SPEED,
                    param2=float(requested_speed),
                    label=(
                        "SET RISK SPEED"
                        if next_segment_risk
                        else "RESTORE NORMAL SPEED"
                    ),
                    phase=sample.phase,
                )
                current_risk_state = next_segment_risk

    # If the mission ends before the configured exit clearance is reached,
    # anchor and execute the remaining exit actions at the final goal waypoint.
    if pending_exit_actions:
        grouped_remaining: Dict[int, List[TargetActionCommand]] = {}
        for pending_target, pending_action in pending_exit_actions:
            grouped_remaining.setdefault(pending_target, []).append(pending_action)
        for pending_target, remaining_actions in sorted(grouped_remaining.items()):
            _append_command_anchor(
                items,
                config,
                label=f"ANCHOR FINAL EXIT TARGET {pending_target}",
                phase=f"mission_end_exit_target_{pending_target}",
                target_number=pending_target,
            )
            for action in sorted(remaining_actions, key=lambda value: value.sequence_order):
                _append_action_command(
                    items=items,
                    action_records=action_records,
                    action=action,
                    target_number=pending_target,
                    phase=f"mission_end_exit_target_{pending_target}",
                    config=config,
                )

    if current_risk_state:
        _append_command_anchor(
            items,
            config,
            label="ANCHOR FINAL SPEED",
            phase="mission_end",
        )
        _append_mission_item(
            items,
            current=0,
            frame=MAV_FRAME_MISSION,
            command=MAV_CMD_DO_CHANGE_SPEED,
            param2=float(config.normal_speed_mps),
            label="FINAL NORMAL SPEED",
            phase="mission_end",
        )

    if len(items) > config.max_mission_items:
        raise RuntimeError(
            f"Generated mission has {len(items)} items, exceeding max_mission_items="
            f"{config.max_mission_items}. Increase mission_waypoint_spacing_m or "
            "raise the verified mission-item limit for the installed firmware/hardware."
        )
    return items, action_records, selected_indices


def write_ardurover_mission_outputs(
    items: Sequence[MissionItem],
    action_records: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    annotation_map: AnnotationMap,
    config: PlannerConfig,
    summary: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    mission_path = output_dir / "ardurover_mission.waypoints"
    mission_lines = ["QGC WPL 110"] + [item.to_wpl_line() for item in items]
    mission_path.write_text("\n".join(mission_lines) + "\n", encoding="utf-8")

    item_rows = [asdict(item) for item in items]
    mission_items_csv = output_dir / "ardurover_mission_items.csv"
    pd.DataFrame(item_rows).to_csv(
        mission_items_csv,
        index=False,
        encoding="utf-8-sig",
    )

    action_plan_csv = output_dir / "target_action_execution_plan.csv"
    pd.DataFrame(action_records).to_csv(
        action_plan_csv,
        index=False,
        encoding="utf-8-sig",
    )

    target_tasks_payload = {
        str(number): {
            "target_description": task.target_description,
            "action_sequence": [asdict(action) for action in task.action_sequence],
        }
        for number, task in sorted(annotation_map.target_tasks.items())
    }
    mission_json_path = output_dir / "ugv_executable_mission.json"
    mission_payload = {
        "format_version": "V4",
        "autopilot": "ArduPilot ArduRover",
        "mission_plain_text_format": "QGC WPL 110",
        "coordinate_reference": {
            "planning_input_crs": config.input_crs,
            "mission_output_crs": config.output_crs,
            "mav_frame": config.mission_frame,
            "mission_altitude_m": config.mission_altitude_m,
        },
        "mission_summary": {
            "mission_item_count": len(items),
            "selected_path_sample_count": len(selected_indices),
            "target_visit_order": summary["target_visit_order"],
            "estimated_travel_time_s": summary["estimated_travel_time_s"],
            "estimated_action_time_s": sum(
                float(config.action_bindings[action["action_id"]].get("duration_s", 0.0))
                + float(config.action_bindings[action["action_id"]].get("restore_duration_s", 0.0))
                for action in action_records
            ),
            "command_anchor_delay_s": config.mission_command_anchor_delay_s,
        },
        "target_tasks": target_tasks_payload,
        "action_bindings": config.action_bindings,
        "trigger_translation": {
            "on_zone_entry": "Anchored after the target waypoint is reached.",
            "within_zone": "Anchored and executed at the target service waypoint.",
            "on_zone_exit": "Deferred until a later waypoint clears the target footprint and configured exit clearance.",
        },
        "mission_items": item_rows,
        "preflight_requirements": [
            "Verify every servo/relay channel and mechanical endpoint on the real vehicle.",
            "For DO_SET_SERVO, configure the selected SERVOx_FUNCTION so ArduPilot allows mission control of that output.",
            "Load the .waypoints file in Mission Planner and inspect every command before writing it to the Pixhawk.",
            "Test the mission with wheels/tracks lifted or in a secured low-risk area before field deployment.",
            "Tune ArduRover waypoint radius, steering, speed, and turn parameters so the vehicle follows the planned path without unsafe corner cutting.",
        ],
    }
    mission_json_path.write_text(
        json.dumps(mission_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ardurover_mission_file": str(mission_path.resolve()),
        "mission_items_csv": str(mission_items_csv.resolve()),
        "action_plan_csv": str(action_plan_csv.resolve()),
        "executable_mission_json": str(mission_json_path.resolve()),
    }


def plot_result(
    annotation_map: AnnotationMap,
    config: PlannerConfig,
    samples: Sequence[SearchSample],
    obstacle_inflated,
    summary: Dict[str, object],
    target_order_numbers: Sequence[int],
    output_path: Path,
    show_plot: bool,
) -> None:
    """
    Export two figures:

    1. planning_result_paper.png
       No title and no statistics box, intended for papers.

    2. planning_result_annotated.png
       Complete title and a statistics box below the axes.

    planning_result.png is retained as a copy of the annotated figure.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle) < 5:
        cycle = [None] * 5

    target_color = cycle[0] if cycle[0] is not None else "tab:blue"
    obstacle_color = cycle[1] if cycle[1] is not None else "tab:orange"
    risk_color = cycle[2] if cycle[2] is not None else "tab:green"
    path_color = cycle[3] if cycle[3] is not None else "tab:red"
    vehicle_color = "black"

    def add_polygons(
        ax,
        polygons: Dict[int, Polygon],
        label: str,
        zone_color: str,
        alpha: float,
        hatch: Optional[str] = None,
    ) -> None:
        for item_index, (number, polygon) in enumerate(sorted(polygons.items())):
            coordinates = np.asarray(polygon.exterior.coords)
            patch = MplPolygon(
                coordinates,
                closed=True,
                facecolor=zone_color,
                edgecolor=zone_color,
                alpha=alpha,
                hatch=hatch,
                label=label if item_index == 0 else None,
            )
            ax.add_patch(patch)

            if label == "Target zone":
                centroid = polygon.centroid
                ax.text(
                    centroid.x,
                    centroid.y,
                    str(number),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.10",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.60,
                    },
                    zorder=9,
                )

    def draw_map(ax) -> None:
        add_polygons(
            ax,
            annotation_map.target_zones,
            "Target zone",
            target_color,
            0.35,
        )
        add_polygons(
            ax,
            annotation_map.obstacle_zones,
            "Obstacle zone",
            obstacle_color,
            0.50,
            hatch="//",
        )
        add_polygons(
            ax,
            annotation_map.risk_zones,
            "Risk zone",
            risk_color,
            0.28,
            hatch="..",
        )

        if not obstacle_inflated.is_empty:
            inflated_geometries = (
                list(obstacle_inflated.geoms)
                if obstacle_inflated.geom_type == "MultiPolygon"
                else [obstacle_inflated]
            )
            for index, polygon in enumerate(inflated_geometries):
                coordinates = np.asarray(polygon.exterior.coords)
                ax.plot(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    linestyle="--",
                    linewidth=1.2,
                    color=obstacle_color,
                    label=(
                        f"Inflated obstacle boundary "
                        f"({config.vehicle_envelope_radius_m:.3f} m)"
                        if index == 0
                        else None
                    ),
                    zorder=3,
                )

        path_x = [sample.x for sample in samples]
        path_y = [sample.y for sample in samples]
        ax.plot(
            path_x,
            path_y,
            linewidth=2.2,
            color=path_color,
            label="Optimized path",
            zorder=7,
        )

        arrow_count = min(16, max(1, len(samples) // 8))
        arrow_indices = np.linspace(
            1,
            len(samples) - 1,
            arrow_count,
            dtype=int,
        )
        for index in arrow_indices:
            sample = samples[index]
            scale = 0.38
            ax.arrow(
                sample.x,
                sample.y,
                scale * math.cos(sample.heading),
                scale * math.sin(sample.heading),
                width=0.012,
                head_width=0.16,
                length_includes_head=True,
                alpha=0.65,
                zorder=8,
            )

        if config.footprint_count > 0:
            indices = np.linspace(
                0,
                len(samples) - 1,
                min(config.footprint_count, len(samples)),
                dtype=int,
            )
            for footprint_index, sample_index in enumerate(indices):
                sample = samples[int(sample_index)]
                vehicle = make_vehicle_polygon(
                    sample.x,
                    sample.y,
                    sample.heading,
                    config,
                )
                coordinates = np.asarray(vehicle.exterior.coords)
                ax.plot(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    linewidth=0.9,
                    color=vehicle_color,
                    alpha=0.65,
                    label="Vehicle footprint" if footprint_index == 0 else None,
                    zorder=6,
                )

        ax.scatter(
            [annotation_map.start[0]],
            [annotation_map.start[1]],
            marker="o",
            s=85,
            label="Start",
            zorder=10,
        )
        ax.scatter(
            [annotation_map.goal[0]],
            [annotation_map.goal[1]],
            marker="*",
            s=130,
            label="Goal",
            zorder=10,
        )

        ax.set_xlabel(f"Easting − {annotation_map.origin_e:.3f} m")
        ax.set_ylabel(f"Northing − {annotation_map.origin_n:.3f} m")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        # V3 requirement: place the legend outside the plot, upper-right on the right side.
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            framealpha=0.88,
        )

    def make_information_text() -> str:
        order_text = " → ".join(str(number) for number in target_order_numbers)
        return (
            f"Target order: {order_text}\n"
            f"Length: {summary['total_path_length_m']:.2f} m\n"
            f"Risk length: {summary['risk_zone_length_m']:.2f} m\n"
            f"Estimated time: {summary['estimated_travel_time_s']:.2f} s\n"
            f"Planning runtime: {summary['planning_runtime_s']:.3f} s\n"
            f"Goal error: {summary['final_goal_position_error_m']:.3f} m"
        )

    paper_path = output_path.with_name("planning_result_paper.png")
    annotated_path = output_path.with_name("planning_result_annotated.png")

    # Figure 1: publication-ready figure without title or information box.
    paper_fig, paper_ax = plt.subplots(figsize=(12, 8))
    draw_map(paper_ax)
    paper_fig.subplots_adjust(
        left=0.085,
        right=0.80,
        bottom=0.10,
        top=0.985,
    )
    paper_fig.savefig(
        paper_path,
        dpi=config.figure_dpi,
        bbox_inches="tight",
    )
    plt.close(paper_fig)

    # Figure 2: annotated result. The information box is below the axes.
    annotated_fig, annotated_ax = plt.subplots(figsize=(12, 8))
    draw_map(annotated_ax)
    annotated_fig.suptitle(
        "Forward-only UGV route planning result",
        fontsize=15,
        y=0.985,
    )
    annotated_fig.text(
        0.025,
        0.020,
        make_information_text(),
        ha="left",
        va="bottom",
        fontsize=10.5,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "none",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )
    annotated_fig.subplots_adjust(
        left=0.085,
        right=0.80,
        bottom=0.205,
        top=0.925,
    )
    annotated_fig.savefig(
        annotated_path,
        dpi=config.figure_dpi,
        bbox_inches="tight",
    )

    # Backward-compatible filename.
    shutil.copyfile(annotated_path, output_path)

    if show_plot:
        plt.show()
    else:
        plt.close(annotated_fig)

def write_text_report(
    summary: Dict[str, object],
    output_path: Path,
) -> None:
    order = " -> ".join(str(value) for value in summary["target_visit_order"])
    lines = [
        "UGV PATH PLANNING REPORT",
        "=" * 60,
        f"Status: {summary['status']}",
        f"Input CSV: {summary['input_csv']}",
        f"Order algorithm: {summary['target_order_algorithm']}",
        f"Target visit order: {order}",
        f"Targets covered: {summary['covered_target_numbers']}",
        f"Missed targets: {summary['missed_target_numbers']}",
        "",
        f"Path length: {summary['total_path_length_m']:.6f} m",
        f"Normal-zone length: {summary['normal_zone_length_m']:.6f} m",
        f"Risk-zone length: {summary['risk_zone_length_m']:.6f} m",
        f"Weighted equivalent length: "
        f"{summary['weighted_equivalent_length_m']:.6f} m",
        f"Estimated travel time: "
        f"{summary['estimated_travel_time_s']:.6f} s",
        f"Estimated travel time: "
        f"{summary['estimated_travel_time_min']:.6f} min",
        "",
        f"Initial heading: {summary['initial_heading_deg']:.6f} deg",
        f"Final heading: {summary['final_heading_deg']:.6f} deg",
        f"Commanded minimum turning radius: "
        f"{summary['commanded_minimum_turning_radius_m']:.6f} m",
        f"Minimum observed turning radius: "
        f"{summary['minimum_observed_turning_radius_m']} m",
        f"Final goal position error: "
        f"{summary['final_goal_position_error_m']:.6f} m",
        f"Exact vehicle collision free: "
        f"{summary['exact_vehicle_collision_free']}",
        f"All targets covered: {summary['all_targets_covered']}",
        "",
        f"Pairwise order-cost runtime: "
        f"{summary['pairwise_cost_runtime_s']:.6f} s",
        f"Curvature-constrained route runtime: "
        f"{summary['route_search_runtime_s']:.6f} s",
        f"Total planning runtime: "
        f"{summary['planning_runtime_s']:.6f} s",
        f"Hybrid A* total expansions: "
        f"{summary['hybrid_astar_total_expansions']}",
        f"Path sample count: {summary['path_sample_count']}",
        "",
        f"ArduRover mission items: {summary.get('ardurover_mission_item_count', 0)}",
        f"Target action commands: {summary.get('target_action_command_count', 0)}",
        f"Mission output files: {summary.get('mission_output_paths', {})}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_planner(
    input_csv: Path,
    config_path: Path,
    output_dir: Path,
    action_csv: Optional[Path] = None,
    force_no_show: bool = False,
) -> Dict[str, object]:
    total_start = time.perf_counter()

    input_csv = resolve_user_path(input_csv)
    config_path = resolve_user_path(config_path)
    output_dir = resolve_user_path(output_dir)
    action_csv = resolve_user_path(action_csv) if action_csv is not None else None

    config = load_config(config_path)
    annotation_map = load_annotations(input_csv, action_csv_path=action_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact inputs used in this run.
    shutil.copy2(input_csv, output_dir / "input_annotations.csv")
    shutil.copy2(config_path, output_dir / "config_used.json")
    if action_csv is not None:
        shutil.copy2(action_csv, output_dir / "input_target_actions.csv")

    obstacle_union = (
        unary_union(list(annotation_map.obstacle_zones.values()))
        if annotation_map.obstacle_zones
        else Polygon()
    )
    risk_union = (
        unary_union(list(annotation_map.risk_zones.values()))
        if annotation_map.risk_zones
        else Polygon()
    )

    obstacle_inflated = obstacle_union.buffer(
        config.vehicle_envelope_radius_m,
        join_style=2,
    )
    # Any part of the conservative vehicle envelope entering a risk zone
    # triggers the reduced risk-zone speed.
    risk_inflated = risk_union.buffer(
        config.vehicle_envelope_radius_m,
        join_style=2,
    )

    pairwise_start = time.perf_counter()
    grid_map = GridCostMap(
        annotation_map,
        config,
        obstacle_inflated,
        risk_inflated,
    )
    task_points, cost_matrix = calculate_pairwise_costs(
        annotation_map,
        grid_map,
        config,
    )
    pairwise_runtime = time.perf_counter() - pairwise_start

    target_numbers = sorted(annotation_map.target_zones)
    node_order, order_algorithm = choose_target_order(
        cost_matrix,
        len(target_numbers),
        config.exact_order_max_targets,
    )
    ordered_target_node_indices = node_order[1:-1]
    ordered_target_numbers = [
        target_numbers[node_index - 1]
        for node_index in ordered_target_node_indices
    ]

    cost_labels = (
        ["start"]
        + [f"target_{number}" for number in target_numbers]
        + ["goal"]
    )
    pd.DataFrame(
        cost_matrix,
        index=cost_labels,
        columns=cost_labels,
    ).to_csv(
        output_dir / "pairwise_weighted_cost_matrix.csv",
        encoding="utf-8-sig",
    )

    route_start = time.perf_counter()
    hybrid_planner = ForwardHybridAStar(
        annotation_map,
        config,
        obstacle_inflated,
        risk_inflated,
    )

    current_xy = annotation_map.start
    current_heading: Optional[float] = None
    all_samples: List[SearchSample] = []
    leg_summaries: List[Dict[str, object]] = []
    total_expansions = 0

    for target_number in ordered_target_numbers:
        phase = f"to_target_{target_number}"
        result = hybrid_planner.search(
            start_xy=current_xy,
            start_heading_rad=current_heading,
            goal_polygon=annotation_map.target_zones[target_number],
            phase=phase,
        )
        result.samples[-1].target_reached = str(target_number)

        if all_samples:
            all_samples.extend(result.samples[1:])
        else:
            all_samples.extend(result.samples)

        current_xy = (result.samples[-1].x, result.samples[-1].y)
        current_heading = result.terminal_heading_rad
        total_expansions += result.expansions
        leg_summaries.append(
            {
                "phase": phase,
                "weighted_length_m": result.weighted_length_m,
                "expansions": result.expansions,
                "terminal_heading_deg": (
                    math.degrees(result.terminal_heading_rad) % 360.0
                ),
            }
        )

    final_result = hybrid_planner.search(
        start_xy=current_xy,
        start_heading_rad=current_heading,
        goal_point=annotation_map.goal,
        phase="to_goal",
    )
    final_result.samples[-1].target_reached = "goal"
    if all_samples:
        all_samples.extend(final_result.samples[1:])
    else:
        all_samples.extend(final_result.samples)
    total_expansions += final_result.expansions
    leg_summaries.append(
        {
            "phase": "to_goal",
            "weighted_length_m": final_result.weighted_length_m,
            "expansions": final_result.expansions,
            "terminal_heading_deg": (
                math.degrees(final_result.terminal_heading_rad) % 360.0
            ),
        }
    )
    route_runtime = time.perf_counter() - route_start

    measurements = validate_and_measure(
        all_samples,
        annotation_map,
        config,
    )
    total_runtime = time.perf_counter() - total_start

    status = "success"
    if (
        not measurements["all_targets_covered"]
        or not measurements["exact_vehicle_collision_free"]
        or measurements["final_goal_position_error_m"]
        > config.goal_tolerance_m + 1e-9
    ):
        status = "validation_failed"

    summary: Dict[str, object] = {
        "status": status,
        "input_csv": str(input_csv.resolve()),
        "config_file": str(config_path.resolve()),
        "target_count": len(target_numbers),
        "obstacle_zone_count": len(annotation_map.obstacle_zones),
        "risk_zone_count": len(annotation_map.risk_zones),
        "target_order_algorithm": order_algorithm,
        "target_visit_order": ordered_target_numbers,
        "target_tasks": {
            str(number): {
                "target_description": task.target_description,
                "action_sequence": [asdict(action) for action in task.action_sequence],
            }
            for number, task in sorted(annotation_map.target_tasks.items())
        },
        "pairwise_cost_runtime_s": pairwise_runtime,
        "route_search_runtime_s": route_runtime,
        "planning_runtime_s": total_runtime,
        "hybrid_astar_total_expansions": total_expansions,
        "leg_summaries": leg_summaries,
        "vehicle_parameters": asdict(config),
        **measurements,
    }

    path_frame = samples_to_dataframe(
        all_samples,
        annotation_map,
        config,
    )
    path_frame.to_csv(
        output_dir / "optimized_path.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mission_output_paths: Dict[str, str] = {}
    if config.emit_ardurover_mission:
        mission_items, action_records, selected_indices = build_ardurover_mission(
            samples=all_samples,
            annotation_map=annotation_map,
            config=config,
        )
        mission_output_paths = write_ardurover_mission_outputs(
            items=mission_items,
            action_records=action_records,
            selected_indices=selected_indices,
            annotation_map=annotation_map,
            config=config,
            summary=summary,
            output_dir=output_dir,
        )
        summary["ardurover_mission_item_count"] = len(mission_items)
        summary["target_action_command_count"] = len(action_records)
        summary["mission_output_paths"] = mission_output_paths

    (output_dir / "planning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_text_report(
        summary,
        output_dir / "planning_report.txt",
    )

    plot_result(
        annotation_map=annotation_map,
        config=config,
        samples=all_samples,
        obstacle_inflated=obstacle_inflated,
        summary=summary,
        target_order_numbers=ordered_target_numbers,
        output_path=output_dir / "planning_result.png",
        show_plot=config.show_plot and not force_no_show,
    )

    print()
    print("=" * 66)
    print("UGV PATH PLANNING COMPLETED")
    print("=" * 66)
    print(f"Status                  : {summary['status']}")
    print(f"Target visit order      : {ordered_target_numbers}")
    print(
        f"Path length             : "
        f"{summary['total_path_length_m']:.3f} m"
    )
    print(
        f"Risk-zone length        : "
        f"{summary['risk_zone_length_m']:.3f} m"
    )
    print(
        f"Estimated travel time   : "
        f"{summary['estimated_travel_time_s']:.3f} s"
    )
    print(
        f"Planning runtime        : "
        f"{summary['planning_runtime_s']:.3f} s"
    )
    print(
        f"Initial / final heading : "
        f"{summary['initial_heading_deg']:.2f} / "
        f"{summary['final_heading_deg']:.2f} deg"
    )
    print(
        f"Final goal error        : "
        f"{summary['final_goal_position_error_m']:.3f} m"
    )
    print(
        f"Targets covered         : "
        f"{summary['covered_target_numbers']}"
    )
    print(
        f"Collision free          : "
        f"{summary['exact_vehicle_collision_free']}"
    )
    print(f"Output directory        : {output_dir.resolve()}")
    if mission_output_paths:
        print(f"ArduRover mission       : {mission_output_paths['ardurover_mission_file']}")
        print(f"Action command count    : {summary['target_action_command_count']}")
    print("=" * 66)

    if status != "success":
        raise RuntimeError(
            "A path was generated, but final validation failed. "
            "Review planning_summary.json and planning_result.png."
        )

    return summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V4 UGV route planner with ArduRover mission and action export. "
            "Edit INPUT_CSV_PATH, CONFIG_JSON_PATH and OUTPUT_ROOT_DIR "
            "at the top of this file, or use optional command-line overrides."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV path override.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config path override.",
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=None,
        help=(
            "Optional target-action CSV. It may use target_number, number, or "
            "zone_index plus target_description and action_sequence columns."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root override. "
            "A timestamped subfolder is created automatically."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening the Matplotlib window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_csv = args.input if args.input is not None else INPUT_CSV_PATH
    config_path = args.config if args.config is not None else CONFIG_JSON_PATH
    action_csv = args.actions if args.actions is not None else ACTION_CSV_PATH
    output_root = (
        args.output_root
        if args.output_root is not None
        else OUTPUT_ROOT_DIR
    )

    try:
        output_dir = create_timestamped_output_dir(output_root)

        print("=" * 66)
        print("UGV PATH PLANNER V4 — ARDUROVER ACTION MISSION")
        print("=" * 66)
        print(f"Input CSV       : {resolve_user_path(input_csv)}")
        print(f"Config JSON     : {resolve_user_path(config_path)}")
        print(f"Action CSV      : {resolve_user_path(action_csv) if action_csv is not None else 'embedded in annotation CSV'}")
        print(f"Output folder   : {output_dir}")
        print("=" * 66)

        run_planner(
            input_csv=input_csv,
            config_path=config_path,
            output_dir=output_dir,
            action_csv=action_csv,
            force_no_show=args.no_show,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
