# UAV–VLM–UGV Task Planning

Core implementation for a UAV–VLM–UGV workflow that converts aerial imagery into task-oriented semantic zones and then generates a geometry- and risk-aware UGV route.

## Repository structure

```text
UAV-VLM-UGV-Task-Planning/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── vlm_based_task_reasoning/
│   ├── vlm_task_reasoning.py
│   ├── vlm_to_metric_coordinates.py
│   └── corner_coordinates_template.json
└── hierarchical_path_planning/
    ├── hierarchical_path_planning.py
    ├── example_annotations.csv
    └── planner_config.json
```

## Workflow

```text
UAV orthographic / bird's-eye image
        ↓
VLM-based task reasoning
        ↓
Target / obstacle / risk polygons in normalized image coordinates [0, 999]
        ↓
Four-corner image-to-metric coordinate transformation
        ↓
Planner-ready metric coordinates (X/E, Y/N, Z/U)
        ↓
Hierarchical path planning
        ↓
Optimized UGV path and optional ArduRover mission export
```

## Requirements

Python 3.10 is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 1. VLM-based task reasoning

The main program is:

```text
vlm_based_task_reasoning/vlm_task_reasoning.py
```

It analyzes UAV bird's-eye images and returns three polygon classes:

- `target_zones`
- `obstacle_zones`
- `risk_zones`

Polygon vertices are reported as integer normalized image coordinates in `[0, 999]`, with the origin at the top-left of each image.

### API credentials

Do **not** store API keys in the repository. The public version reads credentials from environment variables. A single key can be supplied with:

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
python vlm_based_task_reasoning/vlm_task_reasoning.py
```

Linux/macOS:

```bash
export OPENAI_API_KEY="YOUR_KEY"
python vlm_based_task_reasoning/vlm_task_reasoning.py
```

Up to five keys can also be supplied as `OPENAI_API_KEY_1` through `OPENAI_API_KEY_5`.

### Main VLM output

The aggregated zone CSV contains:

```text
image_filename,zone_type,zone_index,point_count,normalized_coordinates
```

The program also saves per-image JSON outputs, processing status information, and overlay visualizations.

## 2. Image-to-metric coordinate transformation

The interface program is:

```text
vlm_based_task_reasoning/vlm_to_metric_coordinates.py
```

It converts the VLM polygons from normalized image coordinates to the metric planning frame using the actual XYZ coordinates of the four image corners:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

For a normalized point `(x_norm, y_norm)`, define:

```text
u = x_norm / 999
v = y_norm / 999
```

The metric point is obtained by bilinear interpolation:

```text
P(u,v) = (1-u)(1-v) P_TL
       + u(1-v)     P_TR
       + uv         P_BR
       + (1-u)v     P_BL
```

Run interactively:

```bash
python vlm_based_task_reasoning/vlm_to_metric_coordinates.py
```

or provide inputs explicitly:

```bash
python vlm_based_task_reasoning/vlm_to_metric_coordinates.py \
  --image path/to/image.jpg \
  --vlm-csv path/to/zone_results.csv \
  --corners vlm_based_task_reasoning/corner_coordinates_template.json \
  --start "100,900" \
  --goal "900,100" \
  --output path/to/planner_annotations.csv \
  --crs EPSG:32650
```

The generated planner CSV uses:

```text
type,number,X/E,Y/N,Z/U
```

The converter can preserve optional `target_description` and `action_sequence` columns when they exist in the VLM CSV.

> **Assumption:** the four-corner mapping is intended for orthographic/georeferenced bird's-eye imagery over approximately planar terrain. Strong perspective distortion or large terrain relief requires a more complete camera/geospatial model.

## 3. Hierarchical path planning

The planner is:

```text
hierarchical_path_planning/hierarchical_path_planning.py
```

The supplied implementation includes:

1. vehicle-envelope-based obstacle inflation;
2. risk-weighted grid A* for the pairwise task cost matrix;
3. Held–Karp dynamic programming for exact target ordering when the number of targets is small;
4. nearest-neighbor initialization plus 2-opt for larger target sets;
5. forward-only Hybrid A* with minimum-turning-radius and obstacle constraints;
6. path validation and quantitative reporting;
7. optional ArduRover/MAVLink mission generation and task-action export.

The example input files are:

```text
hierarchical_path_planning/example_annotations.csv
hierarchical_path_planning/planner_config.json
```

Run:

```bash
cd hierarchical_path_planning
python hierarchical_path_planning.py --no-show
```

or:

```bash
python hierarchical_path_planning/hierarchical_path_planning.py \
  --input hierarchical_path_planning/example_annotations.csv \
  --config hierarchical_path_planning/planner_config.json \
  --no-show
```

### Example planner configuration

The supplied `planner_config.json` contains the vehicle dimensions, safety margin, minimum turning radius, normal speed, risk-zone speed ratio, grid resolution, Hybrid A* discretization, and plotting parameters used by the example.

Because the supplied example annotation CSV contains geometry only and does not contain target action sequences, the public example configuration explicitly sets:

```json
"require_target_actions": false,
"emit_ardurover_mission": false
```

The code still contains the ArduRover mission-export implementation. To enable executable action missions, provide target action assignments and verified hardware bindings, then enable the two mission-related options in the configuration.

## Planner input format

The path planner expects exactly one `start` row and one `goal` row. Polygon zones are represented by multiple rows sharing the same `type` and `number`.

Required columns:

```text
type,number,X/E,Y/N,Z/U
```

Supported types:

```text
start
goal
target_zones
obstacle_zones
risk_zones
```

## Coordinate reference systems

The path-planning implementation defaults to `EPSG:32650` for the metric planning input and `EPSG:4326` for optional ArduRover mission export. If another projected CRS is used, update the configuration accordingly.

## Reproducibility notes

- API credentials are intentionally excluded.
- Generated result folders are excluded through `.gitignore`.
- The VLM output and path-planner input are not the same coordinate representation; the coordinate-transformation stage is required between them.
- Check the four image-corner coordinates and CRS before metric conversion.
- Check vehicle dimensions, safety margin, minimum turning radius, and risk-speed parameters before running the planner on another platform.
- Servo/relay mappings must be calibrated and verified on the actual vehicle before enabling executable ArduRover action missions.

## License

This repository is released under the MIT License. See `LICENSE`.

## Citation

If you use this repository in academic work, please cite the associated UAV–VLM–UGV paper. A DOI/repository citation can be added here after the paper and archived software release are finalized.
