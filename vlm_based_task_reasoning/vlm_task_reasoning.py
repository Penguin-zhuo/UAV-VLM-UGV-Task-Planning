# -*- coding: utf-8 -*-
"""
VLM-Based Task Reasoning
Python 3.10

V5 说明：
1. 使用 Standard Responses API；
2. 支持最多 5 个 API Key 并行处理；
3. 每个 API Key 同一时刻只处理 1 张图；
4. 某个 worker 处理某张图失败时，会自动切换到其他尚未尝试过的 worker；
5. 在提示词中明确写入每张图自己的宽度和高度；
6. 明确规定归一化坐标定义与公式；
7. 输出 CSV、逐图 JSON、状态 CSV 和 overlay 叠加图。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm


OPENAI_API_KEYS = [
]

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_DETAIL = "original"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 16000
DEFAULT_MAX_RETRIES_PER_API = 2
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.2
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_API_KEYS = 5

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

RESULT_COLUMNS = [
    "image_filename",
    "zone_type",
    "zone_index",
    "point_count",
    "normalized_coordinates",
]

STATUS_COLUMNS = [
    "image_filename",
    "status",
    "model",
    "response_id",
    "worker_index",
    "api_key_label",
    "attempted_workers",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "elapsed_seconds",
    "image_width_px",
    "image_height_px",
    "error",
]

ZONE_TYPE_ORDER = ("target_zones", "obstacle_zones", "risk_zones")

ZONE_COLORS = {
    "target_zones": (0, 0, 255, 110),
    "obstacle_zones": (255, 0, 0, 110),
    "risk_zones": (255, 140, 0, 110),
}

ZONE_OUTLINE_COLORS = {
    "target_zones": (0, 0, 255, 220),
    "obstacle_zones": (255, 0, 0, 220),
    "risk_zones": (255, 140, 0, 220),
}

SYSTEM_PROMPT_EN = """
You are an aerial-image analysis assistant for a beach cleaning UGV (unmanned ground vehicle).

Your task is to analyze a UAV beach orthographic / bird's-eye image and identify three kinds of polygonal zones:

1. target_zones
Areas that the autonomous beach-cleaning vehicle should reach and clean, such as:
- beach litter, including bottles, cans, paper, plastic bags, wrappers, and similar debris;
- clusters of small scattered trash;
- other clearly visible cleanable waste suitable for a beach-cleaning vehicle.

Do NOT mark beach chairs, beach tables, wooden posts, people, or vehicles as target_zones.

2. obstacle_zones
Areas that are not safely traversable or should be treated as obstacles, such as:
- seawater, standing water, active waves, or water-covered areas;
- very large rocks, reefs, wooden posts, fences;
- beach chairs, beach tables, vehicles, structures, or other hard obstacles;
- other clearly non-traversable regions.

3. risk_zones
Areas that may still be traversable but with high risk and should usually be slowed down or avoided if possible, such as:
- sand pits or noticeable depressions;
- wet, soft, or low-support sand;

If an area is clearly non-traversable, classify it as obstacle_zones instead of risk_zones.

Output requirements:
- Return ONLY structured data that follows the requested schema.
- Each disconnected region must be returned separately.
- Each region must be represented by a polygon describing its outer boundary.
- Polygon points must be ordered consistently around the boundary (clockwise or counterclockwise).
- Each polygon must contain at least 3 distinct vertices.
- Do not repeat the first point at the end.
- Keep polygons reasonably simple while preserving shape; usually 3 to 15 points.
- All coordinates must be integer normalized coordinates in the range [0, 999].
- Coordinate origin is the top-left corner of THIS image.
- x increases from left to right.
- y increases from top to bottom.
- Do not mark ordinary dry flat traversable sand as any zone.
- Do not mistake shadows alone for obstacles or risk zones.
- Only mark zones that are reasonably visible and sufficiently certain.
- Any of the three zone lists may be empty.
""".strip()


def build_user_prompt(image_width_px: int, image_height_px: int) -> str:
    return f"""
Analyze this specific UAV beach image.

This image has:
- image_width_px = {image_width_px}
- image_height_px = {image_height_px}

Return the following zone lists:
- target_zones
- obstacle_zones
- risk_zones

IMPORTANT NORMALIZATION RULES:
For THIS image only, normalize each polygon vertex using the actual width and height of this image.

Use:
- x_norm = round( x_pixel / max(image_width_px - 1, 1) * 999 )
- y_norm = round( y_pixel / max(image_height_px - 1, 1) * 999 )

This means normalization must always be computed from THIS image's own width and height.
Do NOT assume the image is square.
Do NOT assume a fixed aspect ratio.
Do NOT reuse coordinates from another image size.

Return only integer normalized coordinates in [0, 999].
The top-left corner is (0, 0).
The bottom-right corner is (999, 999).
x grows to the right.
y grows downward.

Return only the structured result.
""".strip()



def build_user_prompt(image_width_px: int, image_height_px: int) -> str:
    return f"""
Analyze this specific UAV image of a gravel, sandy-gravel, paved-brick, or interlocking-brick ground surface.

This image has:
- image_width_px = {image_width_px}
- image_height_px = {image_height_px}

Identify and return:
- target_zones
- obstacle_zones
- risk_zones

Remember:
- All visible children's toys must be classified as target_zones.
- Do not determine whether a toy is abandoned or currently in use.
- Raised curbs, road shoulders, steps, standing water, and other non-traversable areas must be classified as obstacle_zones.
- Manhole covers and other traversable but hazardous surface features must be classified as risk_zones.

IMPORTANT NORMALIZATION RULES:

For THIS image only, normalize every polygon vertex using the actual width and height of this image.

Use:
- x_norm = round(x_pixel / max(image_width_px - 1, 1) * 999)
- y_norm = round(y_pixel / max(image_height_px - 1, 1) * 999)

Normalization must always be computed using THIS image's actual width and height.

Do NOT assume that:
- the image is square;
- the image has a fixed aspect ratio;
- all input images have the same dimensions;
- coordinates from another image size can be reused.

Return only integer normalized coordinates in the range [0, 999].

Coordinate definition:
- top-left corner: (0, 0);
- bottom-right corner: (999, 999);
- x increases from left to right;
- y increases from top to bottom.

Return only the structured result.
""".strip()



def build_user_prompt(image_width_px: int, image_height_px: int) -> str:
    return f"""
Analyze this specific UAV image of a gravel, sandy-gravel, paved-brick, or interlocking-brick ground surface.

This image has:
- image_width_px = {image_width_px}
- image_height_px = {image_height_px}

Return the following zone lists:
- target_zones
- obstacle_zones
- risk_zones

IMPORTANT NORMALIZATION RULES:
For THIS image only, normalize each polygon vertex using the actual width and height of this image.

Use:
- x_norm = round(x_pixel / max(image_width_px - 1, 1) * 999)
- y_norm = round(y_pixel / max(image_height_px - 1, 1) * 999)

Normalization must always be computed using THIS image's actual width and height.

Do NOT assume that:
- the image is square;
- the image has a fixed aspect ratio;
- all input images have the same dimensions;
- coordinates from another image size can be reused.

Return only integer normalized coordinates in the range [0, 999].

Coordinate definition:
- top-left corner: (0, 0);
- bottom-right corner: (999, 999);
- x increases from left to right;
- y increases from top to bottom.

Return only the structured result.
""".strip()



class NormalizedPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0, le=999)
    y: int = Field(ge=0, le=999)


class PolygonZone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: list[NormalizedPoint]


class BeachZoneAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_zones: list[PolygonZone]
    obstacle_zones: list[PolygonZone]
    risk_zones: list[PolygonZone]


@dataclass
class ImageJob:
    image_path: Path
    image_filename: str
    image_width_px: int
    image_height_px: int
    tried_workers: set[int] = field(default_factory=set)


@dataclass
class WorkerConfig:
    worker_index: int
    api_key: str
    client: OpenAI
    api_key_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT-5.6 Sol Standard API to analyze beach UAV images in parallel.")
    parser.add_argument("input_folder", nargs="?", help="Input image folder. If omitted, a dialog will open.")
    parser.add_argument("--output-dir", default=None, help="Output folder. If omitted, a dialog will open.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--detail", choices=["low", "high", "original", "auto"], default=DEFAULT_DETAIL)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries-per-api", type=int, default=DEFAULT_MAX_RETRIES_PER_API)
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ask_directory(title: str) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(selected) if selected else None
    except Exception as exc:
        print(f"Unable to open folder selection dialog: {exc}")
        return None


def resolve_input_folder(folder_arg: str | None) -> Path:
    if folder_arg:
        folder = Path(folder_arg).expanduser().resolve()
    else:
        selected = ask_directory("Select the input image folder")
        if selected is None:
            raise RuntimeError("No input image folder was selected.")
        folder = selected.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a folder: {folder}")
    return folder


def resolve_output_folder(output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        folder = Path(output_dir_arg).expanduser().resolve()
    else:
        selected = ask_directory("Select the output folder")
        if selected is None:
            raise RuntimeError("No output folder was selected.")
        folder = selected.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def discover_images(folder: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.glob("*")
    images = [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(images, key=lambda p: p.relative_to(folder).as_posix().lower())


def relative_image_name(image_path: Path, input_folder: Path) -> str:
    return image_path.relative_to(input_folder).as_posix()


def json_path_for_image(image_path: Path, input_folder: Path, json_root: Path) -> Path:
    relative_path = image_path.relative_to(input_folder)
    output_path = json_root / relative_path
    return output_path.with_suffix(output_path.suffix + ".json")


def overlay_path_for_image(image_path: Path, input_folder: Path, overlay_root: Path) -> Path:
    relative_path = image_path.relative_to(input_folder)
    output_path = overlay_root / relative_path
    return output_path.with_suffix(".png")


def mime_type_for_image(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    fixed = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
    mime = fixed.get(suffix) or mimetypes.guess_type(image_path.name)[0]
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"Could not determine image MIME type: {image_path}")
    return mime


def encode_image_as_data_url(image_path: Path) -> str:
    mime = mime_type_for_image(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width} x {height}")
    return width, height


def clean_polygon(zone: PolygonZone) -> PolygonZone | None:
    cleaned: list[NormalizedPoint] = []
    for point in zone.points:
        current = NormalizedPoint(x=max(0, min(999, int(point.x))), y=max(0, min(999, int(point.y))))
        if not cleaned or current != cleaned[-1]:
            cleaned.append(current)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    unique_points = {(point.x, point.y) for point in cleaned}
    if len(cleaned) < 3 or len(unique_points) < 3:
        return None
    return PolygonZone(points=cleaned)


def clean_analysis(result: BeachZoneAnalysis) -> BeachZoneAnalysis:
    cleaned_data: dict[str, list[PolygonZone]] = {}
    for zone_type in ZONE_TYPE_ORDER:
        cleaned_zones: list[PolygonZone] = []
        for zone in getattr(result, zone_type):
            cleaned = clean_polygon(zone)
            if cleaned is not None:
                cleaned_zones.append(cleaned)
        cleaned_data[zone_type] = cleaned_zones
    return BeachZoneAnalysis(**cleaned_data)


def usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def extract_usage_numbers(usage: dict[str, Any]) -> tuple[Any, Any, Any]:
    input_tokens = usage.get("input_tokens", "")
    output_tokens = usage.get("output_tokens", "")
    total_tokens = usage.get("total_tokens", "")
    if total_tokens == "" and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def save_raw_json(output_path: Path, image_filename: str, model: str, detail: str, reasoning_effort: str,
                  image_width_px: int, image_height_px: int, worker_index: int, api_key_label: str,
                  elapsed_seconds: float, result: BeachZoneAnalysis, response: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_filename": image_filename,
        "model": model,
        "detail": detail,
        "reasoning_effort": reasoning_effort,
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
        "worker_index": worker_index,
        "api_key_label": api_key_label,
        "response_id": getattr(response, "id", ""),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": usage_to_dict(response),
        "result": result.model_dump(),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def append_status_row(status_csv: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        is_new = not status_csv.exists() or status_csv.stat().st_size == 0
        encoding = "utf-8-sig" if is_new else "utf-8"
        with status_csv.open("a", encoding=encoding, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=STATUS_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in STATUS_COLUMNS})


def load_latest_status(status_csv: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not status_csv.exists():
        return latest
    with status_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_filename = row.get("image_filename", "")
            status = row.get("status", "")
            if image_filename:
                latest[image_filename] = status
    return latest


def rebuild_results_csv(json_root: Path, result_csv: Path) -> int:
    rows: list[dict[str, Any]] = []
    if json_root.exists():
        for json_path in sorted(json_root.rglob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                image_filename = payload["image_filename"]
                result = payload["result"]
                for zone_type in ZONE_TYPE_ORDER:
                    zones = result.get(zone_type, [])
                    for zone_index, zone in enumerate(zones, start=1):
                        points = zone.get("points", [])
                        coordinates = [[int(point["x"]), int(point["y"])] for point in points]
                        rows.append({
                            "image_filename": image_filename,
                            "zone_type": zone_type,
                            "zone_index": zone_index,
                            "point_count": len(coordinates),
                            "normalized_coordinates": json.dumps(coordinates, ensure_ascii=False, separators=(",", ":")),
                        })
            except Exception as exc:
                print(f"\\nWarning: failed to parse result JSON, skipped: {json_path}\\nReason: {exc}")
    temporary_csv = result_csv.with_suffix(result_csv.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(result_csv)
    return len(rows)


def save_overlay_image(image_path: Path, overlay_output_path: Path, result: BeachZoneAnalysis) -> None:
    overlay_output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        width, height = base.size

        def denormalize(point_x: int, point_y: int) -> tuple[float, float]:
            px = (point_x / 999.0) * (width - 1 if width > 1 else 0)
            py = (point_y / 999.0) * (height - 1 if height > 1 else 0)
            return px, py

        for zone_type in ZONE_TYPE_ORDER:
            fill_color = ZONE_COLORS[zone_type]
            outline_color = ZONE_OUTLINE_COLORS[zone_type]
            zones = getattr(result, zone_type)
            for zone in zones:
                polygon = [denormalize(pt.x, pt.y) for pt in zone.points]
                if len(polygon) >= 3:
                    draw.polygon(polygon, fill=fill_color, outline=outline_color)
                    draw.line(polygon + [polygon[0]], fill=outline_color, width=3)

        combined = Image.alpha_composite(base, overlay).convert("RGB")
        combined.save(overlay_output_path, format="PNG")


def is_retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    retryable_names = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
    return exc.__class__.__name__ in retryable_names


def analyze_image_with_client(client: OpenAI, image_path: Path, image_width_px: int, image_height_px: int,
                              model: str, detail: str, reasoning_effort: str,
                              max_output_tokens: int, max_retries_per_api: int) -> tuple[BeachZoneAnalysis, Any]:
    image_data_url = encode_image_as_data_url(image_path)
    user_prompt = build_user_prompt(image_width_px=image_width_px, image_height_px=image_height_px)
    last_error: Exception | None = None

    for attempt in range(1, max_retries_per_api + 1):
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                store=False,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT_EN},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": image_data_url, "detail": detail},
                        ],
                    },
                ],
                text_format=BeachZoneAnalysis,
            )
            parsed = response.output_parsed
            if parsed is None:
                output_text = getattr(response, "output_text", "")
                raise RuntimeError("API did not return a parsable structured result." + (f" Raw output: {output_text[:500]}" if output_text else ""))
            cleaned = clean_analysis(parsed)
            return cleaned, response
        except Exception as exc:
            last_error = exc
            retryable = is_retryable_exception(exc)
            if not retryable or attempt >= max_retries_per_api:
                raise
            wait_seconds = min(5.0 * (2 ** (attempt - 1)), 30.0)
            time.sleep(wait_seconds)
    raise RuntimeError(f"Analysis failed: {last_error}")


def process_one_job(worker: WorkerConfig, job: ImageJob, model: str, detail: str,
                    reasoning_effort: str, max_output_tokens: int, max_retries_per_api: int) -> dict[str, Any]:
    start_time = time.perf_counter()
    result, response = analyze_image_with_client(
        client=worker.client,
        image_path=job.image_path,
        image_width_px=job.image_width_px,
        image_height_px=job.image_height_px,
        model=model,
        detail=detail,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        max_retries_per_api=max_retries_per_api,
    )
    elapsed_seconds = time.perf_counter() - start_time
    usage = usage_to_dict(response)
    input_tokens, output_tokens, total_tokens = extract_usage_numbers(usage)
    return {
        "result": result,
        "response": response,
        "elapsed_seconds": elapsed_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def prepare_output_paths(output_root: Path, overwrite: bool) -> tuple[Path, Path, Path, Path, Path]:
    output_dir = output_root / "vlm_task_reasoning_output"
    result_csv = output_dir / "zone_results.csv"
    status_csv = output_dir / "processing_status.csv"
    raw_json_dir = output_dir / "raw_json"
    overlay_dir = output_dir / "overlay_images"
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, result_csv, status_csv, raw_json_dir, overlay_dir


def build_worker_configs() -> list[WorkerConfig]:
    cleaned_keys: list[str] = []
    invalid_placeholders = {"", "在这里填写第1个OpenAI_API_Key", "YOUR_OPENAI_API_KEY", "PASTE_YOUR_API_KEY_HERE"}

    # Preferred for public/reproducible code: read API keys from environment
    # variables so credentials are never stored in the repository.
    environment_keys = []
    primary_key = os.getenv("OPENAI_API_KEY", "").strip()
    if primary_key:
        environment_keys.append(primary_key)
    for index in range(1, DEFAULT_MAX_API_KEYS + 1):
        value = os.getenv(f"OPENAI_API_KEY_{index}", "").strip()
        if value:
            environment_keys.append(value)

    candidate_keys = environment_keys + OPENAI_API_KEYS[:DEFAULT_MAX_API_KEYS]
    for key in candidate_keys:
        key = key.strip()
        if key and key not in invalid_placeholders and key not in cleaned_keys:
            cleaned_keys.append(key)
        if len(cleaned_keys) >= DEFAULT_MAX_API_KEYS:
            break
    if not cleaned_keys:
        raise RuntimeError("No valid API key was detected. Set OPENAI_API_KEY (or OPENAI_API_KEY_1...5) in the environment, or provide a local key in OPENAI_API_KEYS without committing it.")

    workers: list[WorkerConfig] = []
    for index, key in enumerate(cleaned_keys, start=1):
        label = f"API_{index}"
        workers.append(
            WorkerConfig(
                worker_index=index,
                api_key=key,
                api_key_label=label,
                client=OpenAI(api_key=key, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=0),
            )
        )
    return workers


def pop_compatible_job(job_queue: deque[ImageJob], worker_index: int) -> ImageJob | None:
    if not job_queue:
        return None
    queue_length = len(job_queue)
    for _ in range(queue_length):
        job = job_queue.popleft()
        if worker_index not in job.tried_workers:
            return job
        job_queue.append(job)
    return None


def main() -> int:
    args = parse_args()
    try:
        input_folder = resolve_input_folder(args.input_folder)
        output_root = resolve_output_folder(args.output_dir)
        workers = build_worker_configs()
    except Exception as exc:
        print(f"Error: {exc}")
        return 2

    output_dir, result_csv, status_csv, raw_json_dir, overlay_dir = prepare_output_paths(output_root=output_root, overwrite=args.overwrite)

    images = discover_images(input_folder, recursive=args.recursive)
    if output_dir != input_folder:
        images = [image for image in images if output_dir not in image.parents]

    if not images:
        print(f"No supported images were found under: {input_folder}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return 1

    latest_status = load_latest_status(status_csv)
    pending_jobs: list[ImageJob] = []

    for image_path in images:
        image_filename = relative_image_name(image_path, input_folder)
        if latest_status.get(image_filename) == "success":
            continue
        try:
            image_width_px, image_height_px = get_image_size(image_path)
        except Exception as exc:
            print(f"Skipping image due to size read failure: {image_filename} | {exc}")
            continue
        pending_jobs.append(ImageJob(
            image_path=image_path,
            image_filename=image_filename,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        ))

    existing_rows = rebuild_results_csv(raw_json_dir, result_csv)

    print("=" * 80)
    print("VLM-Based Task Reasoning")
    print(f"Input folder: {input_folder}")
    print(f"Output root: {output_root}")
    print(f"Output folder: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Workers: {len(workers)}")
    print(f"Image detail: {args.detail}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Discovered images: {len(images)}")
    print(f"Already successful: {len(images) - len(pending_jobs)}")
    print(f"Pending images: {len(pending_jobs)}")
    print(f"Current CSV zone rows: {existing_rows}")
    print("=" * 80)

    if not pending_jobs:
        print(f"All images have already been processed. Result CSV: {result_csv}")
        return 0

    jobs = deque(pending_jobs)
    status_lock = threading.Lock()
    success_count = 0
    failed_count = 0

    free_workers = {worker.worker_index for worker in workers}
    workers_by_index = {worker.worker_index: worker for worker in workers}
    active_futures: dict[Any, tuple[int, ImageJob]] = {}
    progress_bar = tqdm(total=len(pending_jobs), desc="Analyzing images", unit="image")

    try:
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            while jobs or active_futures:
                scheduled_any = False

                for worker_index in sorted(list(free_workers)):
                    job = pop_compatible_job(jobs, worker_index)
                    if job is None:
                        continue
                    worker = workers_by_index[worker_index]
                    future = executor.submit(
                        process_one_job,
                        worker,
                        job,
                        args.model,
                        args.detail,
                        args.reasoning_effort,
                        args.max_output_tokens,
                        args.max_retries_per_api,
                    )
                    active_futures[future] = (worker_index, job)
                    free_workers.remove(worker_index)
                    scheduled_any = True

                if not active_futures and not scheduled_any and jobs:
                    while jobs:
                        job = jobs.popleft()
                        attempted = ",".join(f"API_{idx}" for idx in sorted(job.tried_workers))
                        append_status_row(status_csv, {
                            "image_filename": job.image_filename,
                            "status": "failed",
                            "model": args.model,
                            "response_id": "",
                            "worker_index": "",
                            "api_key_label": "",
                            "attempted_workers": attempted,
                            "input_tokens": "",
                            "output_tokens": "",
                            "total_tokens": "",
                            "elapsed_seconds": "",
                            "image_width_px": job.image_width_px,
                            "image_height_px": job.image_height_px,
                            "error": "This image was attempted on all available API keys but still failed.",
                        }, status_lock)
                        failed_count += 1
                        progress_bar.update(1)
                    break

                if not active_futures:
                    continue

                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    worker_index, job = active_futures.pop(future)
                    free_workers.add(worker_index)
                    worker = workers_by_index[worker_index]

                    try:
                        payload = future.result()
                        result = payload["result"]
                        response = payload["response"]
                        elapsed_seconds = payload["elapsed_seconds"]
                        input_tokens = payload["input_tokens"]
                        output_tokens = payload["output_tokens"]
                        total_tokens = payload["total_tokens"]

                        raw_json_path = json_path_for_image(job.image_path, input_folder, raw_json_dir)
                        overlay_path = overlay_path_for_image(job.image_path, input_folder, overlay_dir)

                        save_raw_json(
                            output_path=raw_json_path,
                            image_filename=job.image_filename,
                            model=args.model,
                            detail=args.detail,
                            reasoning_effort=args.reasoning_effort,
                            image_width_px=job.image_width_px,
                            image_height_px=job.image_height_px,
                            worker_index=worker.worker_index,
                            api_key_label=worker.api_key_label,
                            elapsed_seconds=elapsed_seconds,
                            result=result,
                            response=response,
                        )
                        save_overlay_image(job.image_path, overlay_path, result)

                        append_status_row(status_csv, {
                            "image_filename": job.image_filename,
                            "status": "success",
                            "model": args.model,
                            "response_id": getattr(response, "id", ""),
                            "worker_index": worker.worker_index,
                            "api_key_label": worker.api_key_label,
                            "attempted_workers": ",".join(f"API_{idx}" for idx in sorted(job.tried_workers)),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "elapsed_seconds": f"{elapsed_seconds:.3f}",
                            "image_width_px": job.image_width_px,
                            "image_height_px": job.image_height_px,
                            "error": "",
                        }, status_lock)

                        rebuild_results_csv(raw_json_dir, result_csv)
                        success_count += 1
                        progress_bar.update(1)

                    except Exception as exc:
                        job.tried_workers.add(worker_index)
                        if len(job.tried_workers) < len(workers):
                            jobs.append(job)
                            tqdm.write(f"Retry: {job.image_filename} failed on {worker.api_key_label}; it will be retried on another API. Reason: {exc}")
                        else:
                            attempted = ",".join(f"API_{idx}" for idx in sorted(job.tried_workers))
                            append_status_row(status_csv, {
                                "image_filename": job.image_filename,
                                "status": "failed",
                                "model": args.model,
                                "response_id": "",
                                "worker_index": worker.worker_index,
                                "api_key_label": worker.api_key_label,
                                "attempted_workers": attempted,
                                "input_tokens": "",
                                "output_tokens": "",
                                "total_tokens": "",
                                "elapsed_seconds": "",
                                "image_width_px": job.image_width_px,
                                "image_height_px": job.image_height_px,
                                "error": str(exc),
                            }, status_lock)
                            failed_count += 1
                            progress_bar.update(1)
                            tqdm.write(f"Failed: {job.image_filename}; all APIs were attempted. Last error: {exc}")

                    if args.request_interval > 0:
                        time.sleep(args.request_interval)

    except KeyboardInterrupt:
        print("\\nInterrupted by user. Completed results have already been saved.")
    finally:
        progress_bar.close()

    final_rows = rebuild_results_csv(raw_json_dir, result_csv)
    print("\\n" + "=" * 80)
    print("Run finished")
    print(f"Success count: {success_count}")
    print(f"Failed count: {failed_count}")
    print(f"Final zone rows: {final_rows}")
    print(f"Result CSV: {result_csv}")
    print(f"Status CSV: {status_csv}")
    print(f"Raw JSON folder: {raw_json_dir}")
    print(f"Overlay folder: {overlay_dir}")
    print("=" * 80)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
