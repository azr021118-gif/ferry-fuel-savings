from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_AIS_PARTS_DIR = BASE_DIR / "yxlan_20250701_20251130_parts"
DEFAULT_DATASET_BASE = "yxlan_20250701_20251130"


def normalize_base_name(name: str) -> str:
    value = Path(name).stem if "." in name else name
    value = re.sub(r"_parts$", "", value)
    value = re.sub(r"_part\d+$", "", value)
    value = re.sub(r"_step\d+.*$", "", value)
    return value


def dataset_base_from_input(input_path: Path) -> str:
    if input_path.is_dir():
        return normalize_base_name(input_path.name)
    return normalize_base_name(input_path.stem)


def read_ais_source(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        csv_parts = sorted(input_path.glob("*.csv"))
        if not csv_parts:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        frames = [pd.read_csv(part) for part in csv_parts]
        return pd.concat(frames, ignore_index=True)
    if input_path.is_file():
        return pd.read_csv(input_path)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def step_output_path(
    output_dir: Path, dataset_base: str, step_suffix: str, extension: str
) -> Path:
    return output_dir / f"{dataset_base}_{step_suffix}.{extension}"



# Direction used by the figure scripts. Default berth_2 -> berth_1 keeps the original chart paths;
# set YXLAN_DIRECTION=berth_1_to_berth_2 to render the reverse direction into charts/berth_1_to_berth_2/.
import os as _os

PLOT_DIRECTION = _os.environ.get("YXLAN_DIRECTION", "berth_2_to_berth_1")
PLOT_CHART_DIR = OUTPUT_DIR / "charts" / ("" if PLOT_DIRECTION == "berth_2_to_berth_1" else PLOT_DIRECTION)
PLOT_SUFFIX = "" if PLOT_DIRECTION == "berth_2_to_berth_1" else f"_{PLOT_DIRECTION}"
PLOT_CHART_DIR.mkdir(parents=True, exist_ok=True)
