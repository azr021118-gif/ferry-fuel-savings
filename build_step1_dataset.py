from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import (
    DEFAULT_AIS_PARTS_DIR,
    OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
    dataset_base_from_input,
    read_ais_source,
)

ENGINE_COUNT = 4
DEFAULT_INPUT = DEFAULT_AIS_PARTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the step-1 cleaned and standardized ship dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Source CSV file or directory containing CSV parts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cleaned outputs.",
    )
    return parser.parse_args()


def haversine_nm(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    radius_m = 6_371_000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lat2 * 0 + lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance_m = radius_m * c
    return pd.Series(distance_m / 1852.0, index=lat1.index)


def angular_diff_deg(current: pd.Series, previous: pd.Series) -> pd.Series:
    diff = (current - previous + 180.0) % 360.0 - 180.0
    return diff


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": df.isna().sum().values,
            "missing_ratio_pct": (df.isna().mean() * 100.0).round(4).values,
            "non_null_count": df.notna().sum().values,
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True])


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    numeric_columns = [col for col in df.columns if col != "time"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(
        subset=["time"], keep="last"
    )
    df = df.reset_index(drop=True)

    if "SOG" in df.columns:
        df["SOG_missing_flag"] = df["SOG"].isna()
    else:
        df["SOG_missing_flag"] = True

    for boundary_col, lower, upper in (
        ("latitude", -90.0, 90.0),
        ("longitude", -180.0, 180.0),
        ("SOG", 0.0, 60.0),
        ("COG", 0.0, 360.0),
    ):
        if boundary_col in df.columns:
            df.loc[~df[boundary_col].between(lower, upper, inclusive="both"), boundary_col] = np.nan

    engine_power_cols = [f"ME{i}_power" for i in range(1, ENGINE_COUNT + 1)]
    engine_rpm_cols = [f"ME{i}_rpm" for i in range(1, ENGINE_COUNT + 1)]
    engine_fuel_cols = [f"ME{i}_fuel" for i in range(1, ENGINE_COUNT + 1)]

    for col_group in (engine_power_cols, engine_rpm_cols, engine_fuel_cols):
        existing = [col for col in col_group if col in df.columns]
        for col in existing:
            df.loc[df[col] < 0, col] = np.nan

    df["fuel_total"] = df[engine_fuel_cols].sum(axis=1, min_count=1)
    df["power_total"] = df[engine_power_cols].sum(axis=1, min_count=1)
    df["rpm_mean"] = df[engine_rpm_cols].mean(axis=1)
    df["rpm_max"] = df[engine_rpm_cols].max(axis=1)
    df["active_engine_count"] = df[engine_rpm_cols].gt(0).sum(axis=1)

    df["timestamp_unix"] = (df["time"].astype("int64") // 10**9).astype("int64")
    df["delta_t_seconds"] = df["time"].diff().dt.total_seconds()

    prev_lat = df["latitude"].shift(1)
    prev_lon = df["longitude"].shift(1)
    df["step_distance_nm"] = haversine_nm(prev_lat, prev_lon, df["latitude"], df["longitude"])
    df.loc[df["delta_t_seconds"].isna() | (df["delta_t_seconds"] <= 0), "step_distance_nm"] = np.nan
    df["cum_distance_nm"] = df["step_distance_nm"].fillna(0).cumsum()

    df["sog_delta"] = df["SOG"].diff()
    df["accel_knots_per_s"] = df["sog_delta"] / df["delta_t_seconds"]
    df["cog_delta_deg"] = angular_diff_deg(df["COG"], df["COG"].shift(1))
    df["turn_rate_deg_per_s"] = df["cog_delta_deg"] / df["delta_t_seconds"]

    valid_distance = df["step_distance_nm"].replace(0, np.nan)
    valid_duration_hours = (df["delta_t_seconds"] / 3600.0).replace(0, np.nan)
    df["fuel_step_est"] = df["fuel_total"] * valid_duration_hours
    df["fuel_per_nm"] = df["fuel_total"] / valid_distance
    df["fuel_per_hour"] = df["fuel_total"] / valid_duration_hours

    ordered_columns = [
        "time",
        "timestamp_unix",
        "latitude",
        "longitude",
        "SOG",
        "SOG_missing_flag",
        "COG",
        "ME1_power",
        "ME1_rpm",
        "ME1_fuel",
        "ME2_power",
        "ME2_rpm",
        "ME2_fuel",
        "ME3_power",
        "ME3_rpm",
        "ME3_fuel",
        "ME4_power",
        "ME4_rpm",
        "ME4_fuel",
        "power_total",
        "rpm_mean",
        "rpm_max",
        "fuel_total",
        "active_engine_count",
        "delta_t_seconds",
        "step_distance_nm",
        "cum_distance_nm",
        "fuel_step_est",
        "sog_delta",
        "accel_knots_per_s",
        "cog_delta_deg",
        "turn_rate_deg_per_s",
        "fuel_per_nm",
        "fuel_per_hour",
    ]
    existing_columns = [col for col in ordered_columns if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in existing_columns]
    return df[existing_columns + remaining_columns]


def build_metadata(df: pd.DataFrame, source_path: Path) -> dict:
    time_min = df["time"].min()
    time_max = df["time"].max()
    return {
        "source_file": str(source_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "time_start_utc": None if pd.isna(time_min) else time_min.isoformat(),
        "time_end_utc": None if pd.isna(time_max) else time_max.isoformat(),
        "total_distance_nm": float(df["cum_distance_nm"].dropna().iloc[-1]) if df["cum_distance_nm"].notna().any() else None,
        "missing_fuel_total_ratio_pct": float(df["fuel_total"].isna().mean() * 100.0),
        "missing_sog_ratio_pct": float(df["SOG"].isna().mean() * 100.0),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = read_ais_source(input_path)
    clean_df = clean_dataframe(raw_df)

    base_name = dataset_base_from_input(input_path)
    cleaned_parquet = output_dir / f"{base_name}_step1_cleaned.parquet"
    cleaned_csv = output_dir / f"{base_name}_step1_cleaned_sample.csv"
    missing_summary_csv = output_dir / f"{base_name}_step1_missing_summary.csv"
    metadata_json = output_dir / f"{base_name}_step1_metadata.json"

    clean_df.to_parquet(cleaned_parquet, index=False)
    clean_df.head(5000).to_csv(cleaned_csv, index=False, encoding="utf-8-sig")
    build_missing_summary(clean_df).to_csv(missing_summary_csv, index=False, encoding="utf-8-sig")

    metadata = build_metadata(clean_df, input_path)
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"cleaned_parquet={cleaned_parquet}")
    print(f"cleaned_csv_sample={cleaned_csv}")
    print(f"missing_summary={missing_summary_csv}")
    print(f"metadata={metadata_json}")
    print(f"rows={len(clean_df)}")
    print(f"columns={len(clean_df.columns)}")


if __name__ == "__main__":
    main()
