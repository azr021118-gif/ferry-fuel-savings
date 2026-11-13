from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR as DEFAULT_OUTPUT_DIR

DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step1_cleaned.parquet"

STOP_SPEED_KNOTS = 0.5
STOP_STEP_DISTANCE_NM = 0.001
BERTH_CORE_RADIUS_NM = 0.18
PORT_ZONE_RADIUS_NM = 1.0
MIN_BERTH_STOP_SECONDS = 300
LOW_SPEED_PORT_KNOTS = 6.0
MIN_SPEED_PROXY_KNOTS = 0.05
SPEED_SMOOTH_WINDOW = 60
CRUISE_QUANTILE = 0.95
CRUISE_SPEED_RATIO = 0.78
MIN_CRUISE_DURATION_SECONDS = 150
MOTION_WINDOW_SECONDS = 240
MOTION_DELTA_THRESHOLD_NM = 0.03
MOTION_RATIO_THRESHOLD = 0.75
MANEUVER_RADIUS_NM = 0.25
DISPLACEMENT_LOOKAHEAD = 45
DISPLACEMENT_THRESHOLD_NM = 0.01
INNER_SETTLE_RADIUS_NM = 0.01
BERTH_MANEUVER_RADIUS_NM = {
    "berth_1": 0.25,
    "berth_2": 0.12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step-2 voyage segmentation labels from the cleaned ship dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Step-1 parquet path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for segmented outputs.",
    )
    return parser.parse_args()


def haversine_nm(lat1, lon1, lat2, lon2):
    radius_m = 6_371_000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return (radius_m * c) / 1852.0


def assign_runs(flag: pd.Series) -> pd.Series:
    return flag.ne(flag.shift(fill_value=False)).cumsum()


def build_stop_like_flag(df: pd.DataFrame) -> pd.Series:
    speed_stop = df["SOG"].notna() & (df["SOG"] <= STOP_SPEED_KNOTS)
    missing_stop = df.get("SOG_missing_flag", pd.Series(False, index=df.index)).fillna(False) & (
        df["step_distance_nm"].fillna(0.0) <= STOP_STEP_DISTANCE_NM
    )
    return speed_stop | missing_stop


def estimate_berth_centers(df: pd.DataFrame, max_berths: int = 2) -> pd.DataFrame:
    stop_like = build_stop_like_flag(df)
    stop_points = df.loc[
        df["latitude"].notna()
        & df["longitude"].notna()
        & stop_like,
        ["latitude", "longitude"],
    ].copy()

    stop_points["lat_bin"] = stop_points["latitude"].round(3)
    stop_points["lon_bin"] = stop_points["longitude"].round(3)

    coarse_counts = (
        stop_points.groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="point_count")
        .sort_values("point_count", ascending=False)
    )

    selected_rows: list[dict] = []
    chosen_seeds: list[tuple[float, float]] = []
    berth_index = 1

    for row in coarse_counts.itertuples(index=False):
        seed_lat = float(row.lat_bin)
        seed_lon = float(row.lon_bin)

        too_close = False
        for chosen_lat, chosen_lon in chosen_seeds:
            seed_distance = haversine_nm(seed_lat, seed_lon, chosen_lat, chosen_lon)
            if seed_distance < 0.5:
                too_close = True
                break
        if too_close:
            continue

        distances = haversine_nm(
            stop_points["latitude"], stop_points["longitude"], seed_lat, seed_lon
        )
        cluster = stop_points.loc[distances <= 0.25]
        if len(cluster) < 100:
            continue

        center_lat = float(cluster["latitude"].median())
        center_lon = float(cluster["longitude"].median())
        selected_rows.append(
            {
                "berth_id": berth_index,
                "berth_name": f"berth_{berth_index}",
                "center_lat": center_lat,
                "center_lon": center_lon,
                "support_points": int(len(cluster)),
            }
        )
        chosen_seeds.append((center_lat, center_lon))
        berth_index += 1

        if len(selected_rows) >= max_berths:
            break

    if not selected_rows:
        raise ValueError("No berth centers could be estimated from stop points.")

    return pd.DataFrame(selected_rows)


def attach_berth_distances(df: pd.DataFrame, berth_centers: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    distance_columns = []
    stop_like = build_stop_like_flag(result)

    for row in berth_centers.itertuples(index=False):
        col = f"dist_{row.berth_name}_nm"
        result[col] = haversine_nm(
            result["latitude"], result["longitude"], row.center_lat, row.center_lon
        )
        distance_columns.append(col)

    result["nearest_berth_distance_nm"] = result[distance_columns].min(axis=1)
    nearest_idx = result[distance_columns].to_numpy().argmin(axis=1)
    berth_names = berth_centers["berth_name"].tolist()
    result["nearest_berth_name"] = [berth_names[idx] for idx in nearest_idx]
    result["near_port_zone"] = result["nearest_berth_distance_nm"] <= PORT_ZONE_RADIUS_NM
    result["at_berth_core"] = (
        result["nearest_berth_distance_nm"] <= BERTH_CORE_RADIUS_NM
    ) & stop_like
    return result


def identify_berth_stays(df: pd.DataFrame) -> pd.DataFrame:
    stay_flag = df["at_berth_core"].fillna(False)
    run_id = assign_runs(stay_flag)
    stays = []

    for current_run, group in df.groupby(run_id):
        if not bool(group["at_berth_core"].iloc[0]):
            continue

        duration_seconds = float(group["delta_t_seconds"].fillna(1).sum())
        if duration_seconds < MIN_BERTH_STOP_SECONDS:
            continue

        stays.append(
            {
                "stay_id": len(stays) + 1,
                "run_id": int(current_run),
                "start_idx": int(group.index[0]),
                "end_idx": int(group.index[-1]),
                "start_time": group["time"].iloc[0],
                "end_time": group["time"].iloc[-1],
                "duration_seconds": duration_seconds,
                "berth_name": group["nearest_berth_name"].mode().iloc[0],
            }
        )

    return pd.DataFrame(stays)


def build_speed_proxy(df: pd.DataFrame) -> pd.Series:
    speed_proxy = df["SOG"].copy()
    inferred_speed = pd.Series(np.nan, index=df.index, dtype=float)
    valid_mask = df["step_distance_nm"].notna() & df["delta_t_seconds"].notna() & (df["delta_t_seconds"] > 0)
    inferred_speed.loc[valid_mask] = (
        df.loc[valid_mask, "step_distance_nm"] / (df.loc[valid_mask, "delta_t_seconds"] / 3600.0)
    )
    speed_proxy = speed_proxy.fillna(inferred_speed)
    return speed_proxy.clip(lower=0.0)


def sustained_true_runs(
    mask: pd.Series, delta_t_seconds: pd.Series, min_duration_seconds: float
) -> list[tuple[int, int]]:
    run_ids = assign_runs(mask.fillna(False))
    qualifying_runs: list[tuple[int, int]] = []

    for _, group in pd.DataFrame(
        {
            "flag": mask.fillna(False),
            "delta_t_seconds": delta_t_seconds.fillna(1.0),
        },
        index=mask.index,
    ).groupby(run_ids):
        if not bool(group["flag"].iloc[0]):
            continue
        run_duration = float(group["delta_t_seconds"].sum())
        if run_duration >= min_duration_seconds:
            qualifying_runs.append((int(group.index[0]), int(group.index[-1])))

    return qualifying_runs


def leading_true_end_index(mask: pd.Series) -> int | None:
    filled = mask.fillna(False)
    if filled.empty or not bool(filled.iloc[0]):
        return None
    false_positions = np.flatnonzero(~filled.to_numpy())
    if len(false_positions) == 0:
        return int(filled.index[-1])
    return int(filled.index[false_positions[0] - 1])


def trailing_true_start_index(mask: pd.Series) -> int | None:
    filled = mask.fillna(False)
    if filled.empty or not bool(filled.iloc[-1]):
        return None
    false_positions = np.flatnonzero(~filled.to_numpy())
    if len(false_positions) == 0:
        return int(filled.index[0])
    return int(filled.index[false_positions[-1] + 1])


def find_departure_start_idx(df: pd.DataFrame, stay: pd.Series) -> int:
    stay_slice = df.loc[stay["start_idx"] : stay["end_idx"]].copy()
    tail = stay_slice.tail(MOTION_WINDOW_SECONDS).copy()
    if tail.empty:
        return int(stay["end_idx"])

    dist = tail["nearest_berth_distance_nm"].rolling(5, min_periods=1).mean().reset_index(drop=True)
    lats = tail["latitude"].reset_index(drop=True)
    lons = tail["longitude"].reset_index(drop=True)
    original_idx = tail.index.to_list()

    max_offset = min(DISPLACEMENT_LOOKAHEAD, max(1, len(tail) - 1))
    candidate_idx = int(original_idx[-1])

    for pos in range(0, len(tail) - max_offset):
        future_pos = min(pos + max_offset, len(tail) - 1)
        displacement = haversine_nm(
            float(lats.iloc[pos]),
            float(lons.iloc[pos]),
            float(lats.iloc[future_pos]),
            float(lons.iloc[future_pos]),
        )
        dist_gain = float(dist.iloc[future_pos] - dist.iloc[pos])
        if (
            displacement >= DISPLACEMENT_THRESHOLD_NM
            and dist_gain >= DISPLACEMENT_THRESHOLD_NM
        ):
            candidate_idx = int(original_idx[pos])
            break

    return candidate_idx


def find_arrival_end_idx(df: pd.DataFrame, stay: pd.Series) -> int:
    stay_slice = df.loc[stay["start_idx"] : stay["end_idx"]].copy()
    head = stay_slice.head(MOTION_WINDOW_SECONDS).copy()
    if head.empty:
        return int(stay["start_idx"])

    dist = head["nearest_berth_distance_nm"].rolling(5, min_periods=1).mean().reset_index(drop=True)
    lats = head["latitude"].reset_index(drop=True)
    lons = head["longitude"].reset_index(drop=True)
    original_idx = head.index.to_list()

    max_offset = min(DISPLACEMENT_LOOKAHEAD, max(1, len(head) - 1))
    candidate_idx = int(original_idx[0])

    for pos in range(0, len(head) - max_offset):
        future_pos = min(pos + max_offset, len(head) - 1)
        displacement = haversine_nm(
            float(lats.iloc[pos]),
            float(lons.iloc[pos]),
            float(lats.iloc[future_pos]),
            float(lons.iloc[future_pos]),
        )
        if (
            float(dist.iloc[pos]) <= INNER_SETTLE_RADIUS_NM
            and displacement <= DISPLACEMENT_THRESHOLD_NM
        ):
            candidate_idx = int(original_idx[pos])
            break

    return candidate_idx


def build_segmented_dataframe(df: pd.DataFrame, berth_stays: pd.DataFrame) -> pd.DataFrame:
    segmented = df.copy()
    segmented["segment_label"] = "unclassified"
    segmented["voyage_id"] = pd.Series(pd.NA, index=segmented.index, dtype="Int64")

    if berth_stays.empty:
        segmented.loc[segmented["at_berth_core"], "segment_label"] = "berthed"
        segmented.loc[
            segmented["segment_label"].eq("unclassified"), "segment_label"
        ] = "main_voyage"
        return segmented

    for stay in berth_stays.itertuples(index=False):
        segmented.loc[stay.start_idx : stay.end_idx, "segment_label"] = "berthed"

    voyage_counter = 1
    for i in range(len(berth_stays) - 1):
        previous_stay = berth_stays.iloc[i]
        next_stay = berth_stays.iloc[i + 1]

        start_idx = find_departure_start_idx(segmented, previous_stay)
        end_idx = find_arrival_end_idx(segmented, next_stay)
        if start_idx > end_idx:
            continue

        voyage_slice = segmented.loc[start_idx:end_idx].copy()
        if voyage_slice.empty:
            continue

        segmented.loc[start_idx:end_idx, "voyage_id"] = voyage_counter
        segmented.loc[start_idx:end_idx, "segment_label"] = "main_voyage"

        voyage_counter += 1

    segmented.loc[
        segmented["segment_label"].eq("unclassified"), "segment_label"
    ] = "main_voyage"

    return segmented


def build_speed_profile_segments(segmented: pd.DataFrame) -> pd.DataFrame:
    result = segmented.copy()
    result["speed_proxy_knots"] = build_speed_proxy(result)
    result["speed_smooth_knots"] = np.nan
    result["speed_segment_label"] = np.where(
        result["voyage_id"].isna(), "berthed", "main_voyage"
    )
    result["speed_departure_turn_point"] = False
    result["speed_arrival_turn_point"] = False

    for voyage_id, group in result.dropna(subset=["voyage_id"]).groupby("voyage_id"):
        voyage = group.copy()
        speed_proxy = voyage["speed_proxy_knots"].copy()
        smooth = (
            speed_proxy.rolling(window=SPEED_SMOOTH_WINDOW, min_periods=1, center=True)
            .median()
            .bfill()
            .ffill()
        )
        result.loc[voyage.index, "speed_smooth_knots"] = smooth

        valid_speed = smooth.dropna()
        if valid_speed.empty:
            result.loc[voyage.index, "speed_segment_label"] = voyage["segment_label"].to_numpy()
            continue

        peak_speed = float(valid_speed.quantile(CRUISE_QUANTILE))
        cruise_threshold = max(MIN_SPEED_PROXY_KNOTS, peak_speed * CRUISE_SPEED_RATIO)
        cruise_mask = smooth >= cruise_threshold
        cruise_runs = sustained_true_runs(
            cruise_mask,
            voyage["delta_t_seconds"],
            min_duration_seconds=MIN_CRUISE_DURATION_SECONDS,
        )

        start_idx = int(voyage.index[0])
        end_idx = int(voyage.index[-1])
        start_berth = str(voyage["nearest_berth_name"].iloc[0])
        end_berth = str(voyage["nearest_berth_name"].iloc[-1])
        dep_col = f"dist_{start_berth}_nm"
        arr_col = f"dist_{end_berth}_nm"
        dep_radius = BERTH_MANEUVER_RADIUS_NM.get(start_berth, MANEUVER_RADIUS_NM)
        arr_radius = BERTH_MANEUVER_RADIUS_NM.get(end_berth, MANEUVER_RADIUS_NM)
        if cruise_runs:
            first_cruise_start = int(cruise_runs[0][0])
            last_cruise_end = int(cruise_runs[-1][1])
        else:
            peak_idx = int(valid_speed.idxmax())
            first_cruise_start = peak_idx
            last_cruise_end = peak_idx

        departure_end_idx = max(start_idx, min(first_cruise_start - 1, end_idx))
        arrival_start_idx = max(start_idx, min(last_cruise_end + 1, end_idx))

        dep_prefix_end = leading_true_end_index(voyage[dep_col] <= dep_radius)
        if dep_prefix_end is not None:
            departure_end_idx = max(departure_end_idx, dep_prefix_end)

        arr_suffix_start = trailing_true_start_index(voyage[arr_col] <= arr_radius)
        if arr_suffix_start is not None:
            arrival_start_idx = min(arrival_start_idx, arr_suffix_start)

        result.loc[voyage.index, "speed_segment_label"] = "main_voyage"
        if departure_end_idx >= start_idx:
            result.loc[start_idx:departure_end_idx, "speed_segment_label"] = "departure"
            result.loc[departure_end_idx, "speed_departure_turn_point"] = True
        if arrival_start_idx <= end_idx:
            result.loc[arrival_start_idx:end_idx, "speed_segment_label"] = "arrival"
            result.loc[arrival_start_idx, "speed_arrival_turn_point"] = True

        if arrival_start_idx <= departure_end_idx:
            boundary_idx = int(valid_speed.idxmax())
            result.loc[voyage.index, "speed_segment_label"] = "main_voyage"
            result.loc[start_idx:boundary_idx, "speed_segment_label"] = "departure"
            result.loc[boundary_idx:end_idx, "speed_segment_label"] = "arrival"
            result.loc[boundary_idx, "speed_departure_turn_point"] = True
            result.loc[boundary_idx, "speed_arrival_turn_point"] = True

    return result


def build_voyage_summary(segmented: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for voyage_id, group in segmented.dropna(subset=["voyage_id"]).groupby("voyage_id"):
        rows.append(
            {
                "voyage_id": int(voyage_id),
                "start_time": group["time"].iloc[0],
                "end_time": group["time"].iloc[-1],
                "duration_hours": float(group["delta_t_seconds"].fillna(1).sum() / 3600.0),
                "start_berth": group["nearest_berth_name"].iloc[0],
                "end_berth": group["nearest_berth_name"].iloc[-1],
                "fuel_rate_sample_sum": float(group["fuel_total"].sum(min_count=1)),
                "fuel_integral_est": float(group["fuel_step_est"].sum(min_count=1)),
                "distance_nm_sum": float(group["step_distance_nm"].sum(min_count=1)),
                "main_voyage_ratio_pct": float((group["segment_label"].eq("main_voyage").mean()) * 100.0),
                "departure_turn_time": group.loc[group["departure_turn_point"], "time"].min(),
                "arrival_turn_time": group.loc[group["arrival_turn_point"], "time"].min(),
                "departure_minutes": float(
                    group.loc[group["segment_label"] == "departure", "delta_t_seconds"].sum() / 60.0
                ),
                "arrival_minutes": float(
                    group.loc[group["segment_label"] == "arrival", "delta_t_seconds"].sum() / 60.0
                ),
                "main_voyage_minutes": float(
                    group.loc[group["segment_label"] == "main_voyage", "delta_t_seconds"].sum() / 60.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_metadata(segmented: pd.DataFrame, berth_centers: pd.DataFrame, berth_stays: pd.DataFrame) -> dict:
    label_counts = segmented["segment_label"].value_counts(dropna=False).to_dict()
    return {
        "row_count": int(len(segmented)),
        "voyage_count": int(segmented["voyage_id"].dropna().nunique()),
        "berth_stay_count": int(len(berth_stays)),
        "berth_centers": berth_centers.to_dict(orient="records"),
        "segment_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "segmentation_method": "speed_profile",
    }


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    berth_centers = estimate_berth_centers(df)
    with_distances = attach_berth_distances(df, berth_centers)
    berth_stays = identify_berth_stays(with_distances)
    segmented = build_segmented_dataframe(with_distances, berth_stays)
    segmented = build_speed_profile_segments(segmented)
    segmented["segment_label"] = segmented["speed_segment_label"]
    segmented["departure_turn_point"] = segmented["speed_departure_turn_point"]
    segmented["arrival_turn_point"] = segmented["speed_arrival_turn_point"]
    segmented = segmented.drop(
        columns=[
            "speed_segment_label",
            "speed_departure_turn_point",
            "speed_arrival_turn_point",
        ],
        errors="ignore",
    )
    voyage_summary = build_voyage_summary(segmented)

    base_name = input_path.stem.replace("_step1_cleaned", "")
    segmented_path = output_dir / f"{base_name}_step2_segmented.parquet"
    segment_sample_path = output_dir / f"{base_name}_step2_segmented_sample.csv"
    voyage_summary_path = output_dir / f"{base_name}_step2_voyage_summary.csv"
    berth_centers_path = output_dir / f"{base_name}_step2_berth_centers.csv"
    metadata_path = output_dir / f"{base_name}_step2_metadata.json"

    segmented.to_parquet(segmented_path, index=False)
    segmented.head(10000).to_csv(segment_sample_path, index=False, encoding="utf-8-sig")
    voyage_summary.to_csv(voyage_summary_path, index=False, encoding="utf-8-sig")
    berth_centers.to_csv(berth_centers_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(
        json.dumps(build_metadata(segmented, berth_centers, berth_stays), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"segmented_parquet={segmented_path}")
    print(f"segment_sample_csv={segment_sample_path}")
    print(f"voyage_summary_csv={voyage_summary_path}")
    print(f"berth_centers_csv={berth_centers_path}")
    print(f"metadata_json={metadata_path}")
    print(f"voyages={segmented['voyage_id'].dropna().nunique()}")
    print(f"berth_stays={len(berth_stays)}")


if __name__ == "__main__":
    main()
