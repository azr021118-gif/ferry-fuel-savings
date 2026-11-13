from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR as DEFAULT_OUTPUT_DIR


DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step2_segmented.parquet"

NORMALIZED_POINTS = 101
VALID_DIRECTIONS = {"berth_2_to_berth_1", "berth_1_to_berth_2"}
MOVING_NO_SOG_DISTANCE_NM = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step-3 voyage fuel and normalized main-voyage profile outputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Step-2 parquet path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for step-3 outputs.",
    )
    return parser.parse_args()


def assign_runs(flag: pd.Series) -> pd.Series:
    return flag.ne(flag.shift(fill_value=False)).cumsum()


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df["fuel_step_est"] = df["fuel_step_est"].fillna(0.0)
    df["fuel_total"] = df["fuel_total"].fillna(0.0)
    df["step_distance_nm"] = df["step_distance_nm"].fillna(0.0)
    df["delta_t_seconds"] = df["delta_t_seconds"].fillna(1.0)
    df["rpm_mean"] = df["rpm_mean"].fillna(0.0)
    df["active_engine_count"] = df["active_engine_count"].fillna(0)
    if "SOG_missing_flag" not in df.columns:
        df["SOG_missing_flag"] = df["SOG"].isna()
    else:
        df["SOG_missing_flag"] = df["SOG_missing_flag"].fillna(df["SOG"].isna())
    df["engine_params_available"] = (
        (df["fuel_total"] > 0)
        | (df["rpm_mean"] > 0)
        | (df["active_engine_count"] > 0)
    )
    df["no_sog_engine_flag"] = df["SOG_missing_flag"] & df["engine_params_available"]
    df["moving_without_sog_flag"] = df["no_sog_engine_flag"] & (
        df["step_distance_nm"] > MOVING_NO_SOG_DISTANCE_NM
    )
    return df


def build_voyage_bounds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for voyage_id, group in df.dropna(subset=["voyage_id"]).groupby("voyage_id"):
        rows.append(
            {
                "voyage_id": int(voyage_id),
                "start_idx": int(group.index[0]),
                "end_idx": int(group.index[-1]),
                "start_time": group["time"].iloc[0],
                "end_time": group["time"].iloc[-1],
                "start_berth": group["nearest_berth_name"].iloc[0],
                "end_berth": group["nearest_berth_name"].iloc[-1],
                "direction": f"{group['nearest_berth_name'].iloc[0]}_to_{group['nearest_berth_name'].iloc[-1]}",
            }
        )
    return pd.DataFrame(rows)


def build_single_voyage_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for voyage_id, group in df.dropna(subset=["voyage_id"]).groupby("voyage_id"):
        start_berth = str(group["nearest_berth_name"].iloc[0])
        end_berth = str(group["nearest_berth_name"].iloc[-1])
        direction = f"{start_berth}_to_{end_berth}"
        fuel_main = float(group.loc[group["segment_label"] == "main_voyage", "fuel_step_est"].sum())
        fuel_departure = float(group.loc[group["segment_label"] == "departure", "fuel_step_est"].sum())
        fuel_arrival = float(group.loc[group["segment_label"] == "arrival", "fuel_step_est"].sum())
        fuel_maneuver = float(
            group.loc[group["segment_label"].isin(["departure", "arrival"]), "fuel_step_est"].sum()
        )
        fuel_departure_cost_eligible = float(
            group.loc[
                (group["segment_label"] == "departure")
                & ~group["no_sog_engine_flag"],
                "fuel_step_est",
            ].sum()
        )
        fuel_arrival_cost_eligible = float(
            group.loc[
                (group["segment_label"] == "arrival")
                & ~group["no_sog_engine_flag"],
                "fuel_step_est",
            ].sum()
        )
        fuel_departure_excluded = float(
            group.loc[
                (group["segment_label"] == "departure")
                & group["no_sog_engine_flag"],
                "fuel_step_est",
            ].sum()
        )
        fuel_arrival_excluded = float(
            group.loc[
                (group["segment_label"] == "arrival")
                & group["no_sog_engine_flag"],
                "fuel_step_est",
            ].sum()
        )
        fuel_maneuver_cost_eligible = float(
            group.loc[
                group["segment_label"].isin(["departure", "arrival"])
                & ~group["no_sog_engine_flag"],
                "fuel_step_est",
            ].sum()
        )
        fuel_maneuver_excluded = float(
            group.loc[
                group["segment_label"].isin(["departure", "arrival"])
                & group["no_sog_engine_flag"],
                "fuel_step_est",
            ].sum()
        )
        fuel_underway = float(group["fuel_step_est"].sum())

        main_distance = float(group.loc[group["segment_label"] == "main_voyage", "step_distance_nm"].sum())
        fuel_per_nm_main = fuel_main / main_distance if main_distance > 0 else np.nan

        summaries.append(
            {
                "voyage_id": int(voyage_id),
                "start_berth": start_berth,
                "end_berth": end_berth,
                "direction": direction,
                "is_valid_direction": direction in VALID_DIRECTIONS,
                "start_time": group["time"].iloc[0],
                "end_time": group["time"].iloc[-1],
                "duration_minutes_underway": float(group["delta_t_seconds"].sum() / 60.0),
                "fuel_main_voyage": fuel_main,
                "fuel_departure": fuel_departure,
                "fuel_arrival": fuel_arrival,
                "fuel_maneuvering": fuel_maneuver,
                "fuel_departure_cost_eligible": fuel_departure_cost_eligible,
                "fuel_arrival_cost_eligible": fuel_arrival_cost_eligible,
                "fuel_maneuvering_cost_eligible": fuel_maneuver_cost_eligible,
                "fuel_departure_excluded_no_sog": fuel_departure_excluded,
                "fuel_arrival_excluded_no_sog": fuel_arrival_excluded,
                "fuel_maneuvering_excluded_no_sog": fuel_maneuver_excluded,
                "fuel_underway_total": fuel_underway,
                "distance_main_voyage_nm": main_distance,
                "fuel_per_nm_main_voyage": fuel_per_nm_main,
                "no_sog_engine_points_total": int(group["no_sog_engine_flag"].sum()),
                "no_sog_engine_points_in_maneuvering": int(
                    group.loc[group["segment_label"].isin(["departure", "arrival"]), "no_sog_engine_flag"].sum()
                ),
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

    return pd.DataFrame(summaries)


def normalize_main_voyage_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tau_grid = np.linspace(0.0, 1.0, NORMALIZED_POINTS)
    records = []

    for voyage_id, group in df.dropna(subset=["voyage_id"]).groupby("voyage_id"):
        direction = f"{group['nearest_berth_name'].iloc[0]}_to_{group['nearest_berth_name'].iloc[-1]}"
        if direction not in VALID_DIRECTIONS:
            continue
        main = group.loc[group["segment_label"] == "main_voyage"].copy()
        if len(main) < 5:
            continue
        sog_series = main["SOG"].interpolate(limit_direction="both")
        if sog_series.isna().all():
            sog_series = pd.Series(0.0, index=main.index)
        else:
            sog_series = sog_series.bfill().ffill()

        elapsed = main["delta_t_seconds"].cumsum().to_numpy(dtype=float)
        elapsed = elapsed - elapsed[0]
        if elapsed[-1] <= 0:
            continue
        tau = elapsed / elapsed[-1]

        for tau_value, sog, fuel_rate, rpm in zip(
            tau_grid,
            np.interp(tau_grid, tau, sog_series.to_numpy(dtype=float)),
            np.interp(tau_grid, tau, main["fuel_total"].to_numpy(dtype=float)),
            np.interp(tau_grid, tau, main["rpm_mean"].to_numpy(dtype=float)),
        ):
            records.append(
                {
                    "voyage_id": int(voyage_id),
                    "direction": direction,
                    "tau": float(tau_value),
                    "SOG_interp": float(sog),
                    "fuel_rate_interp": float(fuel_rate),
                    "rpm_interp": float(rpm),
                }
            )

    normalized_df = pd.DataFrame(records)
    if normalized_df.empty:
        return normalized_df, pd.DataFrame(
            columns=[
                "direction",
                "tau",
                "sog_mean",
                "sog_std",
                "fuel_rate_mean",
                "fuel_rate_std",
                "rpm_mean",
                "rpm_std",
                "voyage_count",
            ]
        )
    stats_df = (
        normalized_df.groupby(["direction", "tau"])
        .agg(
            sog_mean=("SOG_interp", "mean"),
            sog_std=("SOG_interp", "std"),
            fuel_rate_mean=("fuel_rate_interp", "mean"),
            fuel_rate_std=("fuel_rate_interp", "std"),
            rpm_mean=("rpm_interp", "mean"),
            rpm_std=("rpm_interp", "std"),
            voyage_count=("voyage_id", "nunique"),
        )
        .reset_index()
    )
    return normalized_df, stats_df


def build_metadata(
    df: pd.DataFrame,
    single_summary_all: pd.DataFrame,
) -> dict:
    valid_summary = single_summary_all.loc[single_summary_all["is_valid_direction"]]
    return {
        "row_count": int(len(df)),
        "single_voyage_count": int(len(valid_summary)),
        "single_voyage_count_all": int(len(single_summary_all)),
        "directions": valid_summary["direction"].value_counts().to_dict(),
        "excluded_directions": (
            single_summary_all.loc[
                ~single_summary_all["is_valid_direction"], "direction"
            ].value_counts().to_dict()
        ),
        "rows_with_no_sog_engine": int(df["no_sog_engine_flag"].sum()),
        "rows_moving_without_sog": int(df["moving_without_sog_flag"].sum()),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    single_summary_all = build_single_voyage_summary(df)
    single_summary = single_summary_all.loc[single_summary_all["is_valid_direction"]].copy()
    normalized_profiles, normalized_stats = normalize_main_voyage_profiles(df)

    base_name = args.input.stem.replace("_step2_segmented", "")
    step3_parquet = output_dir / f"{base_name}_step3_enriched.parquet"
    single_summary_csv = output_dir / f"{base_name}_step3_single_voyage_energy.csv"
    normalized_profiles_csv = output_dir / f"{base_name}_step3_main_profile_normalized.csv"
    normalized_stats_csv = output_dir / f"{base_name}_step3_main_profile_stats.csv"
    metadata_json = output_dir / f"{base_name}_step3_metadata.json"

    df.to_parquet(step3_parquet, index=False)
    single_summary.to_csv(single_summary_csv, index=False, encoding="utf-8-sig")
    normalized_profiles.to_csv(normalized_profiles_csv, index=False, encoding="utf-8-sig")
    normalized_stats.to_csv(normalized_stats_csv, index=False, encoding="utf-8-sig")
    metadata_json.write_text(
        json.dumps(
            build_metadata(
                df,
                single_summary_all,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"step3_parquet={step3_parquet}")
    print(f"single_voyage_csv={single_summary_csv}")
    print(f"normalized_profiles_csv={normalized_profiles_csv}")
    print(f"normalized_stats_csv={normalized_stats_csv}")
    print(f"metadata_json={metadata_json}")


if __name__ == "__main__":
    main()
