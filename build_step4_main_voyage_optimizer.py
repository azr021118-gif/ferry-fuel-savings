from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR as DEFAULT_OUTPUT_DIR


DEFAULT_STEP3_PARQUET = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step3_enriched.parquet"
DEFAULT_SINGLE_CSV = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step3_single_voyage_energy.csv"

PRIMARY_DIRECTION = "berth_2_to_berth_1"
REVERSE_DIRECTION = "berth_1_to_berth_2"
OPTIMIZATION_DIRECTIONS = (PRIMARY_DIRECTION, REVERSE_DIRECTION)  # both directions since 2026-11-11
TAU_POINTS = 61
PROGRESS_BINS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step-4 main-voyage optimization candidates, scores, and summaries."
    )
    parser.add_argument("--step3-parquet", type=Path, default=DEFAULT_STEP3_PARQUET)
    parser.add_argument("--single-csv", type=Path, default=DEFAULT_SINGLE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def filter_typical_main_voyages(single: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for direction in OPTIMIZATION_DIRECTIONS:
        subset = single.loc[single["direction"] == direction].copy()
        if subset.empty:
            continue
        q1 = subset["main_voyage_minutes"].quantile(0.25)
        q3 = subset["main_voyage_minutes"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        subset["is_typical_main_voyage"] = subset["main_voyage_minutes"] <= upper
        parts.append(subset)
    if not parts:
        raise ValueError(f"No optimizer voyages found for primary direction: {PRIMARY_DIRECTION}")
    return pd.concat(parts, ignore_index=True)


def normalized_progress(group: pd.DataFrame) -> np.ndarray:
    elapsed = group["delta_t_seconds"].cumsum().to_numpy(dtype=float)
    elapsed = elapsed - elapsed[0]
    if elapsed[-1] <= 0:
        return np.linspace(0.0, 1.0, len(group))
    return elapsed / elapsed[-1]


def interpolate_main_voyage(group: pd.DataFrame, tau_grid: np.ndarray) -> pd.DataFrame:
    g = group.sort_values("time").copy()
    tau = normalized_progress(g)

    sog = g["SOG"].interpolate(limit_direction="both")
    if sog.isna().all():
        sog = pd.Series(0.0, index=g.index)
    else:
        sog = sog.bfill().ffill()

    interp = pd.DataFrame(
        {
            "tau": tau_grid,
            "latitude": np.interp(tau_grid, tau, g["latitude"].to_numpy(dtype=float)),
            "longitude": np.interp(tau_grid, tau, g["longitude"].to_numpy(dtype=float)),
            "SOG_interp": np.interp(tau_grid, tau, sog.to_numpy(dtype=float)),
            "fuel_rate_interp": np.interp(tau_grid, tau, g["fuel_total"].to_numpy(dtype=float)),
            "rpm_interp": np.interp(tau_grid, tau, g["rpm_mean"].to_numpy(dtype=float)),
            "turn_rate_interp": np.interp(
                tau_grid,
                tau,
                g["turn_rate_deg_per_s"].fillna(0.0).to_numpy(dtype=float),
            ),
        }
    )
    return interp


def compute_centerlines(step3: pd.DataFrame, single: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tau_grid = np.linspace(0.0, 1.0, TAU_POINTS)
    interp_rows = []

    valid_ids = set(single.loc[single["is_typical_main_voyage"], "voyage_id"].astype(int).tolist())
    for voyage_id, group in step3.loc[
        step3["voyage_id"].notna()
        & step3["segment_label"].eq("main_voyage")
        & step3["voyage_id"].astype("Int64").isin(valid_ids)
    ].groupby("voyage_id"):
        direction = f"{group['nearest_berth_name'].iloc[0]}_to_{group['nearest_berth_name'].iloc[-1]}"
        if direction not in OPTIMIZATION_DIRECTIONS:
            continue
        interp = interpolate_main_voyage(group, tau_grid)
        interp["voyage_id"] = int(voyage_id)
        interp["direction"] = direction
        interp_rows.append(interp)

    interp_df = pd.concat(interp_rows, ignore_index=True)
    centerline = (
        interp_df.groupby(["direction", "tau"])
        .agg(
            center_lat=("latitude", "median"),
            center_lon=("longitude", "median"),
            sog_mean=("SOG_interp", "mean"),
            fuel_rate_mean=("fuel_rate_interp", "mean"),
            rpm_mean=("rpm_interp", "mean"),
            turn_rate_mean=("turn_rate_interp", "mean"),
            voyage_count=("voyage_id", "nunique"),
        )
        .reset_index()
    )
    return interp_df, centerline


def compute_segment_bin_summary(interp_df: pd.DataFrame) -> pd.DataFrame:
    bin_edges = np.linspace(0.0, 1.0, PROGRESS_BINS + 1)
    labels = [f"bin_{i+1}" for i in range(PROGRESS_BINS)]
    df = interp_df.copy()
    df["progress_bin"] = pd.cut(
        df["tau"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return (
        df.groupby(["direction", "voyage_id", "progress_bin"], observed=False)
        .agg(
            fuel_rate_mean=("fuel_rate_interp", "mean"),
            sog_mean=("SOG_interp", "mean"),
            rpm_mean=("rpm_interp", "mean"),
            turn_rate_abs_mean=("turn_rate_interp", lambda s: float(np.mean(np.abs(s)))),
        )
        .reset_index()
    )


def build_candidate_features(
    step3: pd.DataFrame,
    single: pd.DataFrame,
    interp_df: pd.DataFrame,
    centerline: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    center_lookup = {
        direction: grp.sort_values("tau").reset_index(drop=True)
        for direction, grp in centerline.groupby("direction")
    }

    valid_ids = set(single.loc[single["is_typical_main_voyage"], "voyage_id"].astype(int).tolist())
    for voyage_id, group in step3.loc[
        step3["voyage_id"].notna()
        & step3["segment_label"].eq("main_voyage")
        & step3["voyage_id"].astype("Int64").isin(valid_ids)
    ].groupby("voyage_id"):
        direction = f"{group['nearest_berth_name'].iloc[0]}_to_{group['nearest_berth_name'].iloc[-1]}"
        if direction not in OPTIMIZATION_DIRECTIONS:
            continue

        summary = single.loc[single["voyage_id"] == int(voyage_id)].iloc[0]
        interp = interp_df.loc[interp_df["voyage_id"] == int(voyage_id)].sort_values("tau")
        center = center_lookup[direction]

        lateral = haversine_nm(
            interp["latitude"].to_numpy(dtype=float),
            interp["longitude"].to_numpy(dtype=float),
            center["center_lat"].to_numpy(dtype=float),
            center["center_lon"].to_numpy(dtype=float),
        )

        start_lat = float(group["latitude"].iloc[0])
        start_lon = float(group["longitude"].iloc[0])
        end_lat = float(group["latitude"].iloc[-1])
        end_lon = float(group["longitude"].iloc[-1])
        beeline_nm = float(haversine_nm(start_lat, start_lon, end_lat, end_lon))
        sailed_nm = float(group["step_distance_nm"].sum())
        tortuosity = sailed_nm / beeline_nm if beeline_nm > 0 else np.nan

        rows.append(
            {
                "voyage_id": int(voyage_id),
                "direction": direction,
                "start_time": summary["start_time"],
                "end_time": summary["end_time"],
                "fuel_main_voyage": float(summary["fuel_main_voyage"]),
                "fuel_per_nm_main_voyage": float(summary["fuel_per_nm_main_voyage"]),
                "main_voyage_minutes": float(summary["main_voyage_minutes"]),
                "distance_main_voyage_nm": float(summary["distance_main_voyage_nm"]),
                "rpm_peak_main": float(group["rpm_mean"].max()),
                "rpm_std_main": float(group["rpm_mean"].std(ddof=0)),
                "turn_rate_rms_main": float(np.sqrt(np.mean(np.square(group["turn_rate_deg_per_s"].fillna(0.0))))),
                "turn_rate_peak_main": float(group["turn_rate_deg_per_s"].abs().max()),
                "tortuosity_main": float(tortuosity),
                "lateral_dev_mean_nm": float(np.mean(lateral)),
                "lateral_dev_max_nm": float(np.max(lateral)),
                "sog_mean_main": float(group["SOG"].interpolate(limit_direction="both").mean()),
                "sog_std_main": float(group["SOG"].interpolate(limit_direction="both").std(ddof=0)),
            }
        )

    return pd.DataFrame(rows)


def minmax_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    smin = float(s.min())
    smax = float(s.max())
    if np.isclose(smin, smax):
        return pd.Series(0.0, index=s.index)
    return (s - smin) / (smax - smin)


def pareto_efficient(costs: np.ndarray) -> np.ndarray:
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        if np.any(dominates):
            is_efficient[i] = False
    return is_efficient


def build_weight_grid() -> pd.DataFrame:
    rows = []
    for fuel_w in np.arange(0.50, 0.81, 0.05):
        for oper_w in np.arange(0.15, 0.36, 0.05):
            time_w = 1.0 - fuel_w - oper_w
            if 0.05 <= time_w <= 0.25:
                rows.append(
                    {
                        "fuel_weight": round(float(fuel_w), 3),
                        "operation_weight": round(float(oper_w), 3),
                        "time_weight": round(float(time_w), 3),
                    }
                )
    return pd.DataFrame(rows)


def score_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    weights = build_weight_grid()
    all_scored = []
    summaries: dict[str, dict] = {}

    for direction in OPTIMIZATION_DIRECTIONS:
        sub = candidates.loc[candidates["direction"] == direction].copy().reset_index(drop=True)
        sub["fuel_metric"] = 0.5 * minmax_scale(sub["fuel_main_voyage"]) + 0.5 * minmax_scale(
            sub["fuel_per_nm_main_voyage"]
        )
        sub["operation_metric"] = (
            0.30 * minmax_scale(sub["rpm_peak_main"])
            + 0.30 * minmax_scale(sub["turn_rate_rms_main"])
            + 0.20 * minmax_scale(sub["tortuosity_main"])
            + 0.20 * minmax_scale(sub["lateral_dev_mean_nm"])
        )
        sub["time_metric"] = minmax_scale(sub["main_voyage_minutes"])
        sub["pareto_efficient"] = pareto_efficient(
            sub[["fuel_metric", "operation_metric", "time_metric"]].to_numpy(dtype=float)
        )

        rank_rows = []
        for _, w in weights.iterrows():
            score = (
                w["fuel_weight"] * sub["fuel_metric"]
                + w["operation_weight"] * sub["operation_metric"]
                + w["time_weight"] * sub["time_metric"]
            )
            temp = sub[["voyage_id", "direction", "pareto_efficient"]].copy()
            temp["fuel_weight"] = w["fuel_weight"]
            temp["operation_weight"] = w["operation_weight"]
            temp["time_weight"] = w["time_weight"]
            temp["composite_score"] = score
            temp["rank"] = score.rank(method="dense", ascending=True)
            rank_rows.append(temp)

        rank_df = pd.concat(rank_rows, ignore_index=True)
        summary = (
            rank_df.groupby(["voyage_id", "direction", "pareto_efficient"])
            .agg(
                mean_rank=("rank", "mean"),
                top1_frequency=("rank", lambda s: float(np.mean(s == 1))),
                top3_frequency=("rank", lambda s: float(np.mean(s <= 3))),
                mean_score=("composite_score", "mean"),
            )
            .reset_index()
        )
        merged = sub.merge(summary, on=["voyage_id", "direction", "pareto_efficient"], how="left")
        merged["better_than_pct"] = (
            1.0 - (merged["mean_rank"] - 1.0) / max(1.0, len(merged) - 1.0)
        ) * 100.0
        merged = merged.sort_values(
            ["pareto_efficient", "top1_frequency", "mean_rank", "mean_score"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        best = merged.iloc[0].to_dict()
        median_row = merged.median(numeric_only=True).to_dict()
        summaries[direction] = {
            "direction": direction,
            "selected_voyage_id": int(best["voyage_id"]),
            "pareto_efficient": bool(best["pareto_efficient"]),
            "top1_frequency_pct": float(best["top1_frequency"] * 100.0),
            "better_than_pct": float(best["better_than_pct"]),
            "fuel_improvement_vs_median_pct": float(
                (median_row["fuel_main_voyage"] - best["fuel_main_voyage"]) / median_row["fuel_main_voyage"] * 100.0
            ),
            "operation_improvement_vs_median_pct": float(
                (median_row["operation_metric"] - best["operation_metric"])
                / max(median_row["operation_metric"], 1e-9)
                * 100.0
            ),
            "time_improvement_vs_median_pct": float(
                (median_row["main_voyage_minutes"] - best["main_voyage_minutes"])
                / median_row["main_voyage_minutes"]
                * 100.0
            ),
        }

        all_scored.append(merged)

    scored_df = pd.concat(all_scored, ignore_index=True)
    return scored_df, weights, summaries


def build_metadata(single: pd.DataFrame, candidates: pd.DataFrame, summaries: dict) -> dict:
    return {
        "optimization_directions": list(OPTIMIZATION_DIRECTIONS),
        "candidate_count": int(len(candidates)),
        "direction_counts": candidates["direction"].value_counts().to_dict(),
        "typical_main_voyage_count": int(single["is_typical_main_voyage"].sum()),
        "selected_routes": summaries,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step3 = pd.read_parquet(args.step3_parquet)
    single_all = pd.read_csv(args.single_csv)
    base_name = args.step3_parquet.stem.replace("_step3_enriched", "")
    single = single_all.loc[single_all["direction"].isin(OPTIMIZATION_DIRECTIONS)].copy()
    single = filter_typical_main_voyages(single)

    interp_df, centerline = compute_centerlines(step3, single)
    bin_summary = compute_segment_bin_summary(interp_df)
    candidates = build_candidate_features(step3, single, interp_df, centerline)
    scored, weights, summaries = score_candidates(candidates)

    base = f"{base_name}_step4_main_route"
    candidate_csv = args.output_dir / f"{base}_candidates.csv"
    centerline_csv = args.output_dir / f"{base}_centerline.csv"
    tau_csv = args.output_dir / f"{base}_interpolated.csv"
    bins_csv = args.output_dir / f"{base}_progress_bins.csv"
    weights_csv = args.output_dir / f"{base}_weight_grid.csv"
    summary_json = args.output_dir / f"{base}_summary.json"

    scored.to_csv(candidate_csv, index=False, encoding="utf-8-sig")
    centerline.to_csv(centerline_csv, index=False, encoding="utf-8-sig")
    interp_df.to_csv(tau_csv, index=False, encoding="utf-8-sig")
    bin_summary.to_csv(bins_csv, index=False, encoding="utf-8-sig")
    weights.to_csv(weights_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(
            {
                **build_metadata(single, scored, summaries),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"candidate_csv={candidate_csv}")
    print(f"centerline_csv={centerline_csv}")
    print(f"tau_csv={tau_csv}")
    print(f"bins_csv={bins_csv}")
    print(f"weights_csv={weights_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
