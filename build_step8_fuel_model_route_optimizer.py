from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_step6_segment_fuel_model import build_main_voyage_segments, build_model_objects, load_step3
from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR as DEFAULT_OUTPUT_DIR

DEFAULT_STEP3 = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step3_enriched.parquet"
DEFAULT_STEP7_PATHS = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step7_main_route_clusters_candidate_paths.csv"
DEFAULT_STEP7_ASSIGN = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step7_main_route_clusters_assignments.csv"
DEFAULT_STEP4_INTERP = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step4_main_route_interpolated.csv"
DEFAULT_STEP6_COMPARISON = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step6_segment_fuel_model_comparison.csv"
DEFAULT_BERTHS = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step2_berth_centers.csv"
# Written by "nautical chart/enc_route_check.py" (run with the chart venv) from the Step7 candidate paths.
DEFAULT_ENC_CHECK = DEFAULT_OUTPUT_DIR / "enc" / "enc_route_check_summary.csv"
# Net under-keel clearance required after draught, Barrass squat and the low-water allowance
# (all already subtracted in the ENC check). 0.5 m is about 15 % of the 3.4 m draught.
DEFAULT_MIN_UKC_M = 0.5

VALID_DIRECTIONS = ("berth_2_to_berth_1", "berth_1_to_berth_2")
SPEED_STRATEGIES = {
    "eco": (0.96, 0.95, 0.97),
    "balanced": (1.00, 1.00, 1.00),
    "fast": (1.04, 1.03, 1.02),
}
# RPM-setpoint plans (default plan type). eco = the largest evaluated RPM reduction whose passage-time
# increase stays within the 1 min per crossing budget adopted in the revision (uniform -5 %).
RPM_STRATEGIES = {
    "eco": (0.95, 0.95, 0.95),
    "balanced": (1.00, 1.00, 1.00),
    "fast": (1.04, 1.03, 1.02),
}
EPSILON_LEVELS = {
    "strict": 0.35,
    "medium": 0.60,
    "loose": 0.80,
}
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step-8 fuel-model-driven route optimizer using clustered candidate paths."
    )
    parser.add_argument("--step3", type=Path, default=DEFAULT_STEP3)
    parser.add_argument("--step7-paths", type=Path, default=DEFAULT_STEP7_PATHS)
    parser.add_argument("--step7-assign", type=Path, default=DEFAULT_STEP7_ASSIGN)
    parser.add_argument("--step4-interp", type=Path, default=DEFAULT_STEP4_INTERP)
    parser.add_argument("--step6-comparison", type=Path, default=DEFAULT_STEP6_COMPARISON)
    parser.add_argument("--berths", type=Path, default=DEFAULT_BERTHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan-type", default="rpm", choices=["rpm", "speed"],
                        help="rpm: strategies are RPM setpoints (speed from the learned model); speed: as before")
    parser.add_argument("--primary-model", default="lgbm_mono",
                        choices=["gbr_paper", "gbr_norpm", "lgbm_mono", "lgbm_mono_norpm", "greybox"],
                        help="model whose predictions drive the epsilon-constraint selection")
    parser.add_argument("--enc-check", type=Path, default=DEFAULT_ENC_CHECK,
                        help="per-variant chart safety table from nautical chart/enc_route_check.py")
    parser.add_argument("--min-ukc", type=float, default=DEFAULT_MIN_UKC_M,
                        help="minimum net under-keel clearance (m) for a candidate to be chart-feasible")
    parser.add_argument("--no-enc-check", action="store_true",
                        help="skip the chart constraint (all candidates treated as chart-feasible)")
    return parser.parse_args()


def load_chart_feasibility(path: Path, candidate_paths: pd.DataFrame, min_ukc: float) -> pd.DataFrame:
    """Per (direction, cluster, variant) chart-safety flags. Fails if any Step7 variant is missing,
    so a stale ENC table cannot silently pass."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 'nautical chart/enc_route_check.py' with the chart venv "
                                "first, or pass --no-enc-check")
    enc = pd.read_csv(path).rename(columns={"variant": "variant_name"})
    keys = ["direction", "cluster_name", "variant_name"]
    needed = candidate_paths[keys].drop_duplicates()
    merged = needed.merge(enc, on=keys, how="left", indicator=True)
    missing = merged.loc[merged["_merge"].eq("left_only"), keys]
    if len(missing):
        raise ValueError(f"ENC check has no result for these Step7 variants, rerun it:\n{missing.to_string(index=False)}")
    out = merged[keys + ["min_ukc_sounding_m", "min_sounding_m", "corridor_share_lt6m",
                         "clear_land_m", "clear_rock_m", "clear_wreck_m", "clear_lt3m_area_m"]].copy()
    # no sounding inside the swept corridor -> cannot verify -> not feasible
    out["chart_feasible"] = out["min_ukc_sounding_m"].ge(min_ukc).fillna(False)
    return out


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


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def train_best_segment_model(step3_path: Path, comparison_path: Path):
    step3 = load_step3(step3_path)
    segments = build_main_voyage_segments(step3)
    comparison = pd.read_csv(comparison_path)
    best_model_name = str(comparison.sort_values("mae").iloc[0]["model"])

    feature_cols = [
        "direction",
        "segment_duration_s",
        "segment_distance_nm",
        "segment_progress_mean",
        "speed_mean_knots",
        "speed_std_knots",
        "accel_abs_mean",
        "heading_change_abs_sum",
        "turn_rate_abs_mean",
        "turn_rate_peak",
        "curvature_proxy",
        "lat_acc_mean",
        "lat_acc_peak",
        "rpm_mean_seg",
        "rpm_max_seg",
        "active_engine_mean",
        "nearest_berth_dist_mean_nm",
        "no_sog_ratio",
    ]
    feature_cols = [c for c in feature_cols if not segments[c].isna().all()]
    numeric_features = [c for c in feature_cols if c != "direction"]
    categorical_features = ["direction"]

    models = build_model_objects(numeric_features, categorical_features)
    model = models[best_model_name]
    model.fit(segments[feature_cols], segments["fuel_segment"].astype(float))
    return model, feature_cols, step3


def build_cluster_reference(step3: pd.DataFrame, assignments: pd.DataFrame, interp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_interp = interp.merge(
        assignments[["direction", "voyage_id", "cluster_name"]],
        on=["direction", "voyage_id"],
        how="left",
    )
    def positive_mean(series: pd.Series) -> float:
        positive = series.loc[series > 0.2]
        if positive.empty:
            return float(series.mean())
        return float(positive.mean())

    def positive_std(series: pd.Series) -> float:
        positive = series.loc[series > 0.2]
        if len(positive) <= 1:
            return float(series.std())
        return float(positive.std())

    tau_ref = (
        ref_interp.groupby(["direction", "cluster_name", "tau"])
        .agg(
            sog_mean=("SOG_interp", positive_mean),
            sog_std=("SOG_interp", positive_std),
            rpm_mean=("rpm_interp", "mean"),
            rpm_std=("rpm_interp", "std"),
            turn_rate_abs_mean=("turn_rate_interp", lambda s: float(np.mean(np.abs(s)))),
            turn_rate_peak=("turn_rate_interp", lambda s: float(np.max(np.abs(s)))),
        )
        .reset_index()
    )

    cluster_lookup = assignments[["direction", "voyage_id", "cluster_name"]].copy()
    step3m = step3.merge(cluster_lookup, on=["direction", "voyage_id"], how="inner")
    main = step3m.loc[step3m["segment_label"] == "main_voyage"].copy()
    cluster_static = (
        main.groupby(["direction", "cluster_name"])
        .agg(
            active_engine_mean=("active_engine_count", "mean"),
            power_mean=("power_total", "mean"),
            power_max=("power_total", "max"),
        )
        .reset_index()
    )
    return tau_ref, cluster_static


def assign_reference_bins(candidate_group: pd.DataFrame, tau_ref: pd.DataFrame, cluster_static: pd.DataFrame, speed_strategy: tuple[float, float, float], berth_centers: pd.DataFrame) -> pd.DataFrame:
    g = candidate_group.sort_values("tau").reset_index(drop=True).copy()
    g["bin_id"] = pd.cut(g["tau"], bins=np.linspace(0, 1, 4), labels=[1, 2, 3], include_lowest=True).astype(int)
    factors = {1: speed_strategy[0], 2: speed_strategy[1], 3: speed_strategy[2]}
    g["speed_factor"] = g["bin_id"].map(factors)

    ref = tau_ref.loc[
        (tau_ref["direction"] == g["direction"].iloc[0]) & (tau_ref["cluster_name"] == g["cluster_name"].iloc[0])
    ].copy()
    merged = g.merge(ref, on=["direction", "cluster_name", "tau"], how="left")
    static = cluster_static.loc[
        (cluster_static["direction"] == g["direction"].iloc[0]) & (cluster_static["cluster_name"] == g["cluster_name"].iloc[0])
    ].iloc[0]

    merged["sog_mean"] = merged["sog_mean"].interpolate(limit_direction="both").bfill().ffill().fillna(8.0)
    merged["sog_std"] = merged["sog_std"].fillna(0.15)
    merged["rpm_mean"] = merged["rpm_mean"].interpolate(limit_direction="both").bfill().ffill().fillna(1400.0)
    merged["rpm_std"] = merged["rpm_std"].fillna(20.0)
    merged["turn_rate_abs_mean"] = merged["turn_rate_abs_mean"].fillna(0.2)
    merged["turn_rate_peak"] = merged["turn_rate_peak"].fillna(0.4)
    merged["speed_mean_knots_pt"] = merged["sog_mean"] * merged["speed_factor"]
    merged["rpm_mean_pt"] = merged["rpm_mean"] * (merged["speed_factor"] ** 1.15)
    merged["rpm_max_pt"] = (merged["rpm_mean"] + merged["rpm_std"].fillna(0.0)) * (merged["speed_factor"] ** 1.20)
    merged["active_engine_mean_pt"] = float(static["active_engine_mean"])
    merged["power_mean_pt"] = float(static["power_mean"]) * (merged["speed_factor"] ** 1.3)
    merged["power_max_pt"] = float(static["power_max"]) * (merged["speed_factor"] ** 1.35)

    berth_lats = berth_centers["center_lat"].to_numpy(dtype=float)
    berth_lons = berth_centers["center_lon"].to_numpy(dtype=float)
    d1 = haversine_nm(merged["path_lat"], merged["path_lon"], berth_lats[0], berth_lons[0])
    d2 = haversine_nm(merged["path_lat"], merged["path_lon"], berth_lats[1], berth_lons[1])
    merged["nearest_berth_dist_nm_pt"] = np.minimum(d1, d2)
    return merged


def build_candidate_segments(merged_pts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = merged_pts.sort_values("tau").reset_index(drop=True).copy()
    lat = g["path_lat"].to_numpy(dtype=float)
    lon = g["path_lon"].to_numpy(dtype=float)
    bearings = np.array([bearing_deg(lat[i], lon[i], lat[i + 1], lon[i + 1]) for i in range(len(g) - 1)])
    delta = np.zeros(len(g) - 1, dtype=float)
    for i in range(1, len(bearings)):
        delta[i] = abs(angle_diff_deg(bearings[i - 1], bearings[i]))

    rows = []
    M1 = float(np.sum(delta))
    M2 = float(np.max(delta)) if len(delta) else 0.0
    turn_count = int(np.sum(delta >= 5.0))
    path_distance = 0.0
    dynamic_terms = []
    curvature_terms = []

    for i in range(len(g) - 1):
        dist_nm = float(haversine_nm(lat[i], lon[i], lat[i + 1], lon[i + 1]))
        path_distance += dist_nm
        speed_mean = max(0.2, float((g.loc[i, "speed_mean_knots_pt"] + g.loc[i + 1, "speed_mean_knots_pt"]) / 2.0))
        duration_s = max(5.0, dist_nm / speed_mean * 3600.0)
        speed_std = float((g.loc[i, "sog_std"] + g.loc[i + 1, "sog_std"]) / 2.0)
        accel_abs = abs(float(g.loc[i + 1, "speed_mean_knots_pt"] - g.loc[i, "speed_mean_knots_pt"])) / duration_s
        heading_change = float(delta[i])
        turn_rate = heading_change / duration_s
        curvature = heading_change / max(dist_nm, 1e-6)
        speed_ms = speed_mean * 0.514444
        yaw_rate_rad = np.radians(turn_rate)
        lat_acc = abs(speed_ms * yaw_rate_rad)
        dynamic_terms.append((speed_ms**2) * turn_rate)
        curvature_terms.append(curvature)

        rows.append(
            {
                "segment_id": i + 1,
                "direction": str(g.loc[i, "direction"]),
                "segment_duration_s": duration_s,
                "segment_distance_nm": dist_nm,
                "segment_progress_mean": float((g.loc[i, "tau"] + g.loc[i + 1, "tau"]) / 2.0),
                "speed_mean_knots": speed_mean,
                "speed_std_knots": speed_std,
                "accel_abs_mean": accel_abs,
                "heading_change_abs_sum": heading_change,
                "turn_rate_abs_mean": turn_rate,
                "turn_rate_peak": max(turn_rate, float(g.loc[i, "turn_rate_peak"]), float(g.loc[i + 1, "turn_rate_peak"])),
                "curvature_proxy": curvature,
                "lat_acc_mean": lat_acc,
                "lat_acc_peak": lat_acc,
                "rpm_mean_seg": float((g.loc[i, "rpm_mean_pt"] + g.loc[i + 1, "rpm_mean_pt"]) / 2.0),
                "rpm_max_seg": float(max(g.loc[i, "rpm_max_pt"], g.loc[i + 1, "rpm_max_pt"])),
                "power_mean_seg": float((g.loc[i, "power_mean_pt"] + g.loc[i + 1, "power_mean_pt"]) / 2.0),
                "power_max_seg": float(max(g.loc[i, "power_max_pt"], g.loc[i + 1, "power_max_pt"])),
                "active_engine_mean": float((g.loc[i, "active_engine_mean_pt"] + g.loc[i + 1, "active_engine_mean_pt"]) / 2.0),
                "nearest_berth_dist_mean_nm": float((g.loc[i, "nearest_berth_dist_nm_pt"] + g.loc[i + 1, "nearest_berth_dist_nm_pt"]) / 2.0),
                "no_sog_ratio": 0.0,
            }
        )

    beeline_nm = float(haversine_nm(lat[0], lon[0], lat[-1], lon[-1]))
    M3 = float(np.sum(curvature_terms))
    M4 = float(turn_count / max(path_distance, 1e-6))
    M5 = float(np.sum(dynamic_terms))
    metrics = {
        "path_distance_nm": float(path_distance),
        "beeline_distance_nm": beeline_nm,
        "tortuosity_path": float(path_distance / beeline_nm) if beeline_nm > 0 else np.nan,
        "M1_cum_heading_change": M1,
        "M2_max_heading_change": M2,
        "M3_curvature_sum": M3,
        "M4_turn_frequency": M4,
        "M5_dynamic_turn_load": M5,
        "turn_count": turn_count,
    }
    return pd.DataFrame(rows), metrics


def minmax_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    smin = float(s.min())
    smax = float(s.max())
    if np.isclose(smin, smax):
        return pd.Series(0.0, index=s.index)
    return (s - smin) / (smax - smin)


def pareto_efficient(costs: np.ndarray) -> np.ndarray:
    n = costs.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        if np.any(dominates):
            efficient[i] = False
    return efficient


def evaluate_candidates(model, feature_cols: list[str], candidate_paths: pd.DataFrame, tau_ref: pd.DataFrame, cluster_static: pd.DataFrame, berth_centers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, cluster_name, variant_name), group in candidate_paths.groupby(["direction", "cluster_name", "variant_name"]):
        for speed_name, speed_strategy in SPEED_STRATEGIES.items():
            pts = assign_reference_bins(group, tau_ref, cluster_static, speed_strategy, berth_centers)
            segs, metrics = build_candidate_segments(pts)
            X_pred = segs[feature_cols].copy()
            fuel_pred = model.predict(X_pred)
            fuel_total = float(np.sum(fuel_pred))
            time_total = float(segs["segment_duration_s"].sum() / 60.0)
            rpm_peak = float(segs["rpm_max_seg"].max())

            row = {
                "direction": direction,
                "cluster_name": cluster_name,
                "variant_name": variant_name,
                "speed_strategy": speed_name,
                "fuel_total_pred": fuel_total,
                "time_total_min": time_total,
                "rpm_peak": rpm_peak,
                **metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def apply_joint_selection(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out_parts = []
    summary: dict[str, dict] = {}
    m_cols = ["M1_cum_heading_change", "M2_max_heading_change", "M3_curvature_sum", "M4_turn_frequency", "M5_dynamic_turn_load"]
    for direction in VALID_DIRECTIONS:
        sub = candidates.loc[candidates["direction"] == direction].copy().reset_index(drop=True)
        for col in m_cols:
            sub[f"{col}_norm"] = minmax_scale(sub[col])
        sub["maneuver_cost"] = sum(0.2 * sub[f"{col}_norm"] for col in m_cols)
        sub["fuel_metric"] = minmax_scale(sub["fuel_total_pred"])
        sub["time_metric"] = minmax_scale(sub["time_total_min"])
        sub["maneuver_metric"] = sub["maneuver_cost"]
        # chart safety is a hard constraint; normalisation and epsilon quantiles stay on the full
        # candidate set so the manoeuvring thresholds are defined the same way with or without the chart
        if "chart_feasible" not in sub:
            sub["chart_feasible"] = True
        chart_ok = sub["chart_feasible"].astype(bool)
        sub["pareto_efficient"] = False
        sub.loc[chart_ok, "pareto_efficient"] = pareto_efficient(
            sub.loc[chart_ok, ["fuel_metric", "maneuver_metric", "time_metric"]].to_numpy(dtype=float))

        direction_summary = {}
        for level, quantile in EPSILON_LEVELS.items():
            eps = float(sub["maneuver_cost"].quantile(quantile))
            feasible = sub.loc[(sub["maneuver_cost"] <= eps) & chart_ok].copy()
            feasible = feasible.sort_values(["fuel_total_pred", "time_total_min"]).reset_index(drop=True)
            best = feasible.iloc[0]
            med = feasible.median(numeric_only=True)
            direction_summary[level] = {
                "epsilon_quantile": quantile,
                "maneuver_epsilon": eps,
                "selected_cluster_name": str(best["cluster_name"]),
                "selected_variant_name": str(best["variant_name"]),
                "selected_speed_strategy": str(best["speed_strategy"]),
                "selected_fuel_total_pred": float(best["fuel_total_pred"]),
                "selected_time_total_min": float(best["time_total_min"]),
                "selected_maneuver_cost": float(best["maneuver_cost"]),
                "n_maneuver_feasible": int((sub["maneuver_cost"] <= eps).sum()),
                "n_removed_by_chart": int(((sub["maneuver_cost"] <= eps) & ~chart_ok).sum()),
                "n_feasible": int(len(feasible)),
                "fuel_improvement_vs_feasible_median_pct": float((med["fuel_total_pred"] - best["fuel_total_pred"]) / med["fuel_total_pred"] * 100.0),
                "time_improvement_vs_feasible_median_pct": float((med["time_total_min"] - best["time_total_min"]) / med["time_total_min"] * 100.0),
            }
            sub[f"is_selected_{level}"] = (
                (sub["cluster_name"] == best["cluster_name"])
                & (sub["variant_name"] == best["variant_name"])
                & (sub["speed_strategy"] == best["speed_strategy"])
            )
            sub[f"is_feasible_{level}"] = (sub["maneuver_cost"] <= eps) & chart_ok
        summary[direction] = direction_summary
        out_parts.append(sub)
    return pd.concat(out_parts, ignore_index=True), summary


def engine_spread_by_tau(step3: pd.DataFrame, voyage_ids: set[int]) -> pd.DataFrame:
    """Median (rpm_max - rpm_mean) along the main voyage; the synthetic trajectory needs a realistic engine spread."""
    m = step3.loc[step3["segment_label"].eq("main_voyage") & step3["voyage_id"].isin(list(voyage_ids)).fillna(False),
                  ["voyage_id", "delta_t_seconds", "rpm_mean", "rpm_max"]].copy()
    dt = m["delta_t_seconds"].fillna(1.0)
    m["tau"] = (dt.groupby(m["voyage_id"]).cumsum() - 0.5 * dt) / dt.groupby(m["voyage_id"]).transform("sum")
    m["spread"] = m["rpm_max"] - m["rpm_mean"]
    m["tau_bin"] = np.clip((m["tau"] * 60).round() / 60, 0, 1)
    return m.groupby("tau_bin")["spread"].median().rename_axis("tau").reset_index()


def synthesize_trajectory(path: pd.DataFrame, tau_ref: pd.DataFrame, spread: pd.DataFrame, berths: pd.DataFrame,
                          factors: tuple[float, float, float], rpm_exponent: float, active_engines: float,
                          plan_id: int, speed_fn=None, rpm_fn=None) -> pd.DataFrame:
    """1 Hz main-voyage records along a candidate path, in the format the Step6 window builder expects.

    Speed and RPM follow the cluster's median profile (tau_ref) mapped to distance along the path;
    the speed plan multiplies speed by f per third and RPM by f**rpm_exponent; time follows from distance.
    For RPM-setpoint plans, speed_fn(u) and rpm_fn(u) give the speed and RPM factors along the path
    (u = distance fraction) and override factors/rpm_exponent."""
    from scipy.interpolate import CubicSpline

    p = path.sort_values("tau")
    lat, lon = p["path_lat"].to_numpy(float), p["path_lon"].to_numpy(float)
    lat0 = float(lat.mean())
    x = np.radians(lon - lon[0]) * 6371000.0 * np.cos(np.radians(lat0))
    y = np.radians(lat - lat[0]) * 6371000.0
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    s, x, y, tau_pts = s[keep], x[keep], y[keep], p["tau"].to_numpy(float)[keep]
    L = float(s[-1])
    sx, sy = CubicSpline(s, x), CubicSpline(s, y)
    ref = tau_ref.sort_values("tau")
    u_of_tau = s / L                                     # balanced: time progress tau -> distance fraction u
    tau_of_u = lambda u: np.interp(u, u_of_tau, tau_pts)  # noqa: E731
    v_bal = lambda u: np.interp(tau_of_u(u), ref["tau"], ref["sog_mean"])  # noqa: E731
    r_bal = lambda u: np.interp(tau_of_u(u), ref["tau"], ref["rpm_mean"])  # noqa: E731
    sp = lambda u: np.interp(tau_of_u(u), spread["tau"], spread["spread"])  # noqa: E731
    u_edges = np.interp([1 / 3, 2 / 3], tau_pts, u_of_tau)

    pos, rows = 0.0, []
    while pos < L and len(rows) < 20000:
        u = pos / L
        if speed_fn is None:
            f = factors[int(np.searchsorted(u_edges, u, side="right"))]
            fs, fr = f, f ** rpm_exponent
        else:
            fs, fr = float(speed_fn(u)), float(rpm_fn(u))
        v = max(float(v_bal(u)) * fs, 0.5)
        step = v * 1852.0 / 3600.0
        rows.append((pos, v, float(r_bal(u)) * fr, float(sp(u)) * fr, min(step, L - pos)))
        pos += step
    r = pd.DataFrame(rows, columns=["s", "speed_used_knots", "rpm_mean", "spread", "step_m"])
    r["path_u"] = r["s"] / L
    hx, hy = sx(r["s"], 1), sy(r["s"], 1)
    heading = np.degrees(np.arctan2(hx, hy)) % 360.0
    cog_delta = np.concatenate([[0.0], (np.diff(heading) + 180.0) % 360.0 - 180.0])
    px, py = sx(r["s"]), sy(r["s"])
    plat = lat[0] + np.degrees(py / 6371000.0)
    plon = lon[0] + np.degrees(px / (6371000.0 * np.cos(np.radians(lat0))))
    bl, bo = berths["center_lat"].to_numpy(float), berths["center_lon"].to_numpy(float)
    near = np.minimum(haversine_nm(plat, plon, bl[0], bo[0]), haversine_nm(plat, plon, bl[1], bo[1]))
    n = len(r)
    return pd.DataFrame({
        "time": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(np.arange(n), unit="s"),
        "voyage_id": plan_id, "segment_label": "main_voyage", "direction": str(p["direction"].iloc[0]),
        "training_voyage": True, "delta_t_seconds": 1.0, "step_distance_nm": r["step_m"] / 1852.0,
        "speed_used_knots": r["speed_used_knots"], "accel_knots_per_s": np.concatenate([[0.0], np.diff(r["speed_used_knots"])]),
        "cog_delta_deg": cog_delta, "turn_rate_deg_per_s": cog_delta, "rpm_mean": r["rpm_mean"],
        "rpm_max": r["rpm_mean"] + r["spread"], "power_total": 0.0, "active_engine_count": active_engines,
        "nearest_berth_distance_nm": near, "no_sog_engine_flag": 0.0, "fuel_step_est": 0.0,
        "path_u": r["path_u"], "u_edge_1": u_edges[0], "u_edge_2": u_edges[1],
    })


def rpm_setpoint_functions(group, tau_ref, spread, berths, rpm_factors, active_engines, env, h_bundle, plan_id):
    """Speed response to an RPM setpoint plan, from the learned speed model h (Step6c).

    A balanced trajectory is windowed; each window's speed factor is h(RPM*r)/h(RPM); factors are
    interpolated along the path (distance fraction u) and returned with the RPM factor function."""
    import fuel_models as fm

    traj = synthesize_trajectory(group, tau_ref, spread, berths, (1.0, 1.0, 1.0), 0.0, active_engines, plan_id)
    w = fm.build_windows(traj, env)
    wid = np.floor((traj["delta_t_seconds"].cumsum() - traj["delta_t_seconds"].iloc[0]) / 30.0).astype(int)
    u_w = traj.groupby(wid)["path_u"].mean().reindex(w["segment_window_id"]).to_numpy()
    e1, e2 = float(traj["u_edge_1"].iloc[0]), float(traj["u_edge_2"].iloc[0])
    third = lambda u: np.searchsorted([e1, e2], u, side="right")  # noqa: E731
    r_w = np.asarray(rpm_factors, float)[third(u_w)]
    h, feats = h_bundle["h_speed"], h_bundle["H_FEATURES"]
    x = w.copy()
    x["rpm_mean_seg"] *= r_w
    s_w = np.clip(h.predict(x[feats]) / h.predict(w[feats]), 0.8, 1.2)
    order = np.argsort(u_w)
    speed_fn = lambda u: np.interp(u, u_w[order], s_w[order])  # noqa: E731
    rpm_fn = lambda u: float(np.asarray(rpm_factors, float)[third(u)])  # noqa: E731
    thirds = tuple(float(s_w[third(u_w) == k].mean()) for k in range(3))
    return speed_fn, rpm_fn, thirds


def evaluate_candidates_windows(models: dict, meta: dict, candidate_paths: pd.DataFrame, tau_ref: pd.DataFrame,
                                cluster_static: pd.DataFrame, berth_centers: pd.DataFrame, spread: pd.DataFrame,
                                rpm_rules: dict[str, float], plan_type: str = "rpm", h_bundle=None,
                                strategies: dict | None = None, q_time: float = 0.0) -> pd.DataFrame:
    """Score each route variant x plan with every model on 30 s windows.

    plan_type 'rpm'  : factors are RPM setpoints; speed follows from the learned speed model h.
    plan_type 'speed': factors are speed; RPM follows f**exponent for each entry of rpm_rules."""
    import fuel_models as fm

    strategies = strategies or SPEED_STRATEGIES
    rows, plan_id = [], 0
    for (direction, cluster_name, variant_name), group in candidate_paths.groupby(["direction", "cluster_name", "variant_name"]):
        env = meta["env_typical_median"][direction]
        static = cluster_static.loc[(cluster_static["direction"] == direction) & (cluster_static["cluster_name"] == cluster_name)]
        engines = float(static["active_engine_mean"].iloc[0])
        for speed_name, factors in strategies.items():
            fns, speed_thirds = (None, None), factors
            if plan_type == "rpm":
                plan_id += 1
                *fns, speed_thirds = rpm_setpoint_functions(group, tau_ref, spread, berth_centers, factors, engines, env,
                                                            h_bundle, plan_id)
            # geometry indicators M1-M5 keep their original definitions (with the realised speed factors)
            _, metrics = build_candidate_segments(assign_reference_bins(group, tau_ref, cluster_static, speed_thirds, berth_centers))
            for rule, x in (rpm_rules.items() if plan_type == "speed" else [("rpm_setpoint", 0.0)]):
                plan_id += 1
                traj = synthesize_trajectory(group, tau_ref, spread, berth_centers, factors, x, engines, plan_id,
                                             speed_fn=fns[0], rpm_fn=fns[1])
                w = fm.build_windows(traj, env)
                # passage time from the 1 Hz trajectory itself (window sums drop the last partial window)
                minutes = float(traj["delta_t_seconds"].sum() / 60.0)
                # approach speed handed to the arrival model: last 60 s of the main voyage
                end_speed = float(traj["speed_used_knots"].tail(60).mean())
                for name, model in models.items():
                    if plan_type == "speed" and not fm.USES_RPM[name] and rule != next(iter(rpm_rules)):
                        continue  # RPM rule is irrelevant for models without RPM
                    fuel = float(model.predict(w).sum())
                    q = float(meta["conformal_q_rel"][name])
                    rows.append({"direction": direction, "cluster_name": cluster_name, "variant_name": variant_name,
                                 "speed_strategy": speed_name, "plan_type": plan_type, "plan_factors": str(factors),
                                 "speed_factor_thirds": str(tuple(round(v, 4) for v in speed_thirds)), "model": name,
                                 "rpm_rule": rule if (fm.USES_RPM[name] or plan_type == "rpm") else "not_used",
                                 "fuel_total_pred": fuel, "fuel_pi90_lo": fuel * (1 - q), "fuel_pi90_hi": fuel * (1 + q),
                                 "time_total_min": minutes, "end_speed_kn": end_speed,
                                 "time_pi90_lo": minutes * (1 - q_time),
                                 "time_pi90_hi": minutes * (1 + q_time), "windows": int(len(w)),
                                 "window_s_median": float(w["segment_duration_s"].median()),
                                 "rpm_peak": float(w["rpm_max_seg"].max()), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fuel_models as fm

    models, meta = fm.load_bundle()  # fitted by build_step6b_model_comparison.py
    step3 = load_step3(args.step3)
    candidate_paths = pd.read_csv(args.step7_paths)
    assignments = pd.read_csv(args.step7_assign)
    interp = pd.read_csv(args.step4_interp)
    berth_centers = pd.read_csv(args.berths)

    tau_ref, cluster_static = build_cluster_reference(step3, assignments, interp)
    typical = set(assignments.loc[assignments["direction"].isin(VALID_DIRECTIONS), "voyage_id"].astype(int))
    spread = engine_spread_by_tau(step3, typical)
    beta = float(meta["rpm_exponent"]["voyage_level"]["exponent"])
    rpm_rules = {f"data_f^{beta:.2f}": beta, f"paper_f^{meta['paper_rpm_exponent']}": float(meta["paper_rpm_exponent"])}
    h_bundle, q_time = None, 0.0
    if args.plan_type == "rpm":
        import joblib

        h_bundle = joblib.load(fm.MODEL_DIR / "rpm_speed_models.joblib")  # fitted by build_step6c_rpm_models.py
        rpm_summary = json.loads((fm.MODEL_DIR / "rpm_models_summary.json").read_text(encoding="utf-8"))
        q_time = float(rpm_summary["h_time_closure_test"]["conformal90_halfwidth_pct"]) / 100.0
        rpm_rules = {"rpm_setpoint": 0.0}
    long = evaluate_candidates_windows(models, meta, candidate_paths, tau_ref, cluster_static, berth_centers, spread,
                                       rpm_rules, plan_type=args.plan_type, h_bundle=h_bundle, q_time=q_time,
                                       strategies=RPM_STRATEGIES if args.plan_type == "rpm" else SPEED_STRATEGIES)
    keys = ["direction", "cluster_name", "variant_name"]
    if args.no_enc_check:
        chart = candidate_paths[keys].drop_duplicates().assign(chart_feasible=True)
    else:
        chart = load_chart_feasibility(args.enc_check, candidate_paths, args.min_ukc)
    long = long.merge(chart, on=keys, how="left")
    chart_info = {
        "applied": not args.no_enc_check,
        "source": None if args.no_enc_check else str(args.enc_check),
        "min_ukc_m": None if args.no_enc_check else args.min_ukc,
        "rule": "min UKC over soundings/hazard VALSOU in the swept corridor (half beam + margin) after draught, "
                "Barrass squat and low-water allowance >= min_ukc_m; see nautical chart/enc_route_check.py",
        "variants": chart.round(3).to_dict(orient="records"),
        "removed_variants": chart.loc[~chart["chart_feasible"], "variant_name"].tolist(),
    }
    print("chart constraint:", "off" if args.no_enc_check else
          f"min UKC {args.min_ukc} m, removed {chart_info['removed_variants'] or 'none'}")
    primary = (long["model"].eq(args.primary_model)
               & long["rpm_rule"].isin(["not_used", next(iter(rpm_rules))]))
    candidate_eval = long.loc[primary].drop(columns=["model", "rpm_rule"]).reset_index(drop=True)
    scored, summary = apply_joint_selection(candidate_eval)

    # per-model view: balanced vs eco vs fast on the centreline, and each model's selection
    by_model = {}
    for (name, rule), sub in long.groupby(["model", "rpm_rule"]):
        _, s = apply_joint_selection(sub.drop(columns=["model", "rpm_rule"]).reset_index(drop=True))
        for d in VALID_DIRECTIONS:
            sd = sub.loc[sub["direction"].eq(d)]
            cl = sd.loc[sd["variant_name"].eq("centerline")].groupby("speed_strategy").first()
            by_model[f"{name}|{rule}|{d}"] = {
                "centerline_fuel_L": cl["fuel_total_pred"].round(3).to_dict(),
                "centerline_minutes": cl["time_total_min"].round(3).to_dict(),
                "eco_vs_balanced_pct": float((cl.loc["eco", "fuel_total_pred"] / cl.loc["balanced", "fuel_total_pred"] - 1) * 100),
                "fast_vs_balanced_pct": float((cl.loc["fast", "fuel_total_pred"] / cl.loc["balanced", "fuel_total_pred"] - 1) * 100),
                "best_route_gain_at_eco_L": float(cl.loc["eco", "fuel_total_pred"]
                                                  - sd.loc[sd["speed_strategy"].eq("eco") & sd["chart_feasible"],
                                                           "fuel_total_pred"].min()),
                "selected_medium": {k: s[d]["medium"][k] for k in
                                    ("selected_variant_name", "selected_speed_strategy", "selected_fuel_total_pred",
                                     "selected_time_total_min")},
            }
    step4 = pd.read_csv(args.output_dir / f"{args.step3.stem.replace('_step3_enriched', '')}_step4_main_route_candidates.csv")
    hist_main = step4.groupby("direction")["fuel_main_voyage"].median()
    cl_bal = (candidate_eval.loc[candidate_eval["variant_name"].eq("centerline")
                                 & candidate_eval["speed_strategy"].eq("balanced")].groupby("direction").first())

    dataset_base = args.step3.stem.replace("_step3_enriched", "")
    base = f"{dataset_base}_step8_route_optimization"
    eval_csv = args.output_dir / f"{base}_candidates.csv"
    long_csv = args.output_dir / f"{base}_candidates_all_models.csv"
    tau_ref_csv = args.output_dir / f"{base}_cluster_tau_reference.csv"
    summary_json = args.output_dir / f"{base}_summary.json"

    scored.to_csv(eval_csv, index=False, encoding="utf-8-sig")
    long.to_csv(long_csv, index=False, encoding="utf-8-sig")
    if args.plan_type == "rpm":
        # trade-off curve: uniform RPM setpoints on the centreline and on the selected route
        sweep = {f"rpm_{(x - 1) * 100:+.0f}%": (x, x, x) for x in (0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03)}
        parts = []
        for d in VALID_DIRECTIONS:
            sel = summary[d]["medium"]
            paths_sel = candidate_paths.loc[candidate_paths["direction"].eq(d)
                                            & candidate_paths["cluster_name"].eq(sel["selected_cluster_name"])
                                            & candidate_paths["variant_name"].isin(["centerline", sel["selected_variant_name"]])]
            parts.append(evaluate_candidates_windows(models, meta, paths_sel, tau_ref, cluster_static, berth_centers,
                                                     spread, rpm_rules, plan_type="rpm", h_bundle=h_bundle,
                                                     strategies=sweep, q_time=q_time))
        trade = pd.concat(parts, ignore_index=True)
        ref = (trade.loc[trade["variant_name"].eq("centerline") & trade["speed_strategy"].eq("rpm_+0%")]
               .set_index(["direction", "model"]))
        key = pd.MultiIndex.from_frame(trade[["direction", "model"]])
        trade["main_saving_vs_centerline_rpm0_pct"] = (1 - trade["fuel_total_pred"].to_numpy()
                                                       / ref["fuel_total_pred"].reindex(key).to_numpy()) * 100
        trade["extra_minutes_vs_centerline_rpm0"] = (trade["time_total_min"].to_numpy()
                                                     - ref["time_total_min"].reindex(key).to_numpy())
        trade.to_csv(args.output_dir / f"{base}_rpm_tradeoff.csv", index=False, encoding="utf-8-sig")
    tau_ref.to_csv(tau_ref_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(
            {
                "best_segment_model": args.primary_model,
                "primary_rpm_rule": next(iter(rpm_rules)) if fm.USES_RPM[args.primary_model] else "not_used",
                "evaluation": "synthetic 1 Hz trajectory along each candidate path, 30 s windows as in Step6",
                "plan_type": args.plan_type,
                "plan_definition": ("RPM setpoint factors per third (eco 0.95/0.95/0.95 = largest step within a "
                                    "1 min passage-time budget, fast 1.04/1.03/1.02); speed from the learned speed "
                                    "model h" if args.plan_type == "rpm" else
                                    "speed factors per third; RPM from f**exponent"),
                "time_pi90_halfwidth_pct": q_time * 100,
                "speed_strategies": list(SPEED_STRATEGIES.keys()),
                "epsilon_levels": EPSILON_LEVELS,
                "chart_constraint": chart_info,
                "historical_main_fuel_median_L": hist_main.to_dict(),
                "balanced_centerline_vs_historical_median_pct": ((cl_bal["fuel_total_pred"] / hist_main - 1) * 100).dropna().to_dict(),
                "balanced_centerline_minutes": cl_bal["time_total_min"].to_dict(),
                "selected_routes": summary,
                "by_model": by_model,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"candidates_csv={eval_csv}")
    print(f"tau_reference_csv={tau_ref_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
