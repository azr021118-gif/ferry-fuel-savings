from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


INPUT = Path("outputs/yxlan_20250701_20251130_step6_segment_fuel_model_dataset.csv")
OUT_DIR = Path("outputs")
CHART_DIR = OUT_DIR / "charts"

SUMMARY_CSV = OUT_DIR / "yxlan_maneuvering_indicator_sensitivity_summary.csv"
ABLATION_CSV = OUT_DIR / "yxlan_maneuvering_indicator_ablation.csv"
FIG_PATH = CHART_DIR / "yxlan_maneuvering_indicator_sensitivity.png"

RANDOM_STATE = 42
MAX_ROWS_FOR_MODEL = 25000


BASE_FEATURES = [
    "segment_duration_s",
    "segment_distance_nm",
    "segment_progress_mean",
    "speed_mean_knots",
    "speed_std_knots",
    "rpm_mean_seg",
    "rpm_max_seg",
    "nearest_berth_dist_mean_nm",
    "active_engine_mean",
]


MANEUVER_FEATURES = [
    "heading_change_abs_sum",
    "turn_rate_abs_mean",
    "turn_rate_peak",
    "curvature_proxy",
    "lat_acc_mean",
    "lat_acc_peak",
    "accel_abs_mean",
    "heading_change_per_nm",
    "turn_dynamic_load_mean",
    "turn_dynamic_load_peak",
    "speed_variability_ratio",
]


DISPLAY_NAMES = {
    "heading_change_abs_sum": "Accum. heading change",
    "turn_rate_abs_mean": "Mean turning rate",
    "turn_rate_peak": "Peak turning rate",
    "curvature_proxy": "Curvature proxy",
    "lat_acc_mean": "Mean lateral acc.",
    "lat_acc_peak": "Peak lateral acc.",
    "accel_abs_mean": "Mean accel.",
    "heading_change_per_nm": "Heading change per nm",
    "turn_dynamic_load_mean": "Mean dynamic turn load",
    "turn_dynamic_load_peak": "Peak dynamic turn load",
    "speed_variability_ratio": "Speed variability ratio",
}


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["fuel_segment", "fuel_rate_segment", "voyage_id"])

    df["heading_change_per_nm"] = df["heading_change_abs_sum"] / df["segment_distance_nm"].clip(lower=1e-6)
    df["turn_dynamic_load_mean"] = (df["speed_mean_knots"] ** 2) * df["turn_rate_abs_mean"]
    df["turn_dynamic_load_peak"] = (df["speed_mean_knots"] ** 2) * df["turn_rate_peak"]
    df["speed_variability_ratio"] = df["speed_std_knots"] / df["speed_mean_knots"].clip(lower=0.1)

    for col in BASE_FEATURES + MANEUVER_FEATURES + ["fuel_segment", "fuel_rate_segment"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Winsorized copies reduce the impact of implausible turning spikes during sensitivity analysis.
    for col in MANEUVER_FEATURES:
        df[f"{col}_w"] = winsorize(df[col].fillna(0.0))
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    high_threshold = df["fuel_rate_segment"].quantile(0.90)
    high = df["fuel_rate_segment"] > high_threshold
    for col in MANEUVER_FEATURES:
        x = df[f"{col}_w"]
        y = df["fuel_rate_segment"]
        rho, p = spearmanr(x, y, nan_policy="omit")
        q90 = x.quantile(0.90)
        rows.append(
            {
                "feature": col,
                "display_name": DISPLAY_NAMES[col],
                "spearman_rho_fuel_rate": float(rho),
                "spearman_p_value": float(p),
                "normal_mean": float(x.loc[~high].mean()),
                "high_fuel_mean": float(x.loc[high].mean()),
                "high_to_normal_ratio": float(x.loc[high].mean() / max(x.loc[~high].mean(), 1e-12)),
                "feature_90th_percentile": float(q90),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_rho_fuel_rate", key=lambda s: s.abs(), ascending=False)


def evaluate_model(df: pd.DataFrame, features: list[str], target: str = "fuel_rate_segment") -> dict[str, float]:
    use = df.dropna(subset=features + [target, "voyage_id"]).copy()
    if len(use) > MAX_ROWS_FOR_MODEL:
        per_voyage = max(8, int(MAX_ROWS_FOR_MODEL / use["voyage_id"].nunique()))
        sampled = use.groupby("voyage_id", group_keys=False).apply(
            lambda g: g.sample(min(len(g), per_voyage), random_state=RANDOM_STATE)
        )
        use = sampled.sample(min(MAX_ROWS_FOR_MODEL, len(sampled)), random_state=RANDOM_STATE).copy()
    X = use[features].astype(float)
    y = use[target].astype(float)
    groups = use["voyage_id"].astype(int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    return {
        "mae": float(mean_absolute_error(y.iloc[test_idx], pred)),
        "r2": float(r2_score(y.iloc[test_idx], pred)),
        "n_features": len(features),
        "n_test": int(len(test_idx)),
        "n_train_voyages": int(groups.iloc[train_idx].nunique()),
        "n_test_voyages": int(groups.iloc[test_idx].nunique()),
    }


def build_ablation(df: pd.DataFrame) -> pd.DataFrame:
    base = [c for c in BASE_FEATURES if c in df.columns and not df[c].isna().all()]
    maneuver_w = [f"{c}_w" for c in MANEUVER_FEATURES]

    rows = []
    base_metrics = evaluate_model(df, base)
    rows.append({"model": "base_no_maneuver", "added_feature": "", **base_metrics})

    all_metrics = evaluate_model(df, base + maneuver_w)
    rows.append({"model": "base_plus_all_maneuver", "added_feature": "all", **all_metrics})

    for col in MANEUVER_FEATURES:
        metrics = evaluate_model(df, base + [f"{col}_w"])
        rows.append({"model": "base_plus_one_maneuver", "added_feature": col, **metrics})

    out = pd.DataFrame(rows)
    base_mae = float(out.loc[out["model"].eq("base_no_maneuver"), "mae"].iloc[0])
    out["mae_improvement_pct_vs_base"] = (base_mae - out["mae"]) / base_mae * 100.0
    return out.sort_values(["mae_improvement_pct_vs_base"], ascending=False)


def plot_summary(summary: pd.DataFrame, ablation: pd.DataFrame) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    top = summary.copy()
    top["abs_rho"] = top["spearman_rho_fuel_rate"].abs()
    top = top.sort_values("abs_rho", ascending=True)

    one = ablation.loc[ablation["model"].eq("base_plus_one_maneuver")].copy()
    one["display_name"] = one["added_feature"].map(DISPLAY_NAMES)
    one = one.sort_values("mae_improvement_pct_vs_base", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), dpi=300)
    axes[0].barh(top["display_name"], top["spearman_rho_fuel_rate"], color="#2F7FB8")
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlabel("Spearman correlation with segment fuel rate")
    axes[0].set_title("(a) Correlation sensitivity", fontsize=10.5)
    axes[0].tick_params(labelsize=8.5)

    colors = ["#2F7FB8" if v >= 0 else "#B8B8B8" for v in one["mae_improvement_pct_vs_base"]]
    axes[1].barh(one["display_name"], one["mae_improvement_pct_vs_base"], color=colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("MAE improvement after adding one indicator (%)")
    axes[1].set_title("(b) Incremental model contribution", fontsize=10.5)
    axes[1].tick_params(labelsize=8.5)

    for ax in axes:
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.13, top=0.90, wspace=0.44)
    fig.savefig(FIG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = prepare_data()
    summary = build_summary(df)
    ablation = build_ablation(df)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    ablation.to_csv(ABLATION_CSV, index=False, encoding="utf-8-sig")
    plot_summary(summary, ablation)
    print(f"summary={SUMMARY_CSV}")
    print(summary.to_string(index=False))
    print(f"ablation={ABLATION_CSV}")
    print(ablation.to_string(index=False))
    print(f"figure={FIG_PATH}")


if __name__ == "__main__":
    main()
