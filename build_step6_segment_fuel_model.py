from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR as DEFAULT_OUTPUT_DIR


DEFAULT_STEP3 = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step3_enriched.parquet"
VALID_DIRECTIONS = {"berth_2_to_berth_1", "berth_1_to_berth_2"}
SEGMENT_WINDOW_SECONDS = 30.0
MIN_SEGMENT_DURATION_SECONDS = 10.0
MIN_SEGMENT_DISTANCE_NM = 0.003
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step-6 segment-level fuel models for main-voyage fuel prediction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_STEP3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def voyage_level_direction(df: pd.DataFrame, path: Path) -> pd.Series:
    """One direction per voyage.

    The earlier rule joined each *row's* nearest berth with the voyage's last berth,
    so the first part of a berth_2 -> berth_1 voyage was labelled berth_2_to_berth_1
    and the part nearer berth_1 became berth_1_to_berth_1 and was dropped. Here the
    direction comes from the Step3 voyage table (same source as Step4/Step7), with a
    first-row/last-row fallback if that table is missing.
    """
    energy_csv = path.with_name(path.name.replace("_step3_enriched.parquet", "_step3_single_voyage_energy.csv"))
    if energy_csv.exists():
        table = pd.read_csv(energy_csv, usecols=["voyage_id", "direction"])
        lookup = table.set_index(table["voyage_id"].astype(int))["direction"]
        return df["voyage_id"].map(lookup)
    ordered = df.sort_values("time")
    first = ordered.groupby("voyage_id")["nearest_berth_name"].first()
    last = ordered.groupby("voyage_id")["nearest_berth_name"].last()
    return df["voyage_id"].map(first.astype(str) + "_to_" + last.astype(str))


def training_voyage_ids(path: Path) -> set[int] | None:
    """Typical main voyages selected in Step4 (the same population as the fuel reference)."""
    step4_csv = path.with_name(path.name.replace("_step3_enriched.parquet", "_step4_main_route_candidates.csv"))
    if not step4_csv.exists():
        return None
    step4 = pd.read_csv(step4_csv, usecols=["direction", "voyage_id"])
    return set(step4.loc[step4["direction"].isin(VALID_DIRECTIONS), "voyage_id"].astype(int))


def load_step3(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["voyage_id"] = df["voyage_id"].astype("Int64")
    df["direction"] = voyage_level_direction(df, path)
    typical = training_voyage_ids(path)
    df["training_voyage"] = (
        df["voyage_id"].isin(typical).fillna(False).astype(bool) if typical is not None else df["voyage_id"].notna()
    )
    df["speed_used_knots"] = df["SOG"].copy()
    if "speed_proxy_knots" in df.columns:
        df["speed_used_knots"] = df["speed_used_knots"].fillna(df["speed_proxy_knots"])
    df["speed_used_knots"] = df["speed_used_knots"].fillna(0.0)
    if "power_total" in df.columns:
        df["power_total"] = df["power_total"].fillna(0.0)
    return df


def build_main_voyage_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    main = df.loc[
        df["voyage_id"].notna()
        & df["segment_label"].eq("main_voyage")
        & df["direction"].isin(VALID_DIRECTIONS)
        & df.get("training_voyage", pd.Series(True, index=df.index))
    ].copy()

    for voyage_id, group in main.groupby("voyage_id"):
        g = group.sort_values("time").copy()
        if len(g) < 5:
            continue

        dt = g["delta_t_seconds"].fillna(1.0)
        elapsed = dt.cumsum() - dt.iloc[0]
        g["segment_window_id"] = np.floor(elapsed / SEGMENT_WINDOW_SECONDS).astype(int)
        # Position of each record along the main voyage (0-1), matching tau in Step8.
        g["voyage_progress"] = (elapsed + 0.5 * dt) / max(float(dt.sum()), 1e-6)

        for seg_id, seg in g.groupby("segment_window_id"):
            duration_s = float(seg["delta_t_seconds"].sum())
            distance_nm = float(seg["step_distance_nm"].sum())
            if duration_s < MIN_SEGMENT_DURATION_SECONDS or distance_nm < MIN_SEGMENT_DISTANCE_NM:
                continue

            speed_ms = seg["speed_used_knots"].to_numpy(dtype=float) * 0.514444
            yaw_rate_rad = np.radians(seg["turn_rate_deg_per_s"].fillna(0.0).to_numpy(dtype=float))
            lat_acc = np.abs(speed_ms * yaw_rate_rad)
            heading_abs_sum = float(seg["cog_delta_deg"].abs().fillna(0.0).sum())
            curvature_proxy = heading_abs_sum / max(distance_nm, 1e-6)

            rows.append(
                {
                    "voyage_id": int(voyage_id),
                    "direction": str(seg["direction"].iloc[0]),
                    "segment_window_id": int(seg_id),
                    "segment_duration_s": duration_s,
                    "segment_distance_nm": distance_nm,
                    "segment_progress_mean": float(seg["voyage_progress"].mean()),
                    "speed_mean_knots": float(seg["speed_used_knots"].mean()),
                    "speed_std_knots": float(seg["speed_used_knots"].std(ddof=0)),
                    # signed speed change across the window (not a GBR feature; used by the grey-box model)
                    "speed_delta_knots": float(seg["speed_used_knots"].iloc[-1] - seg["speed_used_knots"].iloc[0]),
                    "accel_abs_mean": float(seg["accel_knots_per_s"].abs().fillna(0.0).mean()),
                    "heading_change_abs_sum": heading_abs_sum,
                    "turn_rate_abs_mean": float(seg["turn_rate_deg_per_s"].abs().fillna(0.0).mean()),
                    "turn_rate_peak": float(seg["turn_rate_deg_per_s"].abs().fillna(0.0).max()),
                    "curvature_proxy": curvature_proxy,
                    "lat_acc_mean": float(np.mean(lat_acc)),
                    "lat_acc_peak": float(np.max(lat_acc)),
                    "rpm_mean_seg": float(seg["rpm_mean"].mean()),
                    "rpm_max_seg": float(seg["rpm_max"].max()),
                    "power_mean_seg": float(seg["power_total"].mean()),
                    "power_max_seg": float(seg["power_total"].max()),
                    "active_engine_mean": float(seg["active_engine_count"].mean()),
                    "nearest_berth_dist_mean_nm": float(seg["nearest_berth_distance_nm"].mean()),
                    "no_sog_ratio": float(seg["no_sog_engine_flag"].mean()),
                    "fuel_segment": float(seg["fuel_step_est"].sum()),
                    "fuel_rate_segment": float(seg["fuel_step_est"].sum() / max(duration_s / 3600.0, 1e-6)),
                }
            )

    return pd.DataFrame(rows)


def build_model_objects(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    numeric_pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pre, numeric_features),
            ("cat", categorical_pre, categorical_features),
        ]
    )

    return {
        "LinearRegression": Pipeline(
            [("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=6,
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
    target_col = "fuel_segment"
    numeric_features = [c for c in feature_cols if c != "direction"]
    categorical_features = ["direction"]

    X = segments[feature_cols].copy()
    y = segments[target_col].astype(float)
    groups = segments["voyage_id"].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    voyage_test = segments.iloc[test_idx][["voyage_id", "segment_window_id", "direction"]].reset_index(drop=True)

    models = build_model_objects(numeric_features, categorical_features)
    comparison_rows = []
    prediction_rows = []
    importance_rows = []
    model_store: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-6, None))) * 100.0)
        comparison_rows.append(
            {
                "model": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape_pct": mape,
                "train_segments": int(len(X_train)),
                "test_segments": int(len(X_test)),
                "train_voyages": int(pd.Series(groups.iloc[train_idx]).nunique()),
                "test_voyages": int(pd.Series(groups.iloc[test_idx]).nunique()),
            }
        )

        pred_df = voyage_test.copy()
        pred_df["model"] = model_name
        pred_df["y_true"] = y_test.to_numpy(dtype=float)
        pred_df["y_pred"] = y_pred.astype(float)
        prediction_rows.append(pred_df)

        if model_name == "LinearRegression":
            pre = pipeline.named_steps["preprocessor"]
            feat_names = pre.get_feature_names_out()
            coefs = pipeline.named_steps["model"].coef_
            for feat, coef in sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True):
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": str(feat),
                        "importance_type": "coefficient",
                        "importance_value": float(coef),
                        "importance_abs": float(abs(coef)),
                    }
                )
        else:
            perm = permutation_importance(
                pipeline,
                X_test,
                y_test,
                n_repeats=10,
                random_state=RANDOM_STATE,
                scoring="neg_mean_absolute_error",
            )
            for feat, mean_imp in sorted(
                zip(feature_cols, perm.importances_mean),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": feat,
                        "importance_type": "permutation_neg_mae",
                        "importance_value": float(mean_imp),
                        "importance_abs": float(abs(mean_imp)),
                    }
                )

        model_store[model_name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    importances = pd.DataFrame(importance_rows)
    metadata = {
        "feature_columns": feature_cols,
        "target_column": target_col,
        "best_model_by_mae": str(comparison.iloc[0]["model"]),
        "train_voyage_count": int(pd.Series(groups.iloc[train_idx]).nunique()),
        "test_voyage_count": int(pd.Series(groups.iloc[test_idx]).nunique()),
        "segment_window_seconds": SEGMENT_WINDOW_SECONDS,
        "min_segment_duration_seconds": MIN_SEGMENT_DURATION_SECONDS,
        "min_segment_distance_nm": MIN_SEGMENT_DISTANCE_NM,
    }
    return comparison, predictions, importances, metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_step3(args.input)
    segments = build_main_voyage_segments(df)
    comparison, predictions, importances, metadata = train_and_evaluate(segments)

    dataset_base = args.input.stem.replace("_step3_enriched", "")
    base = f"{dataset_base}_step6_segment_fuel_model"
    dataset_csv = args.output_dir / f"{base}_dataset.csv"
    comparison_csv = args.output_dir / f"{base}_comparison.csv"
    predictions_csv = args.output_dir / f"{base}_predictions.csv"
    importance_csv = args.output_dir / f"{base}_feature_importance.csv"
    metadata_json = args.output_dir / f"{base}_metadata.json"

    segments.to_csv(dataset_csv, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    importances.to_csv(importance_csv, index=False, encoding="utf-8-sig")
    metadata_json.write_text(
        json.dumps(
            {
                "segment_count": int(len(segments)),
                "voyage_count": int(segments["voyage_id"].nunique()),
                "direction_counts": segments["direction"].value_counts().to_dict(),
                "direction_rule": "voyage-level, from step3_single_voyage_energy.csv",
                "training_population": "Step4 typical main voyages (step4_main_route_candidates.csv)",
                "segment_progress_definition": "window-mean of record position along the main voyage (0-1)",
                **metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"dataset_csv={dataset_csv}")
    print(f"comparison_csv={comparison_csv}")
    print(f"predictions_csv={predictions_csv}")
    print(f"importance_csv={importance_csv}")
    print(f"metadata_json={metadata_json}")


if __name__ == "__main__":
    main()
