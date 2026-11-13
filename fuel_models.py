"""Shared window-level fuel models for the revision (Step6b model comparison and Step8).

All models predict fuel (L) of a 30 s main-voyage window built by
build_step6_segment_fuel_model.build_main_voyage_segments. Speed
plans are applied to the 1 Hz records *before* windowing (apply_speed_plan), so
plan windows have the same definition and duration as training windows.

Models
- gbr_paper      : GradientBoosting with the original feature set (RPM as input)
- gbr_norpm      : same learner, RPM removed, environment added
- lgbm_mono      : LightGBM, monotone increasing in speed, duration, RPM, headwind, head-sea Hs
- lgbm_mono_norpm: as above without RPM
- greybox        : fuel rate = a + b*(STW/9)^n + kinetic + wind + wave terms, times a
                   LightGBM residual factor that does not see speed or RPM
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

from build_step6_segment_fuel_model import (
    RANDOM_STATE,
    SEGMENT_WINDOW_SECONDS,
    build_main_voyage_segments,
    build_model_objects,
)
from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR

ENV_CSV = OUTPUT_DIR / "environment" / f"{DEFAULT_DATASET_BASE}_voyage_environment.csv"
ENV_COLS = ["headwind_mean_ms", "crosswind_abs_mean_ms", "hs_mean_m", "hs_head_component_m", "current_along_mean_kn"]

PAPER_FEATURES = [
    "direction", "segment_duration_s", "segment_distance_nm", "segment_progress_mean", "speed_mean_knots",
    "speed_std_knots", "accel_abs_mean", "heading_change_abs_sum", "turn_rate_abs_mean", "turn_rate_peak",
    "curvature_proxy", "lat_acc_mean", "lat_acc_peak", "rpm_mean_seg", "rpm_max_seg",
    "active_engine_mean", "nearest_berth_dist_mean_nm", "no_sog_ratio",
]
# Engine power channels (power_mean_seg / power_max_seg) are empty in the logger data and are not used.
_DROP = {"direction", "rpm_mean_seg", "rpm_max_seg", "no_sog_ratio"}
# direction as a 0/1 flag (1 = berth_1 -> berth_2) for every model except the original GBR, which keeps it categorical
NORPM_FEATURES = [c for c in PAPER_FEATURES if c not in _DROP] + ENV_COLS + ["dir_b1_b2"]
RPM_FEATURES = NORPM_FEATURES + ["rpm_mean_seg", "rpm_max_seg"]
MONOTONE_UP = {"segment_duration_s", "speed_mean_knots", "rpm_mean_seg", "rpm_max_seg", "headwind_mean_ms",
               "hs_head_component_m"}
RESIDUAL_FEATURES = ["segment_progress_mean", "turn_rate_abs_mean", "turn_rate_peak", "heading_change_abs_sum",
                     "curvature_proxy", "speed_std_knots", "accel_abs_mean", "nearest_berth_dist_mean_nm",
                     "active_engine_mean", "crosswind_abs_mean_ms", "hs_mean_m", "dir_b1_b2"]
MODEL_NAMES = ["gbr_paper", "gbr_norpm", "lgbm_mono", "lgbm_mono_norpm", "greybox"]
USES_RPM = {"gbr_paper": True, "gbr_norpm": False, "lgbm_mono": True, "lgbm_mono_norpm": False, "greybox": False}
ROW_COLUMNS = ["time", "voyage_id", "segment_label", "direction", "delta_t_seconds", "step_distance_nm", "SOG",
               "speed_proxy_knots", "turn_rate_deg_per_s", "cog_delta_deg", "accel_knots_per_s", "rpm_mean", "rpm_max",
               "power_total", "active_engine_count", "nearest_berth_distance_nm", "no_sog_engine_flag",
               "fuel_step_est", "latitude", "longitude"]


# ----------------------------------------------------------------------------- data
def load_main_rows(step3_parquet: Path, voyage_ids: set[int]) -> pd.DataFrame:
    """1 Hz main-voyage records of the given voyages, with columns the window builder needs."""
    from build_step6_segment_fuel_model import load_step3

    df = load_step3(step3_parquet)
    keep = df["voyage_id"].isin(list(voyage_ids)).fillna(False) & df["segment_label"].eq("main_voyage")
    rows = df.loc[keep].sort_values(["voyage_id", "time"]).reset_index(drop=True)
    rows["voyage_id"] = rows["voyage_id"].astype(int)
    rows["training_voyage"] = True
    return rows


def load_env() -> pd.DataFrame:
    env = pd.read_csv(ENV_CSV)
    return env.set_index(env["voyage_id"].astype(int))[ENV_COLS]


def add_env(windows: pd.DataFrame, env: pd.DataFrame | dict) -> pd.DataFrame:
    w = windows.copy()
    for c in ENV_COLS:
        w[c] = w["voyage_id"].map(env[c]) if isinstance(env, pd.DataFrame) else float(env[c])
    w["dir_b1_b2"] = w["direction"].astype(str).eq("berth_1_to_berth_2").astype(float)
    return w


def build_windows(rows: pd.DataFrame, env: pd.DataFrame | dict) -> pd.DataFrame:
    return add_env(build_main_voyage_segments(rows), env)


DIRECTIONS = ("berth_2_to_berth_1", "berth_1_to_berth_2")
DIR_LABEL = {"berth_2_to_berth_1": "Berth 2 → Berth 1", "berth_1_to_berth_2": "Berth 1 → Berth 2"}


def voyage_direction() -> pd.Series:
    """voyage_id -> direction, from the Step3 voyage table."""
    e = pd.read_csv(OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step3_single_voyage_energy.csv", usecols=["voyage_id", "direction"])
    return e.set_index(e["voyage_id"].astype(int))["direction"]


def split_voyages(voyage_ids: np.ndarray, test_size=0.2, cal_size=0.2):
    """Test split identical in spirit to Step6 (GroupShuffleSplit, seed 42); calibration carved from train."""
    ids = np.asarray(sorted(voyage_ids))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(gss.split(ids, groups=ids))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cal_size, random_state=RANDOM_STATE)
    tr2, ca = next(gss2.split(ids[tr], groups=ids[tr]))
    return set(ids[tr][tr2]), set(ids[tr][ca]), set(ids[te])


# ----------------------------------------------------------------------------- operating plans
def plan_factor(progress: np.ndarray, factors: tuple[float, float, float]) -> np.ndarray:
    """Speed factor by thirds of (time-normalised) main-voyage progress, as in Step8."""
    k = np.clip(np.floor(np.asarray(progress) * 3).astype(int), 0, 2)
    return np.asarray(factors, float)[k]


def apply_speed_plan(rows: pd.DataFrame, factors: tuple[float, float, float], rpm_exponent: float) -> pd.DataFrame:
    """Scale speed of 1 Hz records by bin factor f; time stretches by 1/f, distance and heading changes stay.

    RPM is scaled by f**rpm_exponent (only used by models that take RPM as input)."""
    r = rows.copy()
    dt = r["delta_t_seconds"].fillna(1.0)
    el = dt.groupby(r["voyage_id"]).cumsum()
    tot = dt.groupby(r["voyage_id"]).transform("sum")
    f = plan_factor(((el - 0.5 * dt) / tot).to_numpy(), factors)
    r["delta_t_seconds"] = dt / f
    r["speed_used_knots"] = r["speed_used_knots"] * f
    r["accel_knots_per_s"] = r["accel_knots_per_s"] * f ** 2
    r["turn_rate_deg_per_s"] = r["turn_rate_deg_per_s"] * f
    r["rpm_mean"] = r["rpm_mean"] * f ** rpm_exponent
    r["rpm_max"] = r["rpm_max"] * f ** rpm_exponent
    return r


# ----------------------------------------------------------------------------- models
class GreyBox:
    """fuel = phys(STW, accel, wind, wave) * exp(residual(non-speed features))."""

    p_names = ["a_Lph", "b_Lph", "n", "c_kin", "d_wind", "e_wave"]

    def __init__(self):
        from lightgbm import LGBMRegressor

        self.resid = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE,
                                   verbose=-1)
        self.params = None

    @staticmethod
    def _terms(X: pd.DataFrame):
        stw = np.clip((X["speed_mean_knots"] - X["current_along_mean_kn"]).to_numpy(float), 0.2, None)
        acc = (X["speed_delta_knots"] / X["segment_duration_s"]).to_numpy(float)  # kn/s, signed
        return stw, acc, X["headwind_mean_ms"].to_numpy(float), X["hs_head_component_m"].to_numpy(float), \
            X["segment_duration_s"].to_numpy(float) / 3600.0

    def phys_rate(self, X: pd.DataFrame, p=None) -> np.ndarray:
        a, b, n, c, d, e = self.params if p is None else p
        stw, acc, hw, hsh, _ = self._terms(X)
        v = stw / 9.0
        rate = a + b * v ** n + c * acc * 3600.0 * v + d * hw * v ** 2 + e * hsh * v ** 2
        return np.clip(rate, 1.0, None)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.to_numpy(float)
        _, _, _, _, hours = self._terms(X)

        def res(p):
            return self.phys_rate(X, p) * hours - y

        sol = least_squares(res, x0=[5.0, 90.0, 3.0, 0.5, 0.5, 5.0],
                            bounds=([0, 0, 0.5, -50, -20, -200], [200, 1000, 6.0, 50, 20, 200]), loss="soft_l1",
                            f_scale=0.02)
        self.params = sol.x
        phys = self.phys_rate(X) * hours
        self.resid.fit(X[RESIDUAL_FEATURES], np.log(np.clip(y, 1e-4, None) / phys))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _, _, _, _, hours = self._terms(X)
        return self.phys_rate(X) * hours * np.exp(self.resid.predict(X[RESIDUAL_FEATURES]))


def _lgbm(features: list[str]):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
                         monotone_constraints=[1 if f in MONOTONE_UP else 0 for f in features],
                         monotone_constraints_method="advanced")


class Wrapped:
    """Uniform predict(DataFrame) for every model, remembering its feature list."""

    def __init__(self, name: str):
        self.name = name
        if name == "gbr_paper":
            self.features = PAPER_FEATURES
            self.model = build_model_objects([f for f in PAPER_FEATURES if f != "direction"], ["direction"])["GradientBoosting"]
        elif name == "gbr_norpm":
            self.features = NORPM_FEATURES
            self.model = build_model_objects(NORPM_FEATURES, [])["GradientBoosting"]
        elif name == "lgbm_mono":
            self.features = RPM_FEATURES
            self.model = _lgbm(RPM_FEATURES)
        elif name == "lgbm_mono_norpm":
            self.features = NORPM_FEATURES
            self.model = _lgbm(NORPM_FEATURES)
        elif name == "greybox":
            self.features = sorted(set(RESIDUAL_FEATURES + ["speed_mean_knots", "speed_delta_knots",
                                                            "segment_duration_s"] + ENV_COLS))
            self.model = GreyBox()
        else:
            raise ValueError(name)

    def fit(self, w: pd.DataFrame):
        self.model.fit(w[self.features], w["fuel_segment"].astype(float))
        return self

    def predict(self, w: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(w[self.features]), float)


# ----------------------------------------------------------------------------- RPM-speed relation
def estimate_rpm_exponent(windows: pd.DataFrame) -> dict:
    """Operating line log(RPM) ~ log(STW) + environment, cruise windows, voyage-level and window-level."""
    import statsmodels.formula.api as smf

    w = windows.loc[windows.segment_progress_mean.between(0.15, 0.85) & windows.speed_mean_knots.between(7, 10.5)
                    & (windows.rpm_mean_seg > 1000)].copy()
    w["stw"] = w.speed_mean_knots - w.current_along_mean_kn
    w["lr"], w["ls"] = np.log(w.rpm_mean_seg), np.log(w.stw)
    env = " + headwind_mean_ms + hs_head_component_m"
    win = smf.ols("lr ~ ls" + env, w).fit(cov_type="cluster", cov_kwds={"groups": w.voyage_id})
    v = w.groupby("voyage_id").agg(lr=("lr", "mean"), ls=("ls", "mean"), headwind_mean_ms=("headwind_mean_ms", "first"),
                                   hs_head_component_m=("hs_head_component_m", "first"))
    voy = smf.ols("lr ~ ls" + env, v).fit(cov_type="HC3")
    out = {}
    for k, r in (("voyage_level", voy), ("window_level_clustered", win)):
        lo, hi = r.conf_int().loc["ls"]
        out[k] = {"exponent": float(r.params["ls"]), "ci95": [float(lo), float(hi)], "n": int(r.nobs)}
    return out


# ----------------------------------------------------------------------------- conformal
def conformal_quantile(rel_err_abs: np.ndarray, alpha: float = 0.10) -> float:
    """Split-conformal quantile of |pred-true|/pred with finite-sample correction."""
    s = np.sort(np.asarray(rel_err_abs, float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1
    return float(s[min(k, len(s) - 1)])


def voyage_totals(windows: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    t = windows[["voyage_id", "fuel_segment", "segment_duration_s"]].copy()
    t["pred"] = pred
    return t.groupby("voyage_id").agg(true=("fuel_segment", "sum"), pred=("pred", "sum"),
                                      minutes=("segment_duration_s", lambda s: s.sum() / 60))


MODEL_DIR = OUTPUT_DIR / "model_comparison"


def save_bundle(models: dict, meta: dict) -> None:
    import joblib

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_DIR / "fitted_models.joblib")
    (MODEL_DIR / "model_bundle_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")


def load_bundle() -> tuple[dict, dict]:
    import joblib

    return joblib.load(MODEL_DIR / "fitted_models.joblib"), json.loads(
        (MODEL_DIR / "model_bundle_meta.json").read_text(encoding="utf-8"))


__all__ = [n for n in dir() if not n.startswith("_")] + ["SEGMENT_WINDOW_SECONDS"]
