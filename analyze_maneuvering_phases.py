"""Departure and arrival (manoeuvring) phases: how throttle handling relates to fuel.

Population: Step4 typical voyages of both directions. Per voyage and phase, RPM-based
handling indicators are computed from the 1 Hz records (SOG is mostly missing near the
berths, so speed comes from position steps). Fuel of each phase is regressed on its
duration, the indicators and environment (HC3 errors); effects are reported per 1 SD.
A matched comparison (strata of phase duration) contrasts the most and least
aggressive thirds. Observational associations only.

Arrival together with the main voyage: the approach speed (last 60 s of the main voyage)
and the throttle-back distance are both entered, per arrival berth, so that an early
throttle-back is not credited with savings that come from a slower approach. The fitted
arrival model (fuel, minutes, probability of a corrective burst) is saved for the
complete-voyage estimate of each Step8 plan.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fuel_models as fm
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

OUT = OUTPUT_DIR / "maneuvering_analysis"
BURST_RPM = 1000.0     # arrival: a "burst" is a rise above this after the first throttle-back
IDLE_RPM = 800.0
HIGH_RPM = 1550.0


def runs_above(x: np.ndarray, thr: float) -> int:
    a = x > thr
    return int(np.sum(a[1:] & ~a[:-1]) + (1 if a[:1].any() else 0))


def phase_indicators(g: pd.DataFrame, phase: str) -> dict:
    dt = g.delta_t_seconds.fillna(1.0).to_numpy(float)
    rpm = g.rpm_mean.to_numpy(float)
    rpm10 = pd.Series(rpm).rolling(10, min_periods=1).mean().to_numpy()
    spread = (g.rpm_max - g.rpm_mean).to_numpy(float)
    dist = g.step_distance_nm.fillna(0).to_numpy(float)
    d = {"minutes": dt.sum() / 60, "fuel_L": float(g.fuel_step_est.sum()), "rpm_mean": float(np.nanmean(rpm)),
         "rpm_peak10": float(np.nanmax(rpm10)), "rpm_cv": float(np.nanstd(rpm) / max(np.nanmean(rpm), 1)),
         "rpm_change_per_s": float(np.nansum(np.abs(np.diff(rpm10))) / max(dt.sum(), 1)),
         "engine_spread_mean": float(np.nanmean(spread)), "distance_nm": float(dist.sum()),
         "speed_mean_kn": float(dist.sum() / max(dt.sum() / 3600, 1e-6))}
    if phase == "departure":
        d["high_rpm_share"] = float(np.mean(rpm > HIGH_RPM))
        # how quickly full power is applied: seconds until the 10 s mean first reaches 95 % of its peak
        d["sec_to_95pct_peak"] = float(np.argmax(rpm10 >= 0.95 * np.nanmax(rpm10)))
    else:
        first_drop = int(np.argmax(rpm10 < IDLE_RPM)) if (rpm10 < IDLE_RPM).any() else len(rpm10) - 1
        after = rpm10[first_drop:]
        d["burst_count"] = runs_above(after, BURST_RPM)
        d["burst_share"] = float(np.mean(after > BURST_RPM)) if len(after) else 0.0
        d["throttle_back_dist_nm"] = float(g.nearest_berth_distance_nm.iloc[first_drop])
        d["idle_share"] = float(np.mean(rpm10 < IDLE_RPM))
    return d


def effect_table(df: pd.DataFrame, y: str, base_terms: list[str], indicators: list[str]) -> list[dict]:
    rows = []
    for ind in indicators:
        z = df.copy()
        z["x"] = (z[ind] - z[ind].mean()) / z[ind].std()
        r = smf.ols(f"{y} ~ x + " + " + ".join(base_terms), z).fit(cov_type="HC3")
        lo, hi = r.conf_int().loc["x"]
        rows.append({"outcome": y, "indicator": ind, "effect_L_per_SD": float(r.params["x"]), "ci_lo": float(lo),
                     "ci_hi": float(hi), "p": float(r.pvalues["x"]),
                     "effect_pct_of_phase_fuel": float(r.params["x"] / df[y].median() * 100),
                     "effect_pct_of_underway_fuel": float(r.params["x"] / df.fuel_underway_total.median() * 100),
                     "indicator_median": float(df[ind].median()), "indicator_sd": float(df[ind].std()), "n": int(r.nobs)})
    return rows


def matched_tercile(df: pd.DataFrame, y: str, strata_on: str, ind: str, n_strata=10) -> dict:
    z = df.copy()
    z["stratum"] = pd.qcut(z[strata_on], n_strata, labels=False, duplicates="drop")
    diffs, w = [], []
    for _, s in z.groupby("stratum"):
        if len(s) < 15:
            continue
        lo, hi = s[ind].quantile([1 / 3, 2 / 3])
        a, b = s.loc[s[ind] >= hi, y], s.loc[s[ind] <= lo, y]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
            w.append(len(s))
    diff = float(np.average(diffs, weights=w))
    return {"outcome": y, "indicator": ind, "strata": len(diffs), "top_minus_bottom_third_L": diff,
            "pct_of_phase_fuel": diff / df[y].median() * 100}


ARR_TERMS = "approach_speed_kn + throttle_back_m + headwind_mean_ms + crosswind_abs_mean_ms + hs_mean_m"
ARR_X = ["approach_speed_kn", "throttle_back_m", "headwind_mean_ms", "crosswind_abs_mean_ms", "hs_mean_m"]


def fit_arrival_model(arr: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Per arrival berth: arrival fuel, minutes and P(corrective burst) on approach speed and throttle-back
    distance (plus wind and waves). Also the raw quintile table, to show why approach speed must be held fixed."""
    out, bins = {}, []
    for d, a in arr.groupby("direction"):
        a = a.dropna(subset=ARR_X).copy()
        a["burst_any"] = (a.burst_count > 0).astype(float)
        fuel = smf.ols("fuel_L ~ " + ARR_TERMS, a).fit(cov_type="HC3")
        mins = smf.ols("minutes ~ " + ARR_TERMS, a).fit(cov_type="HC3")
        burst = smf.logit("burst_any ~ " + ARR_TERMS, a).fit(disp=0)
        raw = smf.ols("fuel_L ~ throttle_back_m", a).fit(cov_type="HC3")
        whole = smf.ols("fuel_underway_total ~ throttle_back_m + approach_speed_kn + main_voyage_minutes + headwind_mean_ms"
                        " + crosswind_abs_mean_ms + hs_mean_m", a).fit(cov_type="HC3")
        x0 = a[ARR_X].median()
        tb50, tb75 = a.throttle_back_m.quantile([.5, .75])
        early = x0.copy()
        early["throttle_back_m"] = tb75

        def pr(m, x):
            return float(np.asarray(m.predict(pd.DataFrame([x])))[0])

        def ci(r, k):
            return [float(v) for v in r.conf_int().loc[k]]

        out[d] = {
            "n": int(fuel.nobs), "r2_fuel": float(fuel.rsquared), "r2_minutes": float(mins.rsquared),
            "fuel_coef": {k: float(v) for k, v in fuel.params.items()},
            "minutes_coef": {k: float(v) for k, v in mins.params.items()},
            "burst_logit_coef": {k: float(v) for k, v in burst.params.items()},
            "fuel_per_100m_earlier_L": float(fuel.params["throttle_back_m"] * 100),
            "fuel_per_100m_earlier_ci": [v * 100 for v in ci(fuel, "throttle_back_m")],
            "fuel_per_100m_earlier_unadjusted_L": float(raw.params["throttle_back_m"] * 100),
            "fuel_per_kn_approach_L": float(fuel.params["approach_speed_kn"]),
            "fuel_per_kn_approach_ci": ci(fuel, "approach_speed_kn"),
            "minutes_per_100m_earlier": float(mins.params["throttle_back_m"] * 100),
            "minutes_per_kn_approach": float(mins.params["approach_speed_kn"]),
            "trip_fuel_per_100m_earlier_L": float(whole.params["throttle_back_m"] * 100),
            "trip_fuel_per_100m_earlier_ci": [v * 100 for v in ci(whole, "throttle_back_m")],
            "reference": {k: float(v) for k, v in x0.items()},
            "throttle_back_m_p50": float(tb50), "throttle_back_m_p75": float(tb75),
            "observed_median_fuel_L": float(a.fuel_L.median()), "observed_median_minutes": float(a.minutes.median()),
            "observed_burst_share": float(a.burst_any.mean()),
            "at_median": {"fuel_L": pr(fuel, x0), "minutes": pr(mins, x0), "p_burst": pr(burst, x0)},
            "early_throttle_back_p75": {"fuel_L": pr(fuel, early), "minutes": pr(mins, early), "p_burst": pr(burst, early)},
        }
        a["tb_quintile"] = pd.qcut(a.throttle_back_m, 5, labels=False)
        q = a.groupby("tb_quintile").agg(n=("voyage_id", "size"), throttle_back_m=("throttle_back_m", "median"),
                                         burst_share=("burst_any", "mean"), arrival_fuel_L=("fuel_L", "median"),
                                         arrival_min=("minutes", "median"), approach_speed_kn=("approach_speed_kn", "median"),
                                         main_fuel_L=("fuel_main_voyage", "median"),
                                         trip_fuel_L=("fuel_underway_total", "median")).reset_index()
        bins.append(q.assign(direction=d))
    return out, pd.concat(bins, ignore_index=True)


def predict_arrival(model: dict, approach_speed_kn: float, throttle_back_m: float) -> dict:
    """Arrival fuel / minutes / P(burst) at the berth's median wind and waves (model from fit_arrival_model)."""
    x = {**model["reference"], "approach_speed_kn": approach_speed_kn, "throttle_back_m": throttle_back_m,
         "Intercept": 1.0}

    def lin(c):
        return sum(v * x[k] for k, v in c.items())

    return {"fuel_L": lin(model["fuel_coef"]), "minutes": lin(model["minutes_coef"]),
            "p_burst": float(1 / (1 + np.exp(-lin(model["burst_logit_coef"]))))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from build_step6_segment_fuel_model import load_step3

    step4 = pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_candidates.csv")
    ids = set(step4.voyage_id.astype(int))
    df = load_step3(OUTPUT_DIR / f"{BASE}_step3_enriched.parquet")
    df = df[df.voyage_id.isin(list(ids)).fillna(False)].sort_values(["voyage_id", "time"])
    # approach = last 60 s of the main voyage (speed from position steps; SOG is sparse near the berths)
    tail = df[df.segment_label.eq("main_voyage")].groupby("voyage_id").tail(60)
    tg = tail.groupby("voyage_id")
    approach = pd.DataFrame({"approach_speed_kn": tg.step_distance_nm.sum() / (tg.delta_t_seconds.sum() / 3600),
                             "approach_rpm": tg.rpm_mean.mean()})
    df = df[df.segment_label.isin(["departure", "arrival"])]
    rec = []
    for (v, ph), g in df.groupby(["voyage_id", "segment_label"]):
        if len(g) >= 20:
            rec.append({"voyage_id": int(v), "phase": ph, **phase_indicators(g, ph)})
    ind = pd.DataFrame(rec)
    energy = pd.read_csv(OUTPUT_DIR / f"{BASE}_step3_single_voyage_energy.csv").set_index("voyage_id")
    env = fm.load_env()
    ind = ind.join(env, on="voyage_id").join(approach, on="voyage_id").join(
        energy[["direction", "fuel_underway_total", "fuel_main_voyage", "main_voyage_minutes",
                "fuel_departure", "fuel_arrival", "start_time"]], on="voyage_id")
    ind["throttle_back_m"] = ind.throttle_back_dist_nm * 1852
    ind["month"] = pd.to_datetime(ind.start_time, utc=True).dt.strftime("%m")
    ind.to_csv(OUT / "phase_indicators_per_voyage.csv", index=False, encoding="utf-8-sig")

    envt = ["headwind_mean_ms", "crosswind_abs_mean_ms", "hs_mean_m", "C(month)", "C(direction)"]
    dep, arr = ind[ind.phase == "departure"].copy(), ind[ind.phase == "arrival"].copy()
    dep["fuel_phase"], arr["fuel_phase"] = dep.fuel_L, arr.fuel_L
    effects = (effect_table(dep, "fuel_phase", ["minutes"] + envt,
                            ["rpm_mean", "rpm_peak10", "rpm_cv", "rpm_change_per_s", "high_rpm_share",
                             "sec_to_95pct_peak", "engine_spread_mean"])
               + effect_table(arr, "fuel_phase", ["minutes"] + envt,
                              ["burst_count", "burst_share", "rpm_cv", "rpm_change_per_s", "engine_spread_mean",
                               "throttle_back_dist_nm", "idle_share"]))
    eff = pd.DataFrame(effects)
    eff.insert(0, "phase", ["departure"] * 7 + ["arrival"] * 7)
    eff.to_csv(OUT / "phase_effects_per_sd.csv", index=False, encoding="utf-8-sig")

    # Does a gentler departure / earlier throttle-back just move fuel into the main voyage? Check whole-trip fuel.
    both = dep.add_prefix("dep_").merge(arr.add_prefix("arr_"), left_on="dep_voyage_id", right_on="arr_voyage_id")
    both = both.rename(columns={"dep_voyage_id": "voyage_id", "dep_fuel_underway_total": "fuel_underway_total",
                                "dep_headwind_mean_ms": "headwind_mean_ms", "dep_crosswind_abs_mean_ms": "crosswind_abs_mean_ms",
                                "dep_hs_mean_m": "hs_mean_m", "dep_month": "month", "dep_direction": "direction",
                                "dep_main_voyage_minutes": "main_voyage_minutes"})
    both["trip_minutes"] = both.dep_minutes + both.arr_minutes + both.main_voyage_minutes
    whole = effect_table(both, "fuel_underway_total", ["trip_minutes", "headwind_mean_ms", "crosswind_abs_mean_ms",
                                                       "hs_mean_m", "C(month)", "C(direction)"],
                         ["dep_rpm_mean", "dep_rpm_cv", "arr_burst_count", "arr_burst_share", "arr_rpm_cv",
                          "arr_throttle_back_dist_nm"])
    pd.DataFrame(whole).to_csv(OUT / "whole_trip_effects_per_sd.csv", index=False, encoding="utf-8-sig")

    matched = [matched_tercile(dep, "fuel_phase", "minutes", "rpm_mean"),
               matched_tercile(dep, "fuel_phase", "minutes", "rpm_cv"),
               matched_tercile(arr, "fuel_phase", "minutes", "burst_share"),
               matched_tercile(arr, "fuel_phase", "minutes", "rpm_change_per_s"),
               matched_tercile(both, "fuel_underway_total", "trip_minutes", "arr_burst_share")]

    # Simple what-if: voyages above the median of the strongest arrival indicator brought to the median
    r = smf.ols("fuel_phase ~ burst_share + minutes + " + " + ".join(envt), arr).fit(cov_type="HC3")
    excess = (arr.burst_share - arr.burst_share.median()).clip(lower=0)
    whatif = float((r.params["burst_share"] * excess).mean())
    arrival_model, tb_bins = fit_arrival_model(arr)
    tb_bins.to_csv(OUT / "arrival_throttle_back_quintiles.csv", index=False, encoding="utf-8-sig")
    summary = {
        "voyages": int(ind.voyage_id.nunique()),
        "voyages_by_direction": ind.drop_duplicates("voyage_id").direction.value_counts().to_dict(),
        "phase_medians": ind.groupby("phase")[["minutes", "fuel_L", "rpm_mean", "rpm_cv", "engine_spread_mean"]]
        .median().round(3).to_dict("index"),
        "phase_medians_by_direction": {f"{d}|{ph}": v for (d, ph), v in ind.groupby(["direction", "phase"])[
            ["minutes", "fuel_L", "rpm_mean", "rpm_peak10", "rpm_cv", "burst_count", "throttle_back_m",
             "approach_speed_kn"]].median().round(3).to_dict("index").items()},
        "arrival_model": arrival_model,
        "arrival_medians": arr[["burst_count", "burst_share", "throttle_back_dist_nm", "idle_share"]].median().round(3).to_dict(),
        "departure_medians": dep[["rpm_peak10", "high_rpm_share", "sec_to_95pct_peak"]].median().round(3).to_dict(),
        "share_of_underway_fuel_pct": float((dep.fuel_L.median() + arr.fuel_L.median()) / ind.fuel_underway_total.median() * 100),
        "duration_only_r2": {"departure": float(smf.ols("fuel_phase ~ minutes", dep).fit().rsquared),
                             "arrival": float(smf.ols("fuel_phase ~ minutes", arr).fit().rsquared)},
        "matched_tercile": matched,
        "whatif_arrival_burst_share_to_median_L": whatif,
        "whatif_pct_of_underway": whatif / ind.fuel_underway_total.median() * 100,
    }
    (OUT / "maneuvering_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(eff.round(3).to_string())
    print(pd.DataFrame(whole).round(3).to_string())
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
