"""Step6c: learned RPM relations, two ways (same voyages and split as Step6b).

Direction 1  RPM = g(speed, acceleration, turning, route position, environment)
             replaces the single exponent f**beta when a *speed* plan is evaluated:
             RPM_cf = RPM_obs * g(cf window) / g(same window at the observed speed).
Direction 2  RPM is the decision variable. Speed = h(RPM, turning, route position,
             environment); a plan scales RPM per third, h gives the speed (hence time),
             and the fuel models with RPM input give fuel.
No water-depth / fairway input: repeated liner route, kept as a transfer limitation.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import fuel_models as fm
from build_step6_segment_fuel_model import RANDOM_STATE, SEGMENT_WINDOW_SECONDS
from build_step8_fuel_model_route_optimizer import SPEED_STRATEGIES
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

OUT = fm.MODEL_DIR
G_FEATURES = ["speed_mean_knots", "speed_delta_knots", "accel_abs_mean", "turn_rate_abs_mean", "heading_change_abs_sum",
              "curvature_proxy", "segment_progress_mean", "nearest_berth_dist_mean_nm", *fm.ENV_COLS, "dir_b1_b2"]
G_MONO = {"speed_mean_knots": 1, "speed_delta_knots": 1, "headwind_mean_ms": 1, "hs_head_component_m": 1,
          "current_along_mean_kn": -1}
H_FEATURES = ["rpm_mean_seg", "turn_rate_abs_mean", "heading_change_abs_sum", "curvature_proxy", "segment_progress_mean",
              "nearest_berth_dist_mean_nm", "active_engine_mean", *fm.ENV_COLS, "dir_b1_b2"]
H_MONO = {"rpm_mean_seg": 1, "headwind_mean_ms": -1, "hs_head_component_m": -1, "current_along_mean_kn": 1}
RPM_PLANS = {  # RPM factor per third of the main voyage
    "rpm_-1%": (0.99, 0.99, 0.99), "rpm_-2%": (0.98, 0.98, 0.98), "rpm_-3%": (0.97, 0.97, 0.97),
    "rpm_-4%": (0.96, 0.96, 0.96), "rpm_-5%": (0.95, 0.95, 0.95),
    "rpm_eco_profile": None,  # eco speed factors ** beta, filled at run time
    "rpm_+3%": (1.03, 1.03, 1.03),
}


def lgbm(features, mono):
    return LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
                         monotone_constraints=[mono.get(f, 0) for f in features], monotone_constraints_method="advanced")


def row_window_id(rows: pd.DataFrame) -> pd.Series:
    """Same window index the Step6 builder assigns (elapsed seconds // 30, per voyage)."""
    dt = rows["delta_t_seconds"].fillna(1.0)
    el = dt.groupby(rows["voyage_id"]).cumsum() - dt.groupby(rows["voyage_id"]).transform("first")
    return np.floor(el / SEGMENT_WINDOW_SECONDS).astype(int)


def apply_row_factors(rows: pd.DataFrame, speed_f: np.ndarray, rpm_f: np.ndarray) -> pd.DataFrame:
    r = rows.copy()
    r["delta_t_seconds"] = r["delta_t_seconds"].fillna(1.0) / speed_f
    r["speed_used_knots"] = r["speed_used_knots"] * speed_f
    r["accel_knots_per_s"] = r["accel_knots_per_s"] * speed_f ** 2
    r["turn_rate_deg_per_s"] = r["turn_rate_deg_per_s"] * speed_f
    r["rpm_mean"] = r["rpm_mean"] * rpm_f
    r["rpm_max"] = r["rpm_max"] * rpm_f
    return r


def unscale(w: pd.DataFrame, f: np.ndarray) -> pd.DataFrame:
    """Window features as they would be at the observed speed (undo a speed factor f)."""
    u = w.copy()
    u["speed_mean_knots"] = u["speed_mean_knots"] / f
    u["speed_delta_knots"] = u["speed_delta_knots"] / f
    u["accel_abs_mean"] = u["accel_abs_mean"] / f ** 2
    u["turn_rate_abs_mean"] = u["turn_rate_abs_mean"] / f
    return u


def voyage_err(tot: pd.DataFrame, pred_col="pred", true_col="true") -> dict:
    rel = (tot[pred_col] - tot[true_col]) / tot[true_col] * 100
    return {"voyage_mape_pct": float(rel.abs().mean()), "bias_median_pct": float(rel.median()),
            "abs_err_p95_pct": float(rel.abs().quantile(.95))}


def main() -> None:
    models, meta = fm.load_bundle()
    beta = float(meta["rpm_exponent"]["voyage_level"]["exponent"])
    q_fuel = meta["conformal_q_rel"]
    split = {k: set(v) for k, v in meta["split"].items()}
    ids = set().union(*split.values())
    rows = fm.load_main_rows(OUTPUT_DIR / f"{BASE}_step3_enriched.parquet", ids)
    env = fm.load_env()
    win = fm.build_windows(rows, env)
    W = {k: win[win.voyage_id.isin(s)] for k, s in split.items()}
    R = {"beta_parametric": beta}

    # ------------------------------------------------------------------ fit g and h
    g = lgbm(G_FEATURES, G_MONO).fit(W["train"][G_FEATURES], W["train"]["rpm_mean_seg"])
    h = lgbm(H_FEATURES, H_MONO).fit(W["train"][H_FEATURES], W["train"]["speed_mean_knots"])
    for name, m, feats, y in (("g_rpm", g, G_FEATURES, "rpm_mean_seg"), ("h_speed", h, H_FEATURES, "speed_mean_knots")):
        p = m.predict(W["test"][feats])
        R[name] = {"window_mae": float(mean_absolute_error(W["test"][y], p)), "window_r2": float(r2_score(W["test"][y], p)),
                   "window_mape_pct": float(np.mean(np.abs(p / W["test"][y] - 1)) * 100)}
    joblib.dump({"g_rpm": g, "h_speed": h, "G_FEATURES": G_FEATURES, "H_FEATURES": H_FEATURES}, OUT / "rpm_speed_models.joblib")

    # local elasticity of learned RPM wrt speed, cruise windows of test voyages
    c = W["test"].loc[W["test"].segment_progress_mean.between(0.15, 0.85) & W["test"].speed_mean_knots.between(7, 10.5)]
    up, dn = c.copy(), c.copy()
    up["speed_mean_knots"] *= 1.02
    dn["speed_mean_knots"] *= 0.98
    el = (np.log(g.predict(up[G_FEATURES])) - np.log(g.predict(dn[G_FEATURES]))) / np.log(1.02 / 0.98)
    third = np.clip(np.floor(c.segment_progress_mean * 3), 0, 2).astype(int)
    R["g_local_elasticity"] = {"median": float(np.median(el)), "p10": float(np.quantile(el, .1)), "p90": float(np.quantile(el, .9)),
                               "by_third_median": {int(k): float(np.median(el[third == k])) for k in range(3)}}
    # inverse check from h: d log v / d log rpm -> implied rpm elasticity 1/e_h
    c2u, c2d = c.copy(), c.copy()
    c2u["rpm_mean_seg"] *= 1.02
    c2d["rpm_mean_seg"] *= 0.98
    eh = (np.log(h.predict(c2u[H_FEATURES])) - np.log(h.predict(c2d[H_FEATURES]))) / np.log(1.02 / 0.98)
    R["h_local_elasticity_speed_wrt_rpm"] = {"median": float(np.median(eh)), "implied_rpm_wrt_speed": float(1 / np.median(eh))}

    # chained accuracy: speed -> g -> RPM-input fuel model (what a speed plan can actually know)
    spread_ratio = float((W["train"].rpm_max_seg / W["train"].rpm_mean_seg).median())
    wt = W["test"].copy()
    wt["rpm_mean_seg"] = g.predict(wt[G_FEATURES])
    wt["rpm_max_seg"] = wt["rpm_mean_seg"] * spread_ratio
    R["chained_speed_to_rpm_to_fuel"] = {}
    for n in ("gbr_paper", "lgbm_mono"):
        R["chained_speed_to_rpm_to_fuel"][n] = {
            "with_true_rpm": voyage_err(fm.voyage_totals(W["test"], models[n].predict(W["test"]))),
            "with_predicted_rpm": voyage_err(fm.voyage_totals(wt, models[n].predict(wt)))}
    print(json.dumps({k: R[k] for k in ("g_rpm", "h_speed", "g_local_elasticity", "h_local_elasticity_speed_wrt_rpm",
                                        "chained_speed_to_rpm_to_fuel")}, indent=1), flush=True)

    # ------------------------------------------------------------------ Direction 1: speed plans, learned RPM
    rows_t = rows[rows.voyage_id.isin(split["test"])]
    base = {n: fm.voyage_totals(W["test"], models[n].predict(W["test"])) for n in fm.MODEL_NAMES}
    t4 = []
    for plan in ("eco", "fast"):
        fac = SPEED_STRATEGIES[plan]
        w0 = fm.build_windows(fm.apply_speed_plan(rows_t, fac, 0.0), env)  # speed changed, RPM untouched
        f_w = fm.plan_factor(w0.segment_progress_mean.to_numpy(), fac)
        ratio = g.predict(w0[G_FEATURES]) / g.predict(unscale(w0, f_w)[G_FEATURES])
        w_cf = w0.copy()
        w_cf["rpm_mean_seg"] *= ratio
        w_cf["rpm_max_seg"] *= ratio
        R[f"learned_rpm_ratio_{plan}"] = {"median": float(np.median(ratio)), "p10": float(np.quantile(ratio, .1)),
                                          "p90": float(np.quantile(ratio, .9)),
                                          "beta_rule_equivalent_median": float(np.median(f_w ** beta))}
        for n in ("gbr_paper", "lgbm_mono"):
            cf = fm.voyage_totals(w_cf, models[n].predict(w_cf))
            s = (base[n].pred - cf.pred) / base[n].pred * 100
            t4.append({"plan": plan, "model": n, "rpm_rule": "learned_g", "main_saving_median_pct": float(s.median()),
                       "p25": float(s.quantile(.25)), "p75": float(s.quantile(.75)),
                       "mean_L": float((base[n].pred - cf.pred).mean()),
                       "extra_minutes_median": float((cf.minutes - base[n].minutes).median())})
    t2 = pd.read_csv(OUT / "table2_speed_plan_savings.csv")
    for _, r in t2[t2.model.isin(["gbr_paper", "lgbm_mono"])].iterrows():
        t4.append({"plan": r.plan, "model": r.model, "rpm_rule": r.rpm_rule, "main_saving_median_pct": r.main_saving_median_pct,
                   "p25": r.main_saving_p25_pct, "p75": r.main_saving_p75_pct, "mean_L": r.main_saving_mean_L,
                   "extra_minutes_median": r.extra_minutes_median})
    table4 = pd.DataFrame(t4).sort_values(["plan", "model", "rpm_rule"])
    table4.to_csv(OUT / "table4_speed_plans_learned_rpm.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ Direction 2: RPM as decision variable
    # time closure of h on cal (conformal) and test voyages: minutes = sum(distance / h)
    def h_minutes(w):
        t = w[["voyage_id", "segment_distance_nm", "segment_duration_s"]].copy()
        t["pred_s"] = w["segment_distance_nm"] / np.clip(h.predict(w[H_FEATURES]), 0.5, None) * 3600
        return t.groupby("voyage_id").agg(true=("segment_duration_s", "sum"), pred=("pred_s", "sum")) / 60

    hc, ht = h_minutes(W["cal"]), h_minutes(W["test"])
    q_time = fm.conformal_quantile(((hc.pred - hc.true).abs() / hc.pred).to_numpy(), meta["conformal_alpha"])
    R["h_time_closure_test"] = {**voyage_err(ht), "conformal90_halfwidth_pct": q_time * 100,
                                "coverage": float(((ht.true >= ht.pred * (1 - q_time)) & (ht.true <= ht.pred * (1 + q_time))).mean())}
    RPM_PLANS["rpm_eco_profile"] = tuple(round(x ** beta, 4) for x in SPEED_STRATEGIES["eco"])
    rows_t = rows_t.copy()
    rows_t["_wid"] = row_window_id(rows_t).to_numpy()
    wt_idx = W["test"].set_index(["voyage_id", "segment_window_id"])
    t5 = []
    for plan, rf in RPM_PLANS.items():
        r_w = fm.plan_factor(W["test"].segment_progress_mean.to_numpy(), rf)
        xw = W["test"].copy()
        xw["rpm_mean_seg"] *= r_w
        s_w = np.clip(h.predict(xw[H_FEATURES]) / h.predict(W["test"][H_FEATURES]), 0.8, 1.2)
        sf = pd.Series(s_w, index=wt_idx.index)
        rfac = pd.Series(r_w, index=wt_idx.index)
        key = pd.MultiIndex.from_arrays([rows_t.voyage_id, rows_t._wid])
        s_row = sf.reindex(key).fillna(1.0).to_numpy()
        r_row = rfac.reindex(key).fillna(1.0).to_numpy()
        w_cf = fm.build_windows(apply_row_factors(rows_t, s_row, r_row), env)
        for n in fm.MODEL_NAMES:
            cf = fm.voyage_totals(w_cf, models[n].predict(w_cf))
            s = (base[n].pred - cf.pred) / base[n].pred * 100
            dmin = cf.minutes - base[n].minutes
            t5.append({"plan": plan, "rpm_factors": str(rf), "model": n, "uses_rpm": fm.USES_RPM[n],
                       "main_saving_median_pct": float(s.median()), "p25": float(s.quantile(.25)),
                       "p75": float(s.quantile(.75)), "mean_L": float((base[n].pred - cf.pred).mean()),
                       "extra_minutes_median": float(dmin.median()), "speed_factor_median": float(np.median(s_w)),
                       "fuel_pi90_halfwidth_pct": q_fuel[n] * 100, "time_pi90_halfwidth_pct": q_time * 100})
        print(plan, "done", flush=True)
    table5 = pd.DataFrame(t5)
    table5.to_csv(OUT / "table5_rpm_setpoint_plans.csv", index=False, encoding="utf-8-sig")
    (OUT / "rpm_models_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps({k: R[k] for k in R if k.startswith(("learned", "h_time"))}, indent=1))
    print(table4.round(2).to_string())
    print(table5.round(2).to_string())


if __name__ == "__main__":
    main()
