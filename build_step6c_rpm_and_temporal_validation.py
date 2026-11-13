"""Step6c: RPM reconstruction test (Fig. D) and temporal validation with a training-period
low-fuel baseline (Fig. E). Writes outputs/model_comparison/step6c_*.

Part D  (voyage-grouped split of Step6b, fitted bundle reused)
  D1  operating line points: voyage means of log RPM vs log STW (training voyages)
  D2  planning-usable RPM: a monotone LightGBM maps speed/manoeuvre/environment features
      to window RPM (training voyages only); fuel on test voyages is predicted with
      reconstructed instead of measured RPM; conformal re-calibrated on calibration voyages
  D3  sensitivity of the eco saving to the RPM exponent beta
  D4  WAPE for all models on the grouped split (complements Table 1)

Part E  (temporal split: Jul-Sep train, Oct calibration, Nov test; typical-voyage
         threshold, RPM exponent, models and conformal fitted without Nov data)
  E1  accuracy on November voyages; E2 expanding-window folds (test Sep, Oct, Nov)
  E3  eco/fast replay on November voyages
  E4  training-period low-fuel baselines (raw lowest quartile / environment-adjusted
      lowest quartile), scored with November conditions next to balanced, eco and the
      selected plan; environment of the selected voyages shows the selection bias.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fuel_models as fm
from build_step8_fuel_model_route_optimizer import SPEED_STRATEGIES, engine_spread_by_tau, synthesize_trajectory
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

OUT = fm.MODEL_DIR
STEP3 = OUTPUT_DIR / f"{BASE}_step3_enriched.parquet"
RPM_PLAN_FEATURES = ["speed_mean_knots", "speed_delta_knots", "speed_std_knots", "accel_abs_mean",
                     "segment_progress_mean", "turn_rate_abs_mean", "heading_change_abs_sum",
                     "nearest_berth_dist_mean_nm"] + fm.ENV_COLS + ["dir_b1_b2"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def accuracy(w: pd.DataFrame, pred: np.ndarray, q: float | None = None) -> dict:
    v = fm.voyage_totals(w, pred)
    rel = (v.pred - v.true) / v.true * 100
    d = {"voyages": len(v), "voyage_mae_L": float((v.pred - v.true).abs().mean()),
         "voyage_wape_pct": float((v.pred - v.true).abs().sum() / v.true.sum() * 100),
         "voyage_mape_pct": float(rel.abs().mean()), "voyage_bias_median_pct": float(rel.median()),
         "voyage_abs_err_p95_pct": float(rel.abs().quantile(.95))}
    if q is not None:
        d["conformal90_halfwidth_pct"] = q * 100
        d["conformal90_coverage"] = float(((v.true >= v.pred * (1 - q)) & (v.true <= v.pred * (1 + q))).mean())
    return d


def rpm_reconstructor(train_w: pd.DataFrame):
    from lightgbm import LGBMRegressor

    mono = [1 if f in {"speed_mean_knots", "headwind_mean_ms", "hs_head_component_m"} else 0 for f in RPM_PLAN_FEATURES]
    models = {}
    for target in ("rpm_mean_seg", "rpm_max_seg"):
        m = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
                          monotone_constraints=mono, monotone_constraints_method="advanced")
        models[target] = m.fit(train_w[RPM_PLAN_FEATURES], train_w[target])
    return models


def with_reconstructed_rpm(w: pd.DataFrame, rec: dict) -> pd.DataFrame:
    r = w.copy()
    for target, m in rec.items():
        r[target] = m.predict(w[RPM_PLAN_FEATURES])
    r["rpm_max_seg"] = np.maximum(r["rpm_max_seg"], r["rpm_mean_seg"])
    return r


def replay_saving(models: dict, rows: pd.DataFrame, base_w: pd.DataFrame, env, plan: str, beta: float) -> dict:
    cf_w = fm.build_windows(fm.apply_speed_plan(rows, SPEED_STRATEGIES[plan], beta), env)
    out = {}
    for n, m in models.items():
        b, c = fm.voyage_totals(base_w, m.predict(base_w)), fm.voyage_totals(cf_w, m.predict(cf_w))
        s = (b.pred - c.pred) / b.pred * 100
        out[n] = {"median_pct": float(s.median()), "p25": float(s.quantile(.25)), "p75": float(s.quantile(.75)),
                  "extra_min_median": float((c.minutes - b.minutes).median())}
    return out


def part_d(windows: pd.DataFrame, rows: pd.DataFrame, env: pd.DataFrame) -> dict:
    models, meta = fm.load_bundle()
    split = {k: set(v) for k, v in meta["split"].items()}
    W = {k: windows[windows.voyage_id.isin(s)] for k, s in split.items()}
    R = {}

    # D4: WAPE for every model (grouped split)
    R["D4_grouped_split_accuracy"] = {n: accuracy(W["test"], m.predict(W["test"]), meta["conformal_q_rel"][n])
                                      for n, m in models.items()}

    # D1: operating line points (training voyages, cruise windows)
    t = W["train"]
    c = t.loc[t.segment_progress_mean.between(0.15, 0.85) & t.speed_mean_knots.between(7, 10.5) & (t.rpm_mean_seg > 1000)]
    pts = c.assign(stw=c.speed_mean_knots - c.current_along_mean_kn).groupby("voyage_id").agg(
        log_stw=("stw", lambda s: float(np.log(s).mean())), log_rpm=("rpm_mean_seg", lambda s: float(np.log(s).mean())),
        headwind_mean_ms=("headwind_mean_ms", "first"), hs_head_component_m=("hs_head_component_m", "first"))
    pts.to_csv(OUT / "step6c_D1_operating_line_points.csv", encoding="utf-8-sig")
    R["D1_operating_line"] = meta["rpm_exponent"]

    # D2: reconstructed RPM
    rec = rpm_reconstructor(W["train"])
    rt, rc = with_reconstructed_rpm(W["test"], rec), with_reconstructed_rpm(W["cal"], rec)
    err = rt.rpm_mean_seg - W["test"].rpm_mean_seg
    vm = pd.DataFrame({"v": W["test"].voyage_id, "meas": W["test"].rpm_mean_seg, "rec": rt.rpm_mean_seg}).groupby("v").mean()
    R["D2_rpm_reconstruction"] = {
        "window_rpm_mae": float(err.abs().mean()), "window_rpm_mape_pct": float((err.abs() / W["test"].rpm_mean_seg).mean() * 100),
        "voyage_mean_rpm_abs_err_pct_median": float(((vm.rec - vm.meas).abs() / vm.meas).median() * 100),
        "voyage_mean_rpm_abs_err_pct_p95": float(((vm.rec - vm.meas).abs() / vm.meas).quantile(.95) * 100)}
    rows_d2 = []
    for n in ("gbr_paper", "lgbm_mono"):
        m = models[n]
        vc = fm.voyage_totals(rc, m.predict(rc))
        q_rec = fm.conformal_quantile(((vc.pred - vc.true).abs() / vc.pred).to_numpy())
        for label, w, q in (("measured_rpm", W["test"], meta["conformal_q_rel"][n]),
                            ("reconstructed_rpm_old_q", rt, meta["conformal_q_rel"][n]),
                            ("reconstructed_rpm_recalibrated_q", rt, q_rec)):
            rows_d2.append({"model": n, "rpm_input": label, **accuracy(w, m.predict(w), q)})
    for n in ("gbr_norpm", "lgbm_mono_norpm", "greybox"):
        rows_d2.append({"model": n, "rpm_input": "not_used", **accuracy(W["test"], models[n].predict(W["test"]),
                                                                       meta["conformal_q_rel"][n])})
    pd.DataFrame(rows_d2).to_csv(OUT / "step6c_D2_reconstructed_rpm_accuracy.csv", index=False, encoding="utf-8-sig")
    R["D2_fuel_with_reconstructed_rpm"] = rows_d2
    # per-voyage pairs for plotting
    pv = pd.DataFrame({"voyage_id": vm.index, "rpm_measured_mean": vm.meas, "rpm_reconstructed_mean": vm.rec})
    for n in ("gbr_paper", "lgbm_mono"):
        a, b = fm.voyage_totals(W["test"], models[n].predict(W["test"])), fm.voyage_totals(rt, models[n].predict(rt))
        pv[f"{n}_err_measured_pct"] = ((a.pred - a.true) / a.true * 100).reindex(pv.voyage_id).to_numpy()
        pv[f"{n}_err_reconstructed_pct"] = ((b.pred - b.true) / b.true * 100).reindex(pv.voyage_id).to_numpy()
    pv.to_csv(OUT / "step6c_D2_per_voyage.csv", index=False, encoding="utf-8-sig")
    log("D2 done")

    # D3: eco saving vs beta (RPM models)
    rows_test = rows[rows.voyage_id.isin(split["test"])]
    rpm_models = {n: models[n] for n in ("gbr_paper", "lgbm_mono")}
    sens = []
    for beta in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        for n, s in replay_saving(rpm_models, rows_test, W["test"], env, "eco", beta).items():
            sens.append({"beta": beta, "model": n, **s})
    pd.DataFrame(sens).to_csv(OUT / "step6c_D3_eco_saving_vs_beta.csv", index=False, encoding="utf-8-sig")
    R["D3_eco_saving_vs_beta"] = sens
    log("D3 done")
    return R


def tau_profile(interp: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    """Reference profile built like Step8's build_cluster_reference: SOG mean over voyages with SOG
    (about half the voyages have no AIS SOG and carry 0 in the Step4 table), RPM mean, median track."""
    sub = interp[interp.voyage_id.isin(ids)]
    g = sub.groupby("tau")
    sog = sub.loc[sub.SOG_interp > 0.2].groupby("tau").SOG_interp.mean()
    return pd.DataFrame({"tau": g.size().index, "sog_mean": sog.reindex(g.size().index).interpolate().bfill().ffill().to_numpy(),
                         "rpm_mean": g.rpm_interp.mean().to_numpy(), "path_lat": g.latitude.median().to_numpy(),
                         "path_lon": g.longitude.median().to_numpy()})


def part_e(energy: pd.DataFrame, env: pd.DataFrame) -> dict:
    R = {}
    e = energy.copy()
    e["month"] = pd.to_datetime(e.start_time, utc=True).dt.month
    train_m, cal_m, test_m = (7, 8, 9), (10,), (11,)
    tr = e[e.month.isin(train_m)]
    q = tr.groupby("direction").main_voyage_minutes.quantile([.25, .75]).unstack()
    upper = (q[0.75] + 1.5 * (q[0.75] - q[0.25])).to_dict()  # per direction, as in Step4
    e["typical_temporal"] = e.main_voyage_minutes <= e.direction.map(upper)
    step4_ids = set(pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_candidates.csv").voyage_id.astype(int))
    typ = e[e.typical_temporal]
    R["E0_population"] = {"upper_main_minutes_train_only": upper,
                          "typical_by_month": typ.groupby("month").size().to_dict(),
                          "differs_from_step4_population": int(len(set(typ.voyage_id) ^ step4_ids))}
    ids = set(typ.voyage_id.astype(int))
    rows = fm.load_main_rows(STEP3, ids)
    windows = fm.build_windows(rows, env)
    month_of = typ.set_index("voyage_id").month
    windows["month"] = windows.voyage_id.map(month_of)
    log(f"E windows {len(windows)}")

    def fold(train_months, cal_months, test_months, with_cal=True):
        Wtr = windows[windows.month.isin(train_months)]
        Wca = windows[windows.month.isin(cal_months)] if with_cal else None
        Wte = windows[windows.month.isin(test_months)]
        mods = {n: fm.Wrapped(n).fit(Wtr) for n in fm.MODEL_NAMES}
        q = {}
        if with_cal:
            for n, m in mods.items():
                v = fm.voyage_totals(Wca, m.predict(Wca))
                q[n] = fm.conformal_quantile(((v.pred - v.true).abs() / v.pred).to_numpy())
        acc = {n: accuracy(Wte, m.predict(Wte), q.get(n)) for n, m in mods.items()}
        return mods, q, acc, Wtr, Wte

    mods, q, acc, Wtr, Wte = fold(train_m, cal_m, test_m)
    beta_t = fm.estimate_rpm_exponent(Wtr)
    R["E1_nov_test_accuracy"] = acc
    R["E1_rpm_exponent_train_months"] = beta_t
    log("E1 done")

    folds = []
    for trm, tem in (((7, 8), (9,)), ((7, 8, 9), (10,)), ((7, 8, 9, 10), (11,))):
        _, _, a, _, _ = fold(trm, (), tem, with_cal=False)
        for n, d in a.items():
            folds.append({"train_months": "-".join(map(str, trm)), "test_month": tem[0], "model": n, **d})
    pd.DataFrame(folds).to_csv(OUT / "step6c_E2_expanding_window.csv", index=False, encoding="utf-8-sig")
    R["E2_expanding_window"] = folds
    log("E2 done")

    beta = beta_t["voyage_level"]["exponent"]
    rows_te = rows[rows.voyage_id.isin(set(Wte.voyage_id))]
    R["E3_nov_replay"] = {plan: replay_saving(mods, rows_te, Wte, env, plan, beta) for plan in ("eco", "fast")}
    log("E3 done")

    # E4: training-period low-fuel baselines, per direction
    trn = typ[typ.month.isin(train_m)].merge(env.reset_index(), on="voyage_id")
    adj = smf.ols("fuel_main_voyage ~ C(direction) + " + " + ".join(fm.ENV_COLS), trn).fit()
    trn["fuel_env_adjusted"] = trn.fuel_main_voyage - adj.fittedvalues + trn.groupby("direction").fuel_main_voyage.transform("mean")
    trn["sog_main_kn"] = trn.distance_main_voyage_nm / (trn.main_voyage_minutes / 60)
    interp = pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_interpolated.csv")
    paths = pd.read_csv(OUTPUT_DIR / f"{BASE}_step7_main_route_clusters_candidate_paths.csv")
    berths = pd.read_csv(OUTPUT_DIR / f"{BASE}_step2_berth_centers.csv")
    from build_step6_segment_fuel_model import load_step3

    step3 = load_step3(STEP3)
    sel_bias, e4, nov_envs = {}, [], {}
    for d in fm.DIRECTIONS:
        td = trn[trn.direction.eq(d)]
        groups = {"all_train": set(td.voyage_id),
                  "lowest_quartile_raw": set(td.loc[td.fuel_main_voyage <= td.fuel_main_voyage.quantile(.25), "voyage_id"]),
                  "lowest_quartile_env_adjusted": set(td.loc[td.fuel_env_adjusted <= td.fuel_env_adjusted.quantile(.25), "voyage_id"])}
        for g, s in groups.items():
            sub = td[td.voyage_id.isin(s)]
            sel_bias[f"{d}|{g}"] = {"n": len(sub),
                                    **sub[["fuel_main_voyage", "fuel_env_adjusted", "main_voyage_minutes", "sog_main_kn"]].median().round(3).to_dict(),
                                    **{f"{c}_mean": v for c, v in sub[fm.ENV_COLS].mean().round(3).to_dict().items()}}
        spread = engine_spread_by_tau(step3, groups["all_train"])
        te_ids = [v for v in set(Wte.voyage_id) if v in set(typ.loc[typ.direction.eq(d), "voyage_id"])]
        nov_env = {c: float(env.loc[te_ids, c].median()) for c in fm.ENV_COLS}
        nov_envs[d] = nov_env
        idir = interp[interp.direction.eq(d)]
        pdir = paths[paths.direction.eq(d)]
        pdir = pdir[pdir.cluster_name.eq(pdir.cluster_name.min())]
        ref_all = tau_profile(idir, groups["all_train"])
        plans = {"balanced_centerline": (pdir[pdir.variant_name.eq("centerline")], ref_all, SPEED_STRATEGIES["balanced"]),
                 "eco_centerline": (pdir[pdir.variant_name.eq("centerline")], ref_all, SPEED_STRATEGIES["eco"]),
                 "selected_port_medium_eco": (pdir[pdir.variant_name.eq("port_shift_medium")], ref_all, SPEED_STRATEGIES["eco"])}
        for g in ("lowest_quartile_raw", "lowest_quartile_env_adjusted"):
            prof = tau_profile(idir, groups[g])
            plans[f"{g}_profile"] = (prof.assign(direction=d)[["tau", "path_lat", "path_lon", "direction"]], prof, (1.0, 1.0, 1.0))
        rows4 = []
        for i, (name, (path, prof, factors)) in enumerate(plans.items()):
            traj = synthesize_trajectory(path.assign(direction=d), prof, spread, berths, factors, beta, 4.0, i)
            w = fm.build_windows(traj, nov_env)
            for n, m in mods.items():
                f = float(m.predict(w).sum())
                rows4.append({"direction": d, "plan": name, "model": n, "fuel_L": f, "pi90_lo": f * (1 - q[n]),
                              "pi90_hi": f * (1 + q[n]), "minutes": float(w.segment_duration_s.sum() / 60),
                              "mean_speed_kn": float(w.speed_mean_knots.mean())})
        r4 = pd.DataFrame(rows4)
        bal = r4[r4.plan.eq("balanced_centerline")].set_index("model").fuel_L
        r4["vs_balanced_pct"] = (r4.fuel_L / r4.model.map(bal) - 1) * 100
        e4.append(r4)
    R["E4_selection_bias_train_groups"] = sel_bias
    e4 = pd.concat(e4, ignore_index=True)
    e4.to_csv(OUT / "step6c_E4_low_fuel_baselines.csv", index=False, encoding="utf-8-sig")
    nov = typ[typ.month.isin(test_m)]
    R["E4_plans"] = e4.round(3).to_dict("records")
    R["E4_nov_observed"] = {d: {"n": len(g), "fuel_main_median": float(g.fuel_main_voyage.median()),
                                "minutes_median": float(g.main_voyage_minutes.median())} for d, g in nov.groupby("direction")}
    R["E4_nov_env_median"] = nov_envs
    log("E4 done")
    return R


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = fm.load_env()
    energy = pd.read_csv(OUTPUT_DIR / f"{BASE}_step3_single_voyage_energy.csv")
    step4_ids = set(pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_candidates.csv").voyage_id.astype(int))
    rows = fm.load_main_rows(STEP3, step4_ids)
    windows = fm.build_windows(rows, env)
    log("windows built")
    R = {"part_D": part_d(windows, rows, env), "part_E": part_e(energy, env)}
    (OUT / "step6c_summary.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
