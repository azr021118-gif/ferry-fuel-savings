"""Data for the revision figures under RPM-setpoint plans (eco = uniform -5 % RPM, 1 min budget).

Temporal split as in build_step6c_rpm_and_temporal_validation.py (train Jul-Sep, calibrate Oct,
test Nov, typical population with a training-only duration threshold). The speed model h is
refitted on the training months so that the whole chain is out of time.

  fig_data_G_rpm_setpoint.csv   per direction: synthetic plans scored by the Jul-Sep models at that
                                direction's November median environment: centreline RPM sweep 0..-5 %,
                                the Step8 selected plan (medium epsilon), training-period low-fuel profiles
  fig_data_C_rpm_setpoint.csv   eco (-5 % RPM) replayed on November voyages (temporal split)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import fuel_models as fm
from build_step6c_rpm_and_temporal_validation import tau_profile
from build_step6c_rpm_models import H_FEATURES, H_MONO, apply_row_factors, lgbm, row_window_id
from build_step6_segment_fuel_model import load_step3
from build_step8_fuel_model_route_optimizer import (RPM_STRATEGIES, engine_spread_by_tau, rpm_setpoint_functions,
                                                    synthesize_trajectory)
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

MC = fm.MODEL_DIR
STEP3 = OUTPUT_DIR / f"{BASE}_step3_enriched.parquet"
TRAIN_M, CAL_M, TEST_M = (7, 8, 9), (10,), (11,)


def main() -> None:
    env = fm.load_env()
    e = pd.read_csv(OUTPUT_DIR / f"{BASE}_step3_single_voyage_energy.csv")
    e["month"] = pd.to_datetime(e.start_time, utc=True).dt.month
    tr = e[e.month.isin(TRAIN_M)]
    qq = tr.groupby("direction").main_voyage_minutes.quantile([.25, .75]).unstack()
    typ = e[e.main_voyage_minutes <= e.direction.map(qq[0.75] + 1.5 * (qq[0.75] - qq[0.25]))]  # per direction
    ids = set(typ.voyage_id.astype(int))
    rows = fm.load_main_rows(STEP3, ids)
    w = fm.build_windows(rows, env)
    w["month"] = w.voyage_id.map(typ.set_index("voyage_id").month)
    Wtr, Wca, Wte = (w[w.month.isin(m)] for m in (TRAIN_M, CAL_M, TEST_M))
    mods = {n: fm.Wrapped(n).fit(Wtr) for n in fm.MODEL_NAMES}
    q = {}
    for n, m in mods.items():
        v = fm.voyage_totals(Wca, m.predict(Wca))
        q[n] = fm.conformal_quantile(((v.pred - v.true).abs() / v.pred).to_numpy())
    h = lgbm(H_FEATURES, H_MONO).fit(Wtr[H_FEATURES], Wtr["speed_mean_knots"])
    hb = {"h_speed": h, "H_FEATURES": H_FEATURES}
    print("models fitted", flush=True)

    # ---------------- C: eco (-5 % RPM) replay on November voyages
    rows_te = rows[rows.voyage_id.isin(set(Wte.voyage_id))].copy()
    rows_te["_wid"] = row_window_id(rows_te).to_numpy()
    key = pd.MultiIndex.from_arrays([rows_te.voyage_id, rows_te._wid])
    widx = Wte.set_index(["voyage_id", "segment_window_id"]).index
    base = {n: fm.voyage_totals(Wte, m.predict(Wte)) for n, m in mods.items()}
    rf = RPM_STRATEGIES["eco"]
    r_w = fm.plan_factor(Wte.segment_progress_mean.to_numpy(), rf)
    x = Wte.copy()
    x["rpm_mean_seg"] *= r_w
    s_w = np.clip(h.predict(x[H_FEATURES]) / h.predict(Wte[H_FEATURES]), 0.8, 1.2)
    wc = fm.build_windows(apply_row_factors(rows_te, pd.Series(s_w, index=widx).reindex(key).fillna(1.0).to_numpy(),
                                            pd.Series(r_w, index=widx).reindex(key).fillna(1.0).to_numpy()), env)
    crow = []
    vdir = typ.set_index("voyage_id").direction
    for n, m in mods.items():
        cf = fm.voyage_totals(wc, m.predict(wc))
        s = (base[n].pred - cf.pred) / base[n].pred * 100
        dm = cf.minutes - base[n].minutes
        for d in ("both",) + fm.DIRECTIONS:
            k = s.index if d == "both" else s.index[np.asarray(s.index.map(vdir)) == d]
            crow.append({"model": n, "direction": d, "split": "temporal_nov", "voyages": len(k),
                         "median_pct": float(s[k].median()), "p25": float(s[k].quantile(.25)),
                         "p75": float(s[k].quantile(.75)), "extra_min_median": float(dm[k].median())})
    pd.DataFrame(crow).to_csv(MC / "fig_data_C_rpm_setpoint.csv", index=False, encoding="utf-8-sig")
    print("C done", flush=True)

    # ---------------- G: synthetic plans, November median environment, per direction
    import json

    trn = typ[typ.month.isin(TRAIN_M)].merge(env.reset_index(), on="voyage_id")
    adj = smf.ols("fuel_main_voyage ~ C(direction) + " + " + ".join(fm.ENV_COLS), trn).fit()
    trn["fuel_env_adjusted"] = trn.fuel_main_voyage - adj.fittedvalues + trn.groupby("direction").fuel_main_voyage.transform("mean")
    interp_all = pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_interpolated.csv")
    paths_all = pd.read_csv(OUTPUT_DIR / f"{BASE}_step7_main_route_clusters_candidate_paths.csv")
    berths = pd.read_csv(OUTPUT_DIR / f"{BASE}_step2_berth_centers.csv")
    step8 = json.loads((OUTPUT_DIR / f"{BASE}_step8_route_optimization_summary.json").read_text(encoding="utf-8"))
    step3 = load_step3(STEP3)
    g_rows = []
    for d in fm.DIRECTIONS:
        td = trn[trn.direction.eq(d)]
        groups = {"all_train": set(td.voyage_id),
                  "lowest_quartile_raw": set(td.loc[td.fuel_main_voyage <= td.fuel_main_voyage.quantile(.25), "voyage_id"]),
                  "lowest_quartile_env_adjusted": set(td.loc[td.fuel_env_adjusted <= td.fuel_env_adjusted.quantile(.25), "voyage_id"])}
        interp = interp_all[interp_all.direction.eq(d)]
        sel = step8["selected_routes"][d]["medium"]
        paths = paths_all[paths_all.direction.eq(d) & paths_all.cluster_name.eq(sel["selected_cluster_name"])]
        spread = engine_spread_by_tau(step3, groups["all_train"])
        nov_env = {c: float(env.loc[[v for v in set(Wte.voyage_id) if vdir.get(v) == d], c].median()) for c in fm.ENV_COLS}
        ref = tau_profile(interp, groups["all_train"])
        cl = paths[paths.variant_name.eq("centerline")]
        pm = paths[paths.variant_name.eq(sel["selected_variant_name"])]
        plans = {f"centerline_rpm_{(x_ - 1) * 100:+.0f}%": (cl, ref, (x_, x_, x_)) for x_ in (1.0, 0.99, 0.98, 0.97, 0.96, 0.95)}
        plans["selected_plan"] = (pm, ref, RPM_STRATEGIES[sel["selected_speed_strategy"]])
        for g in ("lowest_quartile_raw", "lowest_quartile_env_adjusted"):
            prof = tau_profile(interp, groups[g])
            plans[f"{g}_profile"] = (prof.assign(direction=d)[["tau", "path_lat", "path_lon", "direction"]], prof, None)
        for i, (name, (path, prof, rfac)) in enumerate(plans.items()):
            if rfac is None or rfac == (1.0, 1.0, 1.0):  # observed profile as is
                traj = synthesize_trajectory(path, prof, spread, berths, (1.0, 1.0, 1.0), 0.0, 4.0, i)
            else:
                sfn, rfn, _ = rpm_setpoint_functions(path, prof, spread, berths, rfac, 4.0, nov_env, hb, 1000 + i)
                traj = synthesize_trajectory(path, prof, spread, berths, rfac, 0.0, 4.0, i, speed_fn=sfn, rpm_fn=rfn)
            ww = fm.build_windows(traj, nov_env)
            minutes = float(traj.delta_t_seconds.sum() / 60)
            for n, m in mods.items():
                f = float(m.predict(ww).sum())
                g_rows.append({"direction": d, "plan": name, "selected_variant": sel["selected_variant_name"], "model": n,
                               "fuel_L": f, "pi90_lo": f * (1 - q[n]), "pi90_hi": f * (1 + q[n]),
                               "minutes": minutes, "mean_speed_kn": float(ww.speed_mean_knots.mean())})
        print("G", d, "done", flush=True)
    g = pd.DataFrame(g_rows)
    bal = g.loc[g.plan.eq("centerline_rpm_+0%"), ["direction", "model", "fuel_L"]].rename(columns={"fuel_L": "_bal"})
    g = g.merge(bal, on=["direction", "model"])
    g["vs_balanced_pct"] = (g.fuel_L / g.pop("_bal") - 1) * 100
    g.to_csv(MC / "fig_data_G_rpm_setpoint.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(crow).round(2).to_string())
    print(g.pivot_table(index=["direction", "plan"], columns="model", values="vs_balanced_pct").round(2).join(
        g.groupby(["direction", "plan"]).minutes.first().round(3)).to_string())


if __name__ == "__main__":
    main()
