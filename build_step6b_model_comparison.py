"""Step6b: compare window-level fuel models on the same held-out voyages.

Table 1  voyage-level accuracy on test voyages + split-conformal 90 % intervals
Table 2  eco / fast speed-plan savings from each model, replayed on the test voyages'
         own 1 Hz records (re-windowed after the speed change), with RPM rules:
         paper proxy (f**1.15) and the data-estimated operating line (f**beta).
Writes outputs/model_comparison/ and the fitted-model bundle used by Step8.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import fuel_models as fm
from build_step8_fuel_model_route_optimizer import SPEED_STRATEGIES
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

PAPER_RPM_EXPONENT = 1.15
ALPHA = 0.10


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def main() -> None:
    out = fm.MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    step4 = pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_candidates.csv")
    ids = set(step4.voyage_id.astype(int))
    rows = fm.load_main_rows(OUTPUT_DIR / f"{BASE}_step3_enriched.parquet", ids)
    env = fm.load_env()
    windows = fm.build_windows(rows, env)
    assert windows[fm.ENV_COLS].notna().all().all(), "environment covariates missing for some voyages"
    log(f"windows={len(windows)} voyages={windows.voyage_id.nunique()}")

    proper, cal, test = fm.split_voyages(np.array(sorted(ids)))
    W = {k: windows[windows.voyage_id.isin(s)] for k, s in (("train", proper), ("cal", cal), ("test", test))}
    rpm_exp = fm.estimate_rpm_exponent(W["train"])
    beta = rpm_exp["voyage_level"]["exponent"]
    log(f"split train/cal/test voyages = {len(proper)}/{len(cal)}/{len(test)}; rpm exponent {rpm_exp}")

    models, t1, q = {}, [], {}
    for name in fm.MODEL_NAMES:
        m = fm.Wrapped(name).fit(W["train"])
        models[name] = m
        pc, pt = m.predict(W["cal"]), m.predict(W["test"])
        vc, vt = fm.voyage_totals(W["cal"], pc), fm.voyage_totals(W["test"], pt)
        q[name] = fm.conformal_quantile(((vc.pred - vc.true).abs() / vc.pred).to_numpy(), ALPHA)
        rel = (vt.pred - vt.true) / vt.true * 100
        cover = ((vt.true >= vt.pred * (1 - q[name])) & (vt.true <= vt.pred * (1 + q[name]))).mean()
        yt = W["test"].fuel_segment.to_numpy(float)
        t1.append({
            "model": name, "uses_rpm": fm.USES_RPM[name], "n_features": len(m.features),
            "window_mae_L": mean_absolute_error(yt, pt), "window_rmse_L": float(np.sqrt(mean_squared_error(yt, pt))),
            "window_r2": r2_score(yt, pt),
            "voyage_mae_L": float((vt.pred - vt.true).abs().mean()), "voyage_mape_pct": float(rel.abs().mean()),
            "voyage_bias_median_pct": float(rel.median()), "voyage_abs_err_p95_pct": float(rel.abs().quantile(.95)),
            "conformal90_halfwidth_pct": q[name] * 100, "conformal90_test_coverage": float(cover),
            "test_voyages": len(vt), "cal_voyages": len(vc),
        })
        if name == "greybox":
            gp = dict(zip(fm.GreyBox.p_names, map(float, m.model.params)))
        log(f"{name}: voyage MAPE {rel.abs().mean():.2f}%  q90 {q[name]*100:.2f}%")
    table1 = pd.DataFrame(t1)
    table1.to_csv(out / "table1_voyage_accuracy.csv", index=False, encoding="utf-8-sig")

    # ---------------- Table 2: speed plans replayed on test voyages
    rows_test = rows[rows.voyage_id.isin(test)]
    base_tot = {n: fm.voyage_totals(W["test"], models[n].predict(W["test"])) for n in fm.MODEL_NAMES}
    obs_minutes = base_tot["gbr_paper"].minutes
    t2, per_voyage = [], []
    for plan in ("eco", "fast"):
        rules = {"paper_f^1.15": PAPER_RPM_EXPONENT, f"data_f^{beta:.2f}": beta}
        win_cache = {}
        for rule, x in rules.items():
            win_cache[rule] = fm.build_windows(fm.apply_speed_plan(rows_test, SPEED_STRATEGIES[plan], x), env)
        for n in fm.MODEL_NAMES:
            for rule in (rules if fm.USES_RPM[n] else {"not_used": None}):
                w_cf = win_cache[rule if rule != "not_used" else next(iter(rules))]
                cf = fm.voyage_totals(w_cf, models[n].predict(w_cf))
                b = base_tot[n]
                s = (b.pred - cf.pred) / b.pred * 100
                dmin = cf.minutes - b.minutes
                t2.append({"plan": plan, "model": n, "rpm_rule": rule, "voyages": len(s),
                           "main_saving_median_pct": float(s.median()), "main_saving_p25_pct": float(s.quantile(.25)),
                           "main_saving_p75_pct": float(s.quantile(.75)),
                           "main_saving_mean_L": float((b.pred - cf.pred).mean()),
                           "extra_minutes_median": float(dmin.median()),
                           "saving_L_per_extra_min": float((b.pred - cf.pred).mean() / dmin.mean())})
                per_voyage.append(pd.DataFrame({"plan": plan, "model": n, "rpm_rule": rule, "voyage_id": s.index,
                                                "saving_pct": s.values, "extra_minutes": dmin.values}))
        log(f"plan {plan} done")

    # Observational benchmark: environment-controlled minutes-fuel slope on all typical voyages
    import statsmodels.formula.api as smf

    energy = pd.read_csv(OUTPUT_DIR / f"{BASE}_step3_single_voyage_energy.csv")
    d = energy[energy.voyage_id.isin(ids)].merge(env.reset_index(), on="voyage_id")
    r = smf.ols("fuel_main_voyage ~ main_voyage_minutes + C(direction) + " + " + ".join(fm.ENV_COLS), d).fit(cov_type="HC3")
    slope, (lo, hi) = float(r.params["main_voyage_minutes"]), r.conf_int().loc["main_voyage_minutes"].tolist()
    med_fuel = float(d.fuel_main_voyage.median())
    for plan in ("eco", "fast"):
        dm = float(pd.DataFrame(t2).query("plan == @plan and model == 'greybox'").extra_minutes_median.iloc[0])
        t2.append({"plan": plan, "model": "observed_slope_env_controlled", "rpm_rule": "n/a", "voyages": len(d),
                   "main_saving_median_pct": -slope * dm / med_fuel * 100,
                   "main_saving_p25_pct": -hi * dm / med_fuel * 100, "main_saving_p75_pct": -lo * dm / med_fuel * 100,
                   "main_saving_mean_L": -slope * dm, "extra_minutes_median": dm, "saving_L_per_extra_min": -slope})
    table2 = pd.DataFrame(t2)
    table2.to_csv(out / "table2_speed_plan_savings.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_voyage).to_csv(out / "speed_plan_savings_per_voyage.csv", index=False, encoding="utf-8-sig")

    # along-track wind/current change sign with direction, so the typical conditions are kept per direction
    vdir = fm.voyage_direction()
    env_typical = {d: {c: float(env.loc[[v for v in ids if vdir.get(v) == d], c].median()) for c in fm.ENV_COLS}
                   for d in fm.DIRECTIONS}
    meta = {"split": {"train": sorted(map(int, proper)), "cal": sorted(map(int, cal)), "test": sorted(map(int, test))},
            "conformal_alpha": ALPHA, "conformal_q_rel": q, "rpm_exponent": rpm_exp, "paper_rpm_exponent": PAPER_RPM_EXPONENT,
            "greybox_params": gp, "env_typical_median": env_typical,
            "observed_slope_L_per_min_env_controlled": [slope, lo, hi],
            "features": {n: models[n].features for n in fm.MODEL_NAMES}}
    fm.save_bundle(models, meta)
    (out / "model_comparison_summary.json").write_text(json.dumps(
        {"table1": table1.round(4).to_dict("records"), "table2": table2.round(3).to_dict("records"),
         **{k: meta[k] for k in ("conformal_q_rel", "rpm_exponent", "greybox_params", "env_typical_median",
                                 "observed_slope_L_per_min_env_controlled")}}, indent=1, default=float), encoding="utf-8")
    print(table1.round(3).to_string())
    print(table2.round(2).to_string())
    print(json.dumps({"rpm_exponent": rpm_exp, "greybox": gp}, indent=1))


if __name__ == "__main__":
    main()
