"""Voyage-block bootstrap of the complete-voyage fuel saving (2026-11-11, Response to Reviewers Comment 9).

Each replicate redraws the service days (146 local days, both directions) with replacement and keeps every
voyage of a drawn day, so voyages that share a day's wind and sea state stay together (and so do the 30 s
windows of each voyage). The draw is used throughout:
  - the primary fuel model (lgbm_mono) and the speed model h are refitted on the drawn training-split voyages
    (multiplicity as sample weight, same hyper-parameters as Step6b/6c);
  - Step8 scores all candidate plans again and repeats the epsilon-constraint selection (chart constraint fixed);
  - historical medians (main, departure, complete voyage) and the berth arrival models are recomputed.
The complete-voyage fuel follows render_step8_total_fuel_and_route_comparison.build_total_fuel_estimate_table.
Replicate 0 uses the original sample and should reproduce the headline numbers.
Environment medians, candidate paths and chart feasibility are held fixed.

Output: outputs/revision_review/bootstrap_saving_replicates.csv and bootstrap_saving_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import build_step8_fuel_model_route_optimizer as s8
import fuel_models as fm
from analyze_maneuvering_phases import fit_arrival_model, predict_arrival
from build_step6c_rpm_models import H_FEATURES, H_MONO, lgbm
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

OUT = OUTPUT_DIR / "revision_review"
KEYS = ["direction", "cluster_name", "variant_name"]
PARTS = OUT / "bootstrap_parts"  # one CSV per replicate, so an interrupted run resumes


def load_inputs() -> dict:
    sys.argv = sys.argv[:1]
    args = s8.parse_args()
    _, meta = fm.load_bundle()
    ph = pd.read_csv(OUTPUT_DIR / "maneuvering_analysis" / "phase_indicators_per_voyage.csv")
    ph = ph[ph.phase.eq("arrival")].reset_index(drop=True)
    rows = fm.load_main_rows(OUTPUT_DIR / f"{BASE}_step3_enriched.parquet", set(ph.voyage_id.astype(int)))
    win = fm.build_windows(rows, fm.load_env())
    step3 = s8.load_step3(args.step3)
    paths = pd.read_csv(args.step7_paths)
    assign = pd.read_csv(args.step7_assign)
    tau_ref, static = s8.build_cluster_reference(step3, assign, pd.read_csv(args.step4_interp))
    spread = s8.engine_spread_by_tau(step3, set(assign.voyage_id.astype(int)))
    rpm_summary = json.loads((fm.MODEL_DIR / "rpm_models_summary.json").read_text(encoding="utf-8"))
    return {
        "meta": meta, "ph": ph, "win": win, "train": set(meta["split"]["train"]), "paths": paths, "tau_ref": tau_ref,
        "static": static, "spread": spread, "berths": pd.read_csv(args.berths),
        "chart": s8.load_chart_feasibility(args.enc_check, paths, args.min_ukc),
        "q_time": float(rpm_summary["h_time_closure_test"]["conformal90_halfwidth_pct"]) / 100.0,
    }


def draw_counts(ph: pd.DataFrame, b: int, seed: int) -> pd.Series:
    """Number of times each voyage is drawn (day blocks); replicate 0 = original sample."""
    vid = ph.voyage_id.astype(int)
    if b == 0:
        return pd.Series(1, index=vid)
    day = pd.to_datetime(ph.start_time).dt.tz_convert("Asia/Shanghai").dt.date.to_numpy()
    days = np.unique(day)
    drawn = pd.Series(np.random.default_rng([seed, b]).choice(days, len(days))).value_counts()
    return pd.Series(pd.Series(day).map(drawn).fillna(0).astype(int).to_numpy(), index=vid)


def replicate(inp: dict, b: int, seed: int) -> None:
    counts = draw_counts(inp["ph"], b, seed)
    win = inp["win"]
    wt = win.voyage_id.map(counts).fillna(0).to_numpy(float) * win.voyage_id.isin(inp["train"]).to_numpy()
    w = win[wt > 0]
    wt = wt[wt > 0]
    fuel = fm.Wrapped("lgbm_mono")
    fuel.model.set_params(n_jobs=1).fit(w[fuel.features], w["fuel_segment"].astype(float), sample_weight=wt)
    h = lgbm(H_FEATURES, H_MONO).set_params(n_jobs=1).fit(w[H_FEATURES], w["speed_mean_knots"], sample_weight=wt)

    long = s8.evaluate_candidates_windows({"lgbm_mono": fuel}, inp["meta"], inp["paths"], inp["tau_ref"],
                                          inp["static"], inp["berths"], inp["spread"], {"rpm_setpoint": 0.0},
                                          plan_type="rpm", h_bundle={"h_speed": h, "H_FEATURES": H_FEATURES},
                                          strategies=s8.RPM_STRATEGIES, q_time=inp["q_time"])
    cand = long.merge(inp["chart"], on=KEYS, how="left").drop(columns=["model", "rpm_rule"])
    _, summary = s8.apply_joint_selection(cand)

    ph = inp["ph"]
    phb = ph.loc[ph.index.repeat(ph.voyage_id.astype(int).map(counts).fillna(0).astype(int))]
    arrival_models, _ = fit_arrival_model(phb)
    out = []
    for d, hd in phb.groupby("direction"):
        am, s = arrival_models[d], summary[d]["medium"]
        cdir = cand[cand.direction.eq(d)]
        cal = hd.fuel_main_voyage.median() / cdir.fuel_total_pred.median()
        pick = lambda v, st: cdir[cdir.cluster_name.eq(s["selected_cluster_name"]) & cdir.variant_name.eq(v)  # noqa: E731
                                  & cdir.speed_strategy.eq(st)].iloc[0]
        plan, base = pick(s["selected_variant_name"], s["selected_speed_strategy"]), pick("centerline", "balanced")
        v_plan = am["reference"]["approach_speed_kn"] + plan.end_speed_kn - base.end_speed_kn
        hist_total, dep = hd.fuel_underway_total.median(), hd.fuel_departure.median()
        main = plan.fuel_total_pred * cal
        total = main + dep + predict_arrival(am, v_plan, am["throttle_back_m_p50"])["fuel_L"]
        early = main + dep + predict_arrival(am, v_plan, am["throttle_back_m_p75"])["fuel_L"]
        out.append({
            "replicate": b, "direction": d, "selected_variant": s["selected_variant_name"],
            "selected_strategy": s["selected_speed_strategy"],
            "main_saving_vs_balanced_centerline_pct": (1 - plan.fuel_total_pred / base.fuel_total_pred) * 100,
            "extra_minutes": plan.time_total_min - base.time_total_min,
            "hist_total_L": hist_total, "plan_total_L": total, "plan_total_early_tb_L": early,
            "complete_saving_pct": (1 - total / hist_total) * 100,
            "complete_saving_early_tb_pct": (1 - early / hist_total) * 100,
            "arrival_fuel_per_100m_earlier_L": am["fuel_per_100m_earlier_L"],
        })
    pd.DataFrame(out).to_csv(PARTS / f"rep_{b:04d}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    t0 = time.time()
    inp = load_inputs()
    print(f"inputs ready in {time.time() - t0:.0f}s; windows={len(inp['win'])}", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [b for b in range(a.replicates + 1) if not (PARTS / f"rep_{b:04d}.csv").exists()]
    print(f"{len(todo)} replicates to run", flush=True)
    Parallel(n_jobs=a.jobs, verbose=5)(delayed(replicate)(inp, b, a.seed) for b in todo)
    df = pd.concat([pd.read_csv(PARTS / f"rep_{b:04d}.csv") for b in range(a.replicates + 1)], ignore_index=True)
    trip = (df.groupby("replicate")[["hist_total_L", "plan_total_L"]].sum()
            .eval("complete_saving_pct = (1 - plan_total_L / hist_total_L) * 100"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "bootstrap_saving_replicates.csv", index=False, encoding="utf-8-sig")

    def stats(s: pd.Series) -> dict:
        boot = s.drop(index=0, errors="ignore") if s.index.name == "replicate" else s
        return {"point": float(s.iloc[0]), "ci95": [float(boot.quantile(.025)), float(boot.quantile(.975))],
                "sd": float(boot.std())}

    summary = {"replicates": a.replicates, "block": "service day (Asia/Shanghai)", "seed": a.seed, "method": __doc__.split("\n\n")[1], "by_direction": {}}
    for d, g in df.groupby("direction"):
        g = g.set_index("replicate")
        summary["by_direction"][d] = {
            k: stats(g[k]) for k in ("complete_saving_pct", "complete_saving_early_tb_pct",
                                     "main_saving_vs_balanced_centerline_pct", "extra_minutes")}
        sel = (g.selected_variant + " + " + g.selected_strategy).drop(index=0)
        summary["by_direction"][d]["selection_share"] = sel.value_counts(normalize=True).round(3).to_dict()
    summary["round_trip"] = {"complete_saving_pct": stats(trip["complete_saving_pct"])}
    (OUT / "bootstrap_saving_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
