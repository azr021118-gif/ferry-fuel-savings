"""Timetable slack and weather-stratified RPM-setpoint trade-offs (revision scenario analysis).

Part 1  Timetable slack for the 1,119 typical berth_2 -> berth_1 voyages.
        The schedule is inferred from the data: berth_2 departures cluster at HH:00 and berth_1
        departures at HH:30 local time. For each voyage:
          buffer  = scheduled next departure from berth_1 - actual arrival (end of arrival phase)
          slack   = buffer - minimum turnaround
        The minimum turnaround is the median berth_1 dwell when the next departure left late
        (> 1 min after schedule), i.e. the ship left as soon as it was ready.
        A slack-aware policy picks, per voyage, the largest RPM reduction whose extra passage
        time fits the slack (nominal, and with the 90 % time-interval margin).
Part 2  RPM-setpoint plans replayed on held-out voyages (test + calibration, 403 voyages)
        by terciles of headwind and head-sea wave component.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fuel_models as fm
from build_step6c_rpm_models import apply_row_factors, row_window_id
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

OUT = OUTPUT_DIR / "scenario_analysis"
TZ = "Europe/Stockholm"
LEVELS = [0.99, 0.98, 0.97, 0.96, 0.95]
PLANS = {**{f"rpm_{(x - 1) * 100:+.0f}%": (x, x, x) for x in LEVELS}, "eco_setpoint": (0.96, 0.95, 0.97)}


def nearest_slot(ts: pd.Series, minute: int) -> pd.Series:
    """Scheduled departure slot (HH:minute local) nearest to the actual departure."""
    loc = ts.dt.tz_convert(TZ)
    base = loc.dt.floor("h") + pd.Timedelta(minutes=minute)
    cands = pd.concat([base - pd.Timedelta(hours=1), base, base + pd.Timedelta(hours=1)], axis=1)
    diff = cands.sub(loc, axis=0).abs()
    idx = diff.to_numpy().argmin(axis=1)
    return pd.Series(cands.to_numpy()[np.arange(len(cands)), idx], index=ts.index)


def part1_slack(ids: set[int]) -> dict:
    v = pd.read_csv(OUTPUT_DIR / f"{BASE}_step2_voyage_summary.csv", parse_dates=["start_time", "end_time"])
    v = v[v.start_berth != v.end_berth].sort_values("start_time").reset_index(drop=True)
    sched = pd.Series(pd.NaT, index=v.index, dtype="datetime64[ns, UTC]")
    for berth, minute in (("berth_2", 0), ("berth_1", 30)):
        m = v.start_berth.eq(berth)
        sched.loc[m] = pd.to_datetime(nearest_slot(v.loc[m, "start_time"], minute), utc=True)
    v["sched"] = sched
    v["dep_delay_min"] = (v.start_time - v.sched).dt.total_seconds() / 60
    nxt = v.shift(-1)
    d = v[(v.start_berth == "berth_2") & v.voyage_id.isin(ids)].copy()
    d["next_from"], d["next_start"], d["next_sched"], d["next_delay"] = (
        nxt.loc[d.index, "start_berth"], nxt.loc[d.index, "start_time"], nxt.loc[d.index, "sched"], nxt.loc[d.index, "dep_delay_min"])
    d["dwell_min"] = (d.next_start - d.end_time).dt.total_seconds() / 60
    # consecutive service: next leg leaves berth_1 at the slot 30 min after this departure's slot
    d["consecutive"] = d.next_from.eq("berth_1") & ((d.next_sched - d.sched).dt.total_seconds().round() == 1800)
    c = d[d.consecutive].copy()
    late = c[c.next_delay > 1.0]
    t_min = float(late.dwell_min.median())
    c["buffer_min"] = (c.next_sched - c.end_time).dt.total_seconds() / 60
    c["slack_min"] = c.buffer_min - t_min

    trade = pd.read_csv(OUTPUT_DIR / f"{BASE}_step8_route_optimization_rpm_tradeoff.csv")
    cl = trade[trade.variant_name.eq("centerline") & trade.model.eq("lgbm_mono")].set_index("speed_strategy")
    rpm_summary = json.loads((fm.MODEL_DIR / "rpm_models_summary.json").read_text(encoding="utf-8"))
    q_time = rpm_summary["h_time_closure_test"]["conformal90_halfwidth_pct"] / 100
    margin = q_time * float(cl.loc["rpm_+0%", "time_total_min"])
    ladder = [("rpm_+0%", 0.0, 0.0)] + [(f"rpm_{(x - 1) * 100:+.0f}%", float(cl.loc[f"rpm_{(x - 1) * 100:+.0f}%", "extra_minutes_vs_centerline_rpm0"]),
                                         float(cl.loc[f"rpm_{(x - 1) * 100:+.0f}%", "main_saving_vs_centerline_rpm0_pct"])) for x in LEVELS]

    def policy(slack, extra_margin):
        lvl, sav = [], []
        for s in slack:
            best = ladder[0]
            for step in ladder[1:]:
                if step[1] + extra_margin <= s:
                    best = step
            lvl.append(best[0])
            sav.append(best[2])
        return pd.Series(lvl, index=slack.index), np.array(sav)

    # The minimum turnaround cannot be pinned down from AIS alone: sensitivity over plausible values
    sens = []
    for label, tm in (("dwell_p05", c.dwell_min.quantile(.05)), ("dwell_p10", c.dwell_min.quantile(.10)),
                      ("dwell_p25", c.dwell_min.quantile(.25)), ("dwell_p50", c.dwell_min.quantile(.50)),
                      ("late_departure_median", t_min)):
        sl = c.buffer_min - tm
        _, sv = policy(sl, 0.0)
        sens.append({"turnaround_assumption": label, "turnaround_min": float(tm), "slack_median_min": float(sl.median()),
                     "share_slack_ge_0.50": float((sl >= 0.5).mean()), "share_slack_ge_0.83": float((sl >= 0.83).mean()),
                     "policy_mean_main_saving_pct": float(sv.mean())})
    # the opposite end of the round trip (berth_2, departures at HH:00)
    b = v[(v.start_berth == "berth_1") & nxt.start_berth.eq("berth_2")].copy()
    b["buffer2"] = (nxt.loc[b.index, "sched"] - b.end_time).dt.total_seconds() / 60
    b["dwell2"] = (nxt.loc[b.index, "start_time"] - b.end_time).dt.total_seconds() / 60
    b = b[b.buffer2.between(-5, 30)]

    res = {}
    for name, mg in (("nominal", 0.0), ("with_time_pi90_margin", margin)):
        lvl, sav = policy(c.slack_min, mg)
        res[name] = {"margin_min": mg, "level_share": lvl.value_counts(normalize=True).round(4).to_dict(),
                     "mean_main_saving_pct": float(sav.mean()),
                     "share_any_reduction": float((lvl != "rpm_+0%").mean())}
        c[f"policy_{name}"] = lvl.to_numpy()
    q = [.05, .1, .25, .5, .75, .9]
    c["arr_local_hour"] = c.end_time.dt.tz_convert(TZ).dt.hour
    out = {
        "schedule": {"berth_2_departures_within_5min_of_HH00": float(v[v.start_berth == "berth_2"].dep_delay_min.between(-5, 5).mean()),
                     "berth_1_departures_within_5min_of_HH30": float(v[v.start_berth == "berth_1"].dep_delay_min.between(-5, 5).mean()),
                     "own_departure_delay_min_quantiles": d.dep_delay_min.quantile(q).round(2).to_dict()},
        "voyages": int(len(d)), "consecutive_voyages": int(len(c)),
        "min_turnaround_min": t_min, "min_turnaround_n_late_cases": int(len(late)),
        "dwell_min_quantiles_consecutive": c.dwell_min.quantile(q).round(2).to_dict(),
        "buffer_min_quantiles": c.buffer_min.quantile(q).round(2).to_dict(),
        "slack_min_quantiles": c.slack_min.quantile(q).round(2).to_dict(),
        "share_slack_ge": {f"{x:.2f}": float((c.slack_min >= x).mean()) for x in (0, 0.13, 0.28, 0.5, 0.65, 0.83, 1.0, 2.0)},
        "next_departure_late_gt1min_share": float((c.next_delay > 1).mean()),
        "slack_by_local_arrival_hour_median": c.groupby("arr_local_hour").slack_min.median().round(2).to_dict(),
        "ladder_centerline_lgbm_mono": [{"level": a, "extra_min": b, "main_saving_pct": s} for a, b, s in ladder],
        "policy": res,
        "turnaround_sensitivity": sens,
        "berth_2_end": {"legs": int(len(b)), "buffer_min_quantiles": b.buffer2.quantile(q).round(2).to_dict(),
                        "dwell_min_quantiles": b.dwell2.quantile(q).round(2).to_dict()},
    }
    c[["voyage_id", "start_time", "end_time", "sched", "dep_delay_min", "next_sched", "next_delay", "dwell_min", "buffer_min",
       "slack_min", "policy_nominal", "policy_with_time_pi90_margin"]].to_csv(OUT / "slack_per_voyage.csv", index=False,
                                                                              encoding="utf-8-sig")
    return out


def part2_weather() -> pd.DataFrame:
    import joblib

    models, meta = fm.load_bundle()
    hb = joblib.load(fm.MODEL_DIR / "rpm_speed_models.joblib")
    h, hf = hb["h_speed"], hb["H_FEATURES"]
    held = set(meta["split"]["test"]) | set(meta["split"]["cal"])
    rows = fm.load_main_rows(OUTPUT_DIR / f"{BASE}_step3_enriched.parquet", held)
    env = fm.load_env()
    w = fm.build_windows(rows, env)
    rows = rows.copy()
    rows["_wid"] = row_window_id(rows).to_numpy()
    key = pd.MultiIndex.from_arrays([rows.voyage_id, rows._wid])
    widx = w.set_index(["voyage_id", "segment_window_id"]).index
    base = {n: fm.voyage_totals(w, models[n].predict(w)) for n in ("lgbm_mono", "greybox", "lgbm_mono_norpm")}
    voy_env = env.loc[sorted(held)]
    groups = {"headwind": pd.qcut(voy_env.headwind_mean_ms, 3, labels=["tail/low", "mid", "head/high"]),
              "hs_head": pd.qcut(voy_env.hs_head_component_m, 3, labels=["following", "mid", "head"])}
    recs = []
    for plan, rf in PLANS.items():
        r_w = fm.plan_factor(w.segment_progress_mean.to_numpy(), rf)
        x = w.copy()
        x["rpm_mean_seg"] *= r_w
        s_w = np.clip(h.predict(x[hf]) / h.predict(w[hf]), 0.8, 1.2)
        s_row = pd.Series(s_w, index=widx).reindex(key).fillna(1.0).to_numpy()
        r_row = pd.Series(r_w, index=widx).reindex(key).fillna(1.0).to_numpy()
        wc = fm.build_windows(apply_row_factors(rows, s_row, r_row), env)
        for n, b in base.items():
            cf = fm.voyage_totals(wc, models[n].predict(wc))
            df = pd.DataFrame({"saving_pct": (b.pred - cf.pred) / b.pred * 100, "saving_L": b.pred - cf.pred,
                               "extra_min": cf.minutes - b.minutes})
            for gname, g in groups.items():
                for lab, sub in df.join(g.rename("grp")).groupby("grp", observed=True):
                    recs.append({"plan": plan, "model": n, "stratifier": gname, "tercile": str(lab), "voyages": len(sub),
                                 "env_median": float(voy_env.loc[sub.index, "headwind_mean_ms" if gname == "headwind" else "hs_head_component_m"].median()),
                                 "base_fuel_median_L": float(b.pred.loc[sub.index].median()),
                                 "saving_median_pct": float(sub.saving_pct.median()), "saving_p25_pct": float(sub.saving_pct.quantile(.25)),
                                 "saving_p75_pct": float(sub.saving_pct.quantile(.75)), "extra_min_median": float(sub.extra_min.median()),
                                 "L_per_extra_min": float(sub.saving_L.mean() / sub.extra_min.mean())})
        print(plan, "done", flush=True)
    return pd.DataFrame(recs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    step4 = pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_candidates.csv").query("direction == 'berth_2_to_berth_1'")
    s = part1_slack(set(step4.voyage_id.astype(int)))
    (OUT / "slack_summary.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: s[k] for k in ("turnaround_sensitivity", "berth_2_end")}, indent=1, default=str), flush=True)
    import sys

    if "--slack-only" in sys.argv:
        return
    wt = part2_weather()
    wt.to_csv(OUT / "weather_stratified_rpm_plans.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = wt[wt.plan.isin(["rpm_-3%", "rpm_-5%", "eco_setpoint"])]
    print(show.pivot_table(index=["stratifier", "tercile", "plan"], columns="model",
                           values="saving_median_pct").round(2).join(
        show[show.model == "lgbm_mono"].set_index(["stratifier", "tercile", "plan"])[["env_median", "extra_min_median", "L_per_extra_min", "voyages"]].round(3)).to_string())


if __name__ == "__main__":
    main()
