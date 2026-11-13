"""Table 9 (Section 5.3): complete-voyage saving by RPM setpoint on the selected route (2026-11-12).

Uses the Step8 RPM trade-off sweep (monotone LightGBM) and the same accounting as the total-fuel table:
calibrated main voyage + historical median departure + berth arrival model at the plan's approach speed.
Extra time is relative to the balanced plan on the centerline of the selected cluster.

Output: outputs/revision_review/table9_budget_by_rpm.csv
"""
from __future__ import annotations

import json

import pandas as pd

from analyze_maneuvering_phases import predict_arrival
from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR


def main() -> None:
    sweep = pd.read_csv(OUTPUT_DIR / f"{BASE}_step8_route_optimization_rpm_tradeoff.csv")
    total = pd.read_csv(OUTPUT_DIR / f"{BASE}_step8_total_fuel_estimate.csv")
    total = total[total.epsilon_level.eq("medium")].set_index("direction")
    arrival = json.loads((OUTPUT_DIR / "maneuvering_analysis" / "maneuvering_summary.json").read_text(encoding="utf-8"))
    rows = []
    for d, g in sweep[sweep.model.eq("lgbm_mono")].groupby("direction"):
        t, am = total.loc[d], arrival["arrival_model"][d]
        base = g[g.variant_name.eq("centerline") & g.speed_strategy.eq("rpm_+0%")].iloc[0]
        for _, x in g[g.variant_name.eq(t.selected_variant_name)].iterrows():
            v = am["reference"]["approach_speed_kn"] + x.end_speed_kn - base.end_speed_kn
            fuel = (x.fuel_total_pred * t.main_fuel_calibration_factor + t.median_departure_fuel
                    + predict_arrival(am, v, am["throttle_back_m_p50"])["fuel_L"])
            rows.append({"direction": d, "route": t.selected_variant_name, "rpm_change": x.speed_strategy,
                         "extra_minutes": x.time_total_min - base.time_total_min, "complete_fuel_L": fuel,
                         "complete_saving_pct": (1 - fuel / t.historical_total_fuel_median) * 100})
    out = pd.DataFrame(rows).sort_values(["direction", "extra_minutes"])
    path = OUTPUT_DIR / "revision_review" / "table9_budget_by_rpm.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(out.round(2).to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
