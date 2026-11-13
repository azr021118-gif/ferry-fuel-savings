# Operational fuel savings of a coastal Ro-ro passenger ship

Code for the manuscript "Operational fuel-saving potential of a Ro-ro passenger ship on a repeated coastal route: A multi-source data-driven analysis" (Ocean & Coastal Management, OCMA-D-26-02945).

The scripts reproduce the analysis from the raw onboard records: voyage construction and fuel accounting, environmental matching, the window-level fuel models and their validation, the RPM–speed models, candidate routes with chart screening, the arrival throttle-back model, plan selection, the complete-voyage evaluation, and the day-block bootstrap. Plotting scripts are not included.

## Data

The data are not part of this repository.

- **Onboard AIS and engine-monitoring records** (1 Hz, July–November 2025) belong to the ship operator and cannot be shared. The scripts expect them as CSV parts in `yxlan_20250701_20251130_parts/`, with columns `time`, `latitude`, `longitude`, `SOG`, `COG`, and `ME1_rpm`…`ME4_rpm`, `ME1_fuel`…`ME4_fuel` (fuel rate in L/h).
- **ERA5 and CMEMS** fields are downloaded by `fetch_environment_data.py`. Credentials are read from `~/.cdsapirc` and the `copernicusmarine` toolbox and are not stored here.
- **S-57 electronic navigational charts** of the Swedish Maritime Administration are licensed for research use and are not redistributed. `nautical chart/enc_route_check.py` expects the extracted cells in `nautical chart/extracted/`.

All outputs are written to `outputs/`.

## Environment

Python 3.14 with the packages in `requirements-revision.txt`. The chart check uses a separate environment with GDAL's S-57 driver (`nautical chart/requirements-enc.txt`).

```bash
pip install -r requirements-revision.txt
```

## Run order

| Step | Script | Paper |
| --- | --- | --- |
| 1 | `build_step1_dataset.py` — cleaning, total fuel rate, derived motion variables | §2.1 |
| 2 | `build_step2_segments.py` — berths, voyages, departure / main voyage / arrival | §2.2 |
| 3 | `build_step3_voyage_energy.py` — phase fuel per voyage | §2.3 |
| 4 | `build_step4_main_voyage_optimizer.py` — typical voyages, resampled main-voyage tracks | §2.2 |
| 5 | `build_step7_main_voyage_clustering.py` — track clusters, centerlines, offset routes | §4.2 |
| 6 | `fetch_environment_data.py --combined`, then `build_environment_covariates.py` | §2.1 |
| 7 | `build_step6_segment_fuel_model.py` — 30-s windows | §3.1 |
| 8 | `build_step6b_model_comparison.py` — five fuel models, split-conformal intervals, β | §3.2–3.3 |
| 9 | `build_step6c_rpm_models.py`, `build_step6c_rpm_and_temporal_validation.py` — speed model h, RPM model g, temporal split | §3.3–3.4 |
| 10 | `nautical chart/enc_route_check.py` — under-keel clearance of the candidate routes | §4.2 |
| 11 | `analyze_maneuvering_phases.py` — berth arrival models | §4.3 |
| 12 | `build_step8_fuel_model_route_optimizer.py` — RPM setpoint plans, plan selection | §4.1, §4.4 |
| 13 | `build_revision_rpm_setpoint_figdata.py`, `analyze_slack_and_weather.py` — baselines, weather terciles | §4.1, §5.2 |
| 14 | `build_revision_bootstrap_saving.py` — day-block bootstrap of the saving | §5.1 |
| 15 | `build_revision_budget_table.py` — saving by passage-time budget | §5.3 |

`fuel_models.py` holds the shared model definitions, and `project_paths.py` the folder layout. `analyze_maneuvering_indicator_sensitivity.py` reproduces the screening of the maneuvering indicators.
