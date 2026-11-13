"""Attach wind, current, wave and sea-level covariates to every voyage's main phase.

Inputs: outputs from fetch_environment_data.py, Step3 voyage energy table,
Step4 per-voyage interpolated main-voyage tracks, Step7 centreline (fallback).
Each track point gets a time (main-phase start + tau * duration), values from the
nearest *wet* CMEMS cell (linear in time) and the nearest ERA5 sea point.
Along-track components use the local track bearing; positive = with the ship.
These are model/reanalysis values, not on-board measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from project_paths import DEFAULT_DATASET_BASE as BASE, OUTPUT_DIR

ENV = OUTPUT_DIR / "environment"
MS_TO_KN = 1.0 / 0.514444
LAT0 = 56.14


def local_xy(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (np.radians(lon) * 6371000 * np.cos(np.radians(LAT0)),
            np.radians(lat) * 6371000)


def bearings(lat, lon):
    x, y = local_xy(lat, lon)
    dx, dy = np.diff(x, axis=-1), np.diff(y, axis=-1)
    b = np.arctan2(dx, dy)  # radians clockwise from north
    return np.concatenate([b, b[..., -1:]], axis=-1)


def nearest_wet(ds, var, lat, lon):
    da = ds[var]
    wet = da.notnull().any("time").values
    glat, glon = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    gx, gy = local_xy(glat[wet], glon[wet])
    px, py = local_xy(lat.ravel(), lon.ravel())
    dist, k = cKDTree(np.c_[gx, gy]).query(np.c_[px, py])
    iy, ix = np.nonzero(wet)
    return iy[k].reshape(lat.shape), ix[k].reshape(lat.shape), dist.reshape(lat.shape)


def sample(ds, var, iy, ix, times):
    """Linear-in-time sample of ds[var] at cells (iy, ix) and numpy datetime64 times."""
    arr = ds[var].values
    t = ds.time.values.astype("datetime64[s]").astype(np.int64).astype(float)
    q = times.astype("datetime64[s]").astype(np.int64).astype(float)
    pos = np.interp(q, t, np.arange(len(t)))
    i0 = np.clip(np.floor(pos).astype(int), 0, len(t) - 1)
    i1 = np.clip(i0 + 1, 0, len(t) - 1)
    w = pos - i0
    return arr[i0, iy, ix] * (1 - w) + arr[i1, iy, ix] * w


def sample_dir(ds, var, iy, ix, times):
    rad = np.radians(ds[var])
    tmp = xr.Dataset({"s": np.sin(rad), "c": np.cos(rad)}, coords=ds.coords)
    return np.arctan2(sample(tmp, "s", iy, ix, times), sample(tmp, "c", iy, ix, times))


def load_era5():
    """Merge the per-month, per-stepType NetCDF files written by fetch_environment_data.py."""
    months = []
    folders = sorted(p for p in ENV.glob("era5_2025*") if p.is_dir() and any(p.glob("*.nc")))
    for folder in folders:
        parts = []
        for f in sorted(folder.glob("*.nc")):
            with xr.open_dataset(f) as d:
                d = d.rename({"valid_time": "time"}) if "valid_time" in d.dims else d
                parts.append(d.drop_vars([c for c in ("expver", "number") if c in d.coords]).load())
        if parts:
            months.append(xr.merge(parts))
    if not months:
        raise FileNotFoundError("Run fetch_environment_data.py --only era5 first.")
    ds = xr.concat(months, "time").sortby("time")
    ds = ds.isel(time=~ds.get_index("time").duplicated())
    return ds, [p.name for p in folders]


def covered(ds, times, max_gap_h=1.5):
    """True where a dataset time step lies within max_gap_h of the query time, so
    missing months are left NaN instead of being bridged or clamped."""
    t = ds.time.values.astype("datetime64[s]").astype(np.int64)
    q = times.astype("datetime64[s]").astype(np.int64)
    k = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    near = np.minimum(np.abs(q - t[k - 1]), np.abs(q - t[k]))
    return near <= max_gap_h * 3600


def main() -> None:
    energy = pd.read_csv(OUTPUT_DIR / f"{BASE}_step3_single_voyage_energy.csv")
    start = pd.to_datetime(energy.start_time, utc=True) + pd.to_timedelta(energy.departure_minutes, unit="min")
    energy["main_start"] = start
    energy["main_end"] = start + pd.to_timedelta(energy.main_voyage_minutes, unit="min")

    interp = pd.read_csv(OUTPUT_DIR / f"{BASE}_step4_main_route_interpolated.csv",
                         usecols=["tau", "latitude", "longitude", "voyage_id", "direction"])
    paths = pd.read_csv(OUTPUT_DIR / f"{BASE}_step7_main_route_clusters_candidate_paths.csv")
    cl = paths.loc[paths.variant_name.eq("centerline")]
    # one centreline per direction (first cluster), used only when a voyage has no interpolated track
    centres = {d: g.loc[g.cluster_name.eq(g.cluster_name.min())].sort_values("tau") for d, g in cl.groupby("direction")}
    centre = next(iter(centres.values()))
    taus = np.round(centre.tau.to_numpy(), 8)
    P = len(taus)

    tracks = {v: g.sort_values("tau") for v, g in interp.dropna(subset=["latitude", "longitude"]).groupby("voyage_id")}
    vids = energy.voyage_id.astype(int).to_numpy()
    lat = np.empty((len(vids), P))
    lon = np.empty((len(vids), P))
    source = []
    for i, v in enumerate(vids):
        g = tracks.get(v)
        if g is not None and len(g) >= 2:
            lat[i] = np.interp(taus, g.tau, g.latitude)
            lon[i] = np.interp(taus, g.tau, g.longitude)
            source.append("own_track")
        else:
            c = centres[energy.direction.iloc[i]]
            lat[i], lon[i] = np.interp(taus, c.tau, c.path_lat), np.interp(taus, c.tau, c.path_lon)
            source.append("centreline")
    t0 = energy.main_start.dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
    dur = (energy.main_voyage_minutes.to_numpy() * 60e9).astype("timedelta64[ns]")
    times = t0[:, None] + (taus[None, :] * dur[:, None].astype(float)).astype("timedelta64[ns]")
    brg = bearings(lat, lon)
    sb, cb = np.sin(brg), np.cos(brg)

    phy = xr.open_dataset(ENV / "cmems_bal_phy_15min_20250701_20251130.nc").load()
    wav = xr.open_dataset(ENV / "cmems_bal_wav_1h_20250701_20251130.nc").load()
    iy, ix, dist_phy = nearest_wet(phy, "uo", lat, lon)
    u, v = sample(phy, "uo", iy, ix, times), sample(phy, "vo", iy, ix, times)
    sla = sample(phy, "sla", iy, ix, times)
    wy, wx, dist_wav = nearest_wet(wav, "VHM0", lat, lon)
    hs = sample(wav, "VHM0", wy, wx, times)
    mwd = sample_dir(wav, "VMDR", wy, wx, times)  # direction waves come FROM

    era, era_months = load_era5()
    lsm = era["lsm"].isel(time=0).values
    sea = lsm < 0.5
    glat, glon = np.meshgrid(era.latitude.values, era.longitude.values, indexing="ij")
    gx, gy = local_xy(glat[sea], glon[sea])
    px, py = local_xy(lat.ravel(), lon.ravel())
    d_era, k = cKDTree(np.c_[gx, gy]).query(np.c_[px, py])
    ey, ex = (a[k].reshape(lat.shape) for a in np.nonzero(sea))
    u10, v10 = sample(era, "u10", ey, ex, times), sample(era, "v10", ey, ex, times)
    gust = sample(era, "fg10", ey, ex, times) if "fg10" in era else np.full(lat.shape, np.nan)
    gap = ~covered(era, times)  # months not yet downloaded stay NaN instead of being clamped
    for a in (u10, v10, gust):
        a[gap] = np.nan
    for ds_, arrs in ((phy, (u, v, sla)), (wav, (hs, mwd))):
        g = ~covered(ds_, times)
        for a in arrs:
            a[g] = np.nan

    cur_along = u * sb + v * cb                    # m/s, + = following current
    cur_cross = u * cb - v * sb
    wind_along = u10 * sb + v10 * cb               # m/s, + = tailwind
    wind_cross = u10 * cb - v10 * sb
    rel_wave = np.cos(mwd - brg)                   # +1 = head seas (waves from ahead)

    out = pd.DataFrame({
        "voyage_id": vids,
        "track_source": source,
        "main_start_utc": energy.main_start.astype(str).to_numpy(),
        "month": energy.main_start.dt.strftime("%Y-%m").to_numpy(),
        "hour_utc": energy.main_start.dt.hour.to_numpy(),
        "sog_main_mean_kn": energy.distance_main_voyage_nm.to_numpy() / (energy.main_voyage_minutes.to_numpy() / 60),
        "current_along_mean_kn": np.nanmean(cur_along, 1) * MS_TO_KN,
        "current_cross_abs_mean_kn": np.nanmean(np.abs(cur_cross), 1) * MS_TO_KN,
        "current_speed_mean_kn": np.nanmean(np.hypot(u, v), 1) * MS_TO_KN,
        "headwind_mean_ms": -np.nanmean(wind_along, 1),
        "crosswind_abs_mean_ms": np.nanmean(np.abs(wind_cross), 1),
        "wind_speed_mean_ms": np.nanmean(np.hypot(u10, v10), 1),
        "gust_max_ms": np.nanmax(gust, 1),
        "hs_mean_m": np.nanmean(hs, 1),
        "hs_head_component_m": np.nanmean(hs * rel_wave, 1),
        "sla_mean_m": np.nanmean(sla, 1),
    })
    out["stw_main_mean_kn_approx"] = out.sog_main_mean_kn - out.current_along_mean_kn
    target = ENV / f"{BASE}_voyage_environment.csv"
    out.to_csv(target, index=False, encoding="utf-8-sig")

    meta = {
        "voyages": int(len(out)),
        "track_source_counts": out.track_source.value_counts().to_dict(),
        "track_points_per_voyage": P,
        "cmems_phy_dataset": "cmems_mod_bal_phy_anfc_PT15M-i (version 202411, retires 2027-01-12)",
        "cmems_wav_dataset": "cmems_mod_bal_wav_anfc_PT1H-i (version 202311, retires 2027-01-12)",
        "era5_dataset": "reanalysis-era5-single-levels, 0.25 deg, sea points (lsm < 0.5)",
        "era5_months_available": era_months,
        "distance_to_nearest_wet_cmems_cell_m": {q: float(np.quantile(dist_phy, p)) for q, p in
                                                 (("p50", .5), ("p95", .95), ("max", 1.0))},
        "share_points_cmems_cell_beyond_1km": float((dist_phy > 1000).mean()),
        "distance_to_nearest_era5_sea_point_m": {q: float(np.quantile(d_era, p)) for q, p in
                                                 (("p50", .5), ("max", 1.0))},
        "distinct_cmems_cells_used": int(len(set(zip(iy.ravel(), ix.ravel())))),
        "distinct_era5_points_used": int(len(set(zip(ey.ravel(), ex.ravel())))),
        "missing_share": {c: float(out[c].isna().mean()) for c in out.columns if out[c].dtype.kind == "f"},
        "note": "Model/reanalysis covariates, not on-board measurements. STW is SOG minus mean along-track current.",
    }
    (ENV / f"{BASE}_voyage_environment_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
