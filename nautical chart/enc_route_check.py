"""Check YXLAN route variants against the Swedish S-57 ENC cells.

Run with the chart venv:
    "nautical chart/.venv/Scripts/python.exe" "nautical chart/enc_route_check.py"

For each route variant (default: Step7 candidate paths) it builds the swept
corridor (half beam + lateral margin), finds the shallowest charted depth
inside it from the best-scale cell available at each place (DEPARE DRVAL1,
SOUNDG, VALSOU of rocks/obstructions/wrecks), and computes the under-keel
clearance

    UKC = charted depth - low-water allowance - draught - squat

with Barrass squat  S = k * Cb * V^2 / 100  (k = 1 open water, 2 confined).
It also reports clearance to land / rocks / wrecks / <3 m areas and writes a
map. The chart is a 2021 edition (Sjöfartsverket, research licence) and is
not for navigation.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("OGR_S57_OPTIONS", "SPLIT_MULTIPOINT=ON,ADD_SOUNDG_DEPTH=ON")
warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

HERE = Path(__file__).resolve().parent
SHIP = HERE.parent
ENC_DIR = HERE / "extracted"
UTM = 32633
# usage band -> cell, best (largest scale) first
CELLS = {6: "SE6EI0XQ.000", 5: "SE500001.000", 4: "SE4EI0XQ.000", 3: "SE3DI0XQ.000"}
HAZARD_LAYERS = ("UWTROC", "OBSTRN", "WRECKS")
MS_TO_KN = 3600 / 1852
DEFAULT_VARIANTS = SHIP / "outputs" / "yxlan_20250701_20251130_step7_main_route_clusters_candidate_paths.csv"
DEFAULT_SPEED = SHIP / "outputs" / "yxlan_20250701_20251130_step4_main_route_centerline.csv"


def read(cell: str, layer: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(ENC_DIR / cell, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=4326)


def best_scale_layers(layers: tuple[str, ...]) -> dict[str, gpd.GeoDataFrame]:
    """Merge each layer across cells, keeping only the best-scale data at each place."""
    taken = None
    out: dict[str, list] = {l: [] for l in layers}
    meta = []
    for band, cell in CELLS.items():
        cov = read(cell, "M_COVR")
        cov = cov[cov["CATCOV"] == 1].to_crs(UTM).union_all()
        own = cov if taken is None else cov.difference(taken)
        dsid = read(cell, "DSID")
        meta.append(dict(cell=cell, band=band, edition=dsid["DSID_EDTN"].iloc[0], issue=dsid["DSID_ISDT"].iloc[0],
                         scale=dsid["DSPM_CSCL"].iloc[0], vdat=dsid.get("DSPM_VDAT", pd.Series([None])).iloc[0],
                         sdat=dsid.get("DSPM_SDAT", pd.Series([None])).iloc[0]))
        for l in layers:
            g = read(cell, l)
            if len(g):
                g = g.to_crs(UTM)
                g = g[g.intersects(own)]
                g = gpd.clip(g, own) if len(g) else g  # keep only the part not covered by a better cell
                g = g[~g.geometry.is_empty].copy()
                g["band"] = band
                out[l].append(g)
        taken = cov if taken is None else taken.union(cov)
    merged = {l: gpd.GeoDataFrame(pd.concat(v, ignore_index=True), crs=UTM) if v else gpd.GeoDataFrame(geometry=[], crs=UTM)
              for l, v in out.items()}
    merged["_meta"] = pd.DataFrame(meta)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    ap.add_argument("--speed", type=Path, default=DEFAULT_SPEED, help="Step4 centerline with sog_mean per tau")
    ap.add_argument("--draught", type=float, default=3.4, help="m (S&P Global, IMO 8804440)")
    ap.add_argument("--beam", type=float, default=12.41, help="moulded breadth, m")
    ap.add_argument("--lateral-margin", type=float, default=10.0, help="m added to half beam for track uncertainty")
    ap.add_argument("--low-water", type=float, default=0.6, help="m below chart datum to allow for low sea level")
    ap.add_argument("--cb", type=float, default=0.65, help="assumed block coefficient for squat")
    ap.add_argument("--squat-k", type=float, default=2.0, help="Barrass factor: 1 open water, 2 confined")
    ap.add_argument("--speed-factor", type=float, default=1.0, help="scale sog (e.g. 1.05 for a fast plan)")
    ap.add_argument("--skip-ends", type=float, default=0.05, help="tau excluded at each berth for clearance stats")
    ap.add_argument("--out", type=Path, default=SHIP / "outputs" / "enc")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    L = best_scale_layers(("DEPARE", "SOUNDG", "LNDARE") + HAZARD_LAYERS)
    meta = L.pop("_meta")
    print("Cells used:\n" + meta.to_string(index=False))

    dep = L["DEPARE"]
    shoal = dep[dep["DRVAL1"] < 3]
    points = [L["SOUNDG"].assign(depth=L["SOUNDG"]["DEPTH"], kind="SOUNDG")[["geometry", "depth", "kind"]]]
    for h in HAZARD_LAYERS:
        g = L[h]
        if len(g):
            vs = g["VALSOU"] if "VALSOU" in g else pd.Series(np.nan, index=g.index)
            points.append(g.assign(depth=vs, kind=h)[["geometry", "depth", "kind"]])
    depth_pts = gpd.GeoDataFrame(pd.concat(points, ignore_index=True), crs=UTM)

    paths = pd.read_csv(a.variants)
    speed = pd.read_csv(a.speed)
    half = a.beam / 2 + a.lateral_margin
    rows, seg_rows = [], []
    for (dirn, clu, var), g in paths.groupby(["direction", "cluster_name", "variant_name"]):
        g = g.sort_values("tau").reset_index(drop=True)
        pts = gpd.GeoSeries(gpd.points_from_xy(g.path_lon, g.path_lat), crs=4326).to_crs(UTM)
        sp = speed[speed.direction == dirn].sort_values("tau")
        # Step4 sog_mean is in m/s -> knots for Barrass
        v = np.interp(g.tau, sp.tau, sp.sog_mean) * MS_TO_KN * a.speed_factor
        for i in range(len(g) - 1):
            seg = LineString([pts.iloc[i], pts.iloc[i + 1]])
            corr = seg.buffer(half, cap_style="flat")
            hit = dep[dep.intersects(corr)]
            # conservative (ECDIS-style): lower bound of the shallowest depth area touched
            d_area = hit["DRVAL1"].min()
            # area-weighted share of the corridor charted shallower than 6 m
            inter = hit.intersection(corr).area
            shallow_share = inter[hit["DRVAL1"] < 6].sum() / corr.area
            # actual charted values: soundings and VALSOU of rocks/obstructions/wrecks
            d_pt = depth_pts[depth_pts.within(corr)]["depth"].min()
            vmax = max(v[i], v[i + 1])
            squat = a.squat_k * a.cb * vmax ** 2 / 100
            seg_rows.append(dict(direction=dirn, cluster_name=clu, variant=var, tau=g.tau[i], sog_kn=vmax,
                                 depth_area_lower_m=d_area, shallow_lt6m_share=shallow_share,
                                 depth_sounding_m=d_pt, squat_m=squat,
                                 ukc_area_m=d_area - a.low_water - a.draught - squat,
                                 ukc_sounding_m=d_pt - a.low_water - a.draught - squat))
        s = pd.DataFrame([r for r in seg_rows if (r["direction"], r["cluster_name"], r["variant"]) == (dirn, clu, var)])
        inner = g[(g.tau > a.skip_ends) & (g.tau < 1 - a.skip_ends)]
        il = gpd.GeoSeries(gpd.points_from_xy(inner.path_lon, inner.path_lat), crs=4326).to_crs(UTM)
        iline = LineString(list(il))
        si = s[(s.tau > a.skip_ends) & (s.tau < 1 - a.skip_ends)]
        ws = si.loc[si.ukc_sounding_m.idxmin()] if si.ukc_sounding_m.notna().any() else None
        rows.append(dict(direction=dirn, cluster_name=clu, variant=var,
                         max_sog_kn=si.sog_kn.max(), max_squat_m=si.squat_m.max(),
                         min_depth_area_lower_m=si.depth_area_lower_m.min(),
                         min_ukc_area_m=si.ukc_area_m.min(),
                         corridor_share_lt6m=float(np.average(si.shallow_lt6m_share)),
                         n_segments_touch_lt6m=int((si.depth_area_lower_m < 6).sum()),
                         min_sounding_m=si.depth_sounding_m.min(),
                         min_ukc_sounding_m=None if ws is None else ws.ukc_sounding_m,
                         tau_min_ukc_sounding=None if ws is None else ws.tau,
                         clear_land_m=L["LNDARE"].distance(iline).min(),
                         clear_rock_m=L["UWTROC"].distance(iline).min() if len(L["UWTROC"]) else np.nan,
                         clear_wreck_m=L["WRECKS"].distance(iline).min() if len(L["WRECKS"]) else np.nan,
                         clear_lt3m_area_m=shoal.distance(iline).min()))
    res = pd.DataFrame(rows).round(2)
    pd.DataFrame(seg_rows).round(3).to_csv(a.out / "enc_route_ukc_segments.csv", index=False)
    res.to_csv(a.out / "enc_route_check_summary.csv", index=False)
    meta.to_csv(a.out / "enc_cells_used.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\nT={a.draught} m, B={a.beam} m, corridor half-width {half:.1f} m, low water -{a.low_water} m, "
          f"squat {a.squat_k}*Cb({a.cb})*V^2/100, speed x{a.speed_factor}")
    print(res.to_string(index=False))

    # map on the 1:12000 cell
    bb = box(15.545, 56.120, 15.610, 56.168)
    clip = lambda g: g[g.intersects(bb)].clip(bb)
    c5 = CELLS[5]
    d5, l5 = clip(read(c5, "DEPARE")), clip(read(c5, "LNDARE"))
    cols = {"centerline": "k", "port_shift_medium": "tab:red", "port_shift_small": "tab:orange",
            "starboard_shift_small": "tab:cyan", "starboard_shift_medium": "tab:blue"}
    styles = ["-", "--", ":"]
    # one map per direction; berth_2_to_berth_1 keeps the original file name
    for dirn in sorted(paths.direction.unique(), reverse=True):
        fig, ax = plt.subplots(figsize=(7, 8))
        d5.assign(m=d5.DRVAL1.clip(lower=0)).plot(ax=ax, column="m", cmap="Blues", vmin=-3, vmax=20, edgecolor="#9db", linewidth=.2)
        l5.plot(ax=ax, color="#e8dcb5", edgecolor="#a99", linewidth=.3)
        clip(read(c5, "UWTROC")).plot(ax=ax, color="k", markersize=3, marker="+")
        clip(read(c5, "FERYRT")).plot(ax=ax, color="m", lw=1.2, ls="--", label="FERYRT (charted ferry route)")
        pd_ = paths[paths.direction == dirn]
        clusters = sorted(pd_.cluster_name.unique())
        for (clu, var), g in pd_.groupby(["cluster_name", "variant_name"]):
            g = g.sort_values("tau")
            u = res[(res.direction == dirn) & (res.cluster_name == clu) & (res.variant == var)].min_ukc_sounding_m.iloc[0]
            tag = f"c{clusters.index(clu) + 1} " if len(clusters) > 1 else ""
            ax.plot(g.path_lon, g.path_lat, color=cols.get(var, "grey"), lw=1.1, ls=styles[clusters.index(clu) % 3],
                    label=f"{tag}{var} (min UKC {u:.1f} m)")
        ax.set_xlim(15.545, 15.610); ax.set_ylim(56.120, 56.168)
        ax.set_aspect(1 / np.cos(np.radians(56.14)))
        ax.set_xlabel("Longitude [°]"); ax.set_ylabel("Latitude [°]")
        ax.legend(fontsize=6, loc="lower right")
        ax.set_title(f"{dirn}: route variants on ENC {c5} (1:12 000), depth areas DEPARE\n"
                     f"min UKC over soundings in swept corridor: T = {a.draught} m, low water −{a.low_water} m, "
                     f"Barrass squat k={a.squat_k:g}", fontsize=9)
        name = "yxlan_route_on_enc_ukc.png" if dirn == "berth_2_to_berth_1" else f"yxlan_route_on_enc_ukc_{dirn}.png"
        fig.savefig(a.out / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
