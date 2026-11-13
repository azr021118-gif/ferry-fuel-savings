from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_paths import DEFAULT_DATASET_BASE, OUTPUT_DIR as DEFAULT_OUTPUT_DIR


DEFAULT_INTERP = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step4_main_route_interpolated.csv"
DEFAULT_CANDIDATES = DEFAULT_OUTPUT_DIR / f"{DEFAULT_DATASET_BASE}_step4_main_route_candidates.csv"
VALID_DIRECTIONS = ("berth_2_to_berth_1", "berth_1_to_berth_2")
MAX_K = 4
MIN_CLUSTER_SIZE = 8
MIN_SILHOUETTE = 0.12
OFFSET_SCALES = {
    "centerline": 0.0,
    "port_shift_small": -0.35,
    "port_shift_medium": -0.7,
    "starboard_shift_small": 0.35,
    "starboard_shift_medium": 0.7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step-7 main-voyage trajectory clusters, multi-centerlines, and candidate paths."
    )
    parser.add_argument("--interp", type=Path, default=DEFAULT_INTERP)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def haversine_nm(lat1, lon1, lat2, lon2):
    radius_m = 6_371_000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return (radius_m * c) / 1852.0


def build_embeddings(interp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta_rows = []
    for (direction, voyage_id), group in interp.groupby(["direction", "voyage_id"]):
        g = group.sort_values("tau")
        lat = (
            g["latitude"]
            .astype(float)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
        lon = (
            g["longitude"]
            .astype(float)
            .interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
        if np.isnan(lat).any() or np.isnan(lon).any():
            continue
        lat0 = lat[0]
        lon0 = lon[0]
        feat = np.concatenate([lat - lat0, lon - lon0])
        row = {"direction": direction, "voyage_id": int(voyage_id)}
        row.update({f"f_{i}": float(v) for i, v in enumerate(feat)})
        rows.append(row)
        meta_rows.append({"direction": direction, "voyage_id": int(voyage_id), "tau_points": len(g)})
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def choose_cluster_count(X: np.ndarray) -> tuple[np.ndarray, int, float]:
    n = len(X)
    if n < MIN_CLUSTER_SIZE * 2:
        return np.zeros(n, dtype=int), 1, np.nan

    best_labels = np.zeros(n, dtype=int)
    best_k = 1
    best_score = -1.0
    max_k = min(MAX_K, max(2, n // MIN_CLUSTER_SIZE))

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = model.fit_predict(X)
        counts = pd.Series(labels).value_counts()
        if (counts < MIN_CLUSTER_SIZE).any():
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_score < MIN_SILHOUETTE:
        return np.zeros(n, dtype=int), 1, best_score
    return best_labels, best_k, best_score


def cluster_trajectories(embeddings: pd.DataFrame) -> pd.DataFrame:
    cluster_rows = []
    for direction in VALID_DIRECTIONS:
        sub = embeddings.loc[embeddings["direction"] == direction].copy().reset_index(drop=True)
        if sub.empty:
            continue
        feat_cols = [c for c in sub.columns if c.startswith("f_")]
        X_raw = sub[feat_cols].to_numpy(dtype=float)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = StandardScaler().fit_transform(X_raw)
        pca = PCA(n_components=min(8, X_scaled.shape[1], len(sub) - 1), random_state=42)
        X = pca.fit_transform(X_scaled) if len(sub) > 2 else X_scaled
        labels, cluster_count, silhouette = choose_cluster_count(X)
        sub["cluster_id"] = labels.astype(int)
        sub["cluster_name"] = [f"{direction}_cluster_{int(x)+1}" for x in labels]
        sub["cluster_count_direction"] = int(cluster_count)
        sub["silhouette_direction"] = float(silhouette) if silhouette == silhouette else np.nan
        cluster_rows.append(sub)
    if not cluster_rows:
        raise ValueError("No valid embeddings available for main-voyage clustering.")
    return pd.concat(cluster_rows, ignore_index=True)


def build_cluster_centerlines(interp: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    merged = interp.merge(assignments[["direction", "voyage_id", "cluster_id", "cluster_name"]], on=["direction", "voyage_id"], how="left")
    rows = []
    for (direction, cluster_id, cluster_name, tau), group in merged.groupby(["direction", "cluster_id", "cluster_name", "tau"]):
        rows.append(
            {
                "direction": direction,
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_name,
                "tau": float(tau),
                "center_lat": float(group["latitude"].median()),
                "center_lon": float(group["longitude"].median()),
                "spread_lat_std": float(group["latitude"].std(ddof=0)),
                "spread_lon_std": float(group["longitude"].std(ddof=0)),
                "voyage_count": int(group["voyage_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["direction", "cluster_id", "tau"]).reset_index(drop=True)


def build_cluster_summary(assignments: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    merged = assignments.merge(
        candidates[["voyage_id", "direction", "fuel_main_voyage", "main_voyage_minutes", "lateral_dev_mean_nm"]],
        on=["direction", "voyage_id"],
        how="left",
    )
    rows = []
    for (direction, cluster_id, cluster_name), group in merged.groupby(["direction", "cluster_id", "cluster_name"]):
        rows.append(
            {
                "direction": direction,
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_name,
                "voyage_count": int(len(group)),
                "silhouette_direction": float(group["silhouette_direction"].iloc[0]) if "silhouette_direction" in group else np.nan,
                "mean_fuel_main_voyage": float(group["fuel_main_voyage"].mean()),
                "median_fuel_main_voyage": float(group["fuel_main_voyage"].median()),
                "mean_main_voyage_minutes": float(group["main_voyage_minutes"].mean()),
                "mean_lateral_dev_mean_nm": float(group["lateral_dev_mean_nm"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["direction", "cluster_id"]).reset_index(drop=True)


def build_candidate_paths(centerlines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, cluster_id, cluster_name), group in centerlines.groupby(["direction", "cluster_id", "cluster_name"]):
        g = group.sort_values("tau").reset_index(drop=True).copy()
        lat = g["center_lat"].to_numpy(dtype=float)
        lon = g["center_lon"].to_numpy(dtype=float)
        spread_lat = g["spread_lat_std"].fillna(0.0).to_numpy(dtype=float)
        spread_lon = g["spread_lon_std"].fillna(0.0).to_numpy(dtype=float)
        dlat = np.gradient(lat)
        dlon = np.gradient(lon)
        norm = np.sqrt(dlat**2 + dlon**2)
        norm[norm == 0] = 1.0
        nlat = -dlon / norm
        nlon = dlat / norm
        local_scale = np.maximum(np.sqrt(spread_lat**2 + spread_lon**2), 0.00035)

        for variant_name, factor in OFFSET_SCALES.items():
            temp = g.copy()
            temp["variant_name"] = variant_name
            temp["path_lat"] = temp["center_lat"] + factor * local_scale * nlat
            temp["path_lon"] = temp["center_lon"] + factor * local_scale * nlon
            rows.append(temp)
    return pd.concat(rows, ignore_index=True)


def build_metadata(assignments: pd.DataFrame, cluster_summary: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for direction in VALID_DIRECTIONS:
        sub = assignments.loc[assignments["direction"] == direction]
        out[direction] = {
            "voyage_count": int(len(sub)),
            "cluster_count": int(sub["cluster_id"].nunique()),
            "silhouette": float(sub["silhouette_direction"].iloc[0]) if len(sub) else np.nan,
            "cluster_sizes": (
                sub.groupby("cluster_name").size().sort_values(ascending=False).to_dict()
            ),
        }
    return {
        "direction_summary": out,
        "total_cluster_count": int(cluster_summary["cluster_name"].nunique()),
        "candidate_path_family_count": int(cluster_summary["cluster_name"].nunique() * len(OFFSET_SCALES)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interp = pd.read_csv(args.interp)
    candidates = pd.read_csv(args.candidates)

    embeddings, _ = build_embeddings(interp)
    assignments = cluster_trajectories(embeddings)
    centerlines = build_cluster_centerlines(interp, assignments)
    cluster_summary = build_cluster_summary(assignments, candidates)
    candidate_paths = build_candidate_paths(centerlines)
    metadata = build_metadata(assignments, cluster_summary)

    dataset_base = args.interp.stem.replace("_step4_main_route_interpolated", "")
    base = f"{dataset_base}_step7_main_route_clusters"
    assignments_csv = args.output_dir / f"{base}_assignments.csv"
    centerlines_csv = args.output_dir / f"{base}_centerlines.csv"
    cluster_summary_csv = args.output_dir / f"{base}_summary.csv"
    candidate_paths_csv = args.output_dir / f"{base}_candidate_paths.csv"
    metadata_json = args.output_dir / f"{base}_metadata.json"

    assignments.to_csv(assignments_csv, index=False, encoding="utf-8-sig")
    centerlines.to_csv(centerlines_csv, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(cluster_summary_csv, index=False, encoding="utf-8-sig")
    candidate_paths.to_csv(candidate_paths_csv, index=False, encoding="utf-8-sig")
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"assignments_csv={assignments_csv}")
    print(f"centerlines_csv={centerlines_csv}")
    print(f"cluster_summary_csv={cluster_summary_csv}")
    print(f"candidate_paths_csv={candidate_paths_csv}")
    print(f"metadata_json={metadata_json}")


if __name__ == "__main__":
    main()
