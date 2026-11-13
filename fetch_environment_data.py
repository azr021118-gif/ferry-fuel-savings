"""Download ERA5 wind and CMEMS Baltic currents, sea level and waves for the YXLAN route.

Credentials are read automatically by cdsapi (~/.cdsapirc) and by the
copernicusmarine toolbox; nothing is passed or stored here.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from project_paths import OUTPUT_DIR

ENV_DIR = OUTPUT_DIR / "environment"
START, END = "2025-07-01", "2025-11-30"
MONTHS = ["07", "08", "09", "10", "11"]

# Route lies within 56.12-56.17 N, 15.55-15.60 E. ERA5 (0.25 deg) needs a wider box
# so that several sea points are available; CMEMS (~1-2 km) only needs a margin.
ERA5_AREA = [56.50, 15.00, 55.75, 16.25]  # N, W, S, E
CMEMS_BOX = dict(minimum_latitude=56.08, maximum_latitude=56.21,
                 minimum_longitude=15.48, maximum_longitude=15.68)

CMEMS_DATASETS = {
    "cmems_bal_phy_15min": ("cmems_mod_bal_phy_anfc_PT15M-i", ["uo", "vo", "sla"]),
    "cmems_bal_wav_1h": ("cmems_mod_bal_wav_anfc_PT1H-i", ["VHM0", "VMDR", "VTM02", "VHM0_WW", "VMDR_WW"]),
}


def era5_months_present() -> set[str]:
    """Folders are era5_2025MM or era5_2025MM-MM-... (one combined request)."""
    done = set()
    for p in ENV_DIR.glob("era5_2025*"):
        if p.is_dir() and any(p.glob("*.nc")):
            done.update(p.name[len("era5_2025"):].split("-"))
    return done


def unpack_era5(download: Path, month: str) -> Path:
    """CDS returns a zip (one NetCDF per stepType) when instant and max fields are mixed."""
    out_dir = ENV_DIR / f"era5_2025{month}"
    out_dir.mkdir(exist_ok=True)
    if zipfile.is_zipfile(download):
        with zipfile.ZipFile(download) as z:
            for info in z.infolist():
                name = Path(info.filename).name  # never trust paths inside the archive
                if not name.endswith(".nc"):
                    continue
                (out_dir / name).write_bytes(z.read(info))
        download.unlink()
    else:
        download.replace(out_dir / "data.nc")
    return out_dir


def fetch_era5(overwrite: bool, months: list[str], combined: bool) -> None:
    """CDS rejects parallel queued jobs per dataset, so requests run sequentially;
    --combined puts all missing months into one job to queue only once."""
    import cdsapi

    for month in months:
        legacy = ENV_DIR / f"era5_wind_2025{month}.nc"
        if legacy.exists():  # earlier run saved the zip under a .nc name
            unpack_era5(legacy.rename(legacy.with_suffix(".zip")), month)
    todo = list(months) if overwrite else [m for m in months if m not in era5_months_present()]
    for m in sorted(set(months) - set(todo)):
        print(f"skip era5 2025-{m}")
    if not todo:
        return
    client = cdsapi.Client()
    for group in ([todo] if combined else [[m] for m in todo]):
        tag = "-".join(group)
        request = {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                         "10m_wind_gust_since_previous_post_processing", "land_sea_mask"],
            "year": ["2025"],
            "month": group,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": ERA5_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        part = ENV_DIR / f"era5_2025{tag}.download"
        client.retrieve("reanalysis-era5-single-levels", request, str(part))
        print(f"saved {unpack_era5(part, tag).name}")


def fetch_cmems(overwrite: bool) -> None:
    import copernicusmarine

    for name, (dataset_id, variables) in CMEMS_DATASETS.items():
        target = ENV_DIR / f"{name}_20250701_20251130.nc"
        if target.exists() and not overwrite:
            print(f"skip {target.name}")
            continue
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            start_datetime=f"{START}T00:00:00",
            end_datetime=f"{END}T23:59:59",
            coordinates_selection_method="outside",
            output_directory=str(ENV_DIR),
            output_filename=target.name,
            overwrite=overwrite,
            disable_progress_bar=True,
            **CMEMS_BOX,
        )
        print(f"saved {target.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["era5", "cmems"], default=None)
    parser.add_argument("--months", nargs="+", choices=MONTHS, default=MONTHS,
                        help="ERA5 months to fetch (default: all)")
    parser.add_argument("--combined", action="store_true",
                        help="request all missing ERA5 months in one CDS job")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "cmems"):
        fetch_cmems(args.overwrite)
    if args.only in (None, "era5"):
        fetch_era5(args.overwrite, args.months, args.combined)


if __name__ == "__main__":
    main()
