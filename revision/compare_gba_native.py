"""Cross-check the two conservative routes after the splice, where WB2 is unavailable.

For every extension month retrieved both natively (and remapped here with the WB2
conservative method, regrid_era5_primary_native.py) and as CDS grid-box averages
(download_era5_primary_native.py --route gba), the two 64x32 fields are compared
value by value. Agreement at the level seen against WB2 on the overlap windows
shows that the server-side route is equivalent after 2023-01-10 as well.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SUPP = Path(r"D:\Paper2\Major Revision\supplementary_data")
NATIVE = SUPP / "era5_primary_native" / "regridded"
GBA = SUPP / "era5_primary_gba" / "extension"
OUT = Path(r"D:\Paper2\Major Revision\revision_outputs\source_overlap_gba_route")
NAMES = {"t": "temperature_850", "q": "specific_humidity_850", "u": "u_component_of_wind_850",
         "v": "v_component_of_wind_850", "tcc": "total_cloud_cover"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for native_file in sorted(NATIVE.glob("*_20[0-9][0-9][0-9][0-9].nc")):
        gba_file = GBA / native_file.name
        if not gba_file.exists():
            continue
        with xr.open_dataset(native_file) as a, xr.open_dataset(gba_file) as b:
            b = b.sortby("latitude").sortby("longitude")
            for raw, name in NAMES.items():
                if name not in a or raw not in b:
                    continue
                gb = b[raw]
                if "pressure_level" in gb.dims:
                    gb = gb.sel(pressure_level=850.0)
                gb = gb.transpose("valid_time", "latitude", "longitude")
                if not np.array_equal(a.time.values, gb.valid_time.values):
                    raise ValueError(f"time mismatch in {native_file.name}")
                x, y = a[name].values.astype(np.float64), gb.values.astype(np.float64)
                d = y - x
                rows.append({"file": native_file.stem, "variable": name, "n_times": x.shape[0],
                             "mean_difference_gba_minus_native": float(d.mean()), "rmse": float(np.sqrt(np.mean(d ** 2))),
                             "max_abs_difference": float(np.abs(d).max()), "native_sd": float(x.std()),
                             "rmse_over_sd": float(np.sqrt(np.mean(d ** 2)) / x.std()),
                             "pearson_r": float(np.corrcoef(x.ravel(), y.ravel())[0, 1])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "gba_vs_native_after_splice.csv", index=False)
    if table.empty:
        print("no months available in both routes yet")
        return
    summary = table.groupby("variable").agg(months=("file", "nunique"), times=("n_times", "sum"),
                                            max_rmse_over_sd=("rmse_over_sd", "max"), min_r=("pearson_r", "min"),
                                            max_abs_difference=("max_abs_difference", "max"))
    summary.to_csv(OUT / "gba_vs_native_after_splice_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
