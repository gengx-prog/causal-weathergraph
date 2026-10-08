"""Independent fixtures for the ERA5 percent archive and original NetCDF route."""
import numpy as np
import pytest
import xarray as xr

from revision import run_ceres_cloud_substitution as substitution


@pytest.fixture
def portable_cloud():
    timestamps = np.arange("2017-01-01", "2017-01-02", dtype="datetime64[6h]").astype("datetime64[ns]")
    cloud = np.linspace(0.0, 100.0, len(timestamps) * 2048).reshape(len(timestamps), 2048)
    return {"cloud_percent": cloud, "timestamps": timestamps}


def test_portable_cloud_preserves_percent_values_and_needs_no_source_tree(tmp_path, portable_cloud):
    archive = tmp_path / "cloud.npz"
    np.savez(archive, **portable_cloud)
    observed, times = substitution.load_era5_cloud(
        tmp_path / "absent_extension", data_root=tmp_path / "absent_sources", cloud_npz=archive,
    )
    np.testing.assert_array_equal(observed, portable_cloud["cloud_percent"])
    np.testing.assert_array_equal(times, portable_cloud["timestamps"])
    assert observed.dtype == np.float64
    assert observed[-1, -1] == 100.0


@pytest.mark.parametrize("defect, message", [
    ("wrong_grid", "shape"), ("wrong_time_length", "shape"), ("empty", "nonempty"),
    ("float32", "float64"), ("time_strings", "datetime64"), ("time_matrix", "one-dimensional"),
    ("duplicate", "6-hour"), ("reverse", "6-hour"), ("gap", "6-hour"),
    ("offset", "6-hour"), ("nat", "NaT"),
    ("nan", "finite"), ("infinity", "finite"), ("negative", "0..100"), ("above_100", "0..100"),
])
def test_portable_cloud_rejects_invalid_scientific_inputs(tmp_path, portable_cloud, defect, message):
    cloud, times = portable_cloud["cloud_percent"].copy(), portable_cloud["timestamps"].copy()
    if defect == "wrong_grid":
        cloud = cloud[:, :-1]
    elif defect == "wrong_time_length":
        times = times[:-1]
    elif defect == "empty":
        cloud, times = cloud[:0], times[:0]
    elif defect == "float32":
        cloud = cloud.astype(np.float32)
    elif defect == "time_strings":
        times = times.astype(str)
    elif defect == "time_matrix":
        times = times[:, None]
    elif defect == "duplicate":
        times[1] = times[0]
    elif defect == "reverse":
        times = times[::-1]
    elif defect == "gap":
        times[2:] += np.timedelta64(6, "h")
    elif defect == "offset":
        times += np.timedelta64(1, "h")
    elif defect == "nat":
        times[1] = np.datetime64("NaT")
    elif defect == "nan":
        cloud[0, 0] = np.nan
    elif defect == "infinity":
        cloud[0, 0] = np.inf
    elif defect == "negative":
        cloud[0, 0] = -0.01
    elif defect == "above_100":
        cloud[0, 0] = 100.01
    archive = tmp_path / "bad.npz"
    np.savez(archive, cloud_percent=cloud, timestamps=times)
    with pytest.raises(ValueError, match=message):
        substitution.load_era5_cloud(cloud_npz=archive)


def test_portable_cloud_reports_missing_array(tmp_path, portable_cloud):
    archive = tmp_path / "missing.npz"
    np.savez(archive, timestamps=portable_cloud["timestamps"])
    with pytest.raises(ValueError, match="requires cloud_percent and timestamps"):
        substitution.load_era5_cloud(cloud_npz=archive)


def test_portable_cloud_keeps_tolerated_boundary_roundoff(tmp_path, portable_cloud):
    portable_cloud["cloud_percent"][0, 0] = -1e-5
    portable_cloud["cloud_percent"][-1, -1] = 100.00001
    archive = tmp_path / "roundoff.npz"
    np.savez(archive, **portable_cloud)
    observed, _ = substitution.load_era5_cloud(cloud_npz=archive)
    np.testing.assert_array_equal(observed, portable_cloud["cloud_percent"])


def test_netcdf_custom_roots_and_portable_archive_have_identical_order_and_units(tmp_path, monkeypatch):
    times = np.arange("2017-01-01", "2017-01-02T12", dtype="datetime64[6h]").astype("datetime64[ns]")
    segments = [(f"segment_{i}", str(times[2 * i]), str(times[2 * i + 1])) for i in range(3)]
    monkeypatch.setattr(substitution, "SEGMENTS", segments)
    data_root, extension = tmp_path / "sources", tmp_path / "extension"
    latitude = np.linspace(87.1875, -87.1875, 32)
    longitude = np.arange(64) * 5.625
    fraction = np.arange(6 * 32 * 64, dtype=np.float64).reshape(6, 32, 64) / (6 * 32 * 64)
    for i, (name, _, _) in enumerate(segments):
        folder = extension if i == 2 else data_root / name
        folder.mkdir(parents=True)
        xr.Dataset(
            {"tcc": (("time", "latitude", "longitude"), fraction[2 * i:2 * i + 2])},
            coords={"time": times[2 * i:2 * i + 2], "latitude": latitude, "longitude": longitude},
        ).to_netcdf(folder / "total_cloud_cover.nc")
    # The fixture is stored north-to-south; independently specify the expected
    # southernmost row first and convert fractions to percentage points once.
    expected = fraction[:, ::-1, :].reshape(6, 2048) * 100.0
    original_cloud, original_times = substitution.load_era5_cloud(extension, data_root=data_root)
    np.testing.assert_array_equal(original_cloud, expected)
    np.testing.assert_array_equal(original_times, times)
    archive = tmp_path / "same.npz"
    np.savez(archive, cloud_percent=expected, timestamps=times)
    portable, portable_times = substitution.load_era5_cloud(cloud_npz=archive)
    np.testing.assert_array_equal(portable, original_cloud)
    np.testing.assert_array_equal(portable_times, original_times)
    assert substitution.era5_cloud_input_paths(extension, data_root=data_root)["segment_2"] == extension / "total_cloud_cover.nc"
