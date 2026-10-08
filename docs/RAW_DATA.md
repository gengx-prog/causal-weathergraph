# Download the historical ERA5 and CERES inputs

The [raw-data release](https://github.com/gengx-prog/causal-weathergraph/releases/tag/raw-data-20261008)
archives the ERA5 and CERES files acquired for the original study and its
revision. It contains **1,100 scientific NetCDF files totalling
108,693,362,733 bytes (108.69 GB)**, plus eight source and acquisition metadata
files. The files are distributed in **79 ZIP assets**, each smaller than
1.8 GB. Their exact paths, sizes, download URLs and SHA-256 checksums are in
[`artifacts/raw-data-manifest.json`](../artifacts/raw-data-manifest.json).

These are the historical subsets obtained for this study. "Raw data" includes
provider-generated NetCDF subsets and the historical local exports of
WeatherBench 2 data. It does not mean the complete global ERA5 or CERES archive
across every variable, level, resolution and year. The
[source guide](RAW_DATA_SOURCES.md) identifies the official product links,
DOIs, versions, coverage and applicable data licences.

## Choose the required dataset

Run commands from the repository root using the Python environment described
in [ENVIRONMENT.md](ENVIRONMENT.md). The download and checksum scripts use
only the Python standard library. List the available groups and sizes without
downloading any data:

```bash
python scripts/download_artifacts.py --manifest artifacts/raw-data-manifest.json --list
```

Start with the small metadata package:

```bash
python scripts/download_artifacts.py --manifest artifacts/raw-data-manifest.json --dataset source-metadata --output raw-data --cache raw-data-downloads
python scripts/verify_artifacts.py --manifest artifacts/raw-data-manifest.json --dataset source-metadata --data-root raw-data
```

Choose a dataset ID from the table, then use the same ID for both download
and verification. For example, the original WeatherBench 2 primary inputs:

```bash
python scripts/download_artifacts.py --manifest artifacts/raw-data-manifest.json --dataset era5-wb2-primary --output raw-data --cache raw-data-downloads
python scripts/verify_artifacts.py --manifest artifacts/raw-data-manifest.json --dataset era5-wb2-primary --data-root raw-data
```

| Dataset ID | Scientific files | File bytes, decimal GB | Contents |
| --- | ---: | ---: | --- |
| `era5-wb2-primary` | 10 | 2.044 | Historical WeatherBench 2 exports: temperature, specific humidity and u/v wind at 850 hPa, plus total cloud cover; 1979-01-01 through 2023-01-10; six-hourly, 64 by 32 grid. |
| `era5-cds-legacy` | 77 | 0.215 | Original coarse CDS extension, 2023-01-11 through 2025-12-31: 72 monthly provider files and five combined input exports. |
| `era5-native-primary` | 82 | 38.864 | Revision primary-field acquisition on the regular 0.25-degree CDS grid; the 2023-01-11 through 2025-12-31 extension and fixed earlier overlap windows. |
| `era5-gba-primary` | 82 | 0.112 | Separately retained CDS grid-box-average request route at 5.625 degrees, for the same extension and overlap windows. |
| `era5-controls-cds` | 148 | 40.923 | Six physical-control fields at 0.25 degrees: 500/700 hPa vertical velocity, 500 hPa geopotential, 700 hPa temperature, surface pressure and mean sea-level pressure; 144 extension files and four pilot files. |
| `era5-controls-wb2` | 644 | 2.197 | Six control fields exported from WeatherBench 2 as sequential chunks; 1979-01-01 through 2023-01-10; six-hourly, 64 by 32 grid. Also includes two source-coordinate/Zarr metadata files. |
| `era5-source-overlap` | 30 | 1.437 | Fixed 2022/early-2023 source-route comparison acquisitions: coarse primary fields and native CDS controls. |
| `ceres-global` | 23 | 22.815 | Global 1-degree total cloud area fraction: hourly 2017-2025 and daily 2001-2025; original Edition 4.2 provider subsets. |
| `ceres-201901-pilot` | 2 | 0.007 | January 2019 hourly total cloud for two fixed Atlantic boxes. |
| `ceres-2019-extension` | 2 | 0.079 | February-December 2019 hourly total cloud for the same two Atlantic boxes. |
| `source-metadata` | 0 | Small | Six metadata files: public provider licences, source records, acquisition information and audit material. |

The ERA5 overlap windows are 1-7 January, April, July and October 2022 and
1-10 January 2023. The CERES regional acquisitions overlap part of the global
2019 hourly record; they preserve the earlier acquisition history and do not
provide additional independent observations. The source manifest records
each file's dataset and origin.

## Download the complete archive when needed

The full download is approximately **110 GB**, with approximately another
**110 GB** required for extracted files. Allow **more than 220 GB of free
space**, including filesystem overhead and working space. ZIP files remain
in the cache after extraction. You can put `--cache` and `--output` on
different disks by supplying absolute paths. Selecting one dataset avoids
downloading the entire archive.

```bash
python scripts/download_artifacts.py --manifest artifacts/raw-data-manifest.json --asset all --output raw-data --cache raw-data-downloads
python scripts/verify_artifacts.py --manifest artifacts/raw-data-manifest.json --data-root raw-data
```

All release assets are public; no CDS or NASA account is needed for these
archived copies. The downloader uses the versioned manifest to check the
ZIP assets and extracted files. Keep the manifest together with any offline
copy. You can also obtain the ZIP assets from the
[release page](https://github.com/gengx-prog/causal-weathergraph/releases/tag/raw-data-20261008)
and provide their directory to the downloader:

```bash
python scripts/download_artifacts.py --manifest artifacts/raw-data-manifest.json --dataset era5-wb2-primary --from-directory raw-data-downloads --output raw-data --cache raw-data-downloads
```

For this offline example, the directory must already contain all assets for
`era5-wb2-primary`. The manifest identifies those assets. The same approach
works for other dataset selections or the complete archive.

If a transfer is interrupted, rerun the same command. The downloader retains
the partial ZIP in the cache and resumes it when the server supports byte
ranges. It checks the completed ZIP against the manifest before extraction.
If a completed partial ZIP fails its checksum, follow the error's instruction
to remove that temporary file before retrying the transfer.

## Extracted layout and validation scope

The `--output raw-data` commands preserve the relative paths recorded in the
manifest. Historical main inputs appear under `raw-data/legacy_data/`;
revision acquisitions appear under `raw-data/supplementary_data/`; source
notices and audit material appear under `raw-data/provenance/`. Use those
paths when configuring a preprocessing run. No author-specific drive letter
is required to download or verify the published files.

Publication checks cover the selected file inventory, readable NetCDF
headers, recorded acquisition metadata and file checksums. The download and
verification commands establish that the local files match the archived
bytes. They do not, by themselves, rerun all raw-data preprocessing or verify
every scientific result. A complete new run from all native meteorological
files has not been reported as part of this archive publication.

For the tested paper-result reproduction workflow, use
[REPRODUCING.md](REPRODUCING.md) and the smaller
[processed-input release](https://github.com/gengx-prog/causal-weathergraph/releases/tag/reproducibility-20261008.1).
That workflow recomputes the core, CERES-comparison and WHEC analyses from
their archived analysis inputs. The raw-data archive adds the upstream
historical acquisitions for inspection and further preprocessing work.

The publication retains one copy of identical duplicate files and excludes
account credentials, private order/download session data and unrelated
experiment collections. Scientifically distinct acquisition routes remain
separate. Preserve the provider licence and attribution files when copying
or redistributing the data; use the official links and product-specific
citations in [RAW_DATA_SOURCES.md](RAW_DATA_SOURCES.md).
