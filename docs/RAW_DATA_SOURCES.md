# Historical ERA5 and CERES data sources

This page identifies the providers and products behind the historical data
retained for this study. See [the raw-data archive guide](RAW_DATA.md) for the
download and verification commands and
[`raw-data-manifest.json`](../artifacts/raw-data-manifest.json) for the exact
published files, sizes and SHA-256 checksums. The archive is a copy of the
study's acquired subsets, not a mirror of every variable, level and year in
the providers' complete archives.

Official source links and product labels were checked on **2026-10-08**.

## ERA5: WeatherBench 2 and Copernicus CDS

| Retained source route | Data represented in this study | Official source and citation |
| --- | --- | --- |
| WeatherBench 2 ERA5, 64 longitude by 32 latitude, six-hourly | Main fields: 850 hPa temperature, specific humidity, zonal and meridional wind, plus total cloud cover. The study uses 1979-01-01 through 2023-01-10 from this route. | [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html); [pressure-level dataset DOI](https://doi.org/10.24381/cds.bd0915c6); [single-level dataset DOI](https://doi.org/10.24381/cds.adbb2d47). |
| Original CDS extension at 5.625 degrees | The same main fields for 2023-01-11 through 2025-12-31, requested at 00, 06, 12 and 18 UTC. Monthly provider NetCDFs and combined study exports are distinct file types. | [ERA5 hourly pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview); [ERA5 hourly single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview). |
| Revision CDS main fields at 0.25 degrees | Retained global provider NetCDFs used to replace the original coarse extension with an explicitly conservative remapping route. These also include fixed 2022/early-2023 overlap windows. | The same two CDS datasets and DOIs above; acquisition code: [`download_era5_primary_native.py`](../revision/download_era5_primary_native.py). |
| WeatherBench 2 and CDS physical controls | Vertical velocity at 500/700 hPa, geopotential at 500 hPa, temperature at 700 hPa, surface pressure and mean sea-level pressure. WeatherBench 2 covers 1979-01-01 through 2023-01-10; the 0.25-degree CDS extension covers 2023-01-11 through 2025-12-31. | The same WeatherBench 2 source and CDS datasets; acquisition code: [`download_era5_public_controls.py`](../revision/download_era5_public_controls.py) and [`download_era5_cds_controls.py`](../revision/download_era5_cds_controls.py). |
| Source-route and overlap acquisitions | Fixed overlap dates compare WeatherBench 2, directly requested coarse CDS fields, native-grid CDS fields and the separately labelled server-side grid-box-average trial. The archive manifest determines which retained files belong to each route. | [`download_era5_source_overlap.py`](../revision/download_era5_source_overlap.py) and the CDS datasets above. |

The exact WeatherBench 2 object used by both the original main-field exporter
and revision control-field downloader is:

```text
gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr
```

Its public HTTPS base is
[`storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr/`](https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr/.zmetadata).
The link opens the Zarr metadata; the dataset consists of many objects, not
one downloadable NetCDF. The historical NetCDF exports preserve the study's
selected variables and dates. They are already derived from the provider's
conservatively regridded ERA5 product.

The [WeatherBench 2 guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
documents first-order conservative regridding. Asking CDS for a grid with the
same spacing does not establish that it used the same operator. Consequently,
the original coarse CDS files, native-grid files, server-side remapping trial
and locally remapped revision files retain separate provenance. Here,
"native CDS" means the acquired regular 0.25-degree CDS product; it does not
mean the original ERA5 model grid or all 137 model levels.

For a fresh provider acquisition, use the
[official CDS API setup instructions](https://cds.climate.copernicus.eu/how-to-api)
and accept the terms on both dataset download pages. Provider authentication
is needed for a new CDS request. It is not needed to download the copies in
the public GitHub release. A new provider export may have different binary
packaging, so the archived-file SHA-256 values identify the historical files,
not all future exports with the same scientific selection.

## CERES: the actual Edition 4.2 subset products

The historical CERES requests used the official
[SYN1deg Edition 4.2 subsetting tool](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/SYN1degEd42Selection.jsp),
selecting **Observed Cloud Parameters / Cloud Area Fraction / Total**.
The retained NetCDF headers identify NASA Langley Research Center and
`Edition 4.2: Release Date May 7, 2025`.

| Retained subset | Selection | Correct provider product citation |
| --- | --- | --- |
| Global hourly total cloud | 2017-01-01 through 2025-12-31; 1-degree grid; 180 by 360 cells; variable `cldarea_total_1h`, in percent. | CER_SYN1deg-1Hour_Terra-Aqua-NOAA20, Edition4B; [DOI: 10.5067/TERRA-AQUA-NOAA20/CERES/SYN1DEG-1HOUR_L3.004B](https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/SYN1DEG-1HOUR_L3.004B); [official CMR record](https://cmr.earthdata.nasa.gov/search/concepts/C3181056140-LARC_CLOUD.umm_json). |
| Global daily total cloud | 2001-01-01 through 2025-12-31; the same spatial grid; variable `cldarea_total_daily`, in percent. | CER_SYN1deg-Day_Terra-Aqua-NOAA20, Edition4B; [DOI: 10.5067/Terra-Aqua-NOAA20/CERES/SYN1degDay_L3.004B](https://doi.org/10.5067/Terra-Aqua-NOAA20/CERES/SYN1degDay_L3.004B); [official CMR lookup](https://cmr.earthdata.nasa.gov/search/collections.umm_json?short_name=CER_SYN1deg-Day_Terra-Aqua-NOAA20&version=Edition4B). |
| Earlier regional pilot and annual extension | Two fixed Atlantic boxes: January 2019 pilot and February-December 2019 extension, hourly; each box has 34 by 36 one-degree cells. | The same hourly Edition4B DOI; these are retained earlier acquisitions, not extra independent global observations. |

The historical hourly NetCDF header contains a DOI string with
`SYN1deg1Hour_L3.004B`; the current official CMR record gives
`SYN1DEG-1HOUR_L3.004B`. The release preserves the NetCDF bytes and records
the original header value separately from the canonical catalogue citation.
Use the linked CMR DOI above when citing the hourly product.

The [CERES data-quality summary](https://ceres.larc.nasa.gov/documents/DQ_summaries/CERES_SYN1deg_Ed4A_DQS_V1.pdf)
explicitly identifies **Edition4B as Edition4.2 in the ordering tool**. The
filename of that summary still contains `Ed4A`; its opening update explains
the Edition4B release. Do not substitute the older Edition4A dataset DOI or
the monthly `MHour` product for these hourly/daily subsets.

The product changes from Terra+Aqua to NOAA-20 beginning in April 2022, as
documented by the [official selection page](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/SYN1degEd42Selection.jsp)
and data-quality summary. Preserve this transition when interpreting the
record. Keep the original CF time coordinates, missing-value conventions and
units. Hourly and daily satellite products have different temporal support;
their timestamps must not be silently treated as identical to the four
instantaneous ERA5 samples per day. The acquired CERES files contain the
selected cloud field, not every flux and aerosol variable in SYN1deg.

## Attribution and redistribution

**WeatherBench 2 ERA5.** The public bucket's
[dataset-specific LICENSE](https://storage.googleapis.com/weatherbench2/datasets/era5/LICENSE)
is the Copernicus Products licence. It permits redistribution and adaptation
with visible Copernicus attribution, a statement that adapted data were
modified, and the provider-responsibility notice. Preserve this dataset
licence with the copies; the WeatherBench 2 software licence is not a
replacement for the data licence.

**CDS ERA5.** Both current dataset pages specify CC-BY. ECMWF documents the
[CDS licence transition on 2 July 2025](https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464)
and publishes the [CC-BY-4.0 terms](https://ecds.ecmwf.int/licences/creative-commons-attribution-4-0-international-public-licence).
Retain the creator, dataset DOI, licence link and description of changes.
This archive preserves the historical acquisition routes as well as their
subsequent study-specific transformations.

Suggested acknowledgement for the ERA5 subsets and derived inputs:

> ERA5 data were produced by ECMWF and obtained through the Copernicus Climate
> Change Service Climate Data Store and the WeatherBench 2 public distribution.
> This study contains modified Copernicus Climate Change Service information
> acquired in 2026. The modifications include variable and time selection,
> format conversion and the explicitly documented regridding and aggregation.
> The European Commission and ECMWF are not responsible for the use of these
> data in this study. Cite the pressure-level and single-level dataset DOIs
> above and retain the applicable data licences.

**CERES.** The official hourly CMR record marks the dataset as free and open;
both hourly and daily collection records link to
[NASA EOSDIS Data Use and Citation Guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance).
The guidance identifies unrestricted NASA-led mission data as CC0 unless
separately marked and strongly requests data citations. Credit the NASA
Langley CERES Science Team and Atmospheric Science Data Center, cite the
Edition4B DOI for each temporal product used, and identify the cloud-field,
date and geographic subset. Retain original product metadata and identify
the study's later regridding and aggregation. Repository code licensing does
not replace these source notices.

The archived provenance materials include copies of the WeatherBench 2 ERA5
licence, ECMWF CC-BY-4.0 terms and NASA data-use guidance. Their accompanying
`license-sources.json` records the public source URL, retrieval time, byte
count and SHA-256 checksum. See [the archive guide](RAW_DATA.md) for the
metadata package and file-level manifest.
