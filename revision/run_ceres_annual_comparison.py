"""Frozen descriptive CERES/ERA5 comparison for the calendar year 2019.

No network access. Freeze before loading extension cloud values, then run once
after both regional February--December NetCDF deliveries are complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import time
from urllib.parse import unquote, urlparse

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

import run_ceres_pilot_comparison as pilot

BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "revision_outputs/ceres_annual_comparison_2019"
EXTENSION = BASE / "supplementary_data/cloud_validation/ceres_syn1deg_ed42_2019_extension"
REGIONS, SCHEMES = pilot.REGIONS, pilot.SCHEMES
AUTHORIZED_ORDERS = {
    "north_atlantic": {"order_id": "41742", "suborder_id": "99005"},
    "south_atlantic": {"order_id": "41743", "suborder_id": "99006"},
}
MAX_NEW_BYTES = 250_000_000
EXECUTION_ADDENDUM = "execution_addendum_frozen.json"
SEASONS = {1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON",12:"DJF"}
LOCAL_SEASONS = {"north_atlantic":{"DJF":"winter","MAM":"spring","JJA":"summer","SON":"autumn"},
                 "south_atlantic":{"DJF":"summer","MAM":"autumn","JJA":"winter","SON":"spring"}}
LIMITATIONS = [
    "One calendar year and one fixed Atlantic mirror pair only: not global, multiyear, a formal north/south effect test, causal validation, or long-term independent validation.",
    "January results were already examined. This annual protocol freezes processing before this analyst reads the newly acquired February--December cloud values; it is not research preregistration or an unseen-year holdout.",
    "Daily quantities are means of product samples under different sampling operators, not verified equivalent physical 24-hour averages.",
    "The documented Ed4A hourbox/midpoint convention is transferred to Ed4.2 as an explicit version-qualified assumption; exact Ed4.2 within-hour cloud sampling weights remain unverified.",
    "Hourly cloud fields may include interpolation. No sample-specific cloud observation/interpolation flags establish independent direct observations.",
    "Spatial bounds are inferred from regular centers. Both products use the same complete original target cells with spherical area weights and no land/ocean mask; the footprints are not asserted to be ocean-only.",
    "Cloud definitions, retrieval physics and time support differ. Bias, MAE and RMSE are cross-product discrepancies, not errors against ground truth.",
    "Raw physical percentages, not the manuscript's grid-standardized regional anomalies, are compared; these results do not validate causal coefficients or graph edges.",
    "Seasonal groups are calendar-month groups within 2019; DJF joins January--February and December 2019 and is not one continuous winter. Local-season labels are descriptive, with no formal hemisphere-effect test.",
    "Monthly demeaning uses each product's own available 2019 daily values in that month. It removes within-year monthly offsets descriptively; it is not a multiannual climatology, training-only transform, out-of-sample forecast, or bias correction.",
    "All predeclared schemes and groups are reported without outcome-based selection. Serial dependence, shared meteorological structure, interpolation and one-year coverage prevent independent-sample inference; no p-values or calibrated intervals are reported.",
]


def freeze(output, extension):
    output.mkdir(parents=True,exist_ok=True)
    path=output/"comparison_plan.json"
    if path.exists():raise FileExistsError("Frozen annual plan exists; never overwrite it silently")
    oldpath=BASE/"revision_outputs/ceres_pilot_comparison/comparison_plan.json"
    old=json.loads(oldpath.read_text(encoding="utf8"))
    evidence=[oldpath,pilot.PILOT/"metadata/time_semantics_evidence.json",
              pilot.PILOT/"validation_independent_v2.json",pilot.PILOT/"request_plan.json",
              pilot.MANIFEST,Path(pilot.__file__),Path(__file__).with_name("prepare_inputs.py")]
    # Only the already analysed January plan/metadata are inspected here.
    plan={"frozen_utc":pilot.utc_now(),"status":"frozen_before_this_analyst_reads_February_December_CERES_cloud_values",
        "purpose":"One-year descriptive seasonal extension of the existing preprocessing comparison",
        "period":["2019-01-01","2019-12-31"],"expected_daily_rows":730,"expected_hours_per_region":8760,
        "regions":old["regions"],"fixed_existing_inputs":old["inputs"],
        "evidence":[{"path":str(p),"sha256":pilot.sha256(p)} for p in evidence],
        "extension_root":str(extension.resolve()),
        "reader_contract":{"files":"Recursive *.nc beneath north_atlantic and south_atlantic; Jan from old pilot, extension February--December only",
            "variable":"cldarea_total_1h","units":"percent","dimensions":["time","lat","lon"],
            "grid":"Exactly the January source centers/order in each region; no silent remapping to another footprint",
            "time":"Decode original numeric coordinates; containing-hour nominal midpoint must differ by <=15 seconds; exactly one value per expected slot; missing/duplicate time slots cause a stop"},
        "spatial_operator":old["spatial_operator"],"spatial_support_assumption":old["spatial_support_assumption"],
        "schemes":old["schemes"],"missing_policy":old["missing_policy"],
        "groups":{"annual":"All 365 calendar dates", "monthly":"Twelve fixed UTC calendar months",
            "seasonal":"DJF=Jan,Feb,Dec; MAM=Mar,Apr,May; JJA=Jun,Jul,Aug; SON=Sep,Oct,Nov, all within 2019",
            "local_season_labels":LOCAL_SEASONS},
        "metrics":{"unit":"percentage points for differences; percent for means","bias_sign":"CERES minus ERA5",
            "list":["expected_days","valid_product_days","n_paired_days","CERES_mean","ERA5_mean","bias","MAE","RMSE","Pearson_r"],
            "expected_rows":68,"inference":"Descriptive only; no p-values/CI; retain both schemes and every group"},
        "monthly_demeaning":{"products":["CERES_24_slots","CERES_4_halfhour_slots","ERA5_4_times"],
            "fit":"Each region/product independently subtracts its own 2019 calendar-month mean over its valid complete daily sample means",
            "missing":"Preserve invalid days, record separate product reference counts, then use paired finite anomalies for correlation",
            "report":"Four annual region-by-scheme anomaly correlations; all daily anomalies and 24 region-month reference rows saved",
            "interpretation":"Within-year descriptive daily covariation only; neither climatological anomaly validation nor held-out prediction"},
        "limitations":LIMITATIONS}
    pilot.write_json(path,plan)
    print(json.dumps({"plan":str(path),"frozen_utc":plan["frozen_utc"],"sha256":pilot.sha256(path)},indent=2))


def _check_plan(output,extension,require_execution_addendum=True):
    plan=json.loads((output/"comparison_plan.json").read_text(encoding="utf8"))
    if str(extension.resolve())!=plan["extension_root"]:raise ValueError("Extension root differs from the frozen plan")
    if (output/"run_manifest.json").exists():raise FileExistsError("Completed annual output exists; never overwrite it")
    for item in plan["fixed_existing_inputs"]+plan["evidence"]:
        if pilot.sha256(item["path"])!=item["sha256"]:
            raise ValueError(f"Frozen input/evidence changed: {item['path']}")
    if require_execution_addendum:
        addendum=json.loads((output/EXECUTION_ADDENDUM).read_text(encoding="utf8"))
        if addendum["authorized_orders"]!=AUTHORIZED_ORDERS:
            raise ValueError("Authorized orders differ from the frozen execution addendum")
        if addendum["max_new_bytes"]!=MAX_NEW_BYTES:
            raise ValueError("Acquisition cap differs from the frozen execution addendum")
        required={Path(__file__).resolve(),(output/"comparison_plan.json").resolve(),
                  (BASE/"revision_outputs/ceres_pilot_comparison/paired_daily.csv").resolve(),
                  (extension/"request_plan.json").resolve()}
        covered={Path(item["path"]).resolve() for item in addendum["evidence"]}
        if not required.issubset(covered):
            raise ValueError("Execution addendum omits required code, January result, or acquisition evidence")
        for item in addendum["evidence"]:
            if pilot.sha256(item["path"])!=item["sha256"]:
                raise ValueError(f"Frozen execution evidence changed: {item['path']}")
    return plan


def freeze_execution_addendum(output,extension):
    """Append an explicit execution freeze without rewriting the original plan."""
    path=output/EXECUTION_ADDENDUM
    if path.exists():
        raise FileExistsError("Frozen execution addendum exists; never overwrite it")
    plan=_check_plan(output,extension,require_execution_addendum=False)
    new_files=list(extension.rglob("*.nc"))+list(extension.rglob("*.part"))
    if new_files:
        raise ValueError("Execution addendum must be frozen before extension science files arrive")
    paths=[Path(__file__),output/"comparison_plan.json",
           BASE/"revision_outputs/ceres_pilot_comparison/paired_daily.csv",extension/"request_plan.json"]
    paths.extend(Path(item["path"]) for item in plan["fixed_existing_inputs"]+plan["evidence"])
    unique=list(dict.fromkeys(p.resolve() for p in paths))
    addendum={
        "frozen_utc":pilot.utc_now(),
        "status":"execution_addendum_before_new_extension_science_files_arrive_or_are_read",
        "purpose":"Bind the final annual code, January regression reference, original plan and acquisition scope before executing the annual comparison",
        "original_plan_preserved":True,
        "not_original_preregistration":True,
        "scope":"Execution provenance and authorized-input validation; the original annual spatial, temporal, missing-data and descriptive-metric rules remain unchanged",
        "authorized_orders":AUTHORIZED_ORDERS,
        "max_new_bytes":MAX_NEW_BYTES,
        "extension_root":str(extension.resolve()),
        "new_extension_science_files_present":0,
        "acquisition_contract":"Require one complete download_manifest with exactly the two authorized region/order/suborder files; verify URL scope, paths, sizes, SHA256 and absence of extra or partial science files before analysis",
        "evidence":[{"path":str(p),"sha256":pilot.sha256(p)} for p in unique],
        "limitations":"This execution addendum follows the already examined January pilot and is not research preregistration, an unseen-year holdout, or independent scientific validation",
    }
    pilot.write_json(path,addendum)
    print(json.dumps({"execution_addendum":str(path),"frozen_utc":addendum["frozen_utc"],
                      "sha256":pilot.sha256(path)},indent=2))


def _authorized_provider_url(url,region,filename):
    if not isinstance(url,str):
        return False
    try:
        parsed=urlparse(url)
        decoded=unquote(parsed.path)
        parts=PurePosixPath(decoded).parts
        return (parsed.scheme=="https" and parsed.hostname=="ceres-tool.larc.nasa.gov"
                and parsed.port in (None,443) and parsed.username is None and parsed.password is None
                and not parsed.query and not parsed.fragment and "\\" not in decoded
                and not any(part in {".",".."} for part in decoded.split("/"))
                and f'CERES_2026-10-01:{AUTHORIZED_ORDERS[region]["order_id"]}' in parts
                and parts[-1]==filename and filename.lower().endswith(".nc"))
    except (ValueError,IndexError,KeyError):
        return False


def verify_acquisition(extension):
    """Bind annual inputs to completed, authorized downloads before cloud reads."""
    manifest_path=extension/"download_manifest.json"
    manifest_hash=pilot.sha256(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("complete") is not True:
        raise ValueError("Annual acquisition manifest is not complete")
    entries=manifest.get("files")
    if not isinstance(entries,list) or len(entries)!=len(AUTHORIZED_ORDERS):
        raise ValueError("Expected exactly two authorized acquisition entries")
    if list(extension.rglob("*.part")):
        raise ValueError("Partial science files remain in the annual acquisition directory")
    files={}; total=0
    for entry in entries:
        region=entry.get("region")
        if region not in AUTHORIZED_ORDERS or region in files:
            raise ValueError("Unexpected or duplicate acquisition region")
        if any(entry.get(key)!=value for key,value in AUTHORIZED_ORDERS[region].items()):
            raise ValueError("Acquisition order or suborder does not match the authorization")
        path=Path(entry["path"]).resolve()
        if not path.is_relative_to((extension/region).resolve()) or path.suffix.lower()!=".nc":
            raise ValueError("Acquisition file is outside its authorized regional directory")
        if not path.is_file() or not _authorized_provider_url(entry.get("url"),region,path.name):
            raise ValueError("Acquisition file or provider URL is not valid")
        size=entry.get("bytes")
        if type(size) is not int or size<=0 or size!=path.stat().st_size:
            raise ValueError("Acquisition file size does not match its receipt")
        digest=entry.get("sha256")
        if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
            raise ValueError("Invalid acquisition SHA256 receipt")
        if pilot.sha256(path)!=digest:
            raise ValueError("Acquisition file SHA256 differs from its receipt")
        files[region]={**entry,"path":str(path)}
        total+=size
    if total>MAX_NEW_BYTES or type(manifest.get("total_bytes")) is not int or manifest["total_bytes"]!=total:
        raise ValueError("Acquisition total differs from receipts or exceeds the frozen cap")
    actual={path.resolve() for path in extension.rglob("*.nc")}
    expected={Path(entry["path"]) for entry in files.values()}
    if actual!=expected:
        raise ValueError("Unreceipted or missing science files in the acquisition directory")
    if pilot.sha256(manifest_path)!=manifest_hash:
        raise ValueError("Acquisition manifest changed during verification")
    receipt={"path":str(manifest_path.resolve()),"sha256":manifest_hash,"complete":True,
             "total_bytes":total,"authorized_orders":AUTHORIZED_ORDERS,
             "scope":"Every new analysis input matched a completed authorized acquisition receipt and its SHA256"}
    return files,receipt


def _time_metadata(ds):
    t=ds["time"]
    raw=np.asarray(t[:]).copy()
    if raw.ndim!=1 or not np.isfinite(raw).all():raise ValueError("Invalid original CERES time coordinates")
    calendar=getattr(t,"calendar","standard")
    decoded=pd.DatetimeIndex([str(x) for x in netCDF4.num2date(raw.astype(float),t.units,calendar=calendar)])
    nominal=decoded.floor("h")+pd.Timedelta(minutes=30)
    error=(decoded-nominal).total_seconds().to_numpy()
    if np.any(np.abs(error)>15) or nominal.has_duplicates or not nominal.is_monotonic_increasing:
        raise ValueError("CERES nominal slots are duplicate, unsorted, or beyond the frozen 15-second tolerance")
    return raw,decoded,nominal,error,str(t.units),str(calendar)


def inspect_inputs(plan,extension,output):
    """Hash files and inspect only metadata/coordinates before cloud processing."""
    acquired,acquisition_receipt=verify_acquisition(extension)
    fixed={item["role"]:item for item in plan["fixed_existing_inputs"]}
    annual=pd.date_range("2019-01-01 00:30","2019-12-31 23:30",freq="h")
    source_files={}; receipt=[]
    for region,_ in REGIONS:
        new=[Path(acquired[region]["path"])]
        paths=[Path(fixed[region]["path"])]+new
        metadata=[]; reference_lat=reference_lon=None
        for path in paths:
            with netCDF4.Dataset(path) as ds:
                ds.set_auto_maskandscale(False)
                slat,slon=np.asarray(ds["lat"][:]),np.asarray(ds["lon"][:])
                if reference_lat is None:reference_lat,reference_lon=slat,slon
                elif not (np.array_equal(slat,reference_lat) and np.array_equal(slon,reference_lon)):
                    raise ValueError(f"Source grid differs from January: {path}")
                v=ds["cldarea_total_1h"]
                if tuple(v.dimensions)!=("time","lat","lon") or str(v.units).lower()!="percent":
                    raise ValueError(f"Unexpected cloud dimensions/units: {path}")
                if float(getattr(v,"scale_factor",1))!=1 or float(getattr(v,"add_offset",0))!=0:
                    raise ValueError("Packed cloud data require an explicitly reviewed reader; refusing silent conversion")
                version=str(getattr(ds,"Version",""))
                if "4.2" not in version:raise ValueError(f"Expected Edition 4.2 product: {path}")
                raw,decoded,nominal,error,units,calendar=_time_metadata(ds)
                if path!=paths[0] and ((nominal<pd.Timestamp("2019-02-01")).any() or (nominal>=pd.Timestamp("2020-01-01")).any()):
                    raise ValueError("Extension file contains dates outside frozen February--December scope")
                if v.shape!=(len(raw),len(slat),len(slon)):raise ValueError("Cloud shape mismatch")
                fills=[float(v.getncattr(key)) for key in ("_FillValue","missing_value") if key in v.ncattrs()]
                item={"region":region,"path":str(path.resolve()),"sha256":pilot.sha256(path),"bytes":path.stat().st_size,
                    "role":"existing_January" if path==paths[0] else "new_February_December",
                    "version":version,"variable":"cldarea_total_1h","units":"percent","dimensions":list(v.dimensions),
                    "shape":list(v.shape),"fill_values":fills,"time_units":units,"time_calendar":calendar,
                    "original_time_dtype":str(raw.dtype),"nominal_first":nominal[0].isoformat(),"nominal_last":nominal[-1].isoformat(),
                    "time_bounds_variable":str(getattr(ds["time"],"bounds","absent")),"cloud_cell_methods":str(getattr(v,"cell_methods","absent")),
                    "max_abs_nominal_offset_seconds":float(np.max(abs(error)))}
            metadata.append((item,nominal))
        metadata.sort(key=lambda x:x[1][0])
        combined=pd.DatetimeIndex(np.concatenate([n.values for _,n in metadata]))
        if not combined.equals(annual):raise ValueError(f"Expected exactly all 8760 unique hourly slots for {region}")
        source_files[region]=[item for item,_ in metadata]
        receipt.extend(source_files[region])
    pilot.write_json(output/"input_manifest.json",{"created_utc":pilot.utc_now(),"plan_sha256":pilot.sha256(output/"comparison_plan.json"),
        "execution_addendum_sha256":pilot.sha256(output/EXECUTION_ADDENDUM),"acquisition_manifest":acquisition_receipt,
        "stage":"metadata_and_hashes_verified_before_loading_extension_cloud_values","files":receipt,
        "note":"Each new file is bound to the complete authorized download manifest before metadata and scientific processing."})
    return fixed,source_files


def describe(daily):
    metric_rows=[]; anomaly_rows=[]; reference_rows=[]
    product_columns=["ceres_24_slot_mean_percent","ceres_4_halfhour_slot_mean_percent","era5_4_sample_mean_percent"]
    daily=daily.copy()
    for region,region_id in REGIONS:
        region_mask=daily.region==region
        d=daily.loc[region_mask].copy()
        for month in range(1,13):
            selected=region_mask&(daily.month==month)
            row={"region":region,"region_id":region_id,"month":month,"reference_year":2019}
            for product in product_columns:
                x=daily.loc[selected,product].to_numpy(float)
                valid=np.isfinite(x)
                mean=float(x[valid].mean()) if valid.any() else np.nan
                daily.loc[selected,product+"_monthly_demeaned"]=x-mean
                row[product+"_mean"]=mean
                row[product+"_valid_days"]=int(valid.sum())
            reference_rows.append(row)
        groups=[("annual","2019",np.ones(len(d),bool),"")]
        groups.extend(("month",f"2019-{month:02d}",d.month.to_numpy()==month,"") for month in range(1,13))
        groups.extend(("calendar_season",season,d.calendar_season.to_numpy()==season,LOCAL_SEASONS[region][season]) for season in ("DJF","MAM","JJA","SON"))
        for scheme,product in zip(SCHEMES,product_columns[:2]):
            for group_type,group,mask,local_season in groups:
                x=d.loc[mask,product].to_numpy(float); y=d.loc[mask,product_columns[2]].to_numpy(float)
                metric_rows.append({"region":region,"region_id":region_id,"scheme":scheme,"group_type":group_type,"group":group,
                    "local_season":local_season,"expected_days":int(mask.sum()),"valid_ceres_days":int(np.isfinite(x).sum()),
                    "valid_era5_days":int(np.isfinite(y).sum()),**pilot.metric(x,y)})
            x=daily.loc[region_mask,product+"_monthly_demeaned"].to_numpy(float)
            y=daily.loc[region_mask,product_columns[2]+"_monthly_demeaned"].to_numpy(float)
            m=pilot.metric(x,y)
            anomaly_rows.append({"region":region,"region_id":region_id,"scheme":scheme,"year":2019,
                "n_paired_days":m["n_paired_days"],"pearson_r_after_own_2019_monthly_demeaning":m["pearson_r"],
                "interpretation":"descriptive within-year daily variation; own-product monthly references; no held-out fit"})
    if len(metric_rows)!=68 or len(reference_rows)!=24 or len(anomaly_rows)!=4:raise AssertionError("Unexpected fixed group count")
    return daily,metric_rows,reference_rows,anomaly_rows


def run(output,extension):
    started=time.perf_counter(); plan=_check_plan(output,extension)
    execution_addendum_hash=pilot.sha256(output/EXECUTION_ADDENDUM)
    fixed,files=inspect_inputs(plan,extension,output)
    acquisition_receipt=json.loads((output/"input_manifest.json").read_text(encoding="utf8"))["acquisition_manifest"]
    nominal=pd.date_range("2019-01-01 00:30","2019-12-31 23:30",freq="h")
    expected_era=pd.date_range("2019-01-01","2019-12-31 18:00",freq="6h")
    dates=pd.date_range("2019-01-01",periods=365,freq="D")
    with xr.open_dataset(fixed["era5"]["path"]) as ds:
        lat,lon=ds.latitude.values.copy(),ds.longitude.values.copy()
        da=ds[fixed["era5"]["variable"]].sel(time=slice("2019-01-01","2019-12-31T23:59:59"))
        if not np.array_equal(da.time.values,expected_era.values):raise ValueError("ERA5 does not contain the expected 1460 synoptic times")
        era_raw=np.asarray(da.transpose("time","latitude","longitude").values,float)
    valid_era=era_raw[np.isfinite(era_raw)]
    if np.any((valid_era<0)|(valid_era>1)):raise ValueError("Out-of-range ERA5 values; no clipping")
    daily_rows=[]; grid_rows=[]; time_rows=[]; sample_rows=[]; quality={}
    for region,region_id in REGIONS:
        mapped_parts=[]; missing_parts=[]; mapped_missing_parts=[]; raw_times={}; minimum=100.; maximum=0.; max_mass=0.
        for file_id,item in enumerate(files[region]):
            with netCDF4.Dataset(item["path"]) as ds:
                ds.set_auto_maskandscale(False)
                slat,slon=np.asarray(ds["lat"][:]),np.asarray(ds["lon"][:])
                raw_time,decoded,slots,error,units,calendar=_time_metadata(ds)
                cloud=np.asarray(ds["cldarea_total_1h"][:],float)
            missing=~np.isfinite(cloud)
            for fill in item["fill_values"]:missing|=cloud==fill
            good=cloud[~missing]
            if good.size and np.any((good<0)|(good>100)):raise ValueError("Out-of-range CERES percentages; no clipping")
            if good.size:minimum=min(minimum,float(good.min())); maximum=max(maximum,float(good.max()))
            rows,weights,area,kept=pilot.geometry(lat,lon,slat,slon,region_id)
            if rows!=plan["regions"][region]["cells"]:raise ValueError("Grid geometry differs from the frozen original footprint")
            if file_id==0:grid_rows.extend(rows)
            flat=cloud.reshape(len(raw_time),-1); flat_missing=missing.reshape(len(raw_time),-1)
            mapped_missing=flat_missing.astype(np.int32)@(weights.T>0).astype(np.int32)>0
            mapped=np.where(flat_missing,0,flat)@weights.T
            mapped[mapped_missing]=np.nan
            area_weight=area/area.sum()
            # Independent direct rectangular-footprint integral checks area conservation.
            cells=[row for row in rows if row["included"]]
            if len(kept)!=len({r["latitude"] for r in cells})*len({r["longitude"] for r in cells}):
                raise ValueError("Original selected footprint is not a complete rectangle")
            south,north=min(r["south"] for r in cells),max(r["north"] for r in cells)
            west,east=min(r["west"] for r in cells),max(r["east"] for r in cells)
            dy=np.maximum(0,np.sin(np.deg2rad(np.minimum(north,slat.astype(float)+.5)))-np.sin(np.deg2rad(np.maximum(south,slat.astype(float)-.5))))
            sx=slon.astype(float)%360
            dx=np.maximum(0,np.minimum(east,sx+.5)-np.maximum(west,sx-.5))
            direct=(dy[:,None]*np.deg2rad(dx[None,:])).ravel(); direct/=direct.sum()
            direct_mean=np.where(flat_missing,0,flat)@direct
            direct_mean[(flat_missing&(direct[None,:]>0)).any(1)]=np.nan
            regional=mapped@area_weight
            finite=np.isfinite(regional)&np.isfinite(direct_mean)
            if not np.array_equal(np.isfinite(regional),np.isfinite(direct_mean)):raise ValueError("Direct and two-stage masks differ")
            difference=float(np.max(abs(regional[finite]-direct_mean[finite]))) if finite.any() else 0.
            max_mass=max(max_mass,difference)
            if difference>1e-10 or not np.allclose(weights.sum(1),1,atol=1e-12,rtol=0):raise ValueError("Conservative integration verification failed")
            mapped_parts.append(mapped); missing_parts.append(missing); mapped_missing_parts.append(mapped_missing)
            raw_times[f"source_raw_time_file_{file_id:03d}"]=raw_time
            for i,slot in enumerate(slots):
                time_rows.append({"region":region,"source_file_id":file_id,"source_path":item["path"],"source_index":i,
                    "raw_time_value":float(raw_time[i]),"raw_time_units":units,"raw_time_calendar":calendar,
                    "decoded_time_utc":decoded[i].isoformat(),"nominal_midpoint_utc":slot.isoformat(),
                    "nominal_hour_slot_utc":slot.floor("h").isoformat(),"rounding_offset_seconds":error[i],
                    "used_in_4_slot_scheme":slot.hour in (0,6,12,18)})
            del cloud,flat,flat_missing,good
        mapped=np.concatenate(mapped_parts); missing=np.concatenate(missing_parts); mapped_missing=np.concatenate(mapped_missing_parts)
        ceres=mapped@area_weight
        era_cells=era_raw.reshape(1460,-1)[:,kept]*100
        era_missing=~np.isfinite(era_cells); era=era_cells@area_weight
        era_day=era.reshape(365,4).mean(1)
        c24=ceres.reshape(365,24).mean(1); c4=ceres.reshape(365,24)[:,[0,6,12,18]].mean(1)
        for d,date in enumerate(dates):
            season=SEASONS[date.month]
            daily_rows.append({"date_utc":date.date().isoformat(),"month":date.month,"calendar_season":season,
                "local_season":LOCAL_SEASONS[region][season],"region":region,"region_id":region_id,
                "era5_4_sample_mean_percent":era_day[d],"ceres_24_slot_mean_percent":c24[d],"ceres_4_halfhour_slot_mean_percent":c4[d],
                "difference_24_vs_4_pp":c24[d]-era_day[d],"difference_4_vs_4_pp":c4[d]-era_day[d],
                "era5_valid_samples_of_4":int(np.isfinite(era.reshape(365,4)[d]).sum()),
                "ceres_valid_samples_of_24":int(np.isfinite(ceres.reshape(365,24)[d]).sum()),
                "ceres_valid_samples_of_4":int(np.isfinite(ceres.reshape(365,24)[d,[0,6,12,18]]).sum()),
                "fully_covered_target_cells":len(kept)})
        region_times=[row for row in time_rows if row["region"]==region]
        for i,timestamp in enumerate(nominal):
            sample_rows.append({"region":region,"product":"CERES","sample_index":i,
                "time_utc":region_times[i]["decoded_time_utc"],"nominal_slot_utc":timestamp.isoformat(),"regional_cloud_percent":ceres[i]})
        for i,timestamp in enumerate(expected_era):
            sample_rows.append({"region":region,"product":"ERA5","sample_index":i,
                "time_utc":timestamp.isoformat(),"nominal_slot_utc":timestamp.isoformat(),"regional_cloud_percent":era[i]})
        np.savez_compressed(output/f"{region}_processed.npz",source_missing_mask=missing,source_latitude=slat,source_longitude=slon,
            ceres_target_cloud_percent=mapped,ceres_target_missing_mask=mapped_missing,era5_target_cloud_percent=era_cells,
            era5_target_missing_mask=era_missing,target_flat_indices=kept,conservative_weights=weights,target_area_weights=area_weight,
            target_area_m2=area*pilot.EARTH_RADIUS_M**2,**raw_times)
        quality[region]={"original_target_cells":len(rows),"included_target_cells":len(kept),
            "excluded_target_indices":[r["flat_grid_index"] for r in rows if not r["included"]],
            "source_files":len(files[region]),"source_missing_values":int(missing.sum()),"remapped_missing_values":int(mapped_missing.sum()),
            "era5_missing_target_values":int(era_missing.sum()),"source_range_percent":[minimum,maximum],
            "complete_CERES24_days":int(np.isfinite(c24).sum()),"complete_CERES4_days":int(np.isfinite(c4).sum()),
            "complete_ERA5_days":int(np.isfinite(era_day).sum()),"max_abs_time_offset_seconds":max(r["max_abs_nominal_offset_seconds"] for r in files[region]),
            "two_stage_vs_direct_footprint_max_difference_pp":max_mass,"weight_sum_error":float(np.max(abs(weights.sum(1)-1)))}
    daily=pd.DataFrame(daily_rows)
    if len(daily)!=730 or daily.duplicated(["date_utc","region"]).any():raise AssertionError("Expected 365 days by two unchanged regions")
    daily,metrics,references,anomalies=describe(daily)
    # January is a fixed regression check, not an additional result-selection rule.
    old=pd.read_csv(BASE/"revision_outputs/ceres_pilot_comparison/paired_daily.csv").set_index(["region","date_utc"])
    jan=daily[daily.month==1].set_index(["region","date_utc"]).loc[old.index]
    compare=["era5_4_sample_mean_percent","ceres_24_slot_mean_percent","ceres_4_halfhour_slot_mean_percent"]
    jan_error=float(np.nanmax(abs(jan[compare].to_numpy()-old[compare].to_numpy())))
    if not np.allclose(jan[compare],old[compare],atol=1e-8,rtol=0,equal_nan=True):raise AssertionError("January no longer reproduces the fixed pilot")
    tables={"paired_daily.csv":daily,"descriptive_metrics.csv":metrics,"monthly_reference_means.csv":references,
        "monthly_demeaned_correlations.csv":anomalies,"target_grid_support.csv":grid_rows,
        "regional_samples.csv":sample_rows,"ceres_time_mapping.csv":time_rows}
    for name,table in tables.items():pd.DataFrame(table).to_csv(output/name,index=False,float_format="%.12g")
    pilot.write_json(output/"quality_checks.json",quality)
    evidence_dir=output/"source_evidence"; evidence_dir.mkdir(exist_ok=True)
    for item in plan["evidence"]:shutil.copy2(item["path"],evidence_dir/Path(item["path"]).name)
    shutil.copy2(extension/"download_manifest.json",evidence_dir/"annual_download_manifest.json")
    shutil.copy2(extension/"request_plan.json",evidence_dir/"annual_acquisition_request_plan.json")
    shutil.copy2(BASE/"revision_outputs/ceres_pilot_comparison/paired_daily.csv",evidence_dir/"january_paired_daily_reference.csv")
    shutil.copy2(__file__,output/Path(__file__).name)
    for items in files.values():
        for item in items:
            if pilot.sha256(item["path"])!=item["sha256"]:raise ValueError("An input changed during processing")
    if pilot.sha256(acquisition_receipt["path"])!=acquisition_receipt["sha256"]:
        raise ValueError("Acquisition manifest changed during annual processing")
    if pilot.sha256(output/EXECUTION_ADDENDUM)!=execution_addendum_hash:
        raise ValueError("Frozen execution addendum changed during annual processing")
    _check_plan(output,extension)
    make_plot(daily,output)
    manifest={"status":"DESCRIPTIVE_ONE_YEAR_COMPARISON_COMPLETE_NOT_CAUSAL_OR_LONG_TERM_VALIDATION",
        "created_utc":pilot.utc_now(),"plan_sha256":pilot.sha256(output/"comparison_plan.json"),"script_sha256":pilot.sha256(__file__),
        "execution_addendum_sha256":execution_addendum_hash,"acquisition_manifest":acquisition_receipt,
        "input_manifest_sha256":pilot.sha256(output/"input_manifest.json"),"software":{"python":platform.python_version(),
            "numpy":np.__version__,"pandas":pd.__version__,"xarray":xr.__version__,"netCDF4":netCDF4.__version__},
        "daily_rows":730,"metric_rows":68,"monthly_reference_rows":24,"monthly_demeaned_correlation_rows":4,
        "january_pilot_max_difference_pp":jan_error,"quality_checks":quality,
        "annual_metrics":[r for r in metrics if r["group_type"]=="annual"],"monthly_demeaned_correlations":anomalies,
        "limitations":LIMITATIONS,"elapsed_seconds":time.perf_counter()-started}
    write_readme(output,manifest)
    manifest["outputs_sha256"]={str(p.relative_to(output)):pilot.sha256(p) for p in output.rglob("*") if p.is_file() and p.name!="run_manifest.json"}
    pilot.write_json(output/"run_manifest.json",manifest)
    print(json.dumps({"output":str(output),"annual_metrics":manifest["annual_metrics"],"elapsed_seconds":manifest["elapsed_seconds"]},indent=2))


def make_plot(daily,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True,sharey=True,layout="constrained")
    specs=[("era5_4_sample_mean_percent","ERA5: 4 samples/day","#222222","-"),
           ("ceres_24_slot_mean_percent","CERES: 24 slots/day","#0072B2","-"),
           ("ceres_4_halfhour_slot_mean_percent","CERES: 4 half-hour slots/day","#D55E00","--")]
    for ax,(region,region_id) in zip(axes,REGIONS):
        monthly=daily[daily.region==region].groupby("month")[[s[0] for s in specs]].mean()
        for column,label,color,style in specs:ax.plot(monthly.index,monthly[column],style,marker="o",ms=3,color=color,label=label)
        ax.set_title(f"{region.replace('_',' ').title()} | fixed region {region_id}",loc="left",fontsize=10)
        ax.set_ylim(0,100); ax.set_ylabel("Monthly mean of daily samples (%)"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc="lower left")
    axes[-1].set_xticks(range(1,13)); axes[-1].set_xlabel("Calendar month in 2019 | different temporal sampling operators")
    fig.suptitle("One-year descriptive CERES / ERA5 comparison",fontsize=12)
    fig.savefig(output/"monthly_comparison.png",dpi=200); fig.savefig(output/"monthly_comparison.pdf")
    plt.close(fig)


def write_readme(output,manifest):
    lines=["# CERES / ERA5 calendar-year 2019 descriptive comparison","",
        "This extends the fixed January processing comparison to all twelve months for the same two Atlantic footprints. It is not a formal hemisphere-effect test, causal validation, or long-term independent validation.","",
        "The two schemes remain unchanged: A compares daily means of 24 nominal CERES hourly slots with four ERA5 synoptic samples; B compares CERES 00:30/06:30/12:30/18:30 nominal slots with ERA5 00/06/12/18. These sampling operators are not physically equivalent averaging windows.","",
        "Both products share the original, fully covered ERA5 grid cells and spherical area weights. Missing masks propagate through each requested daily mean; no clipping, imputation or outcome-driven cell/scheme selection is used. CERES raw time arrays are preserved separately for each original file, with units and assignment receipts in ceres_time_mapping.csv and input_manifest.json.","",
        "| Region | Scheme | Paired days | Bias (pp) | MAE (pp) | RMSE (pp) | Pearson r |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in manifest["annual_metrics"]:
        values=[row[k] for k in ("bias_pp_ceres_minus_era5","mae_pp","rmse_pp","pearson_r")]
        scheme="A: 24 vs 4" if row["scheme"]==SCHEMES[0] else "B: 4 vs 4"
        lines.append(f"| {row['region']} | {scheme} | {row['n_paired_days']} | "+" | ".join("NA" if v is None else f"{v:.5f}" for v in values)+" |")
    lines += ["","Bias is CERES minus ERA5. Discrepancies do not treat either product as ground truth. No p-values or independent-sample intervals are reported.","",
        "descriptive_metrics.csv retains all 68 rows: annual, each calendar month, and DJF/MAM/JJA/SON for both regions and both schemes. DJF combines January--February and December within 2019, not a continuous cross-year winter. Local-season labels are provided without interpreting differences as a hemisphere effect.","",
        "monthly_demeaned_correlations.csv retains all four annual correlations after each product separately subtracts its own valid 2019 monthly means. All reference means/counts and daily anomalies are saved. This is descriptive within-year variation, not a trained climatology, held-out forecast or bias correction.","",
        "![Monthly descriptive means](monthly_comparison.png)","","## Evidence and reproduction","",
        "- comparison_plan.json: annual rules frozen before reading the new February--December cloud values; January had already been analysed.",
        "- execution_addendum_frozen.json: later execution freeze binding the final annual code, fixed January result and acquisition plan; the original plan is preserved and this is not research preregistration.",
        "- input_manifest.json / source_evidence/: input hashes, coordinate metadata and retained version-qualified temporal-support evidence.",
        "- source_evidence/annual_download_manifest.json: complete two-order receipt, checked against every new file, with authorized order/suborder identifiers, sizes and SHA256.",
        "- paired_daily.csv: exactly 730 date-by-region rows, both schemes, complete-sample counts and product-specific monthly-demeaned values.",
        "- *_processed.npz: source masks, original per-file numeric times, mapped cells, ERA5 cells and fixed area weights.",
        "- descriptive_metrics.csv / monthly_reference_means.csv / monthly_demeaned_correlations.csv: all planned summaries.",
        "- quality_checks.json / run_manifest.json: numerical checks, January replication, software, hashes and limitations.",
        "- monthly_comparison.png / .pdf: exportable figures; no uncertainty or effect-test claim.","",
        "Run revision/run_ceres_annual_comparison.py --freeze into a new output directory, then --freeze-execution before extension science files arrive. Run --run once both authorized regional deliveries and their complete download manifest are available. The original frozen plan and execution addendum are never silently overwritten.","","## Limits",""
    ]
    lines += [f"- {text}" for text in LIMITATIONS]
    (output/"README.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--extension-root",type=Path,default=EXTENSION)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze",action="store_true")
    modes.add_argument("--freeze-execution",action="store_true")
    modes.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.freeze:freeze(args.output,args.extension_root)
    elif args.freeze_execution:freeze_execution_addendum(args.output,args.extension_root)
    else:run(args.output,args.extension_root)


if __name__=="__main__":main()
