"""Bounded downloader for the immutable, authorized global CERES request batch.

No request submission, derived-data processing, or legacy annual analysis.
Email and order-path session cookies arrive over loopback and remain in memory.
Use --probe for a real provider status/link check before starting the monitor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup
import netCDF4
import numpy as np
import pandas as pd
import psutil
import requests

BASE = Path(__file__).resolve().parents[2]
DEFAULT = BASE / "supplementary_data/cloud_validation/ceres_syn1deg_ed42_global_2001_2025"
HOST = "https://ceres-tool.larc.nasa.gov"
DOMAIN = "ceres-tool.larc.nasa.gov"
TOTAL_CAP = 25_000_000_000
FILE_CAP = 1_300_000_000


class ReviewRequired(Exception):
    """Only constant, non-sensitive error codes may be supplied."""
    pass


def require(condition, code):
    if not condition:
        raise ReviewRequired(code)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    temporary.replace(path)


def canonical_path(url):
    p=urlparse(url)
    require(p.scheme=="https" and p.hostname==DOMAIN and p.port in (None,443)
            and not p.username and not p.password and not p.query and not p.fragment,"UNSAFE_PROVIDER_URL")
    path=unquote(p.path)
    require(not any(c in path for c in ("\\","\x00","?","#")),"UNSAFE_PROVIDER_PATH")
    parts=[part for part in path.split("/") if part]
    require(all(part not in (".","..") for part in parts),"UNSAFE_PROVIDER_PATH")
    return "/"+"/".join(parts)


def directory_path(url,order_id):
    path=canonical_path(url)
    pattern=rf"/ord-tool/data[0-9]+/CERES_[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}:{re.escape(order_id)}/dir[0-9]+"
    require(re.fullmatch(pattern,path) is not None,"UNEXPECTED_ORDER_DIRECTORY")
    return path


def allowed_download(url,job):
    try:
        path=canonical_path(url)
        parent,name=path.rsplit("/",1)
        return parent==job["_directory"] and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.nc",name) is not None
    except (ReviewRequired,ValueError):
        return False


def expected_slots(job):
    start=pd.Timestamp(job["start_date"])
    end=pd.Timestamp(job["end_date"])
    if job["frequency"]=="hourly":
        return pd.date_range(start,end+pd.Timedelta(hours=23),freq="h")
    return pd.date_range(start,end,freq="D")


def load_contract(root):
    root=root.resolve()
    planfile=root/"request_plan.json"; receiptfile=root/"submission_receipt.json"
    plan=json.loads(planfile.read_text(encoding="utf8"))
    receipt=json.loads(receiptfile.read_text(encoding="utf8"))
    planhash=sha(planfile); receipthash=sha(receiptfile)
    require(receipt.get("request_plan_sha256")==planhash,"RECEIPT_PLAN_HASH_MISMATCH")
    require(Path(plan["download_directory"]).resolve()==root,"OUTPUT_DIFFERS_FROM_FROZEN_PLAN")
    require(plan["max_total_new_bytes"]==TOTAL_CAP and plan["max_single_file_bytes"]==FILE_CAP,"CAP_DIFFERS_FROM_AUTHORIZED_BATCH")
    require(plan["expected_spatial_shape"]==[180,360] and plan["extent_west_east_south_north"]==[0,360,-90,90],"NOT_GLOBAL_ONE_DEGREE_GRID")
    items=plan["jobs"]; mappings=receipt["jobs"]
    require(len(items)==plan["expected_job_count"]==23 and len(mappings)==23,"INCOMPLETE_JOB_MAPPING")
    by_id={m["job_id"]:m for m in mappings}
    require(len(by_id)==23 and set(by_id)=={j["job_id"] for j in items},"DUPLICATE_OR_UNKNOWN_JOB_MAPPING")
    jobs=[]; pairs=set(); directories=set()
    for source in items:
        item=dict(source); mapped=by_id[item["job_id"]]
        require(re.fullmatch(r"(?:hourly|daily)_[A-Za-z0-9_]+",item["job_id"]) is not None,"INVALID_JOB_IDENTIFIER")
        require(item["frequency"] in ("daily","hourly"),"UNSUPPORTED_FREQUENCY")
        for key in ("order_id","suborder_id"):
            value=mapped[key]
            require(isinstance(value,str) and value.isdigit(),"INVALID_ORDER_IDENTIFIER")
            item[key]=value
        require(not any(k in mapped for k in ("email","cookies","status_url")),"SECRET_OR_STATUS_URL_IN_RECEIPT")
        item["download_directory_url"]=urljoin(HOST,mapped["download_directory_url"])
        item["_directory"]=directory_path(item["download_directory_url"],item["order_id"])
        pair=(item["order_id"],item["suborder_id"])
        require(pair not in pairs and item["_directory"] not in directories,"DUPLICATE_SUBORDER_OR_DIRECTORY")
        pairs.add(pair); directories.add(item["_directory"])
        require(len(expected_slots(item))==item["expected_time_count"],"TIME_COUNT_DIFFERS_FROM_CALENDAR")
        # A confirmed provider mapping may clarify the expected daily name.
        actual=mapped.get("provider_variable",item["expected_variable"])
        if actual!=item["expected_variable"]:
            require(mapped.get("provider_schema_confirmed") is True,"UNCONFIRMED_PROVIDER_VARIABLE_MAPPING")
        require(re.fullmatch(r"cldarea_total_[A-Za-z0-9_]+",actual) is not None,"NOT_TOTAL_CLOUD_VARIABLE")
        item["_variable"]=actual
        jobs.append(item)
    require(sum(j["frequency"]=="hourly" for j in jobs)==18 and sum(j["frequency"]=="daily" for j in jobs)==5,"FREQUENCY_JOB_COUNT_MISMATCH")
    return plan,jobs,{"request_plan_sha256":planhash,"submission_receipt_sha256":receipthash}


def receive_context(port):
    challenge=secrets.token_urlsafe(32); state={}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path!="/challenge":self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(json.dumps({"challenge":challenge}).encode())
        def do_POST(self):
            if self.path!="/provider-context":self.send_error(404); return
            try:
                length=int(self.headers.get("Content-Length","0"))
                require(0<length<=65536,"INVALID_CONTEXT_LENGTH")
                payload=json.loads(self.rfile.read(length))
                require(secrets.compare_digest(str(payload.get("challenge","")),challenge),"INVALID_CHALLENGE")
                email=payload["email"]
                require(isinstance(email,str) and 3<=len(email)<=254 and "@" in email and not any(c in email for c in "\r\n"),"INVALID_EMAIL")
                cookies=payload.get("cookies",[])
                require(isinstance(cookies,list),"INVALID_COOKIE_LIST")
                for cookie in cookies:
                    require(isinstance(cookie,dict) and cookie.get("domain","").lstrip(".")==DOMAIN,"INVALID_COOKIE_DOMAIN")
                    require(cookie.get("path","").startswith("/ord-tool"),"COOKIE_OUTSIDE_ORDER_PATH")
                    require(all(isinstance(cookie.get(k),str) and not any(c in cookie[k] for c in "\r\n") for k in ("name","value")),"INVALID_COOKIE_VALUE")
                state["context"]={"email":email,"cookies":cookies}
            except Exception:
                self.send_error(400); return
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(b'{"accepted":true}')
    server=HTTPServer(("127.0.0.1",port),Handler); server.timeout=5
    deadline=time.monotonic()+300
    try:
        while "context" not in state and time.monotonic()<deadline:server.handle_request()
    finally:server.server_close()
    require("context" in state,"EPHEMERAL_CONTEXT_TIMEOUT")
    return state.pop("context")


def get_page(session,params):
    response=session.get(HOST+"/ord-tool/order",params=params,timeout=(15,60),allow_redirects=False)
    if response.status_code in (401,403) or 300<=response.status_code<400:
        raise ReviewRequired("PROVIDER_SESSION_REFRESH_REQUIRED")
    response.raise_for_status()
    require(len(response.content)<=4*1024*1024,"UNEXPECTED_PROVIDER_PAGE_SIZE")
    response.encoding="utf8"
    return BeautifulSoup(response.text,"html.parser")


def match_suborder_row(doc,job):
    matches=[]
    for tr in doc.find_all("tr"):
        # Provider HTML can omit closing </td>, so html.parser nests later
        # cells inside the product cell. Recover all cells whose nearest row
        # is this row, and exclude text belonging to a nested sibling cell.
        nodes=[c for c in tr.find_all(["td","th"]) if c.find_parent("tr") is tr]
        cells=[" ".join(str(s).strip() for s in c.find_all(string=True)
                        if s.find_parent(["td","th"]) is c and str(s).strip()) for c in nodes]
        if len(cells)<12 or "CERES_SYN1deg_Ed4.2" not in cells[1]:continue
        ids=[]
        for anchor in tr.select("a[href]"):
            query={k.lower():v for k,v in parse_qs(urlparse(anchor["href"]).query).items()}
            ids+=query.get("suborderid",[])
        for element in tr.select("[onclick]"):
            ids+=re.findall(r"\bviewOneSuborder\(\s*([0-9]+)\s*\)",element["onclick"])
        if job["suborder_id"] in ids or cells[0].strip()==job["suborder_id"]:matches.append(cells)
    require(len(matches)==1,"SUBORDER_STATUS_ROW_NOT_UNIQUE_OR_ABSENT")
    return matches[0]


def public_status(doc,job):
    row=match_suborder_row(doc,job)
    counts=[]
    for value in row[6:10]:
        require(re.fullmatch(r"[0-9]+",value.strip()) is not None,"UNRECOGNIZED_PROVIDER_FILE_COUNTS")
        counts.append(int(value))
    require(counts[0]>0 and all(n<=counts[0] for n in counts[1:]),"INCONSISTENT_PROVIDER_FILE_COUNTS")
    require(counts[3]==0,"PROVIDER_REPORTED_FAILED_FILES")
    text=row[5].strip().lower()
    status=text if re.fullmatch(r"[a-z][a-z _-]{0,59}",text) else "unrecognized_status_label"
    return {"job_id":job["job_id"],"order_id":job["order_id"],"suborder_id":job["suborder_id"],
        "checked_utc":now(),"provider_status":status,"expected_files":counts[0],"completed_files":counts[1],
        "processing_files":counts[2],"failed_files":counts[3]}


def status_and_links(session,email,job,order_cache):
    if job["order_id"] not in order_cache:
        order_cache[job["order_id"]]=get_page(session,{"command":"oneOrderStatus","email":email,
            "orderId":job["order_id"],"CERESProducts":"SYN1degEd42"})
    public=public_status(order_cache[job["order_id"]],job)
    doc=get_page(session,{"command":"oneSuborderStatus","email":email,
        "SuborderId":job["suborder_id"],"CERESProducts":"SYN1degEd42"})
    links=[]
    for anchor in doc.select("a[href]"):
        url=urljoin(HOST+"/ord-tool/order",anchor["href"])
        if urlparse(url).path.lower().endswith(".nc"):
            require(allowed_download(url,job),"SCIENCE_LINK_OUTSIDE_MATCHED_SUBORDER_DIRECTORY")
            links.append(url)
    links=sorted(set(links))
    require(len(links)<=public["expected_files"],"MORE_LINKS_THAN_ORDER_FILES")
    public["available_download_links"]=len(links)
    public["state"]="files_available" if links else "queued_or_processing_without_download_links"
    return public,links


def retained_files(root):
    return [p for p in root.rglob("*") if p.is_file() and
            (p.name.lower().endswith(".nc") or p.name.lower().endswith((".part",".partial")))]


def retained_bytes(root):
    files=retained_files(root)
    require(all(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) for p in files),"SCIENCE_PATH_OUTSIDE_BATCH")
    return sum(p.stat().st_size for p in files)


def inspect_dataset(ds,job):
    """Read actual CF time coordinates, all cloud values, and the exact global grid."""
    ds.set_auto_maskandscale(False)
    require("4.2" in str(getattr(ds,"Version","")),"UNEXPECTED_PRODUCT_EDITION")
    require(np.array_equal(np.asarray(ds["lat"][:]),np.arange(-89.5,90,1)),"NOT_180_GLOBAL_LATITUDE_CENTERS")
    require(np.array_equal(np.asarray(ds["lon"][:]),np.arange(.5,360,1)),"NOT_360_GLOBAL_LONGITUDE_CENTERS")
    require(job["_variable"] in ds.variables,"EXPECTED_TOTAL_CLOUD_VARIABLE_ABSENT")
    var=ds[job["_variable"]]; t=ds["time"]
    require(var.dimensions==("time","lat","lon") and var.shape[1:]==(180,360),"WRONG_CLOUD_GRID_OR_DIMENSIONS")
    require([name for name,v in ds.variables.items() if v.dimensions==("time","lat","lon")]==[job["_variable"]],"UNEXPECTED_ADDITIONAL_SCIENCE_VARIABLE")
    require(str(getattr(var,"units","")).lower() in ("percent","%"),"CLOUD_UNIT_NOT_PERCENT")
    require(float(getattr(var,"scale_factor",1))==1 and float(getattr(var,"add_offset",0))==0,"PACKED_CLOUD_REQUIRES_REVIEW")
    require(float(getattr(t,"scale_factor",1))==1 and float(getattr(t,"add_offset",0))==0,"PACKED_TIME_REQUIRES_REVIEW")
    raw=np.asarray(t[:]).copy(); calendar=str(getattr(t,"calendar","standard")); units=str(getattr(t,"units",""))
    require(raw.ndim==1 and len(raw)>0 and np.isfinite(raw).all(),"INVALID_RAW_TIME")
    require(calendar in ("standard","gregorian","proleptic_gregorian"),"UNEXPECTED_CF_CALENDAR")
    decoded=pd.DatetimeIndex([str(v) for v in netCDF4.num2date(raw.astype(float),units,calendar=calendar)])
    unit="h" if job["frequency"]=="hourly" else "D"
    slots=decoded.floor(unit)
    require(not slots.has_duplicates and slots.is_monotonic_increasing,"DUPLICATE_OR_UNSORTED_TIME_SLOTS")
    step=pd.Timedelta(hours=1) if unit=="h" else pd.Timedelta(days=1)
    require(len(slots)==1 or np.all(np.diff(slots.values)==step.to_timedelta64()),"MISSING_TIME_SLOTS_INSIDE_FILE")
    expected=expected_slots(job)
    require(slots[0]>=expected[0] and slots[-1]<=expected[-1] and len(slots)<=len(expected),"FILE_DATES_OUTSIDE_AUTHORIZED_JOB")
    require(var.shape[0]==len(raw),"CLOUD_TIME_LENGTH_MISMATCH")
    offsets=(decoded-slots).total_seconds().to_numpy()
    # No assumed Daily midpoint. Preserve every observed within-day/hour offset.
    decoded_delta=np.diff(decoded.values).astype("timedelta64[us]").astype(np.int64)/1e6
    unit_name=units.split(" since ",1)[0].strip().lower()
    factors={"day":86400.,"days":86400.,"hour":3600.,"hours":3600.,"minute":60.,"minutes":60.,
             "second":1.,"seconds":1.,"millisecond":.001,"milliseconds":.001,"microsecond":.000001,"microseconds":.000001}
    require(unit_name in factors,"UNSUPPORTED_CF_NUMERIC_TIME_UNIT")
    half_ulp_seconds=(float(np.max(np.abs(np.spacing(raw))))*factors[unit_name]/2 if np.issubdtype(raw.dtype,np.floating) else 0.)
    tolerance_seconds=half_ulp_seconds+2e-6
    expected_step_seconds=3600. if unit=="h" else 86400.
    require(not len(decoded_delta) or np.max(abs(decoded_delta-expected_step_seconds))<=2*tolerance_seconds,"CF_TIME_STEPS_EXCEED_NUMERIC_PRECISION")
    hourly_error=float(np.max(abs(offsets-1800.))) if unit=="h" else None
    if unit=="h":require(hourly_error<=tolerance_seconds,"HOURLY_MIDPOINT_OUTSIDE_DTYPE_ROUNDING_BOUND")
    fills=[np.asarray(var.getncattr(k)).ravel() for k in ("_FillValue","missing_value") if k in var.ncattrs()]
    fill_values=[float(v) for array in fills for v in array]
    good_count=missing_count=0; minimum=float("inf"); maximum=float("-inf")
    for start in range(0,len(raw),24):
        values=np.asarray(var[start:start+24],dtype=np.float64)
        missing=~np.isfinite(values)
        for fill in fill_values:missing|=values==fill
        good=values[~missing]
        require(np.all((good>=0)&(good<=100)),"CLOUD_VALUES_OUTSIDE_ZERO_100")
        good_count+=good.size; missing_count+=int(missing.sum())
        if good.size:minimum=min(minimum,float(good.min())); maximum=max(maximum,float(good.max()))
    require(good_count>0,"ALL_CLOUD_VALUES_MISSING")
    temporal={"source_raw_time":raw,"decoded_time_utc":decoded.values.astype("datetime64[us]"),
              "calendar_slots_utc":slots.values.astype("datetime64[us]"),"offset_seconds_from_slot_start":offsets}
    result={"checks":"PASS_FULL_GRID_CF_TIME_CLOUD_RANGE","shape":list(var.shape),"variable":job["_variable"],
        "units":str(var.units),"version":str(getattr(ds,"Version","")),"valid_values":int(good_count),
        "missing_values":int(missing_count),"fill_values":[v if np.isfinite(v) else str(v) for v in fill_values],"cloud_percent_min":minimum,"cloud_percent_max":maximum,
        "time_count":len(raw),"time_units":units,"time_calendar":calendar,"time_dtype":str(raw.dtype),
        "first_decoded_time":decoded[0].isoformat(),"last_decoded_time":decoded[-1].isoformat(),
        "first_calendar_slot":slots[0].isoformat(),"last_calendar_slot":slots[-1].isoformat(),
        "within_slot_offset_seconds_min":float(offsets.min()),"within_slot_offset_seconds_max":float(offsets.max()),
        "actual_time_step_seconds_min":float(decoded_delta.min()) if len(decoded_delta) else None,
        "actual_time_step_seconds_max":float(decoded_delta.max()) if len(decoded_delta) else None,
        "time_half_ulp_seconds_max":half_ulp_seconds,"time_numeric_tolerance_seconds":tolerance_seconds,
        "hourly_nominal_midpoint_max_error_seconds":hourly_error,
        "time_bounds_variable":str(getattr(t,"bounds","absent")),"cloud_cell_methods":str(getattr(var,"cell_methods","absent")),
        "time_interpretation":"Actual CF times retained; calendar-bin continuity checked. Hourly midpoint convention is a numeric consistency check, not proof of within-hour cloud averaging. No assumed Daily midpoint or claim of equivalent temporal support to ERA5."}
    return result,temporal


def inspect_file(path,job):
    with netCDF4.Dataset(path) as ds:return inspect_dataset(ds,job)


def job_complete(job,records):
    selected=[r for r in records if r["job_id"]==job["job_id"]]
    if not selected:return False
    spans=[]
    for item in selected:
        validation=item["validation"]
        first=pd.Timestamp(validation["first_calendar_slot"]); last=pd.Timestamp(validation["last_calendar_slot"])
        slots=pd.date_range(first,last,freq="h" if job["frequency"]=="hourly" else "D")
        require(len(slots)==validation["time_count"],"RECORDED_TIME_COUNT_INCONSISTENT")
        spans.extend(slots)
    actual=pd.DatetimeIndex(sorted(spans))
    require(not actual.has_duplicates,"OVERLAPPING_SCIENCE_FILE_TIME_SPANS")
    return actual.equals(expected_slots(job))


def download(session,url,job,root,deadline):
    require(allowed_download(url,job),"UNAUTHORIZED_DOWNLOAD_URL")
    folder=root/job["job_id"]; folder.mkdir(exist_ok=True)
    name=canonical_path(url).rsplit("/",1)[-1]
    final=folder/name; partial=folder/(name+".partial")
    require(not final.exists() and not partial.exists(),"EXISTING_SCIENCE_OR_PARTIAL_REQUIRES_REVIEW")
    with session.get(url,stream=True,timeout=(15,120),allow_redirects=False) as response:
        require(response.status_code==200,"DOWNLOAD_NOT_HTTP_200")
        raw_length=response.headers.get("Content-Length","0")
        require(raw_length.isdigit(),"INVALID_CONTENT_LENGTH")
        announced=int(raw_length)
        require(announced<=FILE_CAP,"ANNOUNCED_FILE_EXCEEDS_CAP")
        require(retained_bytes(root)+announced<=TOTAL_CAP,"ANNOUNCED_BATCH_EXCEEDS_CAP")
        h=hashlib.sha256(); n=0
        with partial.open("xb") as stream:
            for block in response.iter_content(1024*1024):
                if not block:continue
                require(time.time()<deadline,"MONITORING_TIME_LIMIT_DURING_DOWNLOAD")
                require(n+len(block)<=FILE_CAP,"STREAM_FILE_EXCEEDS_CAP")
                require(retained_bytes(root)+len(block)<=TOTAL_CAP,"STREAM_BATCH_EXCEEDS_CAP")
                stream.write(block); h.update(block); n+=len(block)
    require(n>0 and (not announced or announced==n),"INCOMPLETE_DOWNLOAD")
    with partial.open("rb") as stream:require(stream.read(3)==b"CDF","NOT_EXPECTED_NETCDF3_PAYLOAD")
    validation,coordinates=inspect_file(partial,job)
    coordinate_path=folder/(name+".time_coordinates.npz")
    require(not coordinate_path.exists(),"UNTRACKED_TIME_COORDINATE_ARTIFACT")
    with coordinate_path.open("xb") as stream:np.savez_compressed(stream,**coordinates)
    partial.rename(final)
    return {"job_id":job["job_id"],"order_id":job["order_id"],"suborder_id":job["suborder_id"],
        "path":str(final.resolve()),"url":url,"bytes":n,"sha256":h.hexdigest(),"downloaded_utc":now(),
        "validation":validation,"time_coordinates_path":str(coordinate_path.resolve()),"time_coordinates_sha256":sha(coordinate_path)}


def resume(root,jobs,hashes):
    science=retained_files(root)
    require(retained_bytes(root)<=TOTAL_CAP,"EXISTING_BATCH_EXCEEDS_CAP")
    require(not any(p.name.lower().endswith((".part",".partial")) for p in science),"PRESERVED_PARTIAL_REQUIRES_REVIEW")
    manifest=root/"download_manifest.json"
    if not manifest.exists():
        require(not science,"UNTRACKED_SCIENCE_FILES")
        return []
    saved=json.loads(manifest.read_text(encoding="utf8"))
    require(saved.get("contract_hashes")==hashes,"RESUME_CONTRACT_CHANGED")
    records=saved["files"]; by_id={j["job_id"]:j for j in jobs}; known=set()
    for item in records:
        require(item["job_id"] in by_id,"UNKNOWN_JOB_IN_MANIFEST")
        job=by_id[item["job_id"]]; path=Path(item["path"]).resolve()
        require(path.parent==(root/job["job_id"]).resolve() and path.is_file(),"MISSING_OR_WRONG_RESUME_PATH")
        require(path not in known,"DUPLICATE_RESUME_FILE")
        require(all(item[k]==job[k] for k in ("order_id","suborder_id")) and allowed_download(item["url"],job),"RESUME_PROVIDER_MAPPING_CHANGED")
        require(path.stat().st_size==item["bytes"]<=FILE_CAP and sha(path)==item["sha256"],"RESUME_HASH_OR_SIZE_MISMATCH")
        result,_=inspect_file(path,job)
        require(result==item["validation"],"RESUME_VALIDATION_CHANGED")
        cp=Path(item["time_coordinates_path"]).resolve()
        require(cp.parent==path.parent and cp.is_file() and sha(cp)==item["time_coordinates_sha256"],"RESUME_COORDINATE_ARTIFACT_CHANGED")
        known.add(path)
    require(known=={p.resolve() for p in science},"UNTRACKED_SCIENCE_FILES")
    for job in jobs:job_complete(job,records)
    return records


def save_manifest(root,jobs,records,hashes):
    complete=[job["job_id"] for job in jobs if job_complete(job,records)]
    write_json(root/"download_manifest.json",{"updated_utc":now(),"contract_hashes":hashes,"files":records,
        "verified_science_bytes":sum(r["bytes"] for r in records),"all_retained_science_and_partial_bytes":retained_bytes(root),
        "completed_jobs":complete,"expected_jobs":len(jobs),"complete":len(complete)==len(jobs),
        "validation_scope":"Full cloud range, global one-degree centers, raw CF time/calendar-slot continuity and per-job complete union; no derived analysis or scientific validation claim."})
    return complete


def verify_contract_unchanged(root,hashes):
    require(sha(root/"request_plan.json")==hashes["request_plan_sha256"] and
            sha(root/"submission_receipt.json")==hashes["submission_receipt_sha256"],"IMMUTABLE_CONTRACT_CHANGED_DURING_MONITORING")


@contextmanager
def exclusive_worker(root):
    """Prevent concurrent writers; remove only this batch's dead-process lock."""
    root.mkdir(parents=True,exist_ok=True)
    lock=root/"worker.lock"
    if lock.exists():
        old=json.loads(lock.read_text(encoding="utf8"))
        require(not psutil.pid_exists(int(old["pid"])),"ANOTHER_BATCH_WORKER_IS_ACTIVE")
        require(lock.resolve().parent==root.resolve(),"LOCK_OUTSIDE_BATCH")
        lock.unlink()
    with lock.open("x",encoding="utf8") as stream:
        json.dump({"pid":os.getpid(),"started_utc":now()},stream)
    try:yield
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding="utf8")).get("pid")==os.getpid():lock.unlink()


def run(args):
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    plan,jobs,hashes=load_contract(root)
    status_path=root/("provider_probe.json" if args.probe else "download_status.json")
    previous=json.loads(status_path.read_text(encoding="utf8")) if status_path.exists() else {}
    began=previous.get("started_utc",now()) if not args.probe else now()
    end=datetime.fromisoformat(began).timestamp()+args.max_hours*3600
    status={"started_utc":began,"updated_utc":now(),"state":"awaiting_ephemeral_provider_context",
        "pid":os.getpid(),"contract_hashes":hashes,"code_sha256":sha(__file__),"jobs":len(jobs),
        "poll_seconds":args.poll_seconds,"max_hours":args.max_hours,"max_total_bytes":TOTAL_CAP,"max_file_bytes":FILE_CAP,
        "email_or_cookies_persisted":False,"derived_analysis_started":False,"probe_only":args.probe}
    write_json(status_path,status)
    try:
        records=resume(root,jobs,hashes)
        completed=[j["job_id"] for j in jobs if job_complete(j,records)]
        if len(completed)==len(jobs):
            status.update(state="all_jobs_downloaded_and_verified",updated_utc=now(),completed_jobs=completed)
            write_json(status_path,status); return 0
        if not args.probe:
            probe_path=root/"provider_probe.json"
            require(probe_path.exists(),"REAL_PROVIDER_PROBE_REQUIRED_BEFORE_MONITORING")
            probe=json.loads(probe_path.read_text(encoding="utf8"))
            require(probe.get("state")=="provider_probe_passed" and probe.get("matched_suborders")==len(jobs)
                    and probe.get("contract_hashes")==hashes,"REAL_PROVIDER_PROBE_NOT_COMPLETE_FOR_THIS_CONTRACT")
        require(time.time()<end,"MONITORING_48H_LIMIT_ALREADY_REACHED")
        context=receive_context(args.context_port); email=context.pop("email")
        session=requests.Session()
        session.headers.update({"User-Agent":"CERES-authorized-bounded-download/1.0","Accept-Encoding":"identity"})
        for cookie in context.pop("cookies"):
            session.cookies.set(cookie["name"],cookie["value"],domain=cookie["domain"],path=cookie["path"])
        transient=0
        try:
            while time.time()<end:
                verify_contract_unchanged(root,hashes)
                cycle=[]; cache={}
                for job in jobs:
                    if job["job_id"] in completed and not args.probe:continue
                    status["active_job_id"]=job["job_id"]
                    try:
                        public,links=status_and_links(session,email,job,cache)
                    except requests.RequestException as exc:
                        transient+=1
                        cycle.append({"job_id":job["job_id"],"checked_utc":now(),"network_error_type":type(exc).__name__})
                        continue
                    cycle.append(public)
                    if not args.probe:
                        for url in links:
                            if any(r["job_id"]==job["job_id"] and r["url"]==url for r in records):continue
                            require(time.time()<end,"MONITORING_TIME_LIMIT_REACHED_BEFORE_DOWNLOAD")
                            status.update(state="downloading_serially",updated_utc=now(),provider_status=cycle)
                            write_json(status_path,status)
                            records.append(download(session,url,job,root,end))
                            completed=save_manifest(root,jobs,records,hashes)
                        if public["completed_files"]==public["expected_files"] and len(links)==public["expected_files"]:
                            require(job["job_id"] in completed,"FINISHED_PROVIDER_JOB_HAS_INCOMPLETE_DATE_COVERAGE")
                status.update(updated_utc=now(),provider_status=cycle,completed_jobs=completed,
                    verified_files=len(records),retained_bytes=retained_bytes(root),transient_network_errors=transient)
                if args.probe:
                    matched=sum("provider_status" in x for x in cycle)
                    status.update(state="provider_probe_passed" if matched==len(jobs) else "provider_probe_incomplete",
                        matched_suborders=matched,downloads_started=False)
                    write_json(status_path,status); return 0 if matched==len(jobs) else 2
                if len(completed)==len(jobs):
                    status.update(state="all_jobs_downloaded_and_verified"); write_json(status_path,status); return 0
                status.update(state="waiting_for_provider"); write_json(status_path,status)
                time.sleep(min(args.poll_seconds,max(0,end-time.time())))
            status.update(state="time_limit_reached",updated_utc=now()); write_json(status_path,status); return 2
        finally:
            session.cookies.clear(); session.close(); email=None
    except Exception as exc:
        status.update(state="stopped_requires_review",updated_utc=now(),error_type=type(exc).__name__,
            error_code=str(exc) if isinstance(exc,ReviewRequired) else "UNEXPECTED_ERROR_REDACTED")
        write_json(status_path,status)
        print("CERES global worker stopped: "+status["error_code"],file=sys.stderr)
        return 1


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=DEFAULT)
    ap.add_argument("--context-port",type=int,default=8767)
    ap.add_argument("--poll-seconds",type=int,default=300)
    ap.add_argument("--max-hours",type=float,default=48)
    ap.add_argument("--probe",action="store_true")
    args=ap.parse_args()
    if args.poll_seconds<300 or not 0<args.max_hours<=48 or not 1024<=args.context_port<=65535:
        ap.error("Require poll >=300 seconds, monitoring <=48 hours, and a nonprivileged loopback port")
    # No traceback can expose a credential-bearing requests URL, even during setup.
    try:
        with exclusive_worker(args.output.resolve()):return run(args)
    except Exception as exc:
        print("CERES global worker setup failed: "+(str(exc) if isinstance(exc,ReviewRequired) else type(exc).__name__),file=sys.stderr)
        return 1


if __name__=="__main__":raise SystemExit(main())
