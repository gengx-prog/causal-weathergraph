"""Download NOAA circulation indices and audit their calendar grain.

No resampling, interpolation, lag-label construction, or scientific experiments.
Only the Python standard library is required. Repeat an existing audit without
network traffic with --reuse-existing; immutable raw bytes are SHA256 checked.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import urllib.request


DEFAULT_OUTPUT = Path("D:/Paper2/Major Revision/supplementary_data/circulation_indices")
SOURCES = [
    dict(id="ao", frequency="daily", units="standardized_index",
         field="ao_index_cdas", sentinel=None,
         url="https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.cdas.z1000.19500101_current.csv",
         documentation="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml"),
    dict(id="aao", frequency="daily", units="standardized_index",
         field="aao_index_cdas", sentinel=None,
         url="https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.aao.cdas.z700.19790101_current.csv",
         documentation="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/aao.shtml"),
    dict(id="nao", frequency="daily", units="standardized_index",
         field="nao_index_cdas", sentinel=None,
         url="https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv",
         documentation="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/nao.shtml"),
    dict(id="nino34", frequency="monthly", units="degree_C_anomaly",
         field="NINA34", sentinel=-99.99, observed_sentinel_candidates=(-9999.0,),
         url="https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.csv",
         documentation="https://psl.noaa.gov/data/timeseries/month/Nino34/"),
]


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def expected_dates(start, end, frequency):
    if frequency == "daily":
        return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    current = start.replace(day=1)
    result = []
    while current <= end:
        result.append(current)
        current = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return result


def fetch(source, raw_dir, root):
    request = urllib.request.Request(source["url"], headers={
        "User-Agent": "CausalWeatherGraph-MajorRevision-index-ingest/1.0",
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.1",
        "Accept-Encoding": "identity",
    })
    started = utcnow()
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"{source['id']}: HTTP {response.status}")
        content = response.read()
        receipt = dict(source, requested_at_utc=started, downloaded_at_utc=utcnow(),
                       resolved_url=response.url, http_status=response.status,
                       http_headers={key: response.headers.get(key) for key in
                                     ("Content-Type", "Content-Length", "Last-Modified", "ETag", "Date")})
    if not content or content.lstrip().lower().startswith((b"<html", b"<!doctype")):
        raise ValueError(f"{source['id']}: empty or HTML response instead of CSV")
    filename = source["url"].rsplit("/", 1)[-1]
    path = raw_dir / filename
    path.write_bytes(content)
    receipt.update(raw_path=path.relative_to(root).as_posix(), bytes=len(content), sha256=sha256(content))
    save_json(path.with_suffix(".receipt.json"), receipt)
    return receipt


def parse_source(source, content):
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows:
        raise ValueError(f"{source['id']}: no CSV rows")
    header = [x.strip() for x in rows[0]]
    daily = source["frequency"] == "daily"
    if daily and header != ["year", "month", "day", source["field"]]:
        raise ValueError(f"{source['id']}: unexpected schema {header}")
    if not daily and (len(header) != 2 or header[0] != "Date" or not header[1].startswith("NINA34")):
        raise ValueError(f"{source['id']}: unexpected schema {header}")
    if not daily and "missing value -99.99" not in header[1]:
        raise ValueError("nino34: documented missing-value token changed")
    parsed = []
    for line, row in enumerate(rows[1:], 2):
        if not row or all(not token.strip() for token in row):
            continue
        if len(row) != len(header):
            raise ValueError(f"{source['id']} line {line}: field count mismatch")
        if daily:
            day = dt.date(*(int(token.strip()) for token in row[:3]))
        else:
            day = dt.date.fromisoformat(row[0].strip())
            if day.day != 1:
                raise ValueError(f"nino34 line {line}: monthly date not month start")
        raw_value = row[-1].strip()
        if raw_value.lower() in ("", "na", "n/a", "null", "nan"):
            value, status = None, "missing_token"
        else:
            value = float(raw_value)
            if not math.isfinite(value):
                value, status = None, "nonfinite"
            elif source["sentinel"] is not None and value == source["sentinel"]:
                value, status = None, "documented_missing_sentinel"
            elif value in source.get("observed_sentinel_candidates", ()):
                value, status = None, "observed_missing_sentinel_not_declared_in_header"
            else:
                status = "valid"
        parsed.append(dict(date=day, value=value, source_value=raw_value,
                           value_status=status, source_line=line))
    return header, parsed


def profile(rows, expected=None):
    counts = collections.Counter(row["date"] for row in rows)
    valid = [row["value"] for row in rows if row["value"] is not None]
    missing_dates = sorted(set(expected or []) - set(counts))
    off_calendar = sorted(set(counts) - set(expected or counts))
    missing_values = [dict(date=row["date"].isoformat(), source_line=row["source_line"],
                           source_value=row["source_value"], status=row["value_status"])
                      for row in rows if row["value"] is None]
    # A deliberately broad screening flag, not a truncation or a physical bound.
    range_flags = [dict(date=row["date"].isoformat(), value=row["value"], source_line=row["source_line"])
                   for row in rows if row["value"] is not None and abs(row["value"]) > 20]
    return dict(rows=len(rows), unique_dates=len(counts),
                first_date=min(counts).isoformat() if counts else None,
                last_date=max(counts).isoformat() if counts else None,
                expected_rows=len(expected) if expected is not None else None,
                missing_date_count=len(missing_dates), missing_dates=[d.isoformat() for d in missing_dates],
                unexpected_dates=[d.isoformat() for d in off_calendar],
                duplicate_extra_rows=sum(n - 1 for n in counts.values()),
                duplicate_dates={d.isoformat(): n for d, n in sorted(counts.items()) if n > 1},
                source_order_strictly_increasing=all(a["date"] < b["date"] for a, b in zip(rows, rows[1:])),
                valid_values=len(valid), missing_value_count=len(missing_values), missing_values=missing_values,
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
                range_screen="abs(value) > 20 is flagged, never removed; not a physical validity threshold",
                range_flag_count=len(range_flags), range_flags=range_flags)


def save_csv(path, rows, source):
    fields = ["date", "value", "source_value", "value_status", "index", "frequency", "units", "source_line"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, date=row["date"].isoformat(), index=source["id"],
                                 frequency=source["frequency"], units=source["units"]))
    return dict(path=path.name, bytes=path.stat().st_size, sha256=sha256(path.read_bytes()))


def crosscheck_ascii(root, manifest, reuse_existing=False):
    """Compare separate CPC formats without altering either source or target CSV."""
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    audit_dir = root / "ascii_crosscheck" / run_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    report = dict(audited_at_utc=utcnow(), policy="Read-only comparison: no CSV value replacement or imputation.",
                  comparison="All nonmissing matched dates; report numeric differences and an explicit 5e-4 absolute precision tolerance.", indices=[])
    for source, item in zip(SOURCES[:3], manifest["indices"][:3]):
        start_code = "790101" if source["id"] == "aao" else "500101"
        ascii_source = dict(source, id=source["id"] + "_ascii",
                            url=f"https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.{source['id']}.index.b{start_code}.current.ascii")
        filename = ascii_source["url"].rsplit("/", 1)[-1]
        receipt_candidates = sorted((root / "ascii_crosscheck").glob(f"*/{filename}.receipt.json"))
        if reuse_existing and receipt_candidates:
            receipt = json.loads(receipt_candidates[-1].read_text(encoding="utf-8"))
            if receipt["url"] != ascii_source["url"] or sha256((root / receipt["raw_path"]).read_bytes()) != receipt["sha256"]:
                raise ValueError(f"{source['id']}: ASCII receipt hash or URL mismatch")
        else:
            receipt = fetch(ascii_source, audit_dir, root)
        ascii_content = (root / receipt["raw_path"]).read_text(encoding="utf-8-sig")
        ascii_rows = []
        for line_number, line in enumerate(ascii_content.splitlines(), 1):
            if not line.strip():
                continue
            # CPC uses I4,I3,I3,F7.3: a negative sentinel may touch the day.
            if len(line) < 11:
                raise ValueError(f"Unexpected ASCII row {source['id']}:{line_number}: {line!r}")
            parts = [line[:4], line[4:7], line[7:10], line[10:].strip()]
            date = dt.date(*map(int, parts[:3]))
            value = float(parts[3])
            ascii_rows.append(dict(date=date, value=value if math.isfinite(value) and value != -99 else None,
                                   source_value=parts[3], source_line=line_number,
                                   value_status=("sentinel_candidate_minus_99" if value == -99 else
                                                 "valid" if math.isfinite(value) else "nonfinite")))
        dates = [row["date"] for row in ascii_rows]
        if len(dates) != len(set(dates)):
            raise ValueError(f"Duplicate ASCII dates for {source['id']}")
        ascii_by_date = {row["date"]: row for row in ascii_rows}
        _, csv_rows = parse_source(source, (root / item["download"]["raw_path"]).read_bytes())
        differences = []
        missing_comparison = []
        examples = []
        for row in csv_rows:
            other = ascii_by_date.get(row["date"])
            if row["value"] is None:
                missing_comparison.append(dict(date=row["date"].isoformat(), csv_source_value=row["source_value"],
                                               ascii_value=other["value"] if other else None,
                                               ascii_source_value=other["source_value"] if other else None,
                                               ascii_value_status=other["value_status"] if other else None,
                                               ascii_source_line=other["source_line"] if other else None))
            elif other is not None and other["value"] is not None:
                difference = abs(row["value"] - other["value"])
                differences.append(difference)
                if len(examples) < 5 or row["date"] in (dt.date(1979, 1, 1), dt.date(2019, 1, 1), dt.date(2025, 12, 31)):
                    examples.append(dict(date=row["date"].isoformat(), csv_value=row["value"], ascii_value=other["value"], abs_difference=difference))
        report["indices"].append(dict(id=source["id"], download=receipt,
            ascii_profile=profile(ascii_rows, expected_dates(min(dates), max(dates), "daily")),
            csv_first_date=csv_rows[0]["date"].isoformat(), csv_last_date=csv_rows[-1]["date"].isoformat(),
            csv_dates_absent_from_ascii=[row["date"].isoformat() for row in csv_rows if row["date"] not in ascii_by_date],
            compared_nonmissing_dates=len(differences), maximum_absolute_difference=max(differences) if differences else None,
            values_differing_exactly=sum(delta != 0 for delta in differences),
            differences_greater_than_5e_minus_4=sum(delta > 5e-4 for delta in differences),
            csv_missing_values_checked=missing_comparison, examples=examples))
    save_json(audit_dir / "crosscheck.json", report)
    lines = ["# CPC CSV versus official ASCII cross-check", "", "Original CSV target files were not filled or replaced.", "",
             "| Index | Matched nonmissing dates | Max absolute difference | Differences > 0.0005 | CSV / ASCII end dates | CSV missing dates and ASCII tokens |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for item in report["indices"]:
        missing = "; ".join(f"{row['date']}: {row['ascii_source_value']}" for row in item["csv_missing_values_checked"])
        lines.append(f"| {item['id']} | {item['compared_nonmissing_dates']} | {item['maximum_absolute_difference']} | {item['differences_greater_than_5e_minus_4']} | {item['csv_last_date']} / {item['ascii_profile']['last_date']} | {missing} |")
    lines += ["", "The 0.0005 tolerance is a declared precision comparison, not permission to merge formats. Exact original numeric strings, HTTP receipts, and full comparison details are preserved under ascii_crosscheck/. In particular, -99.000 is flagged as a sentinel candidate and is not a recovered observation. An ASCII value at a CSV-missing date is not an automatically approved replacement. Source revisions, update dates, and differing precision must be considered before any future reconciliation."]
    (audit_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(ascii_crosscheck_report=str(audit_dir / "crosscheck.json"),
                          indices=[dict(id=item["id"], compared=item["compared_nonmissing_dates"],
                                        max_difference=item["maximum_absolute_difference"],
                                        differences_above_tolerance=item["differences_greater_than_5e_minus_4"],
                                        csv_missing_values_checked=item["csv_missing_values_checked"]) for item in report["indices"]]), indent=2))
    return dict(path=(audit_dir / "crosscheck.json").relative_to(root).as_posix(),
                summary=[dict(id=item["id"], compared_nonmissing_dates=item["compared_nonmissing_dates"],
                              maximum_absolute_difference=item["maximum_absolute_difference"],
                              differences_greater_than_5e_minus_4=item["differences_greater_than_5e_minus_4"],
                              csv_missing_values_checked=item["csv_missing_values_checked"],
                              csv_last_date=item["csv_last_date"], ascii_last_date=item["ascii_profile"]["last_date"])
                         for item in report["indices"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(1979, 1, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2025, 12, 31))
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cross-check-ascii", action="store_true", help="Fetch three small official CPC ASCII files for comparison only")
    args = parser.parse_args()
    if args.start > args.end or args.start.day != 1:
        parser.error("start must be on month day 1 and no later than end")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    old = {}
    if args.reuse_existing:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipts = {entry["id"]: entry["download"] for entry in old["indices"]}
    else:
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        raw_dir = root / "raw" / run_id
        raw_dir.mkdir(parents=True, exist_ok=False)
        receipts = {}
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            pending = {executor.submit(fetch, source, raw_dir, root): source for source in SOURCES}
            for task in concurrent.futures.as_completed(pending):
                source = pending[task]
                try:
                    receipts[source["id"]] = task.result()
                except Exception as error:
                    failures.append(dict(id=source["id"], url=source["url"], error=repr(error)))
        if failures:
            save_json(raw_dir / "download_failures.json", failures)
            raise RuntimeError(f"Download failures preserved in {raw_dir}; {failures}")
    manifest = dict(audited_at_utc=utcnow(), python=platform.python_version(),
                    script_path=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__).read_bytes()),
                    target_start=args.start.isoformat(), target_end=args.end.isoformat(),
                    processing="Original frequency retained; no filling, interpolation, resampling, deduplication, or lag labels.",
                    monthly_date_semantics="YYYY-MM-01 labels the entire calendar month, not an instantaneous day-1 measurement.",
                    context_policy="One preceding calendar day for daily indices; one preceding calendar month for monthly indices. Missing context is reported, never invented.",
                    indices=[])
    if old.get("ascii_crosscheck_report"):
        manifest["ascii_crosscheck_report"] = old["ascii_crosscheck_report"]
        manifest["ascii_crosscheck_summary"] = old.get("ascii_crosscheck_summary", [])
    annual_rows = []
    for source in SOURCES:
        receipt = receipts[source["id"]]
        raw_path = (root / receipt["raw_path"]).resolve()
        if not raw_path.is_relative_to(root):
            raise ValueError("Raw receipt path escaped output directory")
        content = raw_path.read_bytes()
        if sha256(content) != receipt["sha256"] or receipt["url"] != source["url"]:
            raise ValueError(f"{source['id']}: raw hash or source URL mismatch")
        header, rows = parse_source(source, content)
        target = [row for row in rows if args.start <= row["date"] <= args.end]
        prior = args.start - dt.timedelta(days=1)
        context_date = prior if source["frequency"] == "daily" else prior.replace(day=1)
        context = [row for row in rows if row["date"] == context_date]
        expectation = expected_dates(args.start, args.end, source["frequency"])
        whole_expectation = expected_dates(min(row["date"] for row in rows), max(row["date"] for row in rows), source["frequency"])
        item = dict(id=source["id"], frequency=source["frequency"], units=source["units"], download=receipt,
                    source_header=header, raw=profile(rows, whole_expectation), target=profile(target, expectation),
                    context=profile(context, [context_date]))
        item["missing_value_conventions"] = dict(documented_sentinel=source["sentinel"],
            observed_additional_sentinel_candidates=list(source.get("observed_sentinel_candidates", ())),
            source_header_discrepancies=[dict(date=row["date"].isoformat(), source_value=row["source_value"],
                                             source_line=row["source_line"])
                                        for row in rows if row["value_status"] == "observed_missing_sentinel_not_declared_in_header"])
        item["target_csv"] = save_csv(root / f"{source['id']}_{args.start.year}_{args.end.year}_{source['frequency']}.csv", target, source)
        item["context_csv"] = save_csv(root / f"{source['id']}_preceding_context.csv", context, source)
        item["target_complete_unique_finite"] = (len(target) == len(expectation)
            and item["target"]["missing_date_count"] == 0 and item["target"]["duplicate_extra_rows"] == 0
            and item["target"]["missing_value_count"] == 0 and not item["target"]["unexpected_dates"])
        for year in range(args.start.year, args.end.year + 1):
            yearly = profile([row for row in target if row["date"].year == year], [d for d in expectation if d.year == year])
            annual_rows.append(dict(index=source["id"], year=year, **{key: yearly[key] for key in
                               ("rows", "unique_dates", "expected_rows", "missing_date_count", "duplicate_extra_rows", "missing_value_count", "minimum", "maximum")}))
        manifest["indices"].append(item)
    with (root / "annual_completeness.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annual_rows[0]))
        writer.writeheader()
        writer.writerows(annual_rows)
    manifest["all_target_periods_complete_unique_finite"] = all(item["target_complete_unique_finite"] for item in manifest["indices"])
    manifest["notes"] = [
        "AAO starts in 1979: no observed 1978-12-31 context exists in this source. A previous-day AAO label for 1979-01-01 is unavailable.",
        "AO/AAO/NAO are standardized indices from differing definitions and projection levels; do not treat their absolute magnitudes as directly comparable.",
        "Nino3.4 is the supplied HadISST1.1 monthly anomaly product; source -99.99 is missing. Raw files retain all provider values, including missing-value sentinels.",
        "The downloaded Nino3.4 CSV declares -99.99 in its header, but 2026-08 through 2026-12 contain -9999.000. These five out-of-target entries are marked as observed missing-sentinel candidates rather than valid observations; exact original tokens remain unchanged in the raw file and manifest. All 564 target months in 1979-2025 remain valid.",
        "The full downloaded files may contain incomplete or sentinel-filled months after the 2025 target end; raw and target quality are reported separately.",
        "A daily label repeated across four ERA5 times is not four independent index observations. No six-hourly values were created.",
        "Freeze any project-fitted threshold using the training period only. Index provider baselines are not project-fitted thresholds; external indices are not guaranteed causally exogenous.",
        "For real-time prediction, verify historical release dates and revisions separately; preceding calendar periods alone do not establish real-time availability.",
    ]
    save_json(manifest_path, manifest)
    lines = ["# Downloaded NOAA circulation indices", "", f"Audit UTC: {manifest['audited_at_utc']}", "",
             f"Target: {args.start} through {args.end}. Daily and monthly grains are preserved.", "",
             "| Index | Raw rows | Target rows / expected | Missing dates | Missing values | Duplicate extra rows | Target range | Context rows / expected |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for item in manifest["indices"]:
        t = item["target"]
        lines.append(f"| {item['id']} | {item['raw']['rows']} | {t['rows']} / {t['expected_rows']} | {t['missing_date_count']} | {t['missing_value_count']} | {t['duplicate_extra_rows']} | {t['minimum']} to {t['maximum']} | {item['context']['rows']} / 1 |")
    lines += ["", "## Files and audit limits", "", "- `raw/`: immutable complete downloaded bytes and HTTP/URL/time/SHA256 receipts.",
              "- `*_1979_2025_*.csv`: original-frequency target-period rows; original numeric token and source line retained.",
              "- `*_preceding_context.csv`: preceding day or month only; AAO is an empty header-only CSV because context is unavailable.",
              "- `manifest.json`: complete date-gap, duplicate, value-status and range audit, source URLs, timestamps and hashes.",
              "- `annual_completeness.csv`: per-index, per-year completeness and value range.",
              "", "No scientific regime classification or hypothesis test has been run. Range checks use a broad absolute-value flag of 20; this is a screening rule, not a physical bound or outlier removal.", ""]
    lines += [f"- {note}" for note in manifest["notes"]]
    lines += ["", "## Missing target values retained", ""]
    for item in manifest["indices"]:
        if item["target"]["missing_values"]:
            dates = ", ".join(row["date"] for row in item["target"]["missing_values"])
            lines.append(f"- {item['id']}: {dates}. These are provider blank values at existing dates, not missing dates. A previous-day label is also unavailable on each following date unless a separate, documented reconciliation is approved.")
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(output_dir=str(root), all_target_periods_complete_unique_finite=manifest["all_target_periods_complete_unique_finite"],
                          indices=[dict(id=item["id"], raw_rows=item["raw"]["rows"], target=item["target"], context=item["context"])
                                   for item in manifest["indices"]]), ensure_ascii=True, indent=2))
    if args.cross_check_ascii:
        comparison = crosscheck_ascii(root, manifest, args.reuse_existing)
        manifest["ascii_crosscheck_report"] = comparison["path"]
        manifest["ascii_crosscheck_summary"] = comparison["summary"]
        save_json(manifest_path, manifest)
    if manifest.get("ascii_crosscheck_report"):
        with (root / "README.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n## Separate official-format cross-check\n\nSee `{manifest['ascii_crosscheck_report']}` and its adjacent README. This comparison does not replace any CSV value.\n")
            for item in manifest.get("ascii_crosscheck_summary", []):
                tokens = "; ".join(f"{row['date']} = {row['ascii_source_value']}" for row in item["csv_missing_values_checked"])
                stream.write(f"\n- {item['id']}: CSV missing dates have ASCII tokens {tokens}; -99.000 is not a recovered measurement. Compared {item['compared_nonmissing_dates']:,} matched nonmissing values: maximum absolute difference {item['maximum_absolute_difference']}; differences greater than 0.0005 = {item['differences_greater_than_5e_minus_4']}. CSV ends {item['csv_last_date']}; ASCII ends {item['ascii_last_date']}.\n")


if __name__ == "__main__":
    main()
