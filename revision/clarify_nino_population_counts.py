"""Clarify unavailable antecedent-month counts without changing model results.

Run once after run_external_nino_regimes.py. Preserve the original count table,
record its hash, and append a manifest amendment. No coefficient is recomputed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "revision_outputs/external_nino_regimes"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt_path = OUT / "population_counts_clarification.json"
    if receipt_path.exists():
        raise FileExistsError("Clarification already applied")
    path = OUT / "population_counts.csv"
    manifest_path = OUT / "manifest.json"
    before_manifest_sha = sha(manifest_path)
    before = sha(path)
    preserved = OUT / "population_counts_before_clarification.csv"
    shutil.copy2(path, preserved)
    model_before = sha(OUT / "nino_regime_contrasts.csv")
    df = pd.read_csv(path)
    df["requested_antecedent_month_labels"] = df.native_months
    df["available_native_months"] = df.native_months.where(df.state != "missing_context", 0)
    df["native_months"] = df.available_native_months
    df.to_csv(path, index=False)
    shutil.copy2(__file__, OUT / "code_snapshot" / Path(__file__).name)
    receipt = {"amended_utc":pd.Timestamp.now(tz="UTC").isoformat(),
               "reason":"The missing_context group refers to one unavailable requested month, November 1978. That is not an available native-month observation.",
               "changes":"Added requested_antecedent_month_labels and available_native_months. native_months now aliases available_native_months. Only the two nonempty training missing_context rows change native_months from 1 to 0; no model-population or coefficient changes.",
               "original_table":preserved.name,"original_table_sha256":before,"amended_table_sha256":sha(path),
               "original_manifest_sha256":before_manifest_sha,"script_sha256":sha(__file__),
               "model_results_sha256":model_before,"model_results_unchanged":sha(OUT/"nino_regime_contrasts.csv")==model_before,
               "reproduction":"Run revision/run_external_nino_regimes.py --freeze and --run, then revision/clarify_nino_population_counts.py. The original run and frozen scientific plan remain unchanged."}
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    readme=OUT/"README.md"
    with readme.open("a",encoding="utf-8") as stream:
        stream.write("\n## Population denominator clarification\n\nThe missing-context label requests November 1978, for which zero native observations are available. The population table now distinguishes requested_antecedent_month_labels from available_native_months; native_months aliases the latter. The original table and an explicit amendment receipt are retained. Contiguous episodes in the missing_context row count missing-data runs, not observed climate-state episodes. To reproduce this reporting clarification after the main script, run revision/clarify_nino_population_counts.py. No regression result changes.\n")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reporting_amendments"]=[receipt]
    manifest["outputs_sha256"]={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.name!="manifest.json"}
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
