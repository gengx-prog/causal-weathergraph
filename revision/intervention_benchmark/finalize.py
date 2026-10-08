"""Verify local document exports and write durable evidence/source receipts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "revision_outputs/intervention_benchmark_v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf8"))
def write(path,x): Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def main():
    audit = read(OUT/"independent_audit.json")
    assert audit["status"] == "PASS_NUMERICS_MODELS_METRICS_SUMMARIES"
    run = read(OUT/"run_manifest.json")
    assert run["status"] == "completed" and len(run["completed"]) == 80
    english = read(OUT/"intervention_supplement_audit.json")
    assert english["status"] == "PASS_STANDALONE_INTERVENTION_SUPPLEMENT"
    for name,expected in run["source_sha256"].items():
        assert sha(ROOT/name) == expected,name
    manifest = read(OUT/"document_manifest.json")
    assert manifest["audit_sha256"] == sha(OUT/"independent_audit.json")
    for name,expected in manifest["files"].items():
        assert sha(OUT/name) == expected,name
    docx = [p for p in OUT.glob("*.docx") if p.name!="document_style.docx"]
    assert len(docx)==1
    pdf = docx[0].with_suffix(".pdf")
    pages=[]
    for filename in (pdf,OUT/"intervention_supplement.pdf"):
        doc=pymupdf.open(filename)
        for i,page in enumerate(doc):
            text=page.get_text()
            assert "\ufffd" not in text and "\x08" not in text and "\x0c" not in text
            spans=[s for b in page.get_text("dict")["blocks"] if "lines" in b for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            outside=[s["bbox"] for s in spans if s["bbox"][0]<10 or s["bbox"][1]<10 or s["bbox"][2]>page.rect.width-10 or s["bbox"][3]>page.rect.height-10]
            assert not outside,(filename,i,outside)
            pages.append(dict(file=filename.name,page=i+1,text_spans=len(spans),within_page=True))
        manifest["files"][filename.name]=sha(filename)
        manifest.setdefault("pdf_pages",{})[filename.name]=len(doc)
        doc.close()
    manifest["pdf_status"]="completed"
    manifest["finalized_utc"]=datetime.now(timezone.utc).isoformat()
    write(OUT/"document_manifest.json",manifest)
    write(OUT/"artifact_qa.json",dict(status="PASS_EXPORT_TEXT_GEOMETRY_AND_HASHES",pages=pages,visual_review="Recorded separately; geometry checks alone do not verify visual layout",document_manifest_sha256=sha(OUT/"document_manifest.json")))
    summary=pd.read_csv(OUT/"summary.csv")
    rows=summary[(summary.horizon_hours==6)&(summary.outcome=="cloud_proxy_pp")&(summary.method.isin(["unadjusted_ols","observed_z_ols","oracle_aipw"]))][["scenario","method","mean_effect_estimate","mean_test_bank_truth_mean","mean_signed_error_vs_test_bank","ate_rmse"]].to_dict("records")
    receipt=dict(schemaVersion=1,items=[dict(id="intervention-benchmark",title="Known intervention effects in an idealized simulator",queries=[dict(id="cloud-proxy-six-hours",source=dict(label="Frozen intervention benchmark and independent numerical audit",files=[dict(label="summary.csv"),dict(label="independent_audit.json"),dict(label="design.json")],metricDefinitions=[dict(label="Mean effect error",definition="Estimated mean effect minus paired test-bank mean effect, averaged over twenty independent repetitions; cloud-proxy percentage points.")],filters=["Four prespecified scenarios; 2048 factual training and 512 held-out test episodes per repetition","A=1 versus A=0 adds +2 versus -2 m/s for the first hour","Six-hour synthetic cloud-proxy outcome; full CSV also retains twelve hours and vapor"],caveats=["Idealized externally thermostatted moist tracers; no momentum dynamics or observed total cloud cover.","The common driver and assignment probabilities are known by construction; this is not real-atmosphere identification.","Scenarios share exogenous draws within a repetition. Twenty repetitions do not establish FDR or precise type-I calibration."]),columns=list(rows[0]),rows=rows)])])
    write(OUT/"answer_sources.json",receipt)
    node=Path("C:/Program Files/nodejs/node.exe")
    cli=Path.home()/".codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11/skills/visualize-data/scripts/render-inline-sources.mjs"
    subprocess.run([str(node),str(cli),"--input",str(OUT/"answer_sources.json"),"--output",str(OUT/"answer-sources.html")],check=True)
    snapshot=OUT/"code_snapshot"
    sources=list(Path(__file__).parent.glob("*.py"))+[Path(__file__).with_name("README.md"),ROOT/"tests/test_intervention_estimators.py",ROOT/"revision/inference.py"]+list((ROOT/"manuscript/revised").glob("intervention_benchmark_*.tex"))
    for path in sources:
        dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    evidence_files=[OUT/name for name in ["design.json","run_manifest.json","independent_audit.json","intervention_supplement_audit.json","all_metrics.csv","summary.csv","numerical_checks.json","resolution_sensitivity.csv","document_manifest.json","artifact_qa.json","answer_sources.json","answer-sources.html"]]+[OUT/name for name in manifest["files"]]
    write(Path(__file__).with_name("evidence_receipt.json"),dict(status="COMPLETED_BOUNDED_SIMULATOR_BENCHMARK",output_directory=str(OUT),artifact_sha256={str(p.relative_to(OUT)):sha(p) for p in evidence_files},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},limits="Model-internal policy total effects only; not atmospheric graph identification or natural mediation. Main manuscript integration pending."))
    print(json.dumps(dict(status="PASS",pdf_pages=manifest["pdf_pages"],metric_rows=len(pd.read_csv(OUT/"all_metrics.csv"))),ensure_ascii=False))


if __name__=="__main__": main()
