"""Verify delivered artifacts and record hashes without rerunning experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "revision_outputs/transport_consistency_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def main():
    audit = read(OUT / "independent_audit.json")
    assert audit["status"] == "PASS_INPUTS_MODELS_EXTERNAL"
    assert audit["audit_code_sha256"] == sha(Path(__file__).with_name("audit.py"))
    stages = {k: v["passed"] for k, v in audit.items() if isinstance(v, dict) and "passed" in v}
    assert len(stages) == 4 and all(stages.values()), stages
    assert audit["design_sha256"] == sha(OUT / "design.json")
    assert audit["inputs"]["input_sha256"] == sha(OUT / "inputs/input_data.npz")
    manifest = read(OUT / "document_manifest.json")
    for group in ["files", "source_files"]:
        for name, expected in manifest[group].items():
            assert sha(OUT / name) == expected, name
    assert manifest["script_sha256"] == sha(Path(__file__).with_name("build_report.py"))
    english = read(OUT / "manuscript_preview_audit.json")
    assert english["status"] == "PASS_STANDALONE_SUPPLEMENT"
    for name, record in english["hashes"].items():
        assert sha(name) == record["sha256"], name

    word_files = [p for p in OUT.glob("*.docx") if p.name != "document_style.docx"]
    assert len(word_files) == 1
    chinese_pdf = word_files[0].with_suffix(".pdf")
    document = fitz.open(chinese_pdf)
    assert len(document) == 5
    page_checks = []
    full_text = ""
    for number, page in enumerate(document, 1):
        text = page.get_text()
        assert "\ufffd" not in text and "\x08" not in text and "\x0c" not in text
        full_text += text
        spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        outside = [s["bbox"] for s in spans if s["bbox"][0] < 20 or s["bbox"][1] < 20 or s["bbox"][2] > page.rect.width - 20 or s["bbox"][3] > page.rect.height - 20]
        assert not outside, (number, outside)
        page_checks.append({"page": number, "text_spans": len(spans), "text_within_20pt_margins": True})
    for token in ["6.35", "2.79", "CERES"]:
        assert token in full_text, token
    document.close()
    manifest["independent_audit_sha256"] = sha(OUT / "independent_audit.json")
    manifest["pdf_export_status"] = "completed"
    manifest["files"][chinese_pdf.name] = sha(chinese_pdf)
    manifest["files"]["manuscript_preview.pdf"] = sha(OUT / "manuscript_preview.pdf")
    manifest["chinese_pdf_pages"] = 5
    manifest["english_pdf_pages"] = english["pages"]
    manifest["finalized_utc"] = datetime.now(timezone.utc).isoformat()
    write(OUT / "document_manifest.json", manifest)
    write(OUT / "artifact_qa.json", {
        "status": "PASS_DELIVERED_ARTIFACTS",
        "scientific_audit_stages": stages,
        "english_source_and_artifact_hashes_verified": True,
        "chinese_page_checks": page_checks,
        "previous_visual_review": "All five Chinese pages as contact sheet and full figure inspected by root; all three English pages as contact sheet inspected by independent agent.",
        "main_manuscript_integration": "Pending: supplementary modules and documents only",
        "document_manifest_sha256": sha(OUT / "document_manifest.json"),
        "finalized_utc": manifest["finalized_utc"],
    })
    snapshot = OUT / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    sources = list(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("README.md")]
    sources += list((ROOT / "manuscript/revised").glob("transport_consistency_*"))
    source_receipts = {}
    for path in sources:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_receipts[str(path.relative_to(ROOT))] = sha(path)
    artifacts = ["design.json", "inputs/input_manifest.json", "inputs/input_data.npz", "model/run_manifest.json", "external/manifest.json", "independent_audit.json", "document_manifest.json", "artifact_qa.json", "manuscript_preview_audit.json", "answer_sources.json", "answer-sources.html"]
    artifacts += list(manifest["files"])
    receipt = {
        "status": "COMPLETED_BOUNDED_EXPERIMENT",
        "output_directory": str(OUT),
        "artifact_sha256": {name: sha(OUT / name) for name in dict.fromkeys(artifacts)},
        "source_sha256": source_receipts,
        "limits": "Retrospective two-region predictive diagnostics; no causal identification; main manuscript integration pending.",
        "finalized_utc": manifest["finalized_utc"],
    }
    write(Path(__file__).with_name("evidence_receipt.json"), receipt)
    print(json.dumps({"status": "PASS_DELIVERED_ARTIFACTS", "pages": [5, 3], "audit_stages": stages, "artifacts": len(receipt["artifact_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
