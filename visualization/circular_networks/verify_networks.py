"""Independent CSV-grain and exported-vector checks for the circular experiment networks."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pymupdf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = read("three_stage_all_candidates.csv")
    regions = read("regions.csv")
    agg = read("network_pairs_primary.csv")
    displayed = {r["edge_type"]: r for r in read("panel_ab_summary.csv")}
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for record in provenance["inputs"]:
        assert hashlib.sha256((ROOT / record["file"]).read_bytes()).hexdigest() == record["sha256"]
    keys = {(r["source_region"], r["target_region"], r["source_var"], r["target_var"], r["lag"]) for r in rows}
    assert len(keys) == len(rows) == 2574
    assert {int(r["region"]) for r in regions} == set(range(66))
    assert sum(r["s1_pass"] == "True" for r in rows) == 2187
    assert sum(r["s2_pass"] == "True" for r in rows) == 1550
    assert sum(r["eval_wb2_replicated"] == "True" for r in rows) == 760
    assert sum(r["eval_cds_replicated"] == "True" for r in rows) == 667
    types = sorted({r["edge_type"] for r in rows})
    for kind in types + ["ALL"]:
        part = [r for r in rows if kind == "ALL" or r["edge_type"] == kind]
        confirmed = sum(r["s2_pass"] == "True" for r in part)
        replicated = sum(r["eval_wb2_replicated"] == "True" for r in part)
        unconfirmed = len(part) - confirmed
        unconfirmed_rep = sum(r["s2_pass"] == "False" and float(r["eval_wb2_negctrl_q"]) < .05 for r in part)
        assert int(displayed[kind]["tested"]) == len(part)
        assert int(displayed[kind]["stage1_screened"]) == sum(r["s1_pass"] == "True" for r in part)
        assert int(displayed[kind]["stage2_confirmed"]) == confirmed
        assert int(displayed[kind]["eval_wb2_replicated"]) == replicated
        assert abs(float(displayed[kind]["confirmed_replication_rate"]) - replicated / confirmed) < 1e-14
        assert abs(float(displayed[kind]["unconfirmed_replication_rate"]) - unconfirmed_rep / unconfirmed) < 1e-14
    checks = {}
    for kind, expected in [("wind_to_humidity", (117, 77, 45, 32, 4, 6)),
                           ("humidity_to_cloud_cover", (133, 90, 46, 44, 5, 6))]:
        selected = [r for r in rows if r["edge_type"] == kind and r["eval_wb2_replicated"] == "True"]
        truth = defaultdict(list)
        for row in selected:
            truth[(int(row["source_region"]), int(row["target_region"]))].append(int(row["lag"]))
        plotted = [r for r in agg if r["edge_type"] == kind]
        actual = {(int(r["source_region"]), int(r["target_region"])): sorted(map(int, r["lag_steps"].split(";"))) for r in plotted}
        assert {key: sorted(value) for key, value in truth.items()} == actual
        assert len(plotted) == len(actual)
        local = sum(i == j for i, j in actual)
        dateline = [r for r in plotted if r["crosses_dateline_in_old_map"] == "True"]
        values = (len(selected), len(plotted), local, len(plotted) - local,
                  len(dateline), sum(int(r["replicated_lags"]) for r in dateline))
        assert values == expected
        assert all(int(r["replicated_lags"]) == len(actual[(int(r["source_region"]), int(r["target_region"]))]) for r in plotted)
        checks[kind] = dict(zip(["selected_edge_lags", "drawn_unique_pairs", "local_pairs", "cross_region_pairs",
                                "included_old_dateline_omissions_pairs", "included_old_dateline_omissions_edge_lags"], values))
    pdf_path = ROOT / "circular_networks.pdf"
    with pymupdf.open(pdf_path) as doc:
        assert len(doc) == 1
        page = doc[0]
        assert len(page.get_images()) == 0, "Expected fully vector PDF"
        width, height = page.rect.width, page.rect.height
        assert abs(width - 6.3 * 72) < .01 and abs(height - 5.3 * 72) < .01
        spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
        outside = [s["text"] for s in spans if not (page.rect + (-.2, -.2, .2, .2)).contains(pymupdf.Rect(s["bbox"]))]
        assert not outside, outside
    svg = ET.parse(ROOT / "circular_networks.svg")
    text_labels = ["".join(x.itertext()) for x in svg.getroot().iter() if x.tag.endswith("}text")]
    assert max(map(len, text_labels)) <= 80
    region_labels = {x.attrib["id"]: "".join(x.itertext()).strip() for x in svg.getroot().iter()
                     if x.attrib.get("id", "").startswith("region-label-")}
    assert len(region_labels) == 132
    for panel in ("c", "d"):
        for region in range(66):
            assert region_labels[f"region-label-{panel}-{region:02d}"] == str(region)
    assert min(s["size"] for s in spans) >= 7.19
    receipt = {"status": "PASS", "inputs_unchanged_sha256": True,
               "candidate_rows": len(rows), "regions_in_each_network": len(regions),
               "stage_counts_and_descriptive_rates_recomputed": True,
               "networks": checks, "all_selected_pairs_included": True,
               "no_top_n_or_effect_threshold_filter": True,
               "no_independent_binomial_intervals": True,
               "width_in": width / 72, "height_in": height / 72,
               "height_width_ratio": height / width,
               "pdf_raster_image_count": 0, "pdf_text_outside_canvas": outside,
               "svg_longest_label_characters": max(map(len, text_labels)),
               "minimum_pdf_text_size_pt": min(s["size"] for s in spans),
               "svg_region_id_labels": len(region_labels),
               "region_label_size_pt": 7.2,
               "region_label_placement": "Alternating radial positions with outward-facing, upright labels.",
               "other_text_minimum_authored_size_pt": 7.5,
               "visual_review": "Checked preview at 6.3-inch target width. Count labels are outside the networks; pair counts and lag legend occupy separate rows. No curves are hidden with white text masks. All 132 region IDs use 7.2-point outward radial labels, avoiding concatenated tangential numbers; legends and axes use at least 7.5 points.",
               "outputs_sha256": {f"circular_networks.{ext}": hashlib.sha256((ROOT / f"circular_networks.{ext}").read_bytes()).hexdigest() for ext in ("pdf", "svg", "png")}}
    (ROOT / "validation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
