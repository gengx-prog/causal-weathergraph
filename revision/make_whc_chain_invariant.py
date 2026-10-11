"""Write whc_chain_invariant_proof.json for a graph_overlap_null_round2 output directory.

Counts two-edge wind-humidity-cloud chains over unique directed edges of the
thresholded symmetric screens (q_hac64_global < 0.05; lags deduplicated) and
records the per-mediator in- and out-degrees whose products sum to the total.
Under any type-specific in/out-degree-preserving reference this total is fixed,
so it cannot serve as an enrichment statistic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["source_region", "target_region", "source_var", "target_var"]


def degrees(path: Path, n_regions: int = 66):
    frame = pd.read_csv(path)
    edges = frame.loc[frame.q_hac64_global < .05, COLS].drop_duplicates()
    wh = edges[(edges.source_var == "wind") & (edges.target_var == "humidity")]
    hc = edges[(edges.source_var == "humidity") & (edges.target_var == "cloud_cover")]
    indeg = np.bincount(wh.target_region, minlength=n_regions)
    outdeg = np.bincount(hc.source_region, minlength=n_regions)
    return indeg, outdeg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--directory", type=Path, required=True, help="graph_overlap_null_round2 output directory with frozen_plan.json")
    args = ap.parse_args()
    plan = json.loads((args.directory / "frozen_plan.json").read_text(encoding="utf-8"))
    cases = []
    for case, key in (("symmetric_full_topology", "full"), ("symmetric_late_vs_early_overlap", "late")):
        indeg, outdeg = degrees(Path(plan["inputs"][key]["path"]))
        products = indeg * outdeg
        cases.append({"case": case, "in_WH_by_mediator": indeg.tolist(), "out_HC_by_mediator": outdeg.tolist(),
                      "per_mediator_products": products.tolist(), "total_unique_edge_chains": int(products.sum())})
    proof = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "Two-edge WHC chains counted by unique directed variable-region edges, summing every W_i -> H_h -> C_j; lag labels have already been deduplicated.",
        "identity": "sum_(i,h,j) A_WH[i,h] A_HC[h,j] = sum_h (sum_i A_WH[i,h]) (sum_j A_HC[h,j]) = sum_h indegree_WH(h) outdegree_HC(h).",
        "consequence": "An exact type-specific in/out-degree-preserving graph reference holds this total fixed for every graph, independent of spatial constraints. More sampling cannot turn this invariant into a pathway-enrichment statistic. Endpoint counts and closure motifs need not be fixed.",
        "cases": cases,
    }
    (args.directory / "whc_chain_invariant_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print({c["case"]: c["total_unique_edge_chains"] for c in cases})


if __name__ == "__main__":
    main()
