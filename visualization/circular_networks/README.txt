Circular regional dependency networks

Rebuild from the packaged scientific data:
    python render_networks.py
    python verify_networks.py

Requirements: Python 3, matplotlib, numpy, pandas, and PyMuPDF for verification.
No reanalysis arrays, external repositories or new statistical fits are needed.
PDF and SVG are vector artwork; SVG preserves editable text. PNG is 300 dpi.
Canvas: 6.3 by 5.3 inches. Region IDs use 7.2-point outward radial labels;
other labels, axis ticks and legends use at least 7.5 points.

Inputs and transformations
- data/three_stage_all_candidates.csv: all 2,574 candidate-lag hypotheses.
- data/regions.csv: region IDs and geographic centroids (66 rows).
- data/three_stage_summary.csv: archived summary checked against every candidate.
- data/analysis_manifest.json: recorded inference rules and temporal boundaries.
- data/network_pairs_primary.csv: derived display table with complete selected
  pairs, lag lists and coordinates; regenerated from candidate rows.
- data/panel_ab_summary.csv: regenerated stage counts and rate denominators.
- provenance.json records exact input SHA-256 hashes. Scientific input bytes
  are preserved; descriptive filenames identify their experimental roles.

Panel interpretation
(a) Tested, screened, confirmed and primary-replicated counts by edge type.
(b) Confirmed/unconfirmed replication fractions, shown as percentages. These
    are dependent hypotheses; the unconfirmed comparison family is not known
    null. Families are adjusted separately; no binomial intervals are shown.
(c,d) Every unique directed region pair with at least one replicated lag is
    included for 2019-01-01 through 2023-01-10. All 66 regions are shown. Six
    sectors group latitude bands; longitude increases clockwise in each.
    Circular positions are schematic. Arrows are conditional predictive
    source-to-target links, not transport fluxes or identified physical causes.
    Halos denote local variable links. Stroke widths encode 1, 2 or 3 retained
    lags for each pair. W = wind speed; H = specific humidity; C = cloud cover.
    W -> H: 117 hypotheses, 77 pairs (45 local, 32 cross-region).
    H -> C: 133 hypotheses, 90 pairs (46 local, 44 cross-region).

Every selected pair is retained, including 4 W -> H and 5 H -> C pairs crossing
an earlier longitude-map boundary (6 retained hypotheses for each type).
Coefficient sign is not encoded: lag signs may alternate under full conditioning.

The repository-level isolated check is:
    python scripts/verify_experiment_figures.py --output outputs/figure-check
Run that command from the repository root. It leaves this source directory intact.
