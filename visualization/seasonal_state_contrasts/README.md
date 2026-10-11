# Seasonal and state contrasts

This folder renders circular views of seasonal and state contrasts from two
unchanged scientific summary CSVs. It does not refit the analysis or create new experiments.

## Rebuild

Requirements: Python 3, NumPy, pandas, Matplotlib. Verification additionally uses
PyMuPDF. Run from any working directory:

```sh
python render_state_contrasts.py
python verify_state_contrasts.py
```

The renderer resolves inputs relative to this file and writes PDF, SVG, PNG,
`plotted_values.csv`, and `value_audit.json` under `outputs/`. A different output
directory can be specified with `--output-dir PATH`. The verification script
checks the default output directory and generates `verification.json` plus a
preview at the intended 6.3-inch reproduction width. No original machine paths
are used to rebuild.

## Encoding and interpretation

- **Panel a:** three edge-type sectors and four fixed-width rings. Inner to outer:
  NH discovery, SH discovery, NH evaluation, SH evaluation. Each fill represents
  the signed local winter-minus-summer slope contrast. Labels use units of
  **10^-3**, rounded to two decimal places; colors retain full source precision on
  one symmetric continuous scale from -0.016 to +0.016. The twelve underlying
  estimates are descriptive; this panel introduces no individual confidence
  intervals. W = wind speed, H = specific humidity, C = cloud cover.
- **Panel b:** the original NH-minus-SH contrast estimates and pointwise 95%
  calendar-HAC64 intervals. The intervals retain the source analysis's same-time
  covariance treatment. They are neither simultaneous nor selection-adjusted
  intervals. The horizontal scale uses unmultiplied fitted slope units.
- **Panel c:** six state-contrast sectors; discovery is the inner ring and
  evaluation the outer ring. Each label is the exact number of edge-lag
  hypotheses with q_HAC64 < 0.05; color is the proportion of tested hypotheses,
  on one continuous 0-65% scale. Four state contrasts have denominator 2,574 each;
  each hemisphere's local-season contrast has denominator 1,287. These are
  dependent hypotheses, not independent observations. Panel c includes all six
  edge types in the original state summary, whereas panel a displays the three
  original edge types. No across-period adjustment is introduced.

D = discovery, 1979-2018; E = evaluation, 2019-2025. Circle radii and sector
angles are fixed categories and do not encode effect magnitude or significance.
Antecedent -24-hour state contrasts do not establish exogeneity. Local winter
and summer are hemisphere-specific; the plot does not equate global DJF with
winter.

## Output and provenance

The page is 7.6 x 3.5 inches, with minimum authored text of 10.4 pt (8.62 pt at
6.3 inches). PDF is entirely vector with embedded fonts, SVG retains editable
text, and PNG is 400 dpi. The diagrams summarize recorded experimental outputs.

`provenance.json` records relative experimental paths, SHA-256 hashes, and the
raw-row reconciliation already performed against the 2,550
hemisphere-pair rows and 25,740 state-test rows. Those larger raw files are not
needed by this renderer and are not duplicated. The packaged design JSON files
document the original statistical definitions. `outputs/verification.json`
separately checks every plotted value against the packaged original summaries.

The repository-level check makes isolated copies before rebuilding:

```sh
python scripts/verify_experiment_figures.py --output outputs/figure-check
```

Run that command from the repository root. It validates source values and vector
exports without refitting statistical models or modifying this directory.
