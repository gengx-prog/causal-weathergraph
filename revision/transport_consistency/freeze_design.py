"""Freeze the bounded transport experiment before deriving new field diagnostics."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "revision_outputs" / "transport_consistency_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "design.json"
    if target.exists():
        raise FileExistsError("Frozen design already exists; use a documented addendum")
    design = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_NEW_TRANSPORT_VALUES_AND_MODEL_RESULTS",
        "scope": "Bounded retrospective physical-transport consistency experiment; not a new causal identification theorem or a full Navier-Stokes solver",
        "prior_exposure": "These historical ERA5 and two-region 2019 CERES data have been examined previously. This is not a pristine temporal holdout or registered confirmatory study.",
        "regions": [{"id": 15, "name": "south_atlantic"}, {"id": 48, "name": "north_atlantic"}],
        "region_selection": "The two unchanged original 36-cell CERES comparison footprints, selected for previously available external coverage, not new transport performance. Ocean-focused, no claim of an all-ocean mask or global representativeness.",
        "data_period": ["1979-01-01", "2025-12-31"],
        "task_horizons": {"humidity_change": 6, "cloud_level": 6},
        "targets": {"humidity_change": "1000*(area_mean_q(t+6h)-area_mean_q(t)); g/kg", "cloud_level": "100*area_mean_total_cloud_cover(t+6h); percentage points"},
        "forecast_origins": "00 UTC daily; inputs at t,t-6h,t-12h; actual six-hour timestamps; no intervening gap",
        "splits": {"train": ["1979-01-01", "2014-12-31"], "validation": ["2015-01-01", "2018-12-31"], "evaluation": ["2019-01-01", "2025-12-31"]},
        "sample_policy": "Every row from t-12h through t+6h is in one split and one input file/source segment. All models use identical eligible rows. No imputation. Forecasts are retrospective reanalysis-based estimates, not operational forecasts.",
        "source_segments": ["WB2_1979_2018", "WB2_2019_2023-01-10", "CDS_2023-01-11_2025"],
        "physics": {
            "earth_radius_m": 6371000.0,
            "operator": "MFC=-spherical_horizontal_divergence(q*u,q*v); central face flux reconstruction, periodic longitude, zero polar boundary length, finite-volume spherical cell areas",
            "resolution": "64x32 resolved ERA5 grid; not native-resolution turbulence",
            "coordinates": "Use original WB2 reference grid and record CDS latitude rounding tolerance up to 0.001 degree; this does not establish equivalence of acquisition/remapping kernels",
            "aggregation": "Compute MFC on grid then area average over complete original region cells. Raw u/v/q/T/C use the same spherical area weights. Six physical controls retain existing arithmetic regional means as common background covariates and are labelled accordingly.",
            "flux_features": ["area_mean(q)*area_mean(u)", "area_mean(q)*area_mean(v)", "area_mean(q*u)-area_mean(q)*area_mean(u)", "area_mean(q*v)-area_mean(q)*area_mean(v)", "area_mean(MFC)"],
            "decomposition_limit": "Regional resolved covariance identity, not a decomposition of MFC into separately differentiated regional means, not turbulent subgrid closure",
            "horizontal_budget_comparator": "Unfitted humidity-change proxy 21600*1000*MFC(t); constant horizontal tendency approximation, missing vertical transport and sources/sinks",
            "terrain": "Record coarse surface-pressure diagnostics; no native terrain mask. No claim that coarse regional sp establishes above-ground validity of every original pressure-level value.",
            "missing_physics": ["matched-layer omega850 and vertical humidity derivative", "complete moisture sources/sinks", "cloud microphysics and vertical cloud overlap"]
        },
        "models": {
            "fit": "Separate model per region, no evaluation refit",
            "common": "u,v,q,T850,C plus six existing physical controls at t,t-6h,t-12h and sine/cosine annual calendar phase",
            "baselines": ["persistence: delta-q=0, cloud=C(t)", "training-only monthly target climatology", "unfitted horizontal-only humidity tendency"],
            "ridge": ["local", "local+mean_flux", "local+mean_flux+covariance", "local+mean_flux+covariance+MFC", "same_grid", "same_grid+all_flux"],
            "same_grid": "All u/v/q/T values in 36 core cells and all immediate cardinal halo cells (60 cells per region), at all three input times, plus common inputs. Exact same raw information supports derived flux features.",
            "ridge_alpha_candidates": [1, 10, 100, 1000],
            "ridge_preprocess": "Fit centering/scaling only on training; choose alpha by validation MSE for each target; save all validation scores and chosen alpha",
            "hgb": {"variants": ["same_grid", "same_grid+all_flux"], "max_iter": 120, "max_leaf_nodes": 15, "min_samples_leaf": 50, "learning_rate": 0.05, "l2_regularization": 1.0, "max_bins": 64, "early_stopping": False, "random_state": 20261001},
            "cloud_bounds": "Apply the same [0,100] clip to every cloud predictor; retain raw predictions and clipping fractions",
            "threads": 2
        },
        "diagnostics": {
            "spatial_mismatch": "Other hemisphere's simultaneous flux added to local model; spatial-correspondence sensitivity, not an exchangeable causal null",
            "future_information": "Flux at t+6h added to local model; deliberate unavailable-future-information diagnostic, stored separately, never used as an implementable forecast or negative control",
            "no_sign_flip_null": "A sign flip followed by refitting is an invertible reparameterization, not a useful null"
        },
        "comparisons": {
            "primary": "Humidity-change paired MSE reduction for ridge same_grid+all_flux versus ridge same_grid; both regions and equal-region pooled estimate",
            "secondary": ["same-information HGB comparison", "local mean-flux/covariance/MFC sequential ablations", "cloud prediction", "unfitted humidity budget comparator"],
            "metrics": ["MSE", "RMSE", "MAE", "bias", "paired MSE reduction", "paired MAE reduction"],
            "groups": ["entire evaluation", "each region", "each source segment", "local DJF/MAM/JJA/SON season with southern names shifted six months"],
            "intervals": "500 paired 30-calendar-day block bootstrap draws, same selected calendar blocks across both regions, sampled within acquisition segments; exploratory 95% percentile intervals conditional on fitted models, not simultaneous/FDR-calibrated or accounting for model-selection uncertainty",
            "seed": 20261001,
            "reporting": "Retain all positive, null and negative results; do not select regions, models, windows or title according to evaluation gains"
        },
        "external_cloud_check": {
            "source": str(ROOT.parent / "revision_outputs/ceres_annual_comparison_2019"),
            "period": "2019 only, both original regions",
            "operator": "Match cloud forecast valid at 06:00 UTC to nominal CERES 06:30 hourly slot over identical area-weighted footprint; retain raw CF time mapping and note distinct temporal averaging operators",
            "models": "All implementable cloud models, raw percentage scale, no fit or bias correction using CERES",
            "limit": "Cross-product forecast comparison, not accuracy against truth, not independent graph validation; CERES hours/product and reanalysis have different sampling/process support"
        },
        "required_checks": ["source hashes and unit/time alignment", "finite values/ranges and area weights", "same-grid halo coverage", "global divergence integral", "regional MFC integral versus direct boundary flux", "covariance identity", "independent source subset recomputation", "training transform isolation and split/source windows", "independent ridge prediction and output metrics", "CERES footprint/time/weight matching"],
        "pdf_inspiration": {"path": r"C:\Users\xinch\OneDrive\Desktop\navier-stokes.pdf", "sha256": sha(r"C:\Users\xinch\OneDrive\Desktop\navier-stokes.pdf"), "pages": [3, 11, 12, 13, 14], "transfer": "Nonlinear flux and complete-residual bookkeeping motivate diagnostics; the incompressible forced-blowup theorem is not an atmospheric causal theorem"},
        "sources": ["https://journals.ametsoc.org/abstract/journals/wefo/20/3/waf858_1.xml", "https://confluence.ecmwf.int/spaces/FUG/pages/673550490/Section%2B2A.1.5.2%2BClouds"],
        "reviewer_mapping": ["R1-I2", "R2-4", "R3-5", "R3-6", "R2-5"],
        "code_sha256": sha(__file__)
    }
    target.write_text(json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"path": str(target), "sha256": sha(target), "frozen": True}))


if __name__ == "__main__":
    main()
