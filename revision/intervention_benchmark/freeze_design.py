"""Write a new frozen design before production outputs are inspected."""
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "revision_outputs/intervention_benchmark_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    design = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_production_simulation_and_estimation",
        "seed": 20261002,
        "replicates": 20,
        "sample_sizes": {"train": 2048, "validation": 512, "test": 512},
        "cells": 48, "domain_m": 1200000, "dt_seconds": 300,
        "fine_exogenous_dt_seconds": 150,
        "horizon_hours": [6, 12], "window_m": [200000, 400000],
        "intervention": "A=1 adds +2 m/s and A=0 adds -2 m/s uniformly during first hour; compare these policies with identical initial states and future exogenous innovations. Later feedback is allowed to evolve.",
        "propensity": "randomized:0.5; otherwise clip(expit(1.5*Z0),0.15,0.85). Assignment uses only pretreatment Z0; observing Z0 suffices for exchangeability in this generator.",
        "scenarios": [
            {"name": "randomized_transport", "randomized": True, "transport": True, "feedback": False, "exact_null": False},
            {"name": "confounded_transport", "randomized": False, "transport": True, "feedback": False, "exact_null": False},
            {"name": "confounded_feedback", "randomized": False, "transport": True, "feedback": True, "exact_null": False},
            {"name": "confounded_null", "randomized": False, "transport": False, "feedback": False, "exact_null": True}
        ],
        "equations": {
            "wind": "8+2*tanh(Z)+1.5*tanh(eta)+feedback*2*tanh(mean_window(rl)/0.001-1)+first_hour*(4*A-2) [m/s]",
            "Z_eta": "independent stationary OU sampled at150s; correlation times21600/10800s, unit stationary variance; common exogenous paths across paired worlds and scenarios",
            "transport": "periodic first-order conservative upwind for rv,rl,T; entire velocity setzero for allthree prognostic variables in exactnull",
            "rv_source": "exact exponential relaxation toward0.010+0.0015*cos(theta)+0.0012*tanh(Z), tau43200s",
            "T_forcing": "exact relaxation toward282+3*sin(theta+0.5)-1.5*tanh(Z), tau21600s",
            "saturation": "rs=0.622*es/(85000-es), es=611.2*exp((2500000/461)*(1/273.15-1/T)); dry-air mixing ratioskg/kg",
            "phase": "D=max(rv-rs,0)-min(rl,max(rs-rv,0)); rv-=D; rl+=D atfixedT",
            "rain": "rl*=exp(-dt/21600); removedwaterrecorded",
            "cloud_proxy": "100*mean_window(1-exp(-rl/0.001)); synthetic dimensionless cloud diagnostic, not ERA5 total cloud fraction",
            "initial_state": "fixed smooth geographical profiles+9 independent N(0,1) coefficients; q/l saturationadjusted atbaseline; initialZ/etaindependent; exactimplementation archived"
        },
        "estimators": ["unadjusted_ols", "local_ols", "observed_z_ols", "local_hgb_gcomp", "observed_z_hgb_gcomp", "full_hgb_gcomp", "oracle_ipw", "oracle_aipw"],
        "hgb": {"max_iter": 120, "max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": .05, "l2_regularization": 1, "max_bins": 64, "early_stopping": False, "random_state": 20261002},
        "fitting": "One assigned factual arm per training episode only; paired truths accessed only on heldouttest. Validation separate, fixed hyperparameters not tuned. No post-treatment controls. Reuses fit_edge coefficient as explicitly adapted endpoint-effect estimator, not original graph score.",
        "evaluation": "ATE error against paired heldouttest average; simulator truth MCSE separately; repeated-dataset meanbias,ATE RMSE and MC intervals across20 independent replicates. FactualpredictionMSE separately. Gcomp no perfitCI. HC3 OLS and oracle IPW/AIPW normal intervals descriptive; exactnull rejection counts with Wilson intervals, not claimed calibrated at20replicates.",
        "numerics": "CFL<=1, positivity, exact tracer waterledger; saved first8 episode complete paired trajectories. Refine dt300to150 and grid48to96 for128 independentstates with matched continuous profiles andOUnoise; retainallchanges regardless of sign. Exactnull requires no wind entrance in any prognostic update.",
        "scope_limits": ["Idealized prescribed-velocity moist tracers, not a Navier-Stokes solver, Hydro-ABC implementation, or atmospheric model validation", "No latent heat, energy closure, pressure/momentum equation, vertical motion or resolved cloud microphysics", "Policy total effect includes feedback and direct temperature/condensate advection; no identified humidity-mediated natural effect", "Not a validation of existing ERA5 graph or causal discovery/FDR; fullgrid vs local changes information, not an equal-information ranking"],
        "references": ["https://arxiv.org/abs/1105.0470", "https://gmd.copernicus.org/articles/16/6067/2023/", "https://www.nature.com/articles/s41467-019-10105-3", "https://imai.fas.harvard.edu/research/files/mediation.pdf"]
    }
    with (OUT / "design.json").open("x", encoding="utf8") as f:
        json.dump(design, f, ensure_ascii=False, indent=2)
    print(str(OUT / "design.json"))


if __name__ == "__main__":
    main()
