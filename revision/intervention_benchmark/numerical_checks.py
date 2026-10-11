"""Analytic numerical checks and frozen effect-resolution sensitivity."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from revision.intervention_benchmark.simulation import advect, phase_adjust, initial_conditions, exogenous, exogenous_paths, rollout


def main():
    out = ROOT.parent / "revision_outputs/intervention_benchmark_v1"
    design = json.loads((out / "design.json").read_text(encoding="utf8"))
    rng = np.random.default_rng(7124)
    f = rng.uniform(.001, .02, size=(20, 48)); velocity = np.linspace(-15, 15, 20)
    fa = advect(f, velocity, 300, 25000)
    mass_error = float(np.max(np.abs(fa.sum(axis=1)-f.sum(axis=1))))
    assert mass_error < 1e-12 and fa.min() >= 0
    constant_error = float(np.max(np.abs(advect(np.ones((20, 48)), velocity, 300, 25000)-1)))
    assert constant_error == 0
    rv, rl = rng.uniform(.001, .025, (2, 20, 48)); T = rng.uniform(275, 290, (20, 48))
    a, b = phase_adjust(rv, rl, T)
    phase_error = float(np.max(np.abs((a+b)-(rv+rl))))
    assert phase_error < 1e-14 and a.min() >= 0 and b.min() >= 0
    analytic = []
    for cells in [48, 96]:
        dx = 1200000 / cells
        x = (np.arange(cells)+.5)*dx
        f = 1 + .2*np.sin(2*np.pi*x/1200000)
        for k in range(72):
            f = advect(f[None], np.array([8.0]), 150, dx)[0]
        exact = 1+.2*np.sin(2*np.pi*(x-8*10800)/1200000)
        analytic.append(float(np.sqrt(np.mean((f-exact)**2))))
    assert analytic[1] < analytic[0]
    rows = []
    n = 128; seed = design["seed"] + 99999
    noise = exogenous(n, seed+1)
    for scenario in design["scenarios"]:
        effects = {}
        for label, cells, dt in [("base", 48, 300), ("half_dt", 48, 150), ("refined_grid", 96, 150)]:
            init = initial_conditions(n, seed, cells)
            paths = exogenous_paths(init, noise)
            y0, c0, _, _ = rollout(init, paths, scenario, 0, dt, retain=0)
            y1, c1, _, _ = rollout(init, paths, scenario, 1, dt, retain=0)
            effects[label] = y1-y0
            if scenario["exact_null"]:
                assert np.array_equal(y0,y1)
            for j,h in enumerate([6,12]):
                for k,target in enumerate(["vapor_g_per_kg", "cloud_proxy_pp"]):
                    diff = effects[label][:,j,k]-effects["base"][:,j,k]
                    rows.append(dict(scenario=scenario["name"], resolution=label, cells=cells, dt_seconds=dt, horizon=h, outcome=target, n=n, mean_effect=float(effects[label][:,j,k].mean()), mean_effect_change=float(diff.mean()), paired_effect_change_rmse=float(np.sqrt(np.mean(diff*diff))), water_residual=max(c0["water_residual"],c1["water_residual"])))
        np.savez_compressed(out / f"resolution_effects_{scenario['name']}.npz", **effects)
    pd.DataFrame(rows).to_csv(out / "resolution_sensitivity.csv", index=False)
    (out / "numerical_checks.json").write_text(json.dumps(dict(status="PASS_NUMERICAL_INVARIANTS_SENSITIVITY_REPORTED", mass_error=mass_error, constant_error=constant_error, phase_error=phase_error, analytic_advection_rmse_48_96=analytic, resolution_sensitivity="All fixed cases retained; changes are diagnostics, not a pass threshold or continuum proof"), indent=2), encoding="utf8")
    print(json.dumps(dict(status="PASS", analytic_rmse=analytic, resolution_rows=len(rows))))


if __name__ == "__main__":
    main()
