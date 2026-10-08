"""Idealized thermally forced moist tracers, NOT a Navier--Stokes cloud model.

Periodic, constant-density 1-D transport. Prognostic fields are dry-air mixing
ratios rv/rl (kg/kg) and externally thermostatted temperature (K). No latent
heat, vertical motion, momentum equation, or observational calibration.
"""
from __future__ import annotations
import hashlib
import numpy as np


def saturation(T):
    es = 611.2 * np.exp((2.5e6 / 461.0) * (1 / 273.15 - 1 / T))
    return 0.622 * es / (85000.0 - es)


def phase_adjust(rv, rl, T):
    change = np.maximum(rv - saturation(T), 0) - np.minimum(rl, np.maximum(saturation(T) - rv, 0))
    return rv - change, rl + change


def advect(field, velocity, dt, dx):
    c = velocity[:, None] * dt / dx
    return field - np.maximum(c, 0) * (field - np.roll(field, 1, axis=1)) - np.minimum(c, 0) * (np.roll(field, -1, axis=1) - field)


def initial_conditions(n, seed, cells=48):
    rng = np.random.default_rng(seed)
    z0, eta0 = rng.normal(size=(2, n))
    coefficients = rng.normal(size=(n, 9))
    theta = 2 * np.pi * (np.arange(cells) + .5) / cells
    wave = np.stack([np.ones(cells), np.cos(theta), np.sin(theta)], axis=0)
    rv = .010 + .0015 * np.cos(theta) + .0004 * (coefficients[:, :3] @ wave)
    rl = np.maximum(.0003 + .00015 * (coefficients[:, 3:6] @ wave), 0)
    T = 282 + 3 * np.sin(theta + .5) + .5 * (coefficients[:, 6:9] @ wave)
    assert np.min(rv) > 0 and np.min(T) > 250 and np.max(T) < 310
    rv, rl = phase_adjust(rv, rl, T)
    uniform = rng.random(n)
    return dict(rv=rv, rl=rl, T=T, z0=z0, eta0=eta0, uniform=uniform,
                coefficients=coefficients, theta=theta)


def exogenous(n, seed, duration=43200, fine_dt=150):
    """All resolutions sample one exact discrete OU path at common clock times."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(duration // fine_dt, 2, n))


def exogenous_paths(initial, noise):
    z, eta = initial["z0"].copy(), initial["eta0"].copy()
    zs, es = [z.copy()], [eta.copy()]
    rz, re = np.exp(-150 / 21600), np.exp(-150 / 10800)
    for e in noise:
        z = rz * z + np.sqrt(1 - rz * rz) * e[0]
        eta = re * eta + np.sqrt(1 - re * re) * e[1]
        zs.append(z.copy()); es.append(eta.copy())
    return np.asarray(zs), np.asarray(es)


def rollout(initial, paths, scenario, arm, dt=300, retain=8):
    rv, rl, T = [initial[k].copy() for k in ("rv", "rl", "T")]
    n, cells = rv.shape
    theta = initial["theta"]
    dx = 1200000 / cells
    x = (np.arange(cells) + .5) * dx
    window = (x >= 200000) & (x < 400000)
    zpath, epath = paths
    reference_rv = .010 + .0015 * np.cos(theta)
    reference_T = 282 + 3 * np.sin(theta + .5)
    initial_water = (rv + rl).mean(axis=1)
    water_source = np.zeros(n); rain_sink = np.zeros(n)
    maxima = dict(cfl=0.0, water_residual=0.0, advection_water_residual=0.0,
                  phase_water_residual=0.0, min_rv=float(rv.min()), min_rl=float(rl.min()),
                  min_T=float(T.min()), max_T=float(T.max()))
    result = np.empty((n, 2, 2))
    trace = [np.stack([rv[:retain], rl[:retain], T[:retain]], axis=1)]
    winds = []
    null_hash = hashlib.sha256() if scenario["exact_null"] else None
    for step in range(43200 // dt):
        clock = step * dt
        z, eta = zpath[clock // 150], epath[clock // 150]
        feedback = 2 * np.tanh(rl[:, window].mean(axis=1) / .001 - 1) if scenario["feedback"] else 0
        wind = 8 + 2 * np.tanh(z) + 1.5 * np.tanh(eta) + feedback
        if clock < 3600:
            wind = wind + (2 if arm == 1 else -2)
        velocity = wind if scenario["transport"] else np.zeros(n)
        cfl = float(np.max(np.abs(velocity)) * dt / dx)
        assert cfl <= 1
        maxima["cfl"] = max(maxima["cfl"], cfl)
        before_adv = (rv + rl).mean(axis=1)
        rv = advect(rv, velocity, dt, dx)
        rl = advect(rl, velocity, dt, dx)
        T = advect(T, velocity, dt, dx)
        maxima["advection_water_residual"] = max(maxima["advection_water_residual"], float(np.max(np.abs((rv + rl).mean(axis=1) - before_adv))))
        rv_eq = reference_rv + .0012 * np.tanh(z[:, None])
        T_eq = reference_T - 1.5 * np.tanh(z[:, None])
        updated_rv = rv_eq + (rv - rv_eq) * np.exp(-dt / 43200)
        water_source += (updated_rv - rv).mean(axis=1)
        rv = updated_rv
        T = T_eq + (T - T_eq) * np.exp(-dt / 21600)
        water_before = rv + rl
        rv, rl = phase_adjust(rv, rl, T)
        maxima["phase_water_residual"] = max(maxima["phase_water_residual"], float(np.max(np.abs(rv + rl - water_before))))
        rain = rl * (1 - np.exp(-dt / 21600))
        rl -= rain
        rain_sink += rain.mean(axis=1)
        residual = (rv + rl).mean(axis=1) - initial_water - water_source + rain_sink
        maxima["water_residual"] = max(maxima["water_residual"], float(np.max(np.abs(residual))))
        maxima["min_rv"] = min(maxima["min_rv"], float(rv.min()))
        maxima["min_rl"] = min(maxima["min_rl"], float(rl.min()))
        maxima["min_T"] = min(maxima["min_T"], float(T.min()))
        maxima["max_T"] = max(maxima["max_T"], float(T.max()))
        assert np.isfinite(T).all() and rv.min() >= -1e-14 and rl.min() >= -1e-14
        if null_hash is not None:
            for field in (rv, rl, T):
                null_hash.update(field.tobytes())
        if clock + dt in (21600, 43200):
            j = 0 if clock + dt == 21600 else 1
            result[:, j, 0] = 1000 * rv[:, window].mean(axis=1)
            result[:, j, 1] = 100 * (-np.expm1(-rl[:, window] / .001)).mean(axis=1)
        trace.append(np.stack([rv[:retain], rl[:retain], T[:retain]], axis=1))
        winds.append(wind[:retain])
    assert maxima["water_residual"] < 1e-12
    assert np.all((result[:, :, 1] >= 0) & (result[:, :, 1] <= 100))
    if null_hash is not None:
        maxima["all_episode_all_step_state_sha256"] = null_hash.hexdigest()
    return result, maxima, np.asarray(trace), np.asarray(winds)


def make_dataset(design, scenario, replicate):
    counts = design["sample_sizes"]
    n = sum(counts.values())
    seed = design["seed"] + replicate * 1000
    initial = initial_conditions(n, seed, design["cells"])
    paths = exogenous_paths(initial, exogenous(n, seed + 1))
    y0, checks0, trace0, wind0 = rollout(initial, paths, scenario, 0)
    y1, checks1, trace1, wind1 = rollout(initial, paths, scenario, 1)
    if scenario["exact_null"]:
        assert np.array_equal(trace0, trace1) and np.array_equal(y0, y1)
        assert checks0["all_episode_all_step_state_sha256"] == checks1["all_episode_all_step_state_sha256"]
    z0 = initial["z0"]
    propensity = np.full(n, .5) if scenario["randomized"] else np.clip(1 / (1 + np.exp(-1.5 * z0)), .15, .85)
    A = (initial["uniform"] < propensity).astype(np.int8)
    y = np.where(A[:, None, None] == 1, y1, y0)
    rv, rl, T = [initial[k] for k in ("rv", "rl", "T")]
    x = (np.arange(design["cells"]) + .5) * (1200000 / design["cells"])
    window = (x >= 200000) & (x < 400000)
    upstream = (x >= 0) & (x < 200000)
    prewind = 8 + 2 * np.tanh(z0) + 1.5 * np.tanh(initial["eta0"])
    if scenario["feedback"]:
        prewind += 2 * np.tanh(rl[:, window].mean(axis=1) / .001 - 1)
    local = np.column_stack([1000 * rv[:, window].mean(axis=1), 1000 * rl[:, window].mean(axis=1), T[:, window].mean(axis=1), 100 * (-np.expm1(-rl[:, window] / .001)).mean(axis=1), 1000 * rv[:, upstream].mean(axis=1), 1000 * (rv[:, window].mean(axis=1) - rv[:, upstream].mean(axis=1)) / 200, prewind])
    full = np.column_stack([rv * 1000, rl * 1000, T, z0, prewind])
    split = np.repeat(np.arange(3, dtype=np.int8), list(counts.values()))
    data = dict(split=split, A=A, propensity=propensity, X_local=local,
                X_observed=np.column_stack([local, z0]), X_full=full,
                Y=y, Y0=y0, Y1=y1, horizons=np.array([6, 12]),
                outcomes=np.array(["vapor_g_per_kg", "cloud_proxy_pp"]),
                exact_null=np.array(scenario["exact_null"]),
                episode_id=np.arange(n), initial_coefficients=initial["coefficients"],
                initial_z=z0, initial_eta=initial["eta0"])
    trace = dict(state0=trace0, state1=trace1, wind0=wind0, wind1=wind1,
                 z=paths[0][:, :8], eta=paths[1][:, :8])
    checks = dict(arm0=checks0, arm1=checks1, n_episodes=n,
                  assigned_counts=np.bincount(A, minlength=2).tolist(),
                  propensity_range=[float(propensity.min()), float(propensity.max())],
                  exact_null_paired_outcomes_equal=bool(np.array_equal(y0, y1)),
                  seed=seed, exogenous_seed=seed + 1)
    return data, checks, trace
