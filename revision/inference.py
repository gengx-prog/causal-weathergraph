"""Calendar-time OLS and Bartlett HAC for a single added lagged source.

Excluded timestamps have ZERO scores. Lag h always means h rows on the
original equally spaced calendar; regime observations are never concatenated
for covariance estimation. HAC tests use the asymptotic normal distribution,
with the explicitly reported n/(n-rank_full) covariance correction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy import fft, linalg, stats


def _invalid(bandwidths, status, n=0):
    out = dict(valid=False, status=status, effect=np.nan, effect_coefficient=np.nan,
               p_ols=1.0, se_ols=np.nan, n_samples=int(n), rank_full=0)
    for b in bandwidths:
        out.update({f'p_hac{b}': 1.0, f'se_hac{b}': np.nan,
                    f'ci95_low_hac{b}': np.nan, f'ci95_high_hac{b}': np.nan})
    return out


def _basis(base):
    q, r, _ = linalg.qr(base, mode='economic', pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    tol = max(base.shape) * np.finfo(float).eps * (diagonal.max() if len(diagonal) else 0)
    rank = int(np.count_nonzero(diagonal > tol))
    return q[:, :rank], rank


def score_autocovariances(scores, max_lag):
    """Unnormalised score cross-products, using linear (not circular) FFT."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    n = len(scores)
    size = fft.next_fast_len(2 * n - 1)
    spectrum = fft.rfft(scores, n=size, axis=0, workers=1)
    return fft.irfft(spectrum.conj() * spectrum, n=size, axis=0, workers=1)[:min(max_lag + 1, n)]


def _cluster_statistics(scores, mask, zss, beta, rank_full, block_lengths, calendar_offset=0):
    """CR1 with t_(G-1), clusters defined on ORIGINAL calendar positions."""
    result={}
    n=int(mask.sum())
    position=np.arange(len(mask),dtype=int)+int(calendar_offset)
    for length in block_lengths:
        if int(length)<1:
            raise ValueError('Calendar block length must be positive')
        labels=position//int(length)
        unique=np.unique(labels[mask])
        g=len(unique)
        if g<2 or n<=rank_full:
            result[int(length)]=(np.full(len(beta),np.nan),np.ones(len(beta)),g,np.nan)
            continue
        sums=np.zeros((int(labels[-1])+1,scores.shape[1]))
        np.add.at(sums,labels,scores)
        correction=g/(g-1)*(n-1)/(n-rank_full)
        variance=correction*np.sum(sums[unique]**2,axis=0)/(zss*zss)
        se=np.sqrt(np.maximum(variance,0))
        test=np.divide(np.abs(beta),se,out=np.full(len(beta),np.inf),where=se>0)
        result[int(length)]=(se,2*stats.t.sf(test,g-1),g,correction)
    return result


def _finish(y, yr, zr, mask, rank_controls, bandwidths, cluster_lengths=(), calendar_offset=0):
    """Calculate inference for columns of already-residualised sources."""
    n, m = zr.shape
    rank_full = rank_controls + 1
    dof = n - rank_full
    zss = np.sum(zr * zr, axis=0)
    zy = yr @ zr
    good = np.isfinite(zss) & (zss > 1e-12)
    denominator = np.where(good, zss, 1.0)
    beta = zy / denominator
    residual = yr[:, None] - zr * beta
    rss = np.sum(residual * residual, axis=0)
    restricted_rss = float(yr @ yr)
    drop = zy * zy / denominator
    se = np.sqrt(np.maximum(rss / dof / denominator, 0))
    fstat = drop / np.maximum(rss / dof, 1e-300)
    p_ols = stats.f.sf(fstat, 1, dof)
    scores = np.zeros((len(mask), m))
    scores[mask] = zr * residual
    max_bandwidth = max(bandwidths, default=0)
    acov = score_autocovariances(scores, max_bandwidth) if bandwidths else None
    tss = float(np.sum((y - y.mean()) ** 2))
    hac = {}
    correction = n / dof
    for b in bandwidths:
        h = np.arange(1, min(b + 1, len(acov)))
        meat = acov[0] + 2 * np.sum((1 - h[:, None] / (b + 1)) * acov[h], axis=0)
        se_hac = np.sqrt(np.maximum(meat * correction, 0)) / denominator
        t_hac = np.divide(np.abs(beta), se_hac, out=np.full(m, np.inf), where=se_hac > 0)
        hac[b] = (se_hac, 2 * stats.norm.sf(t_hac))
    cluster=_cluster_statistics(scores,mask,denominator,beta,rank_full,cluster_lengths,calendar_offset)
    full_residual = np.zeros((len(mask), m))
    full_residual[mask] = residual
    diagnostics = {}
    for lag in (1, 4, 12, 28):
        pairs = mask[lag:] & mask[:-lag]
        a, c = full_residual[lag:][pairs], full_residual[:-lag][pairs]
        norm = np.sqrt(np.sum(a*a, axis=0) * np.sum(c*c, axis=0))
        val = np.divide(np.sum(a*c, axis=0), norm, out=np.full(m, np.nan), where=norm > 0)
        diagnostics[lag] = (val, int(pairs.sum()))
    results = []
    for j in range(m):
        if not good[j]:
            results.append(_invalid(bandwidths, 'source_collinear_with_controls', n))
            continue
        if tss < 1e-20 or rss[j] < 1e-25 * max(1., tss):
            results.append(_invalid(bandwidths, 'constant_target_or_exact_fit', n))
            continue
        out = dict(valid=True, status='ok', effect=float(beta[j]),
                   effect_coefficient=float(beta[j]), effect_abs=float(abs(beta[j])),
                   p_ols=float(p_ols[j]), se_ols=float(se[j]), F_statistic=float(fstat[j]),
                   ci95_low_ols=float(beta[j]-stats.t.ppf(.975, dof)*se[j]),
                   ci95_high_ols=float(beta[j]+stats.t.ppf(.975, dof)*se[j]),
                   n_samples=n, rank_full=rank_full, df_residual=dof,
                   hac_correction=float(correction), partial_r2=float(drop[j]/restricted_rss),
                   delta_r2=float(drop[j]/tss), r2_full=float(1-rss[j]/tss),
                   r2_restricted=float(1-restricted_rss/tss))
        for b, (s, p) in hac.items():
            out.update({f'p_hac{b}': float(p[j]), f'se_hac{b}': float(s[j]),
                        f'ci95_low_hac{b}': float(beta[j]-stats.norm.ppf(.975)*s[j]),
                        f'ci95_high_hac{b}': float(beta[j]+stats.norm.ppf(.975)*s[j])})
        for b,(s,p,g,correction) in cluster.items():
            critical=stats.t.ppf(.975,g-1) if g>1 else np.nan
            out.update({f'p_cluster{b}':float(p[j]),f'se_cluster{b}':float(s[j]),
                        f'ci95_low_cluster{b}':float(beta[j]-critical*s[j]),
                        f'ci95_high_cluster{b}':float(beta[j]+critical*s[j]),
                        f'n_clusters{b}':g,f'cluster_correction{b}':float(correction)})
        for lag, (v, count) in diagnostics.items():
            out[f'residual_acf_{lag}'] = float(v[j])
            out[f'residual_pairs_{lag}'] = count
        results.append(out)
    return results


def fit_edge(y, X_controls, source, mask=None, bandwidths=(32, 64, 128),
             cluster_lengths=(), calendar_offset=0):
    """Fit one source coefficient on a full, equally spaced calendar timeline.

    Inputs are aligned arrays; controls must EXCLUDE the intercept. Missing
    observations and false mask rows are excluded from OLS, but retain their
    calendar positions when constructing HAC score products. Singular controls
    use their numerical rank. Unidentifiable source effects return p=1 and
    valid=False so the attempted hypothesis need not disappear from a family.
    """
    y = np.asarray(y, dtype=float)
    c = np.asarray(X_controls, dtype=float)
    z = np.asarray(source, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    if y.ndim != 1 or z.shape != y.shape or c.ndim != 2 or len(c) != len(y):
        raise ValueError('y, source and X_controls must share the full calendar row axis')
    bandwidths = tuple(int(b) for b in bandwidths)
    if any(b < 0 for b in bandwidths):
        raise ValueError('HAC bandwidth must be nonnegative')
    selected = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if selected.shape != y.shape:
        raise ValueError('mask must have full calendar length')
    selected &= np.isfinite(y) & np.isfinite(z) & np.isfinite(c).all(axis=1)
    n = int(selected.sum())
    if n < c.shape[1] + 3:
        return _invalid(bandwidths, 'insufficient_samples', n)
    base = np.column_stack((np.ones(n), c[selected]))
    q, rank = _basis(base)
    if n <= rank + 1:
        return _invalid(bandwidths, 'insufficient_degrees_of_freedom', n)
    yy, zz = y[selected], z[selected]
    yr = yy - q @ (q.T @ yy)
    zr = zz - q @ (q.T @ zz)
    return _finish(yy, yr, zr[:, None], selected, rank, bandwidths,cluster_lengths,calendar_offset)[0]


def fit_edge_block_cluster(y, X_controls, source, mask=None,
                          block_lengths=(64,128,256), calendar_offset=0):
    """Exploratory calendar-block CR1 covariance with Student t_(G-1).

    Added after finite-sample HAC calibration failed; not a replacement for the
    frozen primary analysis. Populated original-calendar blocks define G.
    Cross-block independence is an approximation, not established by this
    estimator. The effect model and OLS comparator are unchanged.
    """
    result=fit_edge(y,X_controls,source,mask,bandwidths=(),
                    cluster_lengths=block_lengths,calendar_offset=calendar_offset)
    for b in block_lengths:
        result.setdefault(f'p_cluster{b}',1.)
        result.setdefault(f'se_cluster{b}',np.nan)
        result.setdefault(f'ci95_low_cluster{b}',np.nan)
        result.setdefault(f'ci95_high_cluster{b}',np.nan)
        result.setdefault(f'n_clusters{b}',0)
    return result


def adjust_bh(pvalues):
    """BH adjusted p-values; invalid hypotheses are retained with p=1."""
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1)
    return result


def add_multiplicity(table, bandwidths=(32, 64, 128), alpha=.05, cluster_lengths=()):
    table = table.copy()
    for method in ('ols',) + tuple(f'hac{b}' for b in bandwidths) + tuple(f'cluster{b}' for b in cluster_lengths):
        col = f'p_{method}'
        table[f'q_{method}_global'] = adjust_bh(table[col])
        table[f'q_{method}_within'] = table.groupby(['regime', 'target_var'], sort=False)[col].transform(adjust_bh)
        for scope in ('within', 'global'):
            table[f'significant_{method}_{scope}'] = table[f'q_{method}_{scope}'] < alpha
    return table


def run_graph_discovery(data, variable_names, candidates, regimes, max_lag=3,
                        controls=None, bandwidths=(32,64,128), progress: Callable | None=None,
                        cluster_lengths=()):
    """All candidates x lags x regimes; FWL batches share target controls."""
    controls = controls or {'include_target_own_lags': True, 'include_target_region_all_vars': False}
    data = np.asarray(data)
    n_time, _, n_var = data.shape
    var_index = {name: i for i, name in enumerate(variable_names)}
    groups = defaultdict(list)
    for edge in candidates:
        groups[(edge.target_region, edge.target_var)].append(edge)
    records = []
    total = len(regimes) * len(groups)
    done = 0
    for regime_name, regime_mask in regimes.items():
        for (region, target), edges in groups.items():
            ti = var_index[target]
            cv = (list(range(n_var)) if controls.get('include_target_region_all_vars', False)
                  else [ti] if controls.get('include_target_own_lags', True) else [])
            keys = [(v, lag) for v in cv for lag in range(1, max_lag+1)]
            c = np.column_stack([data[max_lag-lag:n_time-lag,region,v] for v,lag in keys]) if keys else np.empty((n_time-max_lag,0))
            y = np.asarray(data[max_lag:,region,ti], dtype=float)
            mask = np.asarray(regime_mask, dtype=bool)[max_lag:].copy()
            mask &= np.isfinite(y) & np.isfinite(c).all(axis=1)
            descriptors = [(edge,lag) for edge in edges for lag in range(1,max_lag+1)]
            source = np.column_stack([data[max_lag-lag:n_time-lag,e.source_region,var_index[e.source_var]] for e,lag in descriptors])
            fits = [None] * len(descriptors)
            regular = [j for j,(e,lag) in enumerate(descriptors)
                       if not(e.source_region == region and (var_index[e.source_var],lag) in keys)
                       and np.isfinite(source[mask,j]).all()]
            if regular and mask.sum() > c.shape[1]+3:
                base = np.column_stack((np.ones(mask.sum()),c[mask]))
                q, rank = _basis(base)
                yy = y[mask]
                yr = yy-q@(q.T@yy)
                zz = source[mask][:,regular]
                zr = zz-q@(q.T@zz)
                batch = _finish(yy,yr,zr,mask,rank,bandwidths,cluster_lengths,max_lag)
                for j,res in zip(regular,batch):
                    fits[j] = res
            for j,(edge,lag) in enumerate(descriptors):
                if fits[j] is None:
                    drop = (var_index[edge.source_var],lag) if edge.source_region == region else None
                    keep = [k for k,key in enumerate(keys) if key != drop]
                    fits[j] = fit_edge(y,c[:,keep],source[:,j],mask,bandwidths,cluster_lengths,max_lag)
                for b in cluster_lengths:
                    fits[j].setdefault(f'p_cluster{b}',1.)
                fits[j].update(edge.to_dict())
                fits[j].update(regime=str(regime_name),lag=int(lag),n_controls=len(keys)-(1 if edge.source_region == region and (var_index[edge.source_var],lag) in keys else 0))
                records.append(fits[j])
            done += 1
            if progress:
                progress(done,total,str(regime_name),int(region))
    table = pd.DataFrame(records)
    return add_multiplicity(table,bandwidths,cluster_lengths=cluster_lengths) if len(table) else table


def sample_calendar_design_blocks(n_rows, block_length, rng):
    """Moving blocks of ORIGINAL pre-lagged design rows, never rebuilt lags.

    Caller constructs every lag before sampling and applies the sampled regime
    mask afterwards. This function is suitable for coefficient stability, not
    automatic validation of an FDR guarantee. Blocks never wrap calendar end.
    """
    if n_rows <= 0:
        return np.empty(0,dtype=int)
    length = min(n_rows,max(1,int(block_length)))
    starts = rng.integers(0,n_rows-length+1,size=int(np.ceil(n_rows/length)))
    return (starts[:,None]+np.arange(length)).ravel()[:n_rows]


def bootstrap_edge_effects(y, X_controls, source, mask=None, block_length=128,
                           n_bootstrap=200, seed=20260929):
    """Resample calendar blocks of an ALREADY LAGGED design, then apply mask.

    Returns coefficient replicates and a pointwise percentile interval. Neither
    selected-edge coverage nor FDR calibration is implied. Regime thresholds
    are held fixed; this is conditional stability, not selection correction.
    HAC is deliberately not computed on the stitched bootstrap timeline.
    """
    y=np.asarray(y,dtype=float)
    c=np.asarray(X_controls,dtype=float)
    z=np.asarray(source,dtype=float)
    if c.ndim==1:
        c=c[:,None]
    if y.ndim!=1 or z.shape!=y.shape or c.ndim!=2 or len(c)!=len(y):
        raise ValueError('Inputs must share the original aligned calendar rows')
    selected=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask,dtype=bool).copy()
    if selected.shape!=y.shape:
        raise ValueError('mask must have full calendar length')
    selected &= np.isfinite(y)&np.isfinite(z)&np.isfinite(c).all(axis=1)
    design=np.column_stack((np.ones(len(y)),c,z))
    rng=np.random.default_rng(seed)
    effects=np.full(int(n_bootstrap),np.nan)
    sample_sizes=np.zeros(int(n_bootstrap),dtype=int)
    for b in range(int(n_bootstrap)):
        rows=sample_calendar_design_blocks(len(y),block_length,rng)
        rows=rows[selected[rows]]
        sample_sizes[b]=len(rows)
        if len(rows)<=design.shape[1]:
            continue
        coef,_,rank,_=np.linalg.lstsq(design[rows],y[rows],rcond=None)
        if rank==design.shape[1]:
            effects[b]=coef[-1]
    valid=effects[np.isfinite(effects)]
    bounds=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
    return dict(effects=effects,sample_sizes=sample_sizes,ci95_low=float(bounds[0]),
                ci95_high=float(bounds[1]),n_valid=int(len(valid)),block_length=int(block_length),
                n_bootstrap=int(n_bootstrap),seed=int(seed),
                method='moving original-calendar blocks of fixed lagged design; percentile interval')
