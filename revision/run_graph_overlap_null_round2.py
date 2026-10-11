"""R2-2 exact constrained support and longer symmetric graph-reference chains.

No refitting, no raw data download, no claim to calibrate edge probabilities.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
from pathlib import Path
from collections import Counter
from itertools import combinations, permutations, product
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from revision.graph_nulls import EDGE_COLUMNS, distance_matrix, distance_bin, graph_metrics
OLD = ROOT.parent / 'revision_outputs/graph_nulls'
OUT = ROOT.parent / 'revision_outputs/graph_overlap_null_round2'


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def fingerprint(edges):
    return hashlib.sha256(repr(sorted(edges)).encode()).hexdigest()


def signature(edges, bins):
    return (Counter((a, b, i) for i, j, a, b in edges),
            Counter((a, b, j) for i, j, a, b in edges),
            Counter((a, b, int(bins[i, j])) for i, j, a, b in edges))


def serialized_signature(edges, bins):
    return {name: [list(k) + [int(v)] for k, v in sorted(counter.items())]
            for name, counter in zip(('typed_outdegree', 'typed_indegree', 'typed_distance_bin_count'), signature(edges, bins))}


def selected(path):
    frame = pd.read_csv(path)
    if set(frame.regime) != {'all'}: raise ValueError('Expected all-regime table')
    return set(frame.loc[frame.q_hac64_global < .05, EDGE_COLUMNS].drop_duplicates().itertuples(index=False, name=None))


def exact_support(edges, support, bins, n):
    """All differences between feasible fixed-degree graphs are alternating cycles.

    Thus an edge can change only if its endpoints are in the same SCC of the
    alternating residual digraph. Enumerate ALL such cells within each type,
    then check joint degree and distance margins; other cells are fixed.
    """
    possible_by_type, diagnostics = [], []
    for a, b in sorted({e[2:] for e in support}):
        pool = {e for e in support if e[2:] == (a, b)}
        original = edges & pool
        adj = np.zeros((2*n, 2*n), dtype=bool)
        for i, j, _, _ in pool:
            if (i,j,a,b) in original: adj[n+j,i] = True
            else: adj[i,n+j] = True
        _, labels = connected_components(csr_matrix(adj), directed=True, connection='strong')
        movable = {e for e in pool if labels[e[0]] == labels[n+e[1]]}
        fixed = original - movable
        choose = len(original & movable)
        combination_count = math.comb(len(movable), choose)
        if combination_count > 1000000: raise ValueError('Exact enumeration exceeds frozen budget')
        target = signature(original, bins)
        legal, degree_count = [], 0
        for combo in combinations(sorted(movable), choose):
            candidate = fixed | set(combo)
            sig = signature(candidate, bins)
            if sig[:2] != target[:2]: continue
            degree_count += 1
            if sig == target: legal.append(candidate)
        if not legal: raise AssertionError('Original graph absent from exact space')
        diagnostics.append({'source_var':a, 'target_var':b,
             'support_edges':len(pool),'selected_edges':len(original),
             'scc_labels_source_nodes':labels[:n].tolist(), 'scc_labels_target_nodes':labels[n:].tolist(),
             'movable_support_edges':[list(e) for e in sorted(movable)],
             'fixed_selected_edges':[list(e) for e in sorted(fixed)],
             'combinations_exhausted':combination_count, 'degree_feasible_count':degree_count,
             'degree_and_distance_feasible_count':len(legal),
             'feasible_type_graphs':[[list(e) for e in sorted(g)] for g in legal]})
        possible_by_type.append(legal)
    total = math.prod(map(len, possible_by_type))
    if total > 100000: raise ValueError('Exact Cartesian graph budget exceeded')
    graphs = [set().union(*parts) for parts in product(*possible_by_type)]
    if not any(g == edges for g in graphs): raise AssertionError('Observed graph missing')
    return graphs, diagnostics


def move(ordered, current, indices, shift, bins, support=None):
    """Permutation of destinations; inverse uses same fixed indices and -shift."""
    old = [ordered[k] for k in indices]
    if len(set(indices)) < len(indices): return 'repeated_index', False
    oldset = set(old)
    new = [(e[0], old[(j+shift) % len(old)][1], e[2], e[3]) for j,e in enumerate(old)]
    newset = set(new)
    if len(newset) != len(new): return 'duplicate_new', False
    if any(e in current and e not in oldset for e in new): return 'duplicate_existing', False
    if support is not None and not newset.issubset(support): return 'outside_support', False
    if sorted(int(bins[e[0],e[1]]) for e in old) != sorted(int(bins[e[0],e[1]]) for e in new):
        return 'distance_margin', False
    changed = oldset != newset
    current.difference_update(oldset); current.update(newset)
    for k,e in zip(indices,new): ordered[k] = e
    return ('accepted_graph_change' if changed else 'accepted_label_only'), changed


def small_graph_checks():
    rng = np.random.default_rng(9212401)
    # Exhaustive all-support enumeration is independent of the SCC shortcut.
    for case in range(24):
        n=3; bins=rng.integers(0,3,size=(n,n)); pool={(i,j,'wind','humidity') for i in range(n) for j in range(n)}
        support=set(sorted(pool)[k] for k in np.flatnonzero(rng.random(9) < .85))
        m=min(max(1,len(support)//2),len(support)); selected_indices=rng.choice(len(support),m,replace=False)
        edges={sorted(support)[k] for k in selected_indices}; target=signature(edges,bins)
        brute={fingerprint(set(c)) for c in combinations(sorted(support),m) if signature(set(c),bins)==target}
        derived,_=exact_support(edges,support,bins,n)
        if {fingerprint(g) for g in derived} != brute: raise AssertionError('SCC exhaustive agreement')
    # Full 24-state labelled matching space: explicitly form transition matrix
    # for global pair, fixed source-near pair, and both directed 3-cycle kernels.
    n=4; states=list(permutations(range(n))); bins=np.zeros((n,n),int)
    pairs=list(combinations(range(n),2)); local=[p for p in pairs if abs(p[0]-p[1])<=1]
    triples=list(product(range(n),repeat=3)); P=np.zeros((len(states),len(states))); lookup={p:i for i,p in enumerate(states)}
    for row,targets in enumerate(states):
        for kernel,pool,weight in [('pair',pairs,.4),('local',local,.4),('triple',triples,.2)]:
            for ids in pool:
                signs=[1] if len(ids)==2 else [-1,1]
                for shift in signs:
                    order=[(i,targets[i],'wind','humidity') for i in range(n)]; current=set(order)
                    move(order,current,ids,shift,bins)
                    dest=tuple(e[1] for e in order)
                    P[row,lookup[dest]]+=weight/len(pool)/len(signs)
    np.testing.assert_allclose(P.sum(1),1,atol=1e-14)
    np.testing.assert_allclose(P,P.T,atol=1e-14)
    np.testing.assert_allclose(np.ones(len(states))/len(states)@P,np.ones(len(states))/len(states),atol=1e-14)
    return {'scc_vs_independent_bruteforce_cases':24,'explicit_kernel_states':24,
            'max_transition_asymmetry':float(np.max(np.abs(P-P.T))),
            'row_sum_max_error':float(np.max(np.abs(P.sum(1)-1))),
            'stationary_uniform_checked':True,
            'scope':'Small exact examples validate implementation, not connectivity of the empirical distance-constrained state space.'}


def chain(edges, reference, bins, distances, seed, chain_id, ensemble, config):
    rng=np.random.default_rng(seed); ordered=sorted(edges); current=set(ordered); n=len(distances)
    groups=[np.array([i for i,e in enumerate(ordered) if e[2:]==typ],dtype=np.int32) for typ in sorted({e[2:] for e in edges})]
    pairs=np.array([pair for group in groups for pair in combinations(group.tolist(),2)],dtype=np.int32)
    source=np.array([e[0] for e in ordered]); local=pairs[(source[pairs[:,0]]!=source[pairs[:,1]]) & (distances[source[pairs[:,0]],source[pairs[:,1]]] <= config['local_source_radius_km'])]
    if not len(local): raise ValueError('Local kernel pool unexpectedly empty')
    group_weights=np.array([len(g)**3 for g in groups],float);group_weights/=group_weights.sum()
    diag=Counter(); kernel_diag=Counter(); target=signature(edges,bins); rows=[]; snapshots=[]
    expected_chain_total=graph_metrics(edges,n,reference)['whc_unique_edge_chains']

    def advance(attempts):
        for start in range(0,attempts,100000):
            size=min(100000,attempts-start)
            kernels=rng.choice(3,size=size,p=[.4,.4,.2])
            global_ids=rng.integers(len(pairs),size=size);local_ids=rng.integers(len(local),size=size)
            group_ids=rng.choice(len(groups),size=size,p=group_weights)
            random_triple=rng.random((size,3)); signs=rng.choice([-1,1],size=size)
            for k,gi,li,tg,ur,sign in zip(kernels,global_ids,local_ids,group_ids,random_triple,signs):
                if k==0: ids=pairs[gi];shift=1;label='global_2'
                elif k==1: ids=local[li];shift=1;label='fixed_local_2'
                else:
                    group=groups[tg];ids=group[np.floor(ur*len(group)).astype(int)];shift=int(sign);label='global_3'
                outcome,_=move(ordered,current,ids,shift,bins)
                diag[outcome]+=1;kernel_diag[label+'__'+outcome]+=1

    burn=config['burnin_attempts_per_edge']*len(edges); gap=config['gap_attempts_per_edge']*len(edges)
    advance(burn);burn_counts=dict(diag)
    for draw in range(config['draws_per_chain']):
        before=diag['accepted_graph_change'];advance(gap)
        if signature(current,bins)!=target:raise AssertionError('Degree or distance margin changed')
        metrics=graph_metrics(current,n,reference)
        if metrics['whc_unique_edge_chains']!=expected_chain_total: raise AssertionError('WHC degree invariant')
        rows.append({'ensemble':ensemble,'chain':chain_id,'draw':draw,'seed':seed,
             'productive_moves_since_previous':diag['accepted_graph_change']-before,
             'changed_fraction_from_observed':1-len(current & edges)/len(edges),
             'graph_sha256':fingerprint(current),**metrics})
        snapshots.append([e[1] for e in ordered])
    attempts=burn+gap*config['draws_per_chain']
    return rows,np.asarray(snapshots,dtype=np.int16),{
         'ensemble':ensemble,'chain':chain_id,'seed':seed,'attempts':attempts,
         'productive_acceptance_fraction':diag['accepted_graph_change']/attempts,
         'accepted_graph_changes':diag['accepted_graph_change'],
         'label_only_moves':diag['accepted_label_only'],'outcome_counts':dict(diag),
         'kernel_outcome_counts':dict(kernel_diag),'burnin_counts':burn_counts,
         'global_pair_pool':len(pairs),'fixed_local_pair_pool':len(local),
         'all_sampled_margin_checks_passed':True}


def diagnostics(frame, observed):
    answer={}
    metrics=[k for k in observed if k!='selected_unique_edges']+['changed_fraction_from_observed']
    for key in metrics:
        values=np.stack([part.sort_values('draw')[key].to_numpy(float) for _,part in frame.groupby('chain')])
        m,n=values.shape;chain_stats=[]
        for chain_id,v in enumerate(values):
            centered=v-v.mean();var=float(centered@centered)
            if var==0: ac1=None;ess=None
            else:
                rho=np.array([1.]+[float(centered[k:]@centered[:-k])/var for k in range(1,n)])
                ac1=float(rho[1]);positive=[]
                for k in range(0,len(rho)-1,2):
                    pair=rho[k]+rho[k+1]
                    if pair<=0:break
                    positive.append(min(pair,positive[-1]) if positive else pair)
                ess=float(min(n,n/max(1,-1+2*sum(positive))))
            chain_stats.append({'chain':chain_id,'mean':float(v.mean()),'sd':float(v.std(ddof=1)),
                 'lag1_autocorrelation':ac1,'initial_monotone_positive_pair_ess_approx':ess})
        halves=np.concatenate([values[:,:n//2],values[:,n//2:]],axis=0)
        nn=halves.shape[1];W=float(np.mean(np.var(halves,axis=1,ddof=1)));B=float(nn*np.var(halves.mean(1),ddof=1))
        rhat=None if W==0 else float(np.sqrt(((nn-1)/nn*W+B/nn)/W))
        obs=observed.get(key)
        answer[key]={'observed':obs,'mean':float(values.mean()),'q025':float(np.quantile(values,.025)),
             'q975':float(np.quantile(values,.975)),'constant_all_chains':bool(np.ptp(values)==0),
             'split_rhat_basic':rhat,'rhat_not_identified_when_constant':W==0,
             'descriptive_upper_tail_fraction':None if obs is None else float(np.mean(values>=obs)),
             'chain_diagnostics':chain_stats}
    return answer


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen_plan.json').exists():raise ValueError('Frozen output exists; do not overwrite an executed plan')
    source_paths={
       'full':OLD/'discovery_symmetric_core_own.csv.gz',
       'early':OLD/'discovery_symmetric_1979_2001_own.csv.gz',
       'late':OLD/'discovery_symmetric_2002_2025_own.csv.gz',
       'support':OLD/'candidates_symmetric_core.csv',
       'coordinates':ROOT.parent/'revision_outputs/inputs/region_trainfit.npz'}
    with np.load(source_paths['coordinates'],allow_pickle=False) as z:
        distances=distance_matrix(z['lat'],z['lon'])
    bins=distance_bin(distances);n=len(distances)
    support=set(pd.read_csv(source_paths['support'])[EDGE_COLUMNS].itertuples(index=False,name=None))
    graphs={name:selected(source_paths[name]) for name in ('full','early','late')}
    cases={'symmetric_full_topology':(graphs['full'],graphs['full']),
           'symmetric_late_vs_early_overlap':(graphs['late'],graphs['early'])}
    small=small_graph_checks()
    config={'chains':4,'draws_per_chain':200,'burnin_attempts_per_edge':200,
       'gap_attempts_per_edge':20,'base_seed':2026093024,'local_source_radius_km':4000.,
       'proposal_mixture':{'global_2':.4,'fixed_local_2':.4,'global_3':.2}}
    plan={'frozen_utc':now(),'analysis':'Post-diagnostic R2-2 constrained graph-reference audit',
       'inputs':{name:{'path':str(path),'sha256':sha(path)} for name,path in source_paths.items()},
       'code_sha256':{str(Path(__file__)):sha(__file__),str(ROOT/'revision/graph_nulls.py'):sha(ROOT/'revision/graph_nulls.py')},
       'configuration':config,'families':{'candidate_support':'symmetric_core 1272 unique variable-region edges; six cross-variable orientations with identical undirected spatial support plus local links',
       'threshold':'q_hac64_global < 0.05 across 1272 candidates x 3 lags = 3816 hypotheses, separately in each all-regime period. Unique graphs deduplicate over lag.',
       'full_period':'1979-2025','early_period':'1979-2001','late_period':'2002-2025',
       'preprocessing':'Inherited original full-period-standardized graph tables; this is retrospective topology/overlap, not training-frozen validation.'},
       'counts':{name:len(g) for name,g in graphs.items()},
       'original_support_exact':'All feasible typed binary graphs with fixed in/out degrees and type-by-distance-bin histograms are enumerated through alternating residual SCCs. Outside-SCC cells are fixed because any feasible difference decomposes into alternating cycles. All within-SCC combinations are checked; this is not sampling.',
       'expanded_support':'Every region pair is allowed within each of the six fixed variable types, subject to exact typed degree and distance-bin margins. This changes candidate support and must not be described as original-candidate stability.',
       'stationary_symmetry_argument':'Edge labels have fixed source and variable type. Global unordered pairs and source-distance-near unordered pairs are fixed before the chain; target assignment never alters their probabilities. Triple type probabilities are proportional to fixed m^3, then three labels are drawn with replacement; repeats yield self transitions. Left/right cycles have equal probability. The inverse uses the same labels with reversed rotation. Feasibility rejection is symmetric. Label-only swaps are retained; the number of labelings of each simple graph is the constant product of typed outdegree factorials. Hence the kernel has a uniform stationary measure on labelled feasible assignments and projects uniformly if the full feasible space is explored. Irreducibility under distance constraints and finite-budget mixing are NOT established.',
       'fixed_attempt_sampling':'Rejected proposals and label-only moves consume attempts; never sample only productive acceptances. Four independent seeds start at observed graph, so cross-chain agreement alone does not establish global exploration.',
       'invariants':'WHC chain total = sum_h indegree_WH(h) * outdegree_HC(h); fixed exactly. Use unique endpoint count, closed chains, and overlap as potentially variable metrics.',
       'diagnostics':'Graph hashes, productive and label-only moves, changed-edge fraction, each chain metric trace, lag1 autocorrelation, initial-positive-pair approximate ESS and basic split Rhat; constants reported undefined, never Rhat=1 proof.',
       'interpretation':'Descriptive graph-reference tails, not calibrated regression p-values or physical causal validation. No adaptive tuning after inspecting samples.',
       'method_sources':[{'url':'https://arxiv.org/abs/1608.00607','use':'Configuration-model space and labelling must be explicit; no imported distance-conditioned mixing guarantee.'},{'url':'https://arxiv.org/abs/0912.0685','use':'Directed switches can have disconnected components; longer cycle moves motivate diagnostic extension, but their connectivity theorem does not cover our extra distance margins.'}],
       'small_exact_checks':small}
    save(OUT/'frozen_plan.json',plan);save(OUT/'small_graph_validation.json',small)
    (OUT/'code_snapshot').mkdir()
    for path in (Path(__file__),ROOT/'revision/graph_nulls.py'):(OUT/'code_snapshot'/path.name).write_bytes(path.read_bytes())
    started=time.perf_counter();summary={}
    for ci,(name,(edges,reference)) in enumerate(cases.items()):
        exact,details=exact_support(edges,support,bins,n)
        exact_rows=[]
        for j,g in enumerate(exact):
            if signature(g,bins)!=signature(edges,bins):raise AssertionError('Exact margin check')
            exact_rows.append({'graph_id':j,'graph_sha256':fingerprint(g),'is_observed':g==edges,
                 'changed_fraction_from_observed':1-len(g&edges)/len(edges),**graph_metrics(g,n,reference)})
        pd.DataFrame(exact_rows).to_csv(OUT/f'{name}_original_support_exact_metrics.csv',index=False)
        save(OUT/f'{name}_original_support_exact.json',{'status':'exhaustive',
            'feasible_graphs':len(exact),'typed_scc_enumeration':details,
            'observed_signature':serialized_signature(edges,bins),
            'graphs':[{'graph_id':j,'edges':[list(e) for e in sorted(g)],'signature':serialized_signature(g,bins)} for j,g in enumerate(exact)],
            'limitation':'Finite support of one/two graphs cannot supply fine-grained tail resolution; exact conditional graph reference is not a test of edge-pvalue calibration.'})
        print(f'{name}: exact original support {len(exact)} graphs',flush=True)
        rows=[];snapshots=[];chain_records=[]
        for c in range(config['chains']):
            seed=config['base_seed']+ci*100+c
            r,s,d=chain(edges,reference,bins,distances,seed,c,name,config)
            rows.extend(r);snapshots.append(s);chain_records.append(d)
            save(OUT/'status.json',{'status':'running','case':name,'chains_complete':c+1,'updated_utc':now()})
            print(f'{name}: chain {c+1}/4 productive={d["accepted_graph_changes"]}/{d["attempts"]}',flush=True)
        frame=pd.DataFrame(rows);frame.to_csv(OUT/f'{name}_expanded_chain_samples.csv',index=False)
        np.savez_compressed(OUT/f'{name}_expanded_snapshots.npz',target_regions=np.stack(snapshots),edge_order=np.array(sorted(edges),dtype=str),distance_bins=bins)
        stats=diagnostics(frame,graph_metrics(edges,n,reference))
        result={'case':name,'original_support_feasible_graphs':len(exact),'expanded_distinct_graphs':int(frame.graph_sha256.nunique()),
             'expanded_mean_changed_fraction':float(frame.changed_fraction_from_observed.mean()),
             'chain_records':chain_records,'metrics':stats,
             'mixing_conclusion':'Diagnostic results only; no uniform finite-sample mixing or full-state-space connectivity certification.'}
        save(OUT/f'{name}_diagnostics.json',result);summary[name]=result
    manifest={'status':'completed','completed_utc':now(),'elapsed_seconds':time.perf_counter()-started,
       'frozen_plan_sha256':sha(OUT/'frozen_plan.json'),'cases':{name:{k:v for k,v in r.items() if k not in ('metrics','chain_records')} for name,r in summary.items()},
       'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name not in ('status.json','manifest.json')},
       'limitations':['All original graph thresholds are exploratory and use legacy full-period preprocessing.','Expanded support is a distinct reference family.','Longer chains and exact invariant checks do not prove mixing.','No graph-wide FDR or physical mechanism claim is established.']}
    save(OUT/'manifest.json',manifest);save(OUT/'status.json',{'status':'completed','updated_utc':now()})
    print(json.dumps({'status':'completed','elapsed_seconds':manifest['elapsed_seconds'],'cases':manifest['cases']}),flush=True)


if __name__=='__main__':main()
