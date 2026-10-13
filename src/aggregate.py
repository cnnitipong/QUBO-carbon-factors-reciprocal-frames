import json, glob, numpy as np, sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mo2_lib as L
def fast_front(F):
    order = np.argsort(F[:, 0] + 1e-9*F[:, 1]); kept = []; keptF = np.zeros((0, F.shape[1])); mask = np.zeros(len(F), bool)
    for i in order:
        f = F[i]
        if len(keptF) and np.any(np.all(keptF <= f, 1) & np.any(keptF < f, 1)): continue
        kept.append(i); keptF = np.vstack([keptF, f]); mask[i] = True
    # second pass to remove points dominated by later-kept points (cannot happen with f1-sorted order except ties) 
    return mask
R = {f.split('/')[-1][:-5]: json.load(open(f)) for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results', '*.json'))}
agg = {'geom': R['geom']}
rows = sum((R[f'E1_exact_{c}']['rows'] for c in range(8)), []); lo = np.array(R['E1_exact_0']['lo']); hi = np.array(R['E1_exact_0']['hi'])
F = np.array([[r[1], r[2], r[3]] for r in rows]); U = np.array([r[4] for r in rows]); feas = U <= 1
Fn = (F - lo)/(hi - lo); mk = np.zeros(len(F), bool); idx = np.where(feas)[0]; mk[idx] = fast_front(Fn[idx])
ex_front = Fn[mk]; agg['E1_exact'] = dict(n=len(rows), n_feas=int(feas.sum()), n_front=int(mk.sum()), hv=L.hv3(ex_front), lo=lo.tolist(), hi=hi.tolist(),
    front=[dict(idx=rows[i][0], ec=rows[i][1], ee=rows[i][2], vw=rows[i][3]) for i in np.where(mk)[0]],
    front2d_cv=int(fast_front(Fn[idx][:, [0, 2]]).sum()), cloud=[[rows[i][1], rows[i][2], rows[i][3], int(feas[i])] for i in range(0, len(rows), 8)])
def summ(prefix, seeds, ex=None):
    out = {}
    for meth in ('ws', 'pbi', 'nsga2'):
        hv = []; ig = []; nf = []; t = []; feas_ = []
        for s in range(seeds):
            k = f'{prefix}_{meth}_s{s}'
            if k not in R: continue
            r = R[k]; hv.append(r['hv']); nf.append(len(r['front'])); t.append(r['time']); feas_.append(r['n_feas']/r['n_eval'])
            if ex is not None: ig.append(L.igd([x['Fn'] for x in r['front']], ex))
        out[meth] = dict(hv=hv, igd=ig, n_front=nf, time=t, feas=feas_, front_s0=R.get(f'{prefix}_{meth}_s0', {}).get('front', []),
                         trace_s0=[[x['ec'], x['ee'], x['vw'], int(x['feas'])] for x in R.get(f'{prefix}_{meth}_s0', {}).get('all', [])])
    allf = {m: [x['Fn'] for s in range(seeds) for x in R.get(f'{prefix}_{m}_s{s}', {}).get('front', [])] for m in out}
    lab = sum(([m]*len(v) for m, v in allf.items()), []); FF = np.array(sum(allf.values(), []))
    if len(FF):
        mkc = L.pareto_front(FF); lab = np.array(lab)
        for m in out: out[m]['share_combined'] = float(mkc[lab == m].mean()) if (lab == m).any() else 0.0
    return out
agg['E1_fan'] = summ('E1_fan_base', 5, ex_front)
for sc in ('base', 'bio', 'eaf'): agg[f'E2_sq3_{sc}'] = summ(f'E2_sq3_{sc}', 5)
for case in ('sq6', 'hex', 'dome'): agg[f'E3_{case}'] = summ(f'E3_{case}_base', 3)
agg['E4'] = [dict(lam=R[k]['lam_types'], n_types=R[k]['n_types'], final=R[k]['final']) for k in sorted(R) if k.startswith('E4_')]
agg['E5'] = {k: dict(front=R[k]['front'], viol=R[k]['all_viol'], hv=R[k]['hv']) for k in sorted(R) if k.startswith('E5_')}
agg['E6'] = {k: dict(front=R[k]['front'], beta=R[k]['beta']) for k in sorted(R) if k.startswith('E6_')}
agg['E7'] = {k: dict(front=R[k]['front'], creep=R[k]['creep'], n_feas=R[k]['n_feas'], n_eval=R[k]['n_eval'], umax=R[k]['umax']) for k in sorted(R) if k.startswith('E7_')}
OUTP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'aggregate_v2.json')
json.dump(agg, open(OUTP, 'w')); print('bytes', os.path.getsize(OUTP))
for k in ('E1_fan', 'E2_sq3_base', 'E2_sq3_bio', 'E2_sq3_eaf', 'E3_sq6', 'E3_hex', 'E3_dome'):
    print(k, {m: (round(np.mean(v['hv']), 4), round(np.std(v['hv']), 4), round(np.mean(v['igd']), 4) if v['igd'] else None, round(v.get('share_combined', 0), 2), int(np.mean(v['n_front'])), round(np.mean(v['feas']), 2), round(np.mean(v['time']))) for m, v in agg[k].items()})
print('E1 exact', agg['E1_exact']['n_feas'], agg['E1_exact']['n_front'], round(agg['E1_exact']['hv'], 4), agg['E1_exact']['front2d_cv'])
print('E4', sorted([(d['lam'], d['n_types'], round(d['final']['ec']), round(d['final']['vw'], 1)) for d in agg['E4']]))
print('E5', {k: (len(v['front']), sum(v['viol']), round(v['hv'], 3)) for k, v in agg['E5'].items()})
print('E6', {k: (len(v['front']), round(min(x['ec'] for x in v['front'])), round(max(x['ecsd'] for x in v['front']))) for k, v in agg['E6'].items()})
print('E7', {k: (v['n_feas'], v['n_eval'], len(v['front']), round(min(x['ec'] for x in v['front']))) for k, v in agg['E7'].items()})
