import sys, os, json, time, itertools, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mo2_lib as L
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results'); os.makedirs(RES, exist_ok=True); T0 = time.time(); BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150
GUARD = float(sys.argv[4]) if len(sys.argv) > 4 else 45
SHARD, NSH = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
SCEN = {'base': dict(steel='steel', timber='glulam'), 'bio': dict(steel='steel', timber='glulamB'), 'eaf': dict(steel='steelEAF', timber='glulam')}
W = L.simplex(4)  # 15 weights
_cache = {}
def space(case, scen, beta=0.0, umax=False, creep=False, stock=0):
    key = (case, scen, beta, umax, creep, stock)
    if key not in _cache:
        m, load, span = L.model(case); cat = L.catalogue(stock=stock, **SCEN[scen]); GL = L.group_len(m, L.lengths(m))
        _cache[key] = L.Space(m, cat, load, GL, beta=beta, umax=(span/250 if umax else None), creep=creep)
    return _cache[key]
def save(name, obj):
    json.dump(obj, open(f'{RES}/{name}.json', 'w'), default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else None)
def done(name): return os.path.exists(f'{RES}/{name}.json')
def left(): return BUDGET - (time.time() - T0)
def run_method(sp, method, seed, n_w=W, n_iters=10, **kw):
    log = []
    if method == 'nsga2':
        budget = len(n_w)*(n_iters+1)
        def obj(x):
            r = sp.eval(x); r.pop('frm', None); log.append(r); return r['Fn'] + (10.0 if not r['feas'] else 0.0)
        L.nsga2(obj, [len(sp.cat)]*sp.m.n_groups, n_evals=budget, pop_size=24, seed=seed)
    else:
        for lam in n_w: L.qubo_run(sp, lam, method, n_iters=n_iters, seed=seed, log=log, **kw)
    fr = L.front(log)
    return dict(n_eval=len(log), n_feas=int(sum(r['feas'] for r in log)), front=[L.strip(r) for r in fr],
                hv=L.hv3([r['Fn'] for r in fr]), all=[L.strip(r) for r in log])
jobs = []
# geometry export + catalogue
jobs.append(('geom', None))
# E1 exact fan enumeration in chunks of 8192
for ch in range(8): jobs.append((f'E1_exact_{ch}', ('exact', ch)))
for meth in ('ws', 'pbi', 'nsga2'):
    for s in range(5): jobs.append((f'E1_fan_base_{meth}_s{s}', ('run', 'fan', 'base', meth, s, {})))
# E2 sq3 scenarios
for scen in ('base', 'bio', 'eaf'):
    for meth in ('ws', 'pbi', 'nsga2'):
        for s in range(5): jobs.append((f'E2_sq3_{scen}_{meth}_s{s}', ('run', 'sq3', scen, meth, s, {})))
# E3 scaling
for case in ('sq6', 'hex', 'dome'):
    for meth in ('pbi', 'ws', 'nsga2'):
        for s in range(3): jobs.append((f'E3_{case}_base_{meth}_s{s}', ('run', case, 'base', meth, s, {})))
# E4 standardisation (sq6, pbi, fixed weight)
for lt in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
    for s in range(3): jobs.append((f'E4_sq6_types_{lt}_s{s}', ('types', 'sq6', lt, s)))
# E5 reclaimed stock (sq6)
for coup in (0, 1):
    for s in range(3): jobs.append((f'E5_sq6_stock_c{coup}_s{s}', ('stock', 'sq6', coup, s)))
# E6 robust beta (sq3)
for beta in (0.0, 1.0, 2.0):
    for s in range(2): jobs.append((f'E6_sq3_beta{beta}_s{s}', ('beta', 'sq3', beta, s)))
# E7 creep / SLS (sq3)
for creep in (0, 1):
    for s in range(2): jobs.append((f'E7_sq3_creep{creep}_s{s}', ('creep', 'sq3', creep, s)))
for ji, (name, spec) in enumerate(jobs):
    if ji % NSH != SHARD or done(name): continue
    if left() < GUARD: print('TIME; stopping before', name); break
    t = time.time()
    if spec is None:
        out = {}
        for case in ('fan', 'sq3', 'sq6', 'hex', 'dome'):
            m, load, span = L.model(case)
            out[case] = dict(nodes=[list(map(float, n)) for n in m.nodes], elements=[dict(i=e['i'], j=e['j'], kind=e['kind'], group=e.get('group', -1)) for e in m.elements],
                             supports=list(map(int, m.support_idx)), loads=list(map(int, m.load_idx)), n_groups=m.n_groups, load=load, span=span)
        cat = L.catalogue(stock=4)
        out['catalogue'] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()} for s in cat]
        out['materials'] = L.MAT; save(name, out)
    elif spec[0] == 'exact':
        sp = space('fan', 'base'); ns = len(sp.cat); ch = spec[1]; rows = []
        allidx = list(itertools.product(range(ns), repeat=4))
        for idx in allidx[ch*8192:(ch+1)*8192]:
            r = sp.eval(idx); rows.append([list(idx), float(r['ec']), float(r['ee']), float(r['vw']), float(r['util']), float(r['umax'])])
            if left() < 10: break
        if len(rows) < len(allidx[ch*8192:(ch+1)*8192]): print('partial chunk, rerun'); break
        save(name, dict(rows=rows, lo=sp.lo.tolist(), hi=sp.hi.tolist()))
    elif spec[0] == 'run':
        _, case, scen, meth, s, kw = spec; sp = space(case, scen); r = run_method(sp, meth, s, **kw); r['time'] = time.time()-t; save(name, r)
    elif spec[0] == 'types' or spec[0] == 'stock' or spec[0] == 'beta' or spec[0] == 'creep':
        pass
    if spec is not None and spec[0] == 'types':
        _, case, lt, s = spec; sp = space(case, 'base'); log = []
        L.qubo_run(sp, (0.4, 0.2, 0.4), 'pbi', seed=s, log=log, n_types=True, lam_types=lt)
        r = log[-1]; save(name, dict(final=L.strip(r), n_types=len(set(r['idx'])), lam_types=lt, time=time.time()-t, trace=[L.strip(x) for x in log]))
    elif spec[0] == 'stock':
        _, case, coup, s = spec; sp = space(case, 'base', stock=4); log = []
        for lam in L.simplex(2): L.qubo_run(sp, lam, 'pbi', seed=s, log=log, stock=bool(coup))
        def viol(r):
            lots = [sp.cat[c].get('lot') for c in r['idx']]; lots = [l for l in lots if l is not None]
            return len(lots) - len(set(lots))
        fr = L.front(log); save(name, dict(front=[dict(L.strip(r), viol=viol(r), n_reused=sum('lot' in sp.cat[c] for c in r['idx'])) for r in fr],
                                            all_viol=[viol(r) for r in log], hv=L.hv3([r['Fn'] for r in fr if viol(r) == 0]), time=time.time()-t))
    elif spec[0] == 'beta':
        _, case, beta, s = spec; sp = space(case, 'base', beta=beta); log = []
        for lam in W: L.qubo_run(sp, lam, 'pbi', seed=s, log=log)
        fr = L.front(log); save(name, dict(front=[L.strip(r) for r in fr], beta=beta, lo=sp.lo.tolist(), scale=sp.scale.tolist(), time=time.time()-t))
    elif spec[0] == 'creep':
        _, case, creep, s = spec; sp = space(case, 'base', umax=True, creep=bool(creep)); log = []
        for lam in W: L.qubo_run(sp, lam, 'pbi', seed=s, log=log)
        fr = L.front(log); save(name, dict(front=[L.strip(r) for r in fr], creep=creep, n_feas=int(sum(r['feas'] for r in log)), n_eval=len(log), umax=sp.umax, time=time.time()-t))
    print(name, round(time.time()-t, 1), 's')
print('REMAINING', sum(not done(n) for n, _ in jobs))
