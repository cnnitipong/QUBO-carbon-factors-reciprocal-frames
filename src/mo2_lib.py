"""Multi-material (steel CHS / glulam / reclaimed steel) carbon-energy-stiffness QUBO sizing
of 3-D reciprocal frames with quadratic constructability coupling. Reuses the v3 FE/QUBO code."""
import sys, os, json, time, itertools, numpy as np
CODE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, CODE)
from reciprocal_geom_v2 import fan_unit, square_nexorade, hex_nexorade, rf_dome
from reciprocal_geom import LINK_SECTION
from frame3d import Frame3D, local_stiffness
from sizing import element_geom_cache, group_element_indices
from sections import CHS_CATALOG, E_STEEL, G_STEEL
from qubo import QUBOBuilder, sa_solve, nsga2, pareto_front, fast_nondominated_sort
# ---------------- materials (A1-A3 per kg; mean, std of carbon factor) ----------------
MAT = {
 'steel':   dict(E=E_STEEL, G=G_STEEL, rho=7850., ec=1.55, ec_sd=0.35, ee=21.5, fd=355e6, kdef=0.0),   # ICE v3 world avg section
 'steelEAF':dict(E=E_STEEL, G=G_STEEL, rho=7850., ec=0.47, ec_sd=0.15, ee=10.0, fd=355e6, kdef=0.0),   # ICE v2 recycled
 'reused':  dict(E=E_STEEL, G=G_STEEL, rho=7850., ec=0.10, ec_sd=0.05, ee=1.5,  fd=275e6, kdef=0.0),   # reclaimed S275 (reconditioning only)
 'glulam':  dict(E=11.5e9, G=0.65e9, rho=420.,  ec=0.42, ec_sd=0.15, ee=12.0, fd=15.4e6, kdef=0.6),   # ICE v2 fossil-only, GL24h, SC1
 'glulamB': dict(E=11.5e9, G=0.65e9, rho=420.,  ec=-1.20, ec_sd=0.30, ee=12.0, fd=15.4e6, kdef=0.6),  # -1/+1 biogenic storage
}
def chs(s, mat):
    return dict(mat=mat, D=s['D'], A=s['A'], Iy=s['Iy'], Iz=s['Iz'], J=s['J'], label=f"CHS {s['D_mm']}x{s['t_mm']}")
def rect(b_mm, mat='glulam'):
    b = b_mm/1e3; h = 2*b
    return dict(mat=mat, D=h, A=b*h, Iy=b*h**3/12, Iz=h*b**3/12, J=0.229*b**3*h, label=f"GL {b_mm:.0f}x{2*b_mm:.0f}")
def finish(cat):
    for s in cat:
        m = MAT[s['mat']]; mpl = s['A']*m['rho']
        s.update(E=m['E'], G=m['G'], fd=m['fd'], kdef=m['kdef'], mass=mpl, ec=mpl*m['ec'], ecsd=mpl*m['ec_sd'], ee=mpl*m['ee'])
    cat.sort(key=lambda s: s['mass']); return cat
def catalogue(steel='steel', timber='glulam', stock=0):
    cat = [chs(s, steel) for s in CHS_CATALOG] + [rect(b, timber) for b in (80,100,120,140,160,180,200,240)]
    if stock:  # reclaimed lots: one lot per group at most (quadratic exclusivity)
        for i, s in enumerate(CHS_CATALOG[2:2+stock]):
            d = chs(s, 'reused'); d['label'] = 'RE ' + d['label'][4:]; d['lot'] = i; cat.append(d)
    return finish(cat)
# ---------------- models ----------------
def model(name):
    if name == 'fan':  return fan_unit(4, 2.0, 0.25, single_group=False), -2200.0, 2.0
    if name == 'sq3':  return square_nexorade(3, 3, 2.5, 0.25), -450.0, 7.5
    if name == 'sq6':  return square_nexorade(6, 6, 2.5, 0.25), -450.0, 15.0
    if name == 'hex':  return hex_nexorade(3, 2.0, 0.25), -1400.0, 12.0
    if name == 'dome': return rf_dome(3, 18.5, 3.15, 0.25), -3500.0, 18.5
    raise ValueError(name)
def lengths(m):
    return np.array([np.linalg.norm(np.array(m.nodes[e['j']])-np.array(m.nodes[e['i']])) for e in m.elements])
def group_len(m, L):
    g = np.zeros(m.n_groups)
    for e, el in enumerate(m.elements):
        if el['kind'] != 'link': g[el['group']] += L[e]
    return g
# ---------------- evaluation ----------------
def build(m, cat, idx, load, creep=False):
    frm = Frame3D()
    for xyz in m.nodes: frm.add_node(*xyz)
    for e in m.elements:
        if e['kind'] == 'link': sp = LINK_SECTION; E, G = E_STEEL, G_STEEL
        else:
            sp = cat[idx[e['group']]]; E, G = sp['E'], sp['G']
            if creep: E, G = E/(1+sp['kdef']), G/(1+sp['kdef'])
        frm.add_element(e['i'], e['j'], E, G, sp['A'], sp['Iy'], sp['Iz'], sp['J'])
    fix_t = (getattr(m, 'meta', None) or {}).get('fix_torsion_at_supports', True)
    for ni in m.support_idx: frm.fix(ni, dofs=(0,1,2,3) if fix_t else (0,1,2))
    for ni in m.load_idx: frm.add_load(ni, fz=load)
    return frm
def evaluate(m, cat, idx, load, GL, creep=False):
    """returns dict: ec (mean carbon), ecsd, ee, vw, util, umax (m), frm"""
    frm = build(m, cat, idx, load, creep); frm.solve(check_cond=False)
    ec = sum(cat[idx[k]]['ec']*GL[k] for k in range(m.n_groups))
    ee = sum(cat[idx[k]]['ee']*GL[k] for k in range(m.n_groups))
    # carbon-factor uncertainty: perfectly correlated within a material, independent across materials
    bym = {}
    for k in range(m.n_groups):
        s = cat[idx[k]]; bym[s['mat']] = bym.get(s['mat'], 0.0) + s['ecsd']*GL[k]
    ecsd = float(np.sqrt(sum(v*v for v in bym.values())))
    util = 0.0
    for e, el in enumerate(m.elements):
        if el['kind'] == 'link': continue
        sp = cat[idx[el['group']]]; f = frm.element_end_forces_local(e)
        N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
        util = max(util, (N/sp['A'] + (My+Mz)*sp['D']/2/sp['Iy'])/sp['fd'])
    return dict(ec=ec, ecsd=ecsd, ee=ee, vw=frm.compliance(), util=util, umax=frm.max_abs_disp(), frm=frm)
# ---------------- objective space ----------------
class Space:
    """normalisation bounds from uniform extreme designs; objectives (carbon[+beta*sd], energy, VW)."""
    def __init__(self, m, cat, load, GL, beta=0.0, umax=None, creep=False):
        self.m, self.cat, self.load, self.GL, self.beta, self.umax, self.creep = m, cat, load, GL, beta, umax, creep
        ex = []
        for c in range(len(cat)):
            r = evaluate(m, cat, [c]*m.n_groups, load, GL, creep); ex.append(self.F(r))
        ex = np.array(ex); self.lo = ex.min(0); self.hi = ex.max(0); self.scale = self.hi - self.lo
        self.n_eval = 0
    def F(self, r):
        return np.array([r['ec'] + self.beta*r['ecsd'], r['ee'], r['vw']])
    def feasible(self, r):
        return r['util'] <= 1.0 and (self.umax is None or r['umax'] <= self.umax)
    def norm(self, F): return (np.asarray(F) - self.lo)/self.scale
    def eval(self, idx):
        r = evaluate(self.m, self.cat, list(idx), self.load, self.GL, self.creep); self.n_eval += 1
        r['F'] = self.F(r); r['Fn'] = self.norm(r['F']); r['feas'] = self.feasible(r); r['idx'] = list(idx); return r
# ---------------- QUBO sizing (frozen-field linearisation + scalarisation + couplings) ----------------
def qubo_run(sp, lam, scal='ws', theta=0.3, n_iters=10, move=3, reads=25, sweeps=120, seed=0,
             n_types=None, lam_types=0.0, stock=False, init=None, log=None, stress_gate=True):
    """lam: weight vector (3,) on (carbon, energy, VW). scal: 'ws' weighted sum or 'pbi' (quadratic
    perpendicular penalty). n_types/lam_types: standardisation coupling (penalise distinct types used).
    stock: reclaimed lots exclusive (quadratic). Returns list of evaluation dicts (trace)."""
    m, cat = sp.m, sp.cat; ng, ns = m.n_groups, len(cat)
    geom = element_geom_cache(m); groups = group_element_indices(m)
    lam = np.asarray(lam, float); lamh = lam/np.linalg.norm(lam)
    cur = list(init) if init is not None else [ns//2]*ng
    trace = []; z = np.zeros(3)  # utopia in normalised space
    for it in range(n_iters):
        r = sp.eval(cur); trace.append(r)
        if log is not None: log.append(r)
        SE = np.zeros((ng, ns)); ec = np.zeros((ng, ns)); ee = np.zeros((ng, ns)); sd = np.zeros((ng, ns)); UT = np.zeros((ng, ns))
        EA = np.array([c_['E']*c_['A'] for c_ in cat]); EIy = np.array([c_['E']*c_['Iy'] for c_ in cat]); EIz = np.array([c_['E']*c_['Iz'] for c_ in cat]); GJ = np.array([c_['G']*c_['J'] for c_ in cat])
        Ac = np.array([c_['A'] for c_ in cat]); Dc = np.array([c_['D'] for c_ in cat]); Ic = np.array([c_['Iy'] for c_ in cat]); fdc = np.array([c_['fd'] for c_ in cat])
        for k, eis in groups.items():
            for ei in eis:
                g = geom[ei]; ul = g['T'] @ r['frm'].U[g['dofs']]; Lk = g['L']
                q = [0.5*ul @ local_stiffness(1.0, 1.0, a_, iy_, iz_, j_, Lk) @ ul for a_, iy_, iz_, j_ in ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))]
                SE[k] += EA*q[0] + EIy*q[1] + EIz*q[2] + GJ*q[3]
                f = r['frm'].element_end_forces_local(ei)
                N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
                UT[k] = np.maximum(UT[k], (N/Ac + (My+Mz)*Dc/2/Ic)/fdc)
            for c in range(ns):
                ec[k, c] = cat[c]['ec']*sp.GL[k]; ee[k, c] = cat[c]['ee']*sp.GL[k]; sd[k, c] = cat[c]['ecsd']*sp.GL[k]
        r.pop('frm', None)
        cur_SE = np.array([SE[k, cur[k]] for k in range(ng)])
        a = np.zeros((3, ng*ns)); b = np.zeros(3)
        for k in range(ng):
            for c in range(ns):
                v = k*ns + c
                a[0, v] = (ec[k, c] + sp.beta*sd[k, c])/sp.scale[0]
                a[1, v] = ee[k, c]/sp.scale[1]
                a[2, v] = -2.0*(SE[k, c] - cur_SE[k])/sp.scale[2]   # dC ~ -2 dSE under frozen field
        b[0] = -sp.lo[0]/sp.scale[0]; b[1] = -sp.lo[1]/sp.scale[1]; b[2] = (r['vw'] - sp.lo[2])/sp.scale[2]
        nb = ng*ns + (ns if n_types is not None else 0)
        qb = QUBOBuilder(nb)
        if scal == 'ws':
            lin = lam @ a
            for v in range(ng*ns): qb.add_linear(v, lin[v])
        else:  # PBI: lamh.(F-z) + theta * ||P (F-z)||^2, P = I - lamh lamh^T  (exactly quadratic in x)
            P = np.eye(3) - np.outer(lamh, lamh); Fz = b - z
            lin = lamh @ a + 2*theta*(Fz @ P @ a)
            Qm = theta*(a.T @ P @ a)
            for v in range(ng*ns):
                qb.add_linear(v, lin[v] + Qm[v, v])
                for w in range(v+1, ng*ns):
                    if abs(Qm[v, w]) > 1e-12: qb.add_quadratic(v, w, 2*Qm[v, w])
        BIG = 50.0
        for k in range(ng):
            lo, hi = cur[k]-move, cur[k]+move
            for c in range(ns):
                same_mat = cat[c]['mat'] == cat[cur[k]]['mat']
                near = lo <= c <= hi
                if not near: qb.add_linear(k*ns+c, BIG)
                elif stress_gate and UT[k, c] > 1.0 and UT[k].min() <= 1.0: qb.add_linear(k*ns+c, BIG)
            qb.add_onehot_penalty([k*ns+c for c in range(ns)], 4.0)
        if n_types is not None:  # y_c = type c used; x_kc <= y_c ; cost lam_types * sum y
            for c in range(ns):
                y = ng*ns + c; qb.add_linear(y, lam_types)
                for k in range(ng): qb.add_quadratic(k*ns+c, y, -4.0); qb.add_linear(k*ns+c, 4.0)
        if stock:  # each reclaimed lot in at most one group
            for c in range(ns):
                if 'lot' in cat[c]:
                    for k in range(ng):
                        for l in range(k+1, ng): qb.add_quadratic(k*ns+c, l*ns+c, 6.0)
        sa = sa_solve(qb.Q, n_reads=reads, n_sweeps=sweeps, seed=seed*7919 + it)
        X = sa['best_x'][:ng*ns].reshape(ng, ns)
        new = [int(np.argmax(X[k])) if X[k].sum() > 0 else cur[k] for k in range(ng)]
        if new == cur and it > 0: break
        cur = new
    r = sp.eval(cur); r.pop('frm', None); trace.append(r)
    if log is not None: log.append(r)
    return trace
def simplex(n=4):
    return [(a/n, b/n, 1-(a+b)/n) for a in range(n+1) for b in range(n+1-a)]
def front(rows):
    F = np.array([r['Fn'] for r in rows if r['feas']]); keep = [r for r in rows if r['feas']]
    if len(F) == 0: return []
    mk = pareto_front(F); return [keep[i] for i in np.where(mk)[0]]
def hv3(Fn, ref=1.05, n=200000, seed=0):
    """Monte-Carlo hypervolume in normalised space (minimisation), fixed sample set."""
    Fn = np.asarray(Fn)
    if len(Fn) == 0: return 0.0
    rng = np.random.default_rng(seed); S = rng.random((n, 3))*ref
    dom = np.zeros(n, bool)
    for f in Fn: dom |= np.all(S >= f, 1)
    return float(dom.mean()*ref**3)
def igd(Fn, ref_front):
    if len(Fn) == 0: return np.inf
    D = np.linalg.norm(np.asarray(ref_front)[:, None, :] - np.asarray(Fn)[None, :, :], axis=2)
    return float(D.min(1).mean())
def strip(r):
    return dict(idx=r['idx'], ec=float(r['ec']), ecsd=float(r['ecsd']), ee=float(r['ee']), vw=float(r['vw']),
                util=float(r['util']), umax=float(r['umax']), feas=bool(r['feas']), Fn=[float(x) for x in r['Fn']])
