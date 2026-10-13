"""Additional methodology and results panels (fig14 to fig21); run after make_figures.py."""
import json, glob, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from figstyle import save, SER, HALF, FULL, SQ, OUT, C, MAT, SCEN
import mo2_lib as L
from sizing import element_geom_cache, group_element_indices
from frame3d import local_stiffness
ROOT = Path(__file__).resolve().parents[1]
A = json.load(open(ROOT / 'data' / 'aggregate_v2.json')); G = A['geom']
NUM = json.load(open(ROOT / 'data' / 'manuscript_numbers.json'))
RES = ROOT / 'data' / 'results'
# ------------------------------------------------------------------ Fig 14: bearing detail and QUBO matrix structure
m, load, span = L.model('fan'); cat = L.catalogue(); GL = L.group_len(m, L.lengths(m)); sp = L.Space(m, cat, load, GL)
nodes = np.array(m.nodes)
fig, ax = plt.subplots(figsize=HALF)
for e in m.elements:
    p, q = nodes[e['i']], nodes[e['j']]
    if e['kind'] == 'link': ax.plot([p[0], q[0]], [p[1], q[1]], color=C['verm'], lw=2.4, solid_capstyle='butt')
    else: ax.plot([p[0], q[0]], [p[1], q[1]], color=C['blue'], lw=4, solid_capstyle='round', alpha=0.85)
sup = nodes[list(m.support_idx)]; ax.plot(sup[:, 0], sup[:, 1], '^', mfc='white', mec='0.0', ms=5)
ld = nodes[list(m.load_idx)]; ax.plot(ld[:, 0], ld[:, 1], 'o', mfc='0.0', mec='0.0', ms=3)
ax.plot([], [], color=C['blue'], lw=4, label='Beam element'); ax.plot([], [], color=C['verm'], lw=2.4, label='Rigid bearing link, $e = 0.07D$')
ax.plot([], [], '^', mfc='white', mec='0.0', label='Support'); ax.plot([], [], 'o', mfc='0.0', mec='0.0', label='Loaded hub node')
ax.set_aspect('equal'); ax.axis('off'); lg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=7.5); save(fig, 'fig14_a', extra=[lg])
captured = {}
_sa = L.sa_solve
def capture(Q, **kw):
    captured['Q'] = Q.copy(); return _sa(Q, **kw)
L.sa_solve = capture
L.qubo_run(sp, (0.4, 0.2, 0.4), 'pbi', n_iters=1, n_types=True, lam_types=0.05, seed=0)
L.sa_solve = _sa
Q = captured['Q']; ng, ns = m.n_groups, len(cat)
fig, ax = plt.subplots(figsize=SQ)
M = np.log10(np.abs(Q) + 1e-6); M[np.abs(Q) < 1e-9] = np.nan
im = ax.imshow(M, cmap='viridis', vmin=-4, vmax=2, interpolation='nearest'); cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label('log$_{10}$ |Q|', fontsize=8)
for k in range(1, ng + 1): ax.axhline(k*ns - 0.5, color='0.0', lw=0.4); ax.axvline(k*ns - 0.5, color='0.0', lw=0.4)
ax.set_xticks([ns*k + ns/2 for k in range(ng)] + [ng*ns + ns/2]); ax.set_xticklabels([f'$g_{k+1}$' for k in range(ng)] + ['$y$'])
ax.set_yticks([ns*k + ns/2 for k in range(ng)] + [ng*ns + ns/2]); ax.set_yticklabels([f'$g_{k+1}$' for k in range(ng)] + ['$y$'])
ax.set_xlabel('Binary variable'); ax.set_ylabel('Binary variable'); save(fig, 'fig14_b')
NUM['Q'] = dict(n_bits=int(Q.shape[0]), nnz=int((np.abs(Q) > 1e-9).sum()), density=float((np.abs(Q) > 1e-9).mean()))
# ------------------------------------------------------------------ Fig 15: surrogate and gate validation on the fan unit
mh, loadh, _ = L.model('hex'); GLh = L.group_len(mh, L.lengths(mh)); ngh = mh.n_groups
geom = element_geom_cache(mh); groups = group_element_indices(mh); ref = [8]*ngh
r0 = L.evaluate(mh, cat, ref, loadh, GLh); frm = r0['frm']
EA = np.array([c['E']*c['A'] for c in cat]); EIy = np.array([c['E']*c['Iy'] for c in cat]); EIz = np.array([c['E']*c['Iz'] for c in cat]); GJ = np.array([c['G']*c['J'] for c in cat])
Ac = np.array([c['A'] for c in cat]); Dc = np.array([c['D'] for c in cat]); Ic = np.array([c['Iy'] for c in cat]); fdc = np.array([c['fd'] for c in cat])
SE = np.zeros((ngh, ns)); UT = np.zeros((ngh, ns))
for k, eis in groups.items():
    for ei in eis:
        g = geom[ei]; ul = g['T'] @ frm.U[g['dofs']]
        q = [0.5*ul @ local_stiffness(1.0, 1.0, a_, iy_, iz_, j_, g['L']) @ ul for a_, iy_, iz_, j_ in ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))]
        SE[k] += EA*q[0] + EIy*q[1] + EIz*q[2] + GJ*q[3]
        f = frm.element_end_forces_local(ei); N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
        UT[k] = np.maximum(UT[k], (N/Ac + (My+Mz)*Dc/2/Ic)/fdc)
def group_util(frm_, m_, cat_, idx_, k_):
    u = 0.0
    for e, el in enumerate(m_.elements):
        if el['kind'] == 'link' or el['group'] != k_: continue
        sp_ = cat_[idx_[k_]]; f = frm_.element_end_forces_local(e); N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
        u = max(u, (N/sp_['A'] + (My+Mz)*sp_['D']/2/sp_['Iy'])/sp_['fd'])
    return u
pred, real, upred, ureal, mats = [], [], [], [], []
for k in range(ngh):
    for c in range(ns):
        if c == ref[k]: continue
        idx = list(ref); idx[k] = c; r = L.evaluate(mh, cat, idx, loadh, GLh)
        pred.append(-2*(SE[k, c] - SE[k, ref[k]])); real.append(r['vw'] - r0['vw']); upred.append(UT[k, c]); ureal.append(group_util(r['frm'], mh, cat, idx, k)); mats.append(cat[c]['mat'])
pred, real, upred, ureal = map(np.array, (pred, real, upred, ureal)); mats = np.array(mats)
fig, ax = plt.subplots(figsize=HALF)
for mt, mk, col, lab in (('steel', 'o', MAT['steel'], 'Steel option'), ('glulam', 's', MAT['glulam'], 'Glulam option')):
    s_ = mats == mt; ax.plot(real[s_], pred[s_], mk, mfc='white', mec=col, ms=3.5, label=lab)
lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]; ax.plot(lim, lim, color='0.0', lw=0.6, ls='--', label='1:1')
ax.set_xscale('symlog', linthresh=50); ax.set_yscale('symlog', linthresh=50); ax.set_xlabel('Actual change of virtual work (J)'); ax.set_ylabel('Frozen-field prediction (J)'); ax.legend(loc='upper left'); save(fig, 'fig15_a')
fig, ax = plt.subplots(figsize=HALF)
for mt, mk, col, lab in (('steel', 'o', MAT['steel'], 'Steel option'), ('glulam', 's', MAT['glulam'], 'Glulam option')):
    s_ = mats == mt; ax.plot(ureal[s_], upred[s_], mk, mfc='white', mec=col, ms=3.5, label=lab)
ax.plot([0, max(ureal.max(), upred.max())], [0, max(ureal.max(), upred.max())], color='0.0', lw=0.6, ls='--', label='1:1')
ax.axhline(1.0, color='0.0', lw=0.5); ax.axvline(1.0, color='0.0', lw=0.5); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Utilisation of the changed group after re-analysis'); ax.set_ylabel('Frozen-force utilisation of the option'); ax.legend(loc='upper left'); save(fig, 'fig15_b')
sgn = float(np.mean(np.sign(pred) == np.sign(real))); rho = float(np.corrcoef(pred, real)[0, 1])
gate_ok = float(np.mean((upred > 1) == (ureal > 1))); false_pass = int(np.sum((upred <= 1) & (ureal > 1))); false_block = int(np.sum((upred > 1) & (ureal <= 1)))
NUM['surrogate'] = dict(case='hex', ref=cat[8]['label'], n_moves=len(pred), stiff_moves_pred_over_actual=float(np.median(pred[real < 0]/real[real < 0])), soft_moves_pred_over_actual=float(np.median(pred[real > 0]/real[real > 0])), n_real_over=int((ureal > 1).sum()), n_pred_over=int((upred > 1).sum()), sign_agreement=sgn, corr=rho, slope=float(np.polyfit(real, pred, 1)[0]), gate_agreement=gate_ok, false_pass=false_pass, false_block=false_block, ref_vw=r0['vw'])
# ------------------------------------------------------------------ Fig 16: exact fan front, carbon against energy and parallel coordinates
E = A['E1_exact']; F = np.array([[d['ec'], d['ee'], d['vw']] for d in E['front']]); CAT16 = [c for c in G['catalogue'] if c['mat'] != 'reused']
nt = np.array([sum(CAT16[i]['mat'] == 'glulam' for i in d['idx']) for d in E['front']])
fig, ax = plt.subplots(figsize=HALF)
sc = ax.scatter(F[:, 0], F[:, 1], s=6 + 60*(np.log(F[:, 2]) - np.log(F[:, 2].min()))/(np.log(F[:, 2].max()) - np.log(F[:, 2].min())), facecolors='white', edgecolors='0.0', linewidths=0.6)
for n_, lab in ((0, 'all steel'), (4, 'all glulam')):
    s_ = nt == n_
    if s_.any(): ax.plot(F[s_, 0], F[s_, 1], 's' if n_ else 'o', mfc=MAT['glulam'] if n_ else MAT['steel'], mec='0.0', ms=3.5, ls='none', label=f'{lab} ({s_.sum()})')
ax.plot([], [], 'o', mfc='white', mec='0.0', ms=3, label='marker size: virtual work')
ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Embodied energy (MJ)'); ax.legend(loc='upper left', fontsize=7.5); save(fig, 'fig16_a')
fig, ax = plt.subplots(figsize=HALF)
cols = ['Carbon', 'Energy', 'log VW', 'Glulam groups']; P = np.c_[F[:, 0], F[:, 1], np.log10(F[:, 2]), nt]; Pn = (P - P.min(0))/(P.max(0) - P.min(0))
cm5 = plt.get_cmap('viridis')
for i in np.argsort(nt): ax.plot(range(4), Pn[i], color=cm5(nt[i]/4), lw=0.6, alpha=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(cols); ax.set_yticks([0, 1]); ax.set_yticklabels(['min', 'max']); ax.set_ylabel('Normalised value')
for v in range(4): ax.axvline(v, color='0.0', lw=0.4)
h = [Line2D([], [], color=cm5(k/4), lw=1.2, label=f'{k} glulam groups') for k in range(5)]; ax.lg = ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=6.5, handlelength=1.4, columnspacing=0.8); save(fig, 'fig16_b', extra=[lg])
NUM['E1_mix'] = {str(k): int((nt == k).sum()) for k in range(5)}
# ------------------------------------------------------------------ Fig 17: hypervolume against evaluation budget
def running_hv(prefix, meth, seeds, step=11):
    curves = []
    for s in range(seeds):
        r = json.load(open(RES / f'{prefix}_{meth}_s{s}.json')); allr = r['all']; xs = list(range(step, len(allr) + 1, step)); hv = []
        for n in xs:
            fe = [x['Fn'] for x in allr[:n] if x['feas']]
            hv.append(L.hv3([f for f in L.front([dict(Fn=np.array(f), feas=True) for f in fe])] and [x['Fn'] for x in L.front([dict(Fn=np.array(f), feas=True) for f in fe])], n=60000) if fe else 0.0)
        curves.append(hv)
    return np.array(xs), np.array(curves)
for name, prefix, seeds, ttl in (('fig17_a', 'E2_sq3_base', 5, '3 x 3 nexorade'), ('fig17_b', 'E3_sq6_base', 3, '6 x 6 nexorade')):
    fig, ax = plt.subplots(figsize=HALF)
    for meth in ('ws', 'pbi', 'nsga2'):
        xs, Cv = running_hv(prefix, meth, seeds); mu = Cv.mean(0); sd = Cv.std(0)
        ax.plot(xs, mu, color=SER[meth]['color'], ls=SER[meth]['ls'], marker=SER[meth]['marker'], ms=2.5, mfc=SER[meth]['mfc'], lw=0.9, label=SER[meth]['label'])
        ax.fill_between(xs, mu - sd, mu + sd, color=SER[meth]['color'], alpha=0.12, lw=0)
        NUM.setdefault('hv_budget', {})[f'{prefix}_{meth}'] = dict(x=xs.tolist(), hv=mu.tolist())
    ax.set_xlabel('Real FE evaluations'); ax.set_ylabel('Hypervolume of feasible set'); ax.legend(loc='lower right'); save(fig, name)
# ------------------------------------------------------------------ Fig 18: pooled fronts on the larger frames
for name, key in (('fig18_a', 'E3_sq6'), ('fig18_b', 'E3_hex'), ('fig18_c', 'E3_dome')):
    pre = {'E3_sq6': 'E3_sq6_base', 'E3_hex': 'E3_hex_base', 'E3_dome': 'E3_dome_base'}[key]
    fig, ax = plt.subplots(figsize=HALF)
    for meth in ('ws', 'pbi', 'nsga2'):
        pts = []
        for s in range(3):
            r = json.load(open(RES / f'{pre}_{meth}_s{s}.json')); pts += [[x['ec'], x['vw']] for x in r['front']]
        pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], marker=SER[meth]['marker'], ls='none', mfc=SER[meth]['mfc'], mec=SER[meth]['color'], ms=3, label=SER[meth]['label'])
    ax.set_yscale('log'); ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.legend(loc='upper right'); save(fig, name)
# ------------------------------------------------------------------ Fig 19: material share along the front and utilisation of front designs
fig, ax = plt.subplots(figsize=HALF); mk = {'sq3': ('o', C['blue']), 'sq6': ('s', C['verm']), 'hex': ('^', C['green']), 'dome': ('D', C['purple'])}
NUM['timber_share'] = {}
for case, pre in (('sq3', 'E2_sq3_base'), ('sq6', 'E3_sq6_base'), ('hex', 'E3_hex_base'), ('dome', 'E3_dome_base')):
    pts = []
    for s in range(3):
        r = json.load(open(RES / f'{pre}_pbi_s{s}.json')); ngp = G[case]['n_groups']
        pts += [[x['ec'], sum(CAT16[i]['mat'] == 'glulam' for i in x['idx'])/ngp] for x in r['front']]
    pts = np.array(pts); o = np.argsort(pts[:, 0]); rank = np.linspace(0, 1, len(o)); nb = 6; edges = np.linspace(0, 1, nb + 1); cen = 0.5*(edges[1:] + edges[:-1])
    bm = [pts[o, 1][(rank >= edges[i]) & (rank <= edges[i+1])].mean() for i in range(nb)]
    ax.plot(cen, bm, marker=mk[case][0], ms=4, lw=0.9, color=mk[case][1], mfc='white', label={'sq3': '3 x 3', 'sq6': '6 x 6', 'hex': 'Hexagonal', 'dome': 'Dome'}[case])
    NUM['timber_share'][case] = dict(low_third=float(pts[o, 1][:len(o)//3].mean()), high_third=float(pts[o, 1][-len(o)//3:].mean()))
ax.set_xlabel('Position along the front (0 = lowest carbon, 1 = stiffest)'); ax.set_ylabel('Share of groups in glulam, bin mean'); ax.set_ylim(0, 1.05); ax.legend(loc='lower left', fontsize=7.5); save(fig, 'fig19_a')
fig, ax = plt.subplots(figsize=HALF); bins = np.linspace(0, 1, 11)
for meth in ('ws', 'pbi', 'nsga2'):
    u = []
    for s in range(3):
        r = json.load(open(RES / f'E3_hex_base_{meth}_s{s}.json')); u += [x['util'] for x in r['front']]
    ax.hist(u, bins=bins, histtype='step', color=SER[meth]['color'], ls=SER[meth]['ls'], lw=1.0, label=SER[meth]['label']); NUM.setdefault('util_front', {})[meth] = dict(mean=float(np.mean(u)), q90=float(np.quantile(u, 0.9)))
ax.set_xlabel('Peak utilisation of front designs, hexagonal nexorade'); ax.set_ylabel('Designs'); ax.legend(loc='upper left'); save(fig, 'fig19_b')
# ------------------------------------------------------------------ Fig 20: utilisation table of the stress gate on the hexagonal nexorade
m3, load3, _ = L.model('hex'); GL3 = L.group_len(m3, L.lengths(m3)); r3 = L.evaluate(m3, cat, [8]*m3.n_groups, load3, GL3); frm3 = r3['frm']
geom3 = element_geom_cache(m3); groups3 = group_element_indices(m3); UT3 = np.zeros((m3.n_groups, ns))
for k, eis in groups3.items():
    for ei in eis:
        f = frm3.element_end_forces_local(ei); N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
        UT3[k] = np.maximum(UT3[k], (N/Ac + (My+Mz)*Dc/2/Ic)/fdc)
fig, ax = plt.subplots(figsize=(6.4, 2.6))
im = ax.imshow(np.log10(np.maximum(UT3, 1e-3)), cmap='RdYlBu_r', aspect='auto', vmin=-1.5, vmax=1.0)
for k in range(m3.n_groups):
    for c in range(ns):
        if UT3[k, c] > 1: ax.text(c, k, 'x', ha='center', va='center', fontsize=7, color='0.0')
ax.set_xticks(range(ns)); ax.set_xticklabels([c['label'].replace('CHS ', 'S ').replace('GL ', 'T ') for c in cat], rotation=60, ha='right', fontsize=6.5)
ax.set_yticks(range(m3.n_groups)); ax.set_yticklabels([f'$g_{k+1}$' for k in range(m3.n_groups)]); ax.set_xlabel('Catalogue option, sorted by mass per metre (S = steel CHS, T = glulam)'); ax.set_ylabel('Member group')
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('log$_{10}$ utilisation'); save(fig, 'fig20_a')
NUM['gate_table'] = dict(n_blocked=int((UT3 > 1).sum()), n_cells=int(UT3.size), ref_label=cat[8]['label'], case='hex')
# ------------------------------------------------------------------ Fig 21: uncertainty bars and material share under biogenic accounting
E6 = A['E6']; fr = sum((E6[k]['front'] for k in E6 if E6[k]['beta'] == 0.0), []); fr.sort(key=lambda d: d['ec'])
fig, ax = plt.subplots(figsize=HALF)
ax.errorbar([d['ec'] for d in fr], [d['vw'] for d in fr], xerr=[d['ecsd'] for d in fr], fmt='o', ms=2.5, mfc='white', mec=C['blue'], ecolor=C['verm'], elinewidth=0.6, capsize=1.5, alpha=0.9)
ax.set_yscale('log'); ax.set_xlabel('Embodied carbon, mean $\\pm$ one standard deviation (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); save(fig, 'fig21_a')
fig, ax = plt.subplots(figsize=HALF)
for key, mk_, col, lab in (('E2_sq3_base', 'o', C['blue'], 'Fossil-only'), ('E2_sq3_bio', 's', C['verm'], 'Biogenic credit'), ('E2_sq3_eaf', '^', C['green'], 'EAF steel')):
    pts = []
    for s in range(5):
        r = json.load(open(RES / f'{key}_pbi_s{s}.json')); pts += [[x['vw'], sum(CAT16[i]['mat'] == 'glulam' for i in x['idx'])/8] for x in r['front']]
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1] + (0.0 if key == 'E2_sq3_base' else (0.02 if key == 'E2_sq3_bio' else -0.02)), mk_, mfc='white', mec=col, ms=3, ls='none', label=lab)
    NUM.setdefault('timber_share_scen', {})[key] = float(pts[:, 1].mean())
ax.set_xscale('log'); ax.set_xlabel('Virtual work $F^\\top u$ (J)'); ax.set_ylabel('Share of groups in glulam'); ax.set_ylim(-0.05, 1.1); lg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5); save(fig, 'fig21_b', extra=[lg])
json.dump(NUM, open(ROOT / 'data' / 'manuscript_numbers.json', 'w'), indent=1, default=float)
print('panels:', len(list(OUT.glob('*.pdf'))), NUM['surrogate'], NUM['Q'], NUM['gate_table'], NUM['timber_share'], NUM['util_front'], NUM['E1_mix'], NUM['timber_share_scen'])
