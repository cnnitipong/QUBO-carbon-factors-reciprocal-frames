"""Regenerates every manuscript panel from data/aggregate_v2.json (greyscale, one file per panel)."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from pathlib import Path
from figstyle import save, SER, HALF, FULL, SQ, GREY, OUT, C, MAT, SCEN
ROOT = Path(__file__).resolve().parents[1]
A = json.load(open(ROOT / 'data' / 'aggregate_v2.json'))
G = A['geom']; CAT20 = G['catalogue']; CAT16 = [c for c in CAT20 if c['mat'] != 'reused']
NUM = {}  # every number quoted in the paper is written here for the audit
# ------------------------------------------------------------------ helpers
def plan(ax, g, idx=None, cat=None, lw0=0.8, title=None, oblique=False):
    nodes = np.array(g['nodes'])
    if oblique:  # cabinet projection: x' = x - 0.45 y, y' = z + 0.35 y
        nodes = np.c_[nodes[:, 0] - 0.45*nodes[:, 1], nodes[:, 2] + 0.35*nodes[:, 1], nodes[:, 2]]
    for e in g['elements']:
        p, q = nodes[e['i']], nodes[e['j']]
        if e['kind'] == 'link':
            ax.plot([p[0], q[0]], [p[1], q[1]], color='0.7', lw=0.4, ls=':')
            continue
        if idx is None:
            ax.plot([p[0], q[0]], [p[1], q[1]], color='0.1', lw=lw0)
        else:
            s = cat[idx[e['group']]]
            lw = 0.6 + 2.6*s['mass']/max(c['mass'] for c in cat)
            col = MAT['steel'] if s['mat'] in ('steel', 'steelEAF') else (MAT['glulam'] if s['mat'].startswith('glulam') else MAT['reused'])
            ls = '-' if s['mat'] != 'reused' else (0, (3, 1.5))
            ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw, ls=ls, solid_capstyle='round')
    sup = nodes[g['supports']]
    ax.plot(sup[:, 0], sup[:, 1], marker='^', ls='none', ms=3.5, mfc='white', mec='0.0', mew=0.6)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=9)
def mat_legend(ax, loc='upper center', reused=False):
    h = [Line2D([], [], color=MAT['steel'], lw=2, label='Steel CHS'), Line2D([], [], color=MAT['glulam'], lw=2, label='Glulam')]
    if reused: h.append(Line2D([], [], color=MAT['reused'], lw=2, ls=(0, (3, 1.5)), label='Reclaimed CHS'))
    h.append(Line2D([], [], marker='^', ls='none', mfc='white', mec='0.0', label='Support'))
    return ax.legend(handles=h, loc=loc, fontsize=8, handlelength=2.2, ncol=2, bbox_to_anchor=(0.5, -0.06))
def front_pts(d, key='ec'):
    return np.array([[p['ec'], p['ee'], p['vw']] for p in d])
# ------------------------------------------------------------------ Fig 1 typologies
for name, key, ttl in (('fig01_a', 'sq3', '3 x 3 square nexorade'), ('fig01_b', 'sq6', '6 x 6 square nexorade'),
                       ('fig01_c', 'hex', 'Hexagonal nexorade'), ('fig01_d', 'dome', 'Reciprocal dome (oblique)')):
    fig, ax = plt.subplots(figsize=SQ); plan(ax, G[key], oblique=(key == 'dome')); save(fig, name)
NUM['cases'] = {k: dict(groups=G[k]['n_groups'], nodes=len(G[k]['nodes']), members=sum(e['kind'] != 'link' for e in G[k]['elements']),
                       links=sum(e['kind'] == 'link' for e in G[k]['elements']), load=G[k]['load'], span=G[k]['span']) for k in ('fan', 'sq3', 'sq6', 'hex', 'dome')}
# ------------------------------------------------------------------ Fig 2 catalogue
fig, ax = plt.subplots(figsize=HALF)
for mat, mk, col, lab in (('steel', 'o', MAT['steel'], 'S355 CHS'), ('glulam', 's', MAT['glulam'], 'GL24h rectangle'), ('reused', 'D', MAT['reused'], 'Reclaimed CHS')):
    c = [s for s in CAT20 if s['mat'] == mat]
    ax.plot([s['E']*s['Iy']/1e6 for s in c], [s['mass'] for s in c], marker=mk, ls='none', mfc=col, mec=col, ms=4.5 if mat != 'reused' else 4, label=lab)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Bending stiffness $EI_y$ (MN m$^2$)'); ax.set_ylabel('Mass per metre (kg/m)'); ax.legend(loc='upper left')
save(fig, 'fig02_a')
fig, ax = plt.subplots(figsize=HALF)
for mat, mk, col, lab in (('steel', 'o', MAT['steel'], 'S355 CHS'), ('glulam', 's', MAT['glulam'], 'GL24h rectangle'), ('reused', 'D', MAT['reused'], 'Reclaimed CHS')):
    c = [s for s in CAT20 if s['mat'] == mat]
    ax.plot([s['E']*s['Iy']/1e6 for s in c], [s['ec'] for s in c], marker=mk, ls='none', mfc=col, mec=col, ms=4.5 if mat != 'reused' else 4, label=lab)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Bending stiffness $EI_y$ (MN m$^2$)'); ax.set_ylabel('Embodied carbon per metre (kgCO$_2$e/m)'); ax.legend(loc='upper left')
save(fig, 'fig02_b')
NUM['catalogue'] = [dict(label=s['label'], mat=s['mat'], mass=s['mass'], EI=s['E']*s['Iy'], ec=s['ec'], ee=s['ee']) for s in CAT20]
# ------------------------------------------------------------------ Fig 3 flowchart
fig = plt.figure(figsize=(6.4, 3.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-1, 100); ax.set_ylim(0, 50); ax.axis('off')
def box(x, y, w, h, text, fill='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', fc=fill, ec='0.0', lw=0.7))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, linespacing=1.25)
def arrow(x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14, lw=0.7, color='0.0', shrinkA=2, shrinkB=2))
    if text: ax.text((x0+x1)/2, (y0+y1)/2 + 1.6, text, ha='center', fontsize=7.5)
box(3, 32, 19, 14, 'Reciprocal frame\n3-D frame FE\n(current sections)', fill='#D6E8F5')
box(27, 32, 20, 14, 'Frozen displacement\nfield: strain energy\nof every option')
box(52, 32, 20, 14, 'Frozen member forces:\nutilisation gate\n(stress)')
box(77, 32, 21, 14, 'Carbon, energy,\nuncertainty per\noption (catalogue)')
box(14, 6, 28, 15, 'One-hot QUBO: scalarised\nobjectives (WS or PBI)\n+ move limit + gates', fill='#FDE7C9')
box(47, 6, 24, 15, 'Quadratic couplings:\nstandardisation,\nreclaimed-lot exclusivity', fill='#FDE7C9')
box(76, 6, 22, 15, 'Simulated-annealing\nsolve, decode\nnew assignment')
arrow(22, 39, 27, 39); arrow(47, 39, 52, 39); arrow(72, 39, 77, 39)
arrow(37, 32, 28, 21); arrow(62, 32, 59, 21); arrow(87, 32, 87, 21)
arrow(42, 13.5, 47, 13.5); arrow(71, 13.5, 76, 13.5)
ax.plot([87, 87, 1, 1], [5.5, 2, 2, 31], color='0.0', lw=0.7)
ax.add_patch(FancyArrowPatch((1, 31), (1, 32), arrowstyle='-|>', mutation_scale=14, lw=0.7, color='0.0', shrinkA=0, shrinkB=0))
ax.text(50, 3.6, 'iterate until the assignment stops changing; one real FE solve per iteration', ha='center', va='bottom', fontsize=7.5, style='italic')
save(fig, 'fig03_a')
# ------------------------------------------------------------------ Fig 4 scalarisation sketch
t = np.linspace(0, 1, 400); fx = t; fy = 1 - np.sqrt(t) - 0.08*np.exp(-((t-0.5)/0.12)**2)  # convex front with a non-convex dent
fig, ax = plt.subplots(figsize=HALF); ax.plot(fx, fy, color='0.0', lw=1.2, label='Pareto front')
for w in (0.3, 0.45, 0.55, 0.7):
    s = w*fx + (1-w)*fy; i = s.argmin(); ax.plot(fx[i], fy[i], 'o', mfc=C['blue'], mec=C['blue'], ms=4.5)
    xs = np.linspace(0, 1, 2); ax.plot(xs, (s[i] - w*xs)/(1-w), color=C['blue'], lw=0.7, ls='--', alpha=0.7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_xlabel('$f_1$ (normalised)'); ax.set_ylabel('$f_2$ (normalised)'); ax.legend(loc='upper right'); save(fig, 'fig04_a')
fig, ax = plt.subplots(figsize=HALF); ax.plot(fx, fy, color='0.0', lw=1.2, label='Pareto front')
for ang in (30, 45, 60):
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); ax.plot([0, d[0]*1.0], [0, d[1]*1.0], color=C['verm'], lw=0.7, ls='--', alpha=0.7)
    P = np.c_[fx, fy]; d1 = P @ d; d2 = np.linalg.norm(P - np.outer(d1, d), axis=1); i = (d1 + 5*d2**2).argmin()
    ax.plot(fx[i], fy[i], 's', mfc=C['verm'], mec=C['verm'], ms=4.5); ax.plot([fx[i], d1[i]*d[0]], [fy[i], d1[i]*d[1]], color=C['verm'], lw=0.8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.set_xlabel('$f_1$ (normalised)'); ax.set_ylabel('$f_2$ (normalised)'); ax.legend(loc='upper right'); save(fig, 'fig04_b')
# ------------------------------------------------------------------ Fig 5 exact validation (fan unit)
E = A['E1_exact']; cloud = np.array(E['cloud']); exf = front_pts(E['front']); wsf = front_pts(A['E1_fan']['ws']['front_s0']); pbf = front_pts(A['E1_fan']['pbi']['front_s0']); nsf = front_pts(A['E1_fan']['nsga2']['front_s0'])
for name, j, lab in (('fig05_a', 0, 'Embodied carbon (kgCO$_2$e)'), ('fig05_b', 1, 'Embodied energy (MJ)')):
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(cloud[:, j], cloud[:, 2], ',', color='0.75', label='All designs (1 in 8 shown)')
    o = np.argsort(exf[:, j]); ax.plot(exf[o, j], exf[o, 2], color='0.0', lw=1.0, label='Exact front (135)')
    ax.plot(wsf[:, j], wsf[:, 2], 'o', mfc='white', mec=C['blue'], ms=3.5, label='WS-QUBO')
    ax.plot(nsf[:, j], nsf[:, 2], '^', mfc='white', mec=C['green'], ms=3.5, label='NSGA-II')
    ax.set_yscale('log'); ax.set_xlabel(lab); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.set_ylim(bottom=0.8); lg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5); save(fig, name, extra=[lg])
NUM['E1'] = dict(n=E['n'], n_feas=E['n_feas'], n_front=E['n_front'], front2d=E['front2d_cv'], hv_exact=E['hv'],
                 **{m: dict(hv_mean=float(np.mean(v['hv'])), hv_sd=float(np.std(v['hv'])), igd_mean=float(np.mean(v['igd'])), igd_sd=float(np.std(v['igd'])),
                            n_front=float(np.mean(v['n_front'])), share=v['share_combined'], time=float(np.mean(v['time']))) for m, v in A['E1_fan'].items()})
NUM['E1']['ws_pct_exact'] = NUM['E1']['ws']['hv_mean']/E['hv']*100
# ------------------------------------------------------------------ Fig 6 hypervolume by case
cases = [('E1_fan', 'Fan'), ('E2_sq3_base', 'Sq3 base'), ('E2_sq3_bio', 'Sq3 biogenic'), ('E2_sq3_eaf', 'Sq3 EAF'), ('E3_sq6', 'Sq6'), ('E3_hex', 'Hex'), ('E3_dome', 'Dome')]
fig, ax = plt.subplots(figsize=FULL); x = np.arange(len(cases)); wbar = 0.26
for i, m in enumerate(('ws', 'pbi', 'nsga2')):
    mu = [np.mean(A[c][m]['hv']) for c, _ in cases]; sd = [np.std(A[c][m]['hv']) for c, _ in cases]
    ax.bar(x + (i-1)*wbar, mu, wbar, yerr=sd, color=SER[m]['color'], edgecolor='0.0', lw=0.4, alpha=0.9, label=SER[m]['label'], error_kw=dict(lw=0.6, capsize=2))
ax.axhline(E['hv'], color='0.0', ls=':', lw=0.7); ax.text(-0.45, E['hv'] + 0.01, 'exact (fan)', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in cases]); ax.set_ylabel('Hypervolume (normalised, ref 1.05$^3$)'); ax.set_ylim(0.7, 1.15); ax.legend(loc='upper right', ncol=3); save(fig, 'fig06_a')
NUM['HV'] = {c: {m: dict(mean=float(np.mean(A[c][m]['hv'])), sd=float(np.std(A[c][m]['hv'])), n=len(A[c][m]['hv']), share=A[c][m]['share_combined'],
                        n_front=float(np.mean(A[c][m]['n_front'])), feas=float(np.mean(A[c][m]['feas'])), time=float(np.mean(A[c][m]['time']))) for m in ('ws', 'pbi', 'nsga2')} for c, _ in cases}
# ------------------------------------------------------------------ Fig 7 sq3 fronts by scenario
for name, key, ttl in (('fig07_a', 'E2_sq3_base', 'fossil-only, world-average steel'), ('fig07_b', 'E2_sq3_bio', 'biogenic storage credited'), ('fig07_c', 'E2_sq3_eaf', 'EAF steel')):
    fig, ax = plt.subplots(figsize=HALF)
    for m in ('ws', 'pbi', 'nsga2'):
        f = front_pts(A[key][m]['front_s0']); o = np.argsort(f[:, 0])
        ax.plot(f[o, 0], f[o, 2], marker=SER[m]['marker'], ls='none', mfc=SER[m]['mfc'], mec=SER[m]['color'], ms=3.5, label=SER[m]['label'])
    ax.set_yscale('log'); ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.legend(loc='upper right')
    if key == 'E2_sq3_bio': ax.axvline(0, color='0.0', lw=0.5); ax.set_ylim(bottom=1.2); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, name)
    NUM.setdefault('fronts', {})[key] = {m: dict(min_ec=float(front_pts(A[key][m]['front_s0'])[:, 0].min()), min_vw=float(front_pts(A[key][m]['front_s0'])[:, 2].min()),
                                            n=len(A[key][m]['front_s0'])) for m in ('ws', 'pbi', 'nsga2')}
# ------------------------------------------------------------------ Fig 8 material maps along the sq6 front (PBI seed 0)
fr = A['E3_sq6']['pbi']['front_s0']; f = front_pts(fr); o = np.argsort(f[:, 0]); fr = [fr[i] for i in o]; f = f[o]
picks = [0, int(np.argmin(np.abs(np.log(f[:, 2]) - np.mean(np.log(f[:, 2]))))), len(fr) - 1]
fig, ax = plt.subplots(figsize=HALF); ax.plot(f[:, 0], f[:, 2], 's', mfc=C['verm'], mec=C['verm'], ms=3, label='PBI-QUBO front')
for lab_, p in zip('ABC', picks): ax.annotate(lab_, (f[p, 0], f[p, 2]), textcoords='offset points', xytext=(5, 3), fontsize=8)
ax.set_yscale('log'); ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.legend(loc='upper right'); save(fig, 'fig08_a')
NUM['E8_designs'] = []
for name, p, lab_ in (('fig08_b', picks[0], 'A'), ('fig08_c', picks[1], 'B'), ('fig08_d', picks[2], 'C')):
    fig, ax = plt.subplots(figsize=SQ); plan(ax, G['sq6'], idx=fr[p]['idx'], cat=CAT16); lg = mat_legend(ax)
    save(fig, name, extra=[lg])
    d = fr[p]; NUM['E8_designs'].append(dict(label=lab_, ec=d['ec'], ee=d['ee'], vw=d['vw'], util=d['util'], sections=[CAT16[c]['label'] for c in d['idx']],
                                             n_timber=sum(CAT16[c]['mat'] == 'glulam' for c in d['idx'])))
# ------------------------------------------------------------------ Fig 9 standardisation
E4 = A['E4']; lams = sorted(set(d['lam'] for d in E4))
nt = {l: [d['n_types'] for d in E4 if d['lam'] == l] for l in lams}; ec4 = {l: [d['final']['ec'] for d in E4 if d['lam'] == l] for l in lams}; vw4 = {l: [d['final']['vw'] for d in E4 if d['lam'] == l] for l in lams}
fig, ax = plt.subplots(figsize=HALF); xs = np.arange(len(lams))
ax.errorbar(xs, [np.mean(nt[l]) for l in lams], yerr=[[np.mean(nt[l]) - min(nt[l]) for l in lams], [max(nt[l]) - np.mean(nt[l]) for l in lams]], marker='o', mfc='white', color=C['blue'], capsize=2, lw=0.9)
ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in lams]); ax.set_xlabel('Standardisation weight $\\lambda_t$'); ax.set_ylabel('Distinct section types used'); ax.set_ylim(0, 9); save(fig, 'fig09_a')
fig, ax = plt.subplots(figsize=HALF)
lam_w = np.array([0.4, 0.2, 0.4]); mks = {0.0: 'o', 0.02: 's', 0.05: '^', 0.1: 'D', 0.2: 'v', 0.4: 'P'}; cmap6 = plt.get_cmap('viridis')
for l in lams:
    pts = [(d['n_types'], float(lam_w @ np.array(d['final']['Fn']))) for d in E4 if d['lam'] == l]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mks[l], ls='none', mfc=cmap6(lams.index(l)/5.5), mec='0.2', ms=5.5, label=f'$\\lambda_t$ = {l}')
ax.set_xlabel('Distinct section types used'); ax.set_ylabel('Scalarised objective $\\lambda^\\top f$ (normalised)'); ax.legend(loc='lower left', ncol=2, fontsize=7.5); save(fig, 'fig09_b')
NUM['E4'] = {str(l): dict(types_mean=float(np.mean(nt[l])), types_min=min(nt[l]), types_max=max(nt[l]), ec_mean=float(np.mean(ec4[l])), vw_mean=float(np.mean(vw4[l]))) for l in lams}
# ------------------------------------------------------------------ Fig 10 reclaimed stock
E5 = A['E5']; v0 = sum((E5[k]['viol'] for k in E5 if '_c0_' in k), []); v1 = sum((E5[k]['viol'] for k in E5 if '_c1_' in k), [])
fig, ax = plt.subplots(figsize=HALF); bins = np.arange(-0.5, max(v0) + 1.5)
ax.hist(v0, bins=bins, color=C['verm'], edgecolor='0.0', alpha=0.7, label='No coupling'); ax.hist(v1, bins=bins, color=C['blue'], edgecolor='0.0', alpha=0.7, label='Lot-exclusivity coupling')
ax.set_xlabel('Duplicate lot assignments per design'); ax.set_ylabel('Designs evaluated'); ax.legend(loc='upper right'); save(fig, 'fig10_a')
fig, ax = plt.subplots(figsize=HALF)
for k, mk, col, lab in ((next(k for k in E5 if '_c1_s0' in k), 's', '0.0', 'Coupled front (feasible)'),):
    fr5 = E5[k]['front']; f5 = front_pts(fr5); o = np.argsort(f5[:, 0]); nre = np.array([p['n_reused'] for p in fr5])[o]
    for i in range(len(o)):
        ax.plot(f5[o[i], 0], f5[o[i], 2], marker='s', ms=3 + 1.3*nre[i], mfc='white', mec=C['green'], ls='none')
    ax.plot([], [], 's', mfc='white', mec=C['green'], ms=3, label='0 reclaimed lots'); ax.plot([], [], 's', mfc='white', mec=C['green'], ms=3 + 1.3*4, label='4 reclaimed lots')
base6 = front_pts(A['E3_sq6']['pbi']['front_s0']); o = np.argsort(base6[:, 0]); ax.plot(base6[o, 0], base6[o, 2], '.', color=C['verm'], ms=4, label='New material only')
ax.set_yscale('log'); ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.legend(loc='upper right', fontsize=7.5); save(fig, 'fig10_b')
k1 = [k for k in E5 if '_c1_' in k]; k0 = [k for k in E5 if '_c0_' in k]
NUM['E5'] = dict(viol_uncoupled_mean=float(np.mean(v0)), viol_uncoupled_share=float(np.mean(np.array(v0) > 0)), viol_coupled=int(sum(v1)), n_eval=len(v0)//3,
                 hv_coupled=[E5[k]['hv'] for k in k1], front_n=[len(E5[k]['front']) for k in k1], max_reused=max(p['n_reused'] for k in k1 for p in E5[k]['front']),
                 min_ec_coupled=min(p['ec'] for k in k1 for p in E5[k]['front']), min_ec_new=float(base6[:, 0].min()))
# ------------------------------------------------------------------ Fig 11 robust fronts
E6 = A['E6']
fig, ax = plt.subplots(figsize=HALF)
for b, mk, col in ((0.0, 'o', C['blue']), (1.0, 's', C['verm']), (2.0, '^', C['green'])):
    fr6 = sum((E6[k]['front'] for k in E6 if E6[k]['beta'] == b), []); f6 = front_pts(fr6); o = np.argsort(f6[:, 0])
    ax.plot(f6[o, 0], f6[o, 2], marker=mk, ls='none', mfc='white', mec=col, ms=3.2, label=f'$\\beta$ = {b:.0f}')
ax.set_yscale('log'); ax.set_xlabel('Mean embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.legend(loc='upper right'); save(fig, 'fig11_a')
fig, ax = plt.subplots(figsize=HALF)
for b, mk, col in ((0.0, 'o', C['blue']), (1.0, 's', C['verm']), (2.0, '^', C['green'])):
    fr6 = sum((E6[k]['front'] for k in E6 if E6[k]['beta'] == b), [])
    ax.plot([p['ec'] for p in fr6], [p['ecsd'] for p in fr6], marker=mk, ls='none', mfc='white', mec=col, ms=3.2, label=f'$\\beta$ = {b:.0f}')
ax.set_xlabel('Mean embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Carbon standard deviation (kgCO$_2$e)'); ax.legend(loc='upper left'); save(fig, 'fig11_b')
NUM['E6'] = {str(b): dict(n_front=sum(len(E6[k]['front']) for k in E6 if E6[k]['beta'] == b), sd_max=max(p['ecsd'] for k in E6 if E6[k]['beta'] == b for p in E6[k]['front']),
                          sd_mean=float(np.mean([p['ecsd'] for k in E6 if E6[k]['beta'] == b for p in E6[k]['front']])),
                          cv_mean=float(np.mean([p['ecsd']/p['ec'] for k in E6 if E6[k]['beta'] == b for p in E6[k]['front']]))) for b in (0.0, 1.0, 2.0)}
# ------------------------------------------------------------------ Fig 12 creep / SLS
E7 = A['E7']
fig, ax = plt.subplots(figsize=HALF)
for c, mk, col, lab in ((0, 'o', C['blue'], 'Instantaneous ($k_{def}$ = 0)'), (1, 's', C['verm'], 'Long-term ($k_{def}$ = 0.6)')):
    fr7 = sum((E7[k]['front'] for k in E7 if E7[k]['creep'] == c), []); f7 = front_pts(fr7); o = np.argsort(f7[:, 0])
    ax.plot(f7[o, 0], f7[o, 2], marker=mk, ls='none', mfc='white', mec=col, ms=3.2, label=lab)
ax.set_yscale('log'); ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Virtual work $F^\\top u$ (J)'); ax.legend(loc='upper right'); save(fig, 'fig12_a')
fig, ax = plt.subplots(figsize=HALF)
for c, mk, col, lab in ((0, 'o', C['blue'], 'Instantaneous'), (1, 's', C['verm'], 'Long-term')):
    fr7 = sum((E7[k]['front'] for k in E7 if E7[k]['creep'] == c), [])
    ax.plot([p['ec'] for p in fr7], [p['umax']*1e3 for p in fr7], marker=mk, ls='none', mfc='white', mec=col, ms=3.2, label=lab)
ax.axhline(E7[list(E7)[0]]['umax']*1e3, color='0.0', lw=0.6, ls='--'); ax.text(ax.get_xlim()[1]*0.98, E7[list(E7)[0]]['umax']*1e3*1.03, 'L/250', ha='right', fontsize=7.5)
ax.set_xlabel('Embodied carbon (kgCO$_2$e)'); ax.set_ylabel('Max. displacement (mm)'); ax.legend(loc='upper right'); save(fig, 'fig12_b')
NUM['E7'] = {str(c): dict(feas=sum(E7[k]['n_feas'] for k in E7 if E7[k]['creep'] == c), n_eval=sum(E7[k]['n_eval'] for k in E7 if E7[k]['creep'] == c),
                          min_ec=min(p['ec'] for k in E7 if E7[k]['creep'] == c for p in E7[k]['front']), n_front=sum(len(E7[k]['front']) for k in E7 if E7[k]['creep'] == c),
                          n_timber_mean=float(np.mean([sum(CAT16[i]['mat'] == 'glulam' for i in p['idx']) for k in E7 if E7[k]['creep'] == c for p in E7[k]['front']]))) for c in (0, 1)}
NUM['E7']['umax_mm'] = E7[list(E7)[0]]['umax']*1e3
# ------------------------------------------------------------------ Fig 13 convergence trace and cost
tr = np.array(A['E3_sq6']['ws']['trace_s0']); W15 = [(a/4, b/4, 1-(a+b)/4) for a in range(5) for b in range(5-a)]
fig, ax = plt.subplots(figsize=HALF); pos = 0; segs = []
for wi, lam in enumerate(W15[:6]):
    n = 11; seg = tr[pos:pos+n]; pos += n
    obj = seg[:, 4:7] @ np.array(lam); segs.append(obj)
    ax.plot(np.arange(n), obj, marker='o', ms=2.5, lw=0.9, color=plt.get_cmap('viridis')(wi/5.5), mfc='white', label=f'$\\lambda$ = ({lam[0]:.2f}, {lam[1]:.2f}, {lam[2]:.2f})')
ax.set_xlabel('Iteration (one real FE solve each)'); ax.set_ylabel('Scalarised objective $\\lambda^\\top f$ (normalised)'); lg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.5); save(fig, 'fig13_a', extra=[lg])
NUM['conv'] = dict(first_iter_mean=float(np.mean([s_[0] for s_ in segs])), last_iter_mean=float(np.mean([s_[-1] for s_ in segs])), min_iter_mean=float(np.mean([s_.min() for s_ in segs])))
fig, ax = plt.subplots(figsize=HALF); xs = np.arange(len(cases))
for i, m in enumerate(('ws', 'pbi', 'nsga2')):
    ax.bar(xs + (i-1)*wbar, [np.mean(A[c][m]['time']) for c, _ in cases], wbar, color=SER[m]['color'], edgecolor='0.0', lw=0.4, alpha=0.9, label=SER[m]['label'])
ax.set_xticks(xs); ax.set_xticklabels([n for _, n in cases], rotation=30, ha='right'); ax.set_ylabel('Wall time per run (s)'); ax.set_yscale('log'); ax.legend(loc='upper left'); save(fig, 'fig13_b')
NUM['feas_share'] = {c: {m: float(np.mean(A[c][m]['feas'])) for m in ('ws', 'pbi', 'nsga2')} for c, _ in cases}
NUM['bounds_fan'] = dict(lo=E['lo'], hi=E['hi'])
json.dump(NUM, open(ROOT / 'data' / 'manuscript_numbers.json', 'w'), indent=1, default=float)
print('panels:', len(list(OUT.glob('*.pdf'))))
