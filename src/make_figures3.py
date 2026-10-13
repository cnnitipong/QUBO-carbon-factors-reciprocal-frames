"""Colour panels added in v3 (fig22 to fig25): encoding and trust region, displacement and utilisation maps,
3-D objective space, catalogue choice along the front, per-seed hypervolume. Run after make_figures2.py."""
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pathlib import Path
from figstyle import save, SER, HALF, FULL, SQ, OUT, C, MAT
import mo2_lib as L
from sizing import element_geom_cache, group_element_indices
ROOT = Path(__file__).resolve().parents[1]
A = json.load(open(ROOT / 'data' / 'aggregate_v2.json')); G = A['geom']
NUM = json.load(open(ROOT / 'data' / 'manuscript_numbers.json'))
RES = ROOT / 'data' / 'results'
cat = L.catalogue(); ns = len(cat); CAT16 = [c for c in G['catalogue'] if c['mat'] != 'reused']
def util_per_element(frm, m, idx):
    u = np.zeros(len(m.elements))
    for e, el in enumerate(m.elements):
        if el['kind'] == 'link': u[e] = np.nan; continue
        sp = cat[idx[el['group']]]; f = frm.element_end_forces_local(e); N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
        u[e] = (N/sp['A'] + (My+Mz)*sp['D']/2/sp['Iy'])/sp['fd']
    return u
# ------------------------------------------------------------------ Fig 22: one-hot encoding, move limit and gate for the hexagonal nexorade
mh, loadh, _ = L.model('hex'); GLh = L.group_len(mh, L.lengths(mh)); ng = mh.n_groups; cur = [8]*ng
r0 = L.evaluate(mh, cat, cur, loadh, GLh); frm = r0['frm']
Ac = np.array([c['A'] for c in cat]); Dc = np.array([c['D'] for c in cat]); Ic = np.array([c['Iy'] for c in cat]); fdc = np.array([c['fd'] for c in cat])
UT = np.zeros((ng, ns)); groups = group_element_indices(mh)
for k, eis in groups.items():
    for ei in eis:
        f = frm.element_end_forces_local(ei); N = max(abs(f[0]), abs(f[6])); My = max(abs(f[4]), abs(f[10])); Mz = max(abs(f[5]), abs(f[11]))
        UT[k] = np.maximum(UT[k], (N/Ac + (My+Mz)*Dc/2/Ic)/fdc)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for k in range(ng):
    for c in range(ns):
        inwin = abs(c - cur[k]) <= 3; blocked = UT[k, c] > 1 and UT[k].min() <= 1
        face = MAT[cat[c]['mat']] if inwin else '#EDEDED'
        alpha = 0.25 if blocked and inwin else (0.85 if inwin else 1.0)
        ax.add_patch(Rectangle((c - 0.5, k - 0.5), 1, 1, facecolor=face, edgecolor='white', lw=0.8, alpha=alpha))
        if blocked: ax.text(c, k, 'x', ha='center', va='center', fontsize=7, color='0.2')
    ax.add_patch(Rectangle((cur[k] - 0.5, k - 0.5), 1, 1, facecolor='none', edgecolor='0.0', lw=1.6))
    ax.add_patch(Rectangle((cur[k] - 3.5, k - 0.5), 7, 1, facecolor='none', edgecolor='0.0', lw=0.6, ls='--'))
ax.set_xlim(-0.5, ns - 0.5); ax.set_ylim(ng - 0.5, -0.5); ax.set_xticks(range(ns)); ax.set_xticklabels([c['label'].replace('CHS ', 'S ').replace('GL ', 'T ') for c in cat], rotation=60, ha='right', fontsize=6.5)
ax.set_yticks(range(ng)); ax.set_yticklabels([f'$g_{k+1}$' for k in range(ng)]); ax.set_xlabel('Catalogue option $c$ (one binary $x_{kc}$ per cell)'); ax.set_ylabel('Member group $k$')
for sp in ax.spines.values(): sp.set_visible(False)
h = [Rectangle((0, 0), 1, 1, fc=MAT['steel'], alpha=0.85, label='Steel option in window'), Rectangle((0, 0), 1, 1, fc=MAT['glulam'], alpha=0.85, label='Glulam option in window'),
     Rectangle((0, 0), 1, 1, fc='#EDEDED', label='Outside move limit'), Line2D([], [], color='0.0', lw=1.6, label='Current option $c_k$'), Line2D([], [], color='0.0', lw=0.6, ls='--', label='Move-limit window'),
     Line2D([], [], marker='x', color='0.2', ls='none', label='Blocked by stress gate')]
lg = ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=7); save(fig, 'fig22_a', extra=[lg])
NUM['fig22'] = dict(window=7, n_in_window=int(sum(min(ns, cur[k] + 4) - max(0, cur[k] - 3) for k in range(ng))), n_blocked_in_window=int(sum(1 for k in range(ng) for c in range(ns) if abs(c - cur[k]) <= 3 and UT[k, c] > 1)))
# ------------------------------------------------------------------ Fig 23: displacement field and utilisation maps of two 6 x 6 front designs
m6, load6, _ = L.model('sq6'); GL6 = L.group_len(m6, L.lengths(m6)); nodes6 = np.array(m6.nodes)
fr = sorted(A['E3_sq6']['pbi']['front_s0'], key=lambda d: d['ec']); dA, dC = fr[0], fr[-1]
def disp_map(ax, m, frm, title):
    nodes = np.array(m.nodes); w = -frm.U.reshape(len(nodes), 6)[:, 2]*1e3
    for e in m.elements:
        if e['kind'] == 'link': continue
        p, q = nodes[e['i']], nodes[e['j']]; ax.plot([p[0], q[0]], [p[1], q[1]], color='0.6', lw=0.5, zorder=1)
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=w, cmap='viridis', s=14, zorder=2, edgecolors='none')
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=8.5); return sc
def util_map(ax, m, frm, idx, title):
    nodes = np.array(m.nodes); u = util_per_element(frm, m, idx); norm = Normalize(0, 1); cm = plt.get_cmap('RdYlBu_r')
    for e, el in enumerate(m.elements):
        if el['kind'] == 'link': continue
        p, q = nodes[el['i']], nodes[el['j']]; sp = cat[idx[el['group']]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=cm(norm(u[e])), lw=0.8 + 2.4*sp['mass']/max(c['mass'] for c in cat), solid_capstyle='round')
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=8.5); return plt.cm.ScalarMappable(norm=norm, cmap=cm), u
rA = L.evaluate(m6, cat, dA['idx'], load6, GL6); rC = L.evaluate(m6, cat, dC['idx'], load6, GL6)
fig, ax = plt.subplots(figsize=SQ); sc = disp_map(ax, m6, rA['frm'], f"Design A: {dA['ec']:.0f} kgCO$_2$e, {dA['vw']:.0f} J"); cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.set_label('Vertical displacement (mm)', fontsize=8); save(fig, 'fig23_a')
fig, ax = plt.subplots(figsize=SQ); sc = disp_map(ax, m6, rC['frm'], f"Design C: {dC['ec']:.0f} kgCO$_2$e, {dC['vw']:.0f} J"); cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.set_label('Vertical displacement (mm)', fontsize=8); save(fig, 'fig23_b')
fig, ax = plt.subplots(figsize=SQ); sm, uA = util_map(ax, m6, rA['frm'], dA['idx'], 'Design A: utilisation'); cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label('Utilisation', fontsize=8); save(fig, 'fig23_c')
fig, ax = plt.subplots(figsize=SQ); sm, uC = util_map(ax, m6, rC['frm'], dC['idx'], 'Design C: utilisation'); cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label('Utilisation', fontsize=8); save(fig, 'fig23_d')
NUM['fig23'] = dict(A=dict(ec=dA['ec'], vw=dA['vw'], umax_mm=rA['umax']*1e3, util_max=float(np.nanmax(uA)), util_mean=float(np.nanmean(uA)), share_over_06=float(np.nanmean(uA > 0.6))),
                    C=dict(ec=dC['ec'], vw=dC['vw'], umax_mm=rC['umax']*1e3, util_max=float(np.nanmax(uC)), util_mean=float(np.nanmean(uC)), share_over_06=float(np.nanmean(uC > 0.6))))
# ------------------------------------------------------------------ Fig 24: three-dimensional objective space on the 3 x 3 nexorade
fig = plt.figure(figsize=(3.3, 3.1)); ax = fig.add_subplot(111, projection='3d')
for meth in ('ws', 'pbi', 'nsga2'):
    pts = []
    for s in range(5):
        r = json.load(open(RES / f'E2_sq3_base_{meth}_s{s}.json')); pts += [x['Fn'] for x in r['front']]
    pts = np.array(pts); ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c=SER[meth]['color'], marker=SER[meth]['marker'], depthshade=False, label=SER[meth]['label'], alpha=0.8)
ax.scatter([0], [0], [0], s=40, c='k', marker='*', label='Utopia point')
ax.set_xlabel('Carbon (norm.)', fontsize=7.5, labelpad=-2); ax.set_ylabel('Energy (norm.)', fontsize=7.5, labelpad=-2); ax.set_zlabel('Virtual work (norm.)', fontsize=7.5, labelpad=-2)
ax.tick_params(labelsize=6.5, pad=-2); ax.view_init(elev=22, azim=-55); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
lg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7); save(fig, 'fig24_a', extra=[lg])
fig = plt.figure(figsize=(3.3, 3.1)); ax = fig.add_subplot(111, projection='3d')
pts = np.array([x['Fn'] for s in range(5) for x in json.load(open(RES / f'E2_sq3_base_pbi_s{s}.json'))['front']])
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, c='0.7', depthshade=False, alpha=0.6, label='PBI-QUBO front')
for lam, col in (((0.4, 0.2, 0.4), C['blue']), ((0.2, 0.2, 0.6), C['verm']), ((0.6, 0.2, 0.2), C['green'])):
    d = np.array(lam)/np.linalg.norm(lam); ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=col, lw=1.2, ls='--')
    proj = pts @ d; perp = np.linalg.norm(pts - np.outer(proj, d), axis=1); i = (proj + 0.3*perp**2).argmin()
    foot = proj[i]*d; ax.plot([pts[i, 0], foot[0]], [pts[i, 1], foot[1]], [pts[i, 2], foot[2]], color=col, lw=1.0)
    ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], s=30, c=col, marker='s', depthshade=False)
ax.scatter([0], [0], [0], s=40, c='k', marker='*')
ax.set_xlabel('Carbon (norm.)', fontsize=7.5, labelpad=-2); ax.set_ylabel('Energy (norm.)', fontsize=7.5, labelpad=-2); ax.set_zlabel('Virtual work (norm.)', fontsize=7.5, labelpad=-2)
ax.tick_params(labelsize=6.5, pad=-2); ax.view_init(elev=22, azim=-55); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
h = [Line2D([], [], color='0.7', marker='o', ls='none', label='PBI-QUBO front'), Line2D([], [], color='0.3', ls='--', label='Weight rays'), Line2D([], [], color='0.3', marker='s', ls='-', label='Selected design, perpendicular')]
lg = ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7); save(fig, 'fig24_b', extra=[lg])
# ------------------------------------------------------------------ Fig 25: catalogue choice per group along the 6 x 6 front, and per-seed hypervolume
fr6 = sorted(sum((json.load(open(RES / f'E3_sq6_base_pbi_s{s}.json'))['front'] for s in range(3)), []), key=lambda d: d['ec'])
M = np.array([d['idx'] for d in fr6]).T  # groups x designs
steel_rank = {c: i for i, c in enumerate([j for j, s in enumerate(cat) if s['mat'] == 'steel'])}; glu_rank = {c: i for i, c in enumerate([j for j, s in enumerate(cat) if s['mat'] == 'glulam'])}
img = np.zeros(M.shape + (3,))
cm_s = LinearSegmentedColormap.from_list('s', ['#CFE3F5', C['blue']]); cm_t = LinearSegmentedColormap.from_list('t', ['#FBE3B8', C['verm']])
for k in range(M.shape[0]):
    for j in range(M.shape[1]):
        c = M[k, j]; img[k, j] = (cm_s(steel_rank[c]/7) if cat[c]['mat'] == 'steel' else cm_t(glu_rank[c]/7))[:3]
fig, ax = plt.subplots(figsize=(6.4, 2.6)); ax.imshow(img, aspect='auto', interpolation='nearest')
ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f'$g_{{{k+1}}}$' for k in range(M.shape[0])]); ax.set_xlabel(f'Front designs of the $6 \\times 6$ nexorade, three seeds pooled, sorted by carbon ({fr6[0]["ec"]:.0f} to {fr6[-1]["ec"]:.0f} kgCO$_2$e)'); ax.set_ylabel('Member group')
ax.set_xticks([0, M.shape[1]//4, M.shape[1]//2, 3*M.shape[1]//4, M.shape[1]-1])
for sp in ax.spines.values(): sp.set_visible(False)
h = [Rectangle((0, 0), 1, 1, fc='#CFE3F5', label='Steel, lightest'), Rectangle((0, 0), 1, 1, fc=C['blue'], label='Steel, heaviest'), Rectangle((0, 0), 1, 1, fc='#FBE3B8', label='Glulam, lightest'), Rectangle((0, 0), 1, 1, fc=C['verm'], label='Glulam, heaviest')]
lg = ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=7.5); save(fig, 'fig25_a', extra=[lg])
share_by_group = (np.vectorize(lambda c: cat[c]['mat'] == 'glulam')(M)).mean(1)
NUM['fig25'] = dict(n_designs=int(M.shape[1]), glulam_share_by_group=[float(x) for x in share_by_group], always_timber=[int(k+1) for k in range(M.shape[0]) if share_by_group[k] > 0.95], mostly_steel=[int(k+1) for k in range(M.shape[0]) if share_by_group[k] < 0.5])
cases = [('E1_fan', 'Fan'), ('E2_sq3_base', 'Sq3 base'), ('E2_sq3_bio', 'Sq3 bio.'), ('E2_sq3_eaf', 'Sq3 EAF'), ('E3_sq6', 'Sq6'), ('E3_hex', 'Hex'), ('E3_dome', 'Dome')]
fig, ax = plt.subplots(figsize=FULL); x = np.arange(len(cases))
for i, m in enumerate(('ws', 'pbi', 'nsga2')):
    for j, (c, _) in enumerate(cases):
        hv = A[c][m]['hv']; ax.plot(np.full(len(hv), j + (i-1)*0.25) + np.linspace(-0.05, 0.05, len(hv)), hv, marker=SER[m]['marker'], ls='none', mfc=SER[m]['mfc'], mec=SER[m]['color'], ms=4, label=SER[m]['label'] if j == 0 else None)
        ax.plot([j + (i-1)*0.25 - 0.09, j + (i-1)*0.25 + 0.09], [np.mean(hv)]*2, color=SER[m]['color'], lw=1.2)
ax.axhline(A['E1_exact']['hv'], color='k', ls=':', lw=0.7); ax.text(-0.4, A['E1_exact']['hv'] + 0.006, 'exact (fan)', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in cases]); ax.set_ylabel('Hypervolume per seed'); ax.legend(loc='lower left', ncol=3); save(fig, 'fig25_b')
json.dump(NUM, open(ROOT / 'data' / 'manuscript_numbers.json', 'w'), indent=1, default=float)
print('panels:', len(list(OUT.glob('*.pdf'))), NUM['fig22'], NUM['fig23'], NUM['fig25'])
