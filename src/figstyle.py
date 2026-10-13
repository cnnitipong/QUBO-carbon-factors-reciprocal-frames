"""Shared greyscale figure style (9 pt base, 8 pt ticks/legend)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / 'figures' / 'panels'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman'], 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'lines.linewidth': 1.0,
    'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'mathtext.fontset': 'dejavuserif'})
GREY = {'k': '0.0', 'd': '0.3', 'm': '0.55', 'l': '0.75', 'xl': '0.88'}
# Okabe-Ito colour-blind-safe palette
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', verm='#D55E00', purple='#CC79A7', sky='#56B4E9', yellow='#F0E442', black='#000000', grey='#7F7F7F')
MAT = dict(steel=C['blue'], glulam=C['orange'], reused=C['green'])
SCEN = [C['blue'], C['verm'], C['green'], C['purple']]
# method series: (color, marker, linestyle, hatch)
SER = {'exact': dict(color='#000000', marker=None, ls='-', label='Exact front'),
       'ws': dict(color='#0072B2', marker='o', ls='-', label='WS-QUBO', hatch='//', mfc='white'),
       'pbi': dict(color='#D55E00', marker='s', ls='--', label='PBI-QUBO', hatch='..', mfc='#D55E00'),
       'nsga2': dict(color='#009E73', marker='^', ls=':', label='NSGA-II', hatch='xx', mfc='white')}
HALF = (3.1, 2.5); FULL = (6.4, 3.0); SQ = (3.1, 3.1)
def save(fig, name, extra=None):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', bbox_extra_artists=extra or [])
    fig.savefig(OUT / f'{name}.png', dpi=130, bbox_inches='tight', bbox_extra_artists=extra or [])
    plt.close(fig)
