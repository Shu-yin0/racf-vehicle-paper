"""Shared print-scale styling for the ICASSP manuscript."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#555555', '#0072B2', '#009E73', '#D55E00']
MARKERS = ['o', 's', '^', 'D']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'stix', 'savefig.dpi': 300})


def save(fig, name):
    fig.savefig(ROOT / 'figures' / (name + '.pdf'))
    fig.savefig(ROOT / 'figures' / (name + '.png'), dpi=300)
    plt.close(fig)
