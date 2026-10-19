"""Paper figure, parameterized to read this reproduction run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_figure(summary, output):
    d=json.loads(Path(summary).read_text())
    out=Path(output)
    periods = np.array([0.80, 1.31, 2.14, 3.49, 5.70]); radii = np.array([0.50, 0.78, 1.22, 1.92, 3.00])
    g = {}
    for k in ('cnn', 'bls'):
        a = np.full((5, 5), np.nan)
        for c in d['cells_descriptive'][k]:
            a[c['radius_index'], c['period_index']] = c['recovery']
        assert np.isfinite(a).all(); g[k] = 100*a
    ctrl = {k: 100*d['operating_points']['primary'][k]['control_trigger_rate'] for k in g}
    style = {'cnn': dict(color='#2a78d6', marker='o', label='CNN'),
             'bls': dict(color='#eb6834', marker='s', label='BLS')}
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True, sharey=True)
    panels = [(axes[0], radii, lambda k: g[k].mean(axis=1), r'Planet radius ($R_\oplus$)'),
              (axes[1], periods, lambda k: g[k].mean(axis=0), 'Orbital period (days)')]
    for ax, x, f, xl in panels:
        for k in ('cnn', 'bls'):
            y = f(k)
            ax.plot(x, y, lw=2, ms=6, mec='white', mew=1.5, solid_capstyle='round', **style[k])
            ax.annotate(style[k]['label'], (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color='#333333')
        ax.set_xscale('log'); ax.set_xticks(x, [f'{v:.2f}' for v in x]); ax.minorticks_off()
        ax.set_xlabel(xl); ax.set_ylim(0, 100); ax.set_xlim(x[0]/1.12, x[-1]*1.35)
        ax.grid(axis='y', color='#e3e3e3', lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.tick_params(length=2, color='#999999')
    axes[0].axhspan(min(ctrl.values()), max(ctrl.values()), color='#bdbdbd', alpha=0.35, lw=0)
    axes[0].annotate('control flag rates', (1.45, max(ctrl.values())), xytext=(0, 3), textcoords='offset points',
                     fontsize=7.5, color='#555555')
    axes[0].set_ylabel('Star-averaged recovery (%)')
    axes[0].set_title('Averaged over period', fontsize=9); axes[1].set_title('Averaged over radius', fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'recovery_lines.pdf'); fig.savefig(out/'recovery_lines.png', dpi=200)
    plt.close(fig)
