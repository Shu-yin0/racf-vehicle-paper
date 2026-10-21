"""Safety rates with reset-cluster intervals for the fixed policy families."""
import json
import numpy as np
from paper_plot_style import ROOT, COLORS, MARKERS, plt, save

p = json.loads((ROOT/'evidence_20260909/rl_statistics.json').read_text())
families = ['mixed_iql', 'mixed_cql', 'mixed_bc_bootstrap', 'expert_acc', 'suboptimal_stress']
methods = ['raw', 'static_racf', 'adaptive_racf', 'pathwise_racf']
fig, ax = plt.subplots(figsize=(3.38, 2.75))
fig.subplots_adjust(left=.24, right=.96, bottom=.21, top=.78)
for j, method in enumerate(methods):
    rows = [next(r for r in p['family_summaries'] if r['family']==f and r['method']==method) for f in families]
    y = np.arange(5) + (j-1.5)*.18
    val = np.array([r['safety']*100 for r in rows])
    low, high = np.array([r['ci95'] for r in rows]).T*100
    ax.errorbar(val, y, xerr=[val-low, high-val], fmt=MARKERS[j], color=COLORS[j],
                markersize=3.8, capsize=1.6, linewidth=.9, label=['Raw','Static','Adaptive','Envelope'][j])
ax.set_yticks(range(5), ['IQL (4)', 'CQL (3)', 'BC (3)', 'IDM (1)', 'Stress (1)'])
ax.invert_yaxis()
ax.set_xlim(-3, 103)
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('Episode-safe rate (%)')
ax.grid(axis='x', color='#dddddd', linewidth=.45)
ax.legend(loc='lower center', bbox_to_anchor=(.38,1.08), ncol=2, frameon=False,
          columnspacing=.8, handletextpad=.25)
save(fig, 'fig2_policy_families')
