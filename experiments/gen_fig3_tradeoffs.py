"""Joint safety/intervention and runtime-stage comparison from frozen rows."""
import json
import numpy as np
from paper_plot_style import ROOT, COLORS, MARKERS, plt, save

p = json.loads((ROOT/'evidence_20260909/rl_statistics.json').read_text())
rows = p['summaries'][1:]
fig, axes = plt.subplots(1, 2, figsize=(7, 2.05))
fig.subplots_adjust(left=.085, right=.985, bottom=.27, top=.92, wspace=.43)
ax = axes[0]
for i, row in enumerate(rows, 1):
    ax.scatter(row['projection_rate']*100, row['episode_safe']*100,
               s=42, marker=MARKERS[i], color=COLORS[i])
    lo,hi = np.array(row['safe_ci95_fixed_policy_reset_cluster'])*100
    ax.plot([row['projection_rate']*100]*2,[lo,hi],color=COLORS[i],lw=1)
    ax.annotate(row['label'], (row['projection_rate']*100,row['episode_safe']*100),
                xytext=(-5,8) if i<3 else (-45,-17), textcoords='offset points', fontsize=9)
ax.set_xlabel('(a) Projected actions (%)')
ax.set_ylabel('Episode-safe rate (%)')
ax.set_xlim(6.44,6.85)
ax.set_ylim(86,98)
ax.grid(color='#dddddd',linewidth=.45)
ax = axes[1]
for i, row in enumerate(rows, 1):
    vals = [row['filter_only']['warm_p50_ms'],row['filter_only']['warm_p95_ms'],
            row['filter_only']['warm_p99_ms']]
    ax.plot([0,1,2],vals,marker=MARKERS[i],color=COLORS[i],lw=1,label=row['label'])
ax.set_xticks([0,1,2], ['P50','P95','P99'])
ax.set_ylabel('Filter latency (ms)')
ax.set_xlabel('(b) Mean within-episode quantiles')
ax.set_ylim(0,26)
ax.grid(axis='y',color='#dddddd',linewidth=.45)
ax.legend(loc='upper left',frameon=False,handlelength=1)
save(fig, 'fig3_tradeoffs')
