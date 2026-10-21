"""Compact four-panel evidence figure for the active ICASSP paper."""
import json

import numpy as np

from paper_plot_style import COLORS, MARKERS, ROOT, plt, save


statistics = json.loads((ROOT / "evidence_20260909/rl_statistics.json").read_text())
coverage = json.loads((ROOT / "evidence_20260909/independent_coverage.json").read_text())
methods = ["raw", "static_racf", "adaptive_racf", "pathwise_racf"]
labels = ["Raw", "Static", "Adaptive", "Envelope"]
families = ["mixed_iql", "mixed_cql", "mixed_bc_bootstrap", "expert_acc", "suboptimal_stress"]
family_labels = ["IQL", "CQL", "BC", "IDM", "Stress"]
conditions = ["nominal", "mass_shift", "slope_shift", "combined"]
condition_labels = ["Nom.", "Mass", "Slope", "Joint"]

fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
fig.subplots_adjust(left=0.067, right=0.992, bottom=0.25, top=0.76, wspace=0.51)

ax = axes[0]
for index, method in enumerate(methods):
    rows = [
        next(row for row in statistics["family_summaries"] if row["family"] == family and row["method"] == method)
        for family in families
    ]
    y = np.arange(len(families)) + (index - 1.5) * 0.16
    values = np.asarray([row["safety"] * 100.0 for row in rows])
    low, high = np.asarray([row["ci95"] for row in rows]).T * 100.0
    ax.errorbar(values, y, xerr=[values - low, high - values], fmt=MARKERS[index],
                color=COLORS[index], markersize=3.3, capsize=1.2, linewidth=0.8, label=labels[index])
ax.set_yticks(range(len(families)), family_labels)
ax.invert_yaxis()
ax.set_xlim(-3, 103)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("(a) Policy family")
ax.grid(axis="x", color="#dddddd", linewidth=0.45)

ax = axes[1]
x = np.arange(len(conditions))
for index, method in enumerate(methods):
    rows = [
        next(row for row in statistics["condition_summaries"] if row["condition"] == condition and row["method"] == method)
        for condition in conditions
    ]
    ax.plot(x, [row["safety"] * 100.0 for row in rows], marker=MARKERS[index],
            color=COLORS[index], linewidth=0.9, markersize=3.5)
ax.set_xticks(x, condition_labels)
ax.set_ylim(15, 103)
ax.set_yticks([20, 60, 100])
ax.set_ylabel("Safe (\%)")
ax.set_xlabel("(b) Condition")
ax.grid(axis="y", color="#dddddd", linewidth=0.45)

ax = axes[2]
x = np.arange(len(coverage["datasets"]))
values = np.asarray([row["coverage"] * 100.0 for row in coverage["datasets"]])
low, high = np.asarray([row["ci95"] for row in coverage["datasets"]]).T * 100.0
ax.errorbar(x, values, yerr=[values - low, high - values], fmt="o", color=COLORS[1],
            capsize=2.0, linewidth=0.9, markersize=4.0)
ax.axhline(90.0, color="#777777", linestyle="--", linewidth=0.8)
ax.set_xticks(x, condition_labels)
ax.set_ylim(84, 101.5)
ax.set_yticks([85, 90, 95, 100])
ax.set_ylabel("Coverage (\%)")
ax.set_xlabel("(c) Coverage")
ax.grid(axis="y", color="#dddddd", linewidth=0.45)

ax = axes[3]
for index, row in enumerate(statistics["summaries"][1:], start=1):
    x_value = row["jerk_p95"]
    y_value = row["episode_safe"] * 100.0
    latency = row["filter_only"]["warm_p95_ms"]
    ax.scatter(x_value, y_value, s=36, marker=MARKERS[index], color=COLORS[index])
    offset = {"Static": (3, -10), "Adaptive": (-5, 6), "Envelope": (-32, 6)}[row["label"]]
    ax.annotate(f"{latency:.1f}", (x_value, y_value),
                xytext=offset, textcoords="offset points", fontsize=8)
ax.set_xlim(2.95, 3.48)
ax.set_ylim(87, 97)
ax.set_yticks([88, 92, 96])
ax.set_ylabel("Safe (\%)")
ax.set_xlabel(r"(d) Jerk P95 (m/s$^3$)")
ax.grid(color="#dddddd", linewidth=0.45)

for ax in axes[1:]:
    ax.tick_params(axis="x", labelrotation=28)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False,
           handlelength=1.0, columnspacing=1.0, handletextpad=0.3)
save(fig, "fig2_evidence")
