"""Plot paired safety episode summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(results_path: Path, output_path: Path) -> None:
    rows = json.loads(results_path.read_text(encoding="utf-8"))["rows"]
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    methods = ["raw", "static_conformal", "adaptive_conformal"]
    values = {method: [np.mean([row["violation_rate"] for row in rows if row["condition"] == condition and row["method"] == method]) for condition in conditions] for method in methods}
    x = np.arange(len(conditions))
    width = 0.24
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    for index, method in enumerate(methods):
        axis.bar(x + (index - 1) * width, values[method], width, label=method.replace("_", " ").title())
    axis.set_xticks(x, [name.replace("_", " ").title() for name in conditions])
    axis.set_ylabel("Violation rate")
    axis.set_title("ACC safety comparison (partial rollout)")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output)


if __name__ == "__main__":
    main()
