"""Plot coverage results without implying a full benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d"]


def _merge_conditions(results_paths: list[Path]) -> tuple[dict, dict]:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in results_paths]
    if not results:
        raise ValueError("at least one coverage result is required")
    base = results[0]
    merged: dict[str, dict] = {}
    for result in results:
        if result["target_coverage"] != base["target_coverage"]:
            raise ValueError("coverage results use different target coverage")
        if result["dynamics_model"] != base["dynamics_model"]:
            raise ValueError("coverage results use different dynamics models")
        for name, values in result["conditions"].items():
            if name in merged:
                raise ValueError(f"duplicate coverage condition: {name}")
            merged[name] = values
    return base, merged


def plot(results_paths: list[Path], output_path: Path, shift_path: Path | None = None) -> None:
    result, conditions = _merge_conditions(results_paths)
    names = list(conditions)
    means = [conditions[name]["coverage_mean"] for name in names]
    stds = [conditions[name]["coverage_std"] for name in names]
    if shift_path is None:
        fig, coverage_axis = plt.subplots(figsize=(5.6, 3.6))
        shift_axis = None
    else:
        fig, (coverage_axis, shift_axis) = plt.subplots(1, 2, figsize=(8.6, 3.45))
    x = np.arange(len(names))
    coverage_axis.bar(x, means, yerr=stds, capsize=4, color=COLORS[: len(names)], edgecolor="black", linewidth=0.7)
    coverage_axis.axhline(result["target_coverage"], color="black", linestyle="--", linewidth=1.2, label="target 1-alpha")
    coverage_axis.set_xticks(x, [name.replace("_", " ").title() for name in names], rotation=20, ha="right")
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_ylabel("Empirical coverage")
    coverage_axis.set_title("One-step coverage (partial rollout)")
    coverage_axis.legend(frameon=False)
    coverage_axis.grid(axis="y", alpha=0.25)
    if shift_axis is not None:
        shift = json.loads(shift_path.read_text(encoding="utf-8"))
        mmd = [shift["conditions"][name]["mmd_rbf_squared"] for name in names]
        shift_axis.bar(x, mmd, color=COLORS[: len(names)], edgecolor="black", linewidth=0.7)
        shift_axis.set_xticks(x, [name.replace("_", " ").title() for name in names], rotation=20, ha="right")
        shift_axis.set_ylabel("RBF MMD-squared")
        shift_axis.set_title("Residual shift proxy (not TV)")
        shift_axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-results", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output, args.shift_results)


if __name__ == "__main__":
    main()
