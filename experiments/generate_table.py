"""Combine verified partial-rollout evidence into a scoped LaTeX table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(coverage_path: Path, safety_path: Path, runtime_path: Path, output_path: Path) -> None:
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    safety = json.loads(safety_path.read_text(encoding="utf-8"))["rows"]
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime_payload.get("status") != "mock test":
        raise ValueError("runtime input must be explicitly labelled mock test")
    runtime = runtime_payload["records"]
    latex_break = r" \\"
    lines = [
        "% Generated from verified partial-rollout and fixed-state mock-test artifacts.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        f"Condition & Coverage & Avg. raw violation & Avg. static violation & Avg. adaptive violation {latex_break}",
        r"\midrule",
    ]
    for condition, value in coverage["conditions"].items():
        raw = [row["violation_rate"] for row in safety if row["condition"] == condition and row["method"] == "raw"]
        static = [row["violation_rate"] for row in safety if row["condition"] == condition and row["method"] == "static_conformal"]
        adaptive = [row["violation_rate"] for row in safety if row["condition"] == condition and row["method"] == "adaptive_conformal"]
        lines.append(
            f"{condition.replace('_', ' ')} & {value['coverage_mean']:.3f} & "
            f"{sum(raw)/len(raw):.3f} & {sum(static)/len(static):.3f} & "
            f"{sum(adaptive)/len(adaptive):.3f} {latex_break}"
        )
    lines.extend(
        [
            r"\midrule",
            "Fixed-state latency (ms; mock test) & "
            f"\\multicolumn{{4}}{{r}}{{static {runtime['static_conformal']['mean_ms']:.3f}; "
            f"adaptive {runtime['adaptive_conformal']['mean_ms']:.3f}; "
            f"mpc {runtime['mpc']['mean_ms']:.3f}}} {latex_break}",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.coverage, args.safety, args.runtime, args.output)


if __name__ == "__main__":
    main()
