"""Summarize paired pathwise qualification results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in ("policy_checkpoint_sha256", "condition", "scenario", "schedule_id", "seed", "trial_id"))


def _mcnemar(first: list[bool], second: list[bool]) -> dict[str, Any]:
    first_only = sum(a and not b for a, b in zip(first, second, strict=True))
    second_only = sum(not a and b for a, b in zip(first, second, strict=True))
    discordant = first_only + second_only
    p = 1.0 if discordant == 0 else float(
        binomtest(min(first_only, second_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    return {"first_only": first_only, "second_only": second_only, "discordant": discordant, "p_two_sided": min(1.0, p)}


def analyze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    methods = sorted({row["method"] for row in rows})
    by_method = {(row["method"], _key(row)): row for row in rows}
    summary = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n": len(subset),
                "episode_violations": sum(row["episode_any_controllable_safety_violation"] for row in subset),
                "episode_violation_rate": float(np.mean([row["episode_any_controllable_safety_violation"] for row in subset])),
                "collision_count": sum(row["collision"] for row in subset),
                "mean_projection_rate": float(np.mean([row["projection_rate"] for row in subset])),
                "mean_fallback_rate": float(np.mean([row["fallback_rate"] for row in subset])),
                "mean_runtime_ms_p95": float(np.mean([row["runtime_ms_p95"] for row in subset])),
                "pathwise_coverage_rate": float(np.mean([row["pathwise_residual_coverage"] for row in subset])),
            }
        )
    comparisons = []
    for baseline, candidate in (("static_conformal", "pathwise_static"), ("contextual_temporal_racf", "pathwise_temporal")):
        pairs = []
        for row in rows:
            if row["method"] != baseline:
                continue
            other = by_method.get((candidate, _key(row)))
            if other is not None:
                pairs.append((row, other))
        baseline_v = [bool(first["episode_any_controllable_safety_violation"]) for first, _ in pairs]
        candidate_v = [bool(second["episode_any_controllable_safety_violation"]) for _, second in pairs]
        comparisons.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "pairs": len(pairs),
                "mcnemar": _mcnemar(baseline_v, candidate_v),
                "mean_projection_delta": float(np.mean([second["projection_rate"] - first["projection_rate"] for first, second in pairs])),
                "mean_runtime_p95_delta_ms": float(np.mean([second["runtime_ms_p95"] - first["runtime_ms_p95"] for first, second in pairs])),
                "mean_fallback_delta": float(np.mean([second["fallback_rate"] - first["fallback_rate"] for first, second in pairs])),
            }
        )
    return {"protocol": payload["protocol"], "status": "partial rollout", "rows": len(rows), "methods": methods, "summary": summary, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
