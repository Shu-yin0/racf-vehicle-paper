"""Independent paired analysis for the frozen pathwise v3 confirmatory run.

The unit of inference is one complete paired episode identified by policy,
condition, scenario, schedule, seed, and trial id.  Binary safety outcomes use
exact McNemar tests; continuous deltas use paired bootstrap intervals.  Holm
adjustment is applied to the frozen cell-level binary comparison family.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


METHODS = (
    "raw",
    "static_conformal",
    "contextual_temporal_racf",
    "pathwise_static",
    "pathwise_temporal",
)
PRIMARY_COMPARISONS = (
    ("static_conformal", "pathwise_static"),
    ("contextual_temporal_racf", "pathwise_temporal"),
)
CONTINUOUS_METRICS = (
    "violation_rate",
    "projection_rate",
    "fallback_rate",
    "mean_action_correction",
    "runtime_ms_p95",
)
RUNTIME_FIELDS = (
    "runtime_ms_mean",
    "runtime_ms_std",
    "runtime_ms_p50",
    "runtime_ms_p95",
    "runtime_ms_p99",
    "runtime_ms_max",
)
GROUP_FIELDS = ("policy_checkpoint_sha256", "condition", "scenario", "schedule_id")
PAIR_FIELDS = (*GROUP_FIELDS, "seed", "trial_id")


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] if field not in {"seed", "trial_id"} else int(row[field]) for field in PAIR_FIELDS)


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in GROUP_FIELDS)


def _exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, int | float]:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    first_only = int(np.sum(left_bool & ~right_bool))
    second_only = int(np.sum(~left_bool & right_bool))
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(first_only, second_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "p_two_sided": p_value,
    }


def _holm(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, float(p_value) * (total - index)))
        adjusted[name] = running
    return adjusted


def _paired_bootstrap(left: np.ndarray, right: np.ndarray, *, seed: int, draws: int = 10000) -> dict[str, float | int]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.ndim != 1 or differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("paired bootstrap inputs must be finite non-empty vectors")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    batch = 250
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        means[start : start + count] = differences[indices].mean(axis=1)
    return {
        "n": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def _runtime_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    result: dict[str, Any] = {"n": len(rows), "fields": {}}
    for field in RUNTIME_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"invalid runtime field {field}")
        result["fields"][field] = {
            "mean_ms": float(values.mean()),
            "std_ms": float(values.std()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(values.max()),
        }
    # Keep the most commonly cited row-P95 summary at a stable top-level key.
    result.update({f"{key}_ms_p95": value for key, value in {
        "mean": result["fields"]["runtime_ms_p95"]["mean_ms"],
        "std": result["fields"]["runtime_ms_p95"]["std_ms"],
        "p50": result["fields"]["runtime_ms_p95"]["p50_ms"],
        "p95": result["fields"]["runtime_ms_p95"]["p95_ms"],
        "p99": result["fields"]["runtime_ms_p95"]["p99_ms"],
        "max": result["fields"]["runtime_ms_p95"]["max_ms"],
    }.items()})
    return result


def _validate_pairing(rows: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, Any]]:
    by_pair: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_pair[_pair_key(row)][row["method"]] = row
    expected = set(METHODS)
    missing = [key for key, methods in by_pair.items() if set(methods) != expected]
    if missing:
        raise ValueError(f"incomplete paired methods for {len(missing)} episodes")
    initial_state_mismatches = []
    for key, methods in by_pair.items():
        reference = np.asarray(methods[METHODS[0]]["initial_state"], dtype=float)
        for method in METHODS[1:]:
            candidate = np.asarray(methods[method]["initial_state"], dtype=float)
            if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
                initial_state_mismatches.append(key)
                break
    if initial_state_mismatches:
        raise ValueError(f"initial-state mismatch for {len(initial_state_mismatches)} paired episodes")
    return by_pair, {
        "paired_trials": len(by_pair),
        "initial_state_mismatches": len(initial_state_mismatches),
        "methods": list(METHODS),
    }


def analyze(path: Path, *, bootstrap_draws: int = 10000, bootstrap_seed: int = 20260907) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "racf_pathwise_confirmatory_v3":
        raise ValueError("unexpected confirmatory protocol")
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("rows must be non-empty")
    by_pair, pairing = _validate_pairing(rows)
    seeds = {int(row["seed"]) for row in rows}
    if seeds != set(range(1250000, 1250400)):
        raise ValueError("confirmatory seed range mismatch")
    if {row.get("split") for row in rows} != {"confirmatory"}:
        raise ValueError("confirmatory split marker mismatch")
    selection_hashes = {row.get("development_selection_sha256") for row in rows}
    if len(selection_hashes) != 1:
        raise ValueError("selection provenance is not unique")

    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for methods in by_pair.values():
        for method in METHODS:
            by_method[method].append(methods[method])
    summary = []
    for method in METHODS:
        subset = by_method[method]
        episode = np.asarray([bool(row["episode_any_controllable_safety_violation"]) for row in subset], dtype=bool)
        summary.append(
            {
                "method": method,
                "n": len(subset),
                "episode_violations": int(episode.sum()),
                "episode_violation_rate": float(episode.mean()),
                "collision_count": int(sum(bool(row["collision"]) for row in subset)),
                "early_termination_count": int(sum(bool(row["early_termination"]) for row in subset)),
                "pathwise_coverage_count": int(sum(bool(row["pathwise_residual_coverage"]) for row in subset)),
                "pathwise_coverage_rate": float(np.mean([bool(row["pathwise_residual_coverage"]) for row in subset])),
                "mean_projection_rate": float(np.mean([float(row["projection_rate"]) for row in subset])),
                "mean_fallback_rate": float(np.mean([float(row["fallback_rate"]) for row in subset])),
                "mean_action_correction": float(np.mean([float(row["mean_action_correction"]) for row in subset])),
                "runtime": _runtime_summary(subset),
            }
        )

    global_comparisons = []
    cell_tests = []
    for baseline, candidate in PRIMARY_COMPARISONS:
        left_episode = np.asarray([bool(methods[baseline]["episode_any_controllable_safety_violation"]) for methods in by_pair.values()])
        right_episode = np.asarray([bool(methods[candidate]["episode_any_controllable_safety_violation"]) for methods in by_pair.values()])
        continuous = {}
        continuous["episode_violation_rate"] = _paired_bootstrap(
            np.asarray([float(methods[candidate]["episode_any_controllable_safety_violation"]) for methods in by_pair.values()]),
            np.asarray([float(methods[baseline]["episode_any_controllable_safety_violation"]) for methods in by_pair.values()]),
            seed=bootstrap_seed + (0 if candidate == "pathwise_static" else 100),
            draws=bootstrap_draws,
        )
        for metric_index, metric in enumerate(CONTINUOUS_METRICS):
            continuous[metric] = _paired_bootstrap(
                np.asarray([float(methods[candidate][metric]) for methods in by_pair.values()]),
                np.asarray([float(methods[baseline][metric]) for methods in by_pair.values()]),
                seed=bootstrap_seed + metric_index + (0 if candidate == "pathwise_static" else 100),
                draws=bootstrap_draws,
            )
        global_comparisons.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "pairs": len(by_pair),
                "mcnemar": _exact_mcnemar(left_episode, right_episode),
                "continuous_candidate_minus_baseline": continuous,
            }
        )
        for group in sorted({_group_key(methods[baseline]) for methods in by_pair.values()}):
            selected = [methods for methods in by_pair.values() if _group_key(methods[baseline]) == group]
            left = np.asarray([bool(methods[baseline]["episode_any_controllable_safety_violation"]) for methods in selected])
            right = np.asarray([bool(methods[candidate]["episode_any_controllable_safety_violation"]) for methods in selected])
            group_id = "|".join(map(str, group))
            cell_tests.append(
                {
                    "id": f"{candidate}_vs_{baseline}|{group_id}",
                    "baseline": baseline,
                    "candidate": candidate,
                    "policy_checkpoint_sha256": group[0],
                    "condition": group[1],
                    "scenario": group[2],
                    "schedule_id": group[3],
                    "pairs": len(selected),
                    "mcnemar": _exact_mcnemar(left, right),
                }
            )
    adjusted = _holm([(item["id"], float(item["mcnemar"]["p_two_sided"])) for item in cell_tests])
    for item in cell_tests:
        item["p_holm"] = adjusted[item["id"]]
        item["significant_holm_alpha_0_05"] = bool(item["p_holm"] < 0.05)

    return {
        "protocol": payload["protocol"],
        "status": "partial rollout confirmatory analysis",
        "scope": "ACC partial rollout; not a full benchmark",
        "selection": {
            "development_selection_sha256": next(iter(selection_hashes)),
            "fixed_B": payload.get("pathwise_q"),
            "fixed_L": payload.get("pathwise_safety_lipschitz"),
        },
        "design": {
            **pairing,
            "rows": len(rows),
            "policies": len({row["policy_checkpoint_sha256"] for row in rows}),
            "conditions": sorted({row["condition"] for row in rows}),
            "scenarios": sorted({row["scenario"] for row in rows}),
            "schedules": sorted({row["schedule_id"] for row in rows}),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "seed_count": len(seeds),
        },
        "summary": summary,
        "global_comparisons": global_comparisons,
        "cell_level_primary_tests": cell_tests,
        "holm_family": {
            "tests": len(cell_tests),
            "scope": "two frozen pathwise-vs-baseline method comparisons across policy-condition-scenario-schedule cells",
            "alpha": 0.05,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    result = analyze(args.input, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": result["design"]["rows"], "paired_trials": result["design"]["paired_trials"], "holm_tests": result["holm_family"]["tests"]}, indent=2))


if __name__ == "__main__":
    main()
