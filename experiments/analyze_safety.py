"""Paired pilot statistics: bootstrap intervals, exact McNemar, and Holm."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def exact_mcnemar_pvalue(left: np.ndarray, right: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for paired boolean outcomes."""
    b = int(np.sum(left & ~right))
    c = int(np.sum(~left & right))
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, index) for index in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int = 0, draws: int = 10_000) -> dict[str, float]:
    differences = left - right
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(draws, len(differences)))].mean(axis=1)
    return {"mean_difference": float(differences.mean()), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975))}


def paired_sign_flip_pvalue(
    left: np.ndarray, right: np.ndarray, seed: int = 0, draws: int = 100_000
) -> float:
    """One-sided p-value for the preregistered direction left > right."""
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired samples must be one-dimensional and non-empty")
    observed = float(differences.mean())
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, differences.size))
    null_means = (signs * differences).mean(axis=1)
    return float((1 + np.count_nonzero(null_means >= observed)) / (draws + 1))


def holm(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[name] = running
    return adjusted


def paired_inference_eligibility(
    raw: dict[str, dict], filtered: dict[str, dict], paired: list[str]
) -> dict[str, object]:
    """Reject pseudo-replication before calculating paired inferential statistics."""
    reasons: list[str] = []
    all_rows = [raw[key] for key in paired] + [filtered[key] for key in paired]
    if not all(row.get("randomized_reset") is True for row in all_rows):
        reasons.append("randomized_reset")
    if not all("initial_state" in row for row in all_rows):
        reasons.append("missing_initial_state")
    else:
        raw_initial_states = {tuple(np.asarray(raw[key]["initial_state"], dtype=float)) for key in paired}
        if len(raw_initial_states) != len(paired):
            reasons.append("repeated_initial_states")
        for key in paired:
            left = np.asarray(raw[key]["initial_state"], dtype=float)
            right = np.asarray(filtered[key]["initial_state"], dtype=float)
            if left.shape != right.shape or not np.array_equal(left, right):
                reasons.append("unpaired_initial_states")
                break
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "effective_pairs": len(paired) if not reasons else 0,
    }


def analyze(
    path: Path,
    output: Path,
    primary_endpoint: str = "episode_any_safety_violation",
    adaptive_method: str = "adaptive_conformal",
    intervention_metric: str | None = None,
) -> dict:
    if primary_endpoint not in {
        "episode_any_safety_violation",
        "violation_rate",
        "controllable_violation_rate",
    }:
        raise ValueError(f"unsupported primary endpoint: {primary_endpoint}")
    if adaptive_method == "static_conformal":
        raise ValueError("adaptive_method must differ from static_conformal")
    if intervention_metric not in {
        None,
        "mean_action_correction",
        "projection_rate",
        "total_correction",
        "total_action_correction",
        "fallback_rate",
    }:
        raise ValueError(f"unsupported intervention metric: {intervention_metric}")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    groups: dict[tuple[str, str, str, str], dict[str, dict]] = {}
    for row in rows:
        policy_key = str(row.get("policy_checkpoint_sha256", row.get("policy", "unspecified")))
        groups.setdefault((policy_key, row["condition"], row["scenario"], row["method"]), {})[
            str(row["seed"])
        ] = row
    tests = []
    primary_tests = []
    mechanism_tests = []
    intervention_tests = []
    policy_cells = {
        (str(row.get("policy_checkpoint_sha256", row.get("policy", "unspecified"))), row["condition"], row["scenario"])
        for row in rows
    }
    for policy_key, condition, scenario in sorted(policy_cells):
        raw = groups.get((policy_key, condition, scenario, "raw"))
        if raw is not None:
            for method in ("static_conformal", adaptive_method):
                filtered = groups[(policy_key, condition, scenario, method)]
                paired = sorted(set(raw).intersection(filtered))
                inference = paired_inference_eligibility(raw, filtered, paired)
                test_id = f"{policy_key}:{condition}:{scenario}:{method}"
                item: dict[str, object] = {
                    "id": test_id,
                    "policy_key": policy_key,
                    "condition": condition,
                    "scenario": scenario,
                    "method": method,
                    "pairs": len(paired),
                    "inference": inference,
                }
                if inference["eligible"]:
                    raw_values = np.asarray([raw[key]["violation_rate"] for key in paired])
                    filtered_values = np.asarray([filtered[key]["violation_rate"] for key in paired])
                    raw_collision = np.asarray([raw[key]["collision"] for key in paired], dtype=bool)
                    filtered_collision = np.asarray([filtered[key]["collision"] for key in paired], dtype=bool)
                    item["violation"] = bootstrap_difference(raw_values, filtered_values)
                    raw_episode_violation = raw_values > 0.0
                    filtered_episode_violation = filtered_values > 0.0
                    item["episode_safety_violation_mcnemar_p"] = exact_mcnemar_pvalue(
                        raw_episode_violation, filtered_episode_violation
                    )
                    item["collision_mcnemar_p"] = exact_mcnemar_pvalue(raw_collision, filtered_collision)
                tests.append(item)

        static = groups[(policy_key, condition, scenario, "static_conformal")]
        adaptive = groups[(policy_key, condition, scenario, adaptive_method)]
        paired = sorted(set(static).intersection(adaptive))
        inference = paired_inference_eligibility(static, adaptive, paired)
        primary: dict[str, object] = {
            "id": f"{policy_key}:{condition}:{scenario}:{adaptive_method}_vs_static",
            "policy_key": policy_key,
            "condition": condition,
            "scenario": scenario,
            "comparison": f"{adaptive_method}_vs_static_conformal",
            "pairs": len(paired),
            "inference": inference,
        }
        if inference["eligible"]:
            static_values = np.asarray([static[key]["violation_rate"] for key in paired])
            adaptive_values = np.asarray([adaptive[key]["violation_rate"] for key in paired])
            if primary_endpoint == "episode_any_safety_violation":
                primary["violation"] = bootstrap_difference(static_values, adaptive_values)
                primary["episode_safety_violation_mcnemar_p"] = exact_mcnemar_pvalue(
                    static_values > 0.0, adaptive_values > 0.0
                )
            else:
                metric = primary_endpoint
                static_values = np.asarray([static[key][metric] for key in paired])
                adaptive_values = np.asarray([adaptive[key][metric] for key in paired])
                result_key = "violation" if metric == "violation_rate" else "controllable_violation"
                primary[result_key] = bootstrap_difference(static_values, adaptive_values)
                primary["violation_rate_sign_flip_p"] = paired_sign_flip_pvalue(
                    static_values, adaptive_values, seed=len(primary_tests)
                )
        primary_tests.append(primary)

        if intervention_metric is not None:
            intervention: dict[str, object] = {
                "id": f"{policy_key}:{condition}:{scenario}:{adaptive_method}_vs_static:{intervention_metric}",
                "policy_key": policy_key,
                "condition": condition,
                "scenario": scenario,
                "comparison": f"{adaptive_method}_vs_static_conformal",
                "metric": intervention_metric,
                "pairs": len(paired),
                "inference": inference,
            }
            if inference["eligible"]:
                static_values = np.asarray([static[key][intervention_metric] for key in paired])
                adaptive_values = np.asarray([adaptive[key][intervention_metric] for key in paired])
                intervention["value"] = bootstrap_difference(static_values, adaptive_values)
                intervention["sign_flip_p"] = paired_sign_flip_pvalue(
                    static_values, adaptive_values, seed=len(intervention_tests)
                )
            intervention_tests.append(intervention)

        # This is a separate, preregistered mechanism family. Its positive
        # direction means that removing the conformal residual margin worsens
        # the per-step violation rate.
        no_margin_key = (policy_key, condition, scenario, "robust_no_margin")
        if no_margin_key in groups:
            no_margin = groups[no_margin_key]
            paired = sorted(set(no_margin).intersection(static))
            inference = paired_inference_eligibility(no_margin, static, paired)
            mechanism: dict[str, object] = {
                "id": f"{policy_key}:{condition}:{scenario}:no_margin_vs_static",
                "policy_key": policy_key,
                "condition": condition,
                "scenario": scenario,
                "comparison": "robust_no_margin_vs_static_conformal",
                "pairs": len(paired),
                "inference": inference,
            }
            if inference["eligible"]:
                no_margin_values = np.asarray([no_margin[key]["violation_rate"] for key in paired])
                static_values = np.asarray([static[key]["violation_rate"] for key in paired])
                mechanism["violation"] = bootstrap_difference(no_margin_values, static_values)
                mechanism["violation_rate_sign_flip_p"] = paired_sign_flip_pvalue(
                    no_margin_values, static_values, seed=len(mechanism_tests)
                )
            mechanism_tests.append(mechanism)

    primary_p_key = "episode_safety_violation_mcnemar_p" if primary_endpoint == "episode_any_safety_violation" else "violation_rate_sign_flip_p"
    primary_adjusted = holm(
        [(str(item["id"]), float(item[primary_p_key])) for item in primary_tests if primary_p_key in item]
    )
    for item in primary_tests:
        if primary_p_key in item:
            item[f"{primary_p_key}_holm"] = primary_adjusted[str(item["id"])]

    mechanism_adjusted = holm(
        [
            (str(item["id"]), float(item["violation_rate_sign_flip_p"]))
            for item in mechanism_tests
            if "violation_rate_sign_flip_p" in item
        ]
    )
    for item in mechanism_tests:
        if "violation_rate_sign_flip_p" in item:
            item["violation_rate_sign_flip_p_holm"] = mechanism_adjusted[str(item["id"])]

    intervention_adjusted = holm(
        [(str(item["id"]), float(item["sign_flip_p"])) for item in intervention_tests if "sign_flip_p" in item]
    )
    for item in intervention_tests:
        if "sign_flip_p" in item:
            item["sign_flip_p_holm"] = intervention_adjusted[str(item["id"])]

    episode_adjusted = holm(
        [
            (str(item["id"]), float(item["episode_safety_violation_mcnemar_p"]))
            for item in tests
            if "episode_safety_violation_mcnemar_p" in item
        ]
    )
    for item in tests:
        if "episode_safety_violation_mcnemar_p" in item:
            item["episode_safety_violation_mcnemar_p_holm"] = episode_adjusted[str(item["id"])]

    collision_adjusted = holm(
        [(str(item["id"]), float(item["collision_mcnemar_p"])) for item in tests if "collision_mcnemar_p" in item]
    )
    for item in tests:
        if "collision_mcnemar_p" in item:
            item["collision_mcnemar_p_holm"] = collision_adjusted[str(item["id"])]
    intervention_summary = []
    for policy_key, condition, scenario, method in sorted(groups):
        cell_rows = list(groups[(policy_key, condition, scenario, method)].values())
        required = {"projection_rate", "mean_action_correction", "steps"}
        if not all(required.issubset(row) for row in cell_rows):
            continue
        totals = [
            float(row.get("total_action_correction", row["mean_action_correction"] * row["steps"]))
            for row in cell_rows
        ]
        normalized_totals = [
            float(row.get("total_correction", row["mean_action_correction"]))
            for row in cell_rows
        ]
        intervention_summary.append(
            {
                "condition": condition,
                "scenario": scenario,
                "method": method,
                "policy_key": policy_key,
                "trials": len(cell_rows),
                "projection_rate": float(np.mean([row["projection_rate"] for row in cell_rows])),
                "mean_action_correction": float(np.mean([row["mean_action_correction"] for row in cell_rows])),
                "total_action_correction": float(np.mean(totals)),
                "total_correction": float(np.mean(normalized_totals)),
            }
        )
    result = {
        "status": "partial rollout",
        "warning": "Paired inference requires seeded randomized resets and distinct paired initial states; other outputs are descriptive only.",
        "primary_endpoint": f"{adaptive_method}_vs_static_conformal {primary_endpoint}",
        "primary_tests": primary_tests,
        "mechanism_ablation_endpoint": "robust_no_margin_vs_static_conformal violation_rate",
        "mechanism_tests": mechanism_tests,
        "intervention_endpoint": None if intervention_metric is None else f"{adaptive_method}_vs_static_conformal {intervention_metric}",
        "intervention_tests": intervention_tests,
        "intervention_summary": intervention_summary,
        "tests": tests,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--primary-endpoint",
        choices=("episode_any_safety_violation", "violation_rate", "controllable_violation_rate"),
        default="episode_any_safety_violation",
    )
    parser.add_argument("--adaptive-method", default="adaptive_conformal")
    parser.add_argument(
        "--intervention-metric",
        choices=(
            "mean_action_correction",
            "projection_rate",
            "total_correction",
            "total_action_correction",
            "fallback_rate",
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.input, args.output, args.primary_endpoint, args.adaptive_method, args.intervention_metric),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
