"""Analyze fixed-bin one-step coverage on independent ACC test transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import beta

from icassp2027.src.acc_adapter import ACTUATOR_GAIN_MODEL, acc_dynamics_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(successes: int, total: int, level: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("binomial interval requires a positive total")
    tail = (1.0 - level) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return low, high


def _bins() -> dict[str, tuple[tuple[str, Callable[[dict[str, float]], bool]], ...]]:
    return {
        "ego_speed": (
            ("low", lambda row: row["ego_speed"] < 10.0),
            ("mid", lambda row: 10.0 <= row["ego_speed"] < 20.0),
            ("high", lambda row: row["ego_speed"] >= 20.0),
        ),
        "closing_speed": (
            ("opening", lambda row: row["closing_speed"] < -0.5),
            ("near_zero", lambda row: -0.5 <= row["closing_speed"] <= 0.5),
            ("closing", lambda row: row["closing_speed"] > 0.5),
        ),
        "gap_margin": (
            ("near_safe", lambda row: row["gap_margin"] < 5.0),
            ("medium", lambda row: 5.0 <= row["gap_margin"] < 20.0),
            ("large", lambda row: row["gap_margin"] >= 20.0),
        ),
        "desired_action": (
            ("braking", lambda row: row["action"] < -0.1),
            ("neutral", lambda row: -0.1 <= row["action"] <= 0.1),
            ("throttle", lambda row: row["action"] > 0.1),
        ),
        "grade": (
            ("negative", lambda row: row["grade"] < -0.25),
            ("near_zero", lambda row: -0.25 <= row["grade"] <= 0.25),
            ("positive", lambda row: row["grade"] > 0.25),
        ),
    }


def analyze(data_dir: Path, *, quantile: float = 0.09078879007355795) -> dict[str, Any]:
    if not np.isfinite(quantile) or quantile < 0.0:
        raise ValueError("quantile must be finite and non-negative")
    manifest_path = data_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sampling_mode") != "independent_one_step":
        raise ValueError("conditional exact intervals require independent_one_step data")
    model = acc_dynamics_model(ACTUATOR_GAIN_MODEL)
    records: list[dict[str, Any]] = []
    verified_files = []
    for name, entry in sorted(manifest.get("datasets", {}).items()):
        if not name.startswith("test_"):
            continue
        path = data_dir / entry["path"]
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"hash mismatch for {path}")
        condition = str(entry["metadata"]["condition"])
        with np.load(path, allow_pickle=False) as data:
            states = np.asarray(data["states"], dtype=np.float64)
            actions = np.asarray(data["actions"], dtype=np.float64)
            next_states = np.asarray(data["next_states"], dtype=np.float64)
        if not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(next_states).all():
            raise ValueError(f"NaN/Inf in {path}")
        predictions = np.asarray(
            [model(state, action, None) for state, action in zip(states, actions, strict=True)],
            dtype=np.float64,
        )
        scores = np.linalg.norm(next_states - predictions, axis=1)
        for state, action, next_state, score in zip(states, actions, next_states, scores, strict=True):
            ego_speed, lead_speed, gap, _, grade = map(float, state)
            next_ego, _, next_gap, _, _ = map(float, next_state)
            records.append(
                {
                    "condition": condition,
                    "ego_speed": ego_speed,
                    "closing_speed": ego_speed - lead_speed,
                    "gap_margin": gap - (3.0 + 1.5 * ego_speed),
                    "action": float(action[0]),
                    "grade": grade,
                    "score": float(score),
                    "covered": bool(score <= quantile),
                    "next_safety_violation": bool(next_gap < 3.0 + 1.5 * next_ego),
                }
            )
        verified_files.append({"path": str(path), "sha256": digest, "rows": len(states)})
    if not records:
        raise ValueError("no independent test transitions found")

    def summarize(selected: list[dict[str, Any]], stratum: str, bin_name: str) -> dict[str, Any]:
        total = len(selected)
        covered = sum(bool(row["covered"]) for row in selected)
        low, high = _interval(covered, total)
        scores = np.asarray([row["score"] for row in selected], dtype=np.float64)
        return {
            "stratum": stratum,
            "bin": bin_name,
            "n": total,
            "covered": covered,
            "coverage": covered / total,
            "exact_ci95_low": low,
            "exact_ci95_high": high,
            "score_median": float(np.median(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_max": float(np.max(scores)),
            "next_safety_violation_rate": float(np.mean([row["next_safety_violation"] for row in selected])),
            "inference": "descriptive_only" if total < 30 else "exact_binomial_independent_transition",
        }

    summaries = [summarize(records, "all", "all")]
    for condition in sorted({row["condition"] for row in records}):
        summaries.append(summarize([row for row in records if row["condition"] == condition], "condition", condition))
    for stratum, bins in _bins().items():
        assigned = 0
        for bin_name, predicate in bins:
            selected = [row for row in records if predicate(row)]
            assigned += len(selected)
            if selected:
                summaries.append(summarize(selected, stratum, bin_name))
        if assigned != len(records):
            raise AssertionError(f"{stratum} bins do not partition the data")
    return {
        "protocol": manifest.get("protocol"),
        "status": "independent one-step conditional coverage analysis",
        "scope": "independent one-step transition coverage; not trajectory safety or conditional guarantee",
        "sampling_mode": manifest["sampling_mode"],
        "quantile": quantile,
        "transitions": len(records),
        "verified_files": verified_files,
        "bins": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.09078879007355795)
    args = parser.parse_args()
    result = analyze(args.data_dir, quantile=args.quantile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "transitions": result["transitions"], "bins": len(result["bins"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
