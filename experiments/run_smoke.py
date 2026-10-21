"""Deterministic, data-free RACF smoke reproduction.

This script validates the public package path with synthetic transitions. It is
not a substitute for the paper's full paired rollout benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racf.conformal import SplitConformalPredictor  # noqa: E402
from racf.pathwise import PathwiseResidualEnvelope  # noqa: E402


def dynamics(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return states + 0.2 * actions


def make_transitions(seed: int, episodes: int, horizon: int):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(episodes, horizon, 2))
    actions = rng.normal(size=(episodes, horizon, 1))
    noise = rng.normal(scale=0.04, size=(episodes, horizon, 2))
    next_states = dynamics(states.reshape(-1, 2), np.repeat(actions, 2, axis=2).reshape(-1, 2))
    next_states = next_states.reshape(episodes, horizon, 2) + noise
    return states, actions, next_states


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    states, actions, next_states = make_transitions(seed=2026, episodes=48, horizon=20)
    split = 32
    one_step = SplitConformalPredictor(alpha=0.1, dynamics_model=dynamics)
    q = one_step.calibrate(
        states[:split].reshape(-1, 2),
        np.repeat(actions[:split], 2, axis=2).reshape(-1, 2),
        next_states[:split].reshape(-1, 2),
    )
    one_step_result = one_step.evaluate_coverage(
        states[split:].reshape(-1, 2),
        np.repeat(actions[split:], 2, axis=2).reshape(-1, 2),
        next_states[split:].reshape(-1, 2),
    )
    pathwise = PathwiseResidualEnvelope(alpha=0.1, dynamics_model=dynamics)
    path_q = pathwise.calibrate(states[:split], actions[:split], next_states[:split])
    pathwise_result = pathwise.evaluate_coverage(states[split:], actions[split:], next_states[split:])
    result = {
        "seed": 2026,
        "calibration_episodes": split,
        "test_episodes": int(states.shape[0] - split),
        "horizon": int(states.shape[1]),
        "one_step": {**one_step_result, "quantile": q},
        "pathwise": {**pathwise_result, "q_pathwise": path_q},
        "scope": "mock test: synthetic transitions only; no simulator or paper benchmark",
    }
    (output_dir / "coverage.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        scores = one_step.score(
            states[split:].reshape(-1, 2),
            np.repeat(actions[split:], 2, axis=2).reshape(-1, 2),
            next_states[split:].reshape(-1, 2),
        )
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.hist(scores, bins=20, color="#4472C4", alpha=0.8)
        ax.axvline(q, color="#C00000", linestyle="--", label=f"q={q:.3f}")
        ax.set(xlabel="one-step residual score", ylabel="count", title="RACF smoke calibration")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "coverage.png", dpi=160)
        plt.close(fig)
    except ImportError:
        result["figure"] = "skipped: matplotlib is unavailable"
        (output_dir / "coverage.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
