"""Finite-horizon pathwise residual-envelope calibration.

The guarantee concerns the maximum residual over a complete episode.  It is
intentionally separate from the closed-loop RACF safety filter: envelope
coverage does not imply trajectory safety without additional dynamics,
containment, feasibility, and deadline assumptions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .conformal import SplitConformalPredictor, finite_sample_quantile


def _episode_batch(values: np.ndarray, name: str, *, allow_action_dim: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} must be a non-empty 3-D episode batch")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


class PathwiseResidualEnvelope:
    """Calibrate a finite-sample quantile of complete-episode max scores."""

    def __init__(
        self,
        alpha: float = 0.1,
        dynamics_model: Callable[..., np.ndarray] | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if dynamics_model is None:
            raise ValueError("dynamics_model is required")
        self.alpha = float(alpha)
        self.f_hat = dynamics_model
        self.q_pathwise: float | None = None
        self.calibration_max_scores: np.ndarray | None = None

    def _episode_scores(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> np.ndarray:
        states_batch = _episode_batch(states, "states")
        actions_batch = _episode_batch(actions, "actions")
        next_batch = _episode_batch(next_states, "next_states")
        if states_batch.shape[:2] != actions_batch.shape[:2] or states_batch.shape[:2] != next_batch.shape[:2]:
            raise ValueError("states, actions, and next_states must share episode and horizon dimensions")
        if states_batch.shape[2] != next_batch.shape[2]:
            raise ValueError("states and next_states must share the state dimension")
        predictor = SplitConformalPredictor(alpha=self.alpha, dynamics_model=self.f_hat)
        max_scores = []
        for state_episode, action_episode, next_episode in zip(states_batch, actions_batch, next_batch, strict=True):
            scores = predictor.score(state_episode, action_episode, next_episode)
            max_scores.append(float(np.max(scores)))
        return np.asarray(max_scores, dtype=np.float64)

    def score_episodes(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> np.ndarray:
        """Return one maximum residual score per complete episode."""
        return self._episode_scores(states, actions, next_states)

    def calibrate(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> float:
        """Calibrate q from complete, disjoint calibration episodes."""
        self.calibration_max_scores = self._episode_scores(states, actions, next_states)
        self.q_pathwise = finite_sample_quantile(self.calibration_max_scores, self.alpha)
        return self.q_pathwise

    def evaluate_coverage(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> dict[str, Any]:
        """Evaluate the empirical complete-episode envelope coverage."""
        if self.q_pathwise is None:
            raise RuntimeError("calibrate must be called before evaluate_coverage")
        scores = self._episode_scores(states, actions, next_states)
        return {
            "coverage": float(np.mean(scores <= self.q_pathwise)),
            "n_episodes": int(scores.size),
            "horizon": int(np.asarray(states).shape[1]),
            "mean_max_score": float(np.mean(scores)),
            "p95_max_score": float(np.percentile(scores, 95)),
            "q_pathwise": float(self.q_pathwise),
            "alpha": self.alpha,
            "scope": "finite-horizon pathwise residual-envelope coverage",
        }
