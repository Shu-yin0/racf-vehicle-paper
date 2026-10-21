"""Split conformal calibration for one-step dynamics residuals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def _as_batch(values: np.ndarray, name: str, dimension: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(f"{name} must have {dimension} columns, got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the split-conformal order statistic, including the infinity case."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite array")
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha)))
    if rank > values.size:
        return float("inf")
    return float(np.sort(values)[rank - 1])


class SplitConformalPredictor:
    """One-step split conformal predictor for transition residuals.

    The guarantee is marginal over an iid calibration sample and an independent
    test transition. Distribution shift is handled by an externally supplied TV
    bound; this class does not estimate that bound.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        dynamics_model: Callable[..., np.ndarray] | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = float(alpha)
        self.f_hat = dynamics_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_alpha: float | None = None
        self.scores: np.ndarray | None = None
        self.n_calib = 0

    def _predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.f_hat is None:
            raise ValueError("dynamics_model is required")
        try:
            prediction = np.asarray(self.f_hat(states, actions), dtype=np.float64)
            if prediction.shape == states.shape:
                return prediction
        except (TypeError, ValueError):
            pass
        rows = [
            np.asarray(self.f_hat(state, action), dtype=np.float64)
            for state, action in zip(states, actions, strict=True)
        ]
        prediction = np.asarray(rows, dtype=np.float64)
        if prediction.shape != states.shape:
            raise ValueError(f"dynamics_model returned {prediction.shape}, expected {states.shape}")
        return prediction

    def score(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> np.ndarray:
        states_batch = _as_batch(states, "states", self.state_dim)
        actions_batch = _as_batch(actions, "actions", self.action_dim)
        next_batch = _as_batch(next_states, "next_states", states_batch.shape[1])
        if states_batch.shape[0] != actions_batch.shape[0] or states_batch.shape[0] != next_batch.shape[0]:
            raise ValueError("states, actions, and next_states must have the same number of rows")
        if self.state_dim is None:
            self.state_dim = states_batch.shape[1]
        if self.action_dim is None:
            self.action_dim = actions_batch.shape[1]
        return np.linalg.norm(next_batch - self._predict(states_batch, actions_batch), axis=1)

    def calibrate(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> float:
        """Calibrate and return the finite-sample residual threshold."""
        self.scores = self.score(states, actions, next_states)
        self.q_alpha = finite_sample_quantile(self.scores, self.alpha)
        self.n_calib = int(self.scores.size)
        return self.q_alpha

    def evaluate_coverage(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> dict[str, Any]:
        """Evaluate empirical one-step coverage on an independent test set."""
        if self.q_alpha is None:
            raise RuntimeError("calibrate must be called before evaluate_coverage")
        scores = self.score(states, actions, next_states)
        return {
            "coverage": float(np.mean(scores <= self.q_alpha)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_test": int(scores.size),
            "quantile": float(self.q_alpha),
            "alpha": self.alpha,
        }

    def predict_set(
        self,
        state: np.ndarray,
        action: np.ndarray,
        param_samples: list[Any],
        dynamics_model: Callable[..., np.ndarray],
    ) -> list[Any]:
        """Return finite parameter samples whose predictions lie in the calibrated ball."""
        if self.q_alpha is None:
            raise RuntimeError("calibrate must be called before predict_set")
        state_array = np.asarray(state, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64)
        nominal = np.asarray(dynamics_model(state_array, action_array, None), dtype=np.float64)
        valid = []
        for params in param_samples:
            prediction = np.asarray(dynamics_model(state_array, action_array, params), dtype=np.float64)
            if np.linalg.norm(prediction - nominal) <= self.q_alpha:
                valid.append(params)
        return valid


class FiniteConditionConformalPredictor:
    """Static split conformal calibration over a finite registered condition set.

    Every registered condition receives an independent calibration stratum. The
    runtime threshold is the maximum stratum threshold, which is conservative
    for each registered condition. An unregistered *test* condition may be
    evaluated with that same global maximum as a shift diagnostic, but receives
    no condition-specific calibration claim. This is not a covariate-shift
    estimator and does not cover adaptive updates.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        dynamics_model: Callable[..., np.ndarray] | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if dynamics_model is None:
            raise ValueError("dynamics_model is required")
        self.alpha = float(alpha)
        self.f_hat = dynamics_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition_quantiles: dict[str, float] = {}
        self.q_alpha: float | None = None

    def _model_for_condition(self, condition: Any) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def predict(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            try:
                prediction = np.asarray(self.f_hat(states, actions, condition), dtype=np.float64)
                if prediction.shape == states.shape:
                    return prediction
            except (TypeError, ValueError):
                pass
            rows = [
                np.asarray(self.f_hat(state, action, condition), dtype=np.float64)
                for state, action in zip(states, actions, strict=True)
            ]
            prediction = np.asarray(rows, dtype=np.float64)
            if prediction.shape != states.shape:
                raise ValueError(f"dynamics_model returned {prediction.shape}, expected {states.shape}")
            return prediction

        return predict

    def _predictor(self, condition: Any) -> SplitConformalPredictor:
        return SplitConformalPredictor(
            alpha=self.alpha,
            dynamics_model=self._model_for_condition(condition),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def calibrate(
        self,
        condition_data: Mapping[str, tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ) -> dict[str, float]:
        """Calibrate one split-conformal threshold per registered condition."""
        if not condition_data:
            raise ValueError("condition_data must be non-empty")
        quantiles: dict[str, float] = {}
        for name, (condition, arrays) in condition_data.items():
            predictor = self._predictor(condition)
            quantiles[str(name)] = predictor.calibrate(*arrays)
            self.state_dim = predictor.state_dim
            self.action_dim = predictor.action_dim
        self.condition_quantiles = quantiles
        self.q_alpha = max(quantiles.values())
        return quantiles.copy()

    def evaluate_coverage(
        self,
        condition_name: str,
        condition: Any,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate with the global threshold and disclose its calibration source."""
        if self.q_alpha is None:
            raise RuntimeError("calibrate must be called before evaluate_coverage")
        predictor = self._predictor(condition)
        predictor.q_alpha = self.q_alpha
        result = predictor.evaluate_coverage(states, actions, next_states)
        result["condition"] = condition_name
        if condition_name in self.condition_quantiles:
            result["condition_quantile"] = self.condition_quantiles[condition_name]
            result["threshold_source"] = "registered_condition_global_maximum"
        else:
            result["condition_quantile"] = None
            result["threshold_source"] = "global_maximum_unregistered_test_condition"
        return result
