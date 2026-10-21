"""Risk-aware conformal action filtering with an auditable finite parameter set."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .conformal import SplitConformalPredictor, finite_sample_quantile


@dataclass(frozen=True)
class FilterInfo:
    success: bool
    fallback: bool
    iterations: int
    min_constraint: float
    residual_margin: float
    projected: bool
    status: str
    correction_norm: float = 0.0
    action_energy: float = 0.0
    action_slew_norm: float = 0.0
    objective_value: float = 0.0
    intervention_budget: float | None = None
    intervention_budget_exceeded: bool = False
    projection_applicable: bool = True
    projection_semantics: str = "racf_robust_projection"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FiniteResidualParameterSelector:
    """Empirically select a small subset from a finite auditable grid.

    The selector is intentionally separate from split-conformal calibration:
    its online updates are an empirical adaptation mechanism, not a coverage
    guarantee. Until enough transitions have arrived it retains every
    candidate. Thereafter it selects the ``max_active`` candidates with the
    smallest windowed finite-sample residual quantiles.
    """

    def __init__(
        self,
        nominal_dynamics: Callable[..., np.ndarray],
        candidates: Iterable[Any],
        alpha: float = 0.1,
        window_size: int = 100,
        update_freq: int = 10,
        min_update_samples: int = 20,
        max_active: int = 2,
        selection_slack: float = 0.0,
        residual_trigger_threshold: float | None = None,
        recovery_updates: int = 0,
        fill_to_max_active: bool = True,
    ) -> None:
        self.f_hat = nominal_dynamics
        self.candidates = list(candidates)
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.update_freq = int(update_freq)
        self.min_update_samples = int(min_update_samples)
        self.max_active = int(max_active)
        self.selection_slack = float(selection_slack)
        self.residual_trigger_threshold = (
            None if residual_trigger_threshold is None else float(residual_trigger_threshold)
        )
        self.recovery_updates = int(recovery_updates)
        self.fill_to_max_active = bool(fill_to_max_active)
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min(self.window_size, self.update_freq, self.min_update_samples, self.max_active) <= 0:
            raise ValueError("window_size, update_freq, min_update_samples, and max_active must be positive")
        if self.max_active > len(self.candidates):
            raise ValueError("max_active cannot exceed the finite candidate count")
        if self.selection_slack < 0.0 or not np.isfinite(self.selection_slack):
            raise ValueError("selection_slack must be finite and non-negative")
        if self.residual_trigger_threshold is not None and (
            self.residual_trigger_threshold <= 0.0 or not np.isfinite(self.residual_trigger_threshold)
        ):
            raise ValueError("residual_trigger_threshold must be finite and positive")
        if self.recovery_updates < 0:
            raise ValueError("recovery_updates must be non-negative")
        self._buffers: list[list[float]] = [[] for _ in self.candidates]
        self._updates = 0
        self._selection_count = 0
        self._event_selection_count = 0
        self._residual_trigger_count = 0
        self._residual_trigger_steps: list[int] = []
        self._recovery_until = 0
        self._residual_trigger_active = False
        self._active_indices = list(range(len(self.candidates)))
        self._quantiles: list[float] | None = None

    @staticmethod
    def _call_dynamics(model: Callable[..., np.ndarray], state: np.ndarray, action: np.ndarray, theta: Any) -> np.ndarray:
        return np.asarray(model(state, action, theta), dtype=np.float64)

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> list[Any] | None:
        """Consume one transition and return a new finite active set when due."""
        observed = np.asarray(next_state, dtype=np.float64)
        transition_residuals: list[float] = []
        for index, theta in enumerate(self.candidates):
            prediction = self._call_dynamics(self.f_hat, np.asarray(state), np.asarray(action), theta)
            residual = float(np.linalg.norm(observed - prediction))
            if not np.isfinite(residual):
                raise ValueError("candidate transition residual is NaN or Inf")
            transition_residuals.append(residual)
            self._buffers[index].append(residual)
            self._buffers[index] = self._buffers[index][-self.window_size :]
        self._updates += 1
        threshold_exceeded = (
            self.residual_trigger_threshold is not None
            and max(transition_residuals) > self.residual_trigger_threshold
        )
        if (
            threshold_exceeded
            and not self._residual_trigger_active
            and self._updates >= self._recovery_until
        ):
            self._active_indices = list(range(len(self.candidates)))
            self._recovery_until = self._updates + self.recovery_updates
            self._residual_trigger_count += 1
            self._residual_trigger_steps.append(self._updates - 1)
        self._residual_trigger_active = bool(threshold_exceeded)
        if self._updates < self._recovery_until:
            return self.active_samples
        if self._updates % self.update_freq != 0 or len(self._buffers[0]) < self.min_update_samples:
            return None
        self._quantiles = [finite_sample_quantile(np.asarray(buffer), self.alpha) for buffer in self._buffers]
        order = sorted(range(len(self.candidates)), key=lambda index: (self._quantiles[index], index))
        cutoff = self._quantiles[order[0]] + self.selection_slack
        selected = [index for index in order if self._quantiles[index] <= cutoff]
        # Keep the configured active-set size stable.  The slack defines the
        # preferred prefix, while the remaining slots are filled by the next
        # best candidates so legacy top-k behavior remains auditable.
        if self.fill_to_max_active and len(selected) < self.max_active:
            selected.extend(index for index in order if index not in selected)
        self._active_indices = selected[: self.max_active]
        self._selection_count += 1
        return self.active_samples

    def select_current_transition(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> list[Any]:
        """Select the finite candidates that best explain one observed event.

        This is an empirical event-response mechanism, separate from the
        windowed conformal quantile and its static coverage statement.
        """
        observed = np.asarray(next_state, dtype=np.float64)
        residuals = [
            float(
                np.linalg.norm(
                    observed - self._call_dynamics(self.f_hat, np.asarray(state), np.asarray(action), theta)
                )
            )
            for theta in self.candidates
        ]
        if not all(np.isfinite(residual) for residual in residuals):
            raise ValueError("candidate transition residual is NaN or Inf")
        self._active_indices = sorted(range(len(self.candidates)), key=lambda index: (residuals[index], index))[
            : self.max_active
        ]
        self._event_selection_count += 1
        return self.active_samples

    @property
    def active_samples(self) -> list[Any]:
        """Return a copy of the currently selected finite parameter samples."""
        return [self.candidates[index] for index in self._active_indices]

    @property
    def update_count(self) -> int:
        return self._updates

    @property
    def selection_count(self) -> int:
        return self._selection_count

    @property
    def event_selection_count(self) -> int:
        return self._event_selection_count

    @property
    def residual_trigger_count(self) -> int:
        return self._residual_trigger_count

    @property
    def residual_trigger_steps(self) -> list[int]:
        return list(self._residual_trigger_steps)

    @property
    def residual_trigger_active(self) -> bool:
        return self._residual_trigger_active

    @property
    def quantiles(self) -> list[float] | None:
        return None if self._quantiles is None else list(self._quantiles)


class RACF:
    """Project a desired action against a finite conformal parameter set.

    ``adaptive=True`` updates the residual threshold from a sliding window. The
    adaptive mode is deliberately empirical and does not claim the static
    split-conformal theorem after updates.
    """

    def __init__(
        self,
        nominal_dynamics: Callable[..., np.ndarray],
        safety_constraint: Callable[[np.ndarray], float],
        action_bounds: tuple[Iterable[float], Iterable[float]],
        parameter_samples: Iterable[Any],
        alpha: float = 0.1,
        adaptive: bool = False,
        window_size: int = 100,
        update_freq: int = 10,
        min_update_samples: int = 20,
        parameter_set_factory: Callable[[float], Iterable[Any]] | None = None,
        adaptive_parameter_selector: FiniteResidualParameterSelector | None = None,
        preserve_calibration_floor: bool = False,
        retain_adaptive_peak: bool = False,
        event_residual_threshold: float | None = None,
        event_selector_threshold: float | None = None,
        event_selector_hold_steps: int = 0,
        prediction_horizon: int = 1,
        emergency_action: Iterable[float] | Callable[[np.ndarray], np.ndarray] | None = None,
        safety_lipschitz: float = 0.0,
        action_energy_weight: float = 0.0,
        action_slew_weight: float = 0.0,
        intervention_budget: float | None = None,
    ) -> None:
        self.f_hat = nominal_dynamics
        self.h = safety_constraint
        self.alpha = float(alpha)
        self.adaptive = bool(adaptive)
        self.window_size = int(window_size)
        self.update_freq = int(update_freq)
        self.min_update_samples = int(min_update_samples)
        if self.window_size <= 0 or self.update_freq <= 0 or self.min_update_samples <= 0:
            raise ValueError("window_size, update_freq, and min_update_samples must be positive")
        self.lower = np.asarray(tuple(action_bounds[0]), dtype=np.float64).reshape(-1)
        self.upper = np.asarray(tuple(action_bounds[1]), dtype=np.float64).reshape(-1)
        if self.lower.size == 0 or self.lower.shape != self.upper.shape or np.any(self.lower > self.upper):
            raise ValueError("action bounds must be non-empty and ordered")
        self.parameter_samples = list(parameter_samples)
        if not self.parameter_samples:
            raise ValueError("parameter_samples must be non-empty")
        self.parameter_set_factory = parameter_set_factory
        self.adaptive_parameter_selector = adaptive_parameter_selector
        self.preserve_calibration_floor = bool(preserve_calibration_floor)
        self.retain_adaptive_peak = bool(retain_adaptive_peak)
        self.event_residual_threshold = (
            None if event_residual_threshold is None else float(event_residual_threshold)
        )
        self.event_selector_threshold = (
            None if event_selector_threshold is None else float(event_selector_threshold)
        )
        self.event_selector_hold_steps = int(event_selector_hold_steps)
        self.prediction_horizon = int(prediction_horizon)
        if self.adaptive_parameter_selector is not None and not self.adaptive:
            raise ValueError("adaptive_parameter_selector requires adaptive=True")
        if self.event_residual_threshold is not None:
            if not self.adaptive:
                raise ValueError("event_residual_threshold requires adaptive=True")
            if not np.isfinite(self.event_residual_threshold) or self.event_residual_threshold <= 0.0:
                raise ValueError("event_residual_threshold must be finite and positive")
        if self.event_selector_threshold is not None:
            if self.adaptive_parameter_selector is None:
                raise ValueError("event_selector_threshold requires adaptive_parameter_selector")
            if not np.isfinite(self.event_selector_threshold) or self.event_selector_threshold <= 0.0:
                raise ValueError("event_selector_threshold must be finite and positive")
        if self.event_selector_hold_steps < 0:
            raise ValueError("event_selector_hold_steps must be non-negative")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        self.emergency_action = emergency_action
        self.safety_lipschitz = float(safety_lipschitz)
        if self.safety_lipschitz < 0.0:
            raise ValueError("safety_lipschitz must be non-negative")
        self.action_energy_weight = float(action_energy_weight)
        self.action_slew_weight = float(action_slew_weight)
        self.intervention_budget = (
            None if intervention_budget is None else float(intervention_budget)
        )
        if (
            not np.isfinite(self.action_energy_weight)
            or self.action_energy_weight < 0.0
            or not np.isfinite(self.action_slew_weight)
            or self.action_slew_weight < 0.0
        ):
            raise ValueError("utility weights must be finite and non-negative")
        if self.intervention_budget is not None and (
            not np.isfinite(self.intervention_budget) or self.intervention_budget < 0.0
        ):
            raise ValueError("intervention_budget must be finite and non-negative")
        self.previous_action: np.ndarray | None = None
        self.q_alpha: float | None = None
        self.calibration_quantile: float | None = None
        self.last_adaptive_window_quantile: float | None = None
        self.event_trigger_count = 0
        self.event_residual_peak: float | None = None
        self.event_selector_trigger_count = 0
        self._event_selector_hold_remaining = 0
        self.residual_buffer: list[float] = []
        self._updates = 0

    @staticmethod
    def _call_dynamics(model: Callable[..., np.ndarray], state: np.ndarray, action: np.ndarray, theta: Any = None) -> np.ndarray:
        try:
            return np.asarray(model(state, action, theta), dtype=np.float64)
        except TypeError:
            if theta is not None:
                raise
            return np.asarray(model(state, action), dtype=np.float64)

    def calibrate(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> float:
        """Calibrate the nominal residual threshold."""
        predictor = SplitConformalPredictor(alpha=self.alpha, dynamics_model=self.f_hat)
        self.q_alpha = predictor.calibrate(states, actions, next_states)
        self.calibration_quantile = self.q_alpha
        self.residual_buffer = [float(value) for value in predictor.scores]
        if self.parameter_set_factory is not None:
            self.parameter_samples = list(self.parameter_set_factory(self.q_alpha))
            if not self.parameter_samples:
                raise ValueError("parameter_set_factory returned an empty parameter set")
        return self.q_alpha

    def set_calibrated_quantile(self, quantile: float) -> float:
        """Install a threshold produced by an auditable external calibration path.

        The finite-condition calibrator uses this method to provide its maximum
        registered-condition threshold to the static filter. Adaptive updates,
        when enabled, remain empirical after this initialization.
        """
        value = float(quantile)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError("quantile must be finite and non-negative")
        self.q_alpha = value
        self.calibration_quantile = value
        if self.parameter_set_factory is not None:
            samples = list(self.parameter_set_factory(value))
            if not samples:
                raise ValueError("parameter_set_factory returned an empty parameter set")
            self.parameter_samples = samples
        return value

    def _constraint_values(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        residual_margin = self._residual_margin()
        values = []
        for theta in self.parameter_samples:
            values.append(self._minimum_predicted_margin(state, action, theta, residual_margin))
        return np.asarray(values, dtype=np.float64)

    def _minimum_predicted_margin(
        self, state: np.ndarray, action: np.ndarray, theta: Any, residual_margin: float
    ) -> float:
        predicted = np.asarray(state, dtype=np.float64)
        minimum = np.inf
        for _ in range(self.prediction_horizon):
            predicted = self._call_dynamics(self.f_hat, predicted, action, theta)
            value = float(np.asarray(self.h(predicted)).reshape(-1)[0]) - residual_margin
            if not np.isfinite(value):
                raise ValueError("safety_constraint returned NaN or Inf")
            minimum = min(minimum, value)
        return float(minimum)

    def _residual_margin(self) -> float:
        """Return the state-to-safety margin implied by static calibration."""
        return self.safety_lipschitz * (self.q_alpha if self.q_alpha is not None else 0.0)

    def _emergency(self, state: np.ndarray) -> np.ndarray:
        if callable(self.emergency_action):
            action = np.asarray(self.emergency_action(state), dtype=np.float64).reshape(-1)
        elif self.emergency_action is None:
            action = self.lower.copy()
        else:
            action = np.asarray(tuple(self.emergency_action), dtype=np.float64).reshape(-1)
        if action.shape != self.lower.shape:
            raise ValueError("emergency_action has the wrong dimension")
        return np.clip(action, self.lower, self.upper)

    def filter_action(self, state: np.ndarray, desired_action: np.ndarray, max_iter: int = 40) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the nearest feasible action, or a bounded emergency action."""
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        desired = np.asarray(desired_action, dtype=np.float64).reshape(-1)
        if desired.shape != self.lower.shape:
            raise ValueError(f"desired_action must have shape {self.lower.shape}")
        desired = np.clip(desired, self.lower, self.upper)

        residual_margin = self._residual_margin()
        previous_action = (
            None if self.previous_action is None else np.asarray(self.previous_action, dtype=np.float64)
        )

        def margin_constraint(action: np.ndarray, theta: Any) -> float:
            return self._minimum_predicted_margin(state_array, action, theta, residual_margin)

        def objective(action: np.ndarray) -> float:
            action_array = np.asarray(action, dtype=np.float64)
            correction = float(np.sum((action_array - desired) ** 2))
            energy = float(np.sum(action_array**2))
            slew = (
                0.0
                if previous_action is None
                else float(np.sum((action_array - previous_action) ** 2))
            )
            value = (
                correction
                + self.action_energy_weight * energy
                + self.action_slew_weight * slew
            )
            if not np.isfinite(value):
                raise ValueError("utility objective returned NaN or Inf")
            return value

        constraints = [
            {"type": "ineq", "fun": lambda action, theta=theta: margin_constraint(action, theta)}
            for theta in self.parameter_samples
        ]
        result = minimize(
            objective,
            desired,
            method="SLSQP",
            bounds=[(self.lower[index], self.upper[index]) for index in range(self.lower.size)],
            constraints=constraints,
            options={"maxiter": int(max_iter), "ftol": 1e-10},
        )
        candidate = np.clip(np.asarray(result.x if result.x is not None else desired, dtype=np.float64), self.lower, self.upper)
        candidate_values = self._constraint_values(state_array, candidate)
        feasible = bool(result.success and np.all(candidate_values >= -1e-8))
        fallback = not feasible
        if fallback:
            candidate = self._emergency(state_array)
            candidate_values = self._constraint_values(state_array, candidate)
        correction_norm = float(np.linalg.norm(candidate - desired))
        action_energy = float(np.sum(candidate**2))
        action_slew_norm = (
            0.0
            if previous_action is None
            else float(np.linalg.norm(candidate - previous_action))
        )
        self.previous_action = candidate.copy()
        info = FilterInfo(
            success=feasible,
            fallback=fallback,
            iterations=int(getattr(result, "nit", max_iter) or 0),
            min_constraint=float(np.min(candidate_values)),
            residual_margin=float(residual_margin),
            projected=bool(correction_norm > 1e-10),
            status=str(getattr(result, "message", "")),
            correction_norm=correction_norm,
            action_energy=action_energy,
            action_slew_norm=action_slew_norm,
            objective_value=float(objective(candidate)),
            intervention_budget=self.intervention_budget,
            intervention_budget_exceeded=bool(
                self.intervention_budget is not None
                and correction_norm > self.intervention_budget + 1e-10
            ),
        )
        return candidate.astype(np.float32), info.as_dict()

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """Update the adaptive residual window, if enabled."""
        if not self.adaptive:
            return
        predicted = self._call_dynamics(self.f_hat, np.asarray(state), np.asarray(action))
        residual = float(np.linalg.norm(np.asarray(next_state, dtype=np.float64) - predicted))
        if not np.isfinite(residual):
            raise ValueError("transition residual is NaN or Inf")
        self.residual_buffer.append(residual)
        self.residual_buffer = self.residual_buffer[-self.window_size :]
        self._updates += 1
        if self.event_residual_threshold is not None and residual > self.event_residual_threshold:
            self.event_trigger_count += 1
            self.event_residual_peak = max(self.event_residual_peak or residual, residual)
            self.q_alpha = max(self.q_alpha or 0.0, residual)
        event_selected = False
        if (
            self.event_selector_threshold is not None
            and residual > self.event_selector_threshold
            and self._event_selector_hold_remaining == 0
        ):
            selected = self.adaptive_parameter_selector.select_current_transition(state, action, next_state)
            self.parameter_samples = list(selected)
            self.event_selector_trigger_count += 1
            self._event_selector_hold_remaining = self.event_selector_hold_steps
            event_selected = True
        if self._updates % self.update_freq == 0 and len(self.residual_buffer) >= self.min_update_samples:
            updated_quantile = finite_sample_quantile(np.asarray(self.residual_buffer), self.alpha)
            self.last_adaptive_window_quantile = updated_quantile
            if self.preserve_calibration_floor and self.calibration_quantile is not None:
                updated_quantile = max(updated_quantile, self.calibration_quantile)
            if self.retain_adaptive_peak and self.q_alpha is not None:
                updated_quantile = max(updated_quantile, self.q_alpha)
            self.q_alpha = updated_quantile
            if self.parameter_set_factory is not None:
                samples = list(self.parameter_set_factory(self.q_alpha))
                if samples:
                    self.parameter_samples = samples
        if self.adaptive_parameter_selector is not None:
            selected = self.adaptive_parameter_selector.update(state, action, next_state)
            if self._event_selector_hold_remaining > 0 and not event_selected:
                self._event_selector_hold_remaining -= 1
            elif selected is not None:
                self.parameter_samples = list(selected)

    @property
    def update_count(self) -> int:
        """Return the number of empirical adaptive transitions consumed."""
        return self._updates
