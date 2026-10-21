"""Reference-preserving smoothing for one-dimensional ACC actions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from .racf import RACF


class ReferencePreservingSmoother:
    """Smooth a feasible RACF action without reducing its same-state reserve.

    This wrapper is deliberately ACC-specific: smaller normalized actions mean
    more braking. ``one_sided`` immediately accepts stronger braking and only
    smooths releases. ``reserve`` solves the regularized objective while
    retaining at least the reference action's predicted robust reserve.
    """

    MODES = ("one_sided", "reserve")

    def __init__(self, reference: RACF, mode: str, weight: float = 1.0) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if reference.lower.size != 1:
            raise ValueError(
                "reference-preserving smoothing currently requires one action dimension"
            )
        if reference.adaptive or reference.action_energy_weight or reference.action_slew_weight:
            raise ValueError("reference must be a static unregularized RACF filter")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and positive")
        self.reference = reference
        self.mode = mode
        self.weight = float(weight)
        self.previous_action: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reference, name)

    def set_calibrated_quantile(self, quantile: float) -> float:
        return self.reference.set_calibrated_quantile(quantile)

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        self.reference.update(state, action, next_state)

    def _candidate_values(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.reference._constraint_values(  # noqa: SLF001
            np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
        )

    def filter_action(
        self, state: np.ndarray, desired_action: np.ndarray, max_iter: int = 40
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        desired = np.clip(
            np.asarray(desired_action, dtype=np.float64).reshape(-1),
            self.reference.lower,
            self.reference.upper,
        )
        reference_action, reference_info = self.reference.filter_action(
            state_array, desired, max_iter=max_iter
        )
        reference_action64 = np.asarray(reference_action, dtype=np.float64)
        reference_values = self._candidate_values(state_array, reference_action64)
        reference_reserve = float(np.min(reference_values))
        smoothing_fallback = False
        smoothing_status = "initial_reference"
        candidate = reference_action64.copy()

        if reference_info["fallback"]:
            smoothing_status = "reference_emergency_fallback"
        elif self.previous_action is not None:
            if self.mode == "one_sided":
                unconstrained = (reference_action64 + self.weight * self.previous_action) / (
                    1.0 + self.weight
                )
                candidate = np.minimum(reference_action64, unconstrained)
                smoothing_status = "one_sided_candidate"
            else:
                required_reserve = max(0.0, reference_reserve)

                def objective(action: np.ndarray) -> float:
                    return float(
                        np.sum((action - desired) ** 2)
                        + self.weight * np.sum((action - self.previous_action) ** 2)
                    )

                constraints = [
                    {
                        "type": "ineq",
                        "fun": lambda action, theta=theta: (
                            self.reference._minimum_predicted_margin(  # noqa: SLF001
                                state_array,
                                action,
                                theta,
                                self.reference._residual_margin(),  # noqa: SLF001
                            )
                            - required_reserve
                        ),
                    }
                    for theta in self.reference.parameter_samples
                ]
                result = minimize(
                    objective,
                    reference_action64,
                    method="SLSQP",
                    bounds=list(zip(self.reference.lower, self.reference.upper, strict=True)),
                    constraints=constraints,
                    options={"maxiter": int(max_iter), "ftol": 1e-10},
                )
                if result.success and result.x is not None:
                    candidate = np.asarray(result.x, dtype=np.float64)
                    smoothing_status = "reserve_candidate"
                else:
                    smoothing_fallback = True
                    smoothing_status = f"reserve_solver_fallback: {result.message}"

            # The simulator consumes float32. Verify that exact applied value.
            candidate = np.clip(candidate, self.reference.lower, self.reference.upper)
            applied = candidate.astype(np.float32).astype(np.float64)
            required = max(0.0, reference_reserve)
            values = self._candidate_values(state_array, applied)
            if np.any(values < required - 1e-8):
                candidate = reference_action64.copy()
                smoothing_fallback = True
                smoothing_status = "post_cast_reserve_fallback"

        applied_action = candidate.astype(np.float32)
        applied_values = self._candidate_values(state_array, applied_action)
        if not reference_info["fallback"] and np.any(
            applied_values < max(0.0, reference_reserve) - 1e-8
        ):
            raise RuntimeError("reference reserve invariant failed after fallback")
        previous = None if self.previous_action is None else self.previous_action.copy()
        self.previous_action = applied_action.astype(np.float64)
        correction = float(np.linalg.norm(applied_action.astype(np.float64) - desired))
        slew = (
            0.0
            if previous is None
            else float(np.linalg.norm(applied_action.astype(np.float64) - previous))
        )
        info = dict(reference_info)
        info.update(
            success=bool(reference_info["success"]),
            fallback=bool(reference_info["fallback"]),
            projected=bool(correction > 1e-10),
            correction_norm=correction,
            action_energy=float(np.sum(applied_action.astype(np.float64) ** 2)),
            action_slew_norm=slew,
            objective_value=float(correction**2 + self.weight * slew**2),
            min_constraint=float(np.min(applied_values)),
            status=smoothing_status,
            smoothing_mode=self.mode,
            smoothing_weight=self.weight,
            smoothing_fallback=smoothing_fallback,
            reference_action=reference_action64.tolist(),
            reference_fallback=bool(reference_info["fallback"]),
            reference_min_constraint=reference_reserve,
            applied_reserve_loss=float(reference_reserve - np.min(applied_values)),
            float32_rechecked=True,
            projection_semantics="reference_preserving_racf_projection",
        )
        return applied_action, info
