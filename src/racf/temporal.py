"""Frozen temporal schedules and predictable candidate tracking for RACF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .racf import FiniteResidualParameterSelector


@dataclass(frozen=True)
class ScheduleSegment:
    """A half-open segment whose parameters are interpolated over time."""

    start_step: int
    end_step: int
    grade_start: float
    grade_end: float
    gain_start: float
    gain_end: float
    mode: str = "linear"

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("schedule segment must have 0 <= start_step < end_step")
        if self.mode not in {"step", "linear"}:
            raise ValueError("schedule segment mode must be step or linear")
        values = (self.grade_start, self.grade_end, self.gain_start, self.gain_end)
        if not all(np.isfinite(values)) or self.gain_start <= 0.0 or self.gain_end <= 0.0:
            raise ValueError("schedule segment values must be finite and gains positive")

    def value_at(self, step: int) -> dict[str, float]:
        if self.mode == "step":
            fraction = 1.0
        elif self.end_step - self.start_step == 1:
            fraction = 1.0
        else:
            fraction = (step - self.start_step) / (self.end_step - self.start_step - 1)
            fraction = float(np.clip(fraction, 0.0, 1.0))
        return {
            "grade": float(self.grade_start + fraction * (self.grade_end - self.grade_start)),
            "actuator_force_gain": float(
                self.gain_start + fraction * (self.gain_end - self.gain_start)
            ),
        }


class TemporalSchedule:
    """Deterministic, hashable schedule evaluated before each environment step."""

    def __init__(
        self,
        schedule_id: str,
        horizon: int,
        base_grade: float = 0.0,
        base_gain: float = 1.0,
        segments: Iterable[ScheduleSegment] = (),
    ) -> None:
        self.schedule_id = str(schedule_id)
        self.horizon = int(horizon)
        self.base_grade = float(base_grade)
        self.base_gain = float(base_gain)
        self.segments = tuple(segments)
        if not self.schedule_id or self.horizon <= 0:
            raise ValueError("schedule_id and positive horizon are required")
        if not np.isfinite((self.base_grade, self.base_gain)).all() or self.base_gain <= 0.0:
            raise ValueError("base schedule values must be finite and base_gain positive")
        previous_end = 0
        for segment in self.segments:
            if segment.start_step < previous_end or segment.end_step > self.horizon:
                raise ValueError("schedule segments must be ordered, non-overlapping, and inside horizon")
            previous_end = segment.end_step

    def parameters_at(self, step: int) -> dict[str, Any]:
        step = int(step)
        if step < 0 or step >= self.horizon:
            raise IndexError("schedule step is outside the frozen horizon")
        result = {"grade": self.base_grade, "actuator_force_gain": self.base_gain}
        for segment in self.segments:
            if segment.start_step <= step < segment.end_step:
                result.update(segment.value_at(step))
                break
        result["schedule_id"] = self.schedule_id
        result["step"] = step
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "horizon": self.horizon,
            "base_grade": self.base_grade,
            "base_gain": self.base_gain,
            "segments": [segment.__dict__.copy() for segment in self.segments],
        }


class PredictableTemporalTracker:
    """Candidate tracker whose update at t can only affect action t+1 onward."""

    def __init__(
        self,
        nominal_dynamics: Callable[..., np.ndarray],
        candidates: Iterable[Any],
        *,
        window_size: int = 40,
        update_frequency: int = 10,
        min_samples: int = 20,
        max_active: int = 4,
    ) -> None:
        self.selector = FiniteResidualParameterSelector(
            nominal_dynamics,
            candidates,
            window_size=window_size,
            update_freq=update_frequency,
            min_update_samples=min_samples,
            max_active=max_active,
        )
        self.window_size = int(window_size)
        self.update_frequency = int(update_frequency)
        self.history: list[dict[str, Any]] = []

    def observe(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, step: int) -> list[Any] | None:
        """Consume transition t and return an active set for the next action."""
        selected = self.selector.update(state, action, next_state)
        self.history.append(
            {
                "step": int(step),
                "updated": selected is not None,
                "active_count": len(self.selector.active_samples),
            }
        )
        return selected

    @property
    def active_samples(self) -> list[Any]:
        return self.selector.active_samples

    @property
    def update_count(self) -> int:
        return self.selector.update_count

    @property
    def selection_count(self) -> int:
        return self.selector.selection_count
