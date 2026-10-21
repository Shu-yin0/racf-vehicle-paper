"""Small adapter between the existing ACC environment and RACF interfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

EXACT_PHYSICS_MODEL = "exact_physics_reference"
REDUCED_PHYSICS_MODEL = "reduced_physics_v1"
ACTUATOR_GAIN_MODEL = "reduced_physics_actuator_gain_v1"
ACC_NOMINAL_MODEL_SPEC = {
    "name": REDUCED_PHYSICS_MODEL,
    "drag_coefficient": 0.24,
    "rolling_coefficient": 0.010,
    "lead_dynamics": "constant_speed",
    "purpose": "Fixed reduced-physics nominal model for a controlled model-error diagnostic.",
}
ACC_ACTUATOR_GAIN_MODEL_SPEC = {
    **ACC_NOMINAL_MODEL_SPEC,
    "name": ACTUATOR_GAIN_MODEL,
    "actuator_force_gain": 0.70,
    "force_midpoint_newtons": -1000.0,
    "purpose": "Fixed reduced-physics and actuator-calibration error model for V4 mechanism tests.",
}


def acc_state_from_env(env: Any) -> np.ndarray:
    """Return physical ACC state used by the pilot dynamics model."""
    return np.asarray(
        [env.ego_speed, env.lead_speed, env.gap, env.desired_speed, np.degrees(env.grade)],
        dtype=np.float64,
    )


def acc_state_from_observation(observation: np.ndarray, env: Any) -> np.ndarray:
    """Invert the existing normalized ACC observation into physical state units."""
    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("ACC observation must be a finite vector with shape (5,)")
    desired_speed = float(env.desired_speed)
    ego_speed = values[0] * float(env.speed_scale) + desired_speed
    closing_speed = values[1] * float(env.relative_speed_scale)
    lead_speed = ego_speed - closing_speed
    gap = values[2] * float(env.gap_scale)
    grade_degrees = values[3] * float(np.degrees(env.grade_scale))
    return np.asarray(
        [ego_speed, lead_speed, gap, desired_speed, grade_degrees], dtype=np.float64
    )


def acc_transition_model(state: np.ndarray, action: np.ndarray, theta: dict[str, float] | None = None, dt: float = 0.1) -> np.ndarray:
    """Predict one deterministic ACC transition for a finite mass/grade sample."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.shape != (5,):
        raise ValueError("ACC state must contain ego speed, lead speed, gap, desired speed, and grade")
    params = theta or {}
    mass = float(params.get("mass", 1500.0))
    grade = float(params.get("grade", values[4]))
    lead_acceleration = float(params.get("lead_acceleration", 0.0))
    if mass <= 0.0 or not np.isfinite(lead_acceleration):
        raise ValueError("mass must be positive and lead_acceleration finite")
    ego_speed, lead_speed, gap, desired_speed, _ = values
    normalized_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
    force = -6000.0 + 0.5 * (normalized_action + 1.0) * 10000.0
    resistance = 0.5 * 1.225 * 0.30 * 2.2 * ego_speed * abs(ego_speed)
    resistance += mass * 9.81 * (0.012 * np.cos(np.deg2rad(grade)) + np.sin(np.deg2rad(grade)))
    acceleration = max((force - resistance) / mass, -ego_speed / dt)
    next_ego = max(0.0, ego_speed + acceleration * dt)
    next_lead = max(0.0, lead_speed + lead_acceleration * dt)
    next_gap = gap + (lead_speed + next_lead - ego_speed - next_ego) * 0.5 * dt
    return np.asarray([next_ego, next_lead, next_gap, desired_speed, grade], dtype=np.float64)


def acc_nominal_transition_model(
    state: np.ndarray,
    action: np.ndarray,
    theta: dict[str, float] | None = None,
    dt: float = 0.1,
) -> np.ndarray:
    """Return the fixed reduced-physics nominal model used in the V3 diagnostic.

    The registered mass and grade remain explicit finite parameters.  Drag and
    rolling resistance are intentionally lower than the simulator's values and
    are fixed before calibration, so this is an auditable model-error setting
    rather than an oracle reconstruction of the environment.
    """
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.shape != (5,):
        raise ValueError("ACC state must contain ego speed, lead speed, gap, desired speed, and grade")
    params = theta or {}
    mass = float(params.get("mass", 1500.0))
    grade = float(params.get("grade", values[4]))
    lead_acceleration = float(params.get("lead_acceleration", 0.0))
    if mass <= 0.0 or not np.isfinite(lead_acceleration):
        raise ValueError("mass must be positive and lead_acceleration finite")
    ego_speed, lead_speed, gap, desired_speed, _ = values
    normalized_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
    force = -6000.0 + 0.5 * (normalized_action + 1.0) * 10000.0
    resistance = (
        0.5
        * 1.225
        * ACC_NOMINAL_MODEL_SPEC["drag_coefficient"]
        * 2.2
        * ego_speed
        * abs(ego_speed)
    )
    resistance += mass * 9.81 * (
        ACC_NOMINAL_MODEL_SPEC["rolling_coefficient"] * np.cos(np.deg2rad(grade))
        + np.sin(np.deg2rad(grade))
    )
    acceleration = max((force - resistance) / mass, -ego_speed / dt)
    next_ego = max(0.0, ego_speed + acceleration * dt)
    next_lead = max(0.0, lead_speed + lead_acceleration * dt)
    next_gap = gap + (lead_speed + next_lead - ego_speed - next_ego) * 0.5 * dt
    return np.asarray([next_ego, next_lead, next_gap, desired_speed, grade], dtype=np.float64)


def acc_actuator_gain_transition_model(
    state: np.ndarray,
    action: np.ndarray,
    theta: dict[str, float] | None = None,
    dt: float = 0.1,
) -> np.ndarray:
    """Return the V4 fixed actuator-gain-error nominal dynamics model."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.shape != (5,):
        raise ValueError("ACC state must contain ego speed, lead speed, gap, desired speed, and grade")
    params = theta or {}
    mass = float(params.get("mass", 1500.0))
    grade = float(params.get("grade", values[4]))
    lead_acceleration = float(params.get("lead_acceleration", 0.0))
    if mass <= 0.0 or not np.isfinite(lead_acceleration):
        raise ValueError("mass must be positive and lead_acceleration finite")
    ego_speed, lead_speed, gap, desired_speed, _ = values
    actuator_force_gain = float(
        params.get("actuator_force_gain", ACC_ACTUATOR_GAIN_MODEL_SPEC["actuator_force_gain"])
    )
    if actuator_force_gain <= 0.0:
        raise ValueError("actuator_force_gain must be positive")
    normalized_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
    force = ACC_ACTUATOR_GAIN_MODEL_SPEC["force_midpoint_newtons"] + (
        actuator_force_gain * 5000.0 * normalized_action
    )
    resistance = (
        0.5
        * 1.225
        * ACC_ACTUATOR_GAIN_MODEL_SPEC["drag_coefficient"]
        * 2.2
        * ego_speed
        * abs(ego_speed)
    )
    resistance += mass * 9.81 * (
        ACC_ACTUATOR_GAIN_MODEL_SPEC["rolling_coefficient"] * np.cos(np.deg2rad(grade))
        + np.sin(np.deg2rad(grade))
    )
    acceleration = max((force - resistance) / mass, -ego_speed / dt)
    next_ego = max(0.0, ego_speed + acceleration * dt)
    next_lead = max(0.0, lead_speed + lead_acceleration * dt)
    next_gap = gap + (lead_speed + next_lead - ego_speed - next_ego) * 0.5 * dt
    return np.asarray([next_ego, next_lead, next_gap, desired_speed, grade], dtype=np.float64)


def acc_dynamics_model(name: str):
    """Resolve a named fixed ACC model without exposing simulator internals."""
    models = {
        EXACT_PHYSICS_MODEL: acc_transition_model,
        REDUCED_PHYSICS_MODEL: acc_nominal_transition_model,
        ACTUATOR_GAIN_MODEL: acc_actuator_gain_transition_model,
    }
    try:
        return models[name]
    except KeyError as error:
        raise ValueError(f"unknown ACC dynamics model: {name}") from error


def acc_safety_margin(state: np.ndarray) -> float:
    """Return the minimum of gap and speed safety margins (non-negative is safe)."""
    ego_speed, _, gap, _, _ = np.asarray(state, dtype=np.float64).reshape(-1)
    safe_gap = 3.0 + 1.5 * max(ego_speed, 0.0)
    return float(min(gap - safe_gap, 30.0 - ego_speed))


def default_parameter_samples(masses: Iterable[float] = (1500.0, 1800.0), grades: Iterable[float] = (0.0, 3.0)) -> list[dict[str, float]]:
    """Return the finite parameter grid used by the MVP and pilot."""
    return [{"mass": float(mass), "grade": float(grade)} for mass in masses for grade in grades]


def actuator_gain_parameter_samples(
    masses: Iterable[float] = (1500.0, 1800.0),
    grades: Iterable[float] = (0.0, 3.0),
    actuator_force_gains: Iterable[float] = (0.7, 1.0),
) -> list[dict[str, float]]:
    """Return the auditable finite mass, grade, and actuator-gain grid.

    This grid is used only by the empirical adaptive-grid diagnostic. It does
    not imply a guarantee beyond its explicitly listed parameter samples.
    """
    samples = [
        {
            "mass": float(mass),
            "grade": float(grade),
            "actuator_force_gain": float(gain),
        }
        for mass in masses
        for grade in grades
        for gain in actuator_force_gains
    ]
    if not samples or any(sample["mass"] <= 0.0 or sample["actuator_force_gain"] <= 0.0 for sample in samples):
        raise ValueError("finite actuator-gain parameter samples must be non-empty and positive")
    return samples


def registered_condition_actuator_gain_parameter_samples(
    condition_pairs: Iterable[tuple[float, float]] = (
        (1500.0, 0.0),
        (1800.0, 0.0),
        (1500.0, 3.0),
        (1800.0, 2.0),
    ),
    actuator_force_gains: Iterable[float] = (0.7, 1.0),
) -> list[dict[str, float]]:
    """Return the exact finite grid registered for the V24 four-condition study.

    Unlike a Cartesian mass/grade product, this grid preserves the four
    protocol condition pairs exactly, including the combined 1800 kg, 2 degree
    condition. The returned ordering is stable and therefore hashable.
    """
    pairs = [(float(mass), float(grade)) for mass, grade in condition_pairs]
    gains = [float(gain) for gain in actuator_force_gains]
    samples = [
        {"mass": mass, "grade": grade, "actuator_force_gain": gain}
        for mass, grade in pairs
        for gain in gains
    ]
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or not gains
        or len(set(gains)) != len(gains)
        or any(mass <= 0.0 or not np.isfinite(grade) for mass, grade in pairs)
        or any(gain <= 0.0 or not np.isfinite(gain) for gain in gains)
    ):
        raise ValueError("registered condition pairs and actuator gains must be unique, non-empty, and valid")
    return samples


def light_mass_actuator_gain_parameter_samples() -> list[dict[str, float]]:
    """Return the finite empirical grid that adds the diagnosed 1200 kg case."""
    return actuator_gain_parameter_samples(masses=(1200.0, 1500.0, 1800.0))


def lead_acceleration_parameter_samples(
    masses: Iterable[float] = (1500.0, 1800.0),
    grades: Iterable[float] = (0.0, 3.0),
    actuator_force_gains: Iterable[float] = (0.7, 1.0),
    lead_accelerations: Iterable[float] = (0.0, -4.0),
) -> list[dict[str, float]]:
    """Return a finite grid including the observed lead-braking mode.

    This is an empirical finite-hypothesis grid. It does not imply coverage
    over arbitrary lead trajectories or unlisted deceleration values.
    """
    samples = [
        {
            "mass": float(mass),
            "grade": float(grade),
            "actuator_force_gain": float(gain),
            "lead_acceleration": float(lead_acceleration),
        }
        for mass in masses
        for grade in grades
        for gain in actuator_force_gains
        for lead_acceleration in lead_accelerations
    ]
    if not samples or any(
        sample["mass"] <= 0.0
        or sample["actuator_force_gain"] <= 0.0
        or not np.isfinite(sample["lead_acceleration"])
        for sample in samples
    ):
        raise ValueError("finite lead-acceleration samples must be non-empty and valid")
    return samples


def parameter_samples_from_residual(residual_quantile: float, trigger: float = 1e-3) -> list[dict[str, float]]:
    """Expand the finite ACC uncertainty grid after a residual-shift trigger.

    This is an empirical adaptive rule: the trigger is a pilot diagnostic, not
    a TV estimate or a conformal coverage guarantee.
    """
    if trigger <= 0.0:
        raise ValueError("trigger must be positive")
    if residual_quantile <= trigger:
        return [{"mass": 1500.0, "grade": 0.0}]
    return default_parameter_samples()
