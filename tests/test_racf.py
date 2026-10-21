import numpy as np
import pytest

from racf.racf import FiniteResidualParameterSelector, RACF


def dynamics(state, action, theta=None):
    theta = theta or {"offset": 0.0}
    return np.asarray(state, dtype=float) + np.asarray(action, dtype=float) + theta.get("offset", 0.0)


def safe(next_state):
    return float(1.0 - np.asarray(next_state).reshape(-1)[0])


def make_filter(adaptive=False, emergency=None):
    return RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}, {"offset": 0.2}],
        adaptive=adaptive,
        emergency_action=emergency,
    )


def test_filter_projects_to_robust_feasible_action():
    filt = make_filter()
    action, info = filt.filter_action(np.array([0.0]), np.array([2.0]))
    assert action[0] <= 0.8 + 1e-5
    assert info["success"] is True
    assert info["projected"] is True


def test_utility_weights_preserve_safety_and_report_intervention_budget():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        action_energy_weight=0.25,
        action_slew_weight=0.5,
        intervention_budget=0.1,
    )
    action, info = filt.filter_action(np.array([0.0]), np.array([2.0]))
    assert -2.0 <= action[0] <= 2.0
    assert info["success"] is True
    assert info["correction_norm"] >= 0.0
    assert info["objective_value"] >= 0.0
    assert info["intervention_budget"] == pytest.approx(0.1)
    assert info["intervention_budget_exceeded"] is True
    assert info["projection_semantics"] == "racf_robust_projection"


def test_calibrated_residual_margin_tightens_the_safety_constraint():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        safety_lipschitz=1.0,
        emergency_action=[-2.0],
    )
    states = np.zeros((20, 1))
    actions = np.zeros((20, 1))
    next_states = np.full((20, 1), 0.2)
    assert filt.calibrate(states, actions, next_states) == pytest.approx(0.2)

    action, info = filt.filter_action(np.array([0.0]), np.array([1.0]))
    assert action[0] <= 0.8 + 1e-5
    assert info["residual_margin"] == pytest.approx(0.2)


def test_registered_threshold_can_initialize_a_static_filter():
    filt = make_filter()
    assert filt.set_calibrated_quantile(0.15) == pytest.approx(0.15)
    _, info = filt.filter_action(np.array([0.0]), np.array([0.0]))
    assert info["residual_margin"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        filt.set_calibrated_quantile(float("nan"))


def test_infeasible_filter_uses_bounded_emergency_action():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=lambda _: -1.0,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        emergency_action=[-1.5],
    )
    action, info = filt.filter_action(np.array([0.0]), np.array([0.0]))
    assert action[0] == pytest.approx(-1.5)
    assert info["fallback"] is True


def test_adaptive_window_updates_threshold_at_frequency():
    filt = make_filter(adaptive=True)
    states = np.zeros((20, 1))
    actions = np.zeros((20, 1))
    next_states = np.zeros((20, 1))
    filt.calibrate(states, actions, next_states)
    initial = filt.q_alpha
    for _ in range(9):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.5]))
    assert filt.q_alpha == initial
    for _ in range(1):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.5]))
    assert filt.q_alpha is not None
    assert filt.q_alpha >= initial


def test_guarded_adaptive_window_never_drops_below_calibration_quantile():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        adaptive=True,
        window_size=10,
        update_freq=10,
        min_update_samples=10,
        preserve_calibration_floor=True,
    )
    filt.set_calibrated_quantile(0.25)
    for _ in range(10):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.01]))
    assert filt.q_alpha == pytest.approx(0.25)
    assert filt.last_adaptive_window_quantile == pytest.approx(0.01)
    for _ in range(10):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.5]))
    assert filt.q_alpha == pytest.approx(0.5)
    assert filt.last_adaptive_window_quantile == pytest.approx(0.5)


def test_monotone_adaptive_window_retains_an_observed_residual_peak():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        adaptive=True,
        window_size=10,
        update_freq=10,
        min_update_samples=10,
        preserve_calibration_floor=True,
        retain_adaptive_peak=True,
    )
    filt.set_calibrated_quantile(0.25)
    for _ in range(10):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.5]))
    assert filt.q_alpha == pytest.approx(0.5)
    for _ in range(10):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.01]))
    assert filt.last_adaptive_window_quantile == pytest.approx(0.01)
    assert filt.q_alpha == pytest.approx(0.5)


def test_event_trigger_retains_an_isolated_extreme_residual_immediately():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        adaptive=True,
        window_size=100,
        update_freq=10,
        min_update_samples=20,
        preserve_calibration_floor=True,
        retain_adaptive_peak=True,
        event_residual_threshold=0.4,
    )
    filt.set_calibrated_quantile(0.25)
    filt.update(np.array([0.0]), np.array([0.0]), np.array([0.8]))

    assert filt.q_alpha == pytest.approx(0.8)
    assert filt.event_trigger_count == 1
    assert filt.event_residual_peak == pytest.approx(0.8)


def test_prediction_horizon_enforces_the_minimum_margin_over_repeated_action_steps():
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=[{"offset": 0.0}],
        prediction_horizon=2,
    )

    action, info = filt.filter_action(np.array([0.0]), np.array([0.8]))

    assert action[0] <= 0.5 + 1e-5
    assert info["success"] is True


def test_selector_can_immediately_choose_candidates_that_explain_an_event():
    selector = FiniteResidualParameterSelector(
        dynamics,
        [{"offset": 0.0}, {"offset": 0.8}],
        max_active=1,
    )

    selected = selector.select_current_transition(
        np.array([0.0]), np.array([0.0]), np.array([0.75])
    )

    assert selected == [{"offset": 0.8}]
    assert selector.event_selection_count == 1


def test_event_selector_does_not_retrigger_during_its_hold_window():
    selector = FiniteResidualParameterSelector(dynamics, [{"offset": 0.0}, {"offset": 0.8}], max_active=1)
    filt = RACF(
        nominal_dynamics=dynamics,
        safety_constraint=safe,
        action_bounds=([-2.0], [2.0]),
        parameter_samples=selector.active_samples,
        adaptive=True,
        adaptive_parameter_selector=selector,
        event_selector_threshold=0.4,
        event_selector_hold_steps=5,
    )
    filt.set_calibrated_quantile(0.1)
    for _ in range(3):
        filt.update(np.array([0.0]), np.array([0.0]), np.array([0.8]))

    assert filt.event_selector_trigger_count == 1
    assert selector.event_selection_count == 1


def test_invalid_action_dimension_is_rejected():
    with pytest.raises(ValueError):
        make_filter().filter_action(np.array([0.0]), np.array([0.0, 0.0]))


def test_finite_residual_selector_keeps_full_grid_then_selects_top_k():
    selector = FiniteResidualParameterSelector(
        dynamics,
        [{"offset": 0.0}, {"offset": 0.2}, {"offset": 1.0}],
        window_size=10,
        update_freq=2,
        min_update_samples=4,
        max_active=2,
    )
    assert selector.active_samples == [{"offset": 0.0}, {"offset": 0.2}, {"offset": 1.0}]
    for _ in range(3):
        assert selector.update(np.array([0.0]), np.array([0.0]), np.array([0.05])) is None
    selected = selector.update(np.array([0.0]), np.array([0.0]), np.array([0.05]))

    assert selected == [{"offset": 0.0}, {"offset": 0.2}]
    assert selector.selection_count == 1
    assert selector.update_count == 4
    assert selector.quantiles is not None

