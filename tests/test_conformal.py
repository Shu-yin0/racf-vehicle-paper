import numpy as np
import pytest

from racf.conformal import (
    FiniteConditionConformalPredictor,
    SplitConformalPredictor,
    finite_sample_quantile,
)
from icassp2027.experiments.compute_shift_diagnostics import rbf_mmd_biased


def model(states, actions):
    return np.asarray(states) + np.asarray(actions)


def make_data(n, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, 2))
    actions = rng.normal(size=(n, 2))
    next_states = model(states, actions) + rng.normal(scale=noise, size=(n, 2))
    return states, actions, next_states


def test_finite_sample_rank_and_infinity_case():
    assert finite_sample_quantile(np.arange(1.0, 11.0), 0.1) == 10.0
    assert np.isinf(finite_sample_quantile(np.array([1.0]), 0.1))


def test_calibrate_and_nominal_coverage():
    predictor = SplitConformalPredictor(alpha=0.1, dynamics_model=model)
    predictor.calibrate(*make_data(500, seed=1))
    result = predictor.evaluate_coverage(*make_data(1000, seed=2))
    assert result["n_test"] == 1000
    assert abs(result["coverage"] - 0.9) < 0.05


def test_shift_changes_residual_not_covariate_mean():
    predictor = SplitConformalPredictor(alpha=0.1, dynamics_model=model)
    predictor.calibrate(*make_data(500, noise=0.05, seed=3))
    states, actions, next_states = make_data(1000, noise=0.30, seed=4)
    result = predictor.evaluate_coverage(states, actions, next_states)
    assert result["coverage"] < 0.9


def test_validation_and_calibration_order():
    predictor = SplitConformalPredictor(alpha=0.1, dynamics_model=model)
    with pytest.raises(RuntimeError):
        predictor.evaluate_coverage(*make_data(2))
    with pytest.raises(ValueError):
        predictor.calibrate(np.zeros((2, 2)), np.zeros((1, 2)), np.zeros((2, 2)))


def test_finite_condition_predictor_uses_conservative_maximum_quantile():
    def dynamics(states, actions, condition):
        return model(states, actions) + float(condition["bias"])

    nominal = make_data(100, noise=0.02, seed=5)
    shifted_states, shifted_actions, shifted_next = make_data(100, noise=0.02, seed=6)
    shifted_next = shifted_next + 0.4
    predictor = FiniteConditionConformalPredictor(alpha=0.1, dynamics_model=dynamics)
    quantiles = predictor.calibrate(
        {
            "nominal": ({"bias": 0.0}, nominal),
            "shifted": ({"bias": 0.0}, (shifted_states, shifted_actions, shifted_next)),
        }
    )

    assert quantiles["shifted"] > quantiles["nominal"]
    assert predictor.q_alpha == quantiles["shifted"]
    result = predictor.evaluate_coverage("nominal", {"bias": 0.0}, *make_data(200, noise=0.02, seed=7))
    assert result["quantile"] == predictor.q_alpha
    assert result["coverage"] >= 0.9


def test_finite_condition_predictor_labels_unregistered_test_condition():
    predictor = FiniteConditionConformalPredictor(
        alpha=0.1,
        dynamics_model=lambda states, actions, condition: model(states, actions),
    )
    predictor.calibrate({"nominal": ({}, make_data(100, seed=8))})

    result = predictor.evaluate_coverage("mass_light", {}, *make_data(100, seed=9))

    assert result["condition_quantile"] is None
    assert result["threshold_source"] == "global_maximum_unregistered_test_condition"


def test_rbf_mmd_detects_a_residual_distribution_shift():
    reference = np.linspace(0.0, 0.1, 100)
    matched_mmd, _ = rbf_mmd_biased(reference, reference.copy())
    shifted_mmd, bandwidth = rbf_mmd_biased(reference, reference + 0.5)

    assert matched_mmd == pytest.approx(0.0)
    assert shifted_mmd > matched_mmd
    assert bandwidth > 0.0

