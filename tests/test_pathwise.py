import numpy as np
import pytest

from racf.pathwise import PathwiseResidualEnvelope


def dynamics(states, actions):
    return np.asarray(states) + np.asarray(actions)


def make_episodes(n=20, horizon=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, horizon, 2))
    actions = rng.normal(size=(n, horizon, 2))
    next_states = dynamics(states, actions) + rng.normal(scale=noise, size=(n, horizon, 2))
    return states, actions, next_states


def test_pathwise_calibration_uses_complete_episode_max_scores():
    predictor = PathwiseResidualEnvelope(alpha=0.1, dynamics_model=dynamics)
    calibration = make_episodes(n=30, seed=1)
    q = predictor.calibrate(*calibration)
    assert q >= 0.0
    assert predictor.calibration_max_scores.shape == (30,)
    result = predictor.evaluate_coverage(*make_episodes(n=100, seed=2))
    assert result["n_episodes"] == 100
    assert result["horizon"] == 5
    assert 0.0 <= result["coverage"] <= 1.0


def test_pathwise_rejects_transition_batches_and_nonfinite_values():
    predictor = PathwiseResidualEnvelope(dynamics_model=dynamics)
    states, actions, next_states = make_episodes()
    with pytest.raises(ValueError):
        predictor.calibrate(states.reshape(-1, 2), actions.reshape(-1, 2), next_states.reshape(-1, 2))
    next_states = next_states.copy()
    next_states[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        predictor.calibrate(states, actions, next_states)


def test_pathwise_does_not_evaluate_before_calibration():
    predictor = PathwiseResidualEnvelope(dynamics_model=dynamics)
    with pytest.raises(RuntimeError):
        predictor.evaluate_coverage(*make_episodes())

