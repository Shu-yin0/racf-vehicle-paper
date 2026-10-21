import numpy as np
import pytest

from racf.racf import RACF
from racf.smoothing import ReferencePreservingSmoother


def dynamics(state, action, theta=None):
    return (
        np.asarray(state, dtype=float)
        + np.asarray(action, dtype=float)
        + (theta or {}).get("offset", 0.0)
    )


def safe(state):
    return float(1.0 - np.asarray(state).reshape(-1)[0])


def make(mode, constraint=safe):
    reference = RACF(
        dynamics,
        constraint,
        ([-2.0], [2.0]),
        [{"offset": 0.0}, {"offset": 0.2}],
        emergency_action=[-2.0],
    )
    reference.set_calibrated_quantile(0.45)
    return ReferencePreservingSmoother(reference, mode, 1.0)


@pytest.mark.parametrize("mode", ["one_sided", "reserve"])
def test_stronger_reference_braking_is_never_delayed(mode):
    smoother = make(mode)
    smoother.previous_action = np.array([0.8])
    action, info = smoother.filter_action(np.array([1.0]), np.array([0.5]))
    assert action[0] == pytest.approx(info["reference_action"][0], abs=1e-6)
    assert info["applied_reserve_loss"] <= 1e-8
    assert info["float32_rechecked"]


def test_one_sided_smooths_release_toward_previous_braking():
    smoother = make("one_sided")
    smoother.previous_action = np.array([-0.8])
    action, info = smoother.filter_action(np.array([0.0]), np.array([0.5]))
    assert action[0] < info["reference_action"][0]
    assert action[0] == pytest.approx(-0.15, abs=1e-5)
    assert info["applied_reserve_loss"] <= 1e-8


def test_reserve_smoothing_never_reduces_same_state_reserve():
    smoother = make("reserve")
    smoother.previous_action = np.array([-0.8])
    action, info = smoother.filter_action(np.array([0.0]), np.array([0.5]))
    assert action[0] <= info["reference_action"][0] + 1e-6
    assert info["applied_reserve_loss"] <= 1e-8


@pytest.mark.parametrize("mode", ["one_sided", "reserve"])
def test_reference_emergency_fallback_is_executed_unchanged(mode):
    smoother = make(mode, lambda _: -1.0)
    smoother.previous_action = np.array([0.8])
    action, info = smoother.filter_action(np.array([0.0]), np.array([0.5]))
    assert action[0] == pytest.approx(-2.0)
    assert info["fallback"] and info["reference_fallback"]
    assert info["status"] == "reference_emergency_fallback"


def test_invalid_combinations_are_rejected():
    reference = RACF(dynamics, safe, ([-2.0], [2.0]), [{"offset": 0.0}], action_slew_weight=1.0)
    with pytest.raises(ValueError, match="unregularized"):
        ReferencePreservingSmoother(reference, "reserve")
    with pytest.raises(ValueError, match="mode"):
        ReferencePreservingSmoother(
            RACF(dynamics, safe, ([-2.0], [2.0]), [{"offset": 0.0}]), "unknown"
        )

