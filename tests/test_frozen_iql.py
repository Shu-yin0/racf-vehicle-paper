import numpy as np
import pytest

from racf.frozen_iql import FrozenIQLActor


def _weights(path):
    w1 = np.zeros((256, 5), dtype=np.float32)
    b1 = np.zeros(256, dtype=np.float32)
    w2 = np.zeros((256, 256), dtype=np.float32)
    b2 = np.zeros(256, dtype=np.float32)
    w3 = np.zeros((1, 256), dtype=np.float32)
    b3 = np.zeros(1, dtype=np.float32)
    w1[0, 0] = 2.0
    b1[0] = -0.5
    w2[0, 0] = 3.0
    b2[0] = -0.25
    w3[0, 0] = 0.5
    b3[0] = 0.1
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)


def test_frozen_iql_actor_matches_explicit_mlp(tmp_path):
    path = tmp_path / "actor.npz"
    _weights(path)
    actor = FrozenIQLActor(path)

    observations = np.array([[1.0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0]], dtype=np.float32)
    result = actor.predict(observations)

    expected = np.tanh(np.array([[(max(2.0 - 0.5, 0.0) * 3.0 - 0.25) * 0.5 + 0.1], [0.1]], dtype=np.float32))
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-7)


def test_frozen_iql_actor_rejects_invalid_observations(tmp_path):
    path = tmp_path / "actor.npz"
    _weights(path)
    actor = FrozenIQLActor(path)

    with pytest.raises(ValueError, match="shape"):
        actor.predict(np.zeros((2, 4), dtype=np.float32))

