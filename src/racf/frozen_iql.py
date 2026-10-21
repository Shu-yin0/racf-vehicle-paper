"""Lightweight inference for the frozen ACC IQL actor.

The deployed actor is an exact export of the frozen d3rlpy NormalPolicy used
by the registered protocol: Linear(5, 256), ReLU, Linear(256, 256), ReLU,
Linear(256, 1), then tanh.  It intentionally contains no training framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


_REQUIRED_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


class FrozenIQLActor:
    """NumPy implementation of the exported deterministic IQL action mean."""

    def __init__(self, weights_path: str | Path) -> None:
        self.weights_path = Path(weights_path)
        with np.load(self.weights_path, allow_pickle=False) as arrays:
            missing = [key for key in _REQUIRED_KEYS if key not in arrays]
            if missing:
                raise ValueError(f"actor export is missing arrays: {missing}")
            self.w1 = np.asarray(arrays["w1"], dtype=np.float32)
            self.b1 = np.asarray(arrays["b1"], dtype=np.float32)
            self.w2 = np.asarray(arrays["w2"], dtype=np.float32)
            self.b2 = np.asarray(arrays["b2"], dtype=np.float32)
            self.w3 = np.asarray(arrays["w3"], dtype=np.float32)
            self.b3 = np.asarray(arrays["b3"], dtype=np.float32)
        if self.w1.shape != (256, 5) or self.b1.shape != (256,):
            raise ValueError("actor export has an invalid first layer")
        if self.w2.shape != (256, 256) or self.b2.shape != (256,):
            raise ValueError("actor export has an invalid second layer")
        if self.w3.shape != (1, 256) or self.b3.shape != (1,):
            raise ValueError("actor export has an invalid output layer")

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Return the same squashed mean action as d3rlpy ``predict``."""
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError(f"observations must have shape (n, 5), got {values.shape}")
        hidden1 = np.maximum(values @ self.w1.T + self.b1, 0.0)
        hidden2 = np.maximum(hidden1 @ self.w2.T + self.b2, 0.0)
        return np.tanh(hidden2 @ self.w3.T + self.b3).astype(np.float32)
