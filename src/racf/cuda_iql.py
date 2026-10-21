"""ctypes wrapper around the node-local CUDA backend for the frozen IQL actor."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class CudaFrozenIQLActor:
    """Inference-only actor that uses the one GPU visible to its Slurm task."""

    def __init__(self, library_path: str | Path, weights_path: str | Path) -> None:
        self._closed = True
        self._library = ctypes.CDLL(str(Path(library_path).resolve()))
        self._library.racf_actor_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self._library.racf_actor_init.restype = ctypes.c_int
        self._library.racf_actor_predict.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_size_t
        ]
        self._library.racf_actor_predict.restype = ctypes.c_int
        self._library.racf_actor_close.argtypes = []
        self._library.racf_actor_close.restype = None
        self._initialize(weights_path)
        self._closed = False

    def _initialize(self, weights_path: str | Path) -> None:
        error = ctypes.create_string_buffer(512)
        result = self._library.racf_actor_init(str(weights_path).encode("utf-8"), error, len(error))
        if result != 0:
            raise RuntimeError(f"CUDA actor initialization failed: {error.value.decode('utf-8', 'replace')}")

    def predict(self, observations: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(np.asarray(observations, dtype=np.float32))
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError(f"observations must have shape (n, 5), got {values.shape}")
        actions = np.empty((values.shape[0], 1), dtype=np.float32)
        error = ctypes.create_string_buffer(512)
        for index, observation in enumerate(values):
            result = self._library.racf_actor_predict(
                observation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                actions[index].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                error,
                len(error),
            )
            if result != 0:
                raise RuntimeError(f"CUDA actor prediction failed: {error.value.decode('utf-8', 'replace')}")
        return actions

    def close(self) -> None:
        if not self._closed:
            self._library.racf_actor_close()
            self._closed = True

    def __del__(self) -> None:
        self.close()
