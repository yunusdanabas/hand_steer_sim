#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keypoint_classifier.py ― thin TFLite wrapper with **optional GPU delegate**.

If `use_gpu=True` we try to load the standard TensorFlow‑Lite GPU delegate.
If that fails (e.g. CPU‑only machine) we silently fall back to pure‑CPU so
the rest of the pipeline keeps working.
"""

from __future__ import annotations
from pathlib import Path
from typing   import List

import numpy as np
import tensorflow as tf


class KeyPointClassifier:
    def __init__(
        self,
        model_path: str | Path = "keypoint_classifier.tflite",
        *,
        num_threads: int = 1,
        use_gpu: bool = False,
    ) -> None:

        delegates: List[tf.lite.experimental.Delegate] | None = None
        if use_gpu:
            try:
                delegates = [
                    tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
                ]
                print("[KeyPointClassifier] GPU delegate loaded ✓")
            except (ValueError, OSError):
                # Could not load GPU delegate → fall back to CPU
                print("[KeyPointClassifier] GPU delegate unavailable, using CPU.")

        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
            experimental_delegates=delegates,
        )
        self.interpreter.allocate_tensors()
        self._in_index  = self.interpreter.get_input_details()[0]["index"]
        self._out_index = self.interpreter.get_output_details()[0]["index"]

    # ------------------------------------------------------------------ #
    def __call__(self, landmark_vec: list[float]) -> int:
        """
        Parameters
        ----------
        landmark_vec : list[float]
            Flattened, normalised key‑point vector (length 42).

        Returns
        -------
        int  →  predicted class ID (0‑based)
        """
        self.interpreter.set_tensor(
            self._in_index, np.asarray([landmark_vec], dtype=np.float32)
        )
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._out_index)[0]
        return int(np.argmax(probs))
