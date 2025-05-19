#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from typing import List


class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='/home/yunusdanabas/catkin_ws/src/hand_steer_sim/hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
        use_gpu=False,
    ):
        delegates: List[tf.lite.experimental.Delegate] | None = None
        if use_gpu:
            try:
                delegates = [
                    tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
                ]
                print("[PointHistoryClassifier] GPU delegate loaded ✓")
            except (ValueError, OSError):
                # Could not load GPU delegate → fall back to CPU
                print("[PointHistoryClassifier] GPU delegate unavailable, using CPU.")

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
            experimental_delegates=delegates,
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
