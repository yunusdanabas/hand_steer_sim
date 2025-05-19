#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dispatcher.py

Factory for selecting the appropriate gesture recognizer
based on control mode.

Modes supported:
  - "static":   Static gesture recognition (discrete commands)
  - "steering": Continuous steering-wheel gesture recognition
"""

from typing import Any


class RecognizerFactory:
    """
    Factory to instantiate the appropriate recognizer:
      - GestureRecognition  (static_mode)
      - SteeringRecognition (steering_mode)
    """

    @staticmethod
    def create(mode: str, **kwargs) -> Any:
        """
        Create and return a recognizer instance based on the given mode.

        Args:
            mode:    One of "static" or "steering" (case-insensitive).
            **kwargs: Passed through to the recognizer constructor
                      (e.g. model_path, label_path, history_length).

        Returns:
            Instance of GestureRecognition or SteeringRecognition.

        Raises:
            ValueError: If mode is not recognized.
        """
        m = mode.lower()
        if m == "static":
            # static_mode uses the legacy static-gesture classifier
            from .model.static_mode.gesture_recognition import GestureRecognition
            return GestureRecognition(**kwargs)

        if m == "steering":
            # steering_mode uses your new two-hand wheel-angle regressor
            from .model.steering_mode.steering_recognition import SteeringRecognition
            return SteeringRecognition(**kwargs)

        raise ValueError(
            f"Unknown control_mode='{mode}'. "
            "Valid options are 'static' or 'steering'."
        )
