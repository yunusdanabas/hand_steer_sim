#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steering_recognition.py

Placeholder “SteeringRecognition”

"""

import cv2
import numpy as np

class SteeringRecognition:
    def __init__(self, label_path: str, model_path: str, **kwargs):
        # You can log initialization if you import rospy here:
        try:
            import rospy
            rospy.loginfo(f"SteeringRecognition initialized with model: {model_path}")
        except ImportError:
            pass

        # Store paths in case you need them later
        self.label_path = label_path
        self.model_path = model_path

    def recognise(self, bgr_img: np.ndarray):

        debug = bgr_img.copy()
        h, w = debug.shape[:2]

        # Draw a simple overlay so you see Steering mode is active
        cv2.putText(
            debug,
            "STEERING MODE (placeholder)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        angle = 0.0
        return debug, angle
