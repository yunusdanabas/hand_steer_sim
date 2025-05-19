#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gesture_recognition.py – MediaPipe-based static-gesture classifier
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import csv
import cv2 as cv
import numpy as np
import mediapipe as mp

from hand_steer_sim.model.static_mode import KeyPointClassifier


mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GestureRecognition:
    """One-shot wrapper around MediaPipe Hands + TFLite classifier."""

    def __init__(
        self,
        label_path: str | Path,
        model_path: str | Path,
        *,
        use_gpu: bool = False,
        static_image_mode: bool = False,
        min_det_conf: float = 0.7,
        min_track_conf: float = 0.7,
    ) -> None:

        self._labels: list[str] = self._load_labels(label_path)
        self._classifier        = KeyPointClassifier(str(model_path), use_gpu=use_gpu)

        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def recognise(self, bgr_img: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Args
        ----
        bgr_img : np.ndarray
            Original BGR frame (will be mirrored for user convenience).

        Returns
        -------
        debug_img : np.ndarray
            Frame annotated with landmarks / label.
        gesture : str
            Predicted gesture label, or "NONE".
        """
        bgr_img = cv.flip(bgr_img, 1)           # mirror for UI
        debug   = bgr_img.copy()

        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        results = self._mp_hands.process(rgb_img)

        gesture = "NONE"
        if results.multi_hand_landmarks:
            for hand_lms, handed in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
                lm_px   = self._pixel_landmarks(debug, hand_lms)
                feature = self._normalise_landmarks(lm_px)
                gesture = self._labels[self._classifier(feature)]

                self._draw_overlay(debug, hand_lms, handed.classification[0].label, gesture)

        return debug, gesture

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_labels(csv_path: str | Path) -> list[str]:
        with open(csv_path, encoding="utf-8-sig") as f:
            return [row[0] for row in csv.reader(f)]

    @staticmethod
    def _pixel_landmarks(img: np.ndarray, hand_lms) -> np.ndarray:
        h, w = img.shape[:2]
        return np.array([[lm.x * w, lm.y * h] for lm in hand_lms.landmark],
                        dtype=np.float32)

    @staticmethod
    def _normalise_landmarks(lms_px: np.ndarray) -> list[float]:
        lms_px -= lms_px[0]                       # wrist-relative
        flat    = lms_px.flatten()
        flat /= (np.abs(flat).max() or 1.0)       # scale to ±1
        return flat.tolist()

    @staticmethod
    def _draw_overlay(img, hand_lms, hand_label, gesture_label) -> None:
        mp_drawing.draw_landmarks(
            img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        h, w = img.shape[:2]
        wrist = hand_lms.landmark[0]
        cv.putText(
            img, f"{hand_label}:{gesture_label}",
            (int(wrist.x * w), int(wrist.y * h) - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA
        )

    # ------------------------------------------------------------------ #
    # misc utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_fps_info(img: np.ndarray, fps: float) -> np.ndarray:
        cv.putText(img, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),   4, cv.LINE_AA)
        cv.putText(img, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return img
