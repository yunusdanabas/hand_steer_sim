#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steering_recognition.py – 16-frame dynamic-gesture classifier
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import csv, cv2 as cv, numpy as np, mediapipe as mp

from hand_steer_sim.model.steering_mode import (
    KeyPointClassifier,        # 4-point "wheel" key-point net
    PointHistoryClassifier     # 16-frame LSTM / MLP net
)

mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

_MCP_IDXS = (5, 9, 13, 17)     # index–pinky MCP joints

class SteeringRecognition:
    def __init__(
        self,
        key_lbl_csv:  str | Path,
        key_tflite:   str | Path,
        hist_lbl_csv: str | Path,
        hist_tflite:  str | Path,
        *,
        use_gpu: bool = False,
        min_det_conf: float = .7,
        min_track_conf: float = .7,
        history_length: int = 8,  # Added history length parameter
    ):
        # label lists
        self._key_lbl  = self._load_labels(key_lbl_csv)
        self._hist_lbl = self._load_labels(hist_lbl_csv)

        # TFLite classifiers
        self._kpc = KeyPointClassifier(key_tflite,  use_gpu=use_gpu)
        self._phc = PointHistoryClassifier(hist_tflite, use_gpu=use_gpu)

        # MediaPipe
        self._mp = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

        # Initialize history buffer with configurable length
        self._history_length = history_length
        self._buf = np.zeros((history_length, 4, 2), np.float32)
        self._idx = 0

    # ────────────────────────── public API ──────────────────────────
    def recognise(self, bgr: np.ndarray) -> Tuple[np.ndarray, str, str]:
        bgr = cv.flip(bgr, 1)
        dbg = bgr.copy()

        results = self._mp.process(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        static_lbl = "NONE"
        dynamic_lbl = "Forward"      # default

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            pts = self._pixel(lms, dbg)

            # Create keypoint vector from all hand landmarks
            kp_vec = self._vec_static(pts)
            
            # static key-point classifier → gate history sampling
            sign_id = self._kpc(kp_vec)
            static_lbl = self._key_lbl[sign_id]

            # fill history only when "holding wheel" (ID 2 in static net)
            if sign_id == 2:
                self._push_history([pts[i] for i in _MCP_IDXS])
            else:
                self._push_history([[0, 0]] * 4)

            dynamic_lbl = self._hist_lbl[self._hist_predict(dbg.shape)]

            self._draw(dbg, lms, f"{static_lbl} | {dynamic_lbl}")

        return dbg, static_lbl, dynamic_lbl

    # ───────────────────────── internal helpers ─────────────────────
    @staticmethod
    def _load_labels(csv_path) -> list[str]:
        with open(csv_path, encoding="utf-8-sig") as f:
            return [r[0] for r in csv.reader(f)]

    def _push_history(self, pts4):
        self._buf[self._idx] = pts4
        self._idx = (self._idx + 1) % self._history_length    # circular

    def _hist_predict(self, img_hw):
        if not (self._buf != 0).any():        # buffer empty
            return 0                     # "None" default
        h, w = img_hw[:2]
        flat = self._buf.copy()
        base  = flat[0, 0]
        flat -= base
        flat[..., 0] /= w; flat[..., 1] /= h
        vec = flat.ravel()
        # Ensure vector length matches model input (64 for 16 frames)
        if len(vec) > 64:
            vec = vec[:64]  # Truncate if longer
        elif len(vec) < 64:
            vec = np.pad(vec, (0, 64 - len(vec)))  # Pad if shorter
        
        # Get prediction and ensure it's within range
        pred = self._phc(vec)
        return min(pred, len(self._hist_lbl) - 1)  # Ensure index is within range

    @staticmethod
    def _pixel(hand, img):
        h, w = img.shape[:2]
        return np.array([[lm.x*w, lm.y*h] for lm in hand.landmark], np.float32)

    @staticmethod
    def _vec_static(pts):
        """Convert 21 hand landmarks to normalized feature vector."""
        # Normalize relative to wrist (first point)
        pts = pts - pts[0]
        # Flatten and normalize by max absolute value
        flat = pts.ravel()
        flat = flat / (np.abs(flat).max() or 1.0)
        return flat.tolist()

    def _draw(self, img, lms, label):
        mp_draw.draw_landmarks(
            img, lms, mp.solutions.hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style())
        h, w = img.shape[:2]
        wrist = lms.landmark[0]
        cv.putText(img, label, (int(wrist.x*w), int(wrist.y*h)-10),
                   cv.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv.LINE_AA)
