#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steering_recognition.py – 16-frame dynamic-gesture classifier
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass
from collections import deque, Counter
import csv, cv2 as cv, numpy as np, mediapipe as mp

from hand_steer_sim.model.steering_mode import (
    KeyPointClassifier,        # 4-point "wheel" key-point net
    PointHistoryClassifier     # 16-frame LSTM / MLP net
)

mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

_MCP_IDXS = (5, 9, 13, 17)     # index–pinky MCP joints

# ───────────────────────────── results struct ──────────────────────────────
@dataclass
class SteeringResult:
    dbg_img:       np.ndarray
    static_label:  str
    dynamic_label: str

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
        history_length: int = 16,  # Added history length parameter
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

        # ---------------------------------------------
        # Majority-vote smoothing over last N dyn preds
        self._vote_len   = 16
        self._pred_hist  = deque(maxlen=self._vote_len)

        # Initialize history buffer using deque
        self._history_length = int(history_length)
        self._hist = deque(maxlen=self._history_length)

        exp_vec = 2 * 4 * self._history_length        # (x,y)*4pts*history
        rospy_msg = f"[SteeringRecognition] ready − dyn-vec {exp_vec}D  " \
                    f"(history={self._history_length})"
        try:    import rospy; rospy.loginfo(rospy_msg)
        except ImportError:  print(rospy_msg)

    # ────────────────────────── public API ──────────────────────────
    def recognise(self, bgr: np.ndarray) -> SteeringResult:
        bgr = cv.flip(bgr, 1)
        dbg = bgr.copy()

        results = self._mp.process(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        static_lbl = "NONE"
        dynamic_lbl = "Forward"      # default

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            pts = self._pixel(lms, dbg)

            # -------- static classifier --------
            sign_id   = self._kpc(self._vec_static(pts))
            static_lbl = self._key_lbl[sign_id]

            # -------- history buffer update ----
            mcp_pts = np.array([pts[i] for i in _MCP_IDXS], np.float32)  # shape (4,2)
            self._push_history(mcp_pts)

            # -------- dynamic classifier -------
            dyn_id = self._hist_predict(dbg.shape)
            self._pred_hist.append(dyn_id)
            stable_dyn = Counter(self._pred_hist).most_common(1)[0][0]
            dynamic_lbl = self._hist_lbl[stable_dyn]

            # -------- draw overlays ------------
            self._draw(dbg, lms, f"{static_lbl} | {dynamic_lbl}")
            dbg = self._draw_point_history(dbg)                # tiny green trail
        else:
            # Push zeros with correct shape when no hand detected
            self._push_history(np.zeros((4, 2), np.float32))

        return SteeringResult(dbg, static_lbl, dynamic_lbl)

    # ───────────────────────── internal helpers ─────────────────────
    @staticmethod
    def _load_labels(csv_path) -> list[str]:
        with open(csv_path, encoding="utf-8-sig") as f:
            labels = [r[0] for r in csv.reader(f)]
        if not labels:
            raise ValueError(f"{csv_path} contained no labels")
        return labels

    def _push_history(self, pts4):
        """Push 4 MCP points to history buffer.
        Args:
            pts4: numpy array of shape (4,2) containing MCP point coordinates
        """
        self._hist.append(pts4)

    def _hist_predict(self, img_hw):
        if len(self._hist) < self._history_length:    # buffer not full
            return 0                     # "None" default
        h, w = img_hw[:2]
        
        # Convert deque to array - each element is (4,2) shape
        buf = np.stack(self._hist)        # shape (L,4,2)
        buf -= buf[0,0]                  # translate to first-frame MCP
        buf[...,0] /= w                  # normalise
        buf[...,1] /= h

        # stack all X then all Y (shape → (hist*4,))
        vec = buf.transpose(2,0,1).reshape(-1).astype(np.float32)
        
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

    def _draw_point_history(self, img):
        """Green breadcrumb trail for steering MCPs."""
        for i, frame in enumerate(self._hist):
            for x, y in frame:
                if x or y:
                    shade = int(200 - 150 * i / self._history_length)
                    cv.circle(img, (int(x), int(y)), 2,
                              (shade, 255, shade), 3)
        return img

    @property
    def last_static_id(self)  -> int: return self._kpc.last_pred  # type: ignore
    @property
    def last_dynamic_id(self) -> int: return self._pred_hist[-1] if self._pred_hist else 0

    def get_labels(self) -> Tuple[List[str], List[str]]:
        """Returns copies of [static_labels], [dynamic_labels]."""
        return list(self._key_lbl), list(self._hist_lbl)
