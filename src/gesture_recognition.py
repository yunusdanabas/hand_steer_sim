#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from typing import List, Tuple


# Built-in MediaPipe drawing helpers
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GestureRecognition:

    def __init__(
        self,
        label_path: str,
        model_path: str,
        *,
        static_image_mode: bool = False,
        min_det_conf: float = 0.7,
        min_track_conf: float = 0.7,
        ) -> None:

        """Create a hand-gesture recogniser.

        Args:
            label_path: CSV with one gesture label per row.
            model_path: TFLite model file for KeyPointClassifier.
            static_image_mode: If True, run detection on every frame.
            min_det_conf:   MediaPipe minimum detection confidence.
            min_track_conf: MediaPipe minimum tracking confidence.
        """
        self._label_path   = label_path
        self._model_path   = model_path
        self._static       = static_image_mode
        self._det_conf     = min_det_conf
        self._track_conf   = min_track_conf

        # --- load MediaPipe + classifier once ---------------------------------
        self._mp_hands, self._classifier, self._labels = self._load_model()


    def _load_model(self):
        """Initialise MediaPipe Hands and KeyPointClassifier."""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=self._static,
            max_num_hands=1,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._track_conf,
        )

        classifier = KeyPointClassifier(self._model_path)

        with open(self._label_path, encoding="utf-8-sig") as f:
            labels = [row[0] for row in csv.reader(f)]

        return hands, classifier, labels


    def recognize(self, bgr_img: np.ndarray) -> Tuple[np.ndarray, str]:
        """Return (debug_image, gesture_label)."""
        # Mirror for UI convenience
        bgr_img = cv.flip(bgr_img, 1)
        debug   = bgr_img.copy()

        # MediaPipe expects RGB
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        results = self._mp_hands.process(rgb_img)

        gesture = "NONE"
        if results.multi_hand_landmarks:
            h,  w  = debug.shape[:2]

            for hand_lms, handed in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
                # 1. Classification
                lm_pixel   = self._calc_landmark_list(debug, hand_lms)
                vec        = self._pre_process_landmark(lm_pixel)
                gesture_id = self._classifier(vec)
                gesture    = self._labels[gesture_id]

                # 2. Draw landmarks & skeleton
                mp_drawing.draw_landmarks(
                    debug,
                    hand_lms,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # 3. Label (left/right + gesture) just above the wrist
                txt = f"{handed.classification[0].label}:{gesture}"
                wrist = hand_lms.landmark[0]
                cv.putText(
                    debug,
                    txt,
                    (int(wrist.x * w), int(wrist.y * h) - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

        return debug, gesture

    @staticmethod
    def _calc_landmark_list(img: np.ndarray, hand_lms) -> List[List[int]]:
        """Pixel (x,y) list for 21 landmarks."""
        h, w = img.shape[:2]
        return [[int(lm.x * w), int(lm.y * h)] for lm in hand_lms.landmark]


    @staticmethod
    def _pre_process_landmark(landmarks: List[List[int]]) -> List[float]:

        """Convert to a flat, normalised vector (length 42)."""
        pts = np.asarray(landmarks, dtype=np.float32)
        pts -= pts[0]                      # make relative to wrist
        flat = pts.flatten()
        max_val = np.abs(flat).max() or 1  # avoid divide-by-zero
        return (flat / max_val).tolist()


    @staticmethod
    def _pre_process_point_history(
        img: np.ndarray, history: List[List[int]]
    ) -> List[float]:
        """Normalise a sequence of tip points to image size."""
        h, w = img.shape[:2]
        hist = np.asarray(history, dtype=np.float32)

        if len(hist) == 0:
            return []

        hist -= hist[0]                    # relative to first point
        hist[:, 0] /= w                    # x normalised
        hist[:, 1] /= h                    # y normalised
        return hist.flatten().tolist()

    @staticmethod
    def draw_fps_info(image: np.ndarray, fps: float) -> np.ndarray:
        cv.putText(image, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image


