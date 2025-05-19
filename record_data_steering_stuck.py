#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_data.py – data‑collection script
--------------------------------------

Usage examples
--------------
# Basic (web‑cam)
python record_data.py

# Custom device and HD resolution
python record_data.py --device 2 --width 1280 --height 720

# Intel RealSense colour stream
python record_data.py --realsense

# Static‑image mode (higher accuracy)
python record_data.py --use_static_image_mode \
                      --min_detection_confidence 0.8


Options
-------
--device N                    OpenCV device index (default 0)
--width  W                    Capture width   (default 960)
--height H                    Capture height  (default 540)
--realsense                   Use Intel RealSense D4xx (colour only)
--use_static_image_mode       Run MediaPipe in single‑shot mode
--min_detection_confidence X  Detection threshold  (default 0.7)
--min_tracking_confidence  X  Tracking  threshold  (default 0.5)


Hot‑keys (window)
-----------------
n      neutral / pause logging
k      log key‑points   → keypoint.csv
h      log point‑history→ point_history.csv
0‑9    class label for logging
Esc    quit
"""

# ───────────────────────────── imports ──────────────────────────────
import argparse
import csv
from collections import Counter, deque
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from scripts.cvfpscalc import CvFpsCalc
from hand_steer_sim.model.steering_mode.keypoint_classifier.keypoint_classifier import (
    KeyPointClassifier,
)
from hand_steer_sim.model.steering_mode.point_history_classifier.point_history_classifier import (
    PointHistoryClassifier,
)

# ───────────────────────────── constants ────────────────────────────
INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP = (5, 9, 13, 17)
MCP_IDXS = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
FONT       = cv.FONT_HERSHEY_SIMPLEX

# ───────────────────────────── CLI ──────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keypoint_model",
        default="hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier.tflite",
        help="path to keypoint_classifier .tflite",
    )
    ap.add_argument(
        "--history_model",
        default="hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier.tflite",
        help="path to point_history_classifier .tflite",
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=960,  help="capture width")
    ap.add_argument("--height", type=int, default=540, help="capture height")
    ap.add_argument("--realsense", action="store_true",
                    help="use Intel RealSense colour stream")
    ap.add_argument("--use_static_image_mode", action="store_true")
    ap.add_argument("--min_detection_confidence", type=float, default=0.7)
    ap.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return ap.parse_args()

# ───────────────────────────── helpers ──────────────────────────────
def select_mode(key: int, mode: int):
    number = key - 48 if 48 <= key <= 57 else -1
    if   key == ord("n"): mode = 0
    elif key == ord("k"): mode = 1
    elif key == ord("h"): mode = 2
    return number, mode


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    pts = np.asarray(landmark_list, np.float32)
    pts -= pts[0]
    flat = pts.ravel()
    flat /= (np.abs(flat).max() or 1.0)
    return flat.tolist()


def pre_process_point_history(image, point_history):
    h, w = image.shape[:2]
    flat = []
    base_x, base_y = point_history[0][0]
    for frame in point_history:
        for x, y in frame:
            flat.extend([(x - base_x) / w, (y - base_y) / h])
    return flat


def logging_csv(number, mode, kp_vec, hist_vec):
    if mode == 1 and 0 <= number <= 9:
        path = Path("hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint.csv")
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([number, *kp_vec])

    if mode == 2 and 0 <= number <= 9:
        path = Path(
            "hand_steer_sim/model/steering_mode/point_history_classifier/point_history.csv"
        )
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([number, *hist_vec])


def draw_point_history(img, history):
    for i, frame in enumerate(history):
        for x, y in frame:
            if x or y:
                cv.circle(img, (x, y), 1 + i // 2, (152, 251, 152), 2)
    return img


def draw_info(img, fps, mode, num):
    h, w = img.shape[:2]
    txt = f"FPS:{fps:.2f}"
    (tw, th), _ = cv.getTextSize(txt, FONT, 1, 2)
    cv.putText(img, txt, (w - tw - 10, th + 10), FONT, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, txt, (w - tw - 10, th + 10), FONT, 1, (255, 255, 255), 2, cv.LINE_AA)

    modes = ["Logging Key Point", "Logging Point History"]
    if 1 <= mode <= 2:
        cv.putText(img, f"MODE:{modes[mode - 1]}", (10, h - 50),
                   FONT, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= num <= 9:
            cv.putText(img, f"NUM:{num}", (10, h - 30),
                       FONT, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return img

# ───────────────────────────── main ─────────────────────────────────
def main():
    args = get_args()
    fps_calc = CvFpsCalc(buffer_len=10)

    # ── camera ───────────────────────────────────────────────────────
    if args.realsense:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
        pipe.start(cfg)
        grab = lambda: np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
    else:
        cap = cv.VideoCapture(args.device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        grab = lambda: cap.read()[1]

    # ── models ───────────────────────────────────────────────────────
    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    kp_classifier = KeyPointClassifier(model_path=args.keypoint_model)
    ph_classifier = PointHistoryClassifier(model_path=args.history_model)

    with open(
        "hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        kp_labels = [row[0] for row in csv.reader(f)]

    with open(
        "hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        ph_labels = [row[0] for row in csv.reader(f)]

    # ── buffers ──────────────────────────────────────────────────────
    history_len = 16
    point_history = deque(maxlen=history_len)
    gesture_hist = deque(maxlen=history_len)

    mode, number = 0, -1

    # ── main loop ────────────────────────────────────────────────────
    while True:
        fps = fps_calc.get()
        key = cv.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        frame = grab()
        if frame is None:
            continue
        frame = cv.flip(frame, 1)
        debug = frame.copy()

        cv.putText(debug, "Hand Tracking Simulation - Data Recording", (10, 30),
                   FONT, 0.8, (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(debug,
                   "[k] log keypoints | [h] log history | [n] neutral | [0-9] class | Esc quit",
                   (10, 60), FONT, 0.5, (0, 0, 0), 2, cv.LINE_AA)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)
        rgb.flags.writeable = True

        if res.multi_hand_landmarks:
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                lm_list = calc_landmark_list(debug, lm)
                kp_vec  = pre_process_landmark(lm_list)
                hist_vec = pre_process_point_history(debug, point_history)
                logging_csv(number, mode, kp_vec, hist_vec)

                sign_id = kp_classifier(kp_vec)

                if sign_id == 2:
                    pts4 = [lm_list[i] for i in MCP_IDXS]
                else:
                    pts4 = [[0, 0]] * 4
                point_history.append(pts4)

                dyn_id = 0
                if len(hist_vec) == history_len * 2 * 4:
                    dyn_id = ph_classifier(hist_vec)

                gesture_hist.append(dyn_id)
                common_dyn = Counter(gesture_hist).most_common(1)[0][0]

                mp_drawing.draw_landmarks(
                    debug, lm, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                wrist = lm.landmark[0]
                h, w = debug.shape[:2]
                origin = (int(wrist.x * w), int(wrist.y * h) - 10)
                label = f"{handed.classification[0].label}:{kp_labels[sign_id]}:{ph_labels[common_dyn]}"
                cv.putText(debug, label, origin, FONT, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        else:
            point_history.append([[0, 0]] * 4)

        debug = draw_point_history(debug, point_history)
        debug = draw_info(debug, fps, mode, number)
        cv.imshow("Yunus Emre Danabas - Data Collecting", debug)

    pipe.stop() if args.realsense else cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
