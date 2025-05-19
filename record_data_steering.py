#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_data.py – data‑collection script (model‑free version)
"""

# ───────────────────────────── imports ──────────────────────────────
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from scripts.cvfpscalc import CvFpsCalc

# ───────────────────────────── constants ────────────────────────────
INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP = (5, 9, 13, 17)
MCP_IDXS = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
FONT       = cv.FONT_HERSHEY_SIMPLEX

# ───────────────────────────── CLI ──────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  type=int, default=0)
    ap.add_argument("--width",   type=int, default=960)
    ap.add_argument("--height",  type=int, default=540)
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
    elif key == ord("k"): mode = 1     # still logs key‑points
    elif key == ord("h"): mode = 2     # logs 4‑MCP trajectories
    return number, mode


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]


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
    if mode == 1 and 0 <= number <= 9:            # key‑points
        path = Path("hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([number, *kp_vec])

    if mode == 2 and 0 <= number <= 9:            # history
        path = Path("hand_steer_sim/model/steering_mode/point_history_classifier/point_history.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
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

    # ── MediaPipe Hands ──────────────────────────────────────────────
    hands = mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # ── buffers ──────────────────────────────────────────────────────
    HISTORY_LEN  = 16
    point_history = deque(maxlen=HISTORY_LEN)

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
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                lm_list = calc_landmark_list(debug, lm)
                kp_vec  = pre_process_landmark(lm_list)

                # always grab the 4 MCPs
                pts4 = [lm_list[i] for i in MCP_IDXS]
                point_history.append(pts4)

                # build history vector when buffer full
                hist_vec = []
                if len(point_history) == HISTORY_LEN:
                    hist_vec = pre_process_point_history(debug, point_history)

                # CSV logging
                logging_csv(number, mode, kp_vec, hist_vec)

                # draw landmarks
                mp_drawing.draw_landmarks(
                    debug, lm, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
        else:
            point_history.append([[0, 0]] * 4)

        debug = draw_point_history(debug, point_history)
        debug = draw_info(debug, fps, mode, number)
        cv.imshow("Yunus Emre Danabas - Data Collecting", debug)

    pipe.stop() if args.realsense else cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
