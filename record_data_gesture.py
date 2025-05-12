#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
record_data.py - Data-collection script

Usage examples:
  # Basic (web-cam) run
  python record_data.py

  # Custom device and HD resolution
  python record_data.py --device 2 --width 1280 --height 720

  # Intel RealSense color stream
  python record_data.py --realsense

  # Static-image mode (higher accuracy)
  python record_data.py --use_static_image_mode \
                        --min_detection_confidence 0.8

Options:
  --device N                   OpenCV device index (default: 0)
  --width W                    Capture width  (default: 960px)
  --height H                   Capture height (default: 540px)
  --realsense                  Use Intel RealSense D4xx (color only)
  --use_static_image_mode      Run MediaPipe in single-shot mode
  --min_detection_confidence X Detection threshold  (default: 0.7)
  --min_tracking_confidence  X Tracking threshold  (default: 0.5)

Hotkeys (in window):
  n        neutral / pause logging
  k        log key-points → keypoint.csv
  h        log point-history → point_history.csv
  0-9      class label for logging
  Esc      quit
'''

import argparse, csv, cv2 as cv, numpy as np, mediapipe as mp 
import pyrealsense2 as rs
from scripts.cvfpscalc import CvFpsCalc
from hand_steer_sim.model.static_mode.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint_model",
                    default="hand_steer_sim/model/static_mode/keypoint_classifier/keypoint_classifier.tflite")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width",  type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--realsense", action="store_true",
                    help="use Intel RealSense colour stream")
    ap.add_argument("--use_static_image_mode", action="store_true")
    ap.add_argument("--min_detection_confidence", type=float, default=0.7)
    ap.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return ap.parse_args()

# ------------------------------------------------------------------------------------ #
def main():
    args   = get_args()
    mp_h   = mp.solutions.hands
    hands  = mp_h.Hands(static_image_mode=args.use_static_image_mode,
                        max_num_hands=1,
                        min_detection_confidence=args.min_detection_confidence,
                        min_tracking_confidence=args.min_tracking_confidence)
    kp_cls = KeyPointClassifier(args.keypoint_model)

    with open("hand_steer_sim/model/static_mode/keypoint_classifier/keypoint_classifier_label.csv",
              encoding="utf‑8‑sig") as f:
        labels = [row[0] for row in csv.reader(f)]

    # ---------- camera ----------------------------------------------------------------
    if args.realsense:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
        pipe.start(cfg)
        grab = lambda: np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
    else:
        cap  = cv.VideoCapture(args.device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        grab = lambda: cap.read()[1]

    fps_calc, mode, class_id = CvFpsCalc(buffer_len=10), 0, -1

    while True:
        fps  = fps_calc.get()
        key  = cv.waitKey(10)
        if key == 27:                                        # Esc
            break
        if key in range(48, 58): class_id = key - 48         # '0'‑'9'
        if key == ord('k'):  mode = 1                        # log key‑points
        if key == ord('n'):  mode = 0                        # neutral

        frame = grab()
        if frame is None: continue
        frame = cv.flip(frame, 1)
        debug = frame.copy()

        # ───────── MediaPipe inference ────────────────────────────────────────────────
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)
        rgb.flags.writeable = True

        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                lm_px  = _pixel_landmarks(debug, lm)
                feat   = _normalise(lm_px)
                pred   = kp_cls(feat)
                label  = labels[pred]

                # draw
                mp.solutions.drawing_utils.draw_landmarks(
                    debug, lm, mp_h.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                h, w = debug.shape[:2]
                wr   = lm.landmark[0]
                cv.putText(debug, f"{hd.classification[0].label}:{label}",
                           (int(wr.x*w), int(wr.y*h)-10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)

                # ----- logging --------------------------------------------------------
                if mode == 1 and 0 <= class_id <= 9:
                    with open("hand_steer_sim/model/static_mode/keypoint_classifier/keypoint.csv",
                              "a", newline="") as f:
                        csv.writer(f).writerow([class_id, *feat])

        # UI overlays ------------------------------------------------------------------
        _put_fps(debug, fps)
        # top-bar instructions
        cv.putText(debug, "Hand Tracking Simulation - Data Recording", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv.LINE_AA)
        cv.putText(debug,
            "[k] -> log keypoints   [h] -> log history   [n] -> neutral   [0-9] -> class   [Esc] -> quit",
            (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv.LINE_AA)
        cv.imshow("Yunus Emre Danabas - Data Collecting", debug)

    # ---------- cleanup ---------------------------------------------------------------
    if args.realsense: pipe.stop()
    else: cap.release()
    cv.destroyAllWindows()

# ------------------------------------------------------------------------------------ #
def _pixel_landmarks(img, lm):
    h, w = img.shape[:2]
    return np.array([[pt.x*w, pt.y*h] for pt in lm.landmark], np.float32)

def _normalise(pts):
    pts -= pts[0]
    flat = pts.flatten()
    flat /= (np.abs(flat).max() or 1.0)
    return flat.tolist()

def _put_fps(img, fps):
    txt = f"FPS:{fps:.2f}"
    (tw,th), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x = img.shape[1] - tw - 10
    cv.putText(img, txt, (x, th+10),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(img, txt, (x, th+10),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

# ------------------------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
