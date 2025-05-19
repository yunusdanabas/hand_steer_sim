#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_data_steering.py – collect 4‑MCP trajectories + static signs
"""

from __future__ import annotations
from pathlib import Path, PurePosixPath
from typing import List
import argparse, csv, cv2 as cv, numpy as np, mediapipe as mp, pyrealsense2 as rs
from collections import deque, Counter
from scripts.cvfpscalc import CvFpsCalc
from hand_steer_sim.model.steering_mode.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from hand_steer_sim.model.steering_mode.point_history_classifier.point_history_classifier import PointHistoryClassifier

# ─────────────────────────────  constants  ──────────────────────────────────────────
MCP_IDX  = np.asarray([5, 9, 13, 17])        # index/middle/ring/pinky MCPs
HIST_LEN = 16                                # frames kept in deque

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_style  = mp.solutions.drawing_styles
font      = cv.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────  CLI  ────────────────────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint_model",
        default="hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier.tflite")
    ap.add_argument("--history_model",
        default="hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier.tflite")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width",  type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--gpu",       action="store_true", help="try GPU delegate for TFLite")
    ap.add_argument("--use_static_image_mode", action="store_true")
    ap.add_argument("--min_detection_confidence", type=float, default=0.7)
    ap.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return ap.parse_args()

# ─────────────────────────────  helpers  ────────────────────────────────────────────
def load_csv(path) -> List[str]:
    with open(path, encoding="utf‑8‑sig") as f:
        return [row[0] for row in csv.reader(f)]

def pixel_landmarks(landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList,
                    w: int, h: int) -> np.ndarray:
    """
    Convert MediaPipe NormalizedLandmarkList to an (N×2) float32 array of pixel coords.
    """
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark],
                   dtype=np.float32)
    return pts

def normalise(pts):
    pts -= pts[0]
    flat = pts.ravel()
    flat /= (np.abs(flat).max() or 1.0)
    return flat.tolist()

def vec_history(hist, w, h):
    if not hist: return []
    base = np.asarray(hist[0][0], dtype=np.float32)
    buf  = np.asarray(hist, dtype=np.float32)        # shape = (F,4,2)
    buf -= base
    buf[...,0] /= w
    buf[...,1] /= h
    return buf.ravel().tolist()

def put_fps(img, fps):
    txt = f"FPS:{fps:.2f}"
    (tw,th), _ = cv.getTextSize(txt, font, 1.0, 2)
    x = img.shape[1]-tw-10
    cv.putText(img, txt,(x,th+10), font,1,(0,0,0),4,cv.LINE_AA)
    cv.putText(img, txt,(x,th+10), font,1,(255,255,255),2,cv.LINE_AA)

# ─────────────────────────────  main  ───────────────────────────────────────────────
def main():
    args  = get_args()
    fps_c = CvFpsCalc(buffer_len=10)

    # camera -------------------------------------------------------------------------
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

    # models -------------------------------------------------------------------------
    kp_cls = KeyPointClassifier(args.keypoint_model,  delegate_gpu=args.gpu)
    ph_cls = PointHistoryClassifier(args.history_model, delegate_gpu=args.gpu)

    # derive label CSVs from the model filenames
    model_path = Path(args.keypoint_model)
    kp_label_path = model_path.parent / f"{model_path.stem}_label.csv"
    kp_labels = load_csv(kp_label_path)

    hist_path = Path(args.history_model)
    ph_label_path = hist_path.parent / f"{hist_path.stem}_label.csv"
    ph_labels = load_csv(ph_label_path)

    hands = mp_hands.Hands(static_image_mode=args.use_static_image_mode,
                           max_num_hands=1,
                           min_detection_confidence=args.min_detection_confidence,
                           min_tracking_confidence=args.min_tracking_confidence)

    # buffers ------------------------------------------------------------------------
    history   = deque(maxlen=HIST_LEN)   # each entry = 4×[x,y]
    fg_hist   = deque(maxlen=HIST_LEN)
    mode      = 0                        # 0=neutral 1=log key 2=log hist
    class_id  = -1

    # ───────────────────────────── loop ─────────────────────────────────────────────
    while True:
        fps  = fps_c.get()
        key  = cv.waitKey(1) & 0xff
        if key == 27: break
        if 48 <= key <= 57: class_id = key-48
        if key == ord('k'): mode = 1
        if key == ord('h'): mode = 2
        if key == ord('n'): mode = 0

        frame = grab()
        if frame is None: continue
        frame = cv.flip(frame,1)
        dbg   = frame.copy()
        h,w   = dbg.shape[:2]

        res = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                pts_px    = pixel_landmarks(lm, w, h)
                feat      = normalise(pts_px)
                sign_id   = kp_cls(feat)

                # 4‑point trajectory --------------------------------------------------
                if sign_id == 2:                                   # “on wheel”
                    bundle = pts_px[MCP_IDX].tolist()
                else:
                    bundle = [[0.0,0.0]]*4
                history.append(bundle)

                # hist feature & dyn‑cls when full -----------------------------------
                dyn_id = 0
                if len(history)==HIST_LEN:
                    dyn_feat = vec_history(history, w, h)
                    dyn_id   = ph_cls(dyn_feat)
                fg_hist.append(dyn_id)
                common_dyn = Counter(fg_hist).most_common(1)[0][0]

                # CSV logging --------------------------------------------------------
                if mode==1 and 0<=class_id<=9:
                    with open("hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint.csv",
                              "a", newline="") as f:
                        csv.writer(f).writerow([class_id,*feat])
                if mode==2 and 0<=class_id<=9 and len(history)==HIST_LEN:
                    with open("hand_steer_sim/model/steering_mode/point_history_classifier/point_history.csv",
                              "a", newline="") as f:
                        csv.writer(f).writerow([class_id,*dyn_feat])

                # draw ----------------------------------------------------------------
                mp_draw.draw_landmarks(
                    dbg, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style())
                wr = lm.landmark[0]
                cv.putText(dbg,
                    f"{hd.classification[0].label}:{kp_labels[sign_id]}:{ph_labels[common_dyn]}",
                    (int(wr.x*w), int(wr.y*h)-10), font, 0.6, (255,255,255),1,cv.LINE_AA)

        else:
            history.append([[0.0,0.0]]*4)

        # show -----------------------------------------------------------------------
        put_fps(dbg, fps)
        cv.imshow("Yunus Emre Danabas - Data Collecting", dbg)

    # cleanup ------------------------------------------------------------------------
    pipe.stop() if args.realsense else cap.release()
    cv.destroyAllWindows()

# ───────────────────────────── KeyPoint / PointHistory wrappers with GPU opt ────────
import tensorflow as tf
class _TFLiteBase:
    def __init__(self, model_path, delegate_gpu=False):
        delegates = []
        if delegate_gpu:
            try:
                delegates.append(tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so"))
            except OSError:  # GPU not available – silently fall back
                pass
        self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()
        self.in_idx  = self.interpreter.get_input_details()[0]["index"]
        self.out_idx = self.interpreter.get_output_details()[0]["index"]
    def __call__(self, x):
        self.interpreter.set_tensor(self.in_idx, np.array([x], np.float32))
        self.interpreter.invoke()
        return int(np.argmax(self.interpreter.get_tensor(self.out_idx)))

class KeyPointClassifier(_TFLiteBase):       pass
class PointHistoryClassifier(_TFLiteBase):   pass

# ─────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
