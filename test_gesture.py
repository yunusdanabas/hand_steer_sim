#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gestures.py  –  real-time viewer for static + dynamic models
NO recording, just live inference.
"""

import argparse, csv, cv2 as cv, mediapipe as mp, numpy as np, pyrealsense2 as rs
import time
from collections import Counter, deque
from pathlib   import Path
from scripts.cvfpscalc import CvFpsCalc
from hand_steer_sim.model.steering_mode.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from hand_steer_sim.model.steering_mode.point_history_classifier.point_history_classifier import PointHistoryClassifier

# ────────────────────────── CLI ──────────────────────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint_model",
                    default="hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier.tflite")
    ap.add_argument("--history_model",
                    default="hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier.tflite")
    ap.add_argument("--device",  type=int, default=0)
    ap.add_argument("--width",   type=int, default=960)
    ap.add_argument("--height",  type=int, default=540)
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--min_detection_confidence", type=float, default=0.7)
    ap.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return ap.parse_args()

# ───────────────────── helpers (unchanged) ─────────────────────
INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP = (5, 9, 13, 17)
MCP_IDXS = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
FONT = cv.FONT_HERSHEY_SIMPLEX

def calc_landmark_list(img, lm):
    h, w = img.shape[:2]
    return [[int(l.x*w), int(l.y*h)] for l in lm.landmark]

def pre_process_landmark(pts):
    arr = np.asarray(pts, np.float32)
    arr -= arr[0]; flat = arr.ravel()
    flat /= (np.abs(flat).max() or 1.0)
    return flat.tolist()

def pre_process_history(img, history):
    h, w = img.shape[:2]; base_x, base_y = history[0][0]
    out = [(x-base_x)/w for fr in history for x,_ in fr] + \
          [(y-base_y)/h for fr in history for _,y in fr]
    return out

def draw_point_history(img, history):
    """Visualize landmark trajectory history."""
    for i, frame in enumerate(history):
        for x, y in frame:
            if x or y:
                cv.circle(img, (x, y), 2 + i // 2, (152, 251, 152), 4)
    return img

def put_fps(img,fps):
    txt=f"FPS:{fps:.2f}"; (tw,th),_=cv.getTextSize(txt,FONT,1,2)
    cv.putText(img,txt,(img.shape[1]-tw-10,th+10),FONT,1,(0,0,0),4,cv.LINE_AA)
    cv.putText(img,txt,(img.shape[1]-tw-10,th+10),FONT,1,(255,255,255),2,cv.LINE_AA)

def init_realsense():
    """Initialize RealSense camera with retries"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            
            # Try to stop any existing pipeline
            try:
                pipe.stop()
            except:
                pass
            
            # Configure the pipeline
            cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            
            # Start the pipeline with a timeout
            profile = pipe.start(cfg)
            
            # Wait for the first frame to ensure the camera is working
            frames = pipe.wait_for_frames(timeout_ms=5000)
            if frames:
                return pipe, lambda: np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if pipe:
                try:
                    pipe.stop()
                except:
                    pass
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception("Failed to initialize RealSense camera after multiple attempts")
    
    raise Exception("Failed to initialize RealSense camera")

# ───────────────────── main ─────────────────────
def main():
    args = get_args()
    fps = CvFpsCalc(10)
    pipe = None
    cap = None

    try:
        # camera
        if args.realsense:
            print("Initializing RealSense camera...")
            pipe, grab = init_realsense()
            print("RealSense camera initialized successfully")
        else:
            cap = cv.VideoCapture(args.device)
            cap.set(3, args.width)
            cap.set(4, args.height)
            grab = lambda: cap.read()[1]

        # models
        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence)

        kp_cls = KeyPointClassifier(args.keypoint_model)
        ph_cls = PointHistoryClassifier(args.history_model)

        kp_labels = [r[0] for r in csv.reader(open(
            "hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"))]
        ph_labels = [r[0] for r in csv.reader(open(
            "hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier_label.csv", encoding="utf-8-sig"))]

        hist_len = 16
        point_history = deque(maxlen=hist_len)
        gesture_hist = deque(maxlen=hist_len)

        while True:
            frame = grab()
            if frame is None: continue
            frame = cv.flip(frame,1)
            rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = calc_landmark_list(frame,lm)
                kp_vec = pre_process_landmark(pts)
                sign_id = kp_cls(kp_vec)

                # update history buffer for dynamic model
                point_history.append([pts[i] for i in MCP_IDXS])
                dyn_id = 0
                dyn_conf = 0.0
                if len(point_history) == hist_len:
                    hist_vec = pre_process_history(frame, point_history)
                    dyn_id, dyn_conf = ph_cls(hist_vec, return_confidence=True)

                gesture_hist.append(dyn_id)
                stable_dyn = Counter(gesture_hist).most_common(1)[0][0]

                # draw landmarks & text
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,lm,mp.solutions.hands.HAND_CONNECTIONS)
                label = f"{kp_labels[sign_id]} | {ph_labels[stable_dyn]} ({dyn_conf:.2f})"
                wrist = lm.landmark[0]; h,w = frame.shape[:2]
                cv.putText(frame,label,(int(wrist.x*w),int(wrist.y*h)-10),
                           FONT,0.6,(255,255,255),2,cv.LINE_AA)
            else:
                point_history.append([[0,0]]*4)

            frame = draw_point_history(frame, point_history)
            put_fps(frame,fps.get())
            cv.imshow("Gesture Test Viewer", frame)
            if cv.waitKey(1)&0xFF==27: break

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        # Cleanup
        if pipe is not None:
            try:
                pipe.stop()
            except:
                pass
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()

if __name__=="__main__":
    main()
