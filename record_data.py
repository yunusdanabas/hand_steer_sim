#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Steer Data Recorder
------------------------
Collects static (single-frame key-points) or history (16-frame MCP trajectories) 
gesture data and saves them to timestamped CSV files that are immediately 
compatible with the training notebooks.

Key Controls (after window opens):
    K  - Choose static mode
    H  - Choose history mode
    0-9 - Select gesture label
    Enter - Take snapshot
    Esc  - Quit (runs validation)

Output Structure:
    data/static_2025-05-30_18-05-41/keypoint.csv      (42-cols)
    data/history_2025-05-30_18-05-41/point_history.csv (128-cols)
"""

# ──────────────────────────────── Imports ────────────────────────────────
import os
import argparse
import datetime
import csv
from pathlib import Path
from collections import deque
from types import SimpleNamespace
from typing import List

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from scripts.cvfpscalc import CvFpsCalc

# ──────────────────────────── Environment Setup ───────────────────────────
# Silence Qt debug spam before importing cv2's Qt plugins
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# ──────────────────────────── Constants ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[0]  # hand_steer_sim/
DATA_ROOT = REPO_ROOT / "data"

# Label file paths
STATIC_LABELS_CSV = REPO_ROOT / "hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier_label.csv"
HISTORY_LABELS_CSV = REPO_ROOT / "hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier_label.csv"

# Configuration constants
EXPECTED_LEN = {"static": 42, "history": 64}
HISTORY_LEN = 8
FONT = cv.FONT_HERSHEY_SIMPLEX

# ──────────────────────────── CLI Parser ────────────────────────────────
def parse_cli() -> argparse.Namespace:
    """Parse command line arguments for the data recorder."""
    ap = argparse.ArgumentParser(
        description="Record gesture datasets for static or dynamic steering models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Recording mode & labels
    ap.add_argument("--mode", choices=["static", "history"],
                   help="Pre-select mode; else choose with K/H after startup")
    ap.add_argument("--classes-file", type=Path,
                   help="Label CSV (one label per line). If omitted, default file is used after mode is chosen.")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT,
                   help="Parent directory to store session folders")

    # Camera / MediaPipe settings
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--realsense", action="store_true",
                   help="Use Intel RealSense colour stream")
    ap.add_argument("--use-static-image-mode", action="store_true",
                   help="MediaPipe single-shot mode (higher latency)")
    ap.add_argument("--min-detection-confidence", type=float, default=0.7)
    ap.add_argument("--min-tracking-confidence", type=float, default=0.5)

    args = ap.parse_args()

    # Set default label file based on mode
    if args.classes_file is None and args.mode is not None:
        args.classes_file = STATIC_LABELS_CSV if args.mode == "static" else HISTORY_LABELS_CSV

    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args

# ──────────────────────────── Camera Setup ──────────────────────────────
def open_camera(args) -> SimpleNamespace:
    """Initialize and configure camera (RealSense or standard webcam)."""
    if args.realsense:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            
            # Try to find and configure RealSense device
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found")
                
            # Configure color stream with more flexible settings
            cfg.enable_stream(rs.stream.color, 
                            args.width, args.height, 
                            rs.format.bgr8, 30)
            
            # Try to start pipeline with retries
            for _ in range(3):
                try:
                    pipe.start(cfg)
                    break
                except RuntimeError as e:
                    if "Couldn't resolve requests" in str(e):
                        # Try alternative configuration
                        cfg = rs.config()
                        cfg.enable_stream(rs.stream.color, 
                                        640, 480,  # Try default resolution
                                        rs.format.bgr8, 30)
                        continue
                    raise
            
            def grab_frame():
                try:
                    frames = pipe.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        return None
                    return np.asanyarray(color_frame.get_data())
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        print("[WARN] Frame timeout - retrying...")
                        return None
                    raise
            
            _grab = grab_frame
            _close = pipe.stop
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize RealSense camera: {e}")
            print("[INFO] Falling back to standard webcam...")
            args.realsense = False
            return open_camera(args)
    else:
        cap = cv.VideoCapture(args.device, cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {args.device}")
        _grab = lambda: cap.read()[1]
        _close = cap.release

    return SimpleNamespace(grab=_grab, close=_close, fps=CvFpsCalc(buffer_len=10))

def init_mediapipe(args):
    """Initialize MediaPipe hands detector with specified parameters."""
    return mp.solutions.hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

def setup_window() -> str:
    """Create and configure the display window."""
    win = "Hand-Steer Recorder"
    cv.namedWindow(win, cv.WINDOW_FULLSCREEN)
    return win

# ──────────────────────────── Data Management ───────────────────────────
def load_labels(csv_path: Path) -> List[str]:
    """Load and validate gesture labels from CSV file."""
    labels = csv_path.read_text(encoding="utf-8").splitlines()
    
    # Clean and validate labels
    labels = [label.strip() for label in labels]
    labels = [label.encode('ascii', 'ignore').decode('ascii') for label in labels]
    labels = [label for label in labels if label]
    
    if len(labels) > 10:
        print(f"[WARN] {csv_path.name} has {len(labels)} labels; truncating to 10.")
        labels = labels[:10]
    
    print(f"[DEBUG] Loaded labels: {labels}")
    return labels

def create_session(mode: str, base_dir: Path):
    """Create a new recording session with appropriate CSV file."""
    mode_dir = base_dir
    mode_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "keypoint.csv" if mode == "static" else "point_history.csv"
    csv_path = mode_dir / csv_name
    new_file = not csv_path.exists()

    fh = csv_path.open("a", newline="")
    writer = csv.writer(fh)

    if new_file:
        header = ["class"] + [f"f{i}" for i in range(EXPECTED_LEN[mode])]
        writer.writerow(["#"] + header)

    print(f"[INFO] Logging to {csv_path}")
    return csv_path, writer, fh

# ──────────────────────────── Feature Processing ────────────────────────
def calc_landmark_list(img, hand_lms):
    """Convert MediaPipe landmarks to pixel coordinates."""
    h, w = img.shape[:2]
    return [[int(lm.x*w), int(lm.y*h)] for lm in hand_lms.landmark]

def vec_static(lm_list):
    """Generate static feature vector from landmarks."""
    pts = np.asarray(lm_list, np.float32)
    pts -= pts[0]
    flat = pts.ravel()
    flat /= (np.abs(flat).max() or 1.0)
    return flat.tolist()  # 42-D

def vec_history(img, hist):
    """Generate history feature vector from landmark trajectories."""
    h, w = img.shape[:2]
    # Convert deque to numpy array and reshape to (8,4,2)
    hist_array = np.array(hist)
    # Get base coordinates from first frame
    base_x, base_y = hist_array[0,0]
    # Normalize coordinates relative to base point and image dimensions
    normalized = np.zeros_like(hist_array, dtype=np.float32)
    normalized[:,:,0] = (hist_array[:,:,0] - base_x) / w
    normalized[:,:,1] = (hist_array[:,:,1] - base_y) / h
    # Flatten to 128-D vector (8 frames * 4 points * 2 coordinates)
    return normalized.ravel().tolist()

# ──────────────────────────── Visualization ────────────────────────────
def draw_overlay(img, fps, mode, labels, label_idx, recording, counters):
    """Draw UI overlay with FPS, mode, labels, and counters."""
    h, w = img.shape[:2]
    
    # FPS display
    txt = f"FPS:{fps:.2f}"
    (tw, th), _ = cv.getTextSize(txt, FONT, 1.0, 2)
    # Draw black outline with thicker stroke
    cv.putText(img, txt, (w-tw-12, th+12), FONT, 1.1, (0,0,0), 6, cv.LINE_AA)
    # Draw white text on top
    cv.putText(img, txt, (w-tw-12, th+12), FONT, 1.1, (255,255,255), 3, cv.LINE_AA)
    
    # Mode and status banner
    banner = []
    if mode is None:
        banner.append("Press K for static poses  or  H for trajectories")
    else:
        banner.append(f"Mode: {mode.upper()}")
        if label_idx >= 0:
            banner.append(f"Selected: [{label_idx}] {labels[label_idx]}")
            banner.append("Press Enter to snapshot")
        else:
            banner.append("Choose label 0-9 before recording")
            
    for i, line in enumerate(banner):
        # Draw black outline with thicker stroke
        cv.putText(img, line, (10, 30+25*i), FONT, 1.0, (0,0,0), 4, cv.LINE_AA)
        # Draw white text on top
        cv.putText(img, line, (10, 30+25*i), FONT, 1.0, (255,255,255), 2, cv.LINE_AA)
    
    # Label list
    if mode and labels:
        for i, lbl in enumerate(labels):
            text = f"{i}: {lbl}"
            # Draw black outline with thicker stroke
            cv.putText(img, text, (10, 120+i*22), FONT, 0.8, (0,0,0), 4, cv.LINE_AA)
            # Draw white text on top
            cv.putText(img, text, (10, 120+i*22), FONT, 0.8, (255,255,255), 2, cv.LINE_AA)
    
    # Sample counters
    if counters:
        # Draw "Samples:" header
        cv.putText(img, "Samples:", (10, h-120), FONT, 0.8, (0,0,0), 4, cv.LINE_AA)
        cv.putText(img, "Samples:", (10, h-120), FONT, 0.8, (255,255,255), 2, cv.LINE_AA)
        
        for i, lbl in enumerate(labels):
            cnt = counters[lbl]
            text = f"{lbl:<12} {cnt:5d}"
            # Draw black outline with thicker stroke
            cv.putText(img, text, (10, h-100+i*20), FONT, 0.7, (0,0,0), 4, cv.LINE_AA)
            # Draw white text on top
            cv.putText(img, text, (10, h-100+i*20), FONT, 0.7, (255,255,255), 2, cv.LINE_AA)
    return img

def draw_point_history(img, history):
    """Visualize landmark trajectory history."""
    for i, frame in enumerate(history):
        for x, y in frame:
            if x or y:
                cv.circle(img, (x, y), 2 + i // 2, (152, 251, 152), 4)
    return img

# ──────────────────────────── Validation ───────────────────────────────
def validate_csv(csv_path: Path, expected_len: int, labels: List[str]) -> None:
    """Validate recorded data file for correctness and completeness."""
    try:
        data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32, comments='#')
    except ValueError as e:
        print(f"[ERROR] {csv_path.name} could not be parsed: {e}")
        return
        
    if data.ndim == 1:
        data = data[None, :] if data.size else np.empty((0, expected_len+1))
        
    rows, cols = data.shape
    ok_cols = cols == expected_len + 1
    ok_nan = not np.isnan(data).any()

    print(f"\n── Validation summary : {csv_path.name} ──")
    print(f"Rows : {rows}")
    print(f"Columns : {cols}  (expected {expected_len+1}) {'✓' if ok_cols else '✗'}")
    print(f"Contains NaNs : {'no ✓' if ok_nan else 'YES ✗'}")
    
    if rows:
        idxs, cnts = np.unique(data[:,0].astype(int), return_counts=True)
        print("Class counts:")
        for idx, cnt in zip(idxs, cnts):
            lbl = labels[idx] if idx < len(labels) else f"<id {idx}>"
            print(f"  {idx:2d} {lbl:<15} {cnt}")
            
    print("Result :", "✔ OK" if ok_cols and ok_nan else "✗ FAILED")

# ──────────────────────────── Main Loop ───────────────────────────────
def run_recorder(args):
    """Main recording loop and state machine."""
    # Initialize components
    cam = open_camera(args)
    hands = init_mediapipe(args)
    window = setup_window()

    # Setup session
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = args.out_dir / run_ts
    base_dir.mkdir(parents=True, exist_ok=True)

    # State variables
    mode = None
    labels = []
    label_idx = -1
    recording = False
    counters = {}
    writer = None
    csv_fh = None
    csv_path = None
    history_buf = deque(maxlen=HISTORY_LEN)

    print("[INFO] Window ready. Press K or H to choose mode.")

    while True:
        # Frame capture and processing
        frame = cam.grab()
        if frame is None:
            cv.waitKey(1)  # Keep UI responsive
            continue
        frame = cv.flip(frame, 1)

        # Key handling
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # Esc → quit
            break

        # Mode selection
        if mode is None and key in (ord('k'), ord('h')):
            mode = 'static' if key == ord('k') else 'history'
            default_csv = STATIC_LABELS_CSV if mode=='static' else HISTORY_LABELS_CSV
            lbl_csv = args.classes_file or default_csv
            labels = load_labels(lbl_csv)
            counters = {lbl:0 for lbl in labels}
            csv_path, writer, csv_fh = create_session(mode, base_dir)
            print(f"[INFO] {mode.upper()} mode selected. Labels loaded from {lbl_csv}")
            continue

        # Label selection
        if mode and key in range(ord('0'), ord('9')+1):
            idx = key - 48
            if idx < len(labels):
                label_idx = idx
                print(f"[INFO] Selected label [{idx}] {labels[idx]}")

        # Snapshot capture
        if key in (10, 13) and mode:
            if label_idx < 0:
                print("[WARN] Choose a label 0-9 first.")
            else:
                if feature_vec and len(feature_vec) == EXPECTED_LEN[mode]:
                    writer.writerow([label_idx, *feature_vec])
                    counters[labels[label_idx]] += 1
                    print(f"[INFO] Snapshot saved for '{labels[label_idx]}' "
                          f"(total {counters[labels[label_idx]]})")
                else:
                    print("[WARN] Feature vector not ready – keep hand steady.")

        # MediaPipe processing
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        feature_vec = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = calc_landmark_list(frame, hand)

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Feature vector generation
            if mode == 'static':
                feature_vec = vec_static(lm)
            elif mode == 'history':
                mcp_idx = (5,9,13,17)
                history_buf.append([lm[i] for i in mcp_idx])
                if len(history_buf) == HISTORY_LEN:
                    feature_vec = vec_history(frame, history_buf)
        else:
            if mode == 'history':
                history_buf.append([[0,0]]*4)

        # Visualization
        if mode == 'history':
            frame = draw_point_history(frame, history_buf)

        fps = cam.fps.get()
        debug = draw_overlay(frame, fps, mode, labels,
                           label_idx, False, counters)
        cv.imshow(window, debug)

    # Cleanup
    cam.close()
    if csv_fh:
        csv_fh.close()
        validate_csv(csv_path, EXPECTED_LEN[mode], labels)

# ──────────────────────────── Entry Point ─────────────────────────────
if __name__ == "__main__":
    run_recorder(parse_cli())
