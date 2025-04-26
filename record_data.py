#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


from scripts import CvFpsCalc
from hand_steer_sim.model.steering_mode.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from hand_steer_sim.model.steering_mode.point_history_classifier.point_history_classifier import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument Parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera Preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model Loading #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Label Loading ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement Module ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate History #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger Gesture History ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Key Processing (ESC: Exit) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera Capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror image
        debug_image = image.copy()

        # Detection Execution #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        # ---------------------------------------------------------------------------
        # Per-frame hand-processing
        # ---------------------------------------------------------------------------
        if results.multi_hand_landmarks:
            for hand_lms, handed in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):

                # ───────────── 1.  Landmark → features & logging ─────────────
                landmark_list = calc_landmark_list(debug_image, hand_lms)

                pre_landmark_vec = pre_process_landmark(landmark_list)
                pre_history_vec  = pre_process_point_history(debug_image, point_history)

                logging_csv(number, mode, pre_landmark_vec, pre_history_vec)

                # Hand-sign (static) classification
                hand_sign_id = keypoint_classifier(pre_landmark_vec)

                # Point history buffer for dynamic gestures
                if hand_sign_id == 2:                          # “pointing” sign
                    point_history.append(landmark_list[8])     # tip of index finger
                else:
                    point_history.append([0, 0])

                # Dynamic-gesture classification when buffer full
                finger_gesture_id = 0
                if len(pre_history_vec) == history_length * 2:
                    finger_gesture_id = point_history_classifier(pre_history_vec)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]

                # ───────────── 2.  Drawing (MediaPipe) ─────────────
                # Landmarks & skeleton
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Text overlay (handedness + static/dynamic labels)
                label_static  = keypoint_classifier_labels[hand_sign_id]
                label_dynamic = point_history_classifier_labels[most_common_fg_id]

                wrist = hand_lms.landmark[0]            # landmark-0: wrist
                img_h, img_w = debug_image.shape[:2]
                txt_origin = (int(wrist.x * img_w), int(wrist.y * img_h) - 10)

                text = f"{handed.classification[0].label}:{label_static}:{label_dynamic}"
                cv.putText(debug_image, text, txt_origin,
                        cv.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv.LINE_AA)

        else:
            point_history.append([0, 0])

        # Shared overlays
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen Reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Key Points
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    # Conversion to Relative Coordinates
    pts = np.asarray(landmark_list, dtype=np.float32)
    pts -= pts[0]
    flat = pts.reshape(-1)
    max_val = np.max(np.abs(flat)) or 1.0
    return (flat / max_val).tolist()


def pre_process_point_history(image, point_history):
    import numpy as np

    pts = np.asarray(point_history, dtype=np.float32)
    if pts.size == 0:
        return []
    pts -= pts[0]
    h, w = image.shape[:2]
    pts[:, 0] /= w
    pts[:, 1] /= h
    return pts.flatten().tolist()


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
