#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steering_sign_recognition_node.py
ROS node: /image_raw  →  /gesture/steering_static, /gesture/steering_dyn   (string)
"""

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from scripts.cvfpscalc import CvFpsCalc
from hand_steer_sim.model.steering_mode import SteeringRecognition


class SteeringSignRecognitionNode:
    def __init__(self):
        # Initialize node
        rospy.init_node('steering_sign_recognition_node')

        # Get parameters
        p = lambda name, default: rospy.get_param(name, default)
        self._img_topic = p("~subscribe_image_topic", "/image_raw")
        stat_topic = p("~publish_static_topic", "/gesture/steering_static")
        dyn_topic = p("~publish_dyn_topic", "/gesture/steering_dyn")

        key_lbl = p("~steering_keypoint_classifier_label",
                    "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier_label.csv")
        key_mod = p("~steering_keypoint_classifier_model",
                    "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier.tflite")
        hist_lbl = p("~steering_classifier_label",
                     "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier_label.csv")
        hist_mod = p("~steering_classifier_model",
                     "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier.tflite")

        self._show_image = p("~show_image", True)
        use_gpu = p("~use_gpu", False)
        self._log_timing = p("~log_timing", False)  

        # Publishers
        self._pub_stat = rospy.Publisher(stat_topic, String, queue_size=10)
        self._pub_dyn = rospy.Publisher(dyn_topic, String, queue_size=10)

        # Subscriber
        self._bridge = CvBridge()
        rospy.Subscriber(self._img_topic, Image, self._img_cb, queue_size=1)

        # Initialize detector
        self._detector = SteeringRecognition(
            key_lbl_csv=key_lbl,
            key_tflite=key_mod,
            hist_lbl_csv=hist_lbl,
            hist_tflite=hist_mod,
            use_gpu=use_gpu
        )

        # FPS calculation
        self._fpscalc = CvFpsCalc(buffer_len=10)

        rospy.loginfo(
            "SteeringSignRecognitionNode ready (sub %s  pubs %s %s)",
            self._img_topic, stat_topic, dyn_topic
        )

    def _img_cb(self, msg: Image):
        # 1) Decode ROS Image → OpenCV frame
        try:
            t_decode_start = rospy.get_time()
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            t_decode = rospy.get_time() - t_decode_start
        except CvBridgeError as e:
            rospy.logerr_throttle(5.0, "cv_bridge: %s", e)
            return

        # 2) Gesture inference
        t_infer_start = rospy.get_time()
        result = self._detector.recognise(frame)
        t_infer = rospy.get_time() - t_infer_start

        # 3) Publish static and dynamic labels
        t_pub_start = rospy.get_time()
        self._pub_stat.publish(result.static_label)
        self._pub_dyn.publish(result.dynamic_label)
        t_pub = rospy.get_time() - t_pub_start

        # 4) Display if enabled
        t_disp = 0.0
        fps = self._fpscalc.get()
        if self._show_image:
            t_disp_start = rospy.get_time()
            # Overlay text: static | dynamic
            cv2.putText(
                result.dbg_img,
                f"{result.static_label} | {result.dynamic_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                result.dbg_img,
                f"{result.static_label} | {result.dynamic_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
            # Second line: FPS
            cv2.putText(
                result.dbg_img,
                f"FPS:{fps:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                result.dbg_img,
                f"FPS:{fps:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow("Steering Recognition", result.dbg_img)
            cv2.waitKey(1)
            t_disp = rospy.get_time() - t_disp_start

        # 5) Optional timing log
        if self._log_timing:
            dt_total = t_decode + t_infer + t_pub + t_disp
            callback_fps = 1.0 / dt_total if dt_total > 0 else 0.0

            rospy.loginfo_throttle(
                1.0,
                "Performance Metrics:\n"
                "  System Mode:\n"
                "    - GPU Enabled: %s\n"
                "  Timing (ms):\n"
                "    - Image Decode: %.1f\n"
                "    - Gesture Inference: %.1f\n"
                "    - Message Publish: %.1f\n"
                "    - Display: %.1f\n"
                "    - Total Processing: %.1f\n"
                "  FPS:\n"
                "    - Callback FPS: %.1f\n"
                "    - Display FPS: %.1f\n"
                "  Last Labels:\n"
                "    - Static: %s\n"
                "    - Dynamic: %s",
                self._detector.use_gpu,
                t_decode * 1e3,
                t_infer * 1e3,
                t_pub * 1e3,
                t_disp * 1e3,
                dt_total * 1e3,
                callback_fps,
                fps,
                result.static_label,
                result.dynamic_label
            )

    @staticmethod
    def cleanup():
        cv2.destroyAllWindows()


def main():
    try:
        node = SteeringSignRecognitionNode()
        rospy.on_shutdown(node.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
