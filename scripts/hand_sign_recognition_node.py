#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand_sign_recognition_node.py
ROS node: subscribe to /image, publish gesture label.
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv

from hand_steer_sim.model.static_mode.gesture_recognition import GestureRecognition
from scripts.cvfpscalc import CvFpsCalc


class HandSignRecognitionNode:
    def __init__(self) -> None:
        # ---------------- parameters ---------------- #
        p     = rospy.get_param
        self.image_topic   = p("~subscribe_image_topic", "/image_raw")
        self.gesture_topic = p("~publish_gesture_topic",  "/gesture/hand_sign")
        label_path = p("~keypoint_classifier_label",
                       "hand_steer_sim/model/static_mode/keypoint_classifier/keypoint_classifier_label.csv")
        model_path = p("~keypoint_classifier_model",
                       "hand_steer_sim/model/static_mode/keypoint_classifier/keypoint_classifier.tflite")
        self.show_image    = p("~show_image", True)
        use_gpu            = p("~use_gpu", False)
        self.log_timing    = p("~log_timing", False)

        # ---------------- helpers ------------------- #
        self._bridge   = CvBridge()
        self._fpscalc  = CvFpsCalc(buffer_len=10)
        self._detector = GestureRecognition(label_path, model_path, use_gpu=use_gpu)
        
        # ---------------- ROS I/O ------------------- #
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self._pub = rospy.Publisher(self.gesture_topic, String, queue_size=10)

        rospy.loginfo("hand_sign_recognition_node ready (sub %s  pub %s)",
                      self.image_topic, self.gesture_topic)

    # ------------------------------------------------------------------ #
    def image_callback(self, msg: Image) -> None:
        """Decode image, run gesture recognition, optionally log timing."""
        # ---------- (1) Decode ROS Image → OpenCV frame ----------
        try:
            t_decode_start = rospy.get_time()
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            t_decode = rospy.get_time() - t_decode_start
        except CvBridgeError as err:
            rospy.logerr_throttle(5.0, "cv_bridge: %s", err)
            return

        # ---------- (2) Gesture inference ----------
        t_infer_start = rospy.get_time()
        dbg_img, gesture = self._detector.recognise(frame)
        t_infer = rospy.get_time() - t_infer_start

        # ---------- (3) Publish gesture ----------
        t_pub_start = rospy.get_time()
        self._pub.publish(gesture)
        t_pub = rospy.get_time() - t_pub_start

        # ---------- (4) Optional display ----------
        t_disp = 0.0
        if self.show_image:
            t_disp_start = rospy.get_time()
            fps = self._fpscalc.get()
            cv.imshow("Gesture-Recognition (ROS)",
                      self._detector.draw_fps_info(dbg_img, fps))
            cv.waitKey(1)
            t_disp = rospy.get_time() - t_disp_start
        else:
            # If no display, still compute fps for logging consistency
            fps = self._fpscalc.get()

        # ---------- (5) Optional timing log ----------
        if self.log_timing:
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
                "  Last Gesture: %s",
                self._detector.use_gpu,
                t_decode*1e3, t_infer*1e3, t_pub*1e3, t_disp*1e3, dt_total*1e3,
                callback_fps, fps, gesture
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def cleanup() -> None:
        cv.destroyAllWindows()


def main() -> None:
    rospy.init_node("hand_sign_recognition")
    node = HandSignRecognitionNode()
    rospy.on_shutdown(node.cleanup)
    rospy.spin()


if __name__ == "__main__":
    main()
