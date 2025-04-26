#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: subscribe to /image, publish gesture label.
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv

from hand_steer_sim.gesture_recognition import GestureRecognition
from cvfpscalc import CvFpsCalc


class HandSignRecognitionNode:
    def __init__(self) -> None:
        # ---------------- parameters ---------------- #
        p     = rospy.get_param
        self.image_topic   = p("~subscribe_image_topic", "/image_raw")
        self.gesture_topic = p("~publish_gesture_topic",  "/gesture/hand_sign")
        label_path = p("~keypoint_classifier_label", "hand_steer_sim/model/keypoint_classifier/keypoint_classifier_label.csv")
        model_path = p("~keypoint_classifier_model", "hand_steer_sim/model/keypoint_classifier/keypoint_classifier.tflite")
        self.show_image    = p("~show_image", True)

        # ---------------- helpers ------------------- #
        self._bridge  = CvBridge()
        self._fpscalc = CvFpsCalc(buffer_len=10)
        self._detector = GestureRecognition(label_path, model_path)

        # ---------------- ROS I/O ------------------- #
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self._pub = rospy.Publisher(self.gesture_topic, String, queue_size=10)

        rospy.loginfo("hand_sign_recognition_node ready (sub %s  pub %s)",
                      self.image_topic, self.gesture_topic)

    # ------------------------------------------------------------------ #


    # ------------------ Original Callbacks ----------------- #
    
    def image_callback(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr_throttle(5.0, "cv_bridge: %s", err)
            return

        dbg_img, gesture = self._detector.recognise(frame)
        self._pub.publish(gesture)

        if self.show_image:
            fps = self._fpscalc.get()
            cv.imshow("Gesture-Recognition (ROS)", self._detector.draw_fps_info(dbg_img, fps))
            cv.waitKey(1)

    # ------------------------------------------------------------------ #

    # ------------------ Timing Callbacks ----------------- #

    """
    def image_callback(self, msg: Image) -> None:
        
        # Callback for image subscriber: times each stage (decode, inference,
        # publish, display) and logs the results once per second.
        
        # Start overall timer
        t_start = rospy.get_time()

        # 1) Decode ROS Image → OpenCV frame
        try:
            t0 = rospy.get_time()
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            dt_decode = rospy.get_time() - t0
        except CvBridgeError as err:
            rospy.logerr_throttle(5.0, "cv_bridge: %s", err)
            return

        # 2) Gesture inference (MediaPipe + TFLite)
        t1 = rospy.get_time()
        dbg_img, gesture = self._detector.recognise(frame)
        dt_infer = rospy.get_time() - t1

        # 3) Publish gesture string
        t2 = rospy.get_time()
        self._pub.publish(gesture)
        dt_pub = rospy.get_time() - t2

        # 4) Display (only if requested)
        dt_disp = 0.0
        if self.show_image:
            t3 = rospy.get_time()
            fps = self._fpscalc.get()
            cv.imshow("Gesture-Recognition (ROS)", 
                    self._detector.draw_fps_info(dbg_img, fps))
            cv.waitKey(1)
            dt_disp = rospy.get_time() - t3

        # Total processing time
        dt_total = rospy.get_time() - t_start

        # Log once per second to avoid flooding
        rospy.loginfo_throttle(
            1.0,
            "timings (ms): decode=%.1f  infer=%.1f  pub=%.1f  disp=%.1f  total=%.1f",
            dt_decode*1e3, dt_infer*1e3, dt_pub*1e3, dt_disp*1e3, dt_total*1e3
        )

    """
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
