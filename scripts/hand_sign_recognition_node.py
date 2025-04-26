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
