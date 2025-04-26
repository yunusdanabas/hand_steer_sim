#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: publish frames from an OpenCV-compatible camera.
"""

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CameraPublisherNode:
    def __init__(self):
        p = rospy.get_param
        device        = p("~camera_name", 0)          # int or /dev/xxx
        self.topic    = p("~topic_name",  "image_raw")
        fps           = p("~publish_rate", 30)

        self._pub    = rospy.Publisher(self.topic, Image, queue_size=10)
        self._bridge = CvBridge()
        self._cap    = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            rospy.logfatal("Cannot open camera %s", device)
            raise SystemExit

        self._loop_rate = rospy.Rate(fps)
        rospy.loginfo("CameraPublisher streaming %s → %s @ %d Hz", device, self.topic, fps)

    # ------------------------------------------------------------------ #
    def spin(self) -> None:
        while not rospy.is_shutdown():
            ok, frame = self._cap.read()
            if not ok:
                rospy.logerr_throttle(5.0, "Camera read failed")
                self._loop_rate.sleep()
                continue
            try:
                msg = self._bridge.cv2_to_imgmsg(frame, "bgr8")
                msg.header.stamp = rospy.Time.now()
                self._pub.publish(msg)
            except CvBridgeError as err:
                rospy.logerr_throttle(5.0, "cv_bridge: %s", err)

            self._loop_rate.sleep()

    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        self._cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    rospy.init_node("camera_interface")
    node = CameraPublisherNode()
    rospy.on_shutdown(node.cleanup)
    node.spin()


if __name__ == "__main__":
    main()
