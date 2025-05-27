#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_publisher_node.py
ROS node: publish frames from an OpenCV-compatible camera or Intel RealSense.
"""

import cv2
import rospy
import numpy as np
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CameraPublisherNode:
    def __init__(self):
        p = rospy.get_param
        device        = p("~camera_name", 0)          # int or /dev/xxx
        self.topic    = p("~topic_name",  "image_raw")
        fps           = p("~publish_rate", 30)
        width         = p("~width", 960)
        height        = p("~height", 540)
        use_realsense = p("~use_realsense", False)

        self._pub    = rospy.Publisher(self.topic, Image, queue_size=10)
        self._bridge = CvBridge()

        # Camera setup based on type
        if use_realsense:
            self._pipe = rs.pipeline()
            self._cfg = rs.config()
            self._cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self._pipe.start(self._cfg)
            self._grab = lambda: np.asanyarray(self._pipe.wait_for_frames().get_color_frame().get_data())
        else:
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._grab = lambda: self._cap.read()[1]

        if not use_realsense and not self._cap.isOpened():
            rospy.logfatal("Cannot open camera %s", device)
            raise SystemExit

        self._loop_rate = rospy.Rate(fps)
        camera_type = "RealSense" if use_realsense else f"Camera {device}"
        rospy.loginfo("%s streaming → %s @ %d Hz", camera_type, self.topic, fps)

    # ------------------------------------------------------------------ #
    def spin(self) -> None:
        while not rospy.is_shutdown():
            try:
                frame = self._grab()
                if frame is None:
                    rospy.logerr_throttle(5.0, "Camera read failed")
                    self._loop_rate.sleep()
                    continue

                msg = self._bridge.cv2_to_imgmsg(frame, "bgr8")
                msg.header.stamp = rospy.Time.now()
                self._pub.publish(msg)
            except CvBridgeError as err:
                rospy.logerr_throttle(5.0, "cv_bridge: %s", err)
            except Exception as e:
                rospy.logerr_throttle(5.0, "Camera error: %s", str(e))

            self._loop_rate.sleep()

    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        if hasattr(self, '_pipe'):
            self._pipe.stop()
        if hasattr(self, '_cap'):
            self._cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    rospy.init_node("camera_interface")
    node = CameraPublisherNode()
    rospy.on_shutdown(node.cleanup)
    node.spin()


if __name__ == "__main__":
    main()
