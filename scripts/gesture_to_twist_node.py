#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate gesture strings to geometry_msgs/Twist.
"""

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class GestureToTwistNode:
    _MAX_LIN =  0.5   # [m/s]  hard safety limits
    _MAX_ANG =  1.5   # [rad/s]

    def __init__(self):
        p = rospy.get_param
        gesture_topic = p("~publish_gesture_topic", "/gesture/hand_sign")
        cmd_vel_topic = p("~cmd_vel_topic",          "/robot_diff_drive_controller/cmd_vel")
        self._dlin    = p("~linear_vel_inc",   0.05)
        self._dang    = p("~angular_vel_inc",  0.2)

        self._twist       = Twist()
        self._publisher   = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        rospy.Subscriber(gesture_topic, String, self.callback, queue_size=5)

        rospy.loginfo("GestureToTwist ready (sub %s  pub %s)", gesture_topic, cmd_vel_topic)


    def callback(self, msg: String) -> None:
        g = msg.data
        if   g == "Forward":    self._twist.linear.x  += self._dlin
        elif g == "Backward":   self._twist.linear.x  -= self._dlin
        elif g == "Turn Right": self._twist.angular.z -= self._dang
        elif g == "Turn Left":  self._twist.angular.z += self._dang
        elif g == "Stop":       self._twist.linear.x = self._twist.angular.z = 0.0
        elif g in ("Go", "NONE"):
            pass
        else:
            rospy.logwarn_throttle(2.0, "Unknown gesture '%s'", g)
            return

        # clamp for safety
        self._twist.linear.x  = max(-self._MAX_LIN,  min(self._MAX_LIN,  self._twist.linear.x))
        self._twist.angular.z = max(-self._MAX_ANG,  min(self._MAX_ANG,  self._twist.angular.z))

        self._publisher.publish(self._twist)
        rospy.loginfo_throttle(0.5, "%s → v=%.2f  ω=%.2f", g, self._twist.linear.x, self._twist.angular.z)


def main() -> None:
    rospy.init_node("gesture_to_twist")
    GestureToTwistNode()
    rospy.spin()


if __name__ == "__main__":
    main()
