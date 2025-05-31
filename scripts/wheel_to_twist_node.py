#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wheel_to_twist_node.py
Map steering-gesture strings → geometry_msgs/Twist.
"""

import rospy
from std_msgs.msg    import String
from geometry_msgs.msg import Twist

class WheelToTwist:
    _MAX_LIN = 0.6   # m/s
    _MAX_ANG = 2.0   # rad/s

    _ANG_INC = {           # angular step per command
        "Turn Left":        +0.2,
        "Turn Left Fast":   +0.4,
        "Turn Right":       -0.2,
        "Turn Right Fast":  -0.4,
    }

    def __init__(self):
        p   = rospy.get_param
        sub = p("~wheel_topic", "/gesture/wheel_sign")
        pub = p("~cmd_vel_topic","/robot_diff_drive_controller/cmd_vel")
        self._tw  = Twist()
        self._pub = rospy.Publisher(pub, Twist, queue_size=10)
        rospy.Subscriber(sub, String, self.cb, queue_size=5)
        rospy.loginfo("WheelToTwist ready (sub %s  pub %s)", sub, pub)

    # ----------------------------------------------------
    def cb(self, msg: String):
        g = msg.data
        if g == "Forward":
            self._tw.linear.x  = min(self._tw.linear.x + 0.05, self._MAX_LIN)
            self._tw.angular.z *= 0.9                     # damp steering
        elif g == "Stop":
            self._tw.linear.x = self._tw.angular.z = 0.0
        elif g in self._ANG_INC:                         # steering
            self._tw.angular.z += self._ANG_INC[g]
        else:
            rospy.logwarn_throttle(2.0, "Unknown gesture '%s'", g); return

        # safety clamp
        self._tw.linear.x  = max(-self._MAX_LIN,  min(self._MAX_LIN,  self._tw.linear.x))
        self._tw.angular.z = max(-self._MAX_ANG,  min(self._MAX_ANG,  self._tw.angular.z))
        self._pub.publish(self._tw)

def main():
    rospy.init_node("wheel_to_twist")
    WheelToTwist()
    rospy.spin()

if __name__ == "__main__":
    main()
