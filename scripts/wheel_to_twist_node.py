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
    _MAX_LIN = 1.0     # m/s   (tune)
    _MAX_ANG = 2.0     # rad/s

    # angular step per dynamic gesture
    _ANG_INC = {
        "Turn Left":       +0.05,
        "Turn Left Fast":  +0.10,
        "Turn Right":      -0.05,
        "Turn Right Fast": -0.10,
    }

    # linear increment for static gestures
    _LIN_STEP = 0.08

    def __init__(self):
        g = rospy.get_param
        stat_topic = g("~static_topic", "/gesture/steering_static")
        dyn_topic  = g("~dyn_topic",    "/gesture/steering_dyn")
        cmd_topic  = g("~cmd_vel_topic","/robot_diff_drive_controller/cmd_vel")

        self._tw           = Twist()
        self._pub          = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self._holding_wheel = False          # ← NEW  (gated steering)

        rospy.Subscriber(stat_topic, String, self._cb_static, queue_size=5)
        rospy.Subscriber(dyn_topic,  String, self._cb_dyn,    queue_size=5)

        rospy.loginfo("WheelToTwist running  (sub %s, %s  →  pub %s)",
                      stat_topic, dyn_topic, cmd_topic)

    # ───────── linear (static gesture) callback ─────────
    def _cb_static(self, msg: String):
        g = msg.data
        # ----------------------------------- update "holding wheel" flag
        self._holding_wheel = (g == "Holding Wheel")

        if g == "NONE":
            return  # Ignore NONE gesture
        elif g == "Stop":
            self._tw.linear.x = 0.0
            self._tw.angular.z = 0.0
        elif g == "Holding Wheel":
            self._tw.linear.x *= 1          # gentle drag so it coasts
        elif g == "Speed Up":
            self._tw.linear.x += self._LIN_STEP
        elif g == "Speed Down":
            self._tw.linear.x -= self._LIN_STEP
        else:
            rospy.logwarn_throttle(2.0, "Unknown static gesture '%s'", g)
            return

        # clamp
        self._tw.linear.x = max(-self._MAX_LIN, min(self._MAX_LIN, self._tw.linear.x))
        self._publish()

    # ───────── angular (dynamic gesture) callback ────────
    def _cb_dyn(self, msg: String):
        g = msg.data

        # Ignore steering commands unless the wheel is currently "held"
        if not self._holding_wheel:
            return

        if g == "NONE":
            return  # Ignore NONE gesture
        elif g in self._ANG_INC:
            self._tw.angular.z += self._ANG_INC[g]
        elif g == "Forward":
            self._tw.angular.z *= 1     # decay back to straight
        else:
            rospy.logwarn_throttle(2.0, "Unknown dynamic gesture '%s'", g)
            return

        # clamp
        self._tw.angular.z = max(-self._MAX_ANG, min(self._MAX_ANG, self._tw.angular.z))
        self._publish()

    # ----------------------------------------------------
    def _publish(self):
        self._pub.publish(self._tw)
        rospy.loginfo_throttle(0.5, "Current velocities: v=%.2f m/s, ω=%.2f rad/s", 
                              self._tw.linear.x, self._tw.angular.z)

def main():
    rospy.init_node("wheel_to_twist")
    WheelToTwist()
    rospy.spin()

if __name__ == "__main__":
    main()
