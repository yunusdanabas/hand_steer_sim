#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class GestureController:
    """
    ROS node that subscribes to hand gesture commands and translates them into Twist commands
    for controlling a robot's differential drive.
    """

    def __init__(self):
        # Initialize the node with a unique name if needed.
        rospy.init_node('hand_sign_control', anonymous=True)

        # Retrieve parameters with default values.
        self.gesture_topic = rospy.get_param("~publish_gesture_topic", "/hand_sign_recognition/gesture")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/robot_diff_drive_controller/cmd_vel")
        self.linear_vel_inc = rospy.get_param("~linear_vel_inc", 0.01)   # [m/s]
        self.angular_vel_inc = rospy.get_param("~angular_vel_inc", 0.1)    # [rad/s]

        # Create subscriber for hand gestures.
        self.gesture_subscriber = rospy.Subscriber(self.gesture_topic, String, self.callback, queue_size=1)

        # Create publisher for velocity commands.
        self.vel_publisher = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        # Initialize the Twist message for velocity commands.
        self.vel_msg = Twist()

        rospy.loginfo("GestureController initialized. Subscribing to: %s, publishing to: %s",
                      self.gesture_topic, self.cmd_vel_topic)

    def callback(self, gesture):
        """
        Callback function that converts received hand gesture commands into
        velocity commands and publishes them.

        Supported gestures:
          - "Forward": Increase forward velocity.
          - "Backward": Increase reverse velocity.
          - "Turn Right": Increase rightward angular velocity.
          - "Turn Left": Increase leftward angular velocity.
          - "Stop": Halt all motion.
          - "Go" or "NONE": Continue with the current velocity (i.e. do nothing).
        
        Args:
            gesture (std_msgs.msg.String): Gesture command.
        """
        if gesture.data == "Forward":
            self.vel_msg.linear.x += self.linear_vel_inc
            self.vel_msg.angular.z = 0.0
        elif gesture.data == "Backward":
            self.vel_msg.linear.x -= self.linear_vel_inc
            self.vel_msg.angular.z = 0.0
        elif gesture.data == "Turn Right":
            self.vel_msg.angular.z -= self.angular_vel_inc
        elif gesture.data == "Turn Left":
            self.vel_msg.angular.z += self.angular_vel_inc
        elif gesture.data == "Stop":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.angular.z = 0.0
        elif gesture.data in ["Go", "NONE"]:
            # "Go" and "NONE": continue with the current velocity command.
            pass
        else:
            rospy.logwarn("Received unknown gesture: %s", gesture.data)
            return

        self.vel_publisher.publish(self.vel_msg)
        rospy.loginfo("%s -> Linear: %.6f m/s, Angular: %.6f rad/s",
                      gesture.data, self.vel_msg.linear.x, self.vel_msg.angular.z)

if __name__ == "__main__":
    try:
        gesture_controller = GestureController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down GestureController node.")
