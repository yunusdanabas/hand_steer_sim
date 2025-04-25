#!/usr/bin/env python3

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class CameraPublisher:
    def __init__(self):
        """
        Initialize the CameraPublisher class
        """

        # Get parameters (with defaults if not provided)
        self.camera_name = rospy.get_param("~camera_name", 0)   # Default to device 0
        self.topic_name = rospy.get_param("~topic_name", "image_raw")
        self.publish_rate = rospy.get_param("~publish_rate", 30)

        # Create a publisher
        self.image_pub = rospy.Publisher(self.topic_name, Image, queue_size=10)

        # CvBridge for OpenCV <-> ROS Image conversion
        self.bridge = CvBridge()

        # Attempt to open the camera device
        self.capture = cv2.VideoCapture(self.camera_name)
        if not self.capture.isOpened():
            rospy.logerr(f"Failed to open camera: {self.camera_name}")

        # Create a ROS rate object for publishing frequency
        self.rate = rospy.Rate(self.publish_rate)

    def publish_image(self):
        """
        Capture frames from the camera and publish to the specified topic
        """
        while not rospy.is_shutdown():
            ret, frame = self.capture.read()
            if not ret:
                rospy.logerr("Could not grab a frame from the camera!")
                # If you want to stop entirely, you could break.
                # But sometimes it's better just to keep trying:
                # break
                continue

            # Convert the OpenCV image to a ROS Image message
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_msg.header.stamp = rospy.Time.now()
                self.image_pub.publish(img_msg)
            except CvBridgeError as e:
                rospy.logerr(f"cv_bridge error: {e}")

            # Sleep to maintain the desired publish rate
            self.rate.sleep()

    def cleanup(self):
        """
        Release resources
        """
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node("camera_interface", anonymous=True)
    rospy.loginfo("Starting camera publisher node...")

    # Create the CameraPublisher object
    cam_pub = CameraPublisher()

    try:
        # Start publishing images
        cam_pub.publish_image()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down...")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard Interrupt received. Shutting down...")
    finally:
        # Release camera and other resources
        cam_pub.cleanup()
        rospy.loginfo("Camera publisher node has been shut down.")
