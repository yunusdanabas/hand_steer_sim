#!/usr/bin/env python3
import rospy
import cv2 as cv

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from gesture_recognition import GestureRecognition
from cvfpscalc import CvFpsCalc


class HandSignRecognition:
    """
    Class that subscribes to an image topic, performs gesture recognition,
    and publishes detected gestures.
    """

    def __init__(self):
        # Retrieve parameters from the private namespace: '~'
        # Using defaults if they are not set on the parameter server
        self.subscribe_image_topic = rospy.get_param(
            "~subscribe_image_topic",
            "/camera/image_raw"
        )
        self.publish_gesture_topic = rospy.get_param(
            "~publish_gesture_topic",
            "/gesture/hand_sign"
        )
        self.keypoint_classifier_label = rospy.get_param(
            "~keypoint_classifier_label",
            "model/keypoint_classifier_label.csv"
        )
        self.keypoint_classifier_model = rospy.get_param(
            "~keypoint_classifier_model",
            "model/keypoint_classifier.tflite"
        )
        # Whether or not to show the debug image in a window
        self.show_image = rospy.get_param("~show_image", True)

        # Create subscriber and publisher
        self.image_subscriber = rospy.Subscriber(
            self.subscribe_image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        self.gesture_publisher = rospy.Publisher(
            self.publish_gesture_topic,
            String,
            queue_size=10
        )

        # Initialize gesture detector
        self.gesture_detector = GestureRecognition(
            self.keypoint_classifier_label,
            self.keypoint_classifier_model
        )

        # Initialize CvBridge for Image <-> OpenCV conversion
        self.bridge = CvBridge()

        # FPS calculator (optional utility for debugging performance)
        self.cv_fps_calc = CvFpsCalc(buffer_len=10)

        rospy.loginfo("HandSignRecognition node initialized.")

    def image_callback(self, image_msg):
        """
        Callback for image subscriber. Performs gesture recognition
        and publishes the recognized gesture as a string.

        Args:
            image_msg (sensor_msgs.msg.Image): ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg)

            # Perform gesture recognition
            debug_image, gesture = self.gesture_detector.recognize(cv_image)

            # Publish recognized gesture
            self.gesture_publisher.publish(gesture)

            # If show_image param is True, display the image in a window
            if self.show_image:
                fps = self.cv_fps_calc.get()  # estimate current FPS
                debug_image = self.gesture_detector.draw_fps_info(debug_image, fps)
                cv.imshow("ROS Gesture Recognition", debug_image)
                cv.waitKey(1)  # small delay so the window can refresh

        except CvBridgeError as err:
            rospy.logerr(f"CvBridge Error: {err}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in image_callback: {e}")

    def cleanup(self):
        """
        Release resources like OpenCV windows when shutting down.
        """
        cv.destroyAllWindows()
        rospy.loginfo("Shut down: Closed all OpenCV windows.")


def main():
    """
    Main entry point for the node. Initializes ROS, creates HandSignRecognition,
    and keeps the node alive.
    """
    rospy.init_node("hand_sign_recognition", anonymous=False)
    rospy.loginfo("Starting 'hand_sign_recognition' node...")

    hand_sign_node = HandSignRecognition()

    # Keep the node alive until it is stopped
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupt received. Shutting down node...")
    finally:
        # Clean up any resources
        hand_sign_node.cleanup()


if __name__ == "__main__":
    main()
