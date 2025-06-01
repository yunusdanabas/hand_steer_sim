#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steering_sign_recognition_node.py
ROS node: /image_raw  →  /gesture/steering_static, /gesture/steering_dyn   (string)
"""

import rospy, cv2 
from sensor_msgs.msg import Image
from std_msgs.msg    import String
from cv_bridge       import CvBridge, CvBridgeError
from scripts.cvfpscalc import CvFpsCalc
from hand_steer_sim.model.steering_mode import SteeringRecognition
import numpy as np

class SteeringSignRecognitionNode:
    def __init__(self):
        rospy.init_node('steering_sign_recognition_node')
        
        # Get parameters
        p = lambda name, default: rospy.get_param(name, default)
        
        # Topics
        self._img_topic = p("~subscribe_image_topic", "/image_raw")
        stat_topic = p("~publish_static_topic", "/gesture/steering_static")
        dyn_topic = p("~publish_dyn_topic", "/gesture/steering_dyn")
        
        # Model paths - using absolute paths from launch file
        key_lbl = p("~steering_keypoint_classifier_label", 
                   "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier_label.csv")
        key_mod = p("~steering_keypoint_classifier_model",
                   "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/keypoint_classifier/keypoint_classifier.tflite")
        hist_lbl = p("~steering_classifier_label",
                    "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier_label.csv")
        hist_mod = p("~steering_classifier_model",
                    "$(find hand_steer_sim)/hand_steer_sim/model/steering_mode/point_history_classifier/point_history_classifier.tflite")
        
        # Other parameters
        self._show_image = p("~show_image", True)
        use_gpu = p("~use_gpu", False)
        
        # Publishers
        self._pub_stat = rospy.Publisher(stat_topic, String, queue_size=10)
        self._pub_dyn = rospy.Publisher(dyn_topic, String, queue_size=10)
        
        # Subscriber
        self._bridge = CvBridge()
        rospy.Subscriber(self._img_topic, Image, self._img_cb, queue_size=1)
        
        # Initialize detector
        self._detector = SteeringRecognition(
            key_lbl_csv=key_lbl,
            key_tflite=key_mod,
            hist_lbl_csv=hist_lbl,
            hist_tflite=hist_mod,
            use_gpu=use_gpu
        )
        
        # FPS calculation
        self._fpscalc = CvFpsCalc(buffer_len=10)
        
        rospy.loginfo("SteeringSignRecognitionNode ready (sub %s  pubs %s %s)",
                      self._img_topic, stat_topic, dyn_topic)
    
    def _img_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr_throttle(5.0, "cv_bridge: %s", e)
            return
            
        # Process frame
        result = self._detector.recognise(frame)
        
        # Publish gestures
        self._pub_stat.publish(result.static_label)
        self._pub_dyn.publish(result.dynamic_label)
        
        # Display if enabled
        if self._show_image:
            fps = self._fpscalc.get()
            # Overlay text
            cv2.putText(result.dbg_img, f"{result.static_label} | {result.dynamic_label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(result.dbg_img, f"{result.static_label} | {result.dynamic_label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
            # Second line: FPS
            cv2.putText(result.dbg_img, f"FPS:{fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(result.dbg_img, f"FPS:{fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display debug image
            cv2.imshow("Steering Recognition", result.dbg_img)
            cv2.waitKey(1)

    @staticmethod
    def cleanup():
        cv2.destroyAllWindows()

def main():
    try:
        node = SteeringSignRecognitionNode()
        rospy.on_shutdown(node.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
