cmake_minimum_required(VERSION 3.0.2)
project(hand_steer_sim)

###############################################################################
# 1.  Find Catkin + the other ROS run-time deps
###############################################################################
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

###############################################################################
# 2.  Tell catkin to run setup.py          (creates proper PYTHONPATH install)
###############################################################################
catkin_python_setup()

###############################################################################
# 3.  Normal catkin_package() declaration  (nothing special for pure-python)
###############################################################################
catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs rospy sensor_msgs std_msgs
)

###############################################################################
# 4.  Include dirs (only catkin headers, since we have no C++ code here)
###############################################################################
include_directories(${catkin_INCLUDE_DIRS})

###############################################################################
# 5.  (Optional) install launch / config files
###############################################################################
## catkin_install_python(PROGRAMS
##   scripts/hand_sign_recognition_node.py
##   scripts/camera_publisher_node.py
##   scripts/gesture_to_twist_node.py
##   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
## )
## install(DIRECTORY launch config
##   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
## )
