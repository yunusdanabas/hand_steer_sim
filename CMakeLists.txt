cmake_minimum_required(VERSION 3.0.2)
project(hand_steer_sim)

###############################################################################
# 1.  Find Catkin & the run-time deps
###############################################################################
find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  sensor_msgs
  geometry_msgs
  std_msgs
)

###############################################################################
# 2.  Pure-Python: let Catkin install our package via setup.py
###############################################################################
catkin_python_setup()

###############################################################################
# 3.  Declare the Catkin package (only dependencies – no libs)
###############################################################################
catkin_package(
  CATKIN_DEPENDS rospy cv_bridge sensor_msgs geometry_msgs std_msgs
)

###############################################################################
# 4.  We have no C++ include dirs, but keep the macro for completeness
###############################################################################
include_directories(${catkin_INCLUDE_DIRS})

###############################################################################
# 5.  Install section
###############################################################################

# 5.1  Python entry-points (rosrun / roslaunch can find them)
catkin_install_python(
  PROGRAMS
    scripts/camera_publisher_node.py
    scripts/hand_sign_recognition_node.py
    scripts/gesture_to_twist_node.py
    scripts/steering_sign_recognition_node.py
    scripts/wheel_to_twist_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 5.2  Non-code assets: launch files, URDF, config, models
install(
  DIRECTORY
    launch
    urdf
    config
    hand_steer_sim/model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

###############################################################################
# 6.  (Optional) Tests – add_rostest(...) here when you have them
###############################################################################
