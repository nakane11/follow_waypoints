cmake_minimum_required(VERSION 2.8.3)

project(follow_waypoints)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  move_base_msgs
  rospy
  smach
  std_msgs
  message_generation
)

add_action_files(
  DIRECTORY action
  FILES
  Waypoints.action
)

catkin_python_setup()

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS actionlib actionlib_msgs geometry_msgs move_base_msgs rospy smach std_msgs
)

catkin_install_python(
  PROGRAMS nodes/follow_waypoints
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

