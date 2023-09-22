cmake_minimum_required(VERSION 2.8.3)
project(modvo)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  qcar
  rospy
  sensor_msgs
  std_msgs
  message_filters
  cv_bridge
)

catkin_package(
 CATKIN_DEPENDS rospy
)

catkin_install_python(PROGRAMS scripts/rgb_stream_node.py
                               scripts/stream_dataloader.py 
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)