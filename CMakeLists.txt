cmake_minimum_required(VERSION 3.1.3)
project(airbot_play_control_upper)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/airbot_play_follow_joint_trajectory_server.py
  scripts/airbot_play_keyboard.py
  scripts/airbot_play_joyControl.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)