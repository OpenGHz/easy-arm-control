cmake_minimum_required(VERSION 3.5)
project(easy_arm_control)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/airbot_play_follow_joint_trajectory_server.py
  scripts/airbot_play_keyboard.py
  scripts/airbot_play_joyControl.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
