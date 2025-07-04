cmake_minimum_required(VERSION 3.0.2)
project(arena)

find_package(catkin REQUIRED)

catkin_package()

# Install folders like launch, worlds, models
install(DIRECTORY launch worlds models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
