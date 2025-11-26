cmake_minimum_required(VERSION 3.8)
project(nebula_vision)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(nebula_interfaces REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_python_install_package(nebula_vision)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY calib
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
