cmake_minimum_required(VERSION 3.0.2)
project(detector)
find_package(catkin REQUIRED)
catkin_python_setup()

catkin_package()

# Executable script(s) to be put in install/bin
catkin_install_python(PROGRAMS
  detector/vanishing_point_node.py        
  detector/detect_aruco.py            
  detector/aruco_pid.py                 
  detector/optical_flow_dense.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)