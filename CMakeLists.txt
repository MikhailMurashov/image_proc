cmake_minimum_required(VERSION 3.9)
project(image_proc)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV COMPONENTS REQUIRED opencv_imgcodecs)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

find_package(OpenMP REQUIRED)

add_executable(image_proc main.cpp)
target_link_libraries(image_proc ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
