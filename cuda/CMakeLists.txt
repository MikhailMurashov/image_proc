cmake_minimum_required(VERSION 3.9)
project(image_proc_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include dir: ${OpenCV_INCLUDE_DIRS}")

include_directories(./include ${OpenCV_INCLUDE_DIRS})

add_library(proc STATIC source/blur.cu source/grayscale.cu source/hello_world.cu
                       include/blur.h include/grayscale.h include/hello_world.h)
target_compile_features(proc PUBLIC cxx_std_11)
set_target_properties(proc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(image_proc_cuda main.cpp)
set_target_properties(image_proc_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(image_proc_cuda proc ${OpenCV_LIBS})
