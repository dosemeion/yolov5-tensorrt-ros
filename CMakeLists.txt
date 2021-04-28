cmake_minimum_required(VERSION 3.0.2)
project(n_camera_obj)

SET(CMAKE_BULID_TYPE "Release")

add_definitions(-std=c++11)
add_definitions(-O2 -pthread)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  message_generation
  cv_bridge
)

find_package(CUDA REQUIRED) 
find_package(OpenCV)

add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxes.msg
)


generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS
  message_runtime
  roscpp 
  std_msgs)

# find_package(CUDA REQUIRED) 
 
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED src/yolov5/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(n_camera_obj src/yolov5/calibrator.cpp src/yolov5/yolov5-p6.cpp)


target_link_libraries(n_camera_obj nvinfer)
target_link_libraries(n_camera_obj cudart)
target_link_libraries(n_camera_obj myplugins)
target_link_libraries(n_camera_obj ${OpenCV_LIBS})
target_link_libraries(n_camera_obj ${catkin_LIBRARIES})







