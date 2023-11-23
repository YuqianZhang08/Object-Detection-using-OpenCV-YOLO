cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE "")

project(yolov5 VERSION 0.1.0)

SET(OpenCV_DIR "E:/opencv/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)

add_library(yourlib SHARED yolov5.cpp)
#add_executable(main yolov5class.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(yourlib ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})