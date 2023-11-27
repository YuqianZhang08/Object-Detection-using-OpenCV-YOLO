cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE "")

project(yolov5 VERSION 0.1.0)

SET(OpenCV_DIR "C:/OpenCV/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)


set(SOURCE_FILES
    yolov5class.cpp
    yolov5class.h
    batchprocess.cpp
    batchprocess.h
)

#add_library(YOLO SHARED ${SOURCE_FILES})

add_executable(main ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})

#target_link_libraries(YOLO ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})


