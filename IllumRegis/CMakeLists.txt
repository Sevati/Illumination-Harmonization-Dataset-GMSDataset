cmake_minimum_required(VERSION 3.14)
project(IllumRegis)

set(CMAKE_CXX_STANDARD 14)

add_executable(IllumRegis main.cpp opticalflow.cpp impl.cpp)

## where to find opencv
set(OpenCV_DIR "E:/opencv-3.4.1/build/install/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#set(OpenCV_LIBS "D:/OpenCV341-MinGW-Build-x64/install/x64/mingw/lib/")
target_link_libraries(IllumRegis ${OpenCV_LIBS} )