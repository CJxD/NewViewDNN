cmake_minimum_required(VERSION 3.1.3)
project(NewViewDNN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/modules/")

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

add_subdirectory(src)
