cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR /opt/homebrew/opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

file(GLOB SOURCES "*.cpp")
add_executable(Project ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Project ${OpenCV_LIBS})
target_compile_options(Project PRIVATE)