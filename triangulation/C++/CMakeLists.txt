cmake_minimum_required(VERSION 3.0.0)
project(triangulation VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(triangulation main.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})