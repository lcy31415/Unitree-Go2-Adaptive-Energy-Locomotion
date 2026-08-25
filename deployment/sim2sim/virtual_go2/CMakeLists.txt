cmake_minimum_required(VERSION 3.12)

project(virtual_go2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# unitree_sdk2 is typically installed under /opt/unitree_robotics
if(NOT EXISTS "${CMAKE_PREFIX_PATH}/lib/cmake/unitree_sdk2" AND EXISTS "/opt/unitree_robotics/lib/cmake/unitree_sdk2")
    list(APPEND CMAKE_PREFIX_PATH /opt/unitree_robotics)
endif()

find_package(unitree_sdk2 REQUIRED)

add_executable(virtual_go2 virtual_go2.cpp)

target_link_libraries(virtual_go2 PRIVATE
    unitree_sdk2
    pthread
)
