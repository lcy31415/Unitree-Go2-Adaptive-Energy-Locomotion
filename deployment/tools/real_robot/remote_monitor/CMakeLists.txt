cmake_minimum_required(VERSION 3.16)

project(go2_remote_monitor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(unitree_sdk2 REQUIRED)

add_executable(go2_remote_monitor
    remote_monitor.cpp
)

target_link_libraries(go2_remote_monitor
    unitree_sdk2
)
