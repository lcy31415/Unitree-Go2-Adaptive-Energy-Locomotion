cmake_minimum_required(VERSION 3.10)

project(dds_read_go2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(unitree_sdk2 REQUIRED)

add_executable(dds_read_go2 dds_read_go2.cpp)

target_link_libraries(dds_read_go2
    unitree_sdk2
)
