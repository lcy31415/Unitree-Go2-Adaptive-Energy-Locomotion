cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(unitree_sdk2 REQUIRED)

set(ONNXRUNTIME_DIR
    ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
    src/*.cpp
)

add_library(${PROJECT_NAME}_lib
    ${ADD_SRC_LIST}
)

target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
        /usr/include/eigen3
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/../../include
        ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(${PROJECT_NAME}_lib
    PUBLIC
        unitree_sdk2
        Boost::program_options
        yaml-cpp
        fmt
        pthread
        rt
        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

add_executable(go2_ctrl
    main.cpp
)

target_link_libraries(go2_ctrl
    PRIVATE
        ${PROJECT_NAME}_lib
)
