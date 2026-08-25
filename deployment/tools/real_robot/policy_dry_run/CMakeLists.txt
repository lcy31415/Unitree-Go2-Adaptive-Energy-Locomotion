cmake_minimum_required(VERSION 3.12)

project(go2_policy_dry_run)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# unitree_sdk2 is typically installed under /opt/unitree_robotics
if(NOT EXISTS "${CMAKE_PREFIX_PATH}/lib/cmake/unitree_sdk2" AND EXISTS "/opt/unitree_robotics/lib/cmake/unitree_sdk2")
    list(APPEND CMAKE_PREFIX_PATH /opt/unitree_robotics)
endif()

find_package(unitree_sdk2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)

# tools/real_robot/policy_dry_run -> repository root
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(DEPLOY_DIR "${REPO_ROOT}/deploy")
set(ONNXRUNTIME_DIR "${DEPLOY_DIR}/thirdparty/onnxruntime-linux-x64-1.22.0")

add_executable(go2_policy_dry_run policy_dry_run.cpp)

target_include_directories(go2_policy_dry_run PRIVATE
    ${DEPLOY_DIR}/include
    ${ONNXRUNTIME_DIR}/include
    /usr/include/eigen3
)

target_link_libraries(go2_policy_dry_run PRIVATE
    unitree_sdk2
    Boost::program_options
    yaml-cpp
    spdlog::spdlog
    pthread
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)

# Offline pre-flight: same deploy stack against mock joint data (no DDS)
add_executable(go2_policy_offline_check offline_policy_check.cpp)

target_include_directories(go2_policy_offline_check PRIVATE
    ${DEPLOY_DIR}/include
    ${ONNXRUNTIME_DIR}/include
    /usr/include/eigen3
)

target_link_libraries(go2_policy_offline_check PRIVATE
    unitree_sdk2
    yaml-cpp
    spdlog::spdlog
    pthread
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
)
