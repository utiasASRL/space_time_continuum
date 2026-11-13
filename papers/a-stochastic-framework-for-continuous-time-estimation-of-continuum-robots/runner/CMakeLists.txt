cmake_minimum_required(VERSION 3.16)

project(tro_revisions_final VERSION 1.0.0 LANGUAGES CXX)

# The TRO results runner, built on the installed spacetime package.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(spacetime 1.0 REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native changes Eigen's alignment, so it must match how spacetime (and GTSAM) were built.
if(spacetime_MARCH_NATIVE AND CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-march=native)
endif()

add_executable(tro_runner main.cpp)
target_link_libraries(tro_runner PRIVATE spacetime::estimator)
