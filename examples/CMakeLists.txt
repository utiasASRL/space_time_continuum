### Example programs: one executable per demo source (each has its own main()).
if(Visualizer)
    # The demos are written against the default (Qt/VTK) backend.
    if(NOT VisualizerBackend STREQUAL "default")
        message(FATAL_ERROR "The demos require VisualizerBackend=default (got '${VisualizerBackend}').")
    endif()
endif()

function(add_demo target source)
    add_executable(${target} ${source})
    # build/examples/<demo> on every generator: $<1:...> stops multi-config generators (Visual
    # Studio, Xcode) from adding a per-config subfolder (python/tests/test_demo_regression.py looks here).
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_BINARY_DIR}>)

    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${gtsam_INCLUDE_DIRS}
    )

    target_link_libraries(${target}
        PRIVATE
        estimator
        interfaces
        Eigen3::Eigen
        JsonCpp::JsonCpp
        gtsam
    )
    if(Visualizer)
        message(STATUS "Linking visualizer to ${target}")
        target_link_libraries(${target} PUBLIC visualizer)
    endif()
endfunction()

add_demo(robot_joining robot_joining.cpp)
add_demo(stewart_gough stewart_gough.cpp)
add_demo(robot_joining_parallel robot_joining_parallel.cpp)
add_demo(bucky_ball bucky_ball.cpp)
