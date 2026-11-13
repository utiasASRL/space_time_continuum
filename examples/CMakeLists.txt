### Example programs: one executable per demo source (each has its own main()).
if(Visualizer)
    # The demos are written against the default (Qt/VTK) backend.
    if(NOT VisualizerBackend STREQUAL "default")
        message(FATAL_ERROR "The demos require VisualizerBackend=default (got '${VisualizerBackend}').")
    endif()
endif()

function(add_demo target source)
    add_executable(${target} ${source})

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
