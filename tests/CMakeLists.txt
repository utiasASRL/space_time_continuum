message(STATUS "Building tests")
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/e88cb95b92acbdce9b058dd894a68e1281b38495.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# googletest is only for the tests here: keep it out of spacetime's install
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Estimator core tests
file(GLOB_RECURSE CPP_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)
# Include header files
file(GLOB_RECURSE H_TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)
# The visualizer-thread tests (tests/visualizer_thread/) are their own executable, below
list(FILTER CPP_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/visualizer_thread/")
# The standalone interface's tests (tests/standalone/), and those running on it
if (NOT InterfaceStandalone)
    list(FILTER CPP_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/standalone/")
    list(FILTER CPP_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/BatchEstimatorTests.cpp")
endif()
# The ROS2 interface's and rviz visualization's tests (tests/ros2/, tests/visualizer/rviz/) are
# their own executable, below: they share one rclcpp context
list(FILTER CPP_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/ros2/")
list(FILTER H_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/ros2/")
list(FILTER CPP_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/visualizer/rviz/")
list(FILTER H_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/visualizer/rviz/")
# Visualizer tests (tests/visualizer/) are only built along with the visualizer
if (NOT Visualizer)
    list(FILTER CPP_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/visualizer/")
    list(FILTER H_TEST_FILES EXCLUDE REGEX "${CMAKE_CURRENT_SOURCE_DIR}/visualizer/")
endif()

add_executable(tests ${CPP_TEST_FILES} ${H_TEST_FILES})
target_link_libraries(tests
    PRIVATE
    estimator
    interface
    gtsam
    Eigen3::Eigen
    JsonCpp::JsonCpp
    GTest::gtest_main
)
# Lets tests read checked-in files (e.g. InterfaceIsThreadFreeTests scans spacetime/interface/).
target_compile_definitions(tests PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
if (InterfaceStandalone)
    target_link_libraries(tests PRIVATE standalone)
endif()
if (Visualizer)
    target_link_libraries(tests
        PRIVATE
        visualizer
    )
endif()
gtest_discover_tests(tests)

# A process holds one QApplication at a time, and the widget tests above keep one alive
# on the main thread, whereas these create theirs on the visualizer's own thread.
if (Visualizer)
    add_executable(visualizer_thread_tests ${CMAKE_CURRENT_SOURCE_DIR}/visualizer_thread/VisualizerThreadTests.cpp)
    target_link_libraries(visualizer_thread_tests PRIVATE visualizer GTest::gtest_main)
    gtest_discover_tests(visualizer_thread_tests)
endif()
# The ROS2 interface's and rviz visualization's tests, run by their own main (tests/ros2/main.cpp)
# holding the rclcpp context.
if (InterfaceROS2)
    find_package(rosgraph_msgs REQUIRED)
    file(GLOB ROS2_TEST_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/ros2/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ros2/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/rviz/*.cpp
    )
    add_executable(ros2_tests ${ROS2_TEST_FILES})
    target_link_libraries(ros2_tests PRIVATE ros2 rviz ${rosgraph_msgs_TARGETS} GTest::gtest)
    gtest_discover_tests(ros2_tests)
endif()
