message("Building estimator core..")
file(GLOB_RECURSE CPP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/estimator/*.cpp
)
file(GLOB_RECURSE H_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/estimator/*.hpp
)
add_library(estimator
    ${CPP_FILES}
    ${H_FILES}
)
# The second path finds the package-level <spacetime> header (spacetime/spacetime): it cannot
# sit next to the spacetime/ directory, and compilers skip that directory when looking it up.
target_include_directories(estimator PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/spacetime>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/spacetime>
    PRIVATE
    ${gtsam_INCLUDE_DIRS}
)
# The estimator's headers (ConfigLoader included) use GTSAM, Eigen and jsoncpp directly.
target_link_libraries(estimator
    PUBLIC
    gtsam
    Eigen3::Eigen
    JsonCpp::JsonCpp
)
target_compile_definitions(estimator PUBLIC SPACETIME_ENABLE_DEBUG_LOG=${SPACETIME_DEBUG_LOG})
add_library(spacetime::estimator ALIAS estimator)

# The interface layer (header-only, the files directly in spacetime/interface/): the
# clock, dataloader, estimation schedule/driver and result-query abstractions every
# interface (one subfolder each, spacetime/interface/<name>/) builds on. Depends on the
# core only, and is thread-free: each interface decides how threads are handled (see
# tests/interface/InterfaceIsThreadFreeTests.cpp).
message("Building interface layer..")
add_library(interface INTERFACE)
target_include_directories(interface INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(interface INTERFACE estimator)
add_library(spacetime::interface ALIAS interface)

if(InterfaceStandalone)
    message("Building standalone interface..")
    # File playback on the interface layer, with its own threads (estimator, dataloaders).
    file(GLOB_RECURSE STANDALONE_CPP_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/standalone/*.cpp
    )
    file(GLOB_RECURSE STANDALONE_H_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/standalone/*.hpp
    )
    add_library(standalone
        ${STANDALONE_CPP_FILES}
        ${STANDALONE_H_FILES}
    )
    target_include_directories(standalone PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(standalone PUBLIC interface Threads::Threads)
    target_compile_definitions(standalone PUBLIC SPACETIME_INTERFACE_STANDALONE=1)
    add_library(spacetime::standalone ALIAS standalone)
endif()

if(InterfaceROS2)
    message("Building ROS2 interface..")
    # Live (or rosbag) estimation on ROS2 nodes: an estimator node, one node per sensor and a
    # snapshot publisher for rviz2, all in one process. ROS owns the threads (executors).
    file(GLOB_RECURSE ROS2_CPP_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/ros2/*.cpp
    )
    file(GLOB_RECURSE ROS2_H_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/ros2/*.hpp
    )
    add_library(ros2
        ${ROS2_CPP_FILES}
        ${ROS2_H_FILES}
    )
    target_include_directories(ros2 PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(ros2 PUBLIC
        interface
        standalone # Standalone::lowerCurrentThreadPriority
        rclcpp::rclcpp
        tf2_ros::tf2_ros
        ${tf2_msgs_TARGETS}
        ${geometry_msgs_TARGETS}
        ${sensor_msgs_TARGETS}
    )
    target_compile_definitions(ros2 PUBLIC SPACETIME_INTERFACE_ROS2=1)
    add_library(spacetime::ros2 ALIAS ros2)

    message("Building rviz visualization..")
    # Publishes the estimator's snapshots as standard messages rviz2 draws (markers, poses,
    # TF). Independent of the Visualizer option: a ROS2 node, not a visualizer backend.
    file(GLOB RVIZ_CPP_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/rviz/*.cpp
    )
    file(GLOB RVIZ_H_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/rviz/*.hpp
    )
    add_library(rviz
        ${RVIZ_CPP_FILES}
        ${RVIZ_H_FILES}
    )
    target_include_directories(rviz PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(rviz PUBLIC
        ros2
        ${visualization_msgs_TARGETS}
    )
    # Lets <spacetime> include the rviz headers.
    target_compile_definitions(rviz PUBLIC SPACETIME_RVIZ=1)
    add_library(spacetime::rviz ALIAS rviz)
endif()

# Every built interface, for downstream code that selects one with Spacetime::Session's
# template argument (see <spacetime>, spacetime/Types.hpp and spacetime/interface/Interfaces.hpp).
add_library(interfaces INTERFACE)
target_link_libraries(interfaces INTERFACE interface)
if(InterfaceStandalone)
    target_link_libraries(interfaces INTERFACE standalone)
endif()
if(InterfaceROS2)
    target_link_libraries(interfaces INTERFACE ros2)
endif()
add_library(spacetime::interfaces ALIAS interfaces)

if(Visualizer)
    message("Building estimator visualizer (${VisualizerBackend} backend)..")
    # Backend-agnostic layer (spacetime/visualizer/*) plus the selected backend's folder.
    file(GLOB CPP_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/*.cpp
    )
    file(GLOB H_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/*.hpp
    )
    file(GLOB_RECURSE BACKEND_CPP_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/${VisualizerBackend}/*.cpp
    )
    # Include header files so MOC can process Q_OBJECT macros
    file(GLOB_RECURSE BACKEND_H_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/${VisualizerBackend}/*.hpp
    )
    add_library(visualizer
        ${CPP_FILES}
        ${H_FILES}
        ${BACKEND_CPP_FILES}
        ${BACKEND_H_FILES}
    )
    string(TOUPPER ${VisualizerBackend} VISUALIZER_BACKEND_UPPER)
    # Qt's bare `emit`/`signals`/`slots` macros collide with identically-named,
    # unrelated identifiers in oneTBB headers pulled in transitively by GTSAM
    # (e.g. oneapi::tbb::profiling_tag::emit()), so everything built against the
    # visualizer disables them; our own code uses Q_SLOTS instead.
    target_compile_definitions(visualizer PUBLIC
        VISUALIZER=1
        SPACETIME_VISUALIZER_BACKEND_${VISUALIZER_BACKEND_UPPER}=1
        QT_NO_KEYWORDS
    )

    target_include_directories(visualizer PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${gtsam_INCLUDE_DIRS}
    )

    target_link_libraries(visualizer
        PUBLIC
        ${VTK_LIBRARIES}
        Qt6::Core
        Qt6::Gui
        Qt6::Xml
        Qt6::Network
        Qt6::Widgets
        Qt6::OpenGLWidgets
        # The object views are header-only templates over the estimator/sensor types.
        estimator
        # Thread priority, for the visualizer's own thread (see default/VisualizerThread).
        standalone
        PRIVATE
        gtsam
        Eigen3::Eigen
    )
    add_library(spacetime::visualizer ALIAS visualizer)
endif()
