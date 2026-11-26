cmake_minimum_required(VERSION 3.16)

project(spacetime VERSION 1.0.0-alpha LANGUAGES CXX)

#Important CMAKE stuff
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 20)

# Add GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

### Build Options
option(BuildTests "BuildTests" ON)
option(Profile "Profile" OFF)
option(Debug "Debug" OFF)
option(Visualizer "Visualizer" ON)
# Which visualizer backend (spacetime/visualizer/<backend>/) to build when Visualizer is ON.
set(VisualizerBackend "default" CACHE STRING "Visualizer backend to build (default)")
set_property(CACHE VisualizerBackend PROPERTY STRINGS default)
# -march=native changes Eigen's alignment, so it must match how GTSAM was built.
option(MarchNative "Compile Release builds with -march=native" ON)
# Interfaces built on the interface layer (spacetime/interface/), one subfolder each
# (e.g. spacetime/interface/standalone/). Standalone replays recorded data; ROS2 runs live
# (or rosbag) estimation on ROS2 nodes (source a ROS2 distribution first); Mujoco is not
# implemented yet.
option(InterfaceStandalone "Build the standalone interface (file playback)" ON)
option(InterfaceMujoco "Build the Mujoco interface" OFF)
option(InterfaceROS2 "Build the ROS2 interface" OFF)
# Python bindings (python/): the `spacetime` package, built against GTSAM's Python module.
option(Python "Build the Python bindings" OFF)
# The demos in examples/ (need InterfaceStandalone). Off when building Python wheels.
option(BuildExamples "Build the demos in examples/" ON)

message("BuildTests: ${BuildTests}")
message("Profile: ${Profile}")
message("Debug: ${Debug}")
message("Visualizer: ${Visualizer}")
message("VisualizerBackend: ${VisualizerBackend}")
message("MarchNative: ${MarchNative}")
message("InterfaceStandalone: ${InterfaceStandalone}")
message("InterfaceMujoco: ${InterfaceMujoco}")
message("InterfaceROS2: ${InterfaceROS2}")
message("BuildExamples: ${BuildExamples}")
message("Python: ${Python}")
if(InterfaceMujoco)
    message(FATAL_ERROR "The Mujoco interface (InterfaceMujoco) is not implemented yet.")
endif()
if(InterfaceROS2 AND NOT InterfaceStandalone)
    message(FATAL_ERROR "The ROS2 interface lowers its non-estimator threads' priority as the standalone interface does: InterfaceROS2=ON requires InterfaceStandalone=ON.")
endif()
if(Visualizer AND NOT InterfaceStandalone)
    message(FATAL_ERROR "The visualizer runs on the standalone interface's threads: Visualizer=ON requires InterfaceStandalone=ON.")
endif()
if(Python AND NOT InterfaceStandalone)
    message(FATAL_ERROR "The Python bindings run problems through the standalone interface: Python=ON requires InterfaceStandalone=ON.")
endif()
if(Python)
    # The static libraries are linked into the Python extension, a shared object.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(Debug)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Og -g3")
    set(SPACETIME_DEBUG_LOG 1) # see spacetime/estimator/utilities/logger.hpp; exported with the estimator target
else()
    set(CMAKE_BUILD_TYPE Release)
    if(MarchNative)
        set(CMAKE_CXX_FLAGS "-march=native -O3 -g0")
    else()
        set(CMAKE_CXX_FLAGS "-O3 -g0")
    endif()
    set(SPACETIME_DEBUG_LOG 0)
endif()

if(Profile)
    string(APPEND CMAKE_CXX_FLAGS " -pg")
endif()

# Core project setup 
find_package(Eigen3 3.4 REQUIRED NO_MODULE) # Eigen
find_package(jsoncpp REQUIRED) # Json
if(Python AND TARGET jsoncpp_lib)
    # JsonCpp::JsonCpp prefers jsoncpp's static library, which is often built without -fPIC and
    # then cannot be linked into the Python extension: use its shared library instead.
    set_target_properties(JsonCpp::JsonCpp PROPERTIES INTERFACE_LINK_LIBRARIES jsoncpp_lib)
endif()
find_package(GTSAM REQUIRED) # GTSAM
if(InterfaceStandalone)
    find_package(Threads REQUIRED) # the standalone interface's estimator/dataloader threads
endif()
if(InterfaceROS2)
    # The ROS2 interface's nodes (spacetime/interface/ros2/): needs a sourced ROS2 distribution.
    find_package(rclcpp REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(tf2_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(visualization_msgs REQUIRED)
endif()
if(Visualizer AND NOT VisualizerBackend STREQUAL "default")
    message(FATAL_ERROR "Unknown VisualizerBackend '${VisualizerBackend}'. Available backends: default")
endif()
if(Visualizer)
    # Found through CMake's usual search: a system install is found as is; for one elsewhere
    # (e.g. a Qt online-installer prefix such as ~/Qt/6.8.0/gcc_64), pass its prefix with
    # -DCMAKE_PREFIX_PATH=<prefix> (or set Qt6_DIR / VTK_DIR to the folders holding
    # Qt6Config.cmake / vtk-config.cmake).
    find_package(VTK 9.3 REQUIRED) # VTK
    find_package(Qt6 REQUIRED COMPONENTS Core Gui Xml Network Widgets OpenGLWidgets) # Qt

    qt_standard_project_setup()
    # QT_NO_KEYWORDS is set on the visualizer target (see spacetime/CMakeLists.txt), so
    # that it reaches everything built against it, here or through the installed package.
endif()

add_subdirectory(spacetime)
if(InterfaceStandalone AND BuildExamples)
    add_subdirectory(examples) # the demo plays recorded data through the standalone interface
endif()

# Build tests if enabled
if(BuildTests)
    add_subdirectory(tests)
endif()

if(Python)
    add_subdirectory(python)
endif()


### Install / export: find_package(spacetime) with the targets spacetime::estimator,
### spacetime::interface, spacetime::interfaces (every built interface) and (if built)
### spacetime::standalone, spacetime::ros2 (with spacetime::rviz) and spacetime::visualizer.
set(SPACETIME_EXPORTED_TARGETS estimator interface interfaces)
if(InterfaceStandalone)
    list(APPEND SPACETIME_EXPORTED_TARGETS standalone)
endif()
if(InterfaceROS2)
    list(APPEND SPACETIME_EXPORTED_TARGETS ros2 rviz)
endif()
if(Visualizer)
    list(APPEND SPACETIME_EXPORTED_TARGETS visualizer)
endif()
install(TARGETS ${SPACETIME_EXPORTED_TARGETS}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
# Headers live next to their sources, under spacetime/, and are included as <spacetime/...>.
# Installs every .hpp file under SOURCE_DIR, and every extensionless aggregation header (one per
# folder, named after it, e.g. spacetime/estimator/estimator), keeping the folder layout.
function(install_headers SOURCE_DIR DEST_PREFIX)
    install(DIRECTORY ${SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${DEST_PREFIX}
        FILES_MATCHING PATTERN "*.hpp"
    )
    file(GLOB_RECURSE ALL_FILES "${SOURCE_DIR}/*")
    foreach(FILE ${ALL_FILES})
        get_filename_component(EXT ${FILE} EXT)
        if(NOT EXT STREQUAL "")
            continue()
        endif()
        file(RELATIVE_PATH REL_PATH ${SOURCE_DIR} ${FILE})
        get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
        # The top-level aggregation header (<spacetime>) cannot go one level up, where the
        # spacetime/ header directory is: it stays in that directory, which is on the include
        # path of every target (see spacetime/CMakeLists.txt).
        install(FILES ${FILE} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${DEST_PREFIX}/${REL_DIR})
    endforeach()
endfunction()
install_headers(${CMAKE_CURRENT_SOURCE_DIR}/spacetime spacetime)
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)


find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen-awesome.css)
    # doxygen-awesome recommends these alongside the stylesheet
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_DISABLE_INDEX NO)
    set(DOXYGEN_FULL_SIDEBAR NO)
    set(DOXYGEN_HTML_COLORSTYLE LIGHT)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    set(DOXYGEN_IMAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/docs/images)

    doxygen_add_docs(docs
        ${CMAKE_CURRENT_SOURCE_DIR}/spacetime
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        COMMENT "Generating API documentation with Doxygen"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    # README uses raw <img> tags, which doxygen does not relocate; mirror the assets
    add_custom_command(TARGET docs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/images
            ${CMAKE_CURRENT_BINARY_DIR}/html/docs/images
    )
endif()
