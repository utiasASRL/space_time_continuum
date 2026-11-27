# The `spacetime` Python package: the pure-Python sources in spacetime/ and the _spacetime
# extension, assembled in ${CMAKE_CURRENT_BINARY_DIR} and pip-installed by `python-install`.
#
# GTSAM types (Pose3, Values, factors, ...) are shared with GTSAM's own Python module, so the
# extension must be built with the pybind11 version GTSAM's wrapper was (see requirements.txt).

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT pybind11_DIR)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE pybind11_found)
    if(NOT pybind11_found EQUAL 0)
        message(FATAL_ERROR "pybind11 not found for ${Python_EXECUTABLE}: pip install -r python/requirements.txt")
    endif()
endif()
find_package(pybind11 3.0 REQUIRED CONFIG)
message("Python bindings: ${Python_EXECUTABLE}, pybind11 ${pybind11_VERSION}")

set(SPACETIME_PYTHON_PACKAGE ${CMAKE_CURRENT_BINARY_DIR}/spacetime)

pybind11_add_module(_spacetime
    src/module.cpp
    src/bind_types.cpp
    src/bind_priors.cpp
    src/bind_joints.cpp
    src/bind_sensors.cpp
    src/bind_actuators.cpp
    src/bind_estimator.cpp
    src/bind_interface.cpp
    src/bind_utilities.cpp
    $<$<BOOL:${Visualizer}>:src/bind_visualizer.cpp>
    $<$<BOOL:${InterfaceROS2}>:src/bind_ros2.cpp>
    $<$<BOOL:${InterfaceROS2}>:src/bind_rviz.cpp>
)
target_include_directories(_spacetime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(_spacetime PRIVATE estimator interfaces gtsam Eigen3::Eigen JsonCpp::JsonCpp)
if(Visualizer)
    target_link_libraries(_spacetime PRIVATE visualizer)
    target_compile_definitions(_spacetime PRIVATE SPACETIME_PYTHON_VISUALIZER=1)
endif()
if(InterfaceROS2)
    # spacetime.interface.ros2 and spacetime.visualizer.rviz (rclcpp is initialized from Python, see bind_ros2.cpp).
    target_link_libraries(_spacetime PRIVATE rviz)
    target_compile_definitions(_spacetime PRIVATE SPACETIME_PYTHON_ROS2=1)
endif()
target_compile_options(_spacetime PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>)
# $<1:...> stops multi-config generators (Visual Studio, Xcode) from adding a per-config subfolder.
set_target_properties(_spacetime PROPERTIES LIBRARY_OUTPUT_DIRECTORY $<1:${SPACETIME_PYTHON_PACKAGE}>)

# Copy the pure-Python sources next to the extension (re-copied whenever one changes).
file(GLOB_RECURSE SPACETIME_PYTHON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/spacetime/*.py)
set(SPACETIME_PYTHON_COPIES)
foreach(source ${SPACETIME_PYTHON_SOURCES})
    file(RELATIVE_PATH relative ${CMAKE_CURRENT_SOURCE_DIR}/spacetime ${source})
    set(copy ${SPACETIME_PYTHON_PACKAGE}/${relative})
    add_custom_command(OUTPUT ${copy} COMMAND ${CMAKE_COMMAND} -E copy ${source} ${copy} DEPENDS ${source})
    list(APPEND SPACETIME_PYTHON_COPIES ${copy})
endforeach()
add_custom_target(spacetime_python_sources ALL DEPENDS ${SPACETIME_PYTHON_COPIES})
add_dependencies(_spacetime spacetime_python_sources)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_target(python-install
    COMMAND ${Python_EXECUTABLE} -m pip install --no-deps --no-build-isolation ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS _spacetime
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Installing the spacetime Python package into ${Python_EXECUTABLE}'s environment")

# Wheel builds (pyproject.toml, scikit-build-core installs only COMPONENT python): the extension
# and the pure-Python sources, as the `spacetime` package at the wheel's root.
install(TARGETS _spacetime LIBRARY DESTINATION spacetime COMPONENT python)
install(DIRECTORY spacetime/ DESTINATION spacetime COMPONENT python
        FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
# Prebuilt wheels also ship GTSAM's Python module, built from the same libgtsam as _spacetime
# (see ci/build_deps.py): SPACETIME_BUNDLE_GTSAM_PY is the gtsam package directory of
# that GTSAM build (<gtsam build>/python/gtsam). auditwheel then bundles one libgtsam for both.
set(SPACETIME_BUNDLE_GTSAM_PY "" CACHE PATH "gtsam Python package directory to bundle into the wheel (optional)")
if(SPACETIME_BUNDLE_GTSAM_PY)
    if(NOT EXISTS ${SPACETIME_BUNDLE_GTSAM_PY}/__init__.py)
        message(FATAL_ERROR "SPACETIME_BUNDLE_GTSAM_PY=${SPACETIME_BUNDLE_GTSAM_PY} is not a gtsam package directory")
    endif()
    message("Bundling GTSAM's Python module from ${SPACETIME_BUNDLE_GTSAM_PY}")
    install(DIRECTORY ${SPACETIME_BUNDLE_GTSAM_PY}/ DESTINATION gtsam COMPONENT python
            PATTERN "__pycache__" EXCLUDE PATTERN "tests" EXCLUDE)
endif()
