#ifndef SPACETIME_VISUALIZER_HPP
#define SPACETIME_VISUALIZER_HPP

/**
 * @file Visualizer.hpp
 * @brief Entry point to the visualizer backend selected at compile time (CMake
 *        option VisualizerBackend). Includes that backend's Visualizer and object
 *        views, and exposes them as Visualization::Backend, so callers can be
 *        written without naming the backend. See BackendConcept.hpp for what each
 *        backend provides.
 */

#include <spacetime/visualizer/BackendConcept.hpp>
#include <spacetime/visualizer/VisualizerOptions.hpp>

#if defined(SPACETIME_VISUALIZER_BACKEND_DEFAULT)
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/objects/SensorViews.hpp>

namespace Spacetime::Visualization
{
    namespace Backend = Default;
}
#else
#error "No visualizer backend selected: set the CMake option VisualizerBackend."
#endif

namespace Spacetime::Visualization
{
    /// The selected backend's visualizer (see Visualization::Backend).
    using Visualizer = Backend::Visualizer;
    static_assert(VisualizerBackend<Visualizer>, "The selected visualizer backend does not satisfy Visualization::VisualizerBackend.");
}

#endif // SPACETIME_VISUALIZER_HPP
