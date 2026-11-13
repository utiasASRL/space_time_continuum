#ifndef VISUALIZERBACKENDCONCEPT_HPP
#define VISUALIZERBACKENDCONCEPT_HPP

#include <concepts>
#include <functional>

#include <spacetime/visualizer/PlaybackController.hpp>

namespace Spacetime::Visualization
{
    /**
     * @brief The contract every visualizer backend (spacetime/visualizer/<backend>/)
     *        provides, so code driving a visualizer (e.g. examples/offline_demos.cpp)
     *        is written against the selected backend through Visualization::Backend
     *        (see spacetime/visualizer/Visualizer.hpp) without naming it.
     *
     *        Beyond this concept, a backend provides, in its own namespace:
     *        - a `Visualizer` class satisfying VisualizerBackend, constructible from
     *          (VisualizerOptions, std::string output_path), with an
     *          `addVisualizable(std::shared_ptr<T>)` taking its own object views;
     *        - one view per drawable state, named after the object it draws and
     *          defined in <backend>/objects/<Object>.hpp (e.g.
     *          <backend>::ContinuumRobot<NodeT> viewing a Types::ContinuumRobotState<NodeT>,
     *          bound to the prior extending it for interpolation), each deriving from
     *          Visualization::Visualizable;
     *        - `makeSensorView<ValuesT>(std::shared_ptr<Types::SensorBase>)`, choosing a
     *          sensor's view from its dynamic type. Sensor views are added on their own,
     *          not through the view of the state their sensor measures.
     */
    template <typename T>
    concept VisualizerBackend = requires(T visualizer, const T const_visualizer, double time, std::function<double()> now_fn) {
        { visualizer.setTimeRange(time, time) } -> std::same_as<void>;
        { visualizer.setLiveTimeSource(now_fn) } -> std::same_as<void>;
        { const_visualizer.isOnline() } -> std::same_as<bool>;
        { visualizer.playback() } -> std::same_as<PlaybackController &>;
    };
} // namespace Spacetime::Visualization

#endif // VISUALIZERBACKENDCONCEPT_HPP
