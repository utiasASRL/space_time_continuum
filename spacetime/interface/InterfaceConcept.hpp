/**
 * @file InterfaceConcept.hpp
 * @brief What an interface must provide, so downstream code can swap interfaces by changing
 *        only the template argument of Spacetime::Session (see spacetime/interface/Interfaces.hpp):
 *        @code
 *        #include <spacetime>
 *
 *        Spacetime::SessionOptions options;
 *        Spacetime::Session session(options); // the default interface; or e.g. Spacetime::Session<Spacetime::INTERFACE_ROS2>
 *        session.start(estimator);
 *        session.join();
 *        @endcode
 *
 *        Each interface, numbered by one of the INTERFACE_* constants, specializes
 *        `Interface::Traits<INTERFACE_X>` (see e.g. spacetime/interface/standalone/Interface.hpp)
 *        with a `name` and a `Session` type. A Session runs the estimator on the interface's own terms: it owns the clock,
 *        the measurement sources and however many threads (or none, e.g. an executor it is
 *        handed) the interface needs. Its hooks are:
 *        - `Session(const SessionOptions &)`: sets up the clock and whatever else the interface
 *          needs before running. Holds no estimator, so it can be built (and e.g. a
 *          visualizer given its now() and requestStop()) before the estimator is complete.
 *        - `void setTickCallback(Interface::EstimatorDriver::TickCallback)`: called with the
 *          time of each estimate, right after it, on the estimating thread. Call before start().
 *        - `void start(Estimator &)`: starts estimating with the estimator, which must outlive
 *          the session, from the start time, filling its measurement queues
 *          (Estimator::getMeasurementQueues(), read here, so every model added by then is
 *          covered) however the interface gets the data (replaying what is already loaded into
 *          them, subscribing to a topic per queue, ...). Throws std::invalid_argument for invalid
 *          options (e.g. a non-positive period, or a batch-mode estimator in real time), and
 *          std::logic_error if already started.
 *        - `void requestStop()`: asks the run to end as soon as possible. Safe to call from
 *          any thread, at any time (e.g. when a visualizer is closed).
 *        - `void join()`: blocks until the run has ended (reached the end time, or stopped),
 *          and rethrows the first error raised while running.
 *        - `bool finished() const`: true once the run has ended.
 *        - `double startTime() const`: the time of the first estimate (SessionOptions::start_time).
 *        - `double endTime() const`: the end time actually used (see SessionOptions::end_time). Valid after start().
 *        - `double now() const`: the current estimation time, e.g. a visualizer's live time
 *          source. Safe to call from any thread.
 *        - `bool isRealTime() const`: whether estimation follows a real-time clock.
 *
 *        Interface::Session (Interfaces.hpp) adds, for every interface, `start(Estimator &,
 *        Viewer &)`, `joinEstimation()` and a `join()` that also runs that viewer (see
 *        ViewerSession.hpp), built on the hooks above.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_INTERFACECONCEPT_HPP
#define SPACETIME_INTERFACE_INTERFACECONCEPT_HPP

#include <concepts>
#include <string_view>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/EstimatorDriver.hpp>

namespace Spacetime::Interface
{
    inline constexpr int INTERFACE_STANDALONE = 0; ///< The standalone ("default") interface: file playback (see standalone/Interface.hpp).
    inline constexpr int INTERFACE_ROS2 = 1;       ///< The ROS2 interface (see ros2/Interface.hpp).
    inline constexpr int INTERFACE_MUJOCO = 2;     ///< The MuJoCo interface (not implemented yet).

    /**
     * @brief An interface's bundle: a `name` and a `Session` satisfying InterfaceType. Each built
     *        interface specializes it for its INTERFACE_* constant; it is left undefined for the others.
     * @tparam I One of the INTERFACE_* constants.
     */
    template <int I>
    struct Traits;

    /// What every interface's Session is constructed with.
    struct SessionOptions
    {
        double start_time = 0.0;    ///< Time of the first estimate.
        double end_time = -1.0;     ///< Time to stop estimating at. If negative: until the data ends (see resolveEndTime).
        double period = 1.0 / 30.0; ///< Time between estimates, and so between nodes.
        bool real_time = false;     ///< Follow a real-time clock, rather than estimate as fast as possible.
    };

    /**
     * @brief An interface bundle: a `name` and a `Session` with the hooks listed in the file comment.
     * @tparam I The bundle type, e.g. Interface::Traits<INTERFACE_STANDALONE>.
     */
    template <typename I>
    concept InterfaceType =
        requires {
            typename I::Session;
            { I::name } -> std::convertible_to<std::string_view>;
        } &&
        std::constructible_from<typename I::Session, const SessionOptions &> &&
        requires(typename I::Session &session, const typename I::Session &const_session, Estimator &estimator,
                 EstimatorDriver::TickCallback callback) {
            { session.setTickCallback(callback) } -> std::same_as<void>;
            { session.start(estimator) } -> std::same_as<void>;
            { session.requestStop() } -> std::same_as<void>;
            { session.join() } -> std::same_as<void>;
            { const_session.finished() } -> std::convertible_to<bool>;
            { const_session.startTime() } -> std::convertible_to<double>;
            { const_session.endTime() } -> std::convertible_to<double>;
            { const_session.now() } -> std::convertible_to<double>;
            { const_session.isRealTime() } -> std::convertible_to<bool>;
        };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_INTERFACECONCEPT_HPP
