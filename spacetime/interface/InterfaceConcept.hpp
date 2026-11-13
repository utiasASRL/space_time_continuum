/**
 * @file InterfaceConcept.hpp
 * @brief What an interface must provide, so downstream code can swap interfaces by changing
 *        only which one it selects:
 *        @code
 *        #include <spacetime/interface/Interfaces.hpp>
 *        using SelectedInterface = Spacetime::Interfaces::Standalone; // the one line to change
 *
 *        SelectedInterface::Session session(estimator, options);
 *        session.addMeasurementQueues(estimator.getMeasurementQueues());
 *        session.start();
 *        session.join();
 *        @endcode
 *
 *        Each interface declares a bundle `Spacetime::Interfaces::<Name>` (see e.g.
 *        spacetime/interface/standalone/Interface.hpp) holding a `name` and a `Session`
 *        type. A Session runs the estimator on the interface's own terms: it owns the clock,
 *        the measurement sources and however many threads (or none, e.g. an executor it is
 *        handed) the interface needs. Its hooks are:
 *        - `Session(Estimator &, const SessionOptions &)`: drives the estimator, which must
 *          outlive the session.
 *        - `void addMeasurementQueues(queues)`: the models' measurement queues (typically
 *          Estimator::getMeasurementQueues()) the interface fills, however it gets the data
 *          (replaying what is already loaded into them, subscribing to a topic per queue,
 *          ...). Call before start(), and before any measurement observer (e.g. a visualizer
 *          view) subscribes to them.
 *        - `void setTickCallback(Interface::EstimatorDriver::TickCallback)`: called with the
 *          time of each estimate, right after it, on the estimating thread. Call before start().
 *        - `void start()`: starts estimating from the start time. Throws std::invalid_argument
 *          for invalid options (e.g. a non-positive period, or a batch-mode estimator in real
 *          time).
 *        - `void requestStop()`: asks the run to end as soon as possible. Safe to call from
 *          any thread, at any time (e.g. when a visualizer is closed).
 *        - `void join()`: blocks until the run has ended (reached the end time, or stopped),
 *          and rethrows the first error raised while running.
 *        - `bool finished() const`: true once the run has ended.
 *        - `double endTime() const`: the end time actually used (see SessionOptions::end_time). Valid after start().
 *        - `double now() const`: the current estimation time, e.g. a visualizer's live time
 *          source. Safe to call from any thread.
 *        - `bool isRealTime() const`: whether estimation follows a real-time clock.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_INTERFACECONCEPT_HPP
#define SPACETIME_INTERFACE_INTERFACECONCEPT_HPP

#include <concepts>
#include <string_view>
#include <vector>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/EstimatorDriver.hpp>

namespace Spacetime::Interface
{
    /// What every interface's Session is constructed with.
    struct SessionOptions
    {
        double start_time = 0.0;    ///< Time of the first estimate.
        double end_time = -1.0;     ///< Time to stop estimating at. If negative: until the data ends (see resolveEndTime).
        double period = 1.0 / 30.0; ///< Time between estimates, and so between nodes.
        bool real_time = false;     ///< Follow a real-time clock, rather than estimate as fast as possible.
    };

    /// An interface bundle: a `name` and a `Session` with the hooks listed in the file comment.
    template <typename I>
    concept InterfaceType =
        requires {
            typename I::Session;
            { I::name } -> std::convertible_to<std::string_view>;
        } &&
        std::constructible_from<typename I::Session, Estimator &, const SessionOptions &> &&
        requires(typename I::Session &session, const typename I::Session &const_session,
                 const std::vector<Types::pMeasurementQueueType> &queues, EstimatorDriver::TickCallback callback) {
            { session.addMeasurementQueues(queues) } -> std::same_as<void>;
            { session.setTickCallback(callback) } -> std::same_as<void>;
            { session.start() } -> std::same_as<void>;
            { session.requestStop() } -> std::same_as<void>;
            { session.join() } -> std::same_as<void>;
            { const_session.finished() } -> std::convertible_to<bool>;
            { const_session.endTime() } -> std::convertible_to<double>;
            { const_session.now() } -> std::convertible_to<double>;
            { const_session.isRealTime() } -> std::convertible_to<bool>;
        };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_INTERFACECONCEPT_HPP
