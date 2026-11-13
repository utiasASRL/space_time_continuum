/**
 * @file EstimatorNode.hpp
 * @brief The ROS2 node stepping the estimator on a timer of the ROS clock.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_ESTIMATORNODE_HPP
#define SPACETIME_ROS2_ESTIMATORNODE_HPP

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/EstimatorDriver.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>

namespace Spacetime::Ros2
{
    /**
     * @brief Estimates every `period` of its ROS clock (so it follows use_sim_time, and
     *        pauses with a paused rosbag), at the clock's current time (see RosClock),
     *        through an Interface::EstimatorDriver. The clock starts on the first tick it
     *        reads non-zero. Ends on the first tick past the end time (as
     *        Interface::EstimationSchedule::isDone: the last estimate is at most at it), on
     *        the first error, or never if the end time is negative. Does nothing else on its
     *        thread, which the session runs at the highest priority.
     */
    class EstimatorNode : public rclcpp::Node
    {
    public:
        /// Called once, on the estimating thread, when the run ends: with the error that ended it, if any.
        using DoneCallback = std::function<void(std::exception_ptr)>;

        /// `estimator` must outlive the node.
        EstimatorNode(Estimator &estimator, const Interface::SessionOptions &options,
                      const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions(), const std::string &name = "spacetime_estimator");

        /// The estimation clock, on this node's ROS clock.
        const std::shared_ptr<RosClock> &clock() const { return mp_clock; }

        /// See Interface::EstimatorDriver::TickCallback. Call before start().
        void setTickCallback(Interface::EstimatorDriver::TickCallback callback) { m_tick_callback = std::move(callback); }
        /// See DoneCallback. Call before start().
        void setDoneCallback(DoneCallback callback) { m_done_callback = std::move(callback); }

        /**
         * @brief Starts the timer.
         * @throws std::invalid_argument for a non-positive period or a batch-mode estimator
         *         (see Interface::EstimatorDriver).
         */
        void start();

        /// The end time: +inf if negative in the options.
        double endTime() const;

    private:
        Estimator &m_estimator;
        Interface::SessionOptions m_options;
        std::shared_ptr<RosClock> mp_clock;
        std::unique_ptr<Interface::EstimatorDriver> mp_driver;
        Interface::EstimatorDriver::TickCallback m_tick_callback;
        DoneCallback m_done_callback;
        rclcpp::TimerBase::SharedPtr mp_timer;
        std::optional<double> m_last_time;
        bool m_done = false;

        void tick();
        void finish(std::exception_ptr error);
    };
} // namespace Spacetime::Ros2

#endif // SPACETIME_ROS2_ESTIMATORNODE_HPP
