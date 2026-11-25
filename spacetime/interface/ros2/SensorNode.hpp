/**
 * @file SensorNode.hpp
 * @brief The base of the ROS2 nodes that turn messages into measurements queued for a
 *        model (e.g. PoseSensorNode, ImuSensorNode, KinematicTendonNode).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_SENSORNODE_HPP
#define SPACETIME_ROS2_SENSORNODE_HPP

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>

#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>

namespace Spacetime::Interface::Ros2
{
    /**
     * @brief A node feeding one measurement queue (typically a sensor's or an actuator's)
     *        from ROS messages, stamped in estimation time (see RosClock). Stamps before the
     *        clock starts, before its start time, or not newer than the last one queued are
     *        dropped: MeasurementQueue drains in insertion order. Its counters are atomic:
     *        safe to read from any thread.
     */
    class SensorNode : public rclcpp::Node
    {
    public:
        /// The queue the node feeds.
        const Types::pMeasurementQueueType &queue() const { return mp_queue; }

        /// Messages received, used or not.
        std::size_t messagesReceived() const { return m_received.load(); }
        /// Measurements queued.
        std::size_t measurementsAdded() const { return m_added.load(); }
        /// Relevant messages dropped (see the class comment and each node's own rules).
        std::size_t measurementsDropped() const { return m_dropped.load(); }

        /// `frame` without a leading '/' (TF frame ids are compared that way, as tf2 does).
        static std::string stripSlash(const std::string &frame);
        /// `prefix` followed by `text` (without a leading '/'), every non-alphanumeric character replaced by '_': a valid node name.
        static std::string nodeName(const std::string &prefix, const std::string &text);
        /// Throws std::invalid_argument (prefixed by `what`) unless `covariance` is symmetric with a positive diagonal.
        static void validateCovariance(const Eigen::MatrixXd &covariance, const std::string &what);

    protected:
        /**
         * @param queue The queue measurements are added to; shared with the node.
         * @param clock Converts stamps to estimation time; typically Session::clock().
         * @param name The node's name.
         * @param type The derived node's type, for error messages (e.g. "PoseSensorNode").
         * @throws std::invalid_argument for a null queue or clock.
         */
        SensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, const std::string &name,
                   const rclcpp::NodeOptions &node_options, const std::string &type);

        /// The estimation clock.
        const std::shared_ptr<const RosClock> &rosClock() const { return mp_clock; }

        /**
         * @brief The estimation time of `stamp`, which becomes the last accepted one; nullopt
         *        (counted as dropped, warning with `what` unless the clock has not started)
         *        before the clock starts, before its start time, or if not newer than the last.
         */
        std::optional<double> acceptStamp(const builtin_interfaces::msg::Time &stamp, const std::string &what);

        /// Queues `measurement` and counts it.
        void queueMeasurement(const std::shared_ptr<Types::MeasurementBase> &measurement);
        /// Counts a received message.
        void countReceived() { m_received++; }
        /// Counts a dropped measurement.
        void countDropped() { m_dropped++; }

    private:
        Types::pMeasurementQueueType mp_queue;
        std::shared_ptr<const RosClock> mp_clock;
        std::optional<double> m_last_time;
        std::atomic<std::size_t> m_received{0}, m_added{0}, m_dropped{0};
    };
} // namespace Spacetime::Interface::Ros2

#endif // SPACETIME_ROS2_SENSORNODE_HPP
