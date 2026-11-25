#include <cctype>
#include <stdexcept>
#include <utility>

#include "SensorNode.hpp"

namespace Spacetime::Interface::Ros2
{
    namespace
    {
        /// Throws (before rclcpp::Node is built) for a null queue or clock.
        const std::string &checked(const Types::pMeasurementQueueType &queue, const std::shared_ptr<const RosClock> &clock,
                                   const std::string &name, const std::string &type)
        {
            if (!queue || !clock)
                throw std::invalid_argument("Ros2::" + type + " requires a non-null queue and clock.");
            return name;
        }
    } // namespace

    SensorNode::SensorNode(Types::pMeasurementQueueType queue, std::shared_ptr<const RosClock> clock, const std::string &name,
                           const rclcpp::NodeOptions &node_options, const std::string &type)
        : rclcpp::Node(checked(queue, clock, name, type), node_options), mp_queue(std::move(queue)), mp_clock(std::move(clock))
    {
    }

    std::optional<double> SensorNode::acceptStamp(const builtin_interfaces::msg::Time &stamp, const std::string &what)
    {
        if (!mp_clock->started())
        {
            m_dropped++;
            return std::nullopt;
        }
        const double time = mp_clock->toEstimatorTime(stamp);
        if (time < mp_clock->startTime() || (m_last_time && time <= *m_last_time))
        {
            m_dropped++;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Dropping %s at t = %.4f s: before the start time or not newer than the last measurement.",
                                 what.c_str(), time);
            return std::nullopt;
        }
        m_last_time = time;
        return time;
    }

    void SensorNode::queueMeasurement(const std::shared_ptr<Types::MeasurementBase> &measurement)
    {
        mp_queue->addMeasurement(measurement);
        m_added++;
    }

    std::string SensorNode::stripSlash(const std::string &frame)
    {
        return (!frame.empty() && frame.front() == '/') ? frame.substr(1) : frame;
    }

    std::string SensorNode::nodeName(const std::string &prefix, const std::string &text)
    {
        std::string name = prefix;
        for (const char c : stripSlash(text))
            name += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        return name;
    }

    void SensorNode::validateCovariance(const Eigen::MatrixXd &covariance, const std::string &what)
    {
        if (!covariance.isApprox(covariance.transpose()) || !(covariance.diagonal().array() > 0.0).all())
            throw std::invalid_argument(what + ": the covariance must be symmetric with a positive diagonal.");
    }
} // namespace Spacetime::Interface::Ros2
