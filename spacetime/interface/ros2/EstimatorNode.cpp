#include <chrono>
#include <limits>
#include <stdexcept>

#include <rclcpp/create_timer.hpp>

#include "EstimatorNode.hpp"

namespace Spacetime::Interface::Ros2
{
    EstimatorNode::EstimatorNode(const Interface::SessionOptions &options,
                                 const rclcpp::NodeOptions &node_options, const std::string &name)
        : rclcpp::Node(name, node_options), m_options(options),
          mp_clock(std::make_shared<RosClock>(get_clock(), options.start_time)) {}

    void EstimatorNode::start(Estimator &estimator)
    {
        if (!(m_options.period > 0.0))
            throw std::invalid_argument("Ros2::EstimatorNode: the period must be positive.");
        mp_driver = std::make_unique<Interface::EstimatorDriver>(estimator, true);
        mp_driver->setTickCallback(m_tick_callback);
        // On the node's clock (not a wall timer), so it follows use_sim_time.
        mp_timer = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(m_options.period), [this]()
                                        { tick(); });
    }

    double EstimatorNode::endTime() const
    {
        return m_options.end_time < 0.0 ? std::numeric_limits<double>::infinity() : m_options.end_time;
    }

    void EstimatorNode::tick()
    {
        if (m_done || !mp_clock->tryStart())
            return;
        const double time = mp_clock->now();
        if (time > endTime())
        {
            // Not clamped to the end time: that would add a row right after the last one,
            // which leaves too little time between them for a well-conditioned prior.
            finish(nullptr);
            return;
        }
        if (m_last_time && time <= *m_last_time)
            return; // e.g. a paused or rewound rosbag
        try
        {
            mp_driver->step(time);
            m_last_time = time;
        }
        catch (...)
        {
            finish(std::current_exception());
        }
    }

    void EstimatorNode::finish(std::exception_ptr error)
    {
        m_done = true;
        mp_timer->cancel();
        if (m_done_callback)
            m_done_callback(error);
    }
} // namespace Spacetime::Interface::Ros2
