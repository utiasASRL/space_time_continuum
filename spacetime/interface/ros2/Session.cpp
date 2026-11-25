#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include <pthread.h>
#include <sched.h>

#include <spacetime/interface/standalone/ThreadPriority.hpp>

#include "Interface.hpp"

namespace Spacetime::Interface::Ros2
{
    namespace
    {
        /// How long an executor thread waits for work before checking for a stop.
        constexpr auto kSpinTimeout = std::chrono::milliseconds(50);

        /// Asks for real-time (SCHED_FIFO) scheduling of the calling thread; returns an error message, empty on success.
        std::string raiseToRealTime()
        {
            sched_param param{};
            param.sched_priority = std::max(1, sched_get_priority_min(SCHED_FIFO) + 10);
            const int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
            return result == 0 ? std::string() : std::string(std::strerror(result));
        }
    } // namespace

    Session::Session(const Interface::SessionOptions &options, const SessionRosOptions &ros_options)
        : m_ros_options(ros_options),
          mp_estimator_node(std::make_shared<EstimatorNode>(options, ros_options.node_options, ros_options.node_prefix + "_estimator"))
    {
        mp_estimator_node->setDoneCallback([this](std::exception_ptr error)
                                           {
            if (error)
            {
                std::lock_guard<std::mutex> lock(m_error_mutex);
                if (!m_error)
                    m_error = error;
            }
            requestStop(); });
    }

    Session::~Session()
    {
        requestStop();
        if (m_estimator_thread.joinable())
            m_estimator_thread.join();
        if (m_node_thread.joinable())
            m_node_thread.join();
    }

    std::shared_ptr<PoseSensorNode> Session::addPoseSensor(Types::pMeasurementQueueType queue, const PoseSensorOptions &options)
    {
        return addSensorNode(std::make_shared<PoseSensorNode>(std::move(queue), mp_estimator_node->clock(), options, m_ros_options.node_options));
    }

    std::shared_ptr<ImuSensorNode> Session::addGyro(Types::pMeasurementQueueType queue, const ImuSensorOptions &options)
    {
        return addSensorNode(std::make_shared<ImuSensorNode>(std::move(queue), mp_estimator_node->clock(), ImuMeasurement::Gyro, options, m_ros_options.node_options));
    }

    std::shared_ptr<ImuSensorNode> Session::addAccelerometer(Types::pMeasurementQueueType queue, const ImuSensorOptions &options)
    {
        return addSensorNode(std::make_shared<ImuSensorNode>(std::move(queue), mp_estimator_node->clock(), ImuMeasurement::Accelerometer, options,
                                                             m_ros_options.node_options));
    }

    std::shared_ptr<ImuSensorNode> Session::addIMU(Types::pMeasurementQueueType queue, const ImuSensorOptions &options)
    {
        return addSensorNode(std::make_shared<ImuSensorNode>(std::move(queue), mp_estimator_node->clock(), ImuMeasurement::IMU, options, m_ros_options.node_options));
    }

    std::shared_ptr<KinematicTendonNode> Session::addKinematicTendon(Types::pMeasurementQueueType queue, const KinematicTendonOptions &options)
    {
        return addSensorNode(std::make_shared<KinematicTendonNode>(std::move(queue), mp_estimator_node->clock(), options, m_ros_options.node_options));
    }

    void Session::addNode(const rclcpp::Node::SharedPtr &node)
    {
        if (!node)
            throw std::invalid_argument("Ros2::Session::addNode: the node must be non-null.");
        m_other_nodes.push_back(node);
        if (m_started)
            m_node_executor.add_node(node);
    }

    std::vector<Types::pMeasurementQueueType> Session::unfedQueues(const Estimator &estimator) const
    {
        std::vector<Types::pMeasurementQueueType> unfed;
        for (const auto &queue : estimator.getMeasurementQueues())
            if (std::none_of(m_sensor_nodes.begin(), m_sensor_nodes.end(), [&](const auto &node)
                             { return node->queue() == queue; }))
                unfed.push_back(queue);
        return unfed;
    }

    void Session::start(Estimator &estimator)
    {
        if (m_started)
            throw std::logic_error("Ros2::Session::start: already started.");
        mp_estimator_node->start(estimator); // validates the options
        m_started = true;
        if (const auto unfed = unfedQueues(estimator); !unfed.empty())
            RCLCPP_WARN(mp_estimator_node->get_logger(), "%zu measurement queue(s) have no node feeding them: their models get no measurements.", unfed.size());

        m_estimator_executor.add_node(mp_estimator_node);
        for (const auto &node : m_sensor_nodes)
            m_node_executor.add_node(node);
        for (const auto &node : m_other_nodes)
            m_node_executor.add_node(node);

        m_estimator_thread = std::jthread([this]()
                                          {
            if (m_ros_options.realtime_estimator)
                if (const std::string error = raiseToRealTime(); !error.empty())
                    RCLCPP_WARN(mp_estimator_node->get_logger(), "Could not schedule the estimator thread in real time (SCHED_FIFO): %s.", error.c_str());
            while (!m_stop.load() && rclcpp::ok())
                m_estimator_executor.spin_once(kSpinTimeout);
            // Also ends the other nodes' thread, e.g. once the end time is reached or on shutdown.
            m_stop.store(true);
            m_finished.store(true); });
        m_node_thread = std::jthread([this]()
                                     {
            Standalone::lowerCurrentThreadPriority(m_ros_options.nice_delta);
            while (!m_stop.load() && rclcpp::ok())
                m_node_executor.spin_once(kSpinTimeout); });
    }

    void Session::requestStop()
    {
        m_stop.store(true);
        m_estimator_executor.cancel();
        m_node_executor.cancel();
    }

    void Session::join()
    {
        if (m_estimator_thread.joinable())
            m_estimator_thread.join();
        if (m_node_thread.joinable())
            m_node_thread.join();
        m_finished.store(true);
        std::lock_guard<std::mutex> lock(m_error_mutex);
        if (m_error)
            std::rethrow_exception(m_error);
    }
} // namespace Spacetime::Interface::Ros2
