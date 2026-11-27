/**
 * @file Interface.hpp
 * @brief The ROS2 interface, selected by Spacetime::Session<INTERFACE_ROS2>: live (or
 *        rosbag) estimation on ROS2 nodes in one process. The session runs two
 *        single-threaded executors, each on its own thread: one runs only the estimator node
 *        (raised where the OS allows, see Standalone::prioritizeEstimatorThread, or real-time
 *        with SessionRosOptions::realtime_estimator), and the other, at a lowered priority
 *        (SessionRosOptions::nice_delta), runs the sensor nodes and any other node added (e.g.
 *        an rviz snapshot publisher, see spacetime/visualizer/rviz/).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_ROS2_INTERFACE_HPP
#define SPACETIME_ROS2_INTERFACE_HPP

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/interface/InterfaceConcept.hpp>
#include <spacetime/interface/ros2/EstimatorNode.hpp>
#include <spacetime/interface/ros2/ImuSensorNode.hpp>
#include <spacetime/interface/ros2/KinematicTendonNode.hpp>
#include <spacetime/interface/ros2/PoseSensorNode.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>
#include <spacetime/interface/ros2/SensorNode.hpp>

namespace Spacetime::Interface::Ros2
{
    /// ROS-specific options of a Session, beyond Interface::SessionOptions.
    struct SessionRosOptions
    {
        /// Options of every node the session creates, e.g. use_sim_time (set globally by `--ros-args -p use_sim_time:=true` too).
        rclcpp::NodeOptions node_options;
        /// The estimator node is named `<node_prefix>_estimator`.
        std::string node_prefix = "spacetime";
        /// How much to lower the thread running every node but the estimator's (see
        /// Standalone::lowerCurrentThreadPriority: a nice-value increment on Linux).
        int nice_delta = 10;
        /// Also request real-time scheduling for the estimator's thread (see
        /// Standalone::raiseCurrentThreadToRealTime: SCHED_FIFO on Linux, which needs the
        /// privilege, e.g. an rtprio limit); only warns if refused.
        bool realtime_estimator = false;
    };

    /**
     * @brief The ROS2 Session (see Interface::InterfaceType for each hook's contract).
     *        Measurements come from nodes subscribing to ROS topics (see SensorNode), added
     *        with addPoseSensor(), addGyro(), addAccelerometer(), addIMU() and
     *        addKinematicTendon(). Time is the estimator node's ROS clock (see RosClock), so
     *        estimation is always real time: live, or following a replayed /clock. The run
     *        also ends on rclcpp shutdown (e.g. Ctrl-C). rclcpp::init must have been called.
     */
    class Session
    {
    public:
        /**
         * @brief Creates the estimator node (nothing runs until start()).
         *        SessionOptions::real_time is ignored (always real time); a negative end_time runs until stopped.
         */
        explicit Session(const Interface::SessionOptions &options = Interface::SessionOptions(), const SessionRosOptions &ros_options = SessionRosOptions());
        /// Requests a stop and joins both executor threads.
        ~Session();

        Session(const Session &) = delete;
        Session &operator=(const Session &) = delete;

        /**
         * @brief Feeds `queue` (typically a Sensors::PoseSensor's) from a TF edge (see PoseSensorNode). Call before start().
         * @return The created node, owned jointly with the session (e.g. to read its counters).
         * @throws std::invalid_argument as PoseSensorNode's constructor does.
         */
        std::shared_ptr<PoseSensorNode> addPoseSensor(Types::pMeasurementQueueType queue, const PoseSensorOptions &options);

        /**
         * @brief Feeds `queue` (typically a Sensors::GyroSensor's) with the angular velocity of
         *        sensor_msgs/Imu messages (see ImuSensorNode). Call before start().
         * @return The created node, owned jointly with the session.
         * @throws std::invalid_argument as ImuSensorNode's constructor does.
         */
        std::shared_ptr<ImuSensorNode> addGyro(Types::pMeasurementQueueType queue, const ImuSensorOptions &options);

        /**
         * @brief Feeds `queue` (typically a Sensors::AccelerometerSensor's) with the linear
         *        acceleration of sensor_msgs/Imu messages (see ImuSensorNode). Call before start().
         * @return The created node, owned jointly with the session.
         * @throws std::invalid_argument as ImuSensorNode's constructor does.
         */
        std::shared_ptr<ImuSensorNode> addAccelerometer(Types::pMeasurementQueueType queue, const ImuSensorOptions &options);

        /**
         * @brief Feeds `queue` (typically a Sensors::IMUSensor's) with the angular velocity and
         *        linear acceleration of sensor_msgs/Imu messages (see ImuSensorNode). Call before start().
         * @return The created node, owned jointly with the session.
         * @throws std::invalid_argument as ImuSensorNode's constructor does.
         */
        std::shared_ptr<ImuSensorNode> addIMU(Types::pMeasurementQueueType queue, const ImuSensorOptions &options);

        /**
         * @brief Feeds `queue` (typically an Actuators::KinematicTendon's) with one joint of
         *        sensor_msgs/JointState messages (see KinematicTendonNode). Call before start().
         * @return The created node, owned jointly with the session.
         * @throws std::invalid_argument as KinematicTendonNode's constructor does.
         */
        std::shared_ptr<KinematicTendonNode> addKinematicTendon(Types::pMeasurementQueueType queue, const KinematicTendonOptions &options);

        /// Runs `node` on the lower-priority executor, e.g. a snapshot publisher. Call before start().
        /// @throws std::invalid_argument for a null node.
        void addNode(const rclcpp::Node::SharedPtr &node);

        /// `estimator`'s measurement queues (see Estimator::getMeasurementQueues) that no sensor node feeds; start() warns about them.
        std::vector<Types::pMeasurementQueueType> unfedQueues(const Estimator &estimator) const;

        /// The estimation clock (see RosClock), e.g. for nodes stamping estimates.
        std::shared_ptr<const RosClock> clock() const { return mp_estimator_node->clock(); }

        /// Options for nodes created next to the session's (its SessionRosOptions::node_options).
        const rclcpp::NodeOptions &nodeOptions() const { return m_ros_options.node_options; }

        /// Called on the estimator executor's thread after each estimate. Call before start().
        void setTickCallback(Interface::EstimatorDriver::TickCallback callback) { mp_estimator_node->setTickCallback(std::move(callback)); }
        /**
         * @brief Starts estimating with `estimator`, which must outlive the session: the estimator
         *        node's timer and both executor threads. Warns about its unfed queues (see unfedQueues).
         * @throws std::invalid_argument for a non-positive period or a batch-mode estimator.
         * @throws std::logic_error if already started.
         */
        void start(Estimator &estimator);
        /// Stops both executor threads as soon as possible. Safe from any thread, at any time.
        void requestStop();
        /// Blocks until both executor threads end (end time reached, error, requestStop, or rclcpp
        /// shutdown); rethrows the estimator's error, if any. Returns at once if never started.
        void join();
        /// True once the estimator thread has ended (or join() has returned).
        bool finished() const { return m_finished.load(); }
        /// The end time, in seconds; +infinity if SessionOptions::end_time was negative.
        double endTime() const { return mp_estimator_node->endTime(); }
        /// The time of the first estimate (SessionOptions::start_time), in seconds.
        double startTime() const { return mp_estimator_node->startTime(); }
        /// The current estimation time (see RosClock::now). Safe from any thread.
        double now() const { return mp_estimator_node->clock()->now(); }
        /// Always true: estimation follows the ROS clock.
        bool isRealTime() const { return true; }

    private:
        SessionRosOptions m_ros_options;
        std::shared_ptr<EstimatorNode> mp_estimator_node;
        std::vector<std::shared_ptr<SensorNode>> m_sensor_nodes;
        std::vector<rclcpp::Node::SharedPtr> m_other_nodes;

        // The executors outlive the threads spinning them (declared before them).
        rclcpp::executors::SingleThreadedExecutor m_estimator_executor, m_node_executor;
        std::atomic<bool> m_stop{false}, m_finished{false};
        std::mutex m_error_mutex;
        std::exception_ptr m_error;
        std::jthread m_estimator_thread, m_node_thread;
        bool m_started = false;

        /// Registers `node` as a sensor node (and runs it, if already started); returns it.
        template <typename NodeT>
        std::shared_ptr<NodeT> addSensorNode(std::shared_ptr<NodeT> node)
        {
            m_sensor_nodes.push_back(node);
            if (m_started)
                m_node_executor.add_node(node);
            return node;
        }
    };
} // namespace Spacetime::Interface::Ros2

namespace Spacetime::Interface
{
    /// The ROS2 interface: live (or rosbag) estimation on ROS2 nodes (see Ros2::Session).
    template <>
    struct Traits<INTERFACE_ROS2>
    {
        static constexpr std::string_view name = "ros2"; ///< Interface name.
        using Session = Ros2::Session;                   ///< See Ros2::Session.
    };
    static_assert(InterfaceType<Traits<INTERFACE_ROS2>>);
} // namespace Spacetime::Interface

#endif // SPACETIME_ROS2_INTERFACE_HPP
