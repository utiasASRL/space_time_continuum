/**
 * @file Interface.hpp
 * @brief The ROS2 interface's bundle, Interfaces::ROS2 (see Interface::InterfaceType): live
 *        (or rosbag) estimation on ROS2 nodes in one process. The session runs two
 *        single-threaded executors, each on its own thread: one runs only the estimator node
 *        (at normal priority, or SCHED_FIFO with SessionRosOptions::realtime_estimator), and
 *        the other, at a lowered priority (nice + SessionRosOptions::nice_delta), runs the
 *        sensor nodes and any other node added (e.g. an rviz snapshot publisher, see
 *        spacetime/visualizer/rviz/).
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
#include <spacetime/interface/ros2/PoseSensorNode.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>

namespace Spacetime::Ros2
{
    /// ROS-specific options of a Session, beyond Interface::SessionOptions.
    struct SessionRosOptions
    {
        /// Options of every node the session creates, e.g. use_sim_time (set globally by `--ros-args -p use_sim_time:=true` too).
        rclcpp::NodeOptions node_options;
        /// The estimator node is named `<node_prefix>_estimator`.
        std::string node_prefix = "spacetime";
        /// Added to the nice value of the thread running every node but the estimator's.
        int nice_delta = 10;
        /// Also request real-time (SCHED_FIFO) scheduling for the estimator's thread; needs the
        /// privilege (e.g. an rtprio limit), and only warns without it.
        bool realtime_estimator = false;
    };

    /**
     * @brief The ROS2 Session (see Interface::InterfaceType for each hook's contract).
     *        Measurements come from nodes subscribing to ROS topics, added with
     *        addPoseSensor(). Time is the estimator node's ROS clock (see RosClock), so
     *        estimation is always real time: live, or following a replayed /clock. The run
     *        also ends on rclcpp shutdown (e.g. Ctrl-C). rclcpp::init must have been called.
     */
    class Session
    {
    public:
        /**
         * @brief Creates the estimator node (nothing runs until start()). `estimator` must outlive the session.
         *        SessionOptions::real_time is ignored (always real time); a negative end_time runs until stopped.
         */
        Session(Estimator &estimator, const Interface::SessionOptions &options, const SessionRosOptions &ros_options = SessionRosOptions());
        /// Requests a stop and joins both executor threads.
        ~Session();

        Session(const Session &) = delete;
        Session &operator=(const Session &) = delete;

        /// Records the queues to feed, to warn at start() about any no node feeds (see unfedQueues).
        void addMeasurementQueues(const std::vector<Types::pMeasurementQueueType> &queues);

        /**
         * @brief Feeds `queue` (typically a Sensors::PoseSensor's) from a TF edge (see PoseSensorNode). Call before start().
         * @return The created node, owned jointly with the session (e.g. to read its counters).
         * @throws std::invalid_argument as PoseSensorNode's constructor does.
         */
        std::shared_ptr<PoseSensorNode> addPoseSensor(Types::pMeasurementQueueType queue, const PoseSensorOptions &options);

        /// Runs `node` on the lower-priority executor, e.g. a snapshot publisher. Call before start().
        /// @throws std::invalid_argument for a null node.
        void addNode(const rclcpp::Node::SharedPtr &node);

        /// The queues added with addMeasurementQueues that no sensor node feeds.
        std::vector<Types::pMeasurementQueueType> unfedQueues() const;

        /// The estimation clock (see RosClock), e.g. for nodes stamping estimates.
        std::shared_ptr<const RosClock> clock() const { return mp_estimator_node->clock(); }

        /// Options for nodes created next to the session's (its SessionRosOptions::node_options).
        const rclcpp::NodeOptions &nodeOptions() const { return m_ros_options.node_options; }

        /// Called on the estimator executor's thread after each estimate. Call before start().
        void setTickCallback(Interface::EstimatorDriver::TickCallback callback) { mp_estimator_node->setTickCallback(std::move(callback)); }
        /**
         * @brief Starts the estimator node's timer and both executor threads. Warns about unfed queues.
         * @throws std::invalid_argument for a non-positive period or a batch-mode estimator.
         * @throws std::logic_error if already started.
         */
        void start();
        /// Stops both executor threads as soon as possible. Safe from any thread, at any time.
        void requestStop();
        /// Blocks until both executor threads end (end time reached, error, requestStop, or rclcpp
        /// shutdown); rethrows the estimator's error, if any. Returns at once if never started.
        void join();
        /// True once the estimator thread has ended (or join() has returned).
        bool finished() const { return m_finished.load(); }
        /// The end time, in seconds; +infinity if SessionOptions::end_time was negative.
        double endTime() const { return mp_estimator_node->endTime(); }
        /// The current estimation time (see RosClock::now). Safe from any thread.
        double now() const { return mp_estimator_node->clock()->now(); }
        /// Always true: estimation follows the ROS clock.
        bool isRealTime() const { return true; }

    private:
        SessionRosOptions m_ros_options;
        std::shared_ptr<EstimatorNode> mp_estimator_node;
        std::vector<std::shared_ptr<PoseSensorNode>> m_sensor_nodes;
        std::vector<rclcpp::Node::SharedPtr> m_other_nodes;
        std::vector<Types::pMeasurementQueueType> m_queues;

        // The executors outlive the threads spinning them (declared before them).
        rclcpp::executors::SingleThreadedExecutor m_estimator_executor, m_node_executor;
        std::atomic<bool> m_stop{false}, m_finished{false};
        std::mutex m_error_mutex;
        std::exception_ptr m_error;
        std::jthread m_estimator_thread, m_node_thread;
        bool m_started = false;
    };
} // namespace Spacetime::Ros2

namespace Spacetime::Interfaces
{
    /// The ROS2 interface: live (or rosbag) estimation on ROS2 nodes (see Ros2::Session).
    struct ROS2
    {
        static constexpr std::string_view name = "ros2"; ///< Interface name.
        using Session = Spacetime::Ros2::Session;   ///< See Ros2::Session.
    };
    static_assert(Interface::InterfaceType<ROS2>);
} // namespace Spacetime::Interfaces

#endif // SPACETIME_ROS2_INTERFACE_HPP
