/**
 * @file SnapshotPublisherNode.hpp
 * @brief A ROS2 node publishing the estimator's snapshots as standard messages rviz2 draws
 *        natively: robot backbones (geometry_msgs/PoseArray), robots, joints and posterior
 *        covariance ellipsoids (visualization_msgs/MarkerArray), and node frames (TF).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_RVIZ_SNAPSHOTPUBLISHERNODE_HPP
#define SPACETIME_RVIZ_SNAPSHOTPUBLISHERNODE_HPP

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/pose_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/RobotShape.hpp>
#include <spacetime/interface/ros2/RosClock.hpp>

namespace Spacetime::Visualization::Rviz
{
    /// Where and how often a SnapshotPublisherNode publishes.
    struct SnapshotPublisherOptions
    {
        /// Checks for a new snapshot per second (wall time); <= 0 publishes only on publishLatest().
        double rate = 30.0;
        /// Topics are `<topic_prefix>/<robot>/poses` and `<topic_prefix>/markers` (relative to the node's namespace unless absolute).
        std::string topic_prefix = "spacetime";
        /// The TFMessage topic node frames are broadcast on.
        std::string tf_topic = "/tf";
    };

    /// How one robot is published (see SnapshotPublisherNode::addRobot).
    struct RobotDisplayOptions
    {
        /// The frame the robot's poses are expressed in (its prior's world frame), e.g. "aurora_field".
        std::string frame_id = "world";
        /// Radius of the drawn tube [m].
        double radius = 0.01;
        /// Extra points per node-to-node segment, for a smoother backbone.
        unsigned int extra_points = 3;
        /// Broadcast each node's pose on TF as `<robot>/node_<i>`.
        bool node_frames = false;
        /// Draw the posterior position-covariance ellipsoids (needs EstimatorOptions::compute_covariance).
        bool covariance = false;
    };

    /**
     * @brief Publishes the latest snapshot whenever it changes. Each robot's backbone is
     *        resolved through its prior's interpolator (see Utilities::RobotShape) at the
     *        snapshot's time, from a ContinuumRobotState::snapshotEstimationNodes() copy, so
     *        it never touches the estimator off its thread. Per snapshot it publishes, stamped
     *        at the snapshot time:
     *        - `<topic_prefix>/<robot>/poses`: the backbone, base to tip (empty while the robot
     *          cannot be placed, e.g. before its first row);
     *        - `<topic_prefix>/markers`: a DELETEALL, then each robot as cylinders (namespace
     *          `<robot>`, alternating grey per node-to-node segment), its covariance ellipsoids
     *          (`<robot>/covariance`) and each joint (namespace `<joint>`): a cylinder between
     *          its points with an offset, a sphere at them without;
     *        - on `tf_topic`, the node frames of robots that ask for them.
     *        Snapshots are skipped until the clock has started (stamps need its origin).
     *        Add robots and joints before the node is spun (adding later is guarded by a
     *        mutex, but takes effect only from the next publish). The backbone and joint
     *        functions run on the thread spinning this node, under that mutex; the prior,
     *        its state and the snapshot source must outlive the node.
     */
    class SnapshotPublisherNode : public rclcpp::Node
    {
    public:
        /// The latest snapshot (e.g. Estimator::getSnapshot), or null before the first.
        using SnapshotSource = std::function<std::shared_ptr<const EstimatorSnapshot>()>;

        /**
         * @brief Advertises `<topic_prefix>/markers` and `tf_topic`, and starts a wall timer at
         *        `options.rate` (if positive) calling publishLatest().
         * @param source Polled on every timer tick, on the spinning thread.
         * @param clock Converts snapshot times to ROS stamps; typically Ros2::Session::clock().
         * @throws std::invalid_argument for an empty source or a null clock.
         */
        SnapshotPublisherNode(SnapshotSource source, std::shared_ptr<const Ros2::RosClock> clock,
                              const SnapshotPublisherOptions &options = SnapshotPublisherOptions(),
                              const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions(), const std::string &name = "spacetime_snapshot_publisher");

        /**
         * @brief Publishes the robot `prior` (e.g. a Priors::WNOA::Prior<Pose3>) extends, as `name`,
         *        and advertises `<topic_prefix>/<name>/poses`.
         * @tparam PriorT A prior with a `NodeType` and a `state()` (a ContinuumRobotState).
         * @throws std::invalid_argument for a null prior, a name that is taken or not a valid
         *         topic token ([A-Za-z][A-Za-z0-9_]*), or a robot added already.
         */
        template <typename PriorT>
        void addRobot(const std::string &name, std::shared_ptr<PriorT> prior, const RobotDisplayOptions &options = RobotDisplayOptions())
        {
            if (!prior)
                throw std::invalid_argument("Rviz::SnapshotPublisherNode::addRobot: the prior must be non-null.");
            using NodeT = typename PriorT::NodeType;
            auto state = prior->state();
            auto shape = std::make_shared<::Spacetime::Utilities::RobotShape<NodeT>>(state);
            shape->bindPriorInterpolator(prior);

            Robot robot;
            robot.name = name;
            robot.options = options;
            robot.state = state.get();
            robot.backbone = [shape, state, extra_points = options.extra_points](const EstimatorSnapshot &snapshot, const PosteriorCovariance *posterior)
                -> std::optional<Backbone>
            {
                const auto backbone = shape->backbone(snapshot.time, snapshot, state->snapshotEstimationNodes(), extra_points, false, posterior);
                if (!backbone)
                    return std::nullopt;
                Backbone result{backbone->points_per_segment, {}};
                result.points.reserve(backbone->points.size());
                for (const auto &point : backbone->points)
                    result.points.push_back({point.pose, point.covariance});
                return result;
            };
            robot.pose_at = [shape, state](const EstimatorSnapshot &snapshot, double arclength)
            { return shape->poseAt(snapshot.time, arclength, snapshot, state->snapshotEstimationNodes(), false); };
            addRobot(std::move(robot));
        }

        /**
         * @brief Draws `joint` (a Joints::Joint) between the robots of its two priors, added before it.
         *        Only its priors' states, arclengths and hasOffset() are read, here; `joint`
         *        need not outlive the node.
         * @tparam JointT A joint with priorA()/priorB(), arclengthA()/arclengthB() and hasOffset().
         * @param radius Radius of the drawn cylinder or sphere [m].
         * @throws std::invalid_argument if the name is taken or invalid, either robot was not
         *         added, or they are published in different frames.
         */
        template <typename JointT>
        void addJoint(const std::string &name, const JointT &joint, double radius = 0.015)
        {
            addJoint(name, joint.priorA()->state().get(), joint.arclengthA(), joint.priorB()->state().get(), joint.arclengthB(),
                     joint.hasOffset(), radius);
        }

        /// Publishes the latest snapshot if it is new (and the clock has started); returns whether it did.
        /// Called by the timer; may also be called directly (thread-safe, serialized by a mutex).
        bool publishLatest();

    private:
        /// One backbone sample: world pose and (if requested and available) its 6 x 6 tangent covariance.
        struct Point
        {
            Eigen::Matrix4d pose;
            Eigen::MatrixXd covariance; ///< Empty when not drawn or unavailable.
        };
        /// A sampled backbone, base to tip; every `points_per_segment`-th point is a node.
        struct Backbone
        {
            unsigned int points_per_segment;
            std::vector<Point> points;
        };
        /// A published robot: type-erased shape queries bound to its prior (see addRobot).
        struct Robot
        {
            std::string name;
            RobotDisplayOptions options;
            const void *state; ///< Identifies the robot (see addJoint).
            /// The backbone at the snapshot's time, or nullopt if it cannot be placed yet.
            std::function<std::optional<Backbone>(const EstimatorSnapshot &, const PosteriorCovariance *)> backbone;
            /// The pose at an arclength [m] at the snapshot's time, or nullopt.
            std::function<std::optional<Eigen::Matrix4d>(const EstimatorSnapshot &, double)> pose_at;
            rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_publisher;
        };
        /// A published joint, by index into m_robots and arclength [m] on each.
        struct Joint
        {
            std::string name;
            std::size_t robot_a, robot_b;
            double s_a, s_b;
            bool has_offset; ///< Drawn as a cylinder from A to B if true, else a sphere at their midpoint.
            double radius;   ///< [m].
        };

        SnapshotSource m_source;
        std::shared_ptr<const Ros2::RosClock> mp_clock;
        SnapshotPublisherOptions m_options;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mp_markers_publisher;
        rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr mp_tf_publisher;
        rclcpp::TimerBase::SharedPtr mp_timer;
        std::mutex m_mutex; ///< Guards everything below (robots/joints may be added while publishing).
        std::vector<Robot> m_robots;
        std::vector<Joint> m_joints;
        std::shared_ptr<const EstimatorSnapshot> mp_last_published;

        /// Registers `robot` and creates its poses publisher. Throws as the public addRobot.
        void addRobot(Robot robot);
        /// Registers a joint between the robots identified by `state_a` and `state_b`. Throws as the public addJoint.
        void addJoint(const std::string &name, const void *state_a, double s_a, const void *state_b, double s_b, bool has_offset, double radius);
        /// Throws std::invalid_argument if `name` is not a valid topic token or is taken by a robot or joint.
        void checkName(const std::string &name) const;
    };
} // namespace Spacetime::Visualization::Rviz

#endif // SPACETIME_RVIZ_SNAPSHOTPUBLISHERNODE_HPP
