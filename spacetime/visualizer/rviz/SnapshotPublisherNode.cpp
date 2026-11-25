#include <array>
#include <chrono>
#include <regex>
#include <utility>

#include <tf2_ros/qos.hpp>

#include <spacetime/interface/ros2/Conversions.hpp>
#include <spacetime/visualizer/rviz/Markers.hpp>

#include "SnapshotPublisherNode.hpp"

namespace Spacetime::Visualization::Rviz
{
    namespace
    {
        using Marker = visualization_msgs::msg::Marker;

        /// Grey levels (0-1) a robot's tube alternates between, one per node-to-node segment (as the VTK view's kBackboneShades).
        constexpr std::array<float, 2> kBackboneShades{110.0f / 255.0f, 150.0f / 255.0f};
        /// Grey level (0-1) of a joint: darker than the robots it joins.
        constexpr float kJointGrey = 0.25f;
        /// Opacity of the covariance ellipsoids.
        constexpr float kEllipsoidAlpha = 0.15f;

        void setColor(Marker &marker, float r, float g, float b, float a = 1.0f)
        {
            marker.color.r = r;
            marker.color.g = g;
            marker.color.b = b;
            marker.color.a = a;
        }

        Marker marker(const std::string &ns, int id, const std_msgs::msg::Header &header)
        {
            Marker m;
            m.header = header;
            m.ns = ns;
            m.id = id;
            m.action = Marker::ADD;
            return m;
        }
    } // namespace

    SnapshotPublisherNode::SnapshotPublisherNode(SnapshotSource source, std::shared_ptr<const Interface::Ros2::RosClock> clock,
                                                 const SnapshotPublisherOptions &options, const rclcpp::NodeOptions &node_options,
                                                 const std::string &name)
        : rclcpp::Node(name, node_options), m_source(std::move(source)), mp_clock(std::move(clock)), m_options(options)
    {
        if (!m_source || !mp_clock)
            throw std::invalid_argument("Rviz::SnapshotPublisherNode requires a snapshot source and a clock.");
        mp_markers_publisher = create_publisher<visualization_msgs::msg::MarkerArray>(m_options.topic_prefix + "/markers", 10);
        mp_tf_publisher = create_publisher<tf2_msgs::msg::TFMessage>(m_options.tf_topic, tf2_ros::DynamicBroadcasterQoS());
        if (m_options.rate > 0.0)
            mp_timer = create_wall_timer(std::chrono::duration<double>(1.0 / m_options.rate), [this]()
                                         { publishLatest(); });
    }

    void SnapshotPublisherNode::checkName(const std::string &name) const
    {
        static const std::regex valid("[A-Za-z][A-Za-z0-9_]*");
        if (!std::regex_match(name, valid))
            throw std::invalid_argument("Rviz::SnapshotPublisherNode: '" + name + "' is not a valid name ([A-Za-z][A-Za-z0-9_]*).");
        for (const auto &robot : m_robots)
            if (robot.name == name)
                throw std::invalid_argument("Rviz::SnapshotPublisherNode: the name '" + name + "' is taken.");
        for (const auto &joint : m_joints)
            if (joint.name == name)
                throw std::invalid_argument("Rviz::SnapshotPublisherNode: the name '" + name + "' is taken.");
    }

    void SnapshotPublisherNode::addRobot(Robot robot)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        checkName(robot.name);
        for (const auto &other : m_robots)
            if (other.state == robot.state)
                throw std::invalid_argument("Rviz::SnapshotPublisherNode::addRobot: this robot is published already, as '" + other.name + "'.");
        robot.poses_publisher = create_publisher<geometry_msgs::msg::PoseArray>(m_options.topic_prefix + "/" + robot.name + "/poses", 10);
        m_robots.push_back(std::move(robot));
    }

    void SnapshotPublisherNode::addJoint(const std::string &name, const Types::State *state_a, double s_a, const Types::State *state_b, double s_b,
                                         bool has_offset, double radius)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        checkName(name);
        auto find = [&](const Types::State *state) -> std::size_t
        {
            for (std::size_t i = 0; i < m_robots.size(); i++)
                if (m_robots[i].state == state)
                    return i;
            throw std::invalid_argument("Rviz::SnapshotPublisherNode::addJoint: add both robots of joint '" + name + "' first.");
        };
        const std::size_t a = find(state_a), b = find(state_b);
        if (m_robots[a].options.frame_id != m_robots[b].options.frame_id)
            throw std::invalid_argument("Rviz::SnapshotPublisherNode::addJoint: the robots of joint '" + name + "' are published in different frames.");
        m_joints.push_back({name, a, b, s_a, s_b, has_offset, radius});
    }

    bool SnapshotPublisherNode::publishLatest()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto snapshot = m_source();
        if (!snapshot || snapshot == mp_last_published || !mp_clock->started())
            return false;
        mp_last_published = snapshot;
        const builtin_interfaces::msg::Time stamp = mp_clock->toRosTime(snapshot->time);

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.emplace_back();
        markers.markers.back().action = Marker::DELETEALL; // clears the previous snapshot's markers
        tf2_msgs::msg::TFMessage frames;

        for (const auto &robot : m_robots)
        {
            std_msgs::msg::Header header;
            header.stamp = stamp;
            header.frame_id = robot.options.frame_id;
            geometry_msgs::msg::PoseArray poses;
            poses.header = header;

            const PosteriorCovariance *posterior = robot.options.covariance ? snapshot->covariance.get() : nullptr;
            const auto backbone = robot.backbone(*snapshot, posterior);
            if (backbone)
            {
                for (std::size_t k = 0; k < backbone->points.size(); k++)
                {
                    const Eigen::Matrix4d &pose = backbone->points[k].pose;
                    poses.poses.push_back(Interface::Ros2::toPoseMsg(gtsam::Pose3(pose)));

                    if (k + 1 < backbone->points.size())
                    {
                        Marker segment = marker(robot.name, static_cast<int>(k), header);
                        setCylinderBetween(segment, pose.block<3, 1>(0, 3), backbone->points[k + 1].pose.block<3, 1>(0, 3), robot.options.radius);
                        const float shade = kBackboneShades[(k / backbone->points_per_segment) % 2];
                        setColor(segment, shade, shade, shade);
                        markers.markers.push_back(segment);
                    }

                    // To first order the body-frame translation perturbation moves the world
                    // position by R * delta_t, so its world covariance is R Sigma_tt R^T.
                    const Eigen::MatrixXd &covariance = backbone->points[k].covariance;
                    if (covariance.rows() >= 6 && covariance.cols() >= 6)
                    {
                        const Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
                        Marker ellipsoid = marker(robot.name + "/covariance", static_cast<int>(k), header);
                        setEllipsoid(ellipsoid, pose.block<3, 1>(0, 3), R * covariance.block<3, 3>(3, 3) * R.transpose());
                        setColor(ellipsoid, 0.0f, 0.0f, 1.0f, kEllipsoidAlpha);
                        markers.markers.push_back(ellipsoid);
                    }

                    if (robot.options.node_frames && k % backbone->points_per_segment == 0)
                    {
                        geometry_msgs::msg::TransformStamped frame;
                        frame.header = header;
                        frame.child_frame_id = robot.name + "/node_" + std::to_string(k / backbone->points_per_segment);
                        frame.transform = Interface::Ros2::toTransformMsg(gtsam::Pose3(pose));
                        frames.transforms.push_back(frame);
                    }
                }
            }
            robot.poses_publisher->publish(poses);
        }

        for (const auto &joint : m_joints)
        {
            const auto pose_a = m_robots[joint.robot_a].pose_at(*snapshot, joint.s_a);
            const auto pose_b = m_robots[joint.robot_b].pose_at(*snapshot, joint.s_b);
            if (!pose_a || !pose_b)
                continue;
            std_msgs::msg::Header header;
            header.stamp = stamp;
            header.frame_id = m_robots[joint.robot_a].options.frame_id;
            const Eigen::Vector3d p_a = pose_a->block<3, 1>(0, 3), p_b = pose_b->block<3, 1>(0, 3);
            Marker m = marker(joint.name, 0, header);
            if (joint.has_offset)
                setCylinderBetween(m, p_a, p_b, joint.radius);
            else
                setSphere(m, 0.5 * (p_a + p_b), joint.radius); // between the two estimates, which a soft joint only holds approximately together
            setColor(m, kJointGrey, kJointGrey, kJointGrey);
            markers.markers.push_back(m);
        }

        mp_markers_publisher->publish(markers);
        if (!frames.transforms.empty())
            mp_tf_publisher->publish(frames);
        return true;
    }
} // namespace Spacetime::Visualization::Rviz
