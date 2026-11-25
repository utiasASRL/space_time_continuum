#include <stdexcept>

#include "Conversions.hpp"

namespace Spacetime::Interface::Ros2
{
    namespace
    {
        gtsam::Rot3 toRot3(double w, double x, double y, double z)
        {
            gtsam::Quaternion q(w, x, y, z);
            const double norm = q.norm();
            if (!(norm > 0.0))
                throw std::invalid_argument("Ros2::toPose3: the rotation quaternion is zero.");
            q.coeffs() /= norm;
            return gtsam::Rot3(q);
        }
    } // namespace

    gtsam::Pose3 toPose3(const geometry_msgs::msg::Transform &transform)
    {
        const auto &q = transform.rotation;
        const auto &t = transform.translation;
        return gtsam::Pose3(toRot3(q.w, q.x, q.y, q.z), gtsam::Point3(t.x, t.y, t.z));
    }

    gtsam::Pose3 toPose3(const geometry_msgs::msg::Pose &pose)
    {
        const auto &q = pose.orientation;
        const auto &p = pose.position;
        return gtsam::Pose3(toRot3(q.w, q.x, q.y, q.z), gtsam::Point3(p.x, p.y, p.z));
    }

    geometry_msgs::msg::Transform toTransformMsg(const gtsam::Pose3 &pose)
    {
        const gtsam::Quaternion q = pose.rotation().toQuaternion();
        geometry_msgs::msg::Transform transform;
        transform.translation.x = pose.x();
        transform.translation.y = pose.y();
        transform.translation.z = pose.z();
        transform.rotation.w = q.w();
        transform.rotation.x = q.x();
        transform.rotation.y = q.y();
        transform.rotation.z = q.z();
        return transform;
    }

    geometry_msgs::msg::Pose toPoseMsg(const gtsam::Pose3 &pose)
    {
        const gtsam::Quaternion q = pose.rotation().toQuaternion();
        geometry_msgs::msg::Pose msg;
        msg.position.x = pose.x();
        msg.position.y = pose.y();
        msg.position.z = pose.z();
        msg.orientation.w = q.w();
        msg.orientation.x = q.x();
        msg.orientation.y = q.y();
        msg.orientation.z = q.z();
        return msg;
    }
} // namespace Spacetime::Interface::Ros2
