#include <Eigen/Geometry>

#include "Markers.hpp"

namespace Spacetime::Visualization::Rviz
{
    namespace
    {
        void setPose(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &position, const Eigen::Quaterniond &orientation)
        {
            marker.pose.position.x = position.x();
            marker.pose.position.y = position.y();
            marker.pose.position.z = position.z();
            marker.pose.orientation.w = orientation.w();
            marker.pose.orientation.x = orientation.x();
            marker.pose.orientation.y = orientation.y();
            marker.pose.orientation.z = orientation.z();
        }
    } // namespace

    void setCylinderBetween(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, double radius)
    {
        const Eigen::Vector3d segment = p1 - p0;
        const double length = segment.norm();
        // Rotation taking the local z axis onto the segment (any rotation for a degenerate one).
        const Eigen::Quaterniond orientation = length > 0.0 ? Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), segment)
                                                            : Eigen::Quaterniond::Identity();
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        setPose(marker, 0.5 * (p0 + p1), orientation);
        marker.scale.x = 2.0 * radius;
        marker.scale.y = 2.0 * radius;
        marker.scale.z = length;
    }

    void setEllipsoid(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &position, const Eigen::Matrix3d &covariance, double scale)
    {
        // The transform's columns are the semi-axes: rviz2's unit-diameter sphere scales by the full axes.
        const Eigen::Matrix4d transform = ::Spacetime::Utilities::ellipsoidTransform(position, covariance, scale);
        Eigen::Matrix3d axes = transform.block<3, 3>(0, 0);
        const Eigen::Vector3d semi_axes = axes.colwise().norm();
        axes.array().rowwise() /= semi_axes.transpose().array();

        marker.type = visualization_msgs::msg::Marker::SPHERE;
        setPose(marker, position, Eigen::Quaterniond(axes).normalized());
        marker.scale.x = 2.0 * semi_axes.x();
        marker.scale.y = 2.0 * semi_axes.y();
        marker.scale.z = 2.0 * semi_axes.z();
    }

    void setSphere(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &position, double radius)
    {
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        setPose(marker, position, Eigen::Quaterniond::Identity());
        marker.scale.x = marker.scale.y = marker.scale.z = 2.0 * radius;
    }
} // namespace Spacetime::Visualization::Rviz
