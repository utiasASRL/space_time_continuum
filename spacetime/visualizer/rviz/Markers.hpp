/**
 * @file Markers.hpp
 * @brief Placing rviz2 marker primitives (visualization_msgs/Marker) for the shapes the
 *        snapshot publisher draws.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_RVIZ_MARKERS_HPP
#define SPACETIME_RVIZ_MARKERS_HPP

#include <Eigen/Core>
#include <visualization_msgs/msg/marker.hpp>

#include <spacetime/estimator/utilities/Ellipsoid.hpp>

namespace Spacetime::Visualization::Rviz
{
    /**
     * @brief Makes `marker` a CYLINDER (unit diameter and height along its local z, as rviz2
     *        draws it) spanning the segment `p0` -> `p1` with the given `radius`. A degenerate
     *        (zero-length) segment gives a flat disc at `p0`.
     */
    void setCylinderBetween(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &p0, const Eigen::Vector3d &p1, double radius);

    /// Makes `marker` a SPHERE drawing the confidence ellipsoid of a 3D Gaussian (see Spacetime::Utilities::ellipsoidTransform).
    void setEllipsoid(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &position, const Eigen::Matrix3d &covariance,
                      double scale = ::Spacetime::Utilities::kEllipsoidScale99_7);

    /// Makes `marker` a SPHERE of `radius` at `position`.
    void setSphere(visualization_msgs::msg::Marker &marker, const Eigen::Vector3d &position, double radius);
} // namespace Spacetime::Visualization::Rviz

#endif // SPACETIME_RVIZ_MARKERS_HPP
