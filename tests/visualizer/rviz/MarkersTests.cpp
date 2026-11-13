#include <cmath>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <spacetime/estimator/utilities/Ellipsoid.hpp>
#include <spacetime/visualizer/rviz/Markers.hpp>

namespace Spacetime
{
    namespace Rviz = Visualization::Rviz;

    namespace
    {
        Eigen::Quaterniond orientation(const visualization_msgs::msg::Marker &marker)
        {
            return Eigen::Quaterniond(marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z);
        }
    } // namespace

    // rviz's CYLINDER is a unit-diameter, unit-height cylinder along its local z.
    TEST(RvizMarkers, CylinderSpansTheSegment)
    {
        visualization_msgs::msg::Marker marker;
        const Eigen::Vector3d p0(1.0, 2.0, 3.0), p1(1.0, 2.0 + 0.6, 3.0 + 0.8);

        Rviz::setCylinderBetween(marker, p0, p1, 0.05);

        EXPECT_EQ(marker.type, visualization_msgs::msg::Marker::CYLINDER);
        EXPECT_DOUBLE_EQ(marker.scale.x, 0.1);
        EXPECT_DOUBLE_EQ(marker.scale.y, 0.1);
        EXPECT_NEAR(marker.scale.z, 1.0, 1e-12);
        EXPECT_NEAR(marker.pose.position.x, 1.0, 1e-12);
        EXPECT_NEAR(marker.pose.position.y, 2.3, 1e-12);
        EXPECT_NEAR(marker.pose.position.z, 3.4, 1e-12);
        const Eigen::Vector3d axis = orientation(marker).toRotationMatrix().col(2);
        EXPECT_TRUE(axis.isApprox((p1 - p0).normalized(), 1e-12));
    }

    TEST(RvizMarkers, DegenerateCylinderIsAFlatDisc)
    {
        visualization_msgs::msg::Marker marker;
        const Eigen::Vector3d p(0.5, 0.5, 0.5);

        Rviz::setCylinderBetween(marker, p, p, 0.05);

        EXPECT_DOUBLE_EQ(marker.scale.z, 0.0);
        EXPECT_TRUE(orientation(marker).isApprox(Eigen::Quaterniond::Identity()));
        EXPECT_DOUBLE_EQ(marker.pose.position.x, 0.5);
    }

    // rviz's SPHERE is unit-diameter: the scale holds the ellipsoid's full axes.
    TEST(RvizMarkers, EllipsoidMatchesTheConfidenceEllipsoid)
    {
        visualization_msgs::msg::Marker marker;
        const Eigen::Vector3d position(0.1, 0.2, 0.3);
        Eigen::Matrix3d covariance = Eigen::Vector3d(4e-4, 1e-4, 9e-4).asDiagonal();
        const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
        covariance = R * covariance * R.transpose();

        Rviz::setEllipsoid(marker, position, covariance);

        EXPECT_EQ(marker.type, visualization_msgs::msg::Marker::SPHERE);
        EXPECT_NEAR(marker.pose.position.x, 0.1, 1e-12);
        EXPECT_NEAR(marker.pose.position.y, 0.2, 1e-12);
        EXPECT_NEAR(marker.pose.position.z, 0.3, 1e-12);

        // The marker's (scaled) axes reproduce the shared ellipsoid transform.
        const Eigen::Matrix4d expected = Utilities::ellipsoidTransform(position, covariance);
        const Eigen::Matrix3d axes = orientation(marker).toRotationMatrix() *
                                     Eigen::Vector3d(marker.scale.x, marker.scale.y, marker.scale.z).asDiagonal() * 0.5;
        EXPECT_TRUE((axes * axes.transpose()).isApprox(expected.block<3, 3>(0, 0) * expected.block<3, 3>(0, 0).transpose(), 1e-9));
        EXPECT_NEAR(marker.scale.z, 2.0 * Utilities::kEllipsoidScale99_7 * 0.03, 1e-9); // largest axis, sqrt(9e-4)
    }
} // namespace Spacetime
