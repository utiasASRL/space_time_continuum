#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/interface/ros2/Conversions.hpp>

namespace Spacetime
{
    namespace
    {
        const gtsam::Pose3 kPose(gtsam::Rot3::RzRyRx(0.3, -0.7, 1.9), gtsam::Point3(0.1, -2.0, 3.5));
    }

    TEST(Ros2Conversions, TransformRoundTrip)
    {
        const geometry_msgs::msg::Transform msg = Ros2::toTransformMsg(kPose);
        EXPECT_DOUBLE_EQ(msg.translation.x, 0.1);
        EXPECT_DOUBLE_EQ(msg.translation.y, -2.0);
        EXPECT_DOUBLE_EQ(msg.translation.z, 3.5);
        EXPECT_TRUE(Ros2::toPose3(msg).equals(kPose, 1e-12));
    }

    TEST(Ros2Conversions, PoseRoundTrip)
    {
        const geometry_msgs::msg::Pose msg = Ros2::toPoseMsg(kPose);
        EXPECT_DOUBLE_EQ(msg.position.x, 0.1);
        EXPECT_DOUBLE_EQ(msg.position.y, -2.0);
        EXPECT_DOUBLE_EQ(msg.position.z, 3.5);
        EXPECT_TRUE(Ros2::toPose3(msg).equals(kPose, 1e-12));
    }

    TEST(Ros2Conversions, QuaternionMatchesRotation)
    {
        const geometry_msgs::msg::Transform msg = Ros2::toTransformMsg(kPose);
        const gtsam::Quaternion q = kPose.rotation().toQuaternion();
        // Either sign is the same rotation.
        const double sign = (q.w() * msg.rotation.w >= 0.0) ? 1.0 : -1.0;
        EXPECT_NEAR(msg.rotation.w, sign * q.w(), 1e-12);
        EXPECT_NEAR(msg.rotation.x, sign * q.x(), 1e-12);
        EXPECT_NEAR(msg.rotation.y, sign * q.y(), 1e-12);
        EXPECT_NEAR(msg.rotation.z, sign * q.z(), 1e-12);
    }

    TEST(Ros2Conversions, UnnormalizedQuaternionIsNormalized)
    {
        geometry_msgs::msg::Transform msg;
        msg.rotation.w = 0.0;
        msg.rotation.x = 0.0;
        msg.rotation.y = 0.0;
        msg.rotation.z = 2.0; // 180 degrees about z, scaled
        const gtsam::Pose3 pose = Ros2::toPose3(msg);
        EXPECT_TRUE(pose.rotation().equals(gtsam::Rot3::Rz(M_PI), 1e-12));
    }

    TEST(Ros2Conversions, ZeroQuaternionThrows)
    {
        geometry_msgs::msg::Transform msg;
        msg.rotation.w = 0.0;
        EXPECT_THROW(Ros2::toPose3(msg), std::invalid_argument);
        geometry_msgs::msg::Pose pose;
        pose.orientation.w = 0.0;
        EXPECT_THROW(Ros2::toPose3(pose), std::invalid_argument);
    }

} // namespace Spacetime
