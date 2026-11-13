#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

// The ROS2 interface's tests share one rclcpp context for the whole run.
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
