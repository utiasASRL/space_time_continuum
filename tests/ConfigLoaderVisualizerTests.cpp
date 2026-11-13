#include <stdexcept>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <json/json.h>

#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/visualizer/VisualizerOptions.hpp>

namespace Spacetime
{
    using Visualization::VisualizerOptions;

    namespace
    {
        /// A JSON array of the three entries of `v`.
        Json::Value array3(const Eigen::Vector3d &v)
        {
            Json::Value array(Json::arrayValue);
            for (int i = 0; i < 3; i++)
                array.append(v(i));
            return array;
        }

        /// A loader holding only `visualizer` as its "visualizer" section.
        Utilities::ConfigLoader loaderWith(const Json::Value &visualizer)
        {
            Json::Value config(Json::objectValue);
            config["visualizer"] = visualizer;
            Utilities::ConfigLoader loader;
            loader.addConfig(config);
            return loader;
        }
    } // namespace

    TEST(ConfigLoaderVisualizer, MissingSectionKeepsTheDefaults)
    {
        const VisualizerOptions options = Utilities::ConfigLoader().loadVisualizerOptions();
        const VisualizerOptions defaults;
        EXPECT_TRUE(options.cameras.empty());
        EXPECT_DOUBLE_EQ(options.robot.radius, defaults.robot.radius);
    }

    TEST(ConfigLoaderVisualizer, ReadsEveryCameraInOrder)
    {
        Json::Value visualizer(Json::objectValue);
        Json::Value first(Json::objectValue);
        first["position"] = array3({0, 1, 0.233});
        first["look_at"] = array3({0, 0, 0.233});
        first["up"] = array3({0, 0, -1});
        Json::Value second(Json::objectValue);
        second["position"] = array3({1, 2, 3});
        second["look_at"] = array3({4, 5, 6});
        second["up"] = array3({0, 1, 0});
        visualizer["cameras"].append(first);
        visualizer["cameras"].append(second);

        const VisualizerOptions options = loaderWith(visualizer).loadVisualizerOptions();
        ASSERT_EQ(options.cameras.size(), 2u);
        EXPECT_TRUE(options.cameras[0].position.isApprox(Eigen::Vector3d(0, 1, 0.233)));
        EXPECT_TRUE(options.cameras[0].focal_point.isApprox(Eigen::Vector3d(0, 0, 0.233)));
        EXPECT_TRUE(options.cameras[0].view_up.isApprox(Eigen::Vector3d(0, 0, -1)));
        EXPECT_TRUE(options.cameras[1].position.isApprox(Eigen::Vector3d(1, 2, 3)));
        EXPECT_TRUE(options.cameras[1].focal_point.isApprox(Eigen::Vector3d(4, 5, 6)));
        EXPECT_TRUE(options.cameras[1].view_up.isApprox(Eigen::Vector3d(0, 1, 0)));
    }

    TEST(ConfigLoaderVisualizer, MissingCameraEntriesKeepTheirDefaults)
    {
        Json::Value visualizer(Json::objectValue);
        Json::Value camera(Json::objectValue);
        camera["position"] = array3({1, 2, 3});
        visualizer["cameras"].append(camera);

        const VisualizerOptions options = loaderWith(visualizer).loadVisualizerOptions();
        const VisualizerOptions::CameraOptions defaults;
        ASSERT_EQ(options.cameras.size(), 1u);
        EXPECT_TRUE(options.cameras[0].position.isApprox(Eigen::Vector3d(1, 2, 3)));
        EXPECT_TRUE(options.cameras[0].focal_point.isApprox(defaults.focal_point));
        EXPECT_TRUE(options.cameras[0].view_up.isApprox(defaults.view_up));
    }

    TEST(ConfigLoaderVisualizer, EmptyCameraListGivesTheDefaultView)
    {
        Json::Value visualizer(Json::objectValue);
        visualizer["cameras"] = Json::Value(Json::arrayValue);
        EXPECT_TRUE(loaderWith(visualizer).loadVisualizerOptions().cameras.empty());
    }

    TEST(ConfigLoaderVisualizer, MalformedCamerasAreRejected)
    {
        Json::Value not_a_list(Json::objectValue);
        not_a_list["cameras"] = Json::Value(Json::objectValue);
        EXPECT_THROW(loaderWith(not_a_list).loadVisualizerOptions(), std::invalid_argument);

        Json::Value not_an_object(Json::objectValue);
        not_an_object["cameras"].append(1.0);
        EXPECT_THROW(loaderWith(not_an_object).loadVisualizerOptions(), std::invalid_argument);

        Json::Value short_vector(Json::objectValue);
        Json::Value camera(Json::objectValue);
        camera["up"] = Json::Value(Json::arrayValue);
        camera["up"].append(0.0);
        camera["up"].append(1.0);
        short_vector["cameras"].append(camera);
        EXPECT_THROW(loaderWith(short_vector).loadVisualizerOptions(), std::invalid_argument);
    }

    TEST(ConfigLoaderVisualizer, ReadsTheRobotRadius)
    {
        Json::Value visualizer(Json::objectValue);
        visualizer["robot"]["radius"] = 0.025;
        visualizer["robot"]["show_frames"] = false; // other legacy robot entries are ignored
        EXPECT_DOUBLE_EQ(loaderWith(visualizer).loadVisualizerOptions().robot.radius, 0.025);
    }

    TEST(ConfigLoaderVisualizer, NonPositiveRobotRadiusIsRejected)
    {
        for (const double radius : {0.0, -0.01})
        {
            Json::Value visualizer(Json::objectValue);
            visualizer["robot"]["radius"] = radius;
            EXPECT_THROW(loaderWith(visualizer).loadVisualizerOptions(), std::invalid_argument);
        }
    }
} // namespace Spacetime
