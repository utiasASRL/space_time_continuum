#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using PriorType = Priors::WNOA::Prior<Pose>;

        std::shared_ptr<PriorType> makePrior(std::shared_ptr<StateType> &state)
        {
            state = std::make_shared<StateType>(1.0, 3);
            ValueType reference;
            reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
            return std::make_shared<PriorType>(state, reference);
        }

        std::string writeFile(const std::filesystem::path &path, const std::string &contents)
        {
            std::ofstream(path) << contents;
            return path.string();
        }
    } // namespace

    // Sensor types the core does not know (e.g. the "lidar" files of the ToF datasets, which
    // the rss2026_tof project loads itself) are skipped; the known ones still load, with or
    // without the optional "ground truth:" header line.
    TEST(ConfigLoaderDemoData, UnknownSensorTypesAreSkipped)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "config_loader_unknown_sensor";
        std::filesystem::create_directories(dir);
        writeFile(dir / "pos1_tof0.csv", "sensor type:,lidar\ntimestamp (s),arclength (m),data\n0.0,0.5,0.3,0.0,0.1\n");
        writeFile(dir / "aurora1.csv",
                  "sensor type:,aurora\nground truth:,1\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0,0,1,0,0,0\n");
        writeFile(dir / "aurora2.csv",
                  "sensor type:,aurora\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0,0,1,0,0,0\n0.1,1.0,0,0,1,0,0,0\n");
        const std::string config = writeFile(dir / "config.json", std::string(R"({
            "demo_data": { "folder_path": ")") + dir.string() + R"(", "start_time": 0, "end_time": 1, "include_aurora": true }
        })");

        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(config));
        std::shared_ptr<StateType> state;
        std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
        EXPECT_NO_THROW(loader.loadDemoData(sensors, gt_sensors, makePrior(state)));

        ASSERT_EQ(sensors.size(), 1u);
        ASSERT_EQ(gt_sensors.size(), 1u);
        EXPECT_NE(std::dynamic_pointer_cast<Sensors::PoseSensor<ValueType>>(sensors.front()), nullptr);
        EXPECT_EQ(sensors.front()->size(), 2u); // the 2-line header file keeps its first row
        EXPECT_EQ(gt_sensors.front()->size(), 1u);
        std::filesystem::remove_all(dir);
    }

    // Rows with too few values, and files without a "<label>,<type>" first line, are skipped
    // rather than read out of bounds.
    TEST(ConfigLoaderDemoData, MalformedRowsAndFilesAreSkipped)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "config_loader_malformed_rows";
        std::filesystem::create_directories(dir);
        writeFile(dir / "aurora1.csv",
                  "sensor type:,aurora\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0,0,1,0,0,0\n0.1,1.0,0,0\n0.2,1.0\n");
        writeFile(dir / "gyro1.csv",
                  "sensor type:,gyro\ntimestamp (s),arclength (m),data\n"
                  "0.0,1.0,0.1,0.2,0.3\n0.1,1.0,0.1\n");
        writeFile(dir / "no_type.csv", "just one cell\n");
        const std::string config = writeFile(dir / "config.json", std::string(R"({
            "demo_data": { "folder_path": ")") + dir.string() + R"(", "start_time": 0, "end_time": 1, "include_aurora": true, "include_gyro": true }
        })");

        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(config));
        std::shared_ptr<StateType> state;
        std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
        EXPECT_NO_THROW(loader.loadDemoData(sensors, gt_sensors, makePrior(state)));

        ASSERT_EQ(sensors.size(), 2u);
        for (const auto &sensor : sensors)
            EXPECT_EQ(sensor->size(), 1u); // only each file's complete row
        std::filesystem::remove_all(dir);
    }

    // A config built in memory (e.g. a file's sections with command-line overrides applied)
    // is added like a file: its sections are read, and top-level keys must not collide.
    TEST(ConfigLoaderDemoData, AddConfigReadsInMemorySections)
    {
        Json::Value config(Json::objectValue);
        config["demo_data"]["robot_length"] = 0.5;
        config["demo_data"]["robot_N"] = 4;

        Utilities::ConfigLoader loader;
        loader.addConfig(config);
        double length = 0.0;
        unsigned int N = 0;
        loader.loadDemoMetadata(length, N);
        EXPECT_DOUBLE_EQ(length, 0.5);
        EXPECT_EQ(N, 4u);

        Json::Value colliding(Json::objectValue);
        colliding["demo_data"]["robot_N"] = 5;
        EXPECT_THROW(loader.addConfig(colliding), std::runtime_error);
        EXPECT_THROW(loader.addConfig(Json::Value(Json::arrayValue)), std::invalid_argument);
    }
} // namespace Spacetime
