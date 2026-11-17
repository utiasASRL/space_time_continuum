#include <memory>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>
#include <vtkActor.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>
#include <spacetime/visualizer/MeasurementHistory.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/objects/PoseSensor.hpp>
#include <spacetime/visualizer/default/objects/SensorViews.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using PriorType = Priors::WNOA::Prior<Pose>;

        std::shared_ptr<Types::MeasurementBase> makePoseMeasurement(double time, double z = 0.0)
        {
            Eigen::Matrix4d value = Eigen::Matrix4d::Identity();
            value(2, 3) = z;
            Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
            return std::make_shared<Types::MeasurementBase>(0.0, time, value, covariance, Eigen::MatrixXi::Ones(6, 1));
        }

        struct SensorRig
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 2);
            std::shared_ptr<PriorType> prior = std::make_shared<PriorType>(state, ValueType());
            std::shared_ptr<Sensors::PoseSensor<ValueType>> pose_sensor =
                std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
        };
    } // namespace

    TEST(MeasurementHistory, ReturnsNullBeforeAnyMeasurement)
    {
        SensorRig rig;
        Visualization::MeasurementHistory history(*rig.pose_sensor);
        EXPECT_EQ(history.latestMeasurementAt(1.0), nullptr);
        EXPECT_EQ(history.size(), 0u);
    }

    TEST(MeasurementHistory, FiltersFutureMeasurements)
    {
        // A ground-truth sensor is never attached to an estimator (getNewFactors is
        // never called on it), so this also verifies the history is recorded at
        // addMeasurement() time, not at factor-drain time.
        SensorRig rig;
        Visualization::MeasurementHistory history(*rig.pose_sensor);
        rig.pose_sensor->addMeasurement(makePoseMeasurement(0.0));
        rig.pose_sensor->addMeasurement(makePoseMeasurement(1.0));

        auto at_half = history.latestMeasurementAt(0.5);
        ASSERT_NE(at_half, nullptr);
        EXPECT_DOUBLE_EQ(at_half->time, 0.0);

        auto at_two = history.latestMeasurementAt(2.0);
        ASSERT_NE(at_two, nullptr);
        EXPECT_DOUBLE_EQ(at_two->time, 1.0);

        EXPECT_EQ(history.latestMeasurementAt(-1.0), nullptr);
    }

    TEST(MeasurementHistory, IncludesMeasurementsQueuedBeforeItWasCreated)
    {
        // The demo loads all sensor data before the visualizer (and so the history) exists.
        SensorRig rig;
        rig.pose_sensor->addMeasurement(makePoseMeasurement(0.0));
        Visualization::MeasurementHistory history(*rig.pose_sensor);
        ASSERT_NE(history.latestMeasurementAt(0.0), nullptr);
        EXPECT_EQ(history.size(), 1u);
    }

    TEST(MeasurementHistory, SurvivesEstimatorConsumption)
    {
        // Consuming measurements via getNewFactors (as the estimator does) drains
        // measurement_queue, but must not erase the separate history.
        SensorRig rig;
        Visualization::MeasurementHistory history(*rig.pose_sensor);
        rig.pose_sensor->addMeasurement(makePoseMeasurement(0.0));
        gtsam::Values new_values;
        rig.pose_sensor->getNewFactors(0.0, new_values);

        EXPECT_TRUE(rig.pose_sensor->empty());
        EXPECT_NE(history.latestMeasurementAt(0.0), nullptr);
    }

    TEST(MeasurementHistory, DropsTheOldestEntriesPastItsCapacity)
    {
        SensorRig rig;
        Visualization::MeasurementHistory history(*rig.pose_sensor, /*capacity=*/2);
        for (double time : {0.0, 1.0, 2.0})
            rig.pose_sensor->addMeasurement(makePoseMeasurement(time));

        EXPECT_EQ(history.size(), 2u);
        EXPECT_EQ(history.latestMeasurementAt(0.5), nullptr); // t = 0 was evicted
        ASSERT_NE(history.latestMeasurementAt(1.5), nullptr);
        EXPECT_DOUBLE_EQ(history.latestMeasurementAt(1.5)->time, 1.0);
    }

    TEST(MeasurementHistory, MeasurementsBetweenIsInclusiveAndOrdered)
    {
        SensorRig rig;
        Visualization::MeasurementHistory history(*rig.pose_sensor);
        for (double time : {0.0, 1.0, 2.0, 3.0})
            rig.pose_sensor->addMeasurement(makePoseMeasurement(time));

        const auto middle = history.measurementsBetween(1.0, 2.0);
        ASSERT_EQ(middle.size(), 2u);
        EXPECT_DOUBLE_EQ(middle[0]->time, 1.0);
        EXPECT_DOUBLE_EQ(middle[1]->time, 2.0);

        EXPECT_EQ(history.measurementsBetween(-10.0, 10.0).size(), 4u);
        EXPECT_TRUE(history.measurementsBetween(3.5, 10.0).empty());
        EXPECT_TRUE(history.measurementsBetween(2.0, 1.0).empty()); // reversed range
    }

    TEST(MeasurementHistory, SensorOutlivingTheHistoryCanStillAddMeasurements)
    {
        SensorRig rig;
        {
            Visualization::MeasurementHistory history(*rig.pose_sensor);
        }
        EXPECT_NO_THROW(rig.pose_sensor->addMeasurement(makePoseMeasurement(0.0)));
    }

    TEST(SensorViews, PoseSensorGetsAPoseSensorView)
    {
        SensorRig rig;
        auto view = Visualization::Default::makeSensorView<ValueType>(rig.pose_sensor);
        ASSERT_NE(std::dynamic_pointer_cast<Visualization::Default::PoseSensor<ValueType>>(view), nullptr);
        EXPECT_EQ(view->info(), "PoseSensor");
    }

    TEST(SensorViews, SensorWithoutAViewGetsANoOpView)
    {
        SensorRig rig;
        auto gyro = std::make_shared<Sensors::GyroSensor<ValueType>>(rig.prior);
        auto view = Visualization::Default::makeSensorView<ValueType>(gyro);
        ASSERT_NE(view, nullptr);
        EXPECT_EQ(std::dynamic_pointer_cast<Visualization::Default::PoseSensor<ValueType>>(view), nullptr);
        EXPECT_EQ(view->info(), std::string());
        EXPECT_NO_THROW(view->updateVisualization(0.0, 0.0));
    }

    TEST(SensorViews, PoseSensorViewDrawsTheLatestMeasurement)
    {
        SensorRig rig;
        rig.pose_sensor->addMeasurement(makePoseMeasurement(0.0, /*z=*/1.0));
        rig.pose_sensor->addMeasurement(makePoseMeasurement(1.0, /*z=*/2.0));
        Visualization::Default::PoseSensor<ValueType> view(rig.pose_sensor);
        view.mp_axes = vtkSmartPointer<vtkActor>::New(); // as visualize() would, without a render window

        view.updateVisualization(0.1, 0.1);
        EXPECT_TRUE(view.mp_axes->GetVisibility());
        EXPECT_DOUBLE_EQ(view.mp_axes->GetUserMatrix()->GetElement(2, 3), 1.0);

        view.updateVisualization(-1.0, -1.0); // before any measurement
        EXPECT_FALSE(view.mp_axes->GetVisibility());
    }

    TEST(SensorViews, NullSensorOrStateIsRejected)
    {
        EXPECT_THROW(Visualization::Default::PoseSensor<ValueType>(nullptr), std::invalid_argument);
        EXPECT_THROW(Visualization::Default::ContinuumRobot<NodeType>(nullptr), std::invalid_argument);
    }
} // namespace Spacetime
