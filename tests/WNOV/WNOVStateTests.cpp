#include <stdexcept>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>

namespace Spacetime::Priors::WNOV
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using ValuesType = Values<Pose>;
        using Keys = InterpolatedKeys<Pose>;
        using TangentVector = gtsam::traits<Pose>::TangentVector;
    }

    // The node type must plug into every generic interface the estimator relies on.
    static_assert(Types::ValuesType<ValuesType>);
    static_assert(Types::NodeType<NodeType>);
    static_assert(Types::InterpolatedKeyMap<Keys, ValuesType>);
    static_assert(Sensors::PoseSensorValuesType<ValuesType>, "a PoseSensor must accept WNOV values");
    static_assert(!Sensors::GyroSensorValuesType<ValuesType>, "a WNOV node has no velocity, so a GyroSensor must not accept it");

    TEST(WNOVState, ValuesArePoseOnly)
    {
        EXPECT_EQ(ValuesType::dim, 6u);
        EXPECT_EQ(NodeType::dim, 6u);

        ValuesType values;
        EXPECT_EQ(values.values.size(), 1u);
        EXPECT_TRUE(values.pose().equals(Pose()));

        const Pose pose(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3), gtsam::Point3(1.0, 2.0, 3.0));
        EXPECT_TRUE(ValuesType(pose).pose().equals(pose));

        values.setRandom();
        EXPECT_FALSE(values.pose().equals(Pose()));
    }

    TEST(WNOVState, ValuesAssignment)
    {
        const ValuesType source(Pose(gtsam::Rot3::Rz(0.5), gtsam::Point3(1.0, 0.0, 0.0)));
        ValuesType target;
        target = source;
        EXPECT_TRUE(target.pose().equals(source.pose()));

        Types::NodeValues &target_base = target;
        const WNOA::Values<Pose> other_type;
        EXPECT_THROW(target_base = other_type, std::runtime_error);
    }

    TEST(WNOVState, NodeRegistersOnlyAPose)
    {
        KeyManager km;
        const NodeType node(km, 0.25, 1.5);
        EXPECT_DOUBLE_EQ(node.arclength, 0.25);
        EXPECT_DOUBLE_EQ(node.time, 1.5);
        ASSERT_EQ(node.names(), std::vector<std::string>{"pose"});
        ASSERT_EQ(node.keys().size(), 1u);
        EXPECT_EQ(node.keys().front(), node.poseKey());
        EXPECT_EQ(gtsam::Symbol(node.poseKey()).chr(), 'p');

        const auto interpolated = node.interpolatedKeys();
        ASSERT_EQ(interpolated.size(), 1u);
        EXPECT_EQ(interpolated.front().first, Keys::Pose());
        EXPECT_EQ(interpolated.front().second, node.poseKey());
    }

    TEST(WNOVState, NodeValuesRoundTrip)
    {
        KeyManager km;
        const NodeType node(km, 0.0, 0.0);
        const Pose pose(gtsam::Rot3::RzRyRx(0.3, 0.2, 0.1), gtsam::Point3(0.1, 0.2, 0.3));

        gtsam::Values values;
        node.setValues(values, ValuesType(pose));
        EXPECT_TRUE(node.getValues(values).pose().equals(pose));

        auto base = std::dynamic_pointer_cast<ValuesType>(node.getValuesBase(values));
        ASSERT_NE(base, nullptr);
        EXPECT_TRUE(base->pose().equals(pose));

        // A second insert of the same key is a programming error.
        EXPECT_ANY_THROW(node.setValues(values, ValuesType(pose)));
    }

    TEST(WNOVState, InterpolatedKeysLayout)
    {
        EXPECT_EQ(Keys::keys(), gtsam::KeyVector{Keys::Pose()});
        EXPECT_EQ(Keys::rowRange(Keys::Pose()), std::make_pair(0, 6));

        const Pose pose(gtsam::Rot3::Ry(0.4), gtsam::Point3(0.0, 1.0, 0.0));
        gtsam::Values values;
        Keys::insert(Keys::Pose(), ValuesType(pose), values);
        EXPECT_TRUE(values.at<Pose>(Keys::Pose()).equals(pose));

        const gtsam::Key unknown = gtsam::Symbol('q', 1);
        EXPECT_THROW(Keys::rowRange(unknown), std::runtime_error);
        EXPECT_THROW(Keys::insert(unknown, ValuesType(pose), values), std::runtime_error);
    }

    TEST(WNOVState, OnlyThePoseCanBeLocked)
    {
        Types::ContinuumRobotState<NodeType> state(1.0, 3);
        EXPECT_NO_THROW(state.lock("pose", 0.0, Pose(), 0.0));
        EXPECT_THROW(state.lock("strain", -1.0, TangentVector(TangentVector::Zero())), std::invalid_argument);
        EXPECT_THROW(state.lock("velocity", 0.0, TangentVector(TangentVector::Zero())), std::invalid_argument);
    }
} // namespace Spacetime::Priors::WNOV
