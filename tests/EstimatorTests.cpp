#include <Eigen/Dense>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    using Pose = gtsam::Pose3;
    using NodeType = Priors::WNOA::Node<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using ValueType = Priors::WNOA::NodeValues<Pose>;

    TEST(Estimator, NonZeroStrainInitialization)
    {
        // 4 spatial node robot (3 spatial factors)
        unsigned int spatial_dim = 4;
        std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, spatial_dim);

        ValueType node_reference;
        node_reference.epsilon((Eigen::Vector<double, 6>() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished());
        std::shared_ptr<Priors::WNOA::Prior<Pose>> prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);

        Estimator estimator(Testing::estimatorOptions(0.1));
        estimator.addPrior(prior);

        estimator.estimate(0.0);

        gtsam::Values values = estimator.getCurrentValues();
        auto node0 = state->estimation_nodes[0];
        auto node_end = state->estimation_nodes.back();
        ASSERT_NE(node0, nullptr);
        ASSERT_NE(node_end, nullptr);

        Eigen::Vector<double, 6> expected_epsilon = (Eigen::Vector<double, 6>() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();
        ValueType node0_values = node0->getValues(values);
        EXPECT_TRUE(node0_values.pose().equals(Pose()));
        EXPECT_TRUE(node0_values.epsilon().isApprox(expected_epsilon));
        EXPECT_TRUE(node0_values.varpi().isZero());

        ValueType node_end_values = node_end->getValues(values);
        Pose expected_end_pose = Pose(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 1.0));
        EXPECT_TRUE(node_end_values.pose().equals(expected_end_pose));
        EXPECT_TRUE(node_end_values.epsilon().isApprox(expected_epsilon));
        EXPECT_TRUE(node_end_values.varpi().isZero());

        estimator.estimate(0.1);

        values = estimator.getCurrentValues();
        node0_values = node0->getValues(values);
        EXPECT_TRUE(node0_values.pose().equals(Pose()));
        EXPECT_TRUE(node0_values.epsilon().isApprox(expected_epsilon));
        EXPECT_TRUE(node0_values.varpi().isZero());

        node_end = state->estimation_nodes.back();
        ASSERT_NE(node_end, nullptr);
        node_end_values = node_end->getValues(values);
        EXPECT_TRUE(node_end_values.pose().equals(expected_end_pose));
        EXPECT_TRUE(node_end_values.epsilon().isApprox(expected_epsilon));
        EXPECT_TRUE(node_end_values.varpi().isZero());
    }

    TEST(Estimator, NonZeroVelocityInitialization)
    {

        // 4 spatial node robot (3 spatial factors)
        unsigned int spatial_dim = 4;
        std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, spatial_dim);

        ValueType node_reference;
        node_reference.varpi((Eigen::Vector<double, 6>() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished());
        std::shared_ptr<Priors::WNOA::Prior<Pose>> prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);

        gtsam::Values current_values, new_values;

        Estimator estimator(Testing::estimatorOptions(0.0));
        estimator.addPrior(prior);

        estimator.estimate(0.0);

        gtsam::Values values = estimator.getCurrentValues();
        auto node0 = state->estimation_nodes[0];
        auto node_end = state->estimation_nodes.back();
        ASSERT_NE(node0, nullptr);
        ASSERT_NE(node_end, nullptr);

        Eigen::Vector<double, 6> expected_varpi = (Eigen::Vector<double, 6>() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();
        ValueType node0_values = node0->getValues(values);
        EXPECT_TRUE(node0_values.pose().equals(Pose()));
        // The default reference strain is straight (see Priors::straightStrain).
        const Eigen::Vector<double, 6> straight = Priors::straightStrain<Pose>();
        EXPECT_TRUE(node0_values.epsilon().isApprox(straight));
        EXPECT_TRUE(node0_values.varpi().isApprox(expected_varpi));

        ValueType node_end_values = node_end->getValues(values);
        EXPECT_TRUE(node_end_values.pose().equals(Pose(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), 1e-9));
        EXPECT_TRUE(node_end_values.epsilon().isApprox(straight));
        EXPECT_TRUE(node_end_values.varpi().isApprox(expected_varpi));
    }
}