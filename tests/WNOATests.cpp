#include <Eigen/Dense>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>

namespace Spacetime::Priors::WNOA
{
    using namespace Legacy;

    using Pose = gtsam::Pose3;
    using NodeType = Node<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using ValueType = WNOA::Values<Pose>; 

    TEST(WNOA, ImportTest)
    {
        EXPECT_TRUE(true); // This test just checks that the test file compiles and runs without errors. The actual tests are in the other TEST cases below
    }

    TEST(WNOA, UnaryFactor)
    {
        KeyManager key_manager;
        std::shared_ptr<NodeType> node0 = std::make_shared<NodeType>(key_manager, 0.0, 0.0);
        ValueType node_reference;
        const Eigen::Matrix<double, 3 * Pose::dimension, 3 * Pose::dimension> covariance = Eigen::Matrix<double, 3 * Pose::dimension, 3 * Pose::dimension>::Identity();
        std::shared_ptr<UnaryFactor::Factor<Pose>> factor = UnaryFactor::createFactor<Pose>(node0, node_reference, covariance);

        ValueType node0_values;
        gtsam::Vector error = factor->evaluateError(node0_values.pose(), node0_values.epsilon(), node0_values.varpi(), nullptr, nullptr, nullptr);
        EXPECT_TRUE(error.isZero());

        node0_values.setRandom();
        gtsam::Values values;
        node0->insert("pose", node0_values.pose(), values);
        node0->insert("strain", node0_values.epsilon(), values);
        node0->insert("velocity", node0_values.varpi(), values);
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOAUnaryFactor", *factor, values, 1e-6, 1e-5));
    }

    TEST(WNOA, BinarySpaceFactor)
    {
        KeyManager key_manager;
        std::shared_ptr<NodeType> node0 = std::make_shared<NodeType>(key_manager, 0.0, 0.0);
        std::shared_ptr<NodeType> node1 = std::make_shared<NodeType>(key_manager, 1.0, 0.0);
        
        std::shared_ptr<BinarySpaceFactor::Factor<Pose>> factor = BinarySpaceFactor::createFactor<Pose>(node0, node1, Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity());

        ValueType node0_values, node1_values;
        gtsam::Vector error = factor->evaluateError(node0_values.pose(), node0_values.epsilon(), node0_values.varpi(), node1_values.pose(), node1_values.epsilon(), node1_values.varpi(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        EXPECT_TRUE(error.isZero());

        node0_values.setRandom();
        node1_values.setRandom();

        gtsam::Values values;
        node0->insert("pose", node0_values.pose(), values);
        node0->insert("strain", node0_values.epsilon(), values);
        node0->insert("velocity", node0_values.varpi(), values);
        node1->insert("pose", node1_values.pose(), values);
        node1->insert("strain", node1_values.epsilon(), values);
        node1->insert("velocity", node1_values.varpi(), values);
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOABinarySpaceFactor", *factor, values, 1e-6, 1e-5));
    }

    TEST(WNOA, BinaryTimeFactor)
    {
        KeyManager key_manager;
        std::shared_ptr<NodeType> node0 = std::make_shared<NodeType>(key_manager, 0.0, 0.0);
        std::shared_ptr<NodeType> node1 = std::make_shared<NodeType>(key_manager, 0.0, 1.0);

        std::shared_ptr<BinaryTimeFactor::Factor<Pose>> factor = BinaryTimeFactor::createFactor<Pose>(node0, node1, Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity());

        ValueType node0_values, node1_values;
        gtsam::Vector error = factor->evaluateError(node0_values.pose(), node0_values.epsilon(), node0_values.varpi(), node1_values.pose(), node1_values.epsilon(), node1_values.varpi(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        EXPECT_TRUE(error.isZero());

        node0_values.setRandom();
        node1_values.setRandom();

        gtsam::Values values;
        node0->insert("pose", node0_values.pose(), values);
        node0->insert("strain", node0_values.epsilon(), values);
        node0->insert("velocity", node0_values.varpi(), values);
        node1->insert("pose", node1_values.pose(), values);
        node1->insert("strain", node1_values.epsilon(), values);
        node1->insert("velocity", node1_values.varpi(), values);
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOABinaryTimeFactor", *factor, values, 1e-6, 1e-5));
    }

    TEST(WNOA, PriorExtension)
    {
        // 4 spatial node robot (3 spatial factors)
        unsigned int spatial_dim = 4;
        auto state = std::make_shared<StateType>(1.0, spatial_dim);

        ValueType node_reference;
        node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
        std::shared_ptr<Prior<Pose>> prior = std::make_shared<Prior<Pose>>(state, node_reference);

        gtsam::Values current_values, new_values;
        KeyManager key_manager;

        // Initialize the state at time 0
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors0 = prior->extendState(key_manager, 0.0, current_values, new_values);
        EXPECT_EQ(state->estimation_nodes.size(), spatial_dim);
        // EXPECT_EQ(new_factors0.size(), spatial_dim); // spatial_dim - 1 spatial factors + 1 unary factor
        EXPECT_EQ(new_factors0.size(), (spatial_dim - 1) * 2 + 3); // spatial factors x 2, temporal factors x 2, and unary factors x3
        current_values.insert(new_values); // Update current values with the new values from the prior extension
        new_values.clear();

        // Every call adds exactly one row, at exactly the requested time -- there is no
        // grid: the estimation loop owns the spacing.
        const std::size_t factors_per_row = (spatial_dim - 1) * 2 + spatial_dim * 2 + 3; // spatial x2, temporal x2, unary x3
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors1 = prior->extendState(key_manager, 0.11, current_values, new_values);
        EXPECT_EQ(state->estimation_nodes.size(), 2 * spatial_dim);
        EXPECT_DOUBLE_EQ(state->estimation_nodes.back()->time, 0.11);
        EXPECT_EQ(new_factors1.size(), factors_per_row);
        current_values.insert(new_values); // Update current values with the new values from the prior extension
        new_values.clear();

        // A much larger gap still adds a single row, rather than back-filling intermediate
        // ones: the prior no longer owns the spacing, the estimation loop does.
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors2 = prior->extendState(key_manager, 0.51, current_values, new_values);
        EXPECT_EQ(state->estimation_nodes.size(), 3 * spatial_dim);
        EXPECT_DOUBLE_EQ(state->estimation_nodes.back()->time, 0.51);
        EXPECT_EQ(new_factors2.size(), factors_per_row);
        current_values.insert(new_values); // Update current values with the new values from the prior extension

        // A measurement at the newest node's time is resolved exactly on it -- no extrapolation.
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(Eigen::Matrix<double, 6, 6>::Identity());
        auto inner_factor = std::make_shared<gtsam::PriorFactor<Pose>>(
            Interpolator<Pose>::InterpolatedKeys::Pose(), Pose(), noise_model);
        auto factor = prior->getOuterFactor(0.51, 0.0, inner_factor);
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(factor->keys().size(), 3u);
    }

    TEST(WNOA, PriorFinerThanTheLastSpacing)
    {
        // 4 spatial node robot (3 spatial factors)
        unsigned int spatial_dim = 4;
        auto state = std::make_shared<StateType>(1.0, spatial_dim);

        ValueType node_reference;
        node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
        std::shared_ptr<Prior<Pose>> prior = std::make_shared<Prior<Pose>>(state, node_reference);

        gtsam::Values current_values, new_values;
        KeyManager key_manager;

        // Add one state at time 0.1
        prior->extendState(key_manager, 0.1, current_values, new_values);
        current_values.insert(new_values); // Update current values with the new values from the prior extension
        new_values.clear();

        // Querying at time 0.15, only 0.05 after the most recent node at 0.1, still adds a
        // node exactly at the requested time.
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors = prior->extendState(key_manager, 0.15, current_values, new_values);
        EXPECT_EQ(state->estimation_nodes.size(), 2 * spatial_dim);
        EXPECT_EQ(new_factors.size(), (spatial_dim - 1) * 2 + spatial_dim * 2 + 3); // spatial x2, temporal x2, unary x3
        EXPECT_DOUBLE_EQ(state->estimation_nodes.back()->time, 0.15);
        current_values.insert(new_values);
        new_values.clear();

        // Calling extendState again with the exact same time must not add a duplicate
        // node -- a zero-length interval would break the WnoaMotionFactor/interpolation math.
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> repeat_factors = prior->extendState(key_manager, 0.15, current_values, new_values);
        EXPECT_EQ(state->estimation_nodes.size(), 2 * spatial_dim); // No new nodes should be added
        EXPECT_EQ(repeat_factors.size(), 0); // No new factors should be added
    }

    TEST(WNOA, StateFailureCases)
    {
        // 4 spatial node robot (3 spatial factors)
        unsigned int spatial_dim = 4;
        auto state = std::make_shared<StateType>(1.0, spatial_dim);
        KeyManager key_manager;

        // Add one state at time 0.1
        state->addStateAtTime(key_manager, 0.1);

        auto next_time_node = state->getNextTemporalNode(state->estimation_nodes.at(0)); // Should return the node at time 0.1
        EXPECT_EQ(next_time_node, nullptr);

        auto next_space_node = state->getNextSpatialNode(state->estimation_nodes.at(0)); // Should return the node at arclength 1.0
        EXPECT_NE(next_space_node, nullptr);

        auto previous_time_node = state->getPreviousTemporalNode(state->estimation_nodes.at(0)); // Should return nullptr since there are no previous time steps
        EXPECT_EQ(previous_time_node, nullptr);

        auto previous_space_node = state->getPreviousSpatialNode(state->estimation_nodes.at(0)); // Should return nullptr since there are no previous spatial steps
        EXPECT_EQ(previous_space_node, nullptr);

        auto last_space_node_plus_one = state->getNextSpatialNode(state->estimation_nodes.at(spatial_dim - 1)); // Should return nullptr since there are only 4 spatial nodes
        EXPECT_EQ(last_space_node_plus_one, nullptr);
    }
}