#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime::Priors::WNOV
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using ValuesType = Values<Pose>;
        using PriorType = Prior<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using FactorVector = std::vector<std::shared_ptr<gtsam::NonlinearFactor>>;

        const Matrix6 Q0 = Vector6(1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3).asDiagonal();
        const Matrix6 Q1 = Vector6(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).asDiagonal();
        const Matrix6 Q2 = Vector6(0.6, 0.5, 0.4, 0.3, 0.2, 0.1).asDiagonal();
        const Vector6 epsilon0 = (Vector6() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished(); // straight along z
        const Pose T0(gtsam::Rot3::Ry(0.2), gtsam::Point3(0.0, 0.0, 0.1));

        /// A 3-node, unit-length robot (ds = 0.5) under the WNOV prior.
        struct Fixture
        {
            KeyManager km;
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 3);
            std::shared_ptr<PriorType> prior = std::make_shared<PriorType>(state, T0, epsilon0, Q0, Q1, Q2);
            gtsam::Values values;

            FactorVector extend(double time)
            {
                gtsam::Values new_values;
                FactorVector factors = prior->extendState(km, time, values, new_values);
                values.insert(new_values);
                return factors;
            }
        };

        Matrix6 covarianceOf(const std::shared_ptr<gtsam::NonlinearFactor> &factor)
        {
            auto noise_factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factor);
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(noise_factor->noiseModel());
            return gaussian->covariance();
        }

        template <typename FactorT>
        std::vector<std::shared_ptr<FactorT>> factorsOfType(const FactorVector &factors)
        {
            std::vector<std::shared_ptr<FactorT>> out;
            for (const auto &factor : factors)
                if (auto typed = std::dynamic_pointer_cast<FactorT>(factor))
                    out.push_back(typed);
            return out;
        }
    }

    TEST(WNOVPrior, FirstRowHasAUnaryAndSpatialFactors)
    {
        Fixture f;
        const FactorVector factors = f.extend(0.0);
        ASSERT_EQ(factors.size(), 3u);
        ASSERT_EQ(f.state->estimation_nodes.size(), 3u);

        const auto unary = factorsOfType<gtsam::PriorFactor<Pose>>(factors);
        ASSERT_EQ(unary.size(), 1u);
        EXPECT_EQ(unary.front()->key(), f.state->estimation_nodes[0]->poseKey());
        EXPECT_TRUE(unary.front()->prior().equals(T0));
        EXPECT_TRUE(covarianceOf(unary.front()).isApprox(Q0));

        const auto spatial = factorsOfType<gtsam::BetweenFactor<Pose>>(factors);
        ASSERT_EQ(spatial.size(), 2u);
        for (unsigned int n = 0; n < 2; n++)
        {
            EXPECT_EQ(spatial[n]->key1(), f.state->estimation_nodes[n]->poseKey());
            EXPECT_EQ(spatial[n]->key2(), f.state->estimation_nodes[n + 1]->poseKey());
            EXPECT_TRUE(spatial[n]->measured().equals(Pose::Expmap(0.5 * epsilon0)));
            EXPECT_TRUE(covarianceOf(spatial[n]).isApprox(0.5 * Q1));
        }
    }

    TEST(WNOVPrior, FirstRowIsInitializedAsAStraightRobot)
    {
        Fixture f;
        f.extend(0.0);
        for (unsigned int n = 0; n < 3; n++)
        {
            const Pose expected = T0 * Pose::Expmap((0.5 * n) * epsilon0);
            EXPECT_TRUE(f.state->estimation_nodes[n]->getValues(f.values).pose().equals(expected, 1e-12)) << "node " << n;
        }
    }

    TEST(WNOVPrior, LaterRowsAddTimeFactorsScaledByTheInterval)
    {
        Fixture f;
        f.extend(0.0);
        const FactorVector factors = f.extend(0.05);
        ASSERT_EQ(f.state->estimation_nodes.size(), 6u);
        ASSERT_EQ(factors.size(), 6u); // 1 unary + 2 spatial + 3 temporal

        EXPECT_EQ(factorsOfType<gtsam::PriorFactor<Pose>>(factors).size(), 1u);
        unsigned int temporal = 0;
        for (const auto &between : factorsOfType<gtsam::BetweenFactor<Pose>>(factors))
        {
            if (!between->measured().equals(Pose()))
                continue; // spatial
            temporal++;
            EXPECT_TRUE(covarianceOf(between).isApprox(0.05 * Q2));
        }
        EXPECT_EQ(temporal, 3u);
    }

    TEST(WNOVPrior, RemoveUnaryKeepsOnlyTheFirstRowsUnaryFactor)
    {
        EXPECT_FALSE(PriorOptions().remove_unary);
        PriorOptions options;
        options.remove_unary = true;
        Fixture kept, removed;
        removed.prior.reset(); // a state is managed by one prior at a time
        removed.prior = std::make_shared<PriorType>(removed.state, T0, epsilon0, Q0, Q1, Q2, options);

        for (const double t : {0.0, 0.1, 0.2})
        {
            const FactorVector with = kept.extend(t), without = removed.extend(t);
            const std::size_t expected_unary = t == 0.0 ? 1u : 0u;
            EXPECT_EQ(factorsOfType<gtsam::PriorFactor<Pose>>(without).size(), expected_unary) << "t = " << t;
            EXPECT_EQ(without.size(), with.size() - (1u - expected_unary)) << "t = " << t; // only the unary is dropped
        }
        // The base nodes are still initialized from T0.
        EXPECT_TRUE(removed.values.equals(kept.values, 1e-12));
    }

    TEST(WNOVPrior, EveryFactorHasZeroErrorAtTheInitialValues)
    {
        Fixture f;
        FactorVector factors = f.extend(0.0);
        const FactorVector second = f.extend(0.1);
        factors.insert(factors.end(), second.begin(), second.end());
        for (const auto &factor : factors)
            EXPECT_NEAR(factor->error(f.values), 0.0, 1e-18);
    }

    TEST(WNOVPrior, InitializationAveragesTheTemporalAndSpatialPredictions)
    {
        Fixture f;
        f.extend(0.0);
        // Move the previous row off the prior mean: each new node starts halfway between
        // the pose it had (time prediction) and its spatial prediction.
        const Pose moved = Pose(gtsam::Rot3::Rx(0.3), gtsam::Point3(0.2, 0.0, 0.0));
        gtsam::Values moved_values;
        for (unsigned int n = 0; n < 3; n++)
            moved_values.insert(f.state->estimation_nodes[n]->poseKey(), moved * f.values.at<Pose>(f.state->estimation_nodes[n]->poseKey()));
        f.values = moved_values;
        f.extend(0.1);

        const Pose base = f.state->estimation_nodes[3]->getValues(f.values).pose();
        EXPECT_TRUE(base.equals(gtsam::interpolate<Pose>(moved * T0, T0, 0.5), 1e-12));
        const Pose second = f.state->estimation_nodes[4]->getValues(f.values).pose();
        const Pose previous = moved_values.at<Pose>(f.state->estimation_nodes[1]->poseKey());
        EXPECT_TRUE(second.equals(gtsam::interpolate<Pose>(previous, base * Pose::Expmap(0.5 * epsilon0), 0.5), 1e-12));
    }

    TEST(WNOVPrior, ExtendStateIsIdempotent)
    {
        Fixture f;
        f.extend(0.0);
        f.extend(0.1);
        EXPECT_TRUE(f.extend(0.1).empty());
        EXPECT_TRUE(f.extend(0.05).empty());
        EXPECT_EQ(f.state->estimation_nodes.size(), 6u);
    }

    TEST(WNOVPrior, CovarianceKeyGroupsAreGridCells)
    {
        Fixture f;
        f.extend(0.0);
        EXPECT_EQ(f.prior->covarianceKeyGroups(f.values).size(), 2u); // lone row: arclength pairs

        f.extend(0.1);
        const auto groups = f.prior->covarianceKeyGroups(f.values);
        ASSERT_EQ(groups.size(), 2u); // spatial_dim - 1 cells
        for (const auto &group : groups)
            EXPECT_EQ(group.size(), 4u); // one pose key per corner
    }

    TEST(WNOVPrior, MeasurementFactorsWithinTheExtentOnly)
    {
        Fixture f;
        f.extend(0.0);
        f.extend(0.1);
        auto inner = std::make_shared<Sensors::PoseSensor<ValuesType>::Factor>(
            gtsam::noiseModel::Gaussian::Covariance(0.01 * Matrix6::Identity()), Pose(), Eigen::MatrixXi::Ones(6, 1));

        auto factor = f.prior->getOuterFactor(0.05, 0.25, inner);
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(factor->keys().size(), 4u);
        EXPECT_EQ(f.prior->getOuterFactor(-0.05, 0.25, inner), nullptr);
        EXPECT_EQ(f.prior->getOuterFactor(0.05, -0.25, inner), nullptr);
    }

    TEST(WNOVPrior, QueryReturnsPoseOnlyValues)
    {
        Fixture f;
        f.extend(0.0);
        f.extend(0.1);

        Eigen::MatrixXd conditional;
        auto queried = std::dynamic_pointer_cast<ValuesType>(f.prior->query(0.05, 0.25, f.values, &conditional));
        ASSERT_NE(queried, nullptr);
        EXPECT_EQ(conditional.rows(), 6);
        EXPECT_EQ(conditional.cols(), 6);

        // Extrapolating a query past the newest row holds the pose.
        auto extrapolated = std::dynamic_pointer_cast<ValuesType>(f.prior->query(0.3, 1.0, f.values));
        ASSERT_NE(extrapolated, nullptr);
        EXPECT_TRUE(extrapolated->pose().equals(f.state->estimation_nodes[5]->getValues(f.values).pose(), 1e-12));

        EXPECT_EQ(f.prior->query(-0.1, 0.0, f.values), nullptr);
        EXPECT_EQ(f.prior->query(0.05, 0.25, gtsam::Values()), nullptr);
    }
} // namespace Spacetime::Priors::WNOV
