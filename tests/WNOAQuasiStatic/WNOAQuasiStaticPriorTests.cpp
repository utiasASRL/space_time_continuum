#include <memory>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/WnoaFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using ValuesType = Values<Pose>;
        using PriorType = Prior<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Matrix12 = Eigen::Matrix<double, 12, 12>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Vector12 = Eigen::Matrix<double, 12, 1>;
        using FactorVector = std::vector<std::shared_ptr<gtsam::NonlinearFactor>>;
        using StrainPrior = gtsam::PriorFactor<Vector6>;

        const Matrix12 Q0 = (Vector12() << 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6).finished().asDiagonal();
        const Matrix6 Q2 = Vector6(0.6, 0.5, 0.4, 0.3, 0.2, 0.1).asDiagonal();
        const Vector6 epsilon0 = (Vector6() << 0.0, 0.3, 0.0, 0.0, 0.0, 1.0).finished(); // bent in y, along z
        const Pose T0(gtsam::Rot3::Ry(0.2), gtsam::Point3(0.0, 0.0, 0.1));

        /// A 3-node, unit-length robot (ds = 0.5) under the quasi-static prior.
        struct Fixture
        {
            KeyManager km;
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 3);
            std::shared_ptr<PriorType> prior = std::make_shared<PriorType>(state, ValuesType(T0, epsilon0), Q0, Q2);
            gtsam::Values values;

            FactorVector extend(double time)
            {
                gtsam::Values new_values;
                FactorVector factors = prior->extendState(km, time, values, new_values);
                values.insert(new_values);
                return factors;
            }

            std::shared_ptr<NodeType> node(unsigned int row, unsigned int n) const { return state->estimation_nodes.at(row * 3 + n); }
        };

        gtsam::Matrix covarianceOf(const std::shared_ptr<gtsam::NonlinearFactor> &factor)
        {
            auto noise_factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factor);
            return std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(noise_factor->noiseModel())->covariance();
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

    TEST(WNOAQuasiStaticPrior, EveryRowHasUnaryAndSpatialFactorsOnly)
    {
        Fixture f;
        for (const double t : {0.0, 0.1, 0.25})
        {
            const FactorVector factors = f.extend(t);
            ASSERT_EQ(factors.size(), 4u) << "t = " << t; // pose + strain unary, 2 spatial
            EXPECT_EQ(factorsOfType<gtsam::PriorFactor<Pose>>(factors).size(), 1u);
            EXPECT_EQ(factorsOfType<StrainPrior>(factors).size(), 1u);
            EXPECT_EQ(factorsOfType<gtsam::WnoaMotionFactor<Pose>>(factors).size(), 2u);
        }
        EXPECT_EQ(f.state->estimation_nodes.size(), 9u);
    }

    TEST(WNOAQuasiStaticPrior, NoFactorConnectsTwoRows)
    {
        Fixture f;
        f.extend(0.0);
        const FactorVector second = f.extend(0.1);

        std::set<gtsam::Key> second_row_keys;
        for (unsigned int n = 0; n < 3; n++)
            for (const gtsam::Key key : f.node(1, n)->keys())
                second_row_keys.insert(key);
        for (const auto &factor : second)
            for (const gtsam::Key key : factor->keys())
                EXPECT_TRUE(second_row_keys.count(key)) << "a second-row factor references a first-row key";
    }

    TEST(WNOAQuasiStaticPrior, UnaryFactorsHoldTheBaseNodeToTheReference)
    {
        Fixture f;
        const FactorVector factors = f.extend(0.0);

        const auto pose_prior = factorsOfType<gtsam::PriorFactor<Pose>>(factors).front();
        EXPECT_EQ(pose_prior->key(), f.node(0, 0)->poseKey());
        EXPECT_TRUE(pose_prior->prior().equals(T0));
        EXPECT_TRUE(covarianceOf(pose_prior).isApprox(Q0.topLeftCorner<6, 6>()));

        const auto strain_prior = factorsOfType<StrainPrior>(factors).front();
        EXPECT_EQ(strain_prior->key(), f.node(0, 0)->strainKey());
        EXPECT_TRUE(strain_prior->prior().isApprox(epsilon0));
        EXPECT_TRUE(covarianceOf(strain_prior).isApprox(Q0.bottomRightCorner<6, 6>()));
    }

    TEST(WNOAQuasiStaticPrior, SpatialFactorsMatchTheWNOAPriorsGtsamFactors)
    {
        // The quasi-static prior is the WNOA prior's arclength dimension alone: its spatial
        // factors are the "wnoa" factor type's (pose, strain) WnoaMotionFactor.
        Fixture f;
        const auto spatial = factorsOfType<gtsam::WnoaMotionFactor<Pose>>(f.extend(0.0));
        ASSERT_EQ(spatial.size(), 2u);

        for (unsigned int n = 0; n < 2; n++)
        {
            const gtsam::WnoaMotionFactor<Pose> expected(
                f.node(0, n)->poseKey(), f.node(0, n)->strainKey(),
                f.node(0, n + 1)->poseKey(), f.node(0, n + 1)->strainKey(), 0.5, Q2.diagonal());
            EXPECT_TRUE(spatial[n]->equals(expected, 1e-12)) << "factor " << n;
        }
    }

    TEST(WNOAQuasiStaticPrior, InitialGuessIsTheZeroErrorConstantStrainChain)
    {
        Fixture f;
        FactorVector factors = f.extend(0.0);
        const FactorVector second = f.extend(0.1);
        factors.insert(factors.end(), second.begin(), second.end());

        for (unsigned int row = 0; row < 2; row++)
            for (unsigned int n = 0; n < 3; n++)
            {
                const ValuesType v = f.node(row, n)->getValues(f.values);
                EXPECT_TRUE(v.pose().equals(T0 * Pose::Expmap((0.5 * n) * epsilon0), 1e-12)) << "row " << row << " node " << n;
                EXPECT_TRUE(v.epsilon().isApprox(epsilon0));
            }
        for (const auto &factor : factors)
            EXPECT_NEAR(factor->error(f.values), 0.0, 1e-18);
    }

    TEST(WNOAQuasiStaticPrior, ExtendingAtOrBeforeTheLastRowAddsNothing)
    {
        Fixture f;
        f.extend(0.1);
        EXPECT_TRUE(f.extend(0.1).empty());
        EXPECT_TRUE(f.extend(0.05).empty());
        EXPECT_EQ(f.state->estimation_nodes.size(), 3u);
    }

    TEST(WNOAQuasiStaticPrior, MatchesTheWNOAPriorsSpatialSliceInABatchSolve)
    {
        // The same pose/strain unary weights and Q2 give the same spatial factors as the
        // WNOA prior with factor_type "wnoa"; the quasi-static unary is that prior's pose
        // and strain unaries (its velocity one dropped).
        auto wnoa_state = std::make_shared<Types::ContinuumRobotState<WNOA::Node<Pose>>>(1.0, 3);
        WNOA::Values<Pose> reference;
        reference.pose(T0);
        reference.epsilon(epsilon0);
        Eigen::Matrix<double, 18, 18> wnoa_Q0 = Eigen::Matrix<double, 18, 18>::Identity();
        wnoa_Q0.topLeftCorner<12, 12>() = Q0;
        WNOA::PriorOptions wnoa_options;
        wnoa_options.factor_type = WNOA::FactorType::WNOA;
        WNOA::Prior<Pose> wnoa(wnoa_state, reference, wnoa_Q0, Matrix6::Identity(), Q2, Matrix6::Identity(), Matrix6::Identity(), wnoa_options);
        KeyManager km;
        gtsam::Values wnoa_values, wnoa_new;
        const FactorVector wnoa_factors = wnoa.extendState(km, 0.0, wnoa_values, wnoa_new);

        Fixture f;
        const FactorVector factors = f.extend(0.0);
        EXPECT_EQ(factorsOfType<gtsam::WnoaMotionFactor<Pose>>(wnoa_factors).size(),
                  factorsOfType<gtsam::WnoaMotionFactor<Pose>>(factors).size());
        const auto wnoa_pose_prior = factorsOfType<gtsam::PriorFactor<Pose>>(wnoa_factors).front();
        const auto pose_prior = factorsOfType<gtsam::PriorFactor<Pose>>(factors).front();
        EXPECT_TRUE(covarianceOf(pose_prior).isApprox(covarianceOf(wnoa_pose_prior)));
    }
} // namespace Spacetime::Priors::WNOAQuasiStatic
