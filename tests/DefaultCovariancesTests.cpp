#include <memory>
#include <string>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/estimator/joint_models/EndEffectorPrior.hpp>
#include <spacetime/estimator/prior_models/StraightStrain.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAPrior.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/WNOAQuasiStaticPrior.hpp>
#include <spacetime/estimator/prior_models/WNOV/WNOVPrior.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector6 = gtsam::Vector6;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using WnoaPrior = Priors::WNOA::Prior<Pose>;
        using QuasiStaticPrior = Priors::WNOAQuasiStatic::Prior<Pose>;
        using WnovPrior = Priors::WNOV::Prior<Pose>;

        const std::string kPaperConfig = std::string(PROJECT_SOURCE_DIR) +
                                         "/papers/a-stochastic-framework-for-continuous-time-estimation-of-continuum-robots/configs/experiment_base.json";

        /// Whether constructing with `make` logs the tangent-axis warning.
        template <typename F>
        bool warnsAboutTangent(F make)
        {
            testing::internal::CaptureStdout();
            make();
            return testing::internal::GetCapturedStdout().find("tangent axis") != std::string::npos;
        }
    } // namespace

    TEST(DefaultCovariances, WnoaMatchesThePaperExperimentConfig)
    {
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(kPaperConfig));
        WnoaPrior::ValuesType reference;
        double time_period = 0.0;
        WnoaPrior::Matrix3N Q0;
        WnoaPrior::MatrixN Q1, Q2, Q3, Q4;
        loader.loadWNOAPrior<Pose>(reference, time_period, Q0, Q1, Q2, Q3, Q4);

        const auto [D0, D1, D2, D3, D4] = WnoaPrior::DefaultCovariances();
        EXPECT_TRUE(D1.isApprox(Q1));
        EXPECT_TRUE(D2.isApprox(Q2));
        EXPECT_TRUE(D3.isApprox(Q3));
        EXPECT_TRUE(D4.isApprox(Q4));

        // The paper's tangent axis is z; the library's is x: the rotational-strain diagonal
        // (Q0 rows 6..8) is permuted z -> x, x -> y, y -> z. Everything else is the same.
        WnoaPrior::Matrix3N expected = Q0;
        expected(6, 6) = Q0(8, 8);
        expected(7, 7) = Q0(6, 6);
        expected(8, 8) = Q0(7, 7);
        EXPECT_TRUE(D0.isApprox(expected));
        EXPECT_TRUE(D0.isDiagonal());
    }

    TEST(DefaultCovariances, QuasiStaticSharesTheWnoaTuning)
    {
        const auto [W0, W1, W2, W3, W4] = WnoaPrior::DefaultCovariances();
        const auto [Q0, Q2] = QuasiStaticPrior::DefaultCovariances();
        EXPECT_TRUE(Q0.isApprox(W0.topLeftCorner<12, 12>()));
        EXPECT_TRUE(Q2.isApprox(W2));
    }

    TEST(DefaultCovariances, WnovAndEndEffectorAreIdentity)
    {
        const auto [Q0, Q1, Q2] = WnovPrior::DefaultCovariances();
        EXPECT_TRUE(Q0.isIdentity());
        EXPECT_TRUE(Q1.isIdentity());
        EXPECT_TRUE(Q2.isIdentity());
        EXPECT_TRUE(Joints::EndEffectorPrior<Pose>::DefaultCovariances().isOnes());
    }

    TEST(DefaultCovariances, PriorsAcceptThem)
    {
        auto state = std::make_shared<WnoaPrior::StateType>(1.0, 3);
        const auto [Q0, Q1, Q2, Q3, Q4] = WnoaPrior::DefaultCovariances();
        EXPECT_NO_THROW(WnoaPrior(state, WnoaPrior::ValuesType(), Q0, Q1, Q2, Q3, Q4));
    }

    TEST(DefaultCovariances, ConstructorsWithoutCovariancesUseThem)
    {
        // Same factors, at the same values, as with the defaults passed explicitly.
        const auto errorAt = [](const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &values) { return graph.error(values); };
        {
            auto a = std::make_shared<WnoaPrior::StateType>(1.0, 3), b = std::make_shared<WnoaPrior::StateType>(1.0, 3);
            const auto [Q0, Q1, Q2, Q3, Q4] = WnoaPrior::DefaultCovariances();
            WnoaPrior implicit(a, WnoaPrior::ValuesType()), explicit_(b, WnoaPrior::ValuesType(), Q0, Q1, Q2, Q3, Q4);
            KeyManager km_a, km_b;
            gtsam::Values va, vb;
            gtsam::NonlinearFactorGraph ga, gb;
            for (double t : {0.0, 0.1})
            {
                for (auto &f : implicit.extendState(km_a, t, va, va))
                    ga.push_back(f);
                for (auto &f : explicit_.extendState(km_b, t, vb, vb))
                    gb.push_back(f);
            }
            gtsam::VectorValues delta = vb.zeroVectors();
            for (auto &[key, vector] : delta)
                vector = 0.05 * gtsam::Vector::Random(vector.size());
            const gtsam::Values perturbed = vb.retract(delta);
            EXPECT_EQ(ga.size(), gb.size());
            EXPECT_NEAR(errorAt(ga, perturbed), errorAt(gb, perturbed), 1e-9);
        }
        Priors::WNOA::PriorOptions options;
        options.remove_unary = true;
        EXPECT_TRUE(WnoaPrior(std::make_shared<WnoaPrior::StateType>(1.0, 3), WnoaPrior::ValuesType(), options).options().remove_unary);
        EXPECT_NO_THROW(QuasiStaticPrior(std::make_shared<QuasiStaticPrior::StateType>(1.0, 3), QuasiStaticPrior::ValuesType()));
        EXPECT_NO_THROW(WnovPrior(std::make_shared<WnovPrior::StateType>(1.0, 3), Pose(), Priors::straightStrain<Pose>()));
        EXPECT_NO_THROW(Joints::EndEffectorPrior<Pose>(std::make_shared<Joints::EndEffectorPrior<Pose>::StateType>(std::vector<double>{0.0}), Pose()));
    }

    TEST(StraightStrain, IsAUnitStrainAlongX)
    {
        EXPECT_TRUE(Priors::straightStrain<Pose>().isApprox((Vector6() << 0, 0, 0, 1, 0, 0).finished()));
        EXPECT_TRUE(Priors::straightStrain<gtsam::Pose2>().isApprox(gtsam::Vector3(1, 0, 0)));
        EXPECT_TRUE(Priors::WNOA::NodeValues<Pose>().epsilon().isApprox(Priors::straightStrain<Pose>()));
        EXPECT_TRUE(Priors::WNOA::NodeValues<Pose>().varpi().isZero());
        EXPECT_TRUE(Priors::WNOAQuasiStatic::NodeValues<Pose>().epsilon().isApprox(Priors::straightStrain<Pose>()));
    }

    TEST(StraightStrain, TangentAxisCheck)
    {
        EXPECT_TRUE(Priors::isTangentX<Pose>((Vector6() << 0, 0, 0, 1, 0, 0).finished()));
        EXPECT_TRUE(Priors::isTangentX<Pose>((Vector6() << 2, -1, 3, 0.9, 0.1, -0.2).finished())); // bent and sheared
        EXPECT_FALSE(Priors::isTangentX<Pose>((Vector6() << 0, 0, 0, 0, 0, 1).finished()));       // z tangent
        EXPECT_FALSE(Priors::isTangentX<Pose>((Vector6() << 0, 0, 0, -1, 0, 0).finished()));      // backwards
        EXPECT_FALSE(Priors::isTangentX<Pose>(Vector6::Zero()));
    }

    TEST(StraightStrain, PriorsWarnWhenTheTangentAxisIsNotX)
    {
        const Vector6 z_tangent = (Vector6() << 0, 0, 0, 0, 0, 1).finished();
        const Vector6 x_tangent = Priors::straightStrain<Pose>();

        auto wnoa_state = std::make_shared<WnoaPrior::StateType>(1.0, 3);
        WnoaPrior::ValuesType bad, good;
        bad.epsilon(z_tangent);
        EXPECT_TRUE(warnsAboutTangent([&] { WnoaPrior(wnoa_state, bad); }));
        EXPECT_FALSE(warnsAboutTangent([&] { WnoaPrior(wnoa_state, good); }));

        auto qs_state = std::make_shared<QuasiStaticPrior::StateType>(1.0, 3);
        EXPECT_TRUE(warnsAboutTangent([&] { QuasiStaticPrior(qs_state, QuasiStaticPrior::ValuesType(Pose(), z_tangent)); }));
        EXPECT_FALSE(warnsAboutTangent([&] { QuasiStaticPrior(qs_state, QuasiStaticPrior::ValuesType()); }));

        auto wnov_state = std::make_shared<WnovPrior::StateType>(1.0, 3);
        EXPECT_TRUE(warnsAboutTangent([&] { WnovPrior(wnov_state, Pose(), Vector6::Zero()); }));
        EXPECT_FALSE(warnsAboutTangent([&] { WnovPrior(wnov_state, Pose(), x_tangent); }));
    }
} // namespace Spacetime
