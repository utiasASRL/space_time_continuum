#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
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
        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        using Factor = Priors::InterpolationFactor<InterpolatorType>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        constexpr double kTime = 0.05, kArclength = 0.4;

        /// A WNOA prior with two time rows, the corners of (kTime, kArclength), and two
        /// different states x and y of those corners.
        struct Rig
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 4);
            std::shared_ptr<PriorType> prior;
            std::shared_ptr<NodeType> n00, n01, n10, n11;
            gtsam::Values x, y;

            Rig()
            {
                ValueType reference;
                reference.epsilon((Vector6() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<PriorType>(state, reference);
                KeyManager km;
                gtsam::Values new_values;
                for (double t : {0.0, 0.1})
                {
                    prior->extendState(km, t, x, new_values);
                    x.insert(new_values);
                    new_values.clear();
                }
                n00 = state->getClosestShiftedNode(kArclength, kTime, -1, -1);
                n01 = state->getNextTemporalNode(n00);
                n10 = state->getNextSpatialNode(n00);
                n11 = state->getNextSpatialNode(n01);
                y = x;
                int i = 0;
                for (const gtsam::Key key : x.keys())
                    if (gtsam::Symbol(key).chr() == 'p')
                    {
                        const Vector6 d = (Vector6() << 0.01, -0.02, 0.015, 0.01, -0.01, 0.005).finished() * (1.0 + 0.1 * i++);
                        x.update(key, x.at<Pose>(key).retract(d));
                        y.update(key, x.at<Pose>(key).retract(-2.0 * d));
                    }
            }

            /// A pose measurement at (kTime, kArclength), with a Gaussian or a (Cauchy) robust noise model.
            gtsam::NoiseModelFactor::shared_ptr inner(bool robust) const
            {
                return inner(robust ? gtsam::noiseModel::mEstimator::Cauchy::Create(1.0) : nullptr);
            }

            /// A pose measurement at (kTime, kArclength), its Gaussian noise wrapped in `loss` unless that is null.
            gtsam::NoiseModelFactor::shared_ptr inner(const gtsam::noiseModel::mEstimator::Base::shared_ptr &loss) const
            {
                gtsam::SharedNoiseModel noise = gtsam::noiseModel::Gaussian::Covariance(Matrix6::Identity() * 1e-3);
                if (loss)
                    noise = gtsam::noiseModel::Robust::Create(loss, noise);
                const Pose measured(gtsam::Rot3::RzRyRx(0.02, -0.01, 0.03), gtsam::Point3(0.01, 0.0, kArclength + 0.02));
                return std::make_shared<Sensors::PoseSensor<ValueType>::Factor>(noise, measured, Eigen::MatrixXi::Ones(6, 1));
            }

            Factor factor(bool robust, bool fixed, bool cache) const
            {
                return Factor(inner(robust), prior->interpolator(), kTime, kArclength, n00, n01, n10, n11, fixed, cache);
            }

            /// The augmented inner covariance at `at`: Sigma + J Pcheck J^T (J = inner Jacobian at the interpolated pose).
            Matrix6 augmentedCovariance(const gtsam::Values &at) const
            {
                Eigen::MatrixXd pcheck;
                auto node = std::dynamic_pointer_cast<ValueType>(
                    prior->interpolator().interpolate(kTime, kArclength, at, n00, n01, n10, n11, nullptr, &pcheck));
                gtsam::Matrix J;
                std::dynamic_pointer_cast<Sensors::PoseSensor<ValueType>::Factor>(inner(false))->evaluateError(node->pose(), &J);
                const auto [offset, width] = ValueType::InterpolatedKeys::rowRange(ValueType::InterpolatedKeys::Pose());
                return Matrix6::Identity() * 1e-3 + J * pcheck.block(offset, offset, width, width) * J.transpose();
            }
        };

        std::string writeFile(const std::filesystem::path &path, const std::string &contents)
        {
            std::ofstream(path) << contents;
            return path.string();
        }
    } // namespace

    TEST(InterpolationFactorCache, LinearizationIsUnchanged)
    {
        Rig rig;
        for (bool robust : {false, true})
        {
            auto a = std::dynamic_pointer_cast<gtsam::JacobianFactor>(rig.factor(robust, false, true).linearize(rig.x));
            auto b = std::dynamic_pointer_cast<gtsam::JacobianFactor>(rig.factor(robust, false, false).linearize(rig.x));
            ASSERT_TRUE(a && b);
            EXPECT_TRUE(a->equals(*b, 1e-12)) << "robust " << robust;
        }
    }

    TEST(InterpolationFactorCache, ErrorElsewhereUsesTheLinearizationPointsNoiseModel)
    {
        Rig rig;
        const Factor cached = rig.factor(false, false, true), exact = rig.factor(false, false, false);
        cached.linearize(rig.x);

        const gtsam::Vector r = cached.unwhitenedError(rig.y);
        const double expected = 0.5 * r.dot(rig.augmentedCovariance(rig.x).ldlt().solve(r));
        EXPECT_NEAR(cached.error(rig.y), expected, 1e-9 * expected);

        // ... which differs from building the noise model at y itself.
        const double at_y = 0.5 * r.dot(rig.augmentedCovariance(rig.y).ldlt().solve(r));
        EXPECT_NEAR(exact.error(rig.y), at_y, 1e-9 * at_y);
        EXPECT_GT(std::abs(cached.error(rig.y) - exact.error(rig.y)), 1e-9 * expected);
    }

    TEST(InterpolationFactorCache, RobustErrorElsewhereUsesTheLinearizationPointsNoiseModel)
    {
        Rig rig;
        const Factor cached = rig.factor(true, false, true);
        cached.linearize(rig.x);
        const gtsam::Vector r = cached.unwhitenedError(rig.y);
        auto robust = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1.0),
                                                        gtsam::noiseModel::Gaussian::Covariance(rig.augmentedCovariance(rig.x)));
        EXPECT_NEAR(cached.error(rig.y), robust->loss(r), 1e-9 * robust->loss(r));
    }

    TEST(InterpolationFactorCache, ErrorBeforeAnyLinearizationIsExact)
    {
        Rig rig;
        EXPECT_NEAR(rig.factor(false, false, true).error(rig.y), rig.factor(false, false, false).error(rig.y), 1e-12);
    }

    TEST(InterpolationFactorCache, RelinearizingRefreshesTheCache)
    {
        Rig rig;
        const Factor cached = rig.factor(false, false, true), exact = rig.factor(false, false, false);
        cached.linearize(rig.x);
        cached.linearize(rig.y);
        EXPECT_NEAR(cached.error(rig.y), exact.error(rig.y), 1e-12 * std::max(1.0, exact.error(rig.y)));
    }

    TEST(InterpolationFactorCache, FixedNoiseModelTakesPrecedence)
    {
        Rig rig;
        const Factor fixed_cached = rig.factor(false, true, true), fixed = rig.factor(false, true, false);
        fixed_cached.linearize(rig.x);
        EXPECT_DOUBLE_EQ(fixed_cached.error(rig.y), fixed.error(rig.y));
        const gtsam::Vector r = fixed.unwhitenedError(rig.y);
        EXPECT_NEAR(fixed.error(rig.y), 0.5 * r.squaredNorm() / 1e-3, 1e-9 * fixed.error(rig.y));
    }

    TEST(InterpolationFactorCache, PriorsPassTheFlagToTheirFactors)
    {
        Rig rig;
        auto factor = std::dynamic_pointer_cast<Factor>(rig.prior->getOuterFactor(kTime, kArclength, rig.inner(false)));
        ASSERT_NE(factor, nullptr);
        const Factor exact = rig.factor(false, false, false);
        factor->linearize(rig.x);
        EXPECT_GT(std::abs(factor->error(rig.y) - exact.error(rig.y)), 0.0); // cached by default

        Priors::WNOA::PriorOptions options;
        options.cache_noise_model = false;
        rig.prior.reset(); // a state is managed by one prior at a time
        auto uncached_prior = std::make_shared<PriorType>(rig.state, ValueType(), PriorType::Matrix3N::Identity(),
                                                          PriorType::MatrixN::Identity(), PriorType::MatrixN::Identity(),
                                                          PriorType::MatrixN::Identity(), PriorType::MatrixN::Identity(), options);
        auto uncached = uncached_prior->getOuterFactor(kTime, kArclength, rig.inner(false));
        const double before = uncached->error(rig.y);
        std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(uncached)->linearize(rig.x);
        EXPECT_DOUBLE_EQ(uncached->error(rig.y), before);
    }

    TEST(InterpolationFactorCache, ConfigLoaderReadsTheFlagWithDefaultTrue)
    {
        const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / "cache_noise_model_config";
        std::filesystem::create_directories(dir);

        Utilities::ConfigLoader defaults;
        ASSERT_TRUE(defaults.loadConfig(writeFile(dir / "defaults.json", R"({ "prior": { "wnoa": {}, "wnov": {} } })")));
        EXPECT_TRUE(defaults.loadWNOAPriorOptions().cache_noise_model);
        EXPECT_TRUE(defaults.loadWNOVPriorOptions().cache_noise_model);

        Utilities::ConfigLoader disabled;
        ASSERT_TRUE(disabled.loadConfig(writeFile(dir / "disabled.json",
                                                  R"({ "prior": { "wnoa": { "cache_noise_model": false }, "wnov": { "cache_noise_model": false } } })")));
        EXPECT_FALSE(disabled.loadWNOAPriorOptions().cache_noise_model);
        EXPECT_FALSE(disabled.loadWNOVPriorOptions().cache_noise_model);
        std::filesystem::remove_all(dir);
    }

    // A robust inner noise model must survive the wrapping: InterpolationFactor unwraps it,
    // augments its Gaussian with the interpolation covariance and rewraps it with the same loss.
    TEST(InterpolationFactorRobust, FixedNoiseModelKeepsThePerRowRobustLoss)
    {
        Rig rig;
        auto cauchy = gtsam::noiseModel::mEstimator::Cauchy::Create(1.0, gtsam::noiseModel::mEstimator::Base::Scalar);
        const Factor robust(rig.inner(cauchy), rig.prior->interpolator(), kTime, kArclength, rig.n00, rig.n01, rig.n10, rig.n11, true, false);
        const Factor gaussian = rig.factor(false, true, false);

        // With a fixed noise model the error is the per-row Cauchy loss of the whitened inner error.
        const gtsam::Vector e = robust.unwhitenedError(rig.x) / std::sqrt(1e-3);
        double expected = 0.0;
        for (int j = 0; j < e.size(); j++)
            expected += cauchy->loss(e(j));
        EXPECT_NEAR(robust.error(rig.x), expected, 1e-9 * expected);
        EXPECT_NEAR(gaussian.error(rig.x), 0.5 * e.squaredNorm(), 1e-9 * gaussian.error(rig.x));
        EXPECT_LT(robust.error(rig.x), gaussian.error(rig.x));

        // Linearized, each row is down-weighted by its own residual (IRLS): sqrt(w(e_j)) e_j.
        auto linear = std::dynamic_pointer_cast<gtsam::JacobianFactor>(robust.linearize(rig.x));
        ASSERT_NE(linear, nullptr);
        const gtsam::Vector b = linear->getb();
        ASSERT_EQ(b.size(), e.size());
        for (int j = 0; j < e.size(); j++)
            EXPECT_NEAR(std::abs(b(j)), std::sqrt(cauchy->weight(e(j))) * std::abs(e(j)), 1e-9) << "row " << j;
    }

    TEST(InterpolationFactorRobust, AugmentedNoiseModelKeepsTheRobustLoss)
    {
        // Off the grid the conditional covariance inflates the inner covariance; the robust
        // loss is applied on top of that augmented Gaussian, and stays below the Gaussian error.
        Rig rig;
        const Factor robust = rig.factor(true, false, false), gaussian = rig.factor(false, false, false);
        const gtsam::Vector r = robust.unwhitenedError(rig.x);
        auto expected = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1.0),
                                                          gtsam::noiseModel::Gaussian::Covariance(rig.augmentedCovariance(rig.x)));
        EXPECT_NEAR(robust.error(rig.x), expected->loss(r), 1e-9 * expected->loss(r));
        EXPECT_GT(robust.error(rig.x), 0.0);
        EXPECT_LT(robust.error(rig.x), gaussian.error(rig.x));
        EXPECT_NE(robust.linearize(rig.x), nullptr);
    }
} // namespace Spacetime
