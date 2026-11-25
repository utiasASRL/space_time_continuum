#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/InterpolationFactor.hpp>
#include <spacetime/estimator/prior_models/TwoSidedInterpolationFactor.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

#include "NumericalDifferentiation.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::NodeValues<Pose>;
        using PriorType = Priors::WNOA::Prior<Pose>;
        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        using Keys = InterpolatorType::InterpolatedKeys;
        using Node = Priors::InterpolatedNode<InterpolatorType>;
        using Factor = Priors::TwoSidedInterpolationFactor<InterpolatorType, InterpolatorType>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        constexpr double kTime = 0.1;

        /// Side B's placeholders: WNOA's own keys moved to the 'r' character.
        std::unordered_map<gtsam::Key, gtsam::Key> sideBKeys()
        {
            std::unordered_map<gtsam::Key, gtsam::Key> map;
            for (const gtsam::Key key : Keys::keys())
                map[key] = gtsam::Symbol('r', gtsam::Symbol(key).index());
            return map;
        }

        /// Two WNOA robots (A: 1 m with 5 nodes, B: 0.6 m with 4 nodes, B's base offset in x)
        /// with node rows at t = 0 and t = kTime, sharing one KeyManager, and perturbed values.
        struct Rig
        {
            std::shared_ptr<StateType> state_a = std::make_shared<StateType>(1.0, 5);
            std::shared_ptr<StateType> state_b = std::make_shared<StateType>(0.6, 4);
            std::shared_ptr<PriorType> prior_a, prior_b;
            gtsam::Values values;

            Rig()
            {
                ValueType reference_a, reference_b;
                reference_a.epsilon((Vector6() << 0, 0.2, 0, 1.0, 0, 0).finished());
                reference_b.pose(Pose(gtsam::Rot3::Rz(0.3), gtsam::Point3(0.4, 0.0, 0.0)));
                reference_b.epsilon((Vector6() << 0.1, 0, 0, 1.0, 0, 0).finished());
                prior_a = std::make_shared<PriorType>(state_a, reference_a);
                prior_b = std::make_shared<PriorType>(state_b, reference_b);

                KeyManager km;
                gtsam::Values new_values;
                for (double t : {0.0, kTime})
                    for (const auto &prior : {prior_a, prior_b})
                    {
                        prior->extendState(km, t, values, new_values);
                        values.insert(new_values);
                        new_values.clear();
                    }

                // Distinct, non-trivial perturbations of every variable.
                int i = 0;
                for (const gtsam::Key key : values.keys())
                {
                    const double scale = 1.0 + 0.1 * i++;
                    const Vector6 d = (Vector6() << 0.01, -0.02, 0.015, 0.01, -0.01, 0.005).finished() * scale;
                    if (gtsam::Symbol(key).chr() == 'p')
                        values.update(key, values.at<Pose>(key).retract(d));
                    else
                        values.update(key, Vector6(values.at<Vector6>(key) + 0.5 * d.reverse()));
                }
            }

            Node nodeA(double s) const { return *prior_a->interpolatedNode(kTime, s); }
            Node nodeB(double s) const { return *prior_b->interpolatedNode(kTime, s, sideBKeys()); }

            static gtsam::Key poseA() { return Keys::Pose(); }
            static gtsam::Key velocityA() { return Keys::Velocity(); }
            static gtsam::Key poseB() { return sideBKeys().at(Keys::Pose()); }
            static gtsam::Key velocityB() { return sideBKeys().at(Keys::Velocity()); }

            static Pose offset() { return Pose(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.05), gtsam::Point3(0.02, -0.01, 0.03)); }
            static Matrix6 sigma() { return (Vector6() << 1e-3, 2e-3, 1e-3, 1e-4, 3e-4, 2e-4).finished().asDiagonal(); }

            /// A between factor from A's interpolated pose to B's.
            static gtsam::NoiseModelFactor::shared_ptr between()
            {
                return std::make_shared<gtsam::BetweenFactor<Pose>>(poseA(), poseB(), offset(), gtsam::noiseModel::Gaussian::Covariance(sigma()));
            }
        };

        /// Absolute tolerance for Jacobians through an off-node interpolation: WNOA::Interpolator's
        /// Jacobians w.r.t. the bordering poses are a documented small-angle approximation (see
        /// Interpolator.hpp), off by up to ~5e-3 at this operating scale. On-node they are exact.
        constexpr double kInterpolationTolerance = 5e-3;

        void expectJacobiansMatchNumerical(const gtsam::NoiseModelFactor &factor, const gtsam::Values &values, double tolerance = 1e-5)
        {
            std::vector<gtsam::Matrix> analytic(factor.size()); // gtsam indexes H without resizing it
            factor.unwhitenedError(values, analytic);
            const std::vector<gtsam::Matrix> numerical = Testing::numericalFactorJacobians(factor, values);
            ASSERT_EQ(analytic.size(), numerical.size());
            for (size_t i = 0; i < analytic.size(); i++)
                EXPECT_TRUE(gtsam::assert_equal(numerical[i], analytic[i], tolerance)) << "key " << gtsam::Symbol(factor.keys()[i]);
        }

        /// The interpolation's conditional covariance (node-sized) at (kTime, s).
        Eigen::MatrixXd conditionalCovariance(const PriorType &prior, double s, const gtsam::Values &values)
        {
            Eigen::MatrixXd covariance;
            prior.query(kTime, s, values, &covariance);
            return covariance;
        }

        ValueType interpolated(const PriorType &prior, double s, const gtsam::Values &values)
        {
            return *std::dynamic_pointer_cast<ValueType>(prior.query(kTime, s, values));
        }
    } // namespace

    TEST(InterpolatedNode, MapsPlaceholdersToInnerKeys)
    {
        Rig rig;
        const Node a = rig.nodeA(0.5), b = rig.nodeB(0.3);
        for (const gtsam::Key key : Keys::keys())
        {
            EXPECT_EQ(a.innerKey(key), key);
            EXPECT_EQ(b.innerKey(key), sideBKeys().at(key));
        }
        EXPECT_TRUE(b.ownsInnerKey(Rig::poseB()));
        EXPECT_FALSE(b.ownsInnerKey(Rig::poseA()));
        EXPECT_TRUE(a.ownsInnerKey(Rig::poseA()));
    }

    TEST(InterpolatedNode, BeforeTheFirstNodeIsUnavailable)
    {
        Rig rig;
        EXPECT_FALSE(rig.prior_a->interpolatedNode(-1.0, 0.5).has_value());
    }

    TEST(TwoSidedInterpolationFactor, KeysAreTheUnionOfBothSidesCorners)
    {
        Rig rig;
        // On-node on both sides: one corner node (3 keys) each.
        Factor on_node(Rig::between(), rig.nodeA(0.5), rig.nodeB(0.4));
        EXPECT_EQ(on_node.keys().size(), 6u);
        // Off-node on both sides: two corner nodes each.
        Factor off_node(Rig::between(), rig.nodeA(0.6), rig.nodeB(0.3));
        EXPECT_EQ(off_node.keys().size(), 12u);
        for (const gtsam::Key key : off_node.keys())
            EXPECT_TRUE(rig.values.exists(key));
    }

    TEST(TwoSidedInterpolationFactor, ErrorIsTheInnerErrorOnBothInterpolatedNodes)
    {
        Rig rig;
        for (const auto &[s_a, s_b] : std::vector<std::pair<double, double>>{{0.5, 0.4}, {0.6, 0.3}, {1.0, 0.1}})
        {
            Factor factor(Rig::between(), rig.nodeA(s_a), rig.nodeB(s_b));
            const Pose pose_a = interpolated(*rig.prior_a, s_a, rig.values).pose();
            const Pose pose_b = interpolated(*rig.prior_b, s_b, rig.values).pose();
            const gtsam::Vector expected = gtsam::BetweenFactor<Pose>(0, 1, Rig::offset(), nullptr).evaluateError(pose_a, pose_b);
            EXPECT_TRUE(gtsam::assert_equal(expected, factor.unwhitenedError(rig.values), 1e-9)) << s_a << ", " << s_b;
        }
    }

    TEST(TwoSidedInterpolationFactor, JacobiansMatchNumericalOnAndOffNode)
    {
        Rig rig;
        expectJacobiansMatchNumerical(Factor(Rig::between(), rig.nodeA(0.5), rig.nodeB(0.4)), rig.values); // exact on-node
        for (const auto &[s_a, s_b] : std::vector<std::pair<double, double>>{{0.6, 0.3}, {0.5, 0.3}, {0.6, 0.4}})
            expectJacobiansMatchNumerical(Factor(Rig::between(), rig.nodeA(s_a), rig.nodeB(s_b)), rig.values, kInterpolationTolerance);
    }

    TEST(TwoSidedInterpolationFactor, MatchesAnInterpolationFactorWhenOneSideIsOnNode)
    {
        // With B on a node, the two-sided factor is InterpolationFactor on A with B's node
        // pose passed through: same error, Jacobians and covariance, to rounding.
        Rig rig;
        const double s_a = 0.6, s_b = 0.4;
        const Factor two_sided(Rig::between(), rig.nodeA(s_a), rig.nodeB(s_b));
        const auto node_b = rig.state_b->getClosestShiftedNode(s_b, kTime, -1, -1);
        const auto n00 = rig.state_a->getClosestShiftedNode(s_a, kTime, -1, -1), n10 = rig.state_a->getNextSpatialNode(n00);
        const Priors::InterpolationFactor<InterpolatorType> one_sided(
            std::make_shared<gtsam::BetweenFactor<Pose>>(Rig::poseA(), node_b->poseKey(), Rig::offset(), gtsam::noiseModel::Gaussian::Covariance(Rig::sigma())),
            rig.prior_a->interpolator(), kTime, s_a, n00, nullptr, n10, nullptr);

        EXPECT_TRUE(gtsam::assert_equal(one_sided.unwhitenedError(rig.values), two_sided.unwhitenedError(rig.values), 1e-12));
        std::vector<gtsam::Matrix> H_one(one_sided.size()), H_two(two_sided.size());
        one_sided.unwhitenedError(rig.values, H_one);
        two_sided.unwhitenedError(rig.values, H_two);
        for (size_t i = 0; i < one_sided.size(); i++)
        {
            const auto it = std::find(two_sided.keys().begin(), two_sided.keys().end(), one_sided.keys()[i]);
            ASSERT_NE(it, two_sided.keys().end());
            EXPECT_TRUE(gtsam::assert_equal(H_one[i], H_two[it - two_sided.keys().begin()], 1e-12)) << gtsam::Symbol(one_sided.keys()[i]);
        }
        auto linear_one = std::dynamic_pointer_cast<gtsam::JacobianFactor>(one_sided.linearize(rig.values));
        EXPECT_NEAR(0.5 * linear_one->getb().squaredNorm(), two_sided.error(rig.values), 1e-9);
    }

    TEST(TwoSidedInterpolationFactor, NoiseAddsBothSidesConditionalCovariance)
    {
        Rig rig;
        const double s_a = 0.6, s_b = 0.3;
        Factor factor(Rig::between(), rig.nodeA(s_a), rig.nodeB(s_b));

        gtsam::Matrix G_a, G_b;
        gtsam::BetweenFactor<Pose>(0, 1, Rig::offset(), nullptr)
            .evaluateError(interpolated(*rig.prior_a, s_a, rig.values).pose(), interpolated(*rig.prior_b, s_b, rig.values).pose(), &G_a, &G_b);
        const auto [offset, width] = Keys::rowRange(Keys::Pose());
        const Matrix6 P_a = conditionalCovariance(*rig.prior_a, s_a, rig.values).block(offset, offset, width, width);
        const Matrix6 P_b = conditionalCovariance(*rig.prior_b, s_b, rig.values).block(offset, offset, width, width);
        const Matrix6 expected = Rig::sigma() + G_a * P_a * G_a.transpose() + G_b * P_b * G_b.transpose();

        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(expected), factor.covarianceAt(rig.values), 1e-9));
        EXPECT_GT((factor.covarianceAt(rig.values) - Rig::sigma()).norm(), 1e-9); // both sides really interpolate
    }

    TEST(TwoSidedInterpolationFactor, OnNodeNoiseIsTheInnerNoisePlusTheExtraCovariance)
    {
        Rig rig;
        const Matrix6 extra = Matrix6::Identity() * 4e-4;
        int calls = 0;
        Factor factor(Rig::between(), rig.nodeA(0.5), rig.nodeB(0.4),
                      [&](const gtsam::Values &inner_values)
                      {
                          calls++;
                          EXPECT_TRUE(inner_values.exists(Rig::poseA()));
                          EXPECT_TRUE(inner_values.exists(Rig::poseB()));
                          return Eigen::MatrixXd(extra);
                      });
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(Rig::sigma() + extra), factor.covarianceAt(rig.values), 1e-12));
        EXPECT_GT(calls, 0);
    }

    TEST(TwoSidedInterpolationFactor, LinearizationIsWhitenedByTheAugmentedNoise)
    {
        Rig rig;
        Factor factor(Rig::between(), rig.nodeA(0.6), rig.nodeB(0.3));
        auto linear = std::dynamic_pointer_cast<gtsam::JacobianFactor>(factor.linearize(rig.values));
        ASSERT_TRUE(linear);
        // 0.5 |b|^2 of the whitened system is the factor's error at the linearization point.
        EXPECT_NEAR(0.5 * linear->getb().squaredNorm(), factor.error(rig.values), 1e-9);
        const gtsam::Vector e = factor.unwhitenedError(rig.values);
        EXPECT_NEAR(factor.error(rig.values), 0.5 * e.dot(factor.covarianceAt(rig.values).ldlt().solve(e)), 1e-9);
    }

    TEST(TwoSidedInterpolationFactor, WorksForAFactorOnVelocities)
    {
        Rig rig;
        auto inner = std::make_shared<gtsam::BetweenFactor<Vector6>>(Rig::velocityA(), Rig::velocityB(), Vector6::Zero(),
                                                                     gtsam::noiseModel::Isotropic::Sigma(6, 1e-2));
        expectJacobiansMatchNumerical(Factor(inner, rig.nodeA(0.6), rig.nodeB(0.3)), rig.values, kInterpolationTolerance);
        // On-node, the interpolation is the identity and the Jacobians are exact.
        expectJacobiansMatchNumerical(Factor(inner, rig.nodeA(0.5), rig.nodeB(0.4)), rig.values);
    }

    TEST(TwoSidedInterpolationFactor, InvalidConstructionThrows)
    {
        Rig rig;
        EXPECT_THROW(Factor(nullptr, rig.nodeA(0.5), rig.nodeB(0.4)), std::invalid_argument);
        // Both sides on WNOA's default placeholders: their inner keys would collide.
        EXPECT_THROW(Factor(Rig::between(), rig.nodeA(0.5), rig.nodeA(0.6)), std::invalid_argument);
        // A passed-through inner key that is also one of the bordering nodes' keys.
        const gtsam::Key corner_key = rig.nodeA(0.5).cornerKeys().front();
        auto clashing = std::make_shared<gtsam::BetweenFactor<Vector6>>(Rig::velocityB(), corner_key, Vector6::Zero(),
                                                                       gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
        EXPECT_THROW(Factor(clashing, rig.nodeA(0.5), rig.nodeB(0.4)), std::invalid_argument);
        // A non-Gaussian (e.g. Constrained) inner noise model cannot be augmented.
        auto constrained = std::make_shared<gtsam::BetweenFactor<Pose>>(Rig::poseA(), Rig::poseB(), Rig::offset(),
                                                                        gtsam::noiseModel::Constrained::All(6));
        EXPECT_THROW(Factor(constrained, rig.nodeA(0.5), rig.nodeB(0.4)), std::invalid_argument);
    }

    TEST(TwoSidedInterpolationFactor, KeysNeitherSideProvidesArePassedThrough)
    {
        // E.g. a sensor bias: an inner key that is a real variable, not an interpolated node's.
        Rig rig;
        const gtsam::Key extra = gtsam::Symbol('z', 0);
        rig.values.insert(extra, Pose(gtsam::Rot3::RzRyRx(0.2, 0.1, -0.1), gtsam::Point3(0.3, -0.1, 0.2)));
        auto inner = std::make_shared<gtsam::BetweenFactor<Pose>>(Rig::poseA(), extra, Rig::offset(), gtsam::noiseModel::Gaussian::Covariance(Rig::sigma()));

        for (const double s_a : {0.5, 0.6})
        {
            Factor factor(inner, rig.nodeA(s_a), rig.nodeB(0.4));
            const auto keys = factor.keys();
            const auto it = std::find(keys.begin(), keys.end(), extra);
            ASSERT_NE(it, keys.end()) << s_a;

            // The error is the inner one on A's interpolated pose and the variable itself.
            const Pose pose_a = interpolated(*rig.prior_a, s_a, rig.values).pose();
            gtsam::Matrix H_a, H_extra;
            const gtsam::Vector expected = gtsam::BetweenFactor<Pose>(0, 1, Rig::offset(), nullptr)
                                               .evaluateError(pose_a, rig.values.at<Pose>(extra), &H_a, &H_extra);
            EXPECT_TRUE(gtsam::assert_equal(expected, factor.unwhitenedError(rig.values), 1e-9));

            // Its Jacobian is the inner one, unchanged.
            std::vector<gtsam::Matrix> H(factor.size());
            factor.unwhitenedError(rig.values, H);
            EXPECT_TRUE(gtsam::assert_equal(H_extra, H[it - keys.begin()], 1e-12));
            expectJacobiansMatchNumerical(factor, rig.values, s_a == 0.5 ? 1e-5 : kInterpolationTolerance);

            // And it adds no conditional covariance: only side A's interpolation does.
            const auto [offset, width] = Keys::rowRange(Keys::Pose());
            const Matrix6 P_a = conditionalCovariance(*rig.prior_a, s_a, rig.values).block(offset, offset, width, width);
            const Matrix6 expected_covariance = Rig::sigma() + H_a * P_a * H_a.transpose();
            EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(expected_covariance), factor.covarianceAt(rig.values), 1e-9));
        }
    }
} // namespace Spacetime
