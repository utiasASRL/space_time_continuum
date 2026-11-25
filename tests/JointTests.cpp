#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/joint_models/JointBase.hpp>
#include <spacetime/estimator/joint_models/Joint.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

#include "NumericalDifferentiation.hpp"
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector3 = Eigen::Vector3d;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix3 = Eigen::Matrix3d;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::NodeValues<Pose>;
        using PriorType = Priors::WNOA::Prior<Pose>;
        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;

        using WNOVNode = Priors::WNOV::Node<Pose>;
        using WNOVPrior = Priors::WNOV::Prior<Pose>;
        using WNOVState = Types::ContinuumRobotState<WNOVNode>;

        using QSPrior = Priors::WNOAQuasiStatic::Prior<Pose>;
        using QSState = QSPrior::StateType;
        using QSValues = QSPrior::ValuesType;

        using JointType = Joints::Joint<PriorType, PriorType>;
        using MixedJointType = Joints::Joint<PriorType, WNOVPrior>;
        using JointFactor = Priors::TwoSidedInterpolationFactor<InterpolatorType, InterpolatorType>;

        constexpr double kPeriod = 0.1;
        constexpr double kLength = 0.4;

        Matrix3 skew(const Vector3 &v) { return gtsam::skewSymmetric(v); }

        /// Base pose of a robot standing up (body x along world z) at `position`.
        Pose upright(const Vector3 &position) { return Pose(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(position)); }

        /// WNOA robot of length kLength with 5 nodes standing up at `position`, its base locked there.
        struct WNOARobot
        {
            std::shared_ptr<StateType> state;
            std::shared_ptr<PriorType> prior;
            Pose base;

            explicit WNOARobot(const Vector3 &position, unsigned int nodes = 5) : base(upright(position))
            {
                state = std::make_shared<StateType>(kLength, nodes);
                ValueType reference;
                reference.pose(base);
                reference.epsilon((Vector6() << 0, 0, 0, 1.0, 0, 0).finished());
                prior = std::make_shared<PriorType>(state, reference);
                state->lock("pose", 0.0, base);
            }

            /// Nominal (straight, upright) pose at arclength s.
            Pose nominal(double s) const { return base * Pose(gtsam::Rot3(), gtsam::Point3(s, 0, 0)); }
        };

        /// Quasi-static robot of length kLength with 5 nodes standing up at `position`, its base locked there.
        struct QSRobot
        {
            std::shared_ptr<QSState> state;
            std::shared_ptr<QSPrior> prior;
            Pose base;

            explicit QSRobot(const Vector3 &position) : base(upright(position))
            {
                state = std::make_shared<QSState>(kLength, 5);
                prior = std::make_shared<QSPrior>(state, QSValues(base, (Vector6() << 0, 0, 0, 1.0, 0, 0).finished()));
                state->lock("pose", 0.0, base);
            }

            Pose nominal(double s) const { return base * Pose(gtsam::Rot3(), gtsam::Point3(s, 0, 0)); }
        };

        Matrix6 rigidCovariance() { return Matrix6::Identity() * 1e-8; }
        Matrix3 sphericalCovariance() { return Matrix3::Identity() * 1e-8; }

        /// Extends both priors to `time` (sharing `km`), appending the new initial guesses to `values`.
        void extend(KeyManager &km, gtsam::Values &values, double time, const std::vector<std::shared_ptr<Types::Model>> &priors)
        {
            for (const auto &prior : priors)
            {
                gtsam::Values new_values;
                prior->extendState(km, time, values, new_values);
                values.insert(new_values);
            }
        }

        /// Perturbs every pose and vector in `values` so no residual is trivially zero.
        void perturb(gtsam::Values &values)
        {
            int i = 0;
            for (const gtsam::Key key : values.keys())
            {
                const Vector6 d = (Vector6() << 0.01, -0.02, 0.015, 0.01, -0.01, 0.005).finished() * (1.0 + 0.1 * i++);
                if (gtsam::Symbol(key).chr() == 'p')
                    values.update(key, values.at<Pose>(key).retract(d));
                else
                    values.update(key, Vector6(values.at<Vector6>(key) + d.reverse()));
            }
        }

        ValueType query(const PriorType &prior, double time, double s, const gtsam::Values &values)
        {
            return *std::dynamic_pointer_cast<ValueType>(prior.query(time, s, values));
        }

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

        Pose someOffset() { return Pose(gtsam::Rot3::RzRyRx(0.2, -0.1, 0.3), gtsam::Point3(0.03, -0.02, 0.01)); }
        Vector6 someVelocity() { return (Vector6() << 0.3, -0.2, 0.5, 0.1, 0.4, -0.3).finished(); }
    } // namespace

    // ---------------------------------------------------------------------------------
    // Inner factors

    TEST(JointFactors, RigidVelocityMatchesFiniteDifferencesOfTheAttachedPose)
    {
        // varpi_b = Ad(T_off^-1) varpi_a is the body twist of T_a(t) * T_off.
        const Pose T_a = upright(Vector3(0.1, 0.2, 0.3)), offset = someOffset();
        const Vector6 varpi_a = someVelocity();
        const double dt = 1e-6;
        const Pose T_b0 = T_a * offset, T_b1 = T_a * Pose::Expmap(varpi_a * dt) * offset;
        const Vector6 varpi_b = Pose::Logmap(T_b0.between(T_b1)) / dt;
        Joints::RigidVelocityFactor factor(0, 1, offset, gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
        EXPECT_LT(factor.evaluateError(varpi_a, varpi_b).norm(), 1e-5);
    }

    TEST(JointFactors, RigidVelocityJacobians)
    {
        gtsam::Values values;
        values.insert(0, someVelocity());
        values.insert(1, Vector6(someVelocity().reverse()));
        expectJacobiansMatchNumerical(Joints::RigidVelocityFactor(0, 1, someOffset(), gtsam::noiseModel::Isotropic::Sigma(6, 1.0)), values);
    }

    TEST(JointFactors, RigidVelocityOffsetCovarianceIsFirstOrderPropagation)
    {
        const Pose offset = someOffset();
        const Vector6 varpi_a = someVelocity();
        const Matrix6 sigma = (Vector6() << 1e-3, 2e-3, 3e-3, 1e-4, 2e-4, 3e-4).finished().asDiagonal();
        const Vector6 w = offset.inverse().AdjointMap() * varpi_a;
        const Matrix6 expected = Pose::adjointMap(w) * sigma * Pose::adjointMap(w).transpose();
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(expected), gtsam::Matrix(Joints::RigidVelocityFactor::offsetCovariance(varpi_a, offset, sigma)), 1e-12));
        // Zero velocity: the offset's uncertainty does not reach the velocity residual.
        EXPECT_LT(Joints::RigidVelocityFactor::offsetCovariance(Vector6::Zero(), offset, sigma).norm(), 1e-15);
    }

    TEST(JointFactors, SphericalPositionIsZeroWheneverThePointsCoincide)
    {
        const Vector3 r(0.03, -0.02, 0.01);
        const Pose T_a = upright(Vector3(0.1, 0.2, 0.3));
        // Any rotation of B: only the position is joined.
        const Pose T_b(gtsam::Rot3::RzRyRx(0.4, 0.5, -0.6), T_a.transformFrom(gtsam::Point3(r)));
        Joints::SphericalPositionFactor factor(0, 1, r, gtsam::noiseModel::Isotropic::Sigma(3, 1.0));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(Vector3::Zero()), factor.evaluateError(T_a, T_b), 1e-12));
    }

    TEST(JointFactors, SphericalPositionJacobians)
    {
        gtsam::Values values;
        values.insert(0, upright(Vector3(0.1, 0.2, 0.3)).retract(someVelocity() * 0.1));
        values.insert(1, Pose(gtsam::Rot3::RzRyRx(0.4, 0.5, -0.6), gtsam::Point3(0.2, 0.1, 0.5)));
        expectJacobiansMatchNumerical(Joints::SphericalPositionFactor(0, 1, Vector3(0.03, -0.02, 0.01), gtsam::noiseModel::Isotropic::Sigma(3, 1.0)), values);
    }

    TEST(JointFactors, SphericalVelocityIsZeroForCoincidingPointVelocities)
    {
        const Vector3 r(0.03, -0.02, 0.01);
        const Pose T_a = upright(Vector3(0.1, 0.2, 0.3));
        const Pose T_b(gtsam::Rot3::RzRyRx(0.4, 0.5, -0.6), T_a.transformFrom(gtsam::Point3(r)));
        const Vector6 varpi_a = someVelocity();
        // World velocity of the joined point, R_a (nu_a + omega_a x r), expressed in B's body frame.
        const Vector3 point_velocity = T_a.rotation().matrix() * (varpi_a.tail<3>() + varpi_a.head<3>().cross(r));
        Vector6 varpi_b;
        varpi_b << 0.7, -0.1, 0.2, T_b.rotation().matrix().transpose() * point_velocity; // B's angular velocity is free
        Joints::SphericalVelocityFactor factor(0, 1, 2, 3, r, gtsam::noiseModel::Isotropic::Sigma(3, 1.0));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(Vector3::Zero()), factor.evaluateError(T_a, varpi_a, T_b, varpi_b), 1e-12));
    }

    TEST(JointFactors, SphericalVelocityJacobians)
    {
        gtsam::Values values;
        values.insert(0, upright(Vector3(0.1, 0.2, 0.3)).retract(someVelocity() * 0.1));
        values.insert(1, someVelocity());
        values.insert(2, Pose(gtsam::Rot3::RzRyRx(0.4, 0.5, -0.6), gtsam::Point3(0.2, 0.1, 0.5)));
        values.insert(3, Vector6(someVelocity().reverse()));
        expectJacobiansMatchNumerical(Joints::SphericalVelocityFactor(0, 1, 2, 3, Vector3(0.03, -0.02, 0.01), gtsam::noiseModel::Isotropic::Sigma(3, 1.0)), values);
    }

    TEST(JointFactors, SphericalVelocityOffsetCovarianceIsFirstOrderPropagation)
    {
        const Vector6 varpi_a = someVelocity();
        const Matrix3 sigma = Vector3(1e-3, 2e-3, 3e-3).asDiagonal();
        const Matrix3 expected = skew(varpi_a.head<3>()) * sigma * skew(varpi_a.head<3>()).transpose();
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(expected), gtsam::Matrix(Joints::SphericalVelocityFactor::offsetCovariance(varpi_a, sigma)), 1e-12));
    }

    // ---------------------------------------------------------------------------------
    // Construction

    TEST(Joint, ValidConstructionKeepsItsInputs)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        Joints::JointOptions options;
        options.offset = someOffset();
        JointType joint(a.prior, kLength, b.prior, 0.2, rigidCovariance(), options);
        EXPECT_EQ(joint.priorA(), a.prior);
        EXPECT_EQ(joint.priorB(), b.prior);
        EXPECT_DOUBLE_EQ(joint.arclengthA(), kLength);
        EXPECT_DOUBLE_EQ(joint.arclengthB(), 0.2);
        EXPECT_TRUE(joint.offset().equals(someOffset()));
        EXPECT_TRUE(joint.hasOffset());
        EXPECT_FALSE(JointType(a.prior, kLength, b.prior, 0.2, rigidCovariance()).hasOffset());

        // A rotation-only offset is not drawn as a link.
        Joints::JointOptions rotation_only;
        rotation_only.offset = Pose(gtsam::Rot3::Rz(0.5), gtsam::Point3(0, 0, 0));
        EXPECT_FALSE(JointType(a.prior, kLength, b.prior, 0.2, rigidCovariance(), rotation_only).hasOffset());
    }

    TEST(Joint, InvalidPriorsThrow)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        EXPECT_THROW(JointType(nullptr, kLength, b.prior, 0.2, rigidCovariance()), std::invalid_argument);
        EXPECT_THROW(JointType(a.prior, kLength, nullptr, 0.2, rigidCovariance()), std::invalid_argument);
        EXPECT_THROW(JointType(a.prior, kLength, a.prior, 0.2, rigidCovariance()), std::invalid_argument);

        // A second prior over A's own state is still the same robot.
        auto same_state = std::make_shared<PriorType>(a.state, ValueType());
        EXPECT_THROW(JointType(a.prior, kLength, same_state, 0.2, rigidCovariance()), std::invalid_argument);

        auto wnov_prior = std::make_shared<WNOVPrior>(std::make_shared<WNOVState>(kLength, 5), Pose(), gtsam::Vector6::Zero());
        EXPECT_NO_THROW(MixedJointType(a.prior, kLength, wnov_prior, 0.2, rigidCovariance()));
    }

    TEST(Joint, ArclengthsOutsideTheRobotThrow)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        EXPECT_THROW(JointType(a.prior, -0.01, b.prior, 0.2, rigidCovariance()), std::invalid_argument);
        EXPECT_THROW(JointType(a.prior, kLength + 0.01, b.prior, 0.2, rigidCovariance()), std::invalid_argument);
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, -0.01, rigidCovariance()), std::invalid_argument);
        EXPECT_NO_THROW(JointType(a.prior, 0.0, b.prior, kLength, rigidCovariance()));
    }

    TEST(Joint, InvalidCovariancesThrow)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, Matrix6::Zero()), std::invalid_argument); // hard constraints are not supported
        Matrix6 indefinite = rigidCovariance();
        indefinite(2, 2) = -1e-8;
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, indefinite), std::invalid_argument);
        Matrix6 asymmetric = rigidCovariance();
        asymmetric(0, 1) = 1e-9;
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, asymmetric), std::invalid_argument);
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, sphericalCovariance()), std::invalid_argument); // rigid needs 6x6

        Joints::JointOptions spherical;
        spherical.type = Joints::JointType::Spherical;
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, rigidCovariance(), spherical), std::invalid_argument); // spherical needs 3x3
        EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, Matrix3::Zero(), spherical), std::invalid_argument);
        EXPECT_NO_THROW(JointType(a.prior, kLength, b.prior, 0.2, sphericalCovariance(), spherical));
    }

    TEST(Joint, NonPositiveVelocityFloorThrows)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        for (double floor : {0.0, -1e-3})
        {
            Joints::JointOptions options;
            options.velocity_sigma_floor = floor;
            EXPECT_THROW(JointType(a.prior, kLength, b.prior, 0.2, rigidCovariance(), options), std::invalid_argument);
        }
    }

    // ---------------------------------------------------------------------------------
    // Factors added to the estimator

    TEST(Joint, AddsPoseAndVelocityFactorsOncePerRow)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        JointType joint(a.prior, kLength, b.prior, 0.2, rigidCovariance());
        KeyManager km;
        gtsam::Values values;

        extend(km, values, 0.0, {a.prior, b.prior});
        EXPECT_EQ(joint.getNewFactors(0.0, values).size(), 2u);
        EXPECT_TRUE(joint.getNewFactors(0.0, values).empty()); // the row at t = 0 is already joined

        extend(km, values, kPeriod, {a.prior, b.prior});
        const auto factors = joint.getNewFactors(kPeriod, values);
        ASSERT_EQ(factors.size(), 2u);
        for (const auto &factor : factors)
        {
            auto joint_factor = std::dynamic_pointer_cast<JointFactor>(factor);
            ASSERT_TRUE(joint_factor);
            EXPECT_EQ(joint_factor->dim(), 6u);
            for (const gtsam::Key key : joint_factor->keys())
                EXPECT_TRUE(values.exists(key));
        }
    }

    TEST(Joint, RobotWithoutVelocityGetsOnlyThePoseFactor)
    {
        WNOARobot a(Vector3::Zero());
        auto wnov_prior = std::make_shared<WNOVPrior>(std::make_shared<WNOVState>(kLength, 5),
                                                      upright(Vector3(0.3, 0, 0)), (Vector6() << 0, 0, 0, 1.0, 0, 0).finished());
        MixedJointType joint(a.prior, kLength, wnov_prior, 0.2, rigidCovariance());
        KeyManager km;
        gtsam::Values values;
        extend(km, values, 0.0, {a.prior, wnov_prior});
        EXPECT_EQ(joint.getNewFactors(0.0, values).size(), 1u);
    }

    TEST(Joint, QuasiStaticRobotsAreJoinedByTheirPosesOnly)
    {
        WNOARobot a(Vector3::Zero());
        QSRobot b(Vector3(0.3, 0, 0)), c(Vector3(0.6, 0, 0));
        static_assert(!Joints::Joint<PriorType, QSPrior>::joins_velocity && !Joints::Joint<QSPrior, QSPrior>::joins_velocity);
        Joints::Joint<PriorType, QSPrior> mixed(a.prior, kLength, b.prior, 0.2, rigidCovariance());
        Joints::Joint<QSPrior, QSPrior> spherical(b.prior, kLength, c.prior, 0.0, sphericalCovariance(),
                                                   Joints::JointOptions{.type = Joints::JointType::Spherical});
        KeyManager km;
        gtsam::Values values;
        extend(km, values, 0.0, {a.prior, b.prior, c.prior});
        EXPECT_EQ(mixed.getNewFactors(0.0, values).size(), 1u);
        EXPECT_EQ(spherical.getNewFactors(0.0, values).size(), 1u);
    }

    TEST(Joint, JointBaseExposesTheJointOfAnyPriorPair)
    {
        WNOARobot a(Vector3::Zero());
        QSRobot b(Vector3(0.3, 0, 0));
        Joints::JointOptions options;
        options.offset = someOffset();
        const std::shared_ptr<Joints::JointBase> joint = std::make_shared<Joints::Joint<PriorType, QSPrior>>(a.prior, kLength, b.prior, 0.2, rigidCovariance(), options);
        EXPECT_EQ(joint->priorModelA(), a.prior);
        EXPECT_EQ(joint->priorModelB(), b.prior);
        EXPECT_EQ(joint->stateA().get(), static_cast<const Types::State *>(a.state.get()));
        EXPECT_EQ(joint->stateB().get(), static_cast<const Types::State *>(b.state.get()));
        EXPECT_DOUBLE_EQ(joint->arclengthA(), kLength);
        EXPECT_DOUBLE_EQ(joint->arclengthB(), 0.2);
        EXPECT_TRUE(joint->offset().equals(someOffset()));
        EXPECT_TRUE(joint->hasOffset());
        EXPECT_FALSE(joint->joinsVelocity());
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(rigidCovariance()), gtsam::Matrix(joint->offsetCovariance())));
        EXPECT_TRUE(JointType(a.prior, kLength, WNOARobot(Vector3(0.3, 0, 0)).prior, 0.0, rigidCovariance()).joinsVelocity());
    }

    TEST(Joint, RobotsWithMismatchedRowsThrow)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        JointType joint(a.prior, kLength, b.prior, 0.2, rigidCovariance());
        KeyManager km;
        gtsam::Values values;
        extend(km, values, 0.0, {a.prior, b.prior});
        joint.getNewFactors(0.0, values);
        extend(km, values, kPeriod, {a.prior}); // B was not extended
        EXPECT_THROW(joint.getNewFactors(kPeriod, values), std::runtime_error);
    }

    TEST(Joint, RigidFactorsEvaluateTheJointOnTheInterpolatedPoints)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0), 4); // B's s = 0.2 lies between nodes
        Joints::JointOptions options;
        options.offset = someOffset();
        options.velocity_sigma_floor = 2e-3;
        const double s_a = 0.35, s_b = 0.2;
        JointType joint(a.prior, s_a, b.prior, s_b, rigidCovariance(), options);
        KeyManager km;
        gtsam::Values values;
        extend(km, values, 0.0, {a.prior, b.prior});
        perturb(values);
        const auto factors = joint.getNewFactors(0.0, values);
        ASSERT_EQ(factors.size(), 2u);
        auto pose_factor = std::dynamic_pointer_cast<JointFactor>(factors[0]);
        auto velocity_factor = std::dynamic_pointer_cast<JointFactor>(factors[1]);
        ASSERT_TRUE(pose_factor && velocity_factor);

        const ValueType point_a = query(*a.prior, 0.0, s_a, values), point_b = query(*b.prior, 0.0, s_b, values);
        const gtsam::Vector pose_error = gtsam::BetweenFactor<Pose>(0, 1, someOffset(), nullptr).evaluateError(point_a.pose(), point_b.pose());
        EXPECT_TRUE(gtsam::assert_equal(pose_error, pose_factor->unwhitenedError(values), 1e-9));
        const gtsam::Vector velocity_error = point_b.varpi() - someOffset().inverse().AdjointMap() * point_a.varpi();
        EXPECT_TRUE(gtsam::assert_equal(velocity_error, velocity_factor->unwhitenedError(values), 1e-9));

        expectJacobiansMatchNumerical(*pose_factor, values, kInterpolationTolerance);
        expectJacobiansMatchNumerical(*velocity_factor, values, kInterpolationTolerance);
    }

    TEST(Joint, VelocityNoiseIsTheFloorPlusThePropagatedOffsetCovariance)
    {
        // On-node on both sides, so no interpolation covariance is added.
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0));
        Joints::JointOptions options;
        options.offset = someOffset();
        options.velocity_sigma_floor = 2e-3;
        const Matrix6 sigma = (Vector6() << 1e-3, 2e-3, 3e-3, 1e-4, 2e-4, 3e-4).finished().asDiagonal();
        JointType joint(a.prior, kLength, b.prior, 0.2, sigma, options);
        KeyManager km;
        gtsam::Values values;
        extend(km, values, 0.0, {a.prior, b.prior});
        perturb(values);
        const auto factors = joint.getNewFactors(0.0, values);
        auto pose_factor = std::dynamic_pointer_cast<JointFactor>(factors[0]);
        auto velocity_factor = std::dynamic_pointer_cast<JointFactor>(factors[1]);

        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(sigma), pose_factor->covarianceAt(values), 1e-12));
        const Vector6 varpi_a = query(*a.prior, 0.0, kLength, values).varpi();
        const Matrix6 expected = Matrix6::Identity() * 4e-6 + Joints::RigidVelocityFactor::offsetCovariance(varpi_a, someOffset(), sigma);
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(expected), velocity_factor->covarianceAt(values), 1e-12));
    }

    TEST(Joint, SphericalFactorsEvaluateTheJointOnTheInterpolatedPoints)
    {
        WNOARobot a(Vector3::Zero()), b(Vector3(0.3, 0, 0), 4);
        Joints::JointOptions options;
        options.type = Joints::JointType::Spherical;
        options.offset = someOffset();
        const double s_a = 0.35, s_b = 0.2;
        const Matrix3 sigma = Vector3(1e-4, 2e-4, 3e-4).asDiagonal();
        JointType joint(a.prior, s_a, b.prior, s_b, sigma, options);
        KeyManager km;
        gtsam::Values values;
        extend(km, values, 0.0, {a.prior, b.prior});
        perturb(values);
        const auto factors = joint.getNewFactors(0.0, values);
        ASSERT_EQ(factors.size(), 2u);
        auto position_factor = std::dynamic_pointer_cast<JointFactor>(factors[0]);
        auto velocity_factor = std::dynamic_pointer_cast<JointFactor>(factors[1]);
        ASSERT_TRUE(position_factor && velocity_factor);

        const ValueType point_a = query(*a.prior, 0.0, s_a, values), point_b = query(*b.prior, 0.0, s_b, values);
        const Vector3 r = someOffset().translation();
        const Matrix3 R_a = point_a.pose().rotation().matrix(), R_b = point_b.pose().rotation().matrix();
        const Vector3 position_error = R_a.transpose() * (point_b.pose().translation() - point_a.pose().translation()) - r;
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(position_error), position_factor->unwhitenedError(values), 1e-9));
        const Vector3 velocity_error = point_a.varpi().tail<3>() + point_a.varpi().head<3>().cross(r) - R_a.transpose() * R_b * point_b.varpi().tail<3>();
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(velocity_error), velocity_factor->unwhitenedError(values), 1e-9));

        expectJacobiansMatchNumerical(*position_factor, values, kInterpolationTolerance);
        expectJacobiansMatchNumerical(*velocity_factor, values, kInterpolationTolerance);
    }

    // ---------------------------------------------------------------------------------
    // Estimation

    namespace
    {
        /// Two upright robots, A at the origin and B 0.3 m along x, optionally joined tip to
        /// tip at their nominal relative pose; only A's mid-point is measured, pulled sideways.
        struct JoinedProblem
        {
            WNOARobot a{Vector3::Zero()}, b{Vector3(0.3, 0, 0)};
            std::shared_ptr<Sensors::PoseSensor<ValueType>> sensor;
            Pose offset = a.nominal(kLength).between(b.nominal(kLength));

            JoinedProblem(Estimator &estimator, int ticks, bool joined)
            {
                estimator.addPrior(a.prior);
                estimator.addPrior(b.prior);
                if (joined)
                {
                    Joints::JointOptions options;
                    options.offset = offset;
                    estimator.addModel(std::make_shared<JointType>(a.prior, kLength, b.prior, kLength, Matrix6::Identity() * 1e-6, options));
                }

                const Matrix6 covariance = Matrix6::Identity() * 1e-6;
                sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(a.prior);
                const Pose measured = a.nominal(kLength / 2) * Pose(gtsam::Rot3(), gtsam::Point3(0, 0.03, 0));
                for (int k = 0; k < ticks; k++)
                    sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(kLength / 2, k * kPeriod, measured.matrix(), covariance,
                                                                                    Eigen::MatrixXi::Ones(6, 1)));
                estimator.addSensor(sensor);
            }

            Pose tip(const WNOARobot &robot, double time, const gtsam::Values &values) const
            {
                return query(*robot.prior, time, kLength, values).pose();
            }
        };
    } // namespace

    TEST(Joint, BatchEstimateKeepsTheJoinedPointsTogether)
    {
        constexpr int ticks = 4;
        Estimator joined_estimator(Testing::estimatorOptions(Estimator::BATCH_LAG)), free_estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        JoinedProblem joined(joined_estimator, ticks, true), free(free_estimator, ticks, false);
        for (int k = 0; k < ticks; k++)
        {
            joined_estimator.estimate(k * kPeriod);
            free_estimator.estimate(k * kPeriod);
        }
        joined_estimator.finalize();
        free_estimator.finalize();

        const double t = (ticks - 1) * kPeriod;
        const gtsam::Values &joined_values = joined_estimator.getCurrentValues(), &free_values = free_estimator.getCurrentValues();
        const Pose tip_a = joined.tip(joined.a, t, joined_values), tip_b = joined.tip(joined.b, t, joined_values);
        // The joint holds (within its 1 mm / 1 mrad sigma)...
        EXPECT_LT(Pose::Logmap(joined.offset.between(tip_a.between(tip_b))).norm(), 5e-3);
        // ...so A's measured deflection drags B's tip along, unlike without the joint.
        const Pose free_tip_b = free.tip(free.b, t, free_values);
        EXPECT_GT((tip_b.translation() - free_tip_b.translation()).norm(), 5e-3);
        EXPECT_LT((free_tip_b.translation() - free.b.nominal(kLength).translation()).norm(), 1e-3); // B alone stays straight
    }

    TEST(Joint, SlidingWindowEstimateKeepsTheJoinedPointsTogether)
    {
        constexpr int ticks = 8;
        Estimator estimator(Testing::estimatorOptions(0.25));
        JoinedProblem problem(estimator, ticks, true);
        for (int k = 0; k < ticks; k++)
        {
            ASSERT_NO_THROW(estimator.estimate(k * kPeriod));
            const gtsam::Values &values = estimator.getCurrentValues();
            const Pose tip_a = problem.tip(problem.a, k * kPeriod, values), tip_b = problem.tip(problem.b, k * kPeriod, values);
            EXPECT_LT(Pose::Logmap(problem.offset.between(tip_a.between(tip_b))).norm(), 5e-3) << "tick " << k;
        }
    }

    TEST(Joint, BatchEstimateJoinsAQuasiStaticRobot)
    {
        // A's measured deflection drags the joined quasi-static robot B's tip along.
        constexpr int ticks = 3;
        WNOARobot a(Vector3::Zero());
        QSRobot b(Vector3(0.3, 0, 0));
        const Pose offset = a.nominal(kLength).between(b.nominal(kLength));
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(a.prior);
        estimator.addPrior(b.prior);
        Joints::JointOptions options;
        options.offset = offset;
        estimator.addModel(std::make_shared<Joints::Joint<PriorType, QSPrior>>(a.prior, kLength, b.prior, kLength, Matrix6::Identity() * 1e-6, options));
        auto sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(a.prior);
        const Pose measured = a.nominal(kLength / 2) * Pose(gtsam::Rot3(), gtsam::Point3(0, 0.03, 0));
        for (int k = 0; k < ticks; k++)
            sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(kLength / 2, k * kPeriod, measured.matrix(), Matrix6::Identity() * 1e-6,
                                                                            Eigen::MatrixXi::Ones(6, 1)));
        estimator.addSensor(sensor);
        for (int k = 0; k < ticks; k++)
            estimator.estimate(k * kPeriod);
        estimator.finalize();

        const double t = (ticks - 1) * kPeriod;
        const gtsam::Values &values = estimator.getCurrentValues();
        const Pose tip_a = query(*a.prior, t, kLength, values).pose();
        const Pose tip_b = std::dynamic_pointer_cast<QSValues>(b.prior->query(t, kLength, values))->pose();
        EXPECT_LT(Pose::Logmap(offset.between(tip_a.between(tip_b))).norm(), 5e-3);
        EXPECT_GT((tip_b.translation() - b.nominal(kLength).translation()).norm(), 5e-3);
    }
} // namespace Spacetime
