#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/factorTesting.h>
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
        using InterpolatorType = Interpolator<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        const Matrix6 Q_space = Vector6(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).asDiagonal();
        const Matrix6 Q_time = Vector6(0.6, 0.5, 0.4, 0.3, 0.2, 0.1).asDiagonal();
        const Vector6 epsilon0 = (Vector6() << 0.0, 0.3, 0.0, 0.0, 0.0, 1.0).finished(); // [rot, trans]

        InterpolatorType makeInterpolator() { return InterpolatorType(Q_space, Q_time, epsilon0); }

        const Pose T_a(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3), gtsam::Point3(0.1, 0.2, 0.3));
        const Pose T_b(gtsam::Rot3::RzRyRx(0.3, 0.1, -0.2), gtsam::Point3(0.4, -0.1, 0.6));

        Matrix6 rightJacobian(const Vector6 &gamma)
        {
            Matrix6 Jr;
            gtsam::traits<Pose>::Expmap(gamma, Jr);
            return Jr;
        }

        /// A 2 x 2 node grid (s in {0, 1}, t in {0, 1}) with the given corner poses, indexed [row * 2 + n].
        struct Grid
        {
            KeyManager km;
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 2);
            gtsam::Values values;

            explicit Grid(const std::array<Pose, 4> &poses)
            {
                state->addStateAtTime(km, 0.0);
                state->addStateAtTime(km, 1.0);
                for (unsigned int i = 0; i < 4; i++)
                    node(i)->setValues(values, ValuesType(poses[i]));
            }
            std::shared_ptr<NodeType> node(unsigned int i) const { return state->estimation_nodes.at(i); }
        };

        Pose interpolatedPose(const InterpolatorType &interp, const Grid &grid, const gtsam::Values &values, double t, double s)
        {
            auto result = interp.interpolate(t, s, values, grid.node(0), grid.node(2), grid.node(1), grid.node(3));
            return std::static_pointer_cast<ValuesType>(result)->pose();
        }
    }

    TEST(WNOVInterpolation, ReproducesTheBorderingNodes)
    {
        const auto interp = makeInterpolator();
        Matrix6 cov;
        EXPECT_TRUE(interp.interpolateTime(0.0, 1.0, 0.0, ValuesType(T_a), ValuesType(T_b), nullptr, &cov).pose().equals(T_a, 1e-12));
        EXPECT_TRUE(cov.isZero());
        EXPECT_TRUE(interp.interpolateTime(0.0, 1.0, 1.0, ValuesType(T_a), ValuesType(T_b), nullptr, &cov).pose().equals(T_b, 1e-9));
        EXPECT_TRUE(cov.isZero(1e-12));
    }

    TEST(WNOVInterpolation, IsTheGeodesicFromEitherEnd)
    {
        const auto interp = makeInterpolator();
        const double alpha = 0.3;
        const Pose T = interp.interpolateTime(0.0, 2.0, 2.0 * alpha, ValuesType(T_a), ValuesType(T_b)).pose();
        EXPECT_TRUE(T.equals(T_a * Pose::Expmap(alpha * Pose::Logmap(T_a.between(T_b))), 1e-12));
        EXPECT_TRUE(T.equals(T_b * Pose::Expmap((1.0 - alpha) * Pose::Logmap(T_b.between(T_a))), 1e-9));
    }

    TEST(WNOVInterpolation, OneAxisJacobiansMatchNumerical)
    {
        const auto interp = makeInterpolator();
        for (const bool space : {false, true})
        {
            auto f = [&](const Pose &T0, const Pose &T1)
            {
                return space ? interp.interpolateSpace(0.0, 0.5, 0.2, ValuesType(T0), ValuesType(T1)).pose()
                             : interp.interpolateTime(1.0, 1.5, 1.2, ValuesType(T0), ValuesType(T1)).pose();
            };
            Eigen::MatrixXd jac;
            if (space)
                interp.interpolateSpace(0.0, 0.5, 0.2, ValuesType(T_a), ValuesType(T_b), &jac);
            else
                interp.interpolateTime(1.0, 1.5, 1.2, ValuesType(T_a), ValuesType(T_b), &jac);
            ASSERT_EQ(jac.rows(), 6);
            ASSERT_EQ(jac.cols(), 12);
            EXPECT_TRUE(jac.leftCols(6).isApprox(gtsam::numericalDerivative21<Pose, Pose, Pose>(f, T_a, T_b), 1e-6)) << "space = " << space;
            EXPECT_TRUE(jac.rightCols(6).isApprox(gtsam::numericalDerivative22<Pose, Pose, Pose>(f, T_a, T_b), 1e-6)) << "space = " << space;
        }
    }

    TEST(WNOVInterpolation, ConditionalCovarianceIsABrownianBridge)
    {
        const auto interp = makeInterpolator();
        const double delta = 2.0, delta_i = 0.5;
        const double bridge = delta_i * (delta - delta_i) / delta;

        // Coincident bordering poses: gamma = 0, so the chart and state frames agree.
        Matrix6 cov_time, cov_space;
        interp.interpolateTime(0.0, delta, delta_i, ValuesType(T_a), ValuesType(T_a), nullptr, &cov_time);
        interp.interpolateSpace(0.0, delta, delta_i, ValuesType(T_a), ValuesType(T_a), nullptr, &cov_space);
        EXPECT_TRUE(cov_time.isApprox(bridge * Q_time, 1e-12));
        EXPECT_TRUE(cov_space.isApprox(bridge * Q_space, 1e-12));

        // Distinct poses: reported in the interpolated pose's frame via Jr(gamma).
        const Vector6 gamma = (delta_i / delta) * Pose::Logmap(T_a.between(T_b));
        const Matrix6 Jr = rightJacobian(gamma);
        interp.interpolateTime(0.0, delta, delta_i, ValuesType(T_a), ValuesType(T_b), nullptr, &cov_time);
        EXPECT_TRUE(cov_time.isApprox(Jr * (bridge * Q_time) * Jr.transpose(), 1e-12));
        EXPECT_TRUE(cov_time.isApprox(cov_time.transpose(), 0.0));

        // Largest in the middle of the interval.
        Matrix6 cov_mid;
        interp.interpolateTime(0.0, delta, delta / 2.0, ValuesType(T_a), ValuesType(T_a), nullptr, &cov_mid);
        EXPECT_GT(cov_mid.trace(), (bridge * Q_time).trace());
    }

    TEST(WNOVInterpolation, SpaceExtrapolationFollowsTheNominalStrain)
    {
        const auto interp = makeInterpolator();
        const double delta_i = 0.4;
        Eigen::MatrixXd jac;
        Matrix6 cov;
        const Pose T = interp.extrapolateSpace(0.2, 0.2 + delta_i, ValuesType(T_a), &jac, &cov).pose();

        const Vector6 gamma = delta_i * epsilon0;
        EXPECT_TRUE(T.equals(T_a * Pose::Expmap(gamma), 1e-12));
        const Matrix6 Jr = rightJacobian(gamma);
        EXPECT_TRUE(cov.isApprox(Jr * (delta_i * Q_space) * Jr.transpose(), 1e-12));

        auto f = [&](const Pose &T0)
        { return interp.extrapolateSpace(0.2, 0.2 + delta_i, ValuesType(T0)).pose(); };
        ASSERT_EQ(jac.rows(), 6);
        ASSERT_EQ(jac.cols(), 6);
        EXPECT_TRUE(jac.isApprox(gtsam::numericalDerivative11<Pose, Pose>(f, T_a), 1e-6));
    }

    TEST(WNOVInterpolation, TimeExtrapolationHoldsThePose)
    {
        const auto interp = makeInterpolator();
        Eigen::MatrixXd jac;
        Matrix6 cov;
        const Pose T = interp.extrapolateTime(3.0, 3.5, ValuesType(T_a), &jac, &cov).pose();
        EXPECT_TRUE(T.equals(T_a, 1e-12));
        EXPECT_TRUE(jac.isApprox(Matrix6::Identity(), 1e-12));
        EXPECT_TRUE(cov.isApprox(0.5 * Q_time, 1e-12));
    }

    TEST(WNOVInterpolation, RejectsQueriesOutsideTheForwardInterval)
    {
        const auto interp = makeInterpolator();
        EXPECT_THROW(interp.extrapolateTime(1.0, 0.5, ValuesType(T_a)), std::runtime_error);
        EXPECT_THROW(interp.interpolateTime(0.0, 1.0, 1.5, ValuesType(T_a), ValuesType(T_b)), std::runtime_error);
        EXPECT_THROW(interp.interpolateSpace(0.0, 1.0, -0.5, ValuesType(T_a), ValuesType(T_b)), std::runtime_error);
    }

    TEST(WNOVInterpolation, TwoAxisTranslationOnlyIsBilinear)
    {
        const std::array<gtsam::Point3, 4> p = {gtsam::Point3(0, 0, 0), gtsam::Point3(0, 0, 1),
                                               gtsam::Point3(1, 0, 0), gtsam::Point3(1, 2, 1)};
        Grid grid({Pose(gtsam::Rot3(), p[0]), Pose(gtsam::Rot3(), p[1]), Pose(gtsam::Rot3(), p[2]), Pose(gtsam::Rot3(), p[3])});
        const auto interp = makeInterpolator();

        const double t = 0.3, s = 0.6;
        const gtsam::Point3 expected = (1 - t) * ((1 - s) * p[0] + s * p[1]) + t * ((1 - s) * p[2] + s * p[3]);
        const Pose T = interpolatedPose(interp, grid, grid.values, t, s);
        EXPECT_TRUE(T.rotation().equals(gtsam::Rot3(), 1e-12));
        EXPECT_TRUE(gtsam::assert_equal(expected, T.translation(), 1e-12));
    }

    TEST(WNOVInterpolation, TwoAxisAgreesWithOneAxisOnTheEdges)
    {
        Grid grid({T_a, T_b, T_b * Pose::Expmap(Vector6::Constant(0.05)), T_a * Pose::Expmap(Vector6::Constant(-0.05))});
        const auto interp = makeInterpolator();
        const auto v = [&](unsigned int i)
        { return grid.node(i)->getValues(grid.values); };

        // s on node00's arclength: time-only between node00 and node01.
        EXPECT_TRUE(interpolatedPose(interp, grid, grid.values, 0.4, 0.0).equals(interp.interpolateTime(0.0, 1.0, 0.4, v(0), v(2)).pose(), 1e-12));
        // t on node00's time: arclength-only between node00 and node10.
        EXPECT_TRUE(interpolatedPose(interp, grid, grid.values, 0.0, 0.7).equals(interp.interpolateSpace(0.0, 1.0, 0.7, v(0), v(1)).pose(), 1e-12));
    }

    TEST(WNOVInterpolation, TwoAxisJacobiansMatchNumerical)
    {
        Grid grid({T_a, T_b, T_b * Pose::Expmap(Vector6::Constant(0.05)), T_a * Pose::Expmap(Vector6::Constant(-0.05))});
        const auto interp = makeInterpolator();
        const double t = 0.35, s = 0.6;

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jacobians;
        Eigen::MatrixXd cov;
        interp.interpolate(t, s, grid.values, grid.node(0), grid.node(2), grid.node(1), grid.node(3), &jacobians, &cov);
        ASSERT_EQ(jacobians.size(), 4u);
        EXPECT_EQ(cov.rows(), 6);
        EXPECT_TRUE(cov.isApprox(cov.transpose(), 1e-12));

        for (unsigned int i = 0; i < 4; i++)
        {
            const gtsam::Key key = grid.node(i)->poseKey();
            auto f = [&](const Pose &corner)
            {
                gtsam::Values values = grid.values;
                values.update(key, corner);
                return interpolatedPose(interp, grid, values, t, s);
            };
            const Eigen::MatrixXd numerical = gtsam::numericalDerivative11<Pose, Pose>(f, grid.values.at<Pose>(key));
            EXPECT_TRUE(jacobians.at(key).isApprox(numerical, 1e-6)) << "corner " << i;
        }
    }

    TEST(WNOVInterpolation, SuppliedButIrrelevantCornersGetZeroJacobians)
    {
        Grid grid({T_a, T_b, T_b, T_a});
        const auto interp = makeInterpolator();
        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jacobians;
        interp.interpolate(0.0, 0.0, grid.values, grid.node(0), grid.node(2), grid.node(1), grid.node(3), &jacobians);
        ASSERT_EQ(jacobians.size(), 4u);
        EXPECT_TRUE(jacobians.at(grid.node(0)->poseKey()).isApprox(Matrix6::Identity()));
        for (unsigned int i = 1; i < 4; i++)
            EXPECT_TRUE(jacobians.at(grid.node(i)->poseKey()).isZero()) << "corner " << i;
    }

    TEST(WNOVInterpolation, InterpolationFactorJacobiansMatchNumerical)
    {
        Grid grid({T_a, T_b, T_b * Pose::Expmap(Vector6::Constant(0.05)), T_a * Pose::Expmap(Vector6::Constant(-0.05))});
        const auto noise = gtsam::noiseModel::Gaussian::Covariance(0.01 * Matrix6::Identity());
        auto inner = std::make_shared<Sensors::PoseSensor<ValuesType>::Factor>(noise, T_b, Eigen::MatrixXi::Ones(6, 1));

        const InterpolationFactor<InterpolatorType> factor(
            inner, makeInterpolator(), 0.35, 0.6, grid.node(0), grid.node(2), grid.node(1), grid.node(3), /*fixed_noise_model=*/true);
        EXPECT_EQ(factor.keys().size(), 4u);
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOVInterpolationFactor", factor, grid.values, 1e-6, 1e-5));
    }

    TEST(WNOVInterpolation, InterpolationFactorAddsTheConditionalCovariance)
    {
        // Coincident translation-free corners: the interpolated pose is T_a, gamma = 0.
        Grid grid({T_a, T_a, T_a, T_a});
        const Matrix6 R = 0.01 * Matrix6::Identity();
        const Pose measured = T_a * Pose::Expmap(Vector6::Constant(0.02));
        auto inner = std::make_shared<Sensors::PoseSensor<ValuesType>::Factor>(
            gtsam::noiseModel::Gaussian::Covariance(R), measured, Eigen::MatrixXi::Ones(6, 1));

        const double t = 0.0, s = 0.25; // arclength-only: bridge 0.25 * 0.75 / 1
        const InterpolationFactor<InterpolatorType> factor(
            inner, makeInterpolator(), t, s, grid.node(0), nullptr, grid.node(1), nullptr, /*fixed_noise_model=*/false);

        // The conditional covariance enters the measurement space through the inner
        // factor's own Jacobian G = de/dT.
        Eigen::MatrixXd G;
        const Vector6 e = inner->evaluateError(T_a, &G);
        const Matrix6 S = R + G * ((0.25 * 0.75) * Q_space) * G.transpose();
        EXPECT_NEAR(factor.error(grid.values), 0.5 * e.dot(S.inverse() * e), 1e-12);
    }
} // namespace Spacetime::Priors::WNOV
