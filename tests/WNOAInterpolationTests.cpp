#include <functional>
#include <memory>
#include <random>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>

#include "NumericalDifferentiation.hpp"

namespace Spacetime::Priors::WNOA
{
    using Pose = gtsam::Pose3;
    using NodeType = Node<Pose>;
    using ValuesType = NodeValues<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using InterpolatorType = Interpolator<Pose>;
    using NumDiff = Spacetime::Testing::NumericalDifferentiation<Pose>;

    static constexpr int dim = gtsam::traits<Pose>::dimension;
    static constexpr int node_dim = NodeType::dim;
    using MatrixN = Eigen::Matrix<double, dim, dim>;
    using MatrixNode = Eigen::Matrix<double, node_dim, node_dim>;
    using VectorNode = Eigen::Matrix<double, node_dim, 1>;

    namespace
    {
        InterpolatorType makeInterpolator()
        {
            return InterpolatorType(
                MatrixN::Identity() * 0.7,  // Q1
                MatrixN::Identity() * 0.5,  // Q2
                MatrixN::Identity() * 0.3,  // Q3
                MatrixN::Identity() * 0.2); // Q4
        }

        /// Node values with a small (few-hundredths-of-a-radian-scale) pose
        /// perturbation and small strain/velocity, representative of the spacing
        /// between neighbouring estimation nodes this interpolator is designed
        /// for. Interpolator's analytic Jacobian relies on a small-angle
        /// approximation (see its class documentation) that is only accurate in
        /// this regime -- unlike ValuesType::setRandom(), which can produce
        /// rotations up to ~1 rad and is fine for mean-only / non-Jacobian tests.
        ValuesType smallRandomValues()
        {
            ValuesType v;
            v.setRandom();
            v.pose(gtsam::traits<Pose>::Expmap(gtsam::traits<Pose>::Logmap(v.pose()) * 0.02));
            v.epsilon(v.epsilon() * 0.02);
            v.varpi(v.varpi() * 0.02);
            return v;
        }

        /// Deterministic (non-random) small-angle-scale node values, indexed by
        /// `index` to vary between corners. Used (instead of smallRandomValues())
        /// wherever the test compares against gtsam::internal::testFactorJacobians,
        /// which -- unlike this file's own isApprox-based (relative) Jacobian
        /// checks -- uses an ABSOLUTE per-entry tolerance and is therefore directly
        /// sensitive to the operating point's magnitude. Interpolator's cross-term
        /// Jacobian (see its class docs, the `-1/2*ad(v)` small-angle
        /// approximation) has an absolute error that scales LINEARLY (not
        /// quadratically) with that magnitude, so an unbounded random draw (however
        /// rare) can push a fixed absolute tolerance into occasional, non-actionable
        /// flakiness; fixed values make the achievable tolerance reproducible.
        ValuesType deterministicSmallValues(int index)
        {
            const double scale = 0.02;
            Eigen::Matrix<double, 6, 1> pose_vec, eps_vec, varpi_vec;
            switch (index % 4)
            {
            case 0:
                pose_vec << 0.3, -0.2, 0.5, 1.0, -0.5, 0.2;
                eps_vec << 0.5, -0.3, 0.2, -0.1, 0.4, 0.3;
                varpi_vec << 0.1, -0.4, 0.3, 0.2, -0.1, 0.5;
                break;
            case 1:
                pose_vec << -0.4, 0.1, 0.2, -0.3, 0.6, -0.1;
                eps_vec << -0.2, 0.4, -0.1, 0.3, -0.5, 0.2;
                varpi_vec << -0.3, 0.2, -0.5, 0.4, 0.1, -0.2;
                break;
            case 2:
                pose_vec << 0.2, 0.3, -0.4, 0.5, -0.2, 0.3;
                eps_vec << 0.3, 0.2, -0.4, 0.1, 0.2, -0.3;
                varpi_vec << 0.4, 0.1, -0.2, -0.3, 0.5, 0.2;
                break;
            default:
                pose_vec << -0.1, -0.3, 0.2, 0.4, 0.1, -0.5;
                eps_vec << -0.4, -0.1, 0.3, -0.2, 0.5, 0.1;
                varpi_vec << -0.2, 0.5, 0.1, -0.4, 0.3, -0.1;
                break;
            }
            ValuesType v;
            v.pose(gtsam::traits<Pose>::Expmap(pose_vec * scale));
            v.epsilon(eps_vec * scale);
            v.varpi(varpi_vec * scale);
            return v;
        }

        void expectValuesEqual(const ValuesType &a, const ValuesType &b, double tol = 1e-6)
        {
            EXPECT_TRUE(a.pose().equals(b.pose(), tol));
            EXPECT_TRUE(a.epsilon().isApprox(b.epsilon(), tol));
            EXPECT_TRUE(a.varpi().isApprox(b.varpi(), tol));
        }

        void expectPSD(const MatrixNode &cov, double tol = 1e-8)
        {
            EXPECT_TRUE(cov.isApprox(cov.transpose(), 1e-9)) << "covariance is not symmetric";
            Eigen::SelfAdjointEigenSolver<MatrixNode> solver(cov);
            EXPECT_GE(solver.eigenvalues().minCoeff(), -tol) << "covariance is not PSD";
        }

        /// Insert a ValuesType's pose/strain/velocity under `node`'s keys into `values`.
        void insertNode(const std::shared_ptr<NodeType> &node, const ValuesType &v, gtsam::Values &values)
        {
            node->insert("pose", v.pose(), values);
            node->insert("strain", v.epsilon(), values);
            node->insert("velocity", v.varpi(), values);
        }

        /// Numerically differentiate a function of the full gtsam::Values (e.g. a call into
        /// Interpolator::interpolate) with respect to a single corner node's node_dim local
        /// perturbation, reading the result back out as a ValuesType local coordinate.
        MatrixNode numericalCornerJacobian(
            const std::function<ValuesType(const gtsam::Values &)> &f,
            const gtsam::Values &values, const std::shared_ptr<NodeType> &node, double h = 1e-6)
        {
            const ValuesType base = f(values);
            const ValuesType node_vals = node->getValues(values);
            MatrixNode J;
            for (int i = 0; i < node_dim; ++i)
            {
                VectorNode delta = VectorNode::Zero();
                delta(i) = h;

                gtsam::Values v_plus = values;
                const ValuesType plus_vals = NumDiff::retractNode(node_vals, delta);
                v_plus.update(node->poseKey(), plus_vals.pose());
                v_plus.update(node->strainKey(), plus_vals.epsilon());
                v_plus.update(node->velocityKey(), plus_vals.varpi());

                gtsam::Values v_minus = values;
                const ValuesType minus_vals = NumDiff::retractNode(node_vals, -delta);
                v_minus.update(node->poseKey(), minus_vals.pose());
                v_minus.update(node->strainKey(), minus_vals.epsilon());
                v_minus.update(node->velocityKey(), minus_vals.varpi());

                const ValuesType f_plus = f(v_plus);
                const ValuesType f_minus = f(v_minus);
                J.col(i) = (NumDiff::localCoordinatesNode(base, f_plus) - NumDiff::localCoordinatesNode(base, f_minus)) / (2 * h);
            }
            return J;
        }

        void expectJacobianMapMatches(
            const std::unordered_map<gtsam::Key, Eigen::MatrixXd> &jacobians,
            const std::shared_ptr<NodeType> &node, const MatrixNode &numerical, double tol)
        {
            ASSERT_TRUE(jacobians.count(node->poseKey()));
            ASSERT_TRUE(jacobians.count(node->strainKey()));
            ASSERT_TRUE(jacobians.count(node->velocityKey()));
            EXPECT_TRUE(jacobians.at(node->poseKey()).isApprox(numerical.block(0, 0, node_dim, dim), tol))
                << "pose-key Jacobian mismatch";
            EXPECT_TRUE(jacobians.at(node->strainKey()).isApprox(numerical.block(0, dim, node_dim, dim), tol))
                << "strain-key Jacobian mismatch";
            EXPECT_TRUE(jacobians.at(node->velocityKey()).isApprox(numerical.block(0, 2 * dim, node_dim, dim), tol))
                << "velocity-key Jacobian mismatch";
        }

        /// Simple inner factor: a gtsam::PriorFactor<Pose> against a fixed reference
        /// pose, referencing only InterpolatorType::InterpolatedKeys::Pose() -- the
        /// pattern any sensor now builds directly (see
        /// Spacetime::Sensors::PoseSensor::getFactor).
        std::shared_ptr<gtsam::PriorFactor<Pose>> posePriorInnerFactor(
            const Pose &reference, const gtsam::SharedNoiseModel &noise_model)
        {
            return std::make_shared<gtsam::PriorFactor<Pose>>(InterpolatorType::InterpolatedKeys::Pose(), reference, noise_model);
        }
    } // namespace

    // ------------------------------------------------------------------
    // 1D (single-axis) interpolation: mean, boundary behaviour, Jacobians
    // ------------------------------------------------------------------

    TEST(WNOAInterpolation, InterpolateTimeBoundaryMatchesNodes)
    {
        InterpolatorType interp = makeInterpolator();
        ValuesType node_k, node_kp1;
        node_k.setRandom();
        node_kp1.setRandom();
        const double t_k = 0.2, t_kp1 = 1.4;

        MatrixNode cov;
        ValuesType at_left = interp.interpolateTime(t_k, t_kp1, t_k, node_k, node_kp1, nullptr, &cov);
        expectValuesEqual(at_left, node_k);
        EXPECT_TRUE(cov.isZero());

        ValuesType at_right = interp.interpolateTime(t_k, t_kp1, t_kp1, node_k, node_kp1);
        expectValuesEqual(at_right, node_kp1, 1e-5);
    }

    TEST(WNOAInterpolation, InterpolateSpaceBoundaryMatchesNodes)
    {
        InterpolatorType interp = makeInterpolator();
        ValuesType node_n, node_np1;
        node_n.setRandom();
        node_np1.setRandom();
        const double s_n = 0.0, s_np1 = 2.0;

        MatrixNode cov;
        ValuesType at_left = interp.interpolateSpace(s_n, s_np1, s_n, node_n, node_np1, nullptr, &cov);
        expectValuesEqual(at_left, node_n);
        EXPECT_TRUE(cov.isZero());

        ValuesType at_right = interp.interpolateSpace(s_n, s_np1, s_np1, node_n, node_np1);
        expectValuesEqual(at_right, node_np1, 1e-5);
    }

    TEST(WNOAInterpolation, InterpolateTimeJacobianMatchesNumerical)
    {
        InterpolatorType interp = makeInterpolator();
        // Small-angle values: the analytic Jacobian's -1/2*ad(v) term is a
        // small-angle approximation (see Interpolator's class documentation),
        // so this uses realistic small inter-node rotations rather than
        // ValuesType::setRandom()'s potentially large (up to ~1 rad) ones.
        // Deterministic rather than smallRandomValues(): this test's assertion is an
        // analytic-vs-numerical Jacobian comparison, whose margin depends on the operating
        // point, so a random draw makes it depend on how much of the global RNG earlier
        // tests happened to consume (i.e. on test execution order).
        ValuesType node_k = deterministicSmallValues(0), node_kp1 = deterministicSmallValues(1);
        const double t_k = 0.3, t_kp1 = 1.7, tau = 0.9;

        Eigen::MatrixXd jac;
        interp.interpolateTime(t_k, t_kp1, tau, node_k, node_kp1, &jac);
        ASSERT_EQ(jac.rows(), node_dim);
        ASSERT_EQ(jac.cols(), 2 * node_dim);

        std::function<ValuesType(const ValuesType &)> f0 = [&](const ValuesType &n0)
        { return interp.interpolateTime(t_k, t_kp1, tau, n0, node_kp1); };
        std::function<ValuesType(const ValuesType &)> f1 = [&](const ValuesType &n1)
        { return interp.interpolateTime(t_k, t_kp1, tau, node_k, n1); };

        const MatrixNode J0_num = NumDiff::numericalJacobian(f0, node_k);
        const MatrixNode J1_num = NumDiff::numericalJacobian(f1, node_kp1);

        EXPECT_TRUE(jac.leftCols(node_dim).isApprox(J0_num, 1e-4));
        EXPECT_TRUE(jac.rightCols(node_dim).isApprox(J1_num, 1e-4));
    }

    TEST(WNOAInterpolation, InterpolateSpaceJacobianMatchesNumerical)
    {
        InterpolatorType interp = makeInterpolator();
        // Deterministic rather than smallRandomValues(): this test's assertion is an
        // analytic-vs-numerical Jacobian comparison, whose margin depends on the operating
        // point, so a random draw makes it depend on how much of the global RNG earlier
        // tests happened to consume (i.e. on test execution order).
        ValuesType node_n = deterministicSmallValues(0), node_np1 = deterministicSmallValues(1);
        const double s_n = 0.1, s_np1 = 2.3, s_tau = 1.5;

        Eigen::MatrixXd jac;
        interp.interpolateSpace(s_n, s_np1, s_tau, node_n, node_np1, &jac);
        ASSERT_EQ(jac.rows(), node_dim);
        ASSERT_EQ(jac.cols(), 2 * node_dim);

        std::function<ValuesType(const ValuesType &)> f0 = [&](const ValuesType &n0)
        { return interp.interpolateSpace(s_n, s_np1, s_tau, n0, node_np1); };
        std::function<ValuesType(const ValuesType &)> f1 = [&](const ValuesType &n1)
        { return interp.interpolateSpace(s_n, s_np1, s_tau, node_n, n1); };

        const MatrixNode J0_num = NumDiff::numericalJacobian(f0, node_n);
        const MatrixNode J1_num = NumDiff::numericalJacobian(f1, node_np1);

        EXPECT_TRUE(jac.leftCols(node_dim).isApprox(J0_num, 1e-4));
        EXPECT_TRUE(jac.rightCols(node_dim).isApprox(J1_num, 1e-4));
    }

    // ------------------------------------------------------------------
    // Conditional covariance
    // ------------------------------------------------------------------

    TEST(WNOAInterpolation, ConditionalCovTimeBoundaryZero)
    {
        InterpolatorType interp = makeInterpolator();
        EXPECT_TRUE(interp.computeConditionalCovTime(0.0, 1.0, 0.0).isZero());
        EXPECT_TRUE(interp.computeConditionalCovTime(0.0, 1.0, 1.0).isZero());
    }

    TEST(WNOAInterpolation, ConditionalCovSpaceBoundaryZero)
    {
        InterpolatorType interp = makeInterpolator();
        EXPECT_TRUE(interp.computeConditionalCovSpace(0.0, 2.0, 0.0).isZero());
        EXPECT_TRUE(interp.computeConditionalCovSpace(0.0, 2.0, 2.0).isZero());
    }

    // ------------------------------------------------------------------
    // Extrapolation primitives
    // ------------------------------------------------------------------

    TEST(WNOAInterpolation, ExtrapolationLambdaIsPhi)
    {
        InterpolatorType interp = makeInterpolator();

        EXPECT_TRUE(interp.getLambdaExtrapTime(0.6).isApprox(Functions<Pose>::getPhiT(0.6), 1e-12));
        EXPECT_TRUE(interp.getLambdaExtrapSpace(0.9).isApprox(Functions<Pose>::getPhiS(0.9), 1e-12));

        // A zero-length extrapolation is the identity, and must not reach getPhi*/getQ*.
        EXPECT_TRUE(interp.getLambdaExtrapTime(0.0).isApprox(MatrixNode::Identity()));
        EXPECT_TRUE(interp.getLambdaExtrapSpace(0.0).isApprox(MatrixNode::Identity()));
    }

    TEST(WNOAInterpolation, ExtrapolationConditionalCovIsQ)
    {
        InterpolatorType interp = makeInterpolator();
        const MatrixN Q1 = MatrixN::Identity() * 0.7, Q2 = MatrixN::Identity() * 0.5;
        const MatrixN Q3 = MatrixN::Identity() * 0.3, Q4 = MatrixN::Identity() * 0.2;

        // With no bordering node, Pcheck = Q(delta) in full -- the interpolation formula's
        // second term, Psi Q Psi^T, vanishes with Psi.
        EXPECT_TRUE(interp.computeConditionalCovExtrapTime(0.6).isApprox(Functions<Pose>::getQt(0.6, Q1, Q3), 1e-12));
        EXPECT_TRUE(interp.computeConditionalCovExtrapSpace(0.9).isApprox(Functions<Pose>::getQs(0.9, Q2, Q4), 1e-12));

        EXPECT_TRUE(interp.computeConditionalCovExtrapTime(0.0).isZero());
        EXPECT_TRUE(interp.computeConditionalCovExtrapSpace(0.0).isZero());

        expectPSD(interp.computeConditionalCovExtrapTime(0.6));
        expectPSD(interp.computeConditionalCovExtrapSpace(0.9));

        // Uncertainty must grow with the extrapolation distance.
        EXPECT_GT(interp.computeConditionalCovExtrapTime(1.2).trace(),
                  interp.computeConditionalCovExtrapTime(0.6).trace());
        EXPECT_GT(interp.computeConditionalCovExtrapSpace(1.8).trace(),
                  interp.computeConditionalCovExtrapSpace(0.9).trace());
    }

    TEST(WNOAInterpolation, ExtrapolationNeverMoreConfidentThanInterpolation)
    {
        InterpolatorType interp = makeInterpolator();
        const double interval = 2.0, tau = 0.8;

        // Conditioning on a second node can only remove uncertainty: the difference is
        // exactly the Psi Q Psi^T term that extrapolation drops, which is PSD by construction.
        expectPSD(MatrixNode(interp.computeConditionalCovExtrapTime(tau) -
                             interp.computeConditionalCovTime(0.0, interval, tau)));
        expectPSD(MatrixNode(interp.computeConditionalCovExtrapSpace(tau) -
                             interp.computeConditionalCovSpace(0.0, interval, tau)));
    }

    TEST(WNOAInterpolation, ExtrapolateZeroDeltaIsAnchor)
    {
        InterpolatorType interp = makeInterpolator();
        ValuesType v0 = smallRandomValues();

        Eigen::MatrixXd jac;
        MatrixNode cov;
        ValuesType at_anchor = interp.extrapolateTime(0.4, 0.4, v0, &jac, &cov);
        expectValuesEqual(at_anchor, v0);
        EXPECT_TRUE(cov.isZero());
        ASSERT_EQ(jac.rows(), node_dim);
        ASSERT_EQ(jac.cols(), node_dim);
        EXPECT_TRUE(jac.isApprox(MatrixNode::Identity()));

        // Just past TOLERANCE the real Phi/Q path runs instead of the coincide guard; it
        // must agree with the anchor, otherwise the two branches disagree at the seam.
        ValuesType just_past = interp.extrapolateTime(0.4, 0.4 + 1e-7, v0);
        expectValuesEqual(just_past, v0, 1e-6);
        ValuesType just_past_s = interp.extrapolateSpace(0.4, 0.4 + 1e-7, v0);
        expectValuesEqual(just_past_s, v0, 1e-6);
    }

    TEST(WNOAInterpolation, ExtrapolateTimeMatchesConstantVelocityPrediction)
    {
        InterpolatorType interp = makeInterpolator();
        ValuesType v0 = smallRandomValues();
        const double t0 = 0.3, dt = 0.45;

        ValuesType result = interp.extrapolateTime(t0, t0 + dt, v0);

        // xi = dt * varpi, so the pose is the constant-velocity prediction and the strain
        // is carried through the chart's left Jacobian (Barfoot eq. 11.51 plus the strain
        // state this framework adds).
        const gtsam::Vector6 xi = dt * v0.varpi();
        Eigen::Matrix<double, dim, dim> J;
        const Pose expected_pose = v0.pose() * gtsam::traits<Pose>::Expmap(xi, J);

        EXPECT_TRUE(result.pose().equals(expected_pose, 1e-9));
        EXPECT_TRUE(result.epsilon().isApprox(J * v0.epsilon(), 1e-9));
        // J(xi) xi == xi, so a pure time extrapolation leaves the velocity untouched.
        EXPECT_TRUE(result.varpi().isApprox(v0.varpi(), 1e-9));
    }

    TEST(WNOAInterpolation, ExtrapolateSpaceMatchesConstantStrainPrediction)
    {
        InterpolatorType interp = makeInterpolator();
        ValuesType v0 = smallRandomValues();
        const double s0 = 0.2, ds = 0.6;

        ValuesType result = interp.extrapolateSpace(s0, s0 + ds, v0);

        const gtsam::Vector6 xi = ds * v0.epsilon();
        Eigen::Matrix<double, dim, dim> J;
        const Pose expected_pose = v0.pose() * gtsam::traits<Pose>::Expmap(xi, J);

        EXPECT_TRUE(result.pose().equals(expected_pose, 1e-9));
        // Mirror image of the time case: strain is preserved, velocity is carried.
        EXPECT_TRUE(result.epsilon().isApprox(v0.epsilon(), 1e-9));
        EXPECT_TRUE(result.varpi().isApprox(J * v0.varpi(), 1e-9));
    }

    TEST(WNOAInterpolation, ExtrapolationIsInterpolationOnAZeroNoiseTrajectory)
    {
        // The core consistency identity between the two schemes. If the second node is
        // itself the extrapolated one, then interpolating between the pair reproduces the
        // extrapolation everywhere on the interval -- exactly, not approximately, since
        //   Lambda*g0 + Psi*Phi(D)*g0 = (Phi_i - Psi*Phi)*g0 + Psi*Phi*g0 = Phi_i*g0.
        InterpolatorType interp = makeInterpolator();
        ValuesType v0 = smallRandomValues();

        const double t0 = 0.3, t1 = 1.7;
        const ValuesType v1 = interp.extrapolateTime(t0, t1, v0);
        for (const double tau : {0.3, 0.55, 0.9, 1.3, 1.7})
        {
            SCOPED_TRACE("time tau=" + std::to_string(tau));
            expectValuesEqual(interp.interpolateTime(t0, t1, tau, v0, v1),
                              interp.extrapolateTime(t0, tau, v0), 1e-9);
        }

        const double s0 = 0.2, s1 = 2.6;
        const ValuesType v1s = interp.extrapolateSpace(s0, s1, v0);
        for (const double sigma : {0.2, 0.7, 1.4, 2.1, 2.6})
        {
            SCOPED_TRACE("arclength sigma=" + std::to_string(sigma));
            expectValuesEqual(interp.interpolateSpace(s0, s1, sigma, v0, v1s),
                              interp.extrapolateSpace(s0, sigma, v0), 1e-9);
        }
    }

    TEST(WNOAInterpolation, ExtrapolateTimeJacobianMatchesNumerical)
    {
        InterpolatorType interp = makeInterpolator();
        // Deterministic rather than smallRandomValues(): this test's assertion is an
        // analytic-vs-numerical Jacobian comparison, whose margin depends on the operating
        // point, so a random draw makes it depend on how much of the global RNG earlier
        // tests happened to consume (i.e. on test execution order).
        ValuesType v0 = deterministicSmallValues(0);
        const double t0 = 0.3, tau = 0.75;

        Eigen::MatrixXd jac;
        interp.extrapolateTime(t0, tau, v0, &jac);
        ASSERT_EQ(jac.rows(), node_dim);
        ASSERT_EQ(jac.cols(), node_dim);

        std::function<ValuesType(const ValuesType &)> f = [&](const ValuesType &n0)
        { return interp.extrapolateTime(t0, tau, n0); };
        EXPECT_TRUE(jac.isApprox(NumDiff::numericalJacobian(f, v0), 1e-4));
    }

    TEST(WNOAInterpolation, ExtrapolateSpaceJacobianMatchesNumerical)
    {
        InterpolatorType interp = makeInterpolator();
        // Deterministic rather than smallRandomValues(): this test's assertion is an
        // analytic-vs-numerical Jacobian comparison, whose margin depends on the operating
        // point, so a random draw makes it depend on how much of the global RNG earlier
        // tests happened to consume (i.e. on test execution order).
        ValuesType v0 = deterministicSmallValues(0);
        const double s0 = 0.2, sigma = 0.8;

        Eigen::MatrixXd jac;
        interp.extrapolateSpace(s0, sigma, v0, &jac);
        ASSERT_EQ(jac.rows(), node_dim);
        ASSERT_EQ(jac.cols(), node_dim);

        std::function<ValuesType(const ValuesType &)> f = [&](const ValuesType &n0)
        { return interp.extrapolateSpace(s0, sigma, n0); };
        EXPECT_TRUE(jac.isApprox(NumDiff::numericalJacobian(f, v0), 1e-4));
    }

    TEST(WNOAInterpolation, BackwardExtrapolationThrows)
    {
        // Extrapolation is forward-only: the WNOA prior is an integrated Wiener process,
        // so Q(delta) has no valid delta < 0 branch. A backwards query must be rejected
        // before it reaches Functions::getQt/getQs (which assert on it).
        InterpolatorType interp = makeInterpolator();
        ValuesType v0 = smallRandomValues();

        EXPECT_THROW(interp.extrapolateTime(1.0, 0.5, v0), std::runtime_error);
        EXPECT_THROW(interp.extrapolateSpace(1.0, 0.5, v0), std::runtime_error);

        // The bare primitives must agree with the entry points above rather than quietly
        // returning Identity/Zero for a backwards offset.
        EXPECT_THROW(interp.getLambdaExtrapTime(-1.0), std::runtime_error);
        EXPECT_THROW(interp.getLambdaExtrapSpace(-1.0), std::runtime_error);
        EXPECT_THROW(interp.computeConditionalCovExtrapTime(-1.0), std::runtime_error);
        EXPECT_THROW(interp.computeConditionalCovExtrapSpace(-1.0), std::runtime_error);

        // A backwards query is rejected even when a bordering node IS supplied: that node
        // lies ahead of the anchor, so it cannot bracket the query either. Letting this
        // through as "interpolation" would reach Functions::getQt/getQs with a negative
        // delta, which asserts in debug and yields an indefinite Q in release.
        ValuesType v1 = smallRandomValues();
        EXPECT_THROW(interp.interpolateTime(1.0, 2.0, 0.5, v0, v1), std::runtime_error);
        EXPECT_THROW(interp.interpolateSpace(1.0, 2.0, 0.5, v0, v1), std::runtime_error);
    }

    // ------------------------------------------------------------------
    // Top-level Interpolator::interpolate dispatch
    // ------------------------------------------------------------------

    TEST(WNOAInterpolation, InterpolateExactMatch)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        ValuesType v00;
        v00.setRandom();
        gtsam::Values values;
        insertNode(node00, v00, values);

        InterpolatorType interp = makeInterpolator();
        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(0.0, 0.0, values, node00, nullptr, nullptr, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, v00);
        EXPECT_TRUE(cov.isZero());

        MatrixNode expected = MatrixNode::Identity();
        expectJacobianMapMatches(jac, node00, expected, 1e-9);
    }

    TEST(WNOAInterpolation, InterpolateTimeOnlyMatchesInterpolateTime)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        ValuesType v00, v01;
        v00.setRandom();
        v01.setRandom();
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);

        InterpolatorType interp = makeInterpolator();
        Eigen::MatrixXd expected_jac;
        MatrixNode expected_cov;
        ValuesType expected = interp.interpolateTime(0.0, 2.0, 0.7, v00, v01, &expected_jac, &expected_cov);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(0.7, 0.0, values, node00, node01, nullptr, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(expected_cov), 1e-9));

        expectJacobianMapMatches(jac, node00, expected_jac.leftCols(node_dim), 1e-9);
        expectJacobianMapMatches(jac, node01, expected_jac.rightCols(node_dim), 1e-9);
    }

    TEST(WNOAInterpolation, InterpolateSpaceOnlyMatchesInterpolateSpace)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 0.0);
        ValuesType v00, v10;
        v00.setRandom();
        v10.setRandom();
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node10, v10, values);

        InterpolatorType interp = makeInterpolator();
        Eigen::MatrixXd expected_jac;
        MatrixNode expected_cov;
        ValuesType expected = interp.interpolateSpace(0.0, 3.0, 1.2, v00, v10, &expected_jac, &expected_cov);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(0.0, 1.2, values, node00, nullptr, node10, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(expected_cov), 1e-9));

        expectJacobianMapMatches(jac, node00, expected_jac.leftCols(node_dim), 1e-9);
        expectJacobianMapMatches(jac, node10, expected_jac.rightCols(node_dim), 1e-9);
    }

    TEST(WNOAInterpolation, InterpolateFullSpaceTimeJacobianMatchesNumerical)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 0.0);
        auto node11 = std::make_shared<NodeType>(km, 3.0, 2.0);

        // Deterministic rather than smallRandomValues(): this test's assertion is an
        // analytic-vs-numerical Jacobian comparison, whose margin depends on the operating
        // point, so a random draw makes it depend on how much of the global RNG earlier
        // tests happened to consume (i.e. on test execution order).
        ValuesType v00 = deterministicSmallValues(0), v01 = deterministicSmallValues(1),
                   v10 = deterministicSmallValues(2), v11 = deterministicSmallValues(3);
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);
        insertNode(node10, v10, values);
        insertNode(node11, v11, values);

        InterpolatorType interp = makeInterpolator();
        const double query_time = 0.8, query_arclength = 1.1;

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(query_time, query_arclength, values, node00, node01, node10, node11, &jac, &cov);
        ASSERT_TRUE(std::dynamic_pointer_cast<ValuesType>(result));

        expectPSD(cov);

        std::function<ValuesType(const gtsam::Values &)> f = [&](const gtsam::Values &vals)
        {
            auto r = interp.interpolate(query_time, query_arclength, vals, node00, node01, node10, node11, nullptr, nullptr);
            return *std::dynamic_pointer_cast<ValuesType>(r);
        };

        for (const auto &node : {node00, node01, node10, node11})
        {
            const MatrixNode numerical = numericalCornerJacobian(f, values, node);
            expectJacobianMapMatches(jac, node, numerical, 1e-4);
        }
    }

    // ------------------------------------------------------------------
    // Top-level dispatch: the extrapolating cells of the 3x3 mode matrix
    // ------------------------------------------------------------------

    TEST(WNOAInterpolation, InterpolateTimeExtrapolationOnly)
    {
        // Query past the last temporal node at a grid arclength: node01 is absent, so the
        // time axis extrapolates off node00 and nothing else is involved.
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        ValuesType v00 = smallRandomValues();
        gtsam::Values values;
        insertNode(node00, v00, values);

        InterpolatorType interp = makeInterpolator();
        const double query_time = 0.6;

        Eigen::MatrixXd expected_jac;
        MatrixNode expected_cov;
        ValuesType expected = interp.extrapolateTime(0.0, query_time, v00, &expected_jac, &expected_cov);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(query_time, 0.0, values, node00, nullptr, nullptr, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(expected_cov), 1e-9));
        expectPSD(cov);
        expectJacobianMapMatches(jac, node00, expected_jac, 1e-9);
        EXPECT_EQ(jac.size(), 3u);
    }

    TEST(WNOAInterpolation, InterpolateSpaceExtrapolationOnly)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        ValuesType v00 = smallRandomValues();
        gtsam::Values values;
        insertNode(node00, v00, values);

        InterpolatorType interp = makeInterpolator();
        const double query_arclength = 0.9;

        Eigen::MatrixXd expected_jac;
        MatrixNode expected_cov;
        ValuesType expected = interp.extrapolateSpace(0.0, query_arclength, v00, &expected_jac, &expected_cov);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(0.0, query_arclength, values, node00, nullptr, nullptr, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(expected_cov), 1e-9));
        expectJacobianMapMatches(jac, node00, expected_jac, 1e-9);
    }

    TEST(WNOAInterpolation, InterpolateSpaceExtrapolationWithTimeInterpolation)
    {
        // The prompt's (s, t) = (1.5, 0.5) example: interpolate in time along the
        // bordering arclength first, THEN extrapolate that intermediate node in space.
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        // Deterministic, not smallRandomValues(): this test's assertion is an
        // analytic-vs-numerical Jacobian comparison, which is sensitive to the operating
        // point (see deterministicSmallValues' rationale above) and would otherwise depend
        // on how much of the global RNG earlier tests happened to consume.
        ValuesType v00 = deterministicSmallValues(0), v01 = deterministicSmallValues(1);
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);

        InterpolatorType interp = makeInterpolator();
        const double query_time = 0.8, query_arclength = 1.1;

        // Reference: the two documented stages composed by hand.
        Eigen::MatrixXd ref_jac_t, ref_jac_s;
        MatrixNode ref_cov_t, ref_cov_s;
        ValuesType mid = interp.interpolateTime(0.0, 2.0, query_time, v00, v01, &ref_jac_t, &ref_cov_t);
        ValuesType expected = interp.extrapolateSpace(0.0, query_arclength, mid, &ref_jac_s, &ref_cov_s);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(query_time, query_arclength, values, node00, node01, nullptr, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(ref_cov_s + ref_jac_s * ref_cov_t * ref_jac_s.transpose()), 1e-9));
        expectPSD(cov);

        std::function<ValuesType(const gtsam::Values &)> f = [&](const gtsam::Values &vals)
        {
            auto r = interp.interpolate(query_time, query_arclength, vals, node00, node01, nullptr, nullptr, nullptr, nullptr);
            return *std::dynamic_pointer_cast<ValuesType>(r);
        };
        for (const auto &node : {node00, node01})
            expectJacobianMapMatches(jac, node, numericalCornerJacobian(f, values, node), 1e-4);
        EXPECT_EQ(jac.size(), 6u);
    }

    TEST(WNOAInterpolation, InterpolateTimeExtrapolationWithSpaceInterpolation)
    {
        // Mirror of the above: interpolate in arclength first, then extrapolate in time.
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 0.0);
        // Deterministic for the same reason as above.
        ValuesType v00 = deterministicSmallValues(0), v10 = deterministicSmallValues(2);
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node10, v10, values);

        InterpolatorType interp = makeInterpolator();
        const double query_time = 0.7, query_arclength = 1.2;

        Eigen::MatrixXd ref_jac_s, ref_jac_t;
        MatrixNode ref_cov_s, ref_cov_t;
        ValuesType mid = interp.interpolateSpace(0.0, 3.0, query_arclength, v00, v10, &ref_jac_s, &ref_cov_s);
        ValuesType expected = interp.extrapolateTime(0.0, query_time, mid, &ref_jac_t, &ref_cov_t);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(query_time, query_arclength, values, node00, nullptr, node10, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(ref_cov_t + ref_jac_t * ref_cov_s * ref_jac_t.transpose()), 1e-9));
        expectPSD(cov);

        std::function<ValuesType(const gtsam::Values &)> f = [&](const gtsam::Values &vals)
        {
            auto r = interp.interpolate(query_time, query_arclength, vals, node00, nullptr, node10, nullptr, nullptr, nullptr);
            return *std::dynamic_pointer_cast<ValuesType>(r);
        };
        for (const auto &node : {node00, node10})
            expectJacobianMapMatches(jac, node, numericalCornerJacobian(f, values, node), 1e-4);
        EXPECT_EQ(jac.size(), 6u);
    }

    TEST(WNOAInterpolation, InterpolateDoubleExtrapolation)
    {
        // The prompt's (s, t) = (1.5, 1.5) example: extrapolate the corner node in space,
        // then extrapolate that result in time. Only node00 participates.
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        // Deterministic for the same reason as above.
        ValuesType v00 = deterministicSmallValues(0);
        gtsam::Values values;
        insertNode(node00, v00, values);

        InterpolatorType interp = makeInterpolator();
        const double query_time = 0.6, query_arclength = 0.9;

        Eigen::MatrixXd ref_jac_s, ref_jac_t;
        MatrixNode ref_cov_s, ref_cov_t;
        ValuesType mid = interp.extrapolateSpace(0.0, query_arclength, v00, &ref_jac_s, &ref_cov_s);
        ValuesType expected = interp.extrapolateTime(0.0, query_time, mid, &ref_jac_t, &ref_cov_t);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        Eigen::MatrixXd cov;
        auto result = interp.interpolate(query_time, query_arclength, values, node00, nullptr, nullptr, nullptr, &jac, &cov);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected);
        EXPECT_TRUE(cov.isApprox(Eigen::MatrixXd(ref_cov_t + ref_jac_t * ref_cov_s * ref_jac_t.transpose()), 1e-9));
        expectPSD(cov);

        std::function<ValuesType(const gtsam::Values &)> f = [&](const gtsam::Values &vals)
        {
            auto r = interp.interpolate(query_time, query_arclength, vals, node00, nullptr, nullptr, nullptr, nullptr, nullptr);
            return *std::dynamic_pointer_cast<ValuesType>(r);
        };
        expectJacobianMapMatches(jac, node00, numericalCornerJacobian(f, values, node00), 1e-4);
        EXPECT_EQ(jac.size(), 3u);
    }

    TEST(WNOAInterpolation, InterpolateSuppliedButUnusedCornersGetZeroJacobians)
    {
        // Priors::InterpolationFactor derives keys() from EVERY corner it was handed, and
        // its Jacobian loop reads this map -- so a supplied corner the resolved query does
        // not depend on must still get a (zero) block, or linearize() sees an empty matrix.
        // Exercised here for a coincident time axis with all four corners present, and for
        // a time-extrapolating query that was handed a now-irrelevant node01.
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 0.0);
        auto node11 = std::make_shared<NodeType>(km, 3.0, 2.0);
        ValuesType v00 = smallRandomValues(), v01 = smallRandomValues();
        ValuesType v10 = smallRandomValues(), v11 = smallRandomValues();
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);
        insertNode(node10, v10, values);
        insertNode(node11, v11, values);

        InterpolatorType interp = makeInterpolator();

        // Coincident time, interpolated arclength: node01/node11 are irrelevant.
        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac;
        interp.interpolate(0.0, 1.2, values, node00, node01, node10, node11, &jac, nullptr);
        EXPECT_EQ(jac.size(), 12u);
        for (const auto &node : {node01, node11})
            for (const gtsam::Key key : {node->poseKey(), node->strainKey(), node->velocityKey()})
            {
                ASSERT_TRUE(jac.count(key)) << "missing Jacobian block for an unused corner";
                EXPECT_TRUE(jac.at(key).isZero()) << "unused corner should contribute nothing";
                EXPECT_EQ(jac.at(key).rows(), node_dim);
                EXPECT_EQ(jac.at(key).cols(), dim);
            }

        // Both axes coincident, all four corners supplied (the pre-existing shape).
        jac.clear();
        interp.interpolate(0.0, 0.0, values, node00, node01, node10, node11, &jac, nullptr);
        EXPECT_EQ(jac.size(), 12u);
    }

    TEST(WNOAInterpolation, InterpolateVisualizerCallShapesUnchanged)
    {
        // ContinuumRobot::updateVisualization passes a null node01 with query_time set to
        // the bit-identical row time. That must keep classifying as "coincide", not as a
        // time extrapolation, or the rendered backbone would silently change.
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.5);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 0.5);
        ValuesType v00 = smallRandomValues(), v10 = smallRandomValues();
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node10, v10, values);

        InterpolatorType interp = makeInterpolator();
        ValuesType expected = interp.interpolateSpace(0.0, 3.0, 1.2, v00, v10);

        auto result = interp.interpolate(node00->time, 1.2, values, node00, nullptr, node10, nullptr);
        auto typed = std::dynamic_pointer_cast<ValuesType>(result);
        ASSERT_TRUE(typed);
        expectValuesEqual(*typed, expected, 1e-12);
    }

    TEST(WNOAInterpolation, InterpolateRejectsInvalidCornerCombinations)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 1.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 1.0);
        ValuesType v00 = smallRandomValues(), v01 = smallRandomValues(), v10 = smallRandomValues();
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);
        insertNode(node10, v10, values);

        InterpolatorType interp = makeInterpolator();

        // Backwards query with no bordering node -- extrapolation is forward-only.
        EXPECT_THROW(interp.interpolate(0.5, 0.0, values, node00, nullptr, nullptr, nullptr), std::runtime_error);
        EXPECT_THROW(interp.interpolate(1.0, -0.5, values, node00, nullptr, nullptr, nullptr), std::runtime_error);

        // Interpolating in both axes requires the fourth corner.
        EXPECT_THROW(interp.interpolate(1.5, 1.2, values, node00, node01, node10, nullptr), std::runtime_error);

        // A bordering node that is not strictly ahead of node00 would silently produce
        // garbage Lambda/Psi; reject it instead.
        auto behind = std::make_shared<NodeType>(km, 0.0, 0.5);
        insertNode(behind, smallRandomValues(), values);
        EXPECT_THROW(interp.interpolate(1.5, 0.0, values, node00, behind, nullptr, nullptr), std::runtime_error);

        // A query PAST the far bordering node is not interpolation either: the
        // Lambda/Psi formulas are only valid between the two nodes. Such a query should
        // have been anchored on the far node and extrapolated from there.
        EXPECT_THROW(interp.interpolate(2.5, 0.0, values, node00, node01, nullptr, nullptr), std::runtime_error);
        EXPECT_THROW(interp.interpolate(1.0, 3.5, values, node00, nullptr, node10, nullptr), std::runtime_error);

        // Backwards query with a bordering node supplied -- rejected for the same reason.
        EXPECT_THROW(interp.interpolate(0.5, 0.0, values, node00, node01, nullptr, nullptr), std::runtime_error);
        EXPECT_THROW(interp.interpolate(1.0, -0.5, values, node00, nullptr, node10, nullptr), std::runtime_error);
    }

    // ------------------------------------------------------------------
    // InterpolationFactor (prior-agnostic wrapper)
    // ------------------------------------------------------------------

    TEST(WNOAInterpolation, InterpolationFactorExactMatchJacobians)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        ValuesType v00;
        v00.setRandom();
        gtsam::Values values;
        insertNode(node00, v00, values);

        InterpolatorType interp = makeInterpolator();
        Pose reference = v00.pose(); // reference == operating point pose -> zero error at this node
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity());
        auto inner_factor = posePriorInnerFactor(reference, noise_model);
        InterpolationFactor<InterpolatorType> factor(
            inner_factor, interp, 0.0, 0.0, node00, nullptr, nullptr, nullptr);

        EXPECT_NEAR(factor.error(values), 0.0, 1e-9);
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("InterpolationFactor-ExactMatch", factor, values, 1e-6, 1e-5));
    }

    TEST(WNOAInterpolation, InterpolationFactorTimeOnlyJacobians)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        // Deterministic (not smallRandomValues()) small-angle values: see the
        // note on deterministicSmallValues() for why this test needs a fixed,
        // reproducible operating point rather than a random draw.
        ValuesType v00 = deterministicSmallValues(0), v01 = deterministicSmallValues(1);
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);

        InterpolatorType interp = makeInterpolator();
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity());
        auto inner_factor = posePriorInnerFactor(Pose(), noise_model);
        // fixed_noise_model=true: testFactorJacobians compares linearize()'s
        // whitened Jacobian against a numerical derivative of whitenedError(),
        // which whitens with the factor's static stored noise model -- it has no
        // way to know that a non-fixed factor's linearize() whitens with a
        // *different*, value-dependent (conditional-covariance-augmented) noise
        // model. That augmentation is validated separately, in
        // InterpolationFactorNoiseAugmentation; this test isolates mean/Jacobian
        // correctness only.
        //
        // Tolerance: gtsam::internal::testFactorJacobians uses an ABSOLUTE
        // per-entry tolerance (unlike the RELATIVE Eigen::isApprox used by the
        // Interpolator-level Jacobian tests above), so it is directly sensitive to
        // the outer inner-factor's own Jacobian magnitude, not just its relative
        // accuracy. Interpolator's cross-term Jacobian (see its class docs, the
        // `-1/2*ad(v)` small-angle approximation) has an absolute error that scales
        // LINEARLY with the operating point's rotation/strain/velocity magnitude
        // (confirmed by direct measurement: halving deterministicSmallValues()'s
        // scale halves the mismatch) rather than shrinking quadratically, so its
        // amplification through a "generic" inner factor's Jacobian (here
        // gtsam::PriorFactor<Pose>, at reference=Identity) is measurably larger
        // than it was through the identity-favoring bespoke error function this
        // test previously used. 1e-2 comfortably covers the ~5e-3 worst-case
        // absolute residual measured across these deterministic tests.
        InterpolationFactor<InterpolatorType> factor(
            inner_factor, interp, 0.7, 0.0, node00, node01, nullptr, nullptr, true);

        EXPECT_TRUE(gtsam::internal::testFactorJacobians("InterpolationFactor-TimeOnly", factor, values, 1e-6, 1e-2));
    }

    TEST(WNOAInterpolation, InterpolationFactorFullSpaceTimeJacobians)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        auto node10 = std::make_shared<NodeType>(km, 3.0, 0.0);
        auto node11 = std::make_shared<NodeType>(km, 3.0, 2.0);
        ValuesType v00 = deterministicSmallValues(0), v01 = deterministicSmallValues(1),
                   v10 = deterministicSmallValues(2), v11 = deterministicSmallValues(3);
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);
        insertNode(node10, v10, values);
        insertNode(node11, v11, values);

        InterpolatorType interp = makeInterpolator();
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity());
        auto inner_factor = posePriorInnerFactor(Pose(), noise_model);
        // See comment in InterpolationFactorTimeOnlyJacobians for why fixed_noise_model=true here.
        InterpolationFactor<InterpolatorType> factor(
            inner_factor, interp, 0.8, 1.1, node00, node01, node10, node11, true);

        EXPECT_TRUE(gtsam::internal::testFactorJacobians("InterpolationFactor-FullSpaceTime", factor, values, 1e-6, 1e-2));
    }

    TEST(WNOAInterpolation, InterpolationFactorNoiseAugmentation)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        ValuesType v00, v01;
        v00.setRandom();
        v01.setRandom();
        gtsam::Values values;
        insertNode(node00, v00, values);
        insertNode(node01, v01, values);

        InterpolatorType interp = makeInterpolator();
        Eigen::MatrixXd base_cov = MatrixN::Identity();
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(base_cov);
        auto inner_factor = posePriorInnerFactor(Pose(), noise_model);

        // Interior query point: conditional covariance is nonzero, so the augmented
        // factor's effective noise should differ from (and be looser than) the fixed one.
        InterpolationFactor<InterpolatorType> augmented(
            inner_factor, interp, 0.7, 0.0, node00, node01, nullptr, nullptr, false);
        InterpolationFactor<InterpolatorType> fixed(
            inner_factor, interp, 0.7, 0.0, node00, node01, nullptr, nullptr, true);

        // Since both factors share the same (nonzero, generally non-orthogonal-to-error)
        // unwhitened residual but different noise models, their whitened errors -- and hence
        // the factor error() -- should generally differ when the conditional covariance is
        // non-negligible.
        EXPECT_NE(augmented.error(values), fixed.error(values));

        // At an exact-match query (zero conditional covariance), the two must coincide exactly.
        InterpolationFactor<InterpolatorType> augmented_exact(
            inner_factor, interp, 0.0, 0.0, node00, nullptr, nullptr, nullptr, false);
        InterpolationFactor<InterpolatorType> fixed_exact(
            inner_factor, interp, 0.0, 0.0, node00, nullptr, nullptr, nullptr, true);
        EXPECT_NEAR(augmented_exact.error(values), fixed_exact.error(values), 1e-9);
    }

    // ------------------------------------------------------------------
    // InterpolationFactor wrapping an extrapolating query
    // ------------------------------------------------------------------

    namespace
    {
        /// Build an InterpolationFactor over the given corners and check its Jacobians
        /// against gtsam's numerical reference. Uses fixed_noise_model = true for the
        /// reason documented on InterpolationFactorTimeOnlyJacobians: testFactorJacobians
        /// cannot account for the value-dependent augmented noise model.
        ///
        /// `tol` is ABSOLUTE and therefore tracks the magnitude of the Jacobian blocks
        /// themselves. Measured at the deterministicSmallValues operating point:
        ///   interpolation (2 or 4 corners)          0.0026 - 0.0049
        ///   extrapolation alongside an interpolated axis (2 corners)  0.0049 - 0.0054
        ///   pure extrapolation (1 corner)           0.0108 - 0.0122
        /// The single-corner cases are ~2.4x worse not because extrapolating is less
        /// accurate -- the residual is flat in the extrapolation distance -- but because
        /// Lambda = Phi concentrates the entire sensitivity on one node instead of
        /// splitting it across several, so each block, and its small-angle error with it,
        /// is correspondingly larger. ExtrapolationJacobianErrorVanishesWithTheOperatingPoint
        /// is the guard that this really is the small-angle term and not a wrong formula.
        void expectFactorJacobiansCorrect(
            const std::string &label, double time, double arclength,
            const std::shared_ptr<NodeType> &node00, const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10, const std::shared_ptr<NodeType> &node11,
            const gtsam::Values &values, double tol)
        {
            InterpolatorType interp = makeInterpolator();
            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity());
            auto inner_factor = posePriorInnerFactor(Pose(), noise_model);

            InterpolationFactor<InterpolatorType> factor(
                inner_factor, interp, time, arclength, node00, node01, node10, node11, true);

            // Every key the factor advertises must receive a Jacobian block; a missing one
            // would surface here as an empty matrix reaching JacobianFactor.
            std::vector<gtsam::Matrix> H(factor.keys().size());
            factor.unwhitenedError(values, &H);
            for (std::size_t i = 0; i < H.size(); ++i)
            {
                EXPECT_GT(H[i].size(), 0) << label << ": empty Jacobian block at index " << i;
                EXPECT_EQ(H[i].cols(), dim) << label << ": wrong Jacobian width at index " << i;
            }

            EXPECT_TRUE(gtsam::internal::testFactorJacobians(label, factor, values, 1e-6, tol));
        }
    } // namespace

    TEST(WNOAInterpolation, InterpolationFactorSpaceExtrapolationWithTimeInterpolationJacobians)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
        gtsam::Values values;
        insertNode(node00, deterministicSmallValues(0), values);
        insertNode(node01, deterministicSmallValues(1), values);

        expectFactorJacobiansCorrect("InterpolationFactor-SpaceExtrapTimeInterp",
                                     /*time=*/0.7, /*arclength=*/1.2,
                                     node00, node01, nullptr, nullptr, values, /*tol=*/1e-2);
    }

    TEST(WNOAInterpolation, InterpolationFactorDoubleExtrapolationJacobians)
    {
        KeyManager km;
        auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
        gtsam::Values values;
        insertNode(node00, deterministicSmallValues(0), values);

        expectFactorJacobiansCorrect("InterpolationFactor-DoubleExtrapolation",
                                     /*time=*/0.7, /*arclength=*/1.2,
                                     node00, nullptr, nullptr, nullptr, values, /*tol=*/2e-2);
    }

    // ------------------------------------------------------------------
    // WNOA::Prior end-to-end: extrapolating factors and the unified query
    // ------------------------------------------------------------------

    namespace
    {
        /// A 3-arclength (s = 0, 1, 2) robot with two temporal rows (t = 0, 0.5).
        /// extendState adds one row per call at exactly the requested time.
        struct PriorFixture
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(2.0, 3);
            std::shared_ptr<Prior<Pose>> prior;
            gtsam::Values values;
            KeyManager km;

            PriorFixture()
            {
                ValuesType node_reference;
                node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<Prior<Pose>>(state, node_reference);

                gtsam::Values new_values;
                prior->extendState(km, 0.0, values, new_values);
                values.insert(new_values);
                new_values.clear();
                prior->extendState(km, 0.5, values, new_values); // one node per call -> rows {0, 0.5}
                values.insert(new_values);
            }
        };
    } // namespace

    TEST(WNOAInterpolation, PriorGetOuterFactorDispatch)
    {
        PriorFixture f;
        // The grid spans t in [0, 0.5] and s in [0, 2].
        ASSERT_DOUBLE_EQ(f.state->estimation_nodes.back()->time, 0.5);
        ASSERT_DOUBLE_EQ(f.state->estimation_nodes.back()->arclength, 2.0);

        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity() * 1e-3);
        auto inner_factor = posePriorInnerFactor(Pose(), noise_model);

        struct Case
        {
            const char *label;
            double time, arclength;
            std::size_t expected_keys;
        };
        // Each corner node contributes 3 keys. Past the end in either axis, the corresponding
        // bordering node is simply absent, so the factor spans fewer nodes than the
        // interpolating equivalent would.
        const Case cases[] = {
            {"exact node match", 0.0, 0.0, 3},
            {"time-only interpolation", 0.25, 0.0, 6},
            {"arclength-only interpolation", 0.0, 0.5, 6},
            {"full space-time interpolation", 0.25, 0.5, 12},
            {"time-extrapolation on a grid arclength", 0.75, 0.0, 3},
            {"time-extrapolation with arclength interpolation", 0.75, 0.5, 6},
            {"arclength-extrapolation with time interpolation", 0.25, 2.5, 6},
            {"extrapolation in both axes", 0.75, 2.5, 3},
        };

        for (const auto &c : cases)
        {
            SCOPED_TRACE(c.label);
            auto factor = f.prior->getOuterFactor(c.time, c.arclength, inner_factor);
            ASSERT_NE(factor, nullptr);
            EXPECT_EQ(factor->keys().size(), c.expected_keys);
            EXPECT_TRUE(std::isfinite(factor->error(f.values)));
            // A linearizable factor is the real requirement -- this is what a missing
            // Jacobian block for an advertised key would break.
            ASSERT_NE(factor->linearize(f.values), nullptr);
        }
    }

    TEST(WNOAInterpolation, PriorRejectsQueriesBeforeTheGraphStart)
    {
        PriorFixture f;
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity() * 1e-3);
        auto inner_factor = posePriorInnerFactor(Pose(), noise_model);

        // Extrapolation is forward-only, so there is no anchor node for these.
        EXPECT_EQ(f.prior->getOuterFactor(-1.0, 0.0, inner_factor), nullptr);
        EXPECT_EQ(f.prior->getOuterFactor(0.25, -0.5, inner_factor), nullptr);
        EXPECT_EQ(f.prior->query(-1.0, 0.0, f.values), nullptr);
        EXPECT_EQ(f.prior->query(0.25, -0.5, f.values), nullptr);
    }

    TEST(WNOAInterpolation, PriorQueryResolvesAcrossTheGraphBoundary)
    {
        PriorFixture f;
        // Prior<Pose> defaults Q1..Q4 to identity, so a standalone interpolator with the
        // same PSDs reproduces its math exactly.
        InterpolatorType reference(MatrixN::Identity(), MatrixN::Identity(),
                                   MatrixN::Identity(), MatrixN::Identity());

        const auto node_00 = f.state->estimation_nodes.at(0); // (s=0, t=0)
        const auto node_01 = f.state->estimation_nodes.at(3); // (s=0, t=0.5)
        const ValuesType v_00 = node_00->getValues(f.values);
        const ValuesType v_01 = node_01->getValues(f.values);

        // Exactly on a node: the node's own values, with no added uncertainty.
        Eigen::MatrixXd cov;
        auto at_node = f.prior->query(0.0, 0.0, f.values, &cov);
        ASSERT_NE(at_node, nullptr);
        expectValuesEqual(*std::dynamic_pointer_cast<ValuesType>(at_node), v_00);
        EXPECT_TRUE(cov.isZero());

        // Interior: ordinary time interpolation.
        auto interior = f.prior->query(0.25, 0.0, f.values, &cov);
        ASSERT_NE(interior, nullptr);
        expectValuesEqual(*std::dynamic_pointer_cast<ValuesType>(interior),
                          reference.interpolateTime(0.0, 0.5, 0.25, v_00, v_01));
        expectPSD(MatrixNode(cov));

        // Past the last row: extrapolation off it, with strictly more uncertainty than
        // the interpolated interior query carried.
        const double interior_trace = cov.trace();
        auto beyond = f.prior->query(0.75, 0.0, f.values, &cov);
        ASSERT_NE(beyond, nullptr);
        expectValuesEqual(*std::dynamic_pointer_cast<ValuesType>(beyond),
                          reference.extrapolateTime(0.5, 0.75, v_01));
        expectPSD(MatrixNode(cov));
        EXPECT_GT(cov.trace(), interior_trace);

        // A caller-supplied Values missing the bordering nodes (e.g. already marginalized
        // out by the fixed-lag smoother) yields nullptr rather than throwing.
        gtsam::Values empty;
        EXPECT_EQ(f.prior->query(0.25, 0.0, empty), nullptr);
    }

    TEST(WNOAInterpolation, PriorQueryMatchesTheFactorItWouldBuild)
    {
        // The read-only query and the factor path must select the same corners and run
        // the same math, otherwise a measurement would be attached somewhere other than
        // where a user inspecting the state was told the robot is.
        PriorFixture f;
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity() * 1e-3);

        for (const auto &qp : {std::make_pair(0.25, 0.5), std::make_pair(0.75, 0.5),
                               std::make_pair(0.25, 2.5), std::make_pair(0.75, 2.5)})
        {
            SCOPED_TRACE("t=" + std::to_string(qp.first) + " s=" + std::to_string(qp.second));
            auto resolved = f.prior->query(qp.first, qp.second, f.values);
            ASSERT_NE(resolved, nullptr);
            const ValuesType state_here = *std::dynamic_pointer_cast<ValuesType>(resolved);

            // A pose prior placed exactly at the queried state must produce zero error.
            auto inner_factor = posePriorInnerFactor(state_here.pose(), noise_model);
            auto factor = f.prior->getOuterFactor(qp.first, qp.second, inner_factor);
            ASSERT_NE(factor, nullptr);
            EXPECT_NEAR(factor->error(f.values), 0.0, 1e-12);
        }
    }


    TEST(WNOAInterpolation, ExtrapolationJacobianErrorVanishesWithTheOperatingPoint)
    {
        // The analytic Jacobian carries a documented small-angle approximation (see
        // Interpolator's class docs, the -1/2*ad(v) term), so it is never exact. What
        // distinguishes that from a WRONG formula is that its residual must vanish with
        // the operating point rather than settling on a floor. Shrinking the node's
        // pose/strain/velocity by 10x must shrink the residual by ~10x.
        //
        // This is the real regression guard for the extrapolation Jacobians; the absolute
        // tolerances used by testFactorJacobians below are only a coarse backstop.
        InterpolatorType interp = makeInterpolator();
        auto noise_model = gtsam::noiseModel::Gaussian::Covariance(MatrixN::Identity());
        auto inner_factor = posePriorInnerFactor(Pose(), noise_model);

        double previous = 0.0;
        for (const double scale : {1.0, 0.1, 0.01})
        {
            KeyManager km;
            auto node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
            const ValuesType base = deterministicSmallValues(0);
            ValuesType v0;
            v0.pose(gtsam::traits<Pose>::Expmap(gtsam::traits<Pose>::Logmap(base.pose()) * scale));
            v0.epsilon(base.epsilon() * scale);
            v0.varpi(base.varpi() * scale);

            gtsam::Values values;
            insertNode(node00, v0, values);

            // Doubly-extrapolated: the shape with the largest residual, since a single
            // corner carries the whole sensitivity.
            InterpolationFactor<InterpolatorType> factor(
                inner_factor, interp, 0.7, 1.2, node00, nullptr, nullptr, nullptr, true);

            const gtsam::Matrix analytic = factor.linearize(values)->jacobian().first;
            const gtsam::Matrix numerical = gtsam::linearizeNumerically(factor, values, 1e-6).jacobian().first;
            const double residual = (analytic - numerical).cwiseAbs().maxCoeff();

            if (previous > 0.0)
                EXPECT_LT(residual, 0.2 * previous)
                    << "residual did not shrink with the operating point at scale " << scale
                    << " -- the Jacobian has a constant error term, not just a small-angle one";
            previous = residual;
        }
    }


    // ------------------------------------------------------------------
    // Conditional covariance frame (see assets/covariance.tex)
    // ------------------------------------------------------------------

    namespace
    {
        /// Independent re-implementation of the local chart of assets/covariance.tex eq. (1):
        /// gamma = [xi; Jr(xi)^-1 eps; Jr(xi)^-1 varpi], xi = Log(T0^-1 T).
        VectorNode chartOf(const ValuesType &node, const ValuesType &anchor)
        {
            const Eigen::Matrix<double, 6, 1> xi = Pose::Logmap(anchor.pose().inverse() * node.pose());
            gtsam::Matrix6 Jr;
            Pose::Expmap(xi, Jr);
            VectorNode gamma;
            gamma << xi, Jr.inverse() * node.epsilon(), Jr.inverse() * node.varpi();
            return gamma;
        }

        /// Inverse of chartOf: T = T0 Exp(xi), eps = Jr(xi) gamma_eps, varpi = Jr(xi) gamma_varpi.
        ValuesType stateOf(const VectorNode &gamma, const ValuesType &anchor)
        {
            gtsam::Matrix6 Jr;
            const Pose T = anchor.pose() * Pose::Expmap(gamma.segment<6>(0), Jr);
            ValuesType out;
            out.pose(T);
            out.epsilon(Jr * gamma.segment<6>(6));
            out.varpi(Jr * gamma.segment<6>(12));
            return out;
        }

        /// Sample covariance, in the state perturbation frame about `mean`, of the states
        /// obtained by perturbing mean's chart coordinate (relative to `anchor`) by
        /// w ~ N(0, chart_covariance) -- i.e. the ground truth that a state-frame
        /// conditional covariance must reproduce.
        MatrixNode sampleStateCovariance(const ValuesType &mean, const ValuesType &anchor,
                                         const MatrixNode &chart_covariance, int samples, unsigned int seed)
        {
            const Eigen::SelfAdjointEigenSolver<MatrixNode> eig(chart_covariance);
            const MatrixNode sqrt_cov = eig.eigenvectors() * eig.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
            const VectorNode gamma_hat = chartOf(mean, anchor);
            std::mt19937 rng(seed);
            std::normal_distribution<double> normal(0.0, 1.0);
            MatrixNode covariance = MatrixNode::Zero();
            for (int i = 0; i < samples; i++)
            {
                VectorNode z;
                for (int k = 0; k < node_dim; k++)
                    z(k) = normal(rng);
                const VectorNode e = NumDiff::localCoordinatesNode(mean, stateOf(gamma_hat + sqrt_cov * z, anchor));
                covariance += e * e.transpose();
            }
            return covariance / samples;
        }

        double relativeBlockError(const MatrixNode &actual, const MatrixNode &expected, int row, int col, int rows, int cols)
        {
            return (actual.block(row, col, rows, cols) - expected.block(row, col, rows, cols)).norm() /
                   expected.block(row, col, rows, cols).norm();
        }

        /// Largest entry-wise deviation in correlation units, max |a_ij - e_ij| / sqrt(e_ii e_jj).
        /// A sample covariance over N draws deviates by ~1/sqrt(N) in these units, so this is
        /// the statistically meaningful check for cross blocks between variables of very
        /// different scale (e.g. strain vs. pose), where a relative block norm is dominated
        /// by sampling noise.
        double maxNormalizedDeviation(const MatrixNode &actual, const MatrixNode &expected)
        {
            const VectorNode sd = expected.diagonal().cwiseSqrt();
            return ((actual - expected).array() / (sd * sd.transpose()).array()).abs().maxCoeff();
        }

        /// Demo-like WNOA hyperparameters (assets/config/default_config.json when
        /// assets/covariance.tex was written); kept fixed so the test does not track the config.
        InterpolatorType makeDemoInterpolator()
        {
            MatrixN Q1 = MatrixN::Zero(), Q2 = MatrixN::Zero(), Q3 = MatrixN::Zero(), Q4 = MatrixN::Zero();
            Q1.diagonal() << 3e2, 3e2, 3e2, 3, 3, 3;
            Q2.diagonal() << 5e2, 5e2, 5e2, 0.8, 0.8, 0.8;
            Q3.diagonal() << 3e3, 3e3, 3e2, 3e1, 3e1, 3e1;
            Q4.diagonal() << 5e1, 5e1, 5e1, 3e1, 3e1, 3e1;
            return InterpolatorType(Q1, Q2, Q3, Q4);
        }

        /// A demo-like extensible-robot node: unit extension strain plus a little curvature.
        ValuesType demoNode(const Pose &pose)
        {
            ValuesType v;
            v.pose(pose);
            v.epsilon((Eigen::Vector<double, 6>() << 0.3, -0.2, 0.1, 0.0, 0.0, 1.0).finished());
            v.varpi((Eigen::Vector<double, 6>() << 0.1, 0.0, 0.4, 0.05, 0.0, 0.0).finished());
            return v;
        }
    } // namespace

    TEST(WNOAInterpolation, ConditionalCovarianceMatchesMonteCarloInStateFrameAlongArclength)
    {
        // Mid-segment arclength interpolation is where the chart-frame error is largest
        // (assets/covariance.tex, Section 8): the omitted lever-arm position variance is
        // ~75% of the chart value, so this case discriminates the two frames.
        const InterpolatorType interp = makeDemoInterpolator();
        const double ds = 0.466 / 4.0, s_query = 0.6 * ds;
        const ValuesType v0 = demoNode(Pose::Expmap((Eigen::Vector<double, 6>() << 0.2, -0.1, 0.3, 0.1, 0.2, -0.1).finished()));
        const ValuesType v1 = demoNode(v0.pose() * Pose::Expmap(ds * v0.epsilon()));

        MatrixNode reported;
        const ValuesType mean = interp.interpolateSpace(0.0, ds, s_query, v0, v1, nullptr, &reported);
        const MatrixNode chart = interp.computeConditionalCovSpace(0.0, ds, s_query);
        const MatrixNode truth = sampleStateCovariance(mean, v0, chart, 20000, 11);

        expectPSD(reported);
        EXPECT_LT(relativeBlockError(reported, truth, 0, 0, 3, 3), 0.05) << "rotation block";
        EXPECT_LT(relativeBlockError(reported, truth, 3, 3, 3, 3), 0.05) << "position block";
        EXPECT_LT(maxNormalizedDeviation(reported, truth), 5.0 / std::sqrt(20000.0)) << "some entry is outside the sampling error";

        // The chart-frame covariance, as previously reported, must NOT pass: this is what
        // makes the test able to detect the error.
        EXPECT_GT(relativeBlockError(chart, truth, 3, 3, 3, 3), 0.25) << "chart frame should visibly disagree";
    }

    TEST(WNOAInterpolation, ConditionalCovarianceMatchesMonteCarloInStateFrameInTime)
    {
        const InterpolatorType interp = makeDemoInterpolator();
        const double dt = 1.0 / 30.0, t_query = 0.5 * dt;
        const ValuesType v0 = demoNode(Pose::Expmap((Eigen::Vector<double, 6>() << -0.1, 0.2, 0.1, 0.3, 0.0, 0.2).finished()));
        const ValuesType v1 = demoNode(v0.pose() * Pose::Expmap(dt * v0.varpi()));

        MatrixNode reported;
        const ValuesType mean = interp.interpolateTime(0.0, dt, t_query, v0, v1, nullptr, &reported);
        const MatrixNode truth = sampleStateCovariance(mean, v0, interp.computeConditionalCovTime(0.0, dt, t_query), 20000, 12);

        expectPSD(reported);
        EXPECT_LT(relativeBlockError(reported, truth, 0, 0, 3, 3), 0.05) << "rotation block";
        EXPECT_LT(relativeBlockError(reported, truth, 3, 3, 3, 3), 0.05) << "position block";
        EXPECT_LT(maxNormalizedDeviation(reported, truth), 5.0 / std::sqrt(20000.0)) << "some entry is outside the sampling error";
    }

    TEST(WNOAInterpolation, ConditionalCovarianceMatchesMonteCarloInStateFrameWhenExtrapolating)
    {
        const InterpolatorType interp = makeDemoInterpolator();
        const double delta = 0.06;
        const ValuesType v0 = demoNode(Pose::Expmap((Eigen::Vector<double, 6>() << 0.1, 0.1, -0.2, 0.0, 0.1, 0.3).finished()));

        MatrixNode reported;
        const ValuesType mean = interp.extrapolateSpace(0.0, delta, v0, nullptr, &reported);
        const MatrixNode truth = sampleStateCovariance(mean, v0, interp.computeConditionalCovExtrapSpace(delta), 20000, 13);

        expectPSD(reported);
        EXPECT_LT(relativeBlockError(reported, truth, 0, 0, 3, 3), 0.05) << "rotation block";
        EXPECT_LT(relativeBlockError(reported, truth, 3, 3, 3, 3), 0.05) << "position block";
        EXPECT_LT(maxNormalizedDeviation(reported, truth), 5.0 / std::sqrt(20000.0)) << "some entry is outside the sampling error";
    }

} // namespace Spacetime::Priors::WNOA
