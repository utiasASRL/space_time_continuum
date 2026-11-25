#include <memory>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

// The read-only accessors tools/FindCovariances.py builds its hypergradient from:
// GridPrior::queryJacobians and InterpolationFactor's inner factor, query point and inner Jacobian.
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
        using Factor = Priors::InterpolationFactor<InterpolatorType>;
        using Jacobians = std::unordered_map<gtsam::Key, Eigen::MatrixXd>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        constexpr double kR = 1e-3;

        /// A WNOA prior with two time rows (t = 0, 0.1) on a 1 m, 4-node robot, at a perturbed state.
        struct Rig
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 4);
            std::shared_ptr<PriorType> prior;
            gtsam::Values x;

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
                int i = 0;
                for (const gtsam::Key key : x.keys())
                    if (gtsam::Symbol(key).chr() == 'p')
                    {
                        const Vector6 d = (Vector6() << 0.01, -0.02, 0.015, 0.01, -0.01, 0.005).finished() * (1.0 + 0.1 * i++);
                        x.update(key, x.at<Pose>(key).retract(d));
                    }
            }

            gtsam::NoiseModelFactor::shared_ptr inner(double arclength) const
            {
                const Pose measured(gtsam::Rot3::RzRyRx(0.02, -0.01, 0.03), gtsam::Point3(0.01, 0.0, arclength + 0.02));
                return std::make_shared<Sensors::PoseSensor<ValueType>::Factor>(
                    gtsam::noiseModel::Gaussian::Covariance(Matrix6::Identity() * kR), measured, Eigen::MatrixXi::Ones(6, 1));
            }

            std::shared_ptr<Factor> factor(double time, double arclength) const
            {
                return std::dynamic_pointer_cast<Factor>(prior->getOuterFactor(time, arclength, inner(arclength)));
            }
        };

        /// Every query shape the grid resolves: on a node, in time only, in arclength only, both, and extrapolated in time.
        const std::vector<std::pair<double, double>> kQueries = {
            {0.1, 2.0 / 3.0}, {0.05, 1.0 / 3.0}, {0.1, 0.4}, {0.05, 0.4}, {0.13, 0.4}};
    } // namespace

    TEST(GridPriorQueryJacobians, MatchTheInterpolatorsJacobiansAndMean)
    {
        Rig rig;
        for (const auto &[t, s] : kQueries)
        {
            Jacobians jacobians;
            auto node = std::dynamic_pointer_cast<ValueType>(rig.prior->queryJacobians(t, s, rig.x, jacobians));
            ASSERT_NE(node, nullptr) << "t " << t << " s " << s;

            auto n00 = rig.state->getClosestShiftedNode(s, t, -1, -1);
            auto n01 = std::abs(t - n00->time) >= TOLERANCE ? rig.state->getNextTemporalNode(n00) : nullptr;
            auto n10 = std::abs(s - n00->arclength) >= TOLERANCE ? rig.state->getNextSpatialNode(n00) : nullptr;
            auto n11 = (n01 && n10) ? rig.state->getNextSpatialNode(n01) : nullptr;
            Jacobians expected;
            auto expected_node = std::dynamic_pointer_cast<ValueType>(
                rig.prior->interpolator().interpolate(t, s, rig.x, n00, n01, n10, n11, &expected));

            EXPECT_TRUE(node->pose().equals(expected_node->pose(), 1e-12));
            EXPECT_TRUE(node->epsilon().isApprox(expected_node->epsilon(), 1e-12));
            EXPECT_TRUE(node->varpi().isApprox(expected_node->varpi(), 1e-12));
            ASSERT_EQ(jacobians.size(), expected.size());
            for (const auto &[key, J] : expected)
            {
                ASSERT_TRUE(jacobians.count(key));
                EXPECT_TRUE(jacobians.at(key).isApprox(J, 1e-12)) << gtsam::Symbol(key);
            }
        }
    }

    TEST(GridPriorQueryJacobians, OnANodeTheJacobianIsTheIdentityOnItsOwnKeys)
    {
        Rig rig;
        Jacobians jacobians;
        ASSERT_NE(rig.prior->queryJacobians(0.1, 2.0 / 3.0, rig.x, jacobians), nullptr);
        const auto node = rig.state->getClosestShiftedNode(2.0 / 3.0, 0.1, -1, -1);
        Eigen::MatrixXd stacked(18, 18);
        int column = 0;
        for (const gtsam::Key key : node->keys())
        {
            ASSERT_TRUE(jacobians.count(key));
            stacked.middleCols(column, jacobians.at(key).cols()) = jacobians.at(key);
            column += jacobians.at(key).cols();
        }
        EXPECT_TRUE(stacked.isApprox(Eigen::MatrixXd::Identity(18, 18), 1e-12));
    }

    TEST(GridPriorQueryJacobians, UnavailableQueriesReturnNullAndNoJacobians)
    {
        Rig rig;
        Jacobians jacobians;
        EXPECT_EQ(rig.prior->queryJacobians(-0.5, 0.4, rig.x, jacobians), nullptr); // before the earliest node
        EXPECT_TRUE(jacobians.empty());

        gtsam::Values missing = rig.x; // a bordering node's variable marginalized out
        const auto n00 = rig.state->getClosestShiftedNode(0.4, 0.05, -1, -1);
        missing.erase(n00->keys().front());
        EXPECT_EQ(rig.prior->queryJacobians(0.05, 0.4, missing, jacobians), nullptr);
        EXPECT_TRUE(jacobians.empty());
    }

    TEST(InterpolationFactorAccessors, ExposeTheInnerFactorAndQueryPoint)
    {
        Rig rig;
        auto inner = rig.inner(0.4);
        auto factor = std::dynamic_pointer_cast<Factor>(rig.prior->getOuterFactor(0.05, 0.4, inner));
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(factor->innerFactor(), inner);
        EXPECT_DOUBLE_EQ(factor->time(), 0.05);
        EXPECT_DOUBLE_EQ(factor->arclength(), 0.4);
        EXPECT_FALSE(factor->fixedNoiseModel()); // the WNOA default
    }

    TEST(InterpolationFactorAccessors, InnerJacobianIsThePoseErrorJacobianOnThePoseRows)
    {
        Rig rig;
        auto factor = rig.factor(0.05, 0.4);
        ASSERT_NE(factor, nullptr);
        const Eigen::MatrixXd G = factor->innerJacobian(rig.x);
        ASSERT_EQ(G.rows(), 6);
        ASSERT_EQ(G.cols(), 18);

        auto node = std::dynamic_pointer_cast<ValueType>(rig.prior->query(0.05, 0.4, rig.x));
        gtsam::Matrix J;
        std::dynamic_pointer_cast<Sensors::PoseSensor<ValueType>::Factor>(rig.inner(0.4))->evaluateError(node->pose(), &J);
        const auto [offset, width] = ValueType::InterpolatedKeys::rowRange(ValueType::InterpolatedKeys::Pose());
        Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(6, 18);
        expected.middleCols(offset, width) = J;
        EXPECT_TRUE(G.isApprox(expected, 1e-12));
    }

    // R + G Pcheck G^T, with Pcheck from GridPrior::query, is the covariance linearize() whitens with:
    // the factor's information A^T A equals Jᵀ (R + G Pcheck Gᵀ)⁻¹ J for its unwhitened Jacobian J.
    TEST(InterpolationFactorAccessors, InnerJacobianAndQueryReproduceTheAugmentedNoise)
    {
        Rig rig;
        for (const auto &[t, s] : kQueries)
        {
            auto factor = rig.factor(t, s);
            ASSERT_NE(factor, nullptr);
            Eigen::MatrixXd pcheck;
            ASSERT_NE(rig.prior->query(t, s, rig.x, &pcheck), nullptr);
            const Eigen::MatrixXd G = factor->innerJacobian(rig.x);
            const Eigen::MatrixXd sigma = Matrix6::Identity() * kR + G * pcheck * G.transpose();

            std::vector<gtsam::Matrix> H;
            const gtsam::Vector e = factor->unwhitenedError(rig.x, &H);
            Eigen::MatrixXd J(6, 0);
            for (const auto &block : H)
            {
                J.conservativeResize(6, J.cols() + block.cols());
                J.rightCols(block.cols()) = block;
            }
            auto linear = std::dynamic_pointer_cast<gtsam::JacobianFactor>(factor->linearize(rig.x));
            ASSERT_NE(linear, nullptr);
            const auto [A, b] = linear->jacobian();
            const Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma);
            EXPECT_TRUE((A.transpose() * A).isApprox(J.transpose() * ldlt.solve(J), 1e-8)) << "t " << t << " s " << s;
            EXPECT_NEAR(b.squaredNorm(), e.dot(ldlt.solve(e)), 1e-8 * std::max(1.0, b.squaredNorm()));
        }
    }
} // namespace Spacetime
