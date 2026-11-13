#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/types/NearestNodeFactor.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

// Tests of Types::StatefulModel's default outer factor: snapping an off-node query to the nearest
// node (nearest arclength, then nearest time), for stateful models that do not override
// getOuterFactor/query.
namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector6 = gtsam::Vector6;
        using NodeType = Priors::WNOA::Node<Pose>;
        using ValuesType = Priors::WNOA::Values<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Keys = Priors::WNOA::InterpolatedKeys<Pose>;

        /// A prior relying entirely on Types::StatefulModel's default outer factors and queries. Each node gets loose
        /// unary priors (so a graph of them is well-posed) and a distinct initial value.
        class SnappingPrior : public Types::StatefulModel<StateType>
        {
        public:
            explicit SnappingPrior(std::shared_ptr<StateType> state) : Types::StatefulModel<StateType>(std::move(state)) {}

            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> extendState(
                KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
            {
                (void)current_values;
                auto state = this->state();
                state->addStateAtTime(km, time);

                std::vector<std::shared_ptr<gtsam::NonlinearFactor>> factors;
                auto loose = gtsam::noiseModel::Isotropic::Sigma(6, 10.0);
                for (size_t i = state->estimation_nodes.size() - state->spatial_dim; i < state->estimation_nodes.size(); i++)
                {
                    const auto &node = state->estimation_nodes[i];
                    ValuesType values;
                    values.pose(Pose(gtsam::Rot3::RzRyRx(0.1 * node->arclength, 0.2 * node->time, 0.3),
                                     gtsam::Point3(node->arclength, node->time, 0.5)));
                    values.epsilon(Vector6::Constant(node->arclength + 1.0));
                    values.varpi(Vector6::Constant(node->time - 2.0));
                    node->setValues(new_values, values);

                    factors.push_back(std::make_shared<gtsam::PriorFactor<Pose>>(node->poseKey(), Pose(), loose));
                    factors.push_back(std::make_shared<gtsam::PriorFactor<Vector6>>(node->strainKey(), Vector6::Zero(), loose));
                    factors.push_back(std::make_shared<gtsam::PriorFactor<Vector6>>(node->velocityKey(), Vector6::Zero(), loose));
                }
                return factors;
            }
        };

        /// A node type that does not define interpolated keys (NodeBase's default).
        struct PlainNode : public Types::NodeBase
        {
            PlainNode(KeyManager &km, double arclength, double time) : Types::NodeBase(km, arclength, time)
            {
                this->registerVariable<Vector6>(km, "x", 'x');
            }
        };

        class PlainPrior : public Types::StatefulModel<Types::ContinuumRobotState<PlainNode>>
        {
        public:
            explicit PlainPrior(std::shared_ptr<Types::ContinuumRobotState<PlainNode>> state)
                : Types::StatefulModel<Types::ContinuumRobotState<PlainNode>>(std::move(state)) {}

            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> extendState(
                KeyManager &km, double time, const gtsam::Values &, gtsam::Values &) override
            {
                this->state()->addStateAtTime(km, time);
                return {};
            }
        };

        /// error = strain + 2 * velocity - landmark: depends on two interpolated keys and one pass-through key.
        class StrainVelocityLandmarkFactor : public gtsam::NoiseModelFactorN<Vector6, Vector6, Vector6>
        {
        public:
            using Base = gtsam::NoiseModelFactorN<Vector6, Vector6, Vector6>;
            StrainVelocityLandmarkFactor(gtsam::Key landmark)
                : Base(gtsam::noiseModel::Isotropic::Sigma(6, 0.1), Keys::Strain(), Keys::Velocity(), landmark) {}

            gtsam::Vector evaluateError(const Vector6 &strain, const Vector6 &velocity, const Vector6 &landmark,
                                        gtsam::OptionalMatrixType H1, gtsam::OptionalMatrixType H2,
                                        gtsam::OptionalMatrixType H3) const override
            {
                if (H1)
                    *H1 = gtsam::Matrix66::Identity();
                if (H2)
                    *H2 = 2.0 * gtsam::Matrix66::Identity();
                if (H3)
                    *H3 = -gtsam::Matrix66::Identity();
                return strain + 2.0 * velocity - landmark;
            }
        };

        std::shared_ptr<gtsam::NoiseModelFactor> makePoseFactor(const Pose &measured, const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(6, 1))
        {
            using Factor = Sensors::PoseSensor<ValuesType>::Factor;
            return std::make_shared<Factor>(gtsam::noiseModel::Isotropic::Sigma(6, 0.01), measured, mask);
        }

        /// Non-uniform arclengths {0, 0.2, 1} and rows at t = {0, 0.1, 0.3}.
        struct Grid
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(std::vector<double>{0.0, 0.2, 1.0});
            std::shared_ptr<SnappingPrior> prior = std::make_shared<SnappingPrior>(state);
            KeyManager km;
            gtsam::Values values;

            Grid()
            {
                for (double t : {0.0, 0.1, 0.3})
                {
                    gtsam::Values new_values;
                    prior->extendState(km, t, values, new_values);
                    values.insert(new_values);
                }
            }

            std::shared_ptr<NodeType> nodeAt(double s, double t) const
            {
                for (const auto &node : state->estimation_nodes)
                    if (std::abs(node->arclength - s) < TOLERANCE && std::abs(node->time - t) < TOLERANCE)
                        return node;
                return nullptr;
            }

            std::shared_ptr<NodeType> nearest(double s, double t) const
            {
                return std::dynamic_pointer_cast<NodeType>(state->getNearestNode(s, t));
            }
        };
    } // namespace

    TEST(StatefulModelDefaultOuterFactor, NearestNodeOnNode)
    {
        Grid grid;
        for (double s : {0.0, 0.2, 1.0})
            for (double t : {0.0, 0.1, 0.3})
                EXPECT_EQ(grid.nearest(s, t), grid.nodeAt(s, t)) << "s=" << s << " t=" << t;
    }

    TEST(StatefulModelDefaultOuterFactor, NearestNodeOffNode)
    {
        Grid grid;
        EXPECT_EQ(grid.nearest(0.05, 0.02), grid.nodeAt(0.0, 0.0));
        EXPECT_EQ(grid.nearest(0.15, 0.18), grid.nodeAt(0.2, 0.1));
        EXPECT_EQ(grid.nearest(0.61, 0.21), grid.nodeAt(1.0, 0.3));
        EXPECT_EQ(grid.nearest(0.59, 0.25), grid.nodeAt(0.2, 0.3));
    }

    TEST(StatefulModelDefaultOuterFactor, NearestNodeTiesGoLower)
    {
        Grid grid;
        EXPECT_EQ(grid.nearest(0.1, 0.0), grid.nodeAt(0.0, 0.0));  // arclength midpoint
        EXPECT_EQ(grid.nearest(0.6, 0.0), grid.nodeAt(0.2, 0.0));  // arclength midpoint
        EXPECT_EQ(grid.nearest(0.0, 0.05), grid.nodeAt(0.0, 0.0)); // time midpoint
        EXPECT_EQ(grid.nearest(0.0, 0.2), grid.nodeAt(0.0, 0.1));  // time midpoint
        EXPECT_EQ(grid.nearest(0.6, 0.2), grid.nodeAt(0.2, 0.1));  // both
    }

    TEST(StatefulModelDefaultOuterFactor, NearestNodeOutsideExtentIsNull)
    {
        Grid grid;
        EXPECT_EQ(grid.nearest(-0.01, 0.1), nullptr);
        EXPECT_EQ(grid.nearest(1.01, 0.1), nullptr);
        EXPECT_EQ(grid.nearest(0.5, -0.01), nullptr);
        EXPECT_EQ(grid.nearest(0.5, 0.31), nullptr);
        // Within TOLERANCE of the extent still resolves.
        EXPECT_EQ(grid.nearest(1.0 + 0.1 * TOLERANCE, 0.3 + 0.1 * TOLERANCE), grid.nodeAt(1.0, 0.3));

        const auto inner = makePoseFactor(Pose());
        EXPECT_EQ(grid.prior->getOuterFactor(0.31, 0.5, inner), nullptr);
        EXPECT_EQ(grid.prior->query(0.31, 0.5, grid.values), nullptr);
    }

    TEST(StatefulModelDefaultOuterFactor, EmptyStateIsNull)
    {
        auto state = std::make_shared<StateType>(std::vector<double>{0.0, 1.0});
        SnappingPrior prior(state);
        EXPECT_EQ(state->getNearestNode(0.0, 0.0), nullptr);
        EXPECT_EQ(prior.getOuterFactor(0.0, 0.0, makePoseFactor(Pose())), nullptr);
        EXPECT_EQ(prior.query(0.0, 0.0, gtsam::Values()), nullptr);
    }

    TEST(StatefulModelDefaultOuterFactor, NodeTypeWithoutInterpolatedKeysIsNull)
    {
        auto state = std::make_shared<Types::ContinuumRobotState<PlainNode>>(std::vector<double>{0.0, 1.0});
        PlainPrior prior(state);
        KeyManager km;
        gtsam::Values current, new_values;
        prior.extendState(km, 0.0, current, new_values);

        ASSERT_NE(state->getNearestNode(0.0, 0.0), nullptr);
        EXPECT_TRUE(state->getNearestNode(0.0, 0.0)->interpolatedKeys().empty());
        EXPECT_EQ(prior.getOuterFactor(0.0, 0.0, makePoseFactor(Pose())), nullptr);
        EXPECT_EQ(prior.query(0.0, 0.0, gtsam::Values()), nullptr);
    }

    TEST(StatefulModelDefaultOuterFactor, WNOANodeInterpolatedKeysMatchRowRanges)
    {
        KeyManager km;
        const NodeType node(km, 0.0, 0.0);
        const auto map = node.interpolatedKeys();
        ASSERT_EQ(map.size(), 3u);
        EXPECT_EQ(map[0], std::make_pair(Keys::Pose(), node.poseKey()));
        EXPECT_EQ(map[1], std::make_pair(Keys::Strain(), node.strainKey()));
        EXPECT_EQ(map[2], std::make_pair(Keys::Velocity(), node.velocityKey()));
        // Tangent order: each component's row range follows the previous one.
        int offset = 0;
        for (const auto &[interpolated, key] : map)
        {
            const auto [row, dim] = Keys::rowRange(interpolated);
            EXPECT_EQ(row, offset);
            offset += dim;
        }
        EXPECT_EQ(offset, static_cast<int>(NodeType::dim));
    }

    TEST(StatefulModelDefaultOuterFactor, FactorSnapsPoseFactorOntoNearestNode)
    {
        Grid grid;
        const auto node = grid.nodeAt(0.2, 0.1);
        const Pose measured(gtsam::Rot3::RzRyRx(0.3, -0.2, 0.1), gtsam::Point3(0.4, -0.1, 0.9));
        const auto inner = makePoseFactor(measured);

        auto factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(grid.prior->getOuterFactor(0.13, 0.28, inner));
        ASSERT_NE(factor, nullptr);
        ASSERT_NE(std::dynamic_pointer_cast<Types::NearestNodeFactor>(factor), nullptr);
        EXPECT_EQ(factor->keys(), gtsam::KeyVector{node->poseKey()});

        // The error is the inner factor's, evaluated on the snapped node.
        gtsam::Values inner_values;
        inner_values.insert(Keys::Pose(), grid.values.at<Pose>(node->poseKey()));
        EXPECT_TRUE(gtsam::assert_equal(inner->unwhitenedError(inner_values), factor->unwhitenedError(grid.values), 1e-12));
        EXPECT_NEAR(inner->error(inner_values), factor->error(grid.values), 1e-12);

        // The Jacobian is the inner factor's (identity through the snap) and correct.
        std::vector<gtsam::Matrix> H_inner(1), H(1);
        inner->unwhitenedError(inner_values, &H_inner);
        factor->unwhitenedError(grid.values, &H);
        EXPECT_TRUE(gtsam::assert_equal(H_inner[0], H[0], 1e-12));
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("NearestNodeFactor", *factor, grid.values, 1e-6, 1e-5));
    }

    TEST(StatefulModelDefaultOuterFactor, FactorKeepsInnerMask)
    {
        Grid grid;
        Eigen::MatrixXi mask = Eigen::MatrixXi::Ones(6, 1);
        mask(1, 0) = 0;
        mask(4, 0) = 0;
        const auto factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(
            grid.prior->getOuterFactor(0.0, 1.0, makePoseFactor(Pose(), mask)));
        ASSERT_NE(factor, nullptr);

        std::vector<gtsam::Matrix> H(1);
        const gtsam::Vector e = factor->unwhitenedError(grid.values, &H);
        EXPECT_EQ(e(1), 0.0);
        EXPECT_EQ(e(4), 0.0);
        EXPECT_TRUE(H[0].row(1).isZero());
        EXPECT_TRUE(H[0].row(4).isZero());
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("NearestNodeFactor", *factor, grid.values, 1e-6, 1e-5));
    }

    TEST(StatefulModelDefaultOuterFactor, FactorMapsSeveralKeysAndPassesThroughOuterKeys)
    {
        Grid grid;
        const auto node = grid.nodeAt(1.0, 0.3);
        const gtsam::Key landmark = gtsam::Symbol('l', 0);
        gtsam::Values values = grid.values;
        values.insert(landmark, Vector6(Vector6::Constant(0.25)));

        const auto inner = std::make_shared<StrainVelocityLandmarkFactor>(landmark);
        const auto factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(grid.prior->getOuterFactor(0.29, 0.9, inner));
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(factor->keys(), (gtsam::KeyVector{node->strainKey(), node->velocityKey(), landmark}));

        const Vector6 expected = values.at<Vector6>(node->strainKey()) + 2.0 * values.at<Vector6>(node->velocityKey()) - Vector6::Constant(0.25);
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(expected), factor->unwhitenedError(values), 1e-12));

        std::vector<gtsam::Matrix> H(3);
        factor->unwhitenedError(values, &H);
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(gtsam::Matrix66::Identity()), H[0]));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(2.0 * gtsam::Matrix66::Identity()), H[1]));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(-gtsam::Matrix66::Identity()), H[2]));
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("NearestNodeFactor", *factor, values, 1e-6, 1e-5));
    }

    TEST(StatefulModelDefaultOuterFactor, FactorRejectsNodeKeyReferencedTwice)
    {
        Grid grid;
        const auto node = grid.nodeAt(0.0, 0.0);
        // Inner factor on the interpolated strain key AND the node's own strain key.
        class Twice : public gtsam::NoiseModelFactorN<Vector6, Vector6>
        {
        public:
            Twice(gtsam::Key a, gtsam::Key b) : gtsam::NoiseModelFactorN<Vector6, Vector6>(gtsam::noiseModel::Unit::Create(6), a, b) {}
            gtsam::Vector evaluateError(const Vector6 &a, const Vector6 &b, gtsam::OptionalMatrixType, gtsam::OptionalMatrixType) const override { return a - b; }
        };
        const auto inner = std::make_shared<Twice>(Keys::Strain(), node->strainKey());
        EXPECT_THROW(grid.prior->getOuterFactor(0.0, 0.0, inner), std::runtime_error);
        EXPECT_THROW(Types::NearestNodeFactor(nullptr, node->interpolatedKeys()), std::runtime_error);
    }

    TEST(StatefulModelDefaultOuterFactor, QueryReturnsNearestNodeValues)
    {
        Grid grid;
        const auto node = grid.nodeAt(0.2, 0.3);
        Eigen::MatrixXd conditional, posterior_cov;
        auto result = std::dynamic_pointer_cast<ValuesType>(
            grid.prior->query(0.27, 0.35, grid.values, &conditional, nullptr, &posterior_cov));
        ASSERT_NE(result, nullptr);

        const ValuesType expected = node->getValues(grid.values);
        EXPECT_TRUE(gtsam::assert_equal(expected.pose(), result->pose()));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(expected.epsilon()), gtsam::Vector(result->epsilon())));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(expected.varpi()), gtsam::Vector(result->varpi())));

        EXPECT_EQ(conditional.rows(), static_cast<Eigen::Index>(NodeType::dim));
        EXPECT_EQ(conditional.cols(), static_cast<Eigen::Index>(NodeType::dim));
        EXPECT_TRUE(conditional.isZero());
        EXPECT_EQ(posterior_cov.size(), 0); // no posterior given
    }

    TEST(StatefulModelDefaultOuterFactor, QueryPosteriorIsNodeMarginal)
    {
        Grid grid;
        const auto node = grid.nodeAt(1.0, 0.1);
        const gtsam::KeyVector keys{node->poseKey(), node->strainKey(), node->velocityKey()};

        const Eigen::MatrixXd A = Eigen::MatrixXd::Random(18, 18);
        const Eigen::MatrixXd sigma = A * A.transpose() + Eigen::MatrixXd::Identity(18, 18);
        PosteriorCovariance posterior;
        for (size_t i = 0; i < 3; i++)
            for (size_t j = i; j < 3; j++)
                posterior.insert(keys[i], keys[j], sigma.block(6 * i, 6 * j, 6, 6));

        Eigen::MatrixXd posterior_cov;
        ASSERT_NE(grid.prior->query(0.12, 0.8, grid.values, nullptr, &posterior, &posterior_cov), nullptr);
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(sigma), gtsam::Matrix(posterior_cov), 1e-12));

        // A missing block leaves the posterior empty rather than partial.
        PosteriorCovariance partial;
        partial.insert(keys[0], keys[0], sigma.block(0, 0, 6, 6));
        posterior_cov = Eigen::MatrixXd::Ones(2, 2);
        ASSERT_NE(grid.prior->query(0.12, 0.8, grid.values, nullptr, &partial, &posterior_cov), nullptr);
        EXPECT_EQ(posterior_cov.size(), 0);
    }

    TEST(StatefulModelDefaultOuterFactor, QueryNodeMissingFromValuesIsNull)
    {
        Grid grid;
        const auto node = grid.nodeAt(0.0, 0.0);
        gtsam::Values values = grid.values;
        values.erase(node->velocityKey()); // e.g. marginalized out
        EXPECT_EQ(grid.prior->query(0.01, 0.01, values), nullptr);
        EXPECT_NE(grid.prior->query(0.01, 0.2, values), nullptr); // another node is unaffected
    }

    TEST(StatefulModelDefaultOuterFactor, WNOAPriorStillOverridesTheDefault)
    {
        auto state = std::make_shared<StateType>(std::vector<double>{0.0, 1.0});
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, ValuesType());
        KeyManager km;
        gtsam::Values values;
        for (double t : {0.0, 0.1})
        {
            gtsam::Values new_values;
            prior->extendState(km, t, values, new_values);
            values.insert(new_values);
        }
        const auto factor = prior->getOuterFactor(0.05, 0.5, makePoseFactor(Pose()));
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(std::dynamic_pointer_cast<Types::NearestNodeFactor>(factor), nullptr);
        EXPECT_NE(std::dynamic_pointer_cast<Priors::InterpolationFactor<Priors::WNOA::Interpolator<Pose>>>(factor), nullptr);
    }

    TEST(StatefulModelDefaultOuterFactor, EstimatorSnapsMeasurementsOntoNearestNode)
    {
        auto state = std::make_shared<StateType>(std::vector<double>{0.0, 0.2, 1.0});
        auto prior = std::make_shared<SnappingPrior>(state);
        Estimator estimator(1.0); // lag long enough to keep the snapped-to row (t = 0.1) in the window
        estimator.addPrior(prior);

        auto sensor = std::make_shared<Sensors::PoseSensor<ValuesType>>(Eigen::Matrix<double, 6, 6>::Identity(), prior);
        estimator.addSensor(sensor);

        const Pose measured(gtsam::Rot3::RzRyRx(0.2, 0.1, -0.3), gtsam::Point3(0.1, 0.2, 0.9));
        const Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
        // (s = 0.9, t = 0.12) snaps to the node at (1.0, 0.1).
        sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(0.9, 0.12, measured.matrix(), covariance, Eigen::MatrixXi::Ones(6, 1)));

        for (double t : {0.0, 0.1, 0.2})
            estimator.estimate(t);

        const auto node = std::dynamic_pointer_cast<NodeType>(state->getNearestNode(1.0, 0.1));
        ASSERT_NE(node, nullptr);
        const gtsam::Values values = estimator.getCurrentValues();
        EXPECT_TRUE(gtsam::assert_equal(measured, values.at<Pose>(node->poseKey()), 1e-3));

        // Nodes without a measurement stay at their (loose) unary prior.
        const auto other = std::dynamic_pointer_cast<NodeType>(state->getNearestNode(0.0, 0.1));
        EXPECT_TRUE(gtsam::assert_equal(Pose(), values.at<Pose>(other->poseKey()), 1e-6));
    }
} // namespace Spacetime
