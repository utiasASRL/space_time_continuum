#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/bias_models/Bias.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime
{
    namespace
    {
        using Vector3 = gtsam::Vector3;
        using Matrix3 = gtsam::Matrix3;
        using Model3 = Biases::BiasModel<Vector3>;
        using Keys3 = Biases::BiasKeys<Vector3>;

        Biases::BiasOptions<Vector3> options3()
        {
            Biases::BiasOptions<Vector3> options;
            options.initial = Vector3(0.1, -0.2, 0.3);
            options.initial_covariance = Vector3(1e-2, 2e-2, 3e-2).asDiagonal();
            options.random_walk_psd = Vector3(1e-4, 2e-4, 3e-4).asDiagonal();
            return options;
        }

        /// Extends `model` at `time`, merging the new values into `values`, and returns its factors.
        Types::Model::Factors extend(Types::Model &model, KeyManager &km, double time, gtsam::Values &values)
        {
            gtsam::Values new_values;
            auto factors = model.extendState(km, time, values, new_values);
            values.insert(new_values);
            return factors;
        }

        /// The covariance of a Gaussian factor's noise model.
        gtsam::Matrix covarianceOf(const gtsam::NonlinearFactor &factor)
        {
            const auto *noise_factor = dynamic_cast<const gtsam::NoiseModelFactor *>(&factor);
            EXPECT_NE(noise_factor, nullptr);
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(noise_factor->noiseModel());
            EXPECT_NE(gaussian, nullptr);
            return gaussian->covariance();
        }

        /// A sensor measuring a bias directly, at its placeholder key (snapped to the nearest bias node).
        struct DirectBiasSensor : public Types::SensorBase
        {
            using Types::SensorBase::SensorBase;
            gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
            {
                return std::make_shared<gtsam::PriorFactor<Vector3>>(Keys3::Bias(), Vector3(measurement->value),
                                                                     gtsam::noiseModel::Gaussian::Covariance(measurement->covariance));
            }
        };
    } // namespace

    TEST(BiasModel, ExtendStateAddsOneNodePerNewTime)
    {
        auto model = std::make_shared<Model3>(options3());
        KeyManager km;
        gtsam::Values values;
        extend(*model, km, 0.0, values);
        extend(*model, km, 0.1, values);
        EXPECT_TRUE(extend(*model, km, 0.1, values).empty());  // repeated time
        EXPECT_TRUE(extend(*model, km, 0.05, values).empty()); // backwards
        ASSERT_EQ(model->state()->estimation_nodes.size(), 2u);
        EXPECT_EQ(model->state()->times(), (std::vector<double>{0.0, 0.1}));
        EXPECT_EQ(values.size(), 2u);
        for (const auto &node : model->state()->estimation_nodes)
        {
            EXPECT_EQ(gtsam::Symbol(node->biasKey()).chr(), 'b');
            EXPECT_TRUE(values.exists(node->biasKey()));
        }
    }

    TEST(BiasModel, FirstNodeGetsAPriorAndLaterNodesARandomWalk)
    {
        const auto options = options3();
        auto model = std::make_shared<Model3>(options);
        KeyManager km;
        gtsam::Values values;

        const auto first = extend(*model, km, 0.0, values);
        ASSERT_EQ(first.size(), 1u);
        auto prior = std::dynamic_pointer_cast<gtsam::PriorFactor<Vector3>>(first.front());
        ASSERT_NE(prior, nullptr);
        EXPECT_TRUE(gtsam::assert_equal(options.initial, prior->prior()));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(options.initial_covariance), covarianceOf(*prior), 1e-12));

        const auto second = extend(*model, km, 0.25, values);
        ASSERT_EQ(second.size(), 1u);
        auto walk = std::dynamic_pointer_cast<gtsam::BetweenFactor<Vector3>>(second.front());
        ASSERT_NE(walk, nullptr);
        EXPECT_TRUE(gtsam::assert_equal(Vector3(Vector3::Zero()), walk->measured()));
        const auto &nodes = model->state()->estimation_nodes;
        EXPECT_EQ(walk->key1(), nodes[0]->biasKey());
        EXPECT_EQ(walk->key2(), nodes[1]->biasKey());
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(0.25 * options.random_walk_psd), covarianceOf(*walk), 1e-12));
    }

    TEST(BiasModel, InitialGuessCarriesTheCurrentEstimateForward)
    {
        auto model = std::make_shared<Model3>(options3());
        KeyManager km;
        gtsam::Values values;
        extend(*model, km, 0.0, values);
        const gtsam::Key first = model->state()->estimation_nodes.front()->biasKey();
        EXPECT_TRUE(gtsam::assert_equal(options3().initial, values.at<Vector3>(first)));

        values.update(first, Vector3(1.0, 2.0, 3.0));
        extend(*model, km, 0.1, values);
        const gtsam::Key second = model->state()->estimation_nodes.back()->biasKey();
        EXPECT_TRUE(gtsam::assert_equal(Vector3(1.0, 2.0, 3.0), values.at<Vector3>(second)));
    }

    TEST(BiasModel, BracketAndWeights)
    {
        auto model = std::make_shared<Model3>(options3());
        KeyManager km;
        gtsam::Values values;
        for (double t : {0.0, 0.1, 0.3})
            extend(*model, km, t, values);
        const auto &nodes = model->state()->estimation_nodes;

        // Between two nodes: both, linearly.
        const auto between = model->bracket(0.2);
        ASSERT_TRUE(between.has_value());
        EXPECT_EQ(between->node0, nodes[1]);
        EXPECT_EQ(between->node1, nodes[2]);
        EXPECT_NEAR(between->tau, 0.5, 1e-12);
        const auto w_between = model->weights(0.15);
        ASSERT_EQ(w_between.size(), 2u);
        EXPECT_EQ(w_between[0].first, nodes[1]->biasKey());
        EXPECT_NEAR(w_between[0].second, 0.75, 1e-12);
        EXPECT_EQ(w_between[1].first, nodes[2]->biasKey());
        EXPECT_NEAR(w_between[1].second, 0.25, 1e-12);

        // On a node: that node only.
        const auto on_node = model->weights(0.1);
        ASSERT_EQ(on_node.size(), 1u);
        EXPECT_EQ(on_node[0].first, nodes[1]->biasKey());
        EXPECT_NEAR(on_node[0].second, 1.0, 1e-12);

        // Past the last node: held at it.
        const auto past = model->bracket(0.4);
        ASSERT_TRUE(past.has_value());
        EXPECT_EQ(past->node0, nodes[2]);
        EXPECT_EQ(past->node1, nullptr);
        const auto w_past = model->weights(0.4);
        ASSERT_EQ(w_past.size(), 1u);
        EXPECT_EQ(w_past[0].first, nodes[2]->biasKey());
        EXPECT_NEAR(w_past[0].second, 1.0, 1e-12);

        // Before the first node: unresolved.
        EXPECT_FALSE(model->bracket(-0.1).has_value());
        EXPECT_TRUE(model->weights(-0.1).empty());

        // No nodes at all.
        EXPECT_FALSE(Model3(options3()).bracket(0.0).has_value());
    }

    TEST(BiasInterpolator, VectorInterpolationIsLinear)
    {
        const Vector3 b0(1.0, 2.0, 3.0), b1(3.0, 2.0, -1.0);
        gtsam::Matrix H0, H1;
        const Vector3 b = Biases::BiasInterpolator<Vector3>::interpolate(b0, b1, 0.25, &H0, &H1);
        EXPECT_TRUE(gtsam::assert_equal(Vector3(0.75 * b0 + 0.25 * b1), b, 1e-12));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(0.75 * Matrix3::Identity()), H0, 1e-12));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(0.25 * Matrix3::Identity()), H1, 1e-12));
    }

    TEST(BiasInterpolator, PoseInterpolationIsGeodesicWithExactJacobians)
    {
        using Pose = gtsam::Pose3;
        const Pose b0(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3), gtsam::Point3(0.1, 0.2, -0.1));
        const Pose b1(gtsam::Rot3::RzRyRx(-0.2, 0.1, 0.4), gtsam::Point3(-0.1, 0.3, 0.2));
        const double tau = 0.3;
        gtsam::Matrix H0, H1;
        const Pose b = Biases::BiasInterpolator<Pose>::interpolate(b0, b1, tau, &H0, &H1);
        EXPECT_TRUE(gtsam::assert_equal(b0 * Pose::Expmap(tau * Pose::Logmap(b0.between(b1))), b, 1e-12));

        auto f = [tau](const Pose &x, const Pose &y) { return Biases::BiasInterpolator<Pose>::interpolate(x, y, tau); };
        EXPECT_TRUE(gtsam::assert_equal(gtsam::numericalDerivative21<Pose, Pose, Pose>(f, b0, b1), H0, 1e-7));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::numericalDerivative22<Pose, Pose, Pose>(f, b0, b1), H1, 1e-7));
    }

    TEST(BiasModel, QueryInterpolatesWithABrownianBridgeCovariance)
    {
        const auto options = options3();
        auto model = std::make_shared<Model3>(options);
        KeyManager km;
        gtsam::Values values;
        extend(*model, km, 0.0, values);
        extend(*model, km, 0.2, values);
        const auto &nodes = model->state()->estimation_nodes;
        values.update(nodes[0]->biasKey(), Vector3(1.0, 0.0, -1.0));
        values.update(nodes[1]->biasKey(), Vector3(3.0, 2.0, 1.0));

        Eigen::MatrixXd conditional;
        auto queried = std::dynamic_pointer_cast<Biases::BiasValues<Vector3>>(model->query(0.05, 0.0, values, &conditional));
        ASSERT_NE(queried, nullptr);
        EXPECT_TRUE(gtsam::assert_equal(Vector3(1.5, 0.5, -0.5), queried->bias(), 1e-12));
        // A random walk conditioned on both ends: tau (1 - tau) dt Q.
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(0.25 * 0.75 * 0.2 * options.random_walk_psd), gtsam::Matrix(conditional), 1e-12));

        // On a node, the node itself with no added uncertainty.
        auto on_node = std::dynamic_pointer_cast<Biases::BiasValues<Vector3>>(model->query(0.2, 0.0, values, &conditional));
        ASSERT_NE(on_node, nullptr);
        EXPECT_TRUE(gtsam::assert_equal(Vector3(3.0, 2.0, 1.0), on_node->bias(), 1e-12));
        EXPECT_TRUE(conditional.isZero(1e-15));

        EXPECT_EQ(model->query(-1.0, 0.0, values), nullptr);
    }

    TEST(BiasModel, DefaultOuterFactorSnapsToTheNearestNode)
    {
        auto model = std::make_shared<Model3>(options3());
        KeyManager km;
        gtsam::Values values;
        extend(*model, km, 0.0, values);
        extend(*model, km, 0.1, values);

        auto inner = std::make_shared<gtsam::PriorFactor<Vector3>>(Keys3::Bias(), Vector3::Zero(), gtsam::noiseModel::Unit::Create(3));
        auto outer = model->getOuterFactor(0.04, 0.0, inner);
        ASSERT_NE(outer, nullptr);
        ASSERT_EQ(outer->keys().size(), 1u);
        EXPECT_EQ(outer->keys().front(), model->state()->estimation_nodes.front()->biasKey());
    }

    TEST(BiasModel, CovarianceGroupsAreOnePerLiveNode)
    {
        auto model = std::make_shared<Model3>(options3());
        KeyManager km;
        gtsam::Values values;
        for (double t : {0.0, 0.1, 0.2})
            extend(*model, km, t, values);
        values.erase(model->state()->estimation_nodes.front()->biasKey()); // e.g. marginalized
        const auto groups = model->covarianceKeyGroups(values);
        ASSERT_EQ(groups.size(), 2u);
        for (const auto &group : groups)
        {
            ASSERT_EQ(group.size(), 1u);
            EXPECT_TRUE(values.exists(group.front()));
        }
    }

    TEST(BiasModel, WorksForALieGroupBias)
    {
        using Pose = gtsam::Pose3;
        Biases::BiasOptions<Pose> options;
        options.initial = Pose(gtsam::Rot3::Rz(0.1), gtsam::Point3(0.0, 0.1, 0.0));
        auto model = std::make_shared<Biases::BiasModel<Pose>>(options);
        KeyManager km;
        gtsam::Values values;
        ASSERT_EQ(extend(*model, km, 0.0, values).size(), 1u);
        const auto walk = extend(*model, km, 0.1, values);
        ASSERT_EQ(walk.size(), 1u);
        EXPECT_NE(std::dynamic_pointer_cast<gtsam::BetweenFactor<Pose>>(walk.front()), nullptr);
        EXPECT_TRUE(gtsam::assert_equal(options.initial, values.at<Pose>(model->state()->estimation_nodes.back()->biasKey())));
    }

    TEST(BiasModel, EstimatorTracksAConstantBias)
    {
        Biases::BiasOptions<Vector3> options;
        options.initial = Vector3::Zero();
        options.initial_covariance = Matrix3::Identity();
        options.random_walk_psd = 1e-4 * Matrix3::Identity();
        auto model = std::make_shared<Model3>(options);
        auto sensor = std::make_shared<DirectBiasSensor>(model);

        Estimator estimator(10.0);
        estimator.addModel(model);
        estimator.addSensor(sensor);

        const Vector3 bias(0.2, -0.1, 0.05);
        for (int k = 0; k <= 10; k++)
            sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(0.0, 0.1 * k, bias, 1e-4 * Eigen::Matrix3d::Identity()));
        for (int k = 0; k <= 10; k++)
            estimator.estimate(0.1 * k);

        auto queried = std::dynamic_pointer_cast<Biases::BiasValues<Vector3>>(model->query(1.0, 0.0, estimator.getCurrentValues()));
        ASSERT_NE(queried, nullptr);
        EXPECT_TRUE(gtsam::assert_equal(bias, queried->bias(), 1e-3));
    }
} // namespace Spacetime
