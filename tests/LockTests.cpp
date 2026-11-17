#include <cstdio>
#include <fstream>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/LockedFactor.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        const Pose kBase(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0.1, 0.2, 0.3));
        const Vector6 kStrain = (Vector6() << 0, 0, 0, 0, 0, 1.0).finished();
        const Vector6 kZero = Vector6::Zero();

        /// A 4-node, 1 m robot with a legacy-factor WNOA prior; the base drifts at the reference velocity.
        struct Fixture
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 4);
            std::shared_ptr<Priors::WNOA::Prior<Pose>> prior;

            Fixture()
            {
                ValueType reference;
                reference.pose(kBase);
                reference.epsilon(kStrain);
                reference.varpi((Vector6() << 0, 0, 0.5, 0, 0.1, 0).finished());
                Priors::WNOA::PriorOptions options;
                options.factor_type = Priors::WNOA::FactorType::Legacy;
                prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, reference, Eigen::Matrix<double, 18, 18>::Identity() * 1e-2,
                                                                    Matrix6::Identity(), Matrix6::Identity(), Matrix6::Identity(), Matrix6::Identity(), options);
            }

            std::shared_ptr<NodeType> node(unsigned int row, unsigned int n) const
            {
                return std::dynamic_pointer_cast<NodeType>(state->estimation_nodes.at(row * 4 + n));
            }

            /// Batch estimator over rows at t = 0, 0.1, ..., with a tip pose measurement per row.
            std::unique_ptr<Estimator> solve(int rows, bool covariance = true)
            {
                EstimatorOptions options;
                options.smootherLag = Estimator::BATCH_LAG;
                options.compute_covariance = covariance;
                auto estimator = std::make_unique<Estimator>(options);
                estimator->addPrior(prior);
                const Matrix6 R = Matrix6::Identity() * 1e-4;
                auto sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
                for (int k = 0; k < rows; k++)
                    sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(
                        1.0, 0.1 * k, (kBase * Pose(gtsam::Rot3(), gtsam::Point3(0, 0, 1.02))).matrix(), R, Eigen::MatrixXi::Ones(6, 1)));
                estimator->addSensor(sensor);
                for (int k = 0; k < rows; k++)
                    estimator->estimate(0.1 * k);
                estimator->finalize();
                return estimator;
            }
        };

        std::string writeTempConfig(const std::string &name, const std::string &contents)
        {
            const std::string path = testing::TempDir() + name;
            std::ofstream(path) << contents;
            return path;
        }
    } // namespace

    // ------------------------------------------------------------------
    // ContinuumRobot::lock
    // ------------------------------------------------------------------

    TEST(Lock, RejectsInvalidLocks)
    {
        Fixture f;
        EXPECT_THROW(f.state->lock("pose", 0.3, kBase), std::invalid_argument);     // not a node arclength
        EXPECT_THROW(f.state->lock("pose", -0.5, kBase), std::invalid_argument);    // only -1 means the tip
        EXPECT_THROW(f.state->lock("curvature", 0.0, kZero), std::invalid_argument); // no such variable
        EXPECT_THROW(f.state->lock("pose", 0.0, kZero), std::invalid_argument);     // wrong value type
        EXPECT_THROW(f.state->lock("strain", 0.0, kBase), std::invalid_argument);
        EXPECT_NO_THROW(f.state->lock("pose", 1.0 / 3.0, kBase));
    }

    TEST(Lock, WithoutATimeEveryRowIsLocked)
    {
        Fixture f;
        f.state->lock("strain", -1.0, kStrain); // the tip
        KeyManager km;
        for (double t : {0.0, 0.1, 0.2})
            f.state->addStateAtTime(km, t);
        EXPECT_EQ(f.state->lockedValues().size(), 3u);
        for (unsigned int row = 0; row < 3; row++)
        {
            EXPECT_TRUE(f.state->lockedValues().exists(f.node(row, 3)->strainKey()));
            EXPECT_FALSE(f.state->lockedValues().exists(f.node(row, 0)->strainKey()));
        }
    }

    TEST(Lock, WithATimeOnlyThatRowIsLocked)
    {
        Fixture f;
        f.state->lock("pose", 0.0, kBase, 0.1);
        f.state->lock("velocity", 0.0, kZero, 0.15); // between rows: never applied (warns)
        KeyManager km;
        for (double t : {0.0, 0.1, 0.2})
            f.state->addStateAtTime(km, t);
        ASSERT_EQ(f.state->lockedValues().size(), 1u);
        EXPECT_TRUE(f.state->lockedValues().exists(f.node(1, 0)->poseKey()));
        EXPECT_TRUE(f.state->lockedValues().at<Pose>(f.node(1, 0)->poseKey()).equals(kBase));
    }

    // ------------------------------------------------------------------
    // LockedFactor
    // ------------------------------------------------------------------

    TEST(Lock, LockedFactorMatchesTheInnerFactorOnItsFreeKeys)
    {
        KeyManager km;
        auto node0 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node1 = std::make_shared<NodeType>(km, 0.3, 0.0);
        auto inner = Priors::WNOA::BinarySpaceFactor::createFactor<Pose>(node0, node1, Matrix6::Identity(), Matrix6::Identity() * 2.0);

        ValueType v0, v1;
        v0.setRandom();
        v1.setRandom();
        gtsam::Values all, locked;
        node0->setValues(all, v0);
        node1->setValues(all, v1);
        locked.insert(node0->poseKey(), v0.pose());
        locked.insert(node0->velocityKey(), v0.varpi());

        LockedFactor factor(inner, locked);
        EXPECT_EQ(factor.keys().size(), 4u);
        EXPECT_FALSE(std::count(factor.keys().begin(), factor.keys().end(), node0->poseKey()));

        gtsam::Values free = all;
        free.erase(node0->poseKey());
        free.erase(node0->velocityKey());
        EXPECT_DOUBLE_EQ(factor.error(free), inner->error(all));

        // Its Jacobians are the inner ones restricted to the free keys.
        auto linear = std::dynamic_pointer_cast<gtsam::JacobianFactor>(factor.linearize(free));
        auto inner_linear = std::dynamic_pointer_cast<gtsam::JacobianFactor>(inner->linearize(all));
        ASSERT_NE(linear, nullptr);
        for (const gtsam::Key key : factor.keys())
            EXPECT_TRUE(linear->getA(linear->find(key)).isApprox(inner_linear->getA(inner_linear->find(key))));
        EXPECT_TRUE(linear->getb().isApprox(inner_linear->getb()));
    }

    TEST(Lock, TouchesDetectsLockedKeys)
    {
        gtsam::Values locked;
        locked.insert(1, Pose());
        EXPECT_TRUE(LockedFactor::touches(gtsam::PriorFactor<Pose>(1, Pose(), gtsam::noiseModel::Unit::Create(6)), locked));
        EXPECT_FALSE(LockedFactor::touches(gtsam::PriorFactor<Pose>(2, Pose(), gtsam::noiseModel::Unit::Create(6)), locked));
        EXPECT_THROW(LockedFactor(nullptr, locked), std::invalid_argument);
    }

    // ------------------------------------------------------------------
    // Estimator
    // ------------------------------------------------------------------

    TEST(Lock, LockedVariablesAreConstantsOfTheEstimate)
    {
        Fixture f;
        f.state->lock("pose", 0.0, kBase);
        f.state->lock("velocity", 0.0, kZero);
        auto estimator = f.solve(3);

        for (unsigned int row = 0; row < 3; row++)
        {
            auto base = f.node(row, 0);
            // Not variables of the graph ...
            EXPECT_FALSE(estimator->getLinearizationPoint().exists(base->poseKey()));
            EXPECT_FALSE(estimator->getLinearizationPoint().exists(base->velocityKey()));
            EXPECT_TRUE(estimator->getLinearizationPoint().exists(base->strainKey()));
            // ... but part of the estimate, exactly at their locked value, with zero covariance.
            const ValueType values = base->getValues(estimator->getCurrentValues());
            EXPECT_TRUE(values.pose().equals(kBase, 1e-12));
            EXPECT_TRUE(values.varpi().isZero());
            EXPECT_TRUE(estimator->covariance(base->poseKey()).isZero());
            EXPECT_TRUE(estimator->covariance(base->poseKey(), f.node(row, 3)->poseKey()).isZero());
            EXPECT_GT(estimator->covariance(base->strainKey()).norm(), 0.0);
        }
    }

    TEST(Lock, CovarianceExtractionAndQueriesHandleLockedCorners)
    {
        Fixture f;
        f.state->lock("pose", 0.0, kBase);
        f.state->lock("velocity", 0.0, kZero);
        auto estimator = f.solve(3);
        auto posterior = estimator->getCurrentCovariance();
        ASSERT_NE(posterior, nullptr);

        // On the locked base: a zero pose covariance; between rows near the base: small but finite.
        Eigen::MatrixXd at_base, near_base;
        ASSERT_NE(f.prior->query(0.1, 0.0, estimator->getCurrentValues(), nullptr, posterior.get(), &at_base), nullptr);
        ASSERT_EQ(at_base.rows(), 18);
        EXPECT_LT((at_base.block<6, 6>(0, 0).norm()), 1e-12);
        ASSERT_NE(f.prior->query(0.05, 0.1, estimator->getCurrentValues(), nullptr, posterior.get(), &near_base), nullptr);
        EXPECT_GT((near_base.block<6, 6>(0, 0).norm()), 0.0);
        EXPECT_TRUE(near_base.allFinite());
    }

    TEST(Lock, LockingEverythingOnAFactorDropsIt)
    {
        // A single-node robot with its whole state locked leaves no variables.
        auto state = std::make_shared<StateType>(std::vector<double>{0.0});
        ValueType reference;
        reference.pose(kBase);
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, reference);
        state->lock("pose", 0.0, kBase);
        state->lock("strain", 0.0, kZero);
        state->lock("velocity", 0.0, kZero);
        Estimator estimator(Estimator::BATCH_LAG);
        estimator.addPrior(prior);
        estimator.estimate(0.0);
        estimator.finalize();
        EXPECT_EQ(estimator.getFactors().size(), 0u);
        EXPECT_EQ(estimator.getCurrentValues().size(), 3u);
    }

    TEST(Lock, SlidingWindowKeepsRecentLockedValuesAndPrunesMarginalizedOnes)
    {
        Fixture f;
        f.state->lock("pose", 0.0, kBase);
        Estimator estimator(0.15);
        estimator.addPrior(f.prior);
        for (double t : {0.0, 0.1, 0.2, 0.3, 0.4})
            estimator.estimate(t);

        // Rows older than current time - lag (0.4 - 0.15) are marginalized, and so are
        // their locked constants; the ones inside the window stay, at their value.
        const gtsam::Values values = estimator.getCurrentValues();
        for (unsigned int row : {0u, 1u, 2u})
        {
            EXPECT_FALSE(values.exists(f.node(row, 0)->poseKey())) << "row " << row;
            EXPECT_FALSE(values.exists(f.node(row, 1)->poseKey())) << "row " << row; // same rule as the variables
        }
        for (unsigned int row : {3u, 4u})
        {
            ASSERT_TRUE(values.exists(f.node(row, 0)->poseKey())) << "row " << row;
            EXPECT_TRUE(values.at<Pose>(f.node(row, 0)->poseKey()).equals(kBase, 1e-12));
            EXPECT_TRUE(values.exists(f.node(row, 1)->poseKey()));
        }
    }

    TEST(Lock, IncrementalSmootherPrunesLockedValuesLikeTheBatchSmoother)
    {
        Fixture f;
        f.state->lock("pose", 0.0, kBase);
        EstimatorOptions options;
        options.smootherLag = 0.15;
        options.smoother = EstimatorOptions::Smoother::Incremental;
        Estimator estimator(options);
        estimator.addPrior(f.prior);
        for (double t : {0.0, 0.1, 0.2, 0.3, 0.4})
            estimator.estimate(t);

        const gtsam::Values values = estimator.getCurrentValues();
        for (unsigned int row : {0u, 1u, 2u})
        {
            EXPECT_FALSE(values.exists(f.node(row, 0)->poseKey())) << "row " << row;
            EXPECT_FALSE(values.exists(f.node(row, 1)->poseKey())) << "row " << row;
        }
        for (unsigned int row : {3u, 4u})
        {
            ASSERT_TRUE(values.exists(f.node(row, 0)->poseKey())) << "row " << row;
            EXPECT_TRUE(values.at<Pose>(f.node(row, 0)->poseKey()).equals(kBase, 1e-12));
            EXPECT_FALSE(estimator.getLinearizationPoint().exists(f.node(row, 0)->poseKey())); // a constant, not a variable
            EXPECT_TRUE(values.exists(f.node(row, 1)->poseKey()));
        }
    }

    // ------------------------------------------------------------------
    // ConfigLoader
    // ------------------------------------------------------------------

    TEST(Lock, ConfigLoaderAppliesTheLockList)
    {
        const std::string path = writeTempConfig("locks.json", R"({ "demo_data": { "locks": [
            { "type": "pose", "s": 0, "value": [1,0,0,0, 0,1,0,0, 0,0,1,0.5, 0,0,0,1] },
            { "type": "strain", "s": -1, "t": 0.1, "value": [0,0,0, 0,0,1] }
        ] } })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        Fixture f;
        loader.loadLocks(*f.state);
        KeyManager km;
        for (double t : {0.0, 0.1})
            f.state->addStateAtTime(km, t);
        EXPECT_EQ(f.state->lockedValues().size(), 3u); // base pose x2, tip strain at t = 0.1
        EXPECT_TRUE(f.state->lockedValues().at<Pose>(f.node(0, 0)->poseKey()).equals(Pose(gtsam::Rot3(), gtsam::Point3(0, 0, 0.5))));
        EXPECT_TRUE(f.state->lockedValues().at<gtsam::Vector6>(f.node(1, 3)->strainKey()).isApprox(kStrain));
        std::remove(path.c_str());
    }

    TEST(Lock, ConfigLoaderRejectsMalformedLocks)
    {
        for (const std::string &locks : {R"([{ "type": "pose", "s": 0, "value": [1, 2] }])",
                                         R"([{ "type": "velocity", "s": 0, "value": [1, 2, 3] }])",
                                         R"([{ "type": "pose", "value": [] }])",
                                         R"({ "type": "pose" })"})
        {
            const std::string path = writeTempConfig("bad_locks.json", R"({ "demo_data": { "locks": )" + locks + "} }");
            Utilities::ConfigLoader loader;
            ASSERT_TRUE(loader.loadConfig(path));
            Fixture f;
            EXPECT_THROW(loader.loadLocks(*f.state), std::invalid_argument) << locks;
            std::remove(path.c_str());
        }
    }
} // namespace Spacetime
