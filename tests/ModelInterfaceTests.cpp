#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/estimator/types/Model.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using Factors = std::vector<std::shared_ptr<gtsam::NonlinearFactor>>;

        /// A WNOA prior that logs its extendState calls into `log`.
        class RecordingPrior : public Priors::WNOA::Prior<Pose>
        {
        public:
            RecordingPrior(std::shared_ptr<StateType> state, std::vector<std::string> &log)
                : Priors::WNOA::Prior<Pose>(state, ValueType()), m_log(log) {}

            Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
            {
                m_log.push_back("extend:prior");
                return Priors::WNOA::Prior<Pose>::extendState(km, time, current_values, new_values);
            }

            Factors getNewFactors(double, gtsam::Values &) override
            {
                m_log.push_back("factors:prior");
                return {};
            }

        private:
            std::vector<std::string> &m_log;
        };

        /// A model with no factors that logs every call it receives into `log`.
        class RecordingModel : public Types::Model
        {
        public:
            RecordingModel(std::string name, std::vector<std::string> &log) : m_name(std::move(name)), m_log(log) {}

            Factors extendState(KeyManager &, double, const gtsam::Values &, gtsam::Values &) override
            {
                m_log.push_back("extend:" + m_name);
                return {};
            }

            Factors getNewFactors(double, gtsam::Values &) override
            {
                m_log.push_back("factors:" + m_name);
                return {};
            }

            void onMarginalization(double cutoff, const gtsam::Values &) override { cutoffs.push_back(cutoff); }

            std::vector<double> cutoffs;

        private:
            std::string m_name;
            std::vector<std::string> &m_log;
        };

        std::shared_ptr<StateType> makeState() { return std::make_shared<StateType>(1.0, 2); }

        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior() { return std::make_shared<Priors::WNOA::Prior<Pose>>(makeState(), ValueType()); }

        std::shared_ptr<Types::MeasurementBase> makeMeasurement(double time)
        {
            return std::make_shared<Types::MeasurementBase>(0.0, time, Eigen::Vector2d(1.0, 2.0), Eigen::Matrix2d::Identity());
        }
    } // namespace

    namespace
    {
        /// A model with one scalar variable x0, measured at 1; it locks x0 at 5 if `lock` is set,
        /// and asks for x0's covariance.
        class ScalarModel : public Types::Model
        {
        public:
            explicit ScalarModel(bool lock) : m_lock(lock) {}

            Factors extendState(KeyManager &km, double time, const gtsam::Values &, gtsam::Values &new_values) override
            {
                if (m_key)
                    return {};
                m_key = km.nextKey('x', time);
                new_values.insert(*m_key, gtsam::Vector1(1.0));
                if (m_lock)
                    m_locked.insert(*m_key, gtsam::Vector1(5.0));
                return {std::make_shared<gtsam::PriorFactor<gtsam::Vector1>>(*m_key, gtsam::Vector1(1.0), gtsam::noiseModel::Isotropic::Sigma(1, 0.5))};
            }

            const gtsam::Values &lockedValues() const override { return m_locked; }

            std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
            {
                if (m_key && values.exists(*m_key))
                    return {{*m_key}};
                return {};
            }

            gtsam::Key key() const { return *m_key; }

        private:
            bool m_lock;
            std::optional<gtsam::Key> m_key;
            gtsam::Values m_locked;
        };
    } // namespace

    // Any model, not only a prior, can hold variables fixed.
    TEST(ModelInterface, AnyModelCanLockVariables)
    {
        auto model = std::make_shared<ScalarModel>(true);
        Estimator estimator;
        estimator.addModel(model);
        estimator.estimate(0.0);
        EXPECT_DOUBLE_EQ(estimator.getCurrentValues().at<gtsam::Vector1>(model->key())(0), 5.0);
        EXPECT_TRUE(estimator.covariance(model->key()).isZero());
    }

    // Any model, not only a prior, chooses covariance blocks to extract.
    TEST(ModelInterface, AnyModelCanAskForCovarianceBlocks)
    {
        auto model = std::make_shared<ScalarModel>(false);
        EstimatorOptions options;
        options.compute_covariance = true;
        Estimator estimator(options);
        estimator.addModel(model);
        estimator.estimate(0.0);
        auto covariance = estimator.getCurrentCovariance();
        ASSERT_NE(covariance, nullptr);
        Eigen::MatrixXd block;
        ASSERT_TRUE(covariance->joint({model->key()}, block));
        EXPECT_NEAR(block(0, 0), 0.25, 1e-9);
    }

    // Every model is extended before any is asked for factors; both phases run the models in
    // the order they were added, whatever their type.
    TEST(ModelInterface, EstimatorExtendsEveryModelThenCollectsFactorsInOrder)
    {
        std::vector<std::string> log;
        auto first = std::make_shared<RecordingModel>("a", log);
        auto prior = std::make_shared<RecordingPrior>(makeState(), log);
        auto second = std::make_shared<RecordingModel>("b", log);

        Estimator estimator;
        estimator.addModel(first);
        estimator.addModel(prior);
        estimator.addModel(second);
        estimator.estimate(0.0);

        const std::vector<std::string> expected = {"extend:a", "extend:prior", "extend:b", "factors:a", "factors:prior", "factors:b"};
        EXPECT_EQ(log, expected);
    }

    TEST(ModelInterface, OnMarginalizationReachesEveryModel)
    {
        std::vector<std::string> log;
        auto model = std::make_shared<RecordingModel>("a", log);
        Estimator estimator(1.0);
        estimator.addPrior(makePrior());
        estimator.addModel(model);
        estimator.estimate(0.0);
        estimator.estimate(0.1);
        EXPECT_EQ(model->cutoffs.size(), 2u);

        Estimator batch(Estimator::BATCH_LAG);
        auto batch_model = std::make_shared<RecordingModel>("b", log);
        batch.addPrior(makePrior());
        batch.addModel(batch_model);
        batch.estimate(0.0);
        EXPECT_TRUE(batch_model->cutoffs.empty()); // nothing is solved before finalize()
        batch.finalize();
        ASSERT_EQ(batch_model->cutoffs.size(), 1u);
        EXPECT_TRUE(std::isinf(batch_model->cutoffs.front()));
    }

    TEST(MeasurementQueue, DrainsUpToTimeInOrder)
    {
        Types::MeasurementQueue queue;
        for (double t : {0.0, 0.1, 0.2, 0.3})
            queue.addMeasurement(makeMeasurement(t));

        const auto drained = queue.drainUpTo(0.2); // boundary included
        ASSERT_EQ(drained.size(), 3u);
        EXPECT_DOUBLE_EQ(drained[0]->time, 0.0);
        EXPECT_DOUBLE_EQ(drained[1]->time, 0.1);
        EXPECT_DOUBLE_EQ(drained[2]->time, 0.2);
        ASSERT_EQ(queue.size(), 1u);
        EXPECT_DOUBLE_EQ(queue.peekAll().front()->time, 0.3);

        EXPECT_TRUE(queue.drainUpTo(0.25).empty());
        EXPECT_EQ(queue.drainUpTo(1.0).size(), 1u);
        EXPECT_TRUE(queue.empty());
    }

    // As for SensorBase: a callback first sees the measurements already queued, then every new one.
    TEST(MeasurementQueue, CallbacksSeeQueuedAndNewMeasurements)
    {
        Types::MeasurementQueue queue;
        queue.addMeasurement(makeMeasurement(0.0));
        std::vector<double> seen;
        queue.addMeasurementCallback([&](const Types::MeasurementQueue::pMeasurementType &m)
                                     { seen.push_back(m->time); });
        queue.addMeasurement(makeMeasurement(0.1));
        EXPECT_EQ(seen, (std::vector<double>{0.0, 0.1}));
        EXPECT_EQ(queue.size(), 2u); // observing doesn't consume
    }
} // namespace Spacetime
