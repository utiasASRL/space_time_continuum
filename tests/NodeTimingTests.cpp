#include <memory>
#include <string>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime
{
    using Pose = gtsam::Pose3;
    using NodeType = Priors::WNOA::Node<Pose>;
    using StateType = Types::ContinuumRobotState<NodeType>;
    using ValueType = Priors::WNOA::Values<Pose>;
    using PriorT = Priors::WNOA::Prior<Pose>;

    namespace
    {
        constexpr unsigned int kSpatialDim = 4;

        const char *const kExtrapolationWarning = "EXTRAPOLATED measurement factor";

        /// Robot + prior, kept together so the state outlives the prior.
        struct Rig
        {
            std::shared_ptr<StateType> state;
            std::shared_ptr<PriorT> prior;
            gtsam::Values values;
            KeyManager km;

            Rig()
                : state(std::make_shared<StateType>(1.0, kSpatialDim))
            {
                ValueType node_reference;
                node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
                prior = std::make_shared<PriorT>(state, node_reference);
            }

            /// extendState at `time`, returning everything it logged.
            std::string extend(double time)
            {
                gtsam::Values new_values;
                testing::internal::CaptureStdout();
                prior->extendState(km, time, values, new_values);
                const std::string output = testing::internal::GetCapturedStdout();
                values.insert(new_values);
                return output;
            }

            std::size_t rows() const { return state->estimation_nodes.size() / kSpatialDim; }
            double newestTime() const { return state->estimation_nodes.back()->time; }
        };

        gtsam::NoiseModelFactor::shared_ptr posePrior()
        {
            return std::make_shared<gtsam::PriorFactor<Pose>>(
                Priors::WNOA::Interpolator<Pose>::InterpolatedKeys::Pose(), Pose(),
                gtsam::noiseModel::Gaussian::Covariance(Eigen::Matrix<double, 6, 6>::Identity()));
        }

        bool contains(const std::string &haystack, const char *needle)
        {
            return haystack.find(needle) != std::string::npos;
        }

        bool graphContains(const gtsam::NonlinearFactorGraph &graph, const gtsam::NonlinearFactor *factor)
        {
            for (std::size_t i = 0; i < graph.size(); ++i)
                if (graph[i].get() == factor)
                    return true;
            return false;
        }
    } // namespace

    // ------------------------------------------------------------------
    // One node per extendState call, at exactly the requested time
    // ------------------------------------------------------------------

    TEST(NodeTiming, RepeatedOrEarlierTimesAddNothing)
    {
        // A zero-length or backwards interval would break the WNOA motion factors.
        Rig rig;
        rig.extend(0.0);
        rig.extend(0.1);
        ASSERT_EQ(rig.rows(), 2u);

        rig.extend(0.1);
        rig.extend(0.05);
        EXPECT_EQ(rig.rows(), 2u);
        EXPECT_DOUBLE_EQ(rig.newestTime(), 0.1);
    }

    // ------------------------------------------------------------------
    // Extrapolated measurement factors: still possible, always warned about
    // ------------------------------------------------------------------

    TEST(NodeTiming, ExtrapolatedFactorWarnsOnEitherAxis)
    {
        Rig rig;
        rig.extend(0.0);

        auto build = [&](double time, double arclength, std::size_t *keys)
        {
            testing::internal::CaptureStdout();
            auto factor = rig.prior->getOuterFactor(time, arclength, posePrior());
            const std::string output = testing::internal::GetCapturedStdout();
            EXPECT_NE(factor, nullptr);
            if (factor && keys)
                *keys = factor->keys().size();
            return output;
        };

        std::size_t keys = 0;
        const std::string in_time = build(0.05, 0.0, &keys);
        EXPECT_TRUE(contains(in_time, kExtrapolationWarning));
        EXPECT_TRUE(contains(in_time, "in time"));
        EXPECT_EQ(keys, 3u) << "extrapolated factors are still built";

        const std::string in_space = build(0.0, 1.5, &keys);
        EXPECT_TRUE(contains(in_space, kExtrapolationWarning));
        EXPECT_TRUE(contains(in_space, "in arclength"));

        const std::string in_both = build(0.05, 1.5, &keys);
        EXPECT_TRUE(contains(in_both, "in time"));
        EXPECT_TRUE(contains(in_both, "in arclength"));
    }

    TEST(NodeTiming, InterpolatedFactorsAndQueriesDoNotWarn)
    {
        Rig rig;
        rig.extend(0.0);
        rig.extend(0.1);

        testing::internal::CaptureStdout();
        auto on_node = rig.prior->getOuterFactor(0.1, 1.0, posePrior());
        auto between = rig.prior->getOuterFactor(0.05, 0.5, posePrior());
        // Extrapolating a state QUERY is a legitimate use, e.g. predicting past the newest node.
        auto predicted = rig.prior->query(0.2, 1.0, rig.values);
        const std::string output = testing::internal::GetCapturedStdout();

        EXPECT_NE(on_node, nullptr);
        EXPECT_NE(between, nullptr);
        EXPECT_NE(predicted, nullptr) << "query() must still extrapolate";
        EXPECT_FALSE(contains(output, kExtrapolationWarning)) << output;
    }

    // ------------------------------------------------------------------
    // Once in the graph, an extrapolated factor stays until marginalized
    // ------------------------------------------------------------------

    TEST(NodeTiming, SpatiallyExtrapolatedFactorPersistsInTheEstimator)
    {
        // Through the estimator, the only extrapolation that can still occur is spatial: a
        // sensor past the last arclength node. No later node can ever bracket it, and nothing
        // may try to replace it.
        Rig rig;
        Estimator estimator(/*smootherLag=*/10.0);
        estimator.addPrior(rig.prior);
        auto sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(rig.prior);
        estimator.addSensor(sensor);

        estimator.estimate(0.0);
        // NOTE: MeasurementBase takes (arclength, time, ...).
        sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(
            1.5, 0.1, Eigen::Matrix4d::Identity(), Eigen::Matrix<double, 6, 6>::Identity() * 1e-4,
            Eigen::MatrixXi::Ones(6, 1)));

        testing::internal::CaptureStdout();
        estimator.estimate(0.1);
        EXPECT_TRUE(contains(testing::internal::GetCapturedStdout(), kExtrapolationWarning));

        // The one measurement factor in the graph (everything else is a prior factor).
        const gtsam::NonlinearFactor *extrapolated = nullptr;
        for (const auto &factor : estimator.getFactors())
            if (factor && dynamic_cast<const Priors::InterpolationFactor<Priors::WNOA::Interpolator<Pose>> *>(factor.get()))
                extrapolated = factor.get();
        ASSERT_NE(extrapolated, nullptr);
        EXPECT_EQ(extrapolated->keys().size(), 3u);

        for (const double t : {0.2, 0.3, 0.4})
            estimator.estimate(t);

        EXPECT_TRUE(graphContains(estimator.getFactors(), extrapolated))
            << "the extrapolated factor was removed or replaced";
        EXPECT_EQ(extrapolated->keys().size(), 3u);
    }

    TEST(NodeTiming, TemporallyExtrapolatedFactorIsOnlyRemovedByMarginalization)
    {
        // The estimator never builds a temporally-extrapolated factor itself (every tick adds
        // a node at the current time first), but one can still reach the graph -- e.g. a
        // caller inserting it directly. Later nodes that bracket it must NOT upgrade it; only
        // the smoother's own marginalization may take it out.
        Rig rig;
        Estimator estimator(/*smootherLag=*/0.25);
        estimator.addPrior(rig.prior);
        estimator.estimate(0.0);

        testing::internal::CaptureStdout();
        auto factor = rig.prior->getOuterFactor(0.05, 1.0, posePrior());
        testing::internal::GetCapturedStdout();
        ASSERT_NE(factor, nullptr);
        ASSERT_EQ(factor->keys().size(), 3u);

        gtsam::NonlinearFactorGraph graph;
        graph.push_back(factor);
        estimator.smoother().update(graph); // straight into the smoother, bypassing the sensor path

        estimator.estimate(0.1); // brackets t = 0.05
        estimator.estimate(0.2);
        EXPECT_TRUE(graphContains(estimator.getFactors(), factor.get()))
            << "a bracketing node must not cause the factor to be replaced";

        // Far enough on that its anchor at t = 0 ages out of the 0.25 s lag window.
        for (const double t : {0.3, 0.4, 0.5, 0.6})
            estimator.estimate(t);
        EXPECT_FALSE(graphContains(estimator.getFactors(), factor.get()))
            << "marginalization should have folded it into a marginal factor";
    }

    // ------------------------------------------------------------------
    // Regression: ticks further apart than the smoother lag
    // ------------------------------------------------------------------

    TEST(NodeTiming, TicksFurtherApartThanTheSmootherLagDoNotThrow)
    {
        // Under the old grid placement, a prior whose node spacing exceeded the smoother lag
        // initialized each new row from a predecessor already marginalized out of the
        // values, and threw ValuesKeyDoesNotExist ("key p4 does not exist"). With a node at
        // every tick, each prior's predecessor is always the previous tick's, which is never
        // older than the lag window -- so the precondition for that crash no longer exists.
        Rig first, second;
        Estimator estimator(/*smootherLag=*/0.5);
        estimator.addPrior(first.prior);
        estimator.addPrior(second.prior);

        for (double time : {0.0, 1.0, 2.0, 3.0})
            EXPECT_NO_THROW(estimator.estimate(time));
    }
    // ------------------------------------------------------------------
    // Late measurements: bordering rows already marginalized
    // ------------------------------------------------------------------

    namespace
    {
        const char *const kMarginalizedWarning = "already marginalized";

        /// A pose measurement at (arclength, time).
        std::shared_ptr<Types::MeasurementBase> poseMeasurement(double arclength, double time)
        {
            // NOTE: MeasurementBase takes (arclength, time, ...).
            return std::make_shared<Types::MeasurementBase>(
                arclength, time, Eigen::Matrix4d::Identity(), Eigen::Matrix<double, 6, 6>::Identity() * 1e-4,
                Eigen::MatrixXi::Ones(6, 1));
        }

        /// The number of measurement (InterpolationFactor) factors in `graph`.
        std::size_t measurementFactorCount(const gtsam::NonlinearFactorGraph &graph)
        {
            std::size_t count = 0;
            for (const auto &factor : graph)
                if (factor && dynamic_cast<const Priors::InterpolationFactor<Priors::WNOA::Interpolator<Pose>> *>(factor.get()))
                    ++count;
            return count;
        }

        /// Estimator + prior + pose sensor, for the late-measurement tests.
        struct SensorRig
        {
            Rig rig;
            Estimator estimator;
            std::shared_ptr<Sensors::PoseSensor<ValueType>> sensor;

            explicit SensorRig(double smoother_lag) : estimator(smoother_lag)
            {
                estimator.addPrior(rig.prior);
                sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(rig.prior);
                estimator.addSensor(sensor);
            }

            /// estimate(time), returning everything it logged. Rethrows anything estimate()
            /// throws, after ending the capture (only one capturer may exist at a time).
            std::string estimate(double time)
            {
                testing::internal::CaptureStdout();
                try
                {
                    estimator.estimate(time);
                }
                catch (...)
                {
                    testing::internal::GetCapturedStdout();
                    throw;
                }
                return testing::internal::GetCapturedStdout();
            }
        };
    } // namespace

    TEST(NodeTiming, FactorOnMarginalizedRowIsRefused)
    {
        Rig rig;
        for (const double t : {0.0, 0.1, 0.2})
            rig.extend(t);
        rig.prior->onMarginalization(/*cutoff=*/0.2, rig.values);

        auto build = [&](double time, std::string &output)
        {
            testing::internal::CaptureStdout();
            auto factor = rig.prior->getOuterFactor(time, 0.5, posePrior());
            output = testing::internal::GetCapturedStdout();
            return factor;
        };

        std::string output;
        EXPECT_EQ(build(0.05, output), nullptr) << "both bordering rows are marginalized";
        EXPECT_TRUE(contains(output, kMarginalizedWarning)) << output;
        EXPECT_EQ(build(0.15, output), nullptr) << "the earlier bordering row is marginalized";
        EXPECT_TRUE(contains(output, kMarginalizedWarning)) << output;

        EXPECT_NE(build(0.2, output), nullptr) << "a query on the earliest live row is fine";
        EXPECT_FALSE(contains(output, kMarginalizedWarning)) << output;
    }

    TEST(NodeTiming, MarginalizationCutoffNeverMovesBackwards)
    {
        Rig rig;
        for (const double t : {0.0, 0.1, 0.2})
            rig.extend(t);
        rig.prior->onMarginalization(0.2, rig.values);
        rig.prior->onMarginalization(0.1, rig.values);

        testing::internal::CaptureStdout();
        auto factor = rig.prior->getOuterFactor(0.15, 0.5, posePrior());
        testing::internal::GetCapturedStdout();
        EXPECT_EQ(factor, nullptr);
    }

    TEST(NodeTiming, LateMeasurementInFilterModeIsDroppedNotThrown)
    {
        // Regression: in real time, a dataloader thread can push a measurement after the
        // estimate at a later time has already drained its queue. In filter mode
        // (smootherLag 0) the row before it is then gone, and the factor used to reach the
        // smoother anyway: ValuesKeyDoesNotExist ("key p... does not exist").
        SensorRig rig(/*smoother_lag=*/0.0);
        rig.estimate(0.0);
        rig.estimate(0.1); // marginalizes the row at t = 0

        rig.sensor->addMeasurement(poseMeasurement(0.5, 0.05));
        std::string output;
        EXPECT_NO_THROW(output = rig.estimate(0.2));
        EXPECT_TRUE(contains(output, kMarginalizedWarning)) << output;
        EXPECT_EQ(measurementFactorCount(rig.estimator.getFactors()), 0u);
    }

    TEST(NodeTiming, MeasurementBorderingTheLiveRowIsStillAttachedInFilterMode)
    {
        // The row the previous update kept is still in the smoother: a measurement between it
        // and the new row is attached, in the same update that then marginalizes that row.
        SensorRig rig(/*smoother_lag=*/0.0);
        rig.estimate(0.0);
        rig.estimate(0.1);

        rig.sensor->addMeasurement(poseMeasurement(0.5, 0.15));
        std::string output;
        EXPECT_NO_THROW(output = rig.estimate(0.2));
        EXPECT_FALSE(contains(output, kMarginalizedWarning)) << output;
        EXPECT_NO_THROW(rig.estimate(0.3));
    }

    TEST(NodeTiming, LateMeasurementOutsideTheLagWindowIsDropped)
    {
        SensorRig rig(/*smoother_lag=*/0.25);
        for (const double t : {0.0, 0.1, 0.2, 0.3, 0.4, 0.5})
            rig.estimate(t); // the window now starts at t = 0.3

        rig.sensor->addMeasurement(poseMeasurement(0.5, 0.15)); // before the window: dropped
        // Inside it, and between rows that also survive estimate(0.6) (whose window starts at
        // t = 0.4), so its factor stays in the graph rather than folding into a marginal.
        rig.sensor->addMeasurement(poseMeasurement(0.5, 0.45));
        std::string output;
        EXPECT_NO_THROW(output = rig.estimate(0.6));
        EXPECT_TRUE(contains(output, kMarginalizedWarning)) << output;
        EXPECT_EQ(measurementFactorCount(rig.estimator.getFactors()), 1u);
    }

    TEST(NodeTiming, QueriesStillWorkAfterBatchFinalize)
    {
        // finalize() reports an infinite cutoff (nothing is left to estimate); that must only
        // gate new factors, not queries of the solved state.
        SensorRig rig(Estimator::BATCH_LAG);
        rig.sensor->addMeasurement(poseMeasurement(0.5, 0.05));
        for (const double t : {0.0, 0.1, 0.2})
            rig.estimate(t);
        testing::internal::CaptureStdout();
        rig.estimator.finalize();
        auto queried = rig.rig.prior->query(0.05, 0.5, rig.estimator.getCurrentValues());
        testing::internal::GetCapturedStdout();
        EXPECT_NE(queried, nullptr);
    }
}
