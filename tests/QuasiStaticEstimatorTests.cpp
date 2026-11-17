#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using QSValues = Priors::WNOAQuasiStatic::Values<Pose>;
        using QSPrior = Priors::WNOAQuasiStatic::Prior<Pose>;
        using QSState = Types::ContinuumRobotState<Priors::WNOAQuasiStatic::Node<Pose>>;
        using QSSensor = Sensors::PoseSensor<QSValues>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Matrix12 = Eigen::Matrix<double, 12, 12>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        const Vector6 epsilon0 = (Vector6() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();
        const Matrix6 R = 1e-8 * Matrix6::Identity(); // near-exact measurements

        Pose tip(double bend) { return Pose(gtsam::Rot3::Ry(bend), gtsam::Point3(0.2 * bend, 0.0, 0.95)); }

        EstimatorOptions quasiStaticOptions(double smoother_lag)
        {
            EstimatorOptions options;
            options.smootherLag = smoother_lag;
            options.quasi_static = true;
            return options;
        }

        /// A 3-node, unit-length robot under the quasi-static prior, with two tip pose sensors.
        struct Fixture
        {
            std::shared_ptr<QSState> state = std::make_shared<QSState>(1.0, 3);
            std::shared_ptr<QSPrior> prior = std::make_shared<QSPrior>(state, QSValues(Pose(), epsilon0), 1e-6 * Matrix12::Identity(), 1e-2 * Matrix6::Identity());
            std::shared_ptr<QSSensor> a = std::make_shared<QSSensor>(prior);
            std::shared_ptr<QSSensor> b = std::make_shared<QSSensor>(prior);
            Estimator estimator;

            explicit Fixture(const EstimatorOptions &options = quasiStaticOptions(Estimator::BATCH_LAG)) : estimator(options)
            {
                estimator.addPrior(prior);
                estimator.addSensor(a);
                estimator.addSensor(b);
            }

            static void measure(const std::shared_ptr<QSSensor> &sensor, double time, const Pose &pose = tip(0.0))
            {
                sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(1.0, time, pose.matrix(), R, Eigen::MatrixXi::Ones(6, 1)));
            }

            std::vector<double> rowTimes() const
            {
                std::vector<double> times;
                for (std::size_t i = 0; i < state->estimation_nodes.size(); i += 3)
                    times.push_back(state->estimation_nodes[i]->time);
                return times;
            }

            Pose tipAt(unsigned int row) const
            {
                return state->estimation_nodes.at(row * 3 + 2)->getValues(estimator.getCurrentValues()).pose();
            }
        };

    }

    TEST(QuasiStaticEstimator, AddsOneRowPerUniqueMeasurementTime)
    {
        Fixture f;
        Fixture::measure(f.a, 0.013);
        Fixture::measure(f.a, 0.05);
        Fixture::measure(f.b, 0.05);
        Fixture::measure(f.b, 0.07);
        f.estimator.estimate(0.1);

        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.013, 0.05, 0.07}));
        EXPECT_TRUE(f.a->empty());
        EXPECT_TRUE(f.b->empty());
    }

    TEST(QuasiStaticEstimator, MeasurementsSharingATimestampShareTheirRow)
    {
        Fixture f;
        Fixture::measure(f.a, 0.05);
        Fixture::measure(f.a, 0.05);
        Fixture::measure(f.b, 0.05);
        f.estimator.estimate(0.1);
        f.estimator.finalize();

        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.05}));
        // 4 prior factors (pose and strain unary, two spatial) and all 3 measurements.
        EXPECT_EQ(f.estimator.getFactors().size(), 7u);
    }

    TEST(QuasiStaticEstimator, IncludesMeasurementsAtTheEstimateTimeButNotAfter)
    {
        Fixture f;
        Fixture::measure(f.a, 0.1);
        Fixture::measure(f.a, 0.15);
        f.estimator.estimate(0.1);

        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.1}));
        ASSERT_EQ(f.a->size(), 1u);
        EXPECT_DOUBLE_EQ(f.a->peekAll().front()->time, 0.15);

        f.estimator.estimate(0.2);
        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.1, 0.15}));
    }

    TEST(QuasiStaticEstimator, NoNewMeasurementsAddNoRow)
    {
        Fixture f;
        f.estimator.estimate(0.1);
        EXPECT_TRUE(f.state->estimation_nodes.empty());

        Fixture::measure(f.a, 0.12);
        f.estimator.estimate(0.2);
        f.estimator.estimate(0.3);
        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.12}));
    }

    TEST(QuasiStaticEstimator, LateMeasurementsAreDrainedWithoutANewRow)
    {
        Fixture f;
        Fixture::measure(f.a, 0.1);
        f.estimator.estimate(0.1);
        Fixture::measure(f.b, 0.05); // arrives after its time's estimate
        Fixture::measure(f.a, 0.2);
        f.estimator.estimate(0.2);

        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.1, 0.2}));
        EXPECT_TRUE(f.a->empty());
        EXPECT_TRUE(f.b->empty());
    }

    TEST(QuasiStaticEstimator, BatchSolveFitsEveryRowToItsOwnMeasurements)
    {
        Fixture f;
        const std::vector<double> bends = {0.0, 0.3, -0.2};
        for (std::size_t k = 0; k < bends.size(); k++)
            Fixture::measure(f.a, 0.1 * k, tip(bends[k]));
        f.estimator.estimate(1.0);
        f.estimator.finalize();

        ASSERT_EQ(f.rowTimes().size(), 3u);
        for (unsigned int row = 0; row < 3; row++)
            EXPECT_TRUE(f.tipAt(row).equals(tip(bends[row]), 1e-3)) << "row " << row;
    }

    TEST(QuasiStaticEstimator, SlidingWindowMatchesTheBatchSolve)
    {
        Fixture batch, window(quasiStaticOptions(10.0));
        for (Fixture *f : {&batch, &window})
        {
            Fixture::measure(f->a, 0.0, tip(0.1));
            Fixture::measure(f->a, 0.04, tip(0.2));
            f->estimator.estimate(0.05);
            Fixture::measure(f->a, 0.08, tip(0.3));
            f->estimator.estimate(0.1);
        }
        batch.estimator.finalize();

        ASSERT_EQ(window.rowTimes(), batch.rowTimes());
        for (unsigned int row = 0; row < 3; row++)
            EXPECT_TRUE(window.tipAt(row).equals(batch.tipAt(row), 1e-6)) << "row " << row;
    }

    TEST(QuasiStaticEstimator, OptionOffKeepsRowsAtTheEstimateTimes)
    {
        EstimatorOptions options;
        options.smootherLag = Estimator::BATCH_LAG;
        Fixture f(options);
        Fixture::measure(f.a, 0.013);
        f.estimator.estimate(0.1);
        f.estimator.estimate(0.2);
        EXPECT_EQ(f.rowTimes(), (std::vector<double>{0.1, 0.2}));
    }

    TEST(QuasiStaticEstimator, WorksWithTheWNOAPriorToo)
    {
        using WNOAValues = Priors::WNOA::Values<Pose>;
        auto state = std::make_shared<Types::ContinuumRobotState<Priors::WNOA::Node<Pose>>>(1.0, 2);
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, WNOAValues());
        auto sensor = std::make_shared<Sensors::PoseSensor<WNOAValues>>(prior);
        Estimator estimator(quasiStaticOptions(Estimator::BATCH_LAG));
        estimator.addPrior(prior);
        estimator.addSensor(sensor);
        for (const double t : {0.02, 0.09})
            sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(1.0, t, Pose().matrix(), R, Eigen::MatrixXi::Ones(6, 1)));
        estimator.estimate(0.1);

        ASSERT_EQ(state->estimation_nodes.size(), 4u);
        EXPECT_DOUBLE_EQ(state->estimation_nodes[0]->time, 0.02);
        EXPECT_DOUBLE_EQ(state->estimation_nodes[2]->time, 0.09);
    }
} // namespace Spacetime
