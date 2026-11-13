#include <memory>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime::Priors::WNOV
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using ValuesType = Values<Pose>;
        using PriorType = Prior<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Sensor = Sensors::PoseSensor<ValuesType>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        const Vector6 epsilon0 = (Vector6() << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished();
        const Matrix6 R = 1e-8 * Matrix6::Identity(); // near-exact measurements

        /// A 5-node, unit-length robot under the WNOV prior with a near-exact pose sensor.
        struct Fixture
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 5);
            std::shared_ptr<PriorType> prior;
            std::shared_ptr<Sensor> sensor;
            Estimator estimator;

            Fixture(double smoother_lag, bool compute_covariance = false, const PriorOptions &prior_options = PriorOptions())
                : estimator([&]
                            { EstimatorOptions options; options.smootherLag = smoother_lag; options.compute_covariance = compute_covariance; return options; }())
            {
                prior = std::make_shared<PriorType>(state, Pose(), epsilon0, 1e-4 * Matrix6::Identity(),
                                                    1e-2 * Matrix6::Identity(), 1e-1 * Matrix6::Identity(), prior_options);
                sensor = std::make_shared<Sensor>(R, prior);
                estimator.addPrior(prior);
                estimator.addSensor(sensor);
            }

            void measure(double time, double arclength, const Pose &pose)
            {
                sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(arclength, time, pose.matrix(), R, Eigen::MatrixXi::Ones(6, 1)));
            }

            Pose poseAt(unsigned int row, unsigned int n) const
            {
                return state->estimation_nodes.at(row * 5 + n)->getValues(estimator.getCurrentValues()).pose();
            }
        };

        Pose bentTip(double time) { return Pose(gtsam::Rot3::Ry(0.3 + time), gtsam::Point3(0.2 + time, 0.0, 0.9)); }
    }

    TEST(WNOVEstimator, BatchEstimateIsLinearBetweenMeasurements)
    {
        // A tip along the nominal strain direction, but shorter than the nominal length:
        // by symmetry nothing rotates, so the problem is an exact vector space, where the
        // MAP of a random walk pinned at both ends is the straight line between them
        // whatever its drift -- the nodes are evenly spaced, not at the nominal spacing.
        // (A chord off the body z axis is NOT a vector-space problem: epsilon0 is a body-
        // frame drift, so the interior frames tilt toward the chord.)
        Fixture f(Estimator::BATCH_LAG);
        const Pose tip(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.9));
        for (const double t : {0.0, 0.1, 0.2})
        {
            f.measure(t, 0.0, Pose());
            f.measure(t, 1.0, tip);
            f.estimator.estimate(t);
        }
        f.estimator.finalize();

        for (unsigned int row = 0; row < 3; row++)
            for (unsigned int n = 0; n < 5; n++)
                EXPECT_TRUE(f.poseAt(row, n).equals(Pose(gtsam::Rot3(), 0.25 * n * tip.translation()), 1e-6)) << "row " << row << ", node " << n;
    }

    TEST(WNOVEstimator, BatchEstimateIsPiecewiseLinear)
    {
        Fixture f(Estimator::BATCH_LAG);
        for (const double t : {0.0, 0.1, 0.2})
        {
            f.measure(t, 0.0, Pose());
            f.measure(t, 1.0, bentTip(t));
            f.estimator.estimate(t);
        }
        f.estimator.finalize();

        for (unsigned int row = 0; row < 3; row++)
        {
            EXPECT_TRUE(f.poseAt(row, 0).equals(Pose(), 1e-3)) << "row " << row;
            EXPECT_TRUE(f.poseAt(row, 4).equals(bentTip(0.1 * row), 1e-3)) << "row " << row;
        }

        // Queries between nodes interpolate the estimate linearly, in arclength and in time.
        const gtsam::Values values = f.estimator.getCurrentValues();
        auto in_space = std::dynamic_pointer_cast<ValuesType>(f.prior->query(0.1, 0.125, values));
        ASSERT_NE(in_space, nullptr);
        EXPECT_TRUE(in_space->pose().equals(gtsam::interpolate<Pose>(f.poseAt(1, 0), f.poseAt(1, 1), 0.5), 1e-12));
        auto in_time = std::dynamic_pointer_cast<ValuesType>(f.prior->query(0.13, 0.75, values));
        ASSERT_NE(in_time, nullptr);
        EXPECT_TRUE(in_time->pose().equals(gtsam::interpolate<Pose>(f.poseAt(1, 3), f.poseAt(2, 3), 0.3), 1e-9));
    }

    TEST(WNOVEstimator, RemoveUnaryLetsLaterBasesFollowTheirMeasurements)
    {
        // The base is measured at T0 on the first row and away from it afterwards: without
        // the later rows' unary factors, nothing but the time prior holds those bases to T0.
        const Pose moved_base(gtsam::Rot3::Rz(0.1), gtsam::Point3(0.05, 0.0, 0.0));
        auto baseMiss = [&](bool remove_unary)
        {
            PriorOptions options;
            options.remove_unary = remove_unary;
            Fixture f(Estimator::BATCH_LAG, false, options);
            for (const double t : {0.0, 0.1, 0.2})
            {
                f.measure(t, 0.0, t == 0.0 ? Pose() : moved_base);
                f.measure(t, 1.0, bentTip(t));
                f.estimator.estimate(t);
            }
            f.estimator.finalize();
            EXPECT_TRUE(f.poseAt(0, 0).equals(Pose(), 1e-3)) << "the first row keeps its unary factor";
            return Pose::Logmap(f.poseAt(2, 0).between(moved_base)).norm();
        };
        const double kept = baseMiss(false), removed = baseMiss(true);
        EXPECT_LT(removed, 1e-5);
        EXPECT_LT(removed, kept);
    }

    TEST(WNOVEstimator, OffNodeMeasurementsAreMatchedThroughTheInterpolation)
    {
        // With the measurement's own noise model (no conditional covariance added), a
        // near-exact measurement between nodes must be reproduced by the interpolated pose.
        PriorOptions options;
        options.fixed_noise_model = true;
        Fixture f(Estimator::BATCH_LAG, false, options);
        const Pose measured(gtsam::Rot3::Rx(0.2), gtsam::Point3(0.3, 0.0, 0.375));
        f.measure(0.0, 0.0, Pose());
        f.measure(0.0, 0.375, measured); // between nodes 1 (s = 0.25) and 2 (s = 0.5)
        f.estimator.estimate(0.0);
        f.estimator.finalize();

        auto queried = std::dynamic_pointer_cast<ValuesType>(f.prior->query(0.0, 0.375, f.estimator.getCurrentValues()));
        ASSERT_NE(queried, nullptr);
        EXPECT_TRUE(queried->pose().equals(measured, 1e-4));
    }

    TEST(WNOVEstimator, ConditionalCovarianceDownweightsOffNodeMeasurements)
    {
        // The default InterpolationFactor adds the GP conditional covariance to the
        // measurement's, so an off-node measurement is trusted less than an on-node one.
        const Pose measured(gtsam::Rot3(), gtsam::Point3(0.3, 0.0, 0.375));
        auto miss = [&](bool fixed_noise_model)
        {
            PriorOptions options;
            options.fixed_noise_model = fixed_noise_model;
            Fixture f(Estimator::BATCH_LAG, false, options);
            f.measure(0.0, 0.0, Pose());
            f.measure(0.0, 0.375, measured);
            f.estimator.estimate(0.0);
            f.estimator.finalize();
            auto queried = std::dynamic_pointer_cast<ValuesType>(f.prior->query(0.0, 0.375, f.estimator.getCurrentValues()));
            return Pose::Logmap(queried->pose().between(measured)).norm();
        };
        const double straight_miss = Pose::Logmap(Pose(gtsam::Rot3(), gtsam::Point3(0.0, 0.0, 0.375)).between(measured)).norm();
        const double augmented = miss(false), fixed = miss(true);
        EXPECT_LT(fixed, 1e-4);
        EXPECT_GT(augmented, fixed);
        EXPECT_LT(augmented, 0.5 * straight_miss);
    }

    TEST(WNOVEstimator, PosteriorCovarianceAtAndBetweenNodes)
    {
        Fixture f(Estimator::BATCH_LAG, /*compute_covariance=*/true);
        for (const double t : {0.0, 0.1})
        {
            f.measure(t, 0.0, Pose());
            f.measure(t, 1.0, bentTip(t));
            f.estimator.estimate(t);
        }
        f.estimator.finalize();
        const auto posterior = f.estimator.getCurrentCovariance();
        ASSERT_NE(posterior, nullptr);
        const gtsam::Values values = f.estimator.getCurrentValues();

        // On a node, the posterior is that node's marginal.
        Eigen::MatrixXd on_node;
        ASSERT_NE(f.prior->query(0.0, 0.5, values, nullptr, posterior.get(), &on_node), nullptr);
        ASSERT_EQ(on_node.rows(), 6);
        EXPECT_TRUE(on_node.isApprox(posterior->covariance(f.state->estimation_nodes[2]->poseKey()), 1e-9));

        // Between nodes, it adds the conditional covariance to the propagated marginals.
        Eigen::MatrixXd conditional, between;
        ASSERT_NE(f.prior->query(0.05, 0.625, values, &conditional, posterior.get(), &between), nullptr);
        ASSERT_EQ(between.rows(), 6);
        EXPECT_TRUE(between.isApprox(between.transpose(), 1e-12));
        EXPECT_GT((between - conditional).trace(), 0.0);
        EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(between).eigenvalues().minCoeff(), 0.0);
    }

    TEST(WNOVEstimator, FixedLagSmootherRuns)
    {
        Fixture f(0.25);
        for (int k = 0; k <= 10; k++)
        {
            const double t = 0.1 * k;
            f.measure(t, 0.0, Pose());
            f.measure(t, 1.0, bentTip(0.0));
            EXPECT_NO_THROW(f.estimator.estimate(t)) << "t = " << t;
        }

        const unsigned int last_row = 10;
        EXPECT_TRUE(f.poseAt(last_row, 4).equals(bentTip(0.0), 1e-3));
        // Rows past the lag have been marginalized out, so they cannot be queried.
        EXPECT_EQ(f.prior->query(0.0, 0.5, f.estimator.getCurrentValues()), nullptr);
        EXPECT_NE(f.prior->query(1.0, 0.5, f.estimator.getCurrentValues()), nullptr);
    }
} // namespace Spacetime::Priors::WNOV
