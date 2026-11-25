#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtest/gtest.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/actuation_models/KinematicTendon.hpp>
#include <spacetime/estimator/prior_models/MultiInterpolationFactor.hpp>
#include <spacetime/estimator/prior_models/StraightStrain.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAPrior.hpp>
#include <spacetime/estimator/types/Measurement.hpp>
#include "NumericalDifferentiation.hpp"
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Prior = Priors::WNOA::Prior<Pose>;
        using StateType = Prior::StateType;
        using ValueType = Prior::ValuesType;
        using Tendon = Actuators::KinematicTendon<Prior>;
        using Vector6 = gtsam::Vector6;

        /// A straight `length` m robot with `nodes` nodes and an identity-weighted WNOA prior.
        std::shared_ptr<Prior> makePrior(double length = 1.0, unsigned int nodes = 5)
        {
            auto state = std::make_shared<StateType>(length, nodes);
            return std::make_shared<Prior>(state, ValueType());
        }

        /// The pose at arclength s of a constant-strain robot from the identity.
        Pose arcPose(const Vector6 &strain, double s) { return Pose::Expmap(s * strain); }

        std::shared_ptr<Types::MeasurementBase> lengthMeasurement(double time, double length, double variance)
        {
            return std::make_shared<Types::MeasurementBase>(0.0, time, Eigen::MatrixXd::Constant(1, 1, length),
                                                            Eigen::MatrixXd::Constant(1, 1, variance));
        }
    } // namespace

    TEST(KinematicTendon, DiskArclengthsEndAtTheTermination)
    {
        const std::vector<double> even = Tendon::diskArclengths(1.0, 0.2);
        const std::vector<double> expected_even{0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
        ASSERT_EQ(even.size(), expected_even.size());
        for (size_t i = 0; i < even.size(); ++i)
            EXPECT_NEAR(even[i], expected_even[i], 1e-12);

        const std::vector<double> uneven = Tendon::diskArclengths(1.0, 0.3);
        const std::vector<double> expected_uneven{0.0, 0.3, 0.6, 0.9, 1.0};
        ASSERT_EQ(uneven.size(), expected_uneven.size());
        for (size_t i = 0; i < uneven.size(); ++i)
            EXPECT_NEAR(uneven[i], expected_uneven[i], 1e-12);

        EXPECT_EQ(Tendon::diskArclengths(0.5, 1.0), (std::vector<double>{0.0, 0.5}));
    }

    TEST(KinematicTendon, ConstructorStoresTheRouting)
    {
        auto prior = makePrior();
        Tendon tendon(prior, 0.01, M_PI / 2, 0.8, 0.2);
        EXPECT_EQ(tendon.arclengths().size(), 5u);
        EXPECT_EQ(tendon.inputDim(), 1u);
        EXPECT_DOUBLE_EQ(tendon.radius(), 0.01);
        EXPECT_DOUBLE_EQ(tendon.angle(), M_PI / 2);
        EXPECT_DOUBLE_EQ(tendon.termination(), 0.8);
        ASSERT_TRUE(tendon.discretization().has_value());
        EXPECT_DOUBLE_EQ(*tendon.discretization(), 0.2);
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0.0, 0.0, 0.01), tendon.offset(), 1e-12));
        EXPECT_EQ(tendon.prior(), prior);
        EXPECT_DOUBLE_EQ(tendon.lengthSigma(), Tendon::DEFAULT_LENGTH_SIGMA);
        tendon.setLengthSigma(1e-3);
        EXPECT_DOUBLE_EQ(tendon.lengthSigma(), 1e-3);
        EXPECT_THROW(tendon.setLengthSigma(0.0), std::invalid_argument);
    }

    TEST(KinematicTendon, ExplicitDiskArclengths)
    {
        auto prior = makePrior();
        const std::vector<double> disks{0.1, 0.25, 0.7, 0.95};
        Tendon tendon(prior, 0.01, 0.0, disks);
        EXPECT_EQ(tendon.arclengths(), disks);
        EXPECT_DOUBLE_EQ(tendon.termination(), 0.95);
        EXPECT_FALSE(tendon.discretization().has_value());

        std::vector<Pose> poses;
        for (const double s : disks)
            poses.push_back(arcPose(Priors::straightStrain<Pose>(), s));
        EXPECT_NEAR(tendon.predictedLength(poses), 0.85, 1e-12); // from the first disk to the last

        EXPECT_THROW(Tendon(prior, 0.01, 0.0, std::vector<double>{0.5}), std::invalid_argument);           // one disk
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, std::vector<double>{0.0, 0.5, 0.5}), std::invalid_argument); // not strictly ascending
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, std::vector<double>{0.6, 0.3}), std::invalid_argument);      // descending
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, std::vector<double>{-0.1, 0.5}), std::invalid_argument);     // before the base
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, std::vector<double>{0.0, 1.2}), std::invalid_argument);      // past the tip
        EXPECT_THROW(Tendon(prior, -0.01, 0.0, disks), std::invalid_argument);
        EXPECT_THROW(Tendon(nullptr, 0.01, 0.0, disks), std::runtime_error);
    }

    TEST(KinematicTendon, ExplicitDisksMatchTheEvenlySpacedOnes)
    {
        auto prior = makePrior();
        const Tendon even(prior, 0.01, 0.3, 1.0, 0.2);
        const Tendon listed(prior, 0.01, 0.3, Tendon::diskArclengths(1.0, 0.2));
        EXPECT_EQ(even.arclengths(), listed.arclengths());
        EXPECT_TRUE(gtsam::assert_equal(even.offset(), listed.offset()));
    }

    TEST(KinematicTendon, InvalidArgumentsThrow)
    {
        auto prior = makePrior();
        EXPECT_THROW(Tendon(nullptr, 0.01, 0.0, 1.0, 0.2), std::runtime_error);
        EXPECT_THROW(Tendon(prior, -0.01, 0.0, 1.0, 0.2), std::invalid_argument);
        EXPECT_THROW(Tendon(prior, 0.01, NAN, 1.0, 0.2), std::invalid_argument);
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, 1.5, 0.2), std::invalid_argument); // past the robot
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, 0.0, 0.2), std::invalid_argument);
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, 1.0, 0.0), std::invalid_argument);
        EXPECT_THROW(Tendon(prior, 0.01, 0.0, 1.0, -0.2), std::invalid_argument);
    }

    TEST(KinematicTendon, StraightRobotLengthIsTheTermination)
    {
        auto prior = makePrior();
        for (const double angle : {0.0, 1.0, -2.5})
        {
            Tendon tendon(prior, 0.02, angle, 0.9, 0.2);
            std::vector<Pose> poses;
            for (const double s : tendon.arclengths())
                poses.push_back(arcPose(Priors::straightStrain<Pose>(), s));
            EXPECT_NEAR(tendon.predictedLength(poses), 0.9, 1e-12);
        }
    }

    TEST(KinematicTendon, ConstantCurvatureLengthIsTheSumOfChords)
    {
        auto prior = makePrior();
        const double kappa = 2.0, r = 0.01;
        const Vector6 strain = (Vector6() << 0, 0, kappa, 1, 0, 0).finished(); // bending about z, towards +y
        for (const double angle : {0.0, M_PI})
        {
            Tendon tendon(prior, r, angle, 1.0, 0.3);
            std::vector<Pose> poses;
            for (const double s : tendon.arclengths())
                poses.push_back(arcPose(strain, s));
            // A tendon offset towards the centre of curvature (+y) runs on a tighter circle.
            const double radius = 1.0 / kappa - r * std::cos(angle);
            double expected = 0.0;
            for (size_t i = 0; i + 1 < tendon.arclengths().size(); ++i)
                expected += 2.0 * radius * std::sin(kappa * (tendon.arclengths()[i + 1] - tendon.arclengths()[i]) / 2.0);
            EXPECT_NEAR(tendon.predictedLength(poses), expected, 1e-12);
        }
    }

    TEST(KinematicTendon, InnerFactorJacobiansMatchNumerical)
    {
        auto prior = makePrior();
        Tendon tendon(prior, 0.015, 0.7, 1.0, 0.3);
        const gtsam::Key u = gtsam::Symbol('u', 0);
        const auto factor = tendon.getFactors(*lengthMeasurement(0.0, 1.0, 1e-4), u).at(0);
        ASSERT_NE(factor, nullptr);
        ASSERT_EQ(factor->size(), tendon.arclengths().size() + 1);

        gtsam::Values values;
        const Vector6 strain = (Vector6() << 0.3, -0.5, 1.2, 1.05, 0.02, -0.01).finished();
        for (size_t i = 0; i < tendon.arclengths().size(); ++i)
            values.insert(Tendon::diskPoseKey(i), arcPose(strain, tendon.arclengths()[i]));
        values.insert(u, gtsam::Vector(gtsam::Vector1(0.97)));

        std::vector<gtsam::Matrix> H;
        const gtsam::Vector error = factor->unwhitenedError(values, &H);
        std::vector<Pose> poses;
        for (size_t i = 0; i < tendon.arclengths().size(); ++i)
            poses.push_back(values.at<Pose>(Tendon::diskPoseKey(i)));
        EXPECT_NEAR(error(0), tendon.predictedLength(poses) - 0.97, 1e-12);

        const auto numerical = Testing::numericalFactorJacobians(*factor, values);
        ASSERT_EQ(H.size(), numerical.size());
        for (size_t i = 0; i < H.size(); ++i)
            EXPECT_TRUE(gtsam::assert_equal(numerical[i], H[i], 1e-6)) << "key " << i;
    }

    TEST(KinematicTendon, InterpolatedFactorJacobiansMatchNumerical)
    {
        auto prior = makePrior(1.0, 4); // nodes at 0, 1/3, 2/3, 1: the disks fall between them
        Tendon tendon(prior, 0.01, 0.4, 0.9, 0.25);
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(prior);
        estimator.estimate(0.0);
        estimator.estimate(0.1);

        // Perturb the guess so that the robot is bent and moving.
        gtsam::Values values = estimator.getCurrentValues();
        gtsam::VectorValues delta = values.zeroVectors();
        for (auto &[key, vector] : delta)
            vector = 0.1 * gtsam::Vector::Random(vector.size());
        values = values.retract(delta);
        const gtsam::Key u = gtsam::Symbol('u', 0);
        values.insert(u, gtsam::Vector(gtsam::Vector1(0.95)));

        // The prior interpolates every disk jointly.
        const auto outer = prior->getMultiPointOuterFactor(0.05, tendon.arclengths(), tendon.getFactors(*lengthMeasurement(0.05, 0.95, 1e-4), u).at(0));
        const auto typed = std::dynamic_pointer_cast<Tendon::FactorType>(outer);
        ASSERT_NE(typed, nullptr);
        const Tendon::FactorType &factor = *typed;

        std::vector<gtsam::Matrix> H;
        factor.unwhitenedError(values, &H);
        const auto numerical = Testing::numericalFactorJacobians(factor, values);
        ASSERT_EQ(H.size(), numerical.size());
        // The pose-key columns carry the interpolator's small-angle approximation error (see
        // ActuatorBase.InterpolatedFactorJacobiansMatchNumerical), so each block is checked
        // relative to its size.
        for (size_t i = 0; i < H.size(); ++i)
            EXPECT_LT((H[i] - numerical[i]).cwiseAbs().maxCoeff(), 1e-3 * std::max(1.0, numerical[i].cwiseAbs().maxCoeff()))
                << gtsam::DefaultKeyFormatter(factor.keys()[i]);

        // The interpolation adds its conditional covariance to the tight factor noise.
        EXPECT_GT(factor.covarianceAt(values)(0, 0), tendon.lengthSigma() * tendon.lengthSigma());
    }

    TEST(KinematicTendon, MultiInterpolationFactorRejectsSharedInnerKeys)
    {
        auto prior = makePrior();
        Tendon tendon(prior, 0.01, 0.0, 1.0, 0.5);
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(prior);
        estimator.estimate(0.0);
        auto a = prior->interpolatedNode(0.0, 0.0), b = prior->interpolatedNode(0.0, 0.5); // both keep the placeholders
        ASSERT_TRUE(a && b);
        const auto inner = tendon.getFactors(*lengthMeasurement(0.0, 1.0, 1e-4), gtsam::Symbol('u', 0)).at(0);
        EXPECT_THROW(Tendon::FactorType(inner, {*a, *b}), std::invalid_argument);
        EXPECT_THROW(Tendon::FactorType(inner, {}), std::invalid_argument);
    }

    TEST(KinematicTendon, EstimatorFitsTheRobotToTheMeasuredLength)
    {
        auto prior = makePrior(1.0, 5);
        prior->state()->lock("pose", 0.0);
        auto tendon = std::make_shared<Tendon>(prior, 0.01, 0.0, 1.0, 0.2);
        tendon->addMeasurement(lengthMeasurement(0.0, 0.97, 1e-8));

        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(prior);
        estimator.addActuator(tendon);
        ASSERT_EQ(estimator.getMeasurementQueues().size(), 1u);
        estimator.estimate(0.0);
        estimator.finalize();

        const gtsam::Values &values = estimator.getCurrentValues();
        ASSERT_EQ(tendon->inputKeys().size(), 1u);
        const double u = values.at<gtsam::Vector>(tendon->inputKeys().front().second)(0);
        EXPECT_NEAR(u, 0.97, 1e-4);

        std::vector<Pose> poses;
        for (const double s : tendon->arclengths())
        {
            auto node = std::dynamic_pointer_cast<ValueType>(prior->query(0.0, s, values));
            ASSERT_NE(node, nullptr);
            poses.push_back(node->pose());
        }
        EXPECT_NEAR(tendon->predictedLength(poses), u, 1e-4);
    }

    TEST(KinematicTendon, MeasurementsBeforeTheStateAreDropped)
    {
        auto prior = makePrior();
        auto tendon = std::make_shared<Tendon>(prior, 0.01, 0.0, 1.0, 0.2);
        tendon->addMeasurement(lengthMeasurement(-1.0, 1.0, 1e-4)); // before the first row
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(prior);
        estimator.addActuator(tendon);
        EXPECT_NO_THROW(estimator.estimate(0.0));
        EXPECT_NO_THROW(estimator.finalize());
    }
} // namespace Spacetime
