#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WnoaFactor.h>
#include <gtsam/nonlinear/WnoaInterpolator.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/joint_models/EndEffector.hpp>
#include <spacetime/estimator/prior_models/InterpolationFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

#include "NumericalDifferentiation.hpp"
#include "TestEstimatorOptions.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;

        using EndEffector = Joints::EndEffector<Pose>;
        using EEPrior = Joints::EndEffectorPrior<Pose>;
        using EENode = Joints::EndEffectorNode<Pose>;
        using EEValues = Joints::EndEffectorNodeValues<Pose>;
        using EEKeys = Joints::EndEffectorKeys<Pose>;
        using EEInterpolator = Joints::EndEffectorInterpolator<Pose>;
        using EEState = Types::ContinuumRobotState<EENode>;

        using RobotNode = Priors::WNOA::Node<Pose>;
        using RobotValues = Priors::WNOA::NodeValues<Pose>;
        using RobotPrior = Priors::WNOA::Prior<Pose>;
        using RobotState = Types::ContinuumRobotState<RobotNode>;

        constexpr double kPeriod = 0.1;
        constexpr double kLength = 0.3;

        Vector6 qPsd() { return (Vector6() << 1.0, 2.0, 1.5, 0.5, 0.8, 1.2).finished(); }
        Pose eePose() { return Pose(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3), gtsam::Point3(0.1, 0.0, 0.4)); }
        Vector6 someVelocity() { return (Vector6() << 0.3, -0.2, 0.5, 0.1, 0.4, -0.3).finished(); }

        /// An upright WNOA robot with its base locked at `position`.
        struct UprightRobot
        {
            std::shared_ptr<RobotState> state;
            std::shared_ptr<RobotPrior> prior;
            Pose base;

            explicit UprightRobot(const Eigen::Vector3d &position) : base(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(position))
            {
                state = std::make_shared<RobotState>(kLength, 5);
                RobotValues reference;
                reference.pose(base);
                reference.epsilon((Vector6() << 0, 0, 0, 1, 0, 0).finished());
                prior = std::make_shared<RobotPrior>(state, reference);
                state->lock("pose", 0.0, base);
            }

            Pose tip() const { return base * Pose(gtsam::Rot3(), gtsam::Point3(kLength, 0, 0)); }
        };

        /// An end effector extended over rows at t = 0 and t = kPeriod, with distinct pose/velocity values.
        struct EERig
        {
            std::shared_ptr<EndEffector> ee = std::make_shared<EndEffector>(eePose(), qPsd());
            gtsam::Values values;
            KeyManager km;

            EERig()
            {
                UprightRobot holder(Eigen::Vector3d::Zero()); // the end effector refuses to be estimated alone
                ee->attach(holder.prior, kLength, Matrix6::Identity() * 1e-6);
                for (double t : {0.0, kPeriod})
                    extendPrior(t);
                const auto nodes = ee->state()->estimation_nodes;
                values.update(nodes[0]->velocityKey(), someVelocity());
                values.update(nodes[1]->poseKey(), Pose(eePose() * Pose::Expmap(someVelocity() * kPeriod)).retract(Vector6::Constant(0.01)));
                values.update(nodes[1]->velocityKey(), Vector6(someVelocity() + Vector6::Constant(0.05)));
            }

            void extendPrior(double t)
            {
                gtsam::Values new_values;
                ee->endEffectorPrior()->extendState(km, t, values, new_values);
                values.insert(new_values);
            }

            EEValues node(int row) const { return ee->state()->estimation_nodes[row]->getValues(values); }
        };

        /// Absolute tolerance for Jacobians through a time interpolation: gtsam::Interpolator's
        /// pose blocks use the same small-angle approximation as WNOA::Interpolator's.
        constexpr double kInterpolationTolerance = 5e-3;

        void expectJacobiansMatchNumerical(const gtsam::NoiseModelFactor &factor, const gtsam::Values &values, double tolerance = 1e-5)
        {
            std::vector<gtsam::Matrix> analytic(factor.size()); // gtsam indexes H without resizing it
            factor.unwhitenedError(values, analytic);
            const std::vector<gtsam::Matrix> numerical = Testing::numericalFactorJacobians(factor, values);
            ASSERT_EQ(analytic.size(), numerical.size());
            for (size_t i = 0; i < analytic.size(); i++)
                EXPECT_TRUE(gtsam::assert_equal(numerical[i], analytic[i], tolerance)) << "key " << gtsam::Symbol(factor.keys()[i]);
        }

        /// A factor on the end effector's interpolated pose (or velocity) at `time`.
        Priors::InterpolationFactor<EEInterpolator> interpolatedFactor(const EERig &rig, double time, bool on_velocity = false)
        {
            gtsam::NoiseModelFactor::shared_ptr inner;
            if (on_velocity)
                inner = std::make_shared<gtsam::PriorFactor<Vector6>>(EEKeys::Velocity(), someVelocity(), gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
            else
                inner = std::make_shared<gtsam::PriorFactor<Pose>>(EEKeys::Pose(), eePose(), gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
            const auto state = rig.ee->state();
            const auto n00 = state->getClosestShiftedNode(0.0, time, -1, -1);
            const auto n01 = std::abs(time - n00->time) > 1e-9 ? state->getNextTemporalNode(n00) : nullptr;
            return Priors::InterpolationFactor<EEInterpolator>(inner, rig.ee->endEffectorPrior()->interpolator(), time, 0.0, n00, n01, nullptr, nullptr);
        }

        EEValues query(const EERig &rig, double time, Eigen::MatrixXd *conditional = nullptr)
        {
            return *std::dynamic_pointer_cast<EEValues>(rig.ee->endEffectorPrior()->query(time, 0.0, rig.values, conditional));
        }
    } // namespace

    // ---------------------------------------------------------------------------------
    // Node, state and prior

    // As a stateful model, the end effector answers queries and builds outer factors through its prior.
    TEST(EndEffector, QueriesAndOuterFactorsAreItsPriors)
    {
        EERig rig;
        for (double t : {0.0, 0.37 * kPeriod, kPeriod + 0.04})
        {
            Eigen::MatrixXd conditional, prior_conditional;
            const auto own = std::dynamic_pointer_cast<EEValues>(rig.ee->query(t, 0.0, rig.values, &conditional));
            const auto prior = std::dynamic_pointer_cast<EEValues>(rig.ee->endEffectorPrior()->query(t, 0.0, rig.values, &prior_conditional));
            ASSERT_TRUE(own && prior) << "t = " << t;
            EXPECT_TRUE(own->pose().equals(prior->pose(), 1e-12));
            EXPECT_TRUE(conditional.isApprox(prior_conditional, 1e-12));

            const auto inner = std::make_shared<gtsam::PriorFactor<Pose>>(EEKeys::Pose(), eePose(), gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
            const auto factor = rig.ee->getOuterFactor(t, 0.0, inner);
            const auto prior_factor = rig.ee->endEffectorPrior()->getOuterFactor(t, 0.0, inner);
            ASSERT_TRUE(factor && prior_factor);
            EXPECT_EQ(factor->keys(), prior_factor->keys());
            EXPECT_NEAR(factor->error(rig.values), prior_factor->error(rig.values), 1e-12);
        }
    }

    TEST(EndEffectorPrior, FirstRowIsUnanchoredAndLaterRowsAreLinkedByWnoaMotion)
    {
        EERig rig;
        KeyManager km;
        auto ee = std::make_shared<EndEffector>(eePose(), qPsd());
        gtsam::Values values, new_values;
        EXPECT_TRUE(ee->endEffectorPrior()->extendState(km, 0.0, values, new_values).empty()); // no anchor in time
        // Initial guess: the given pose, at rest.
        const auto first = ee->state()->estimation_nodes[0];
        EXPECT_TRUE(new_values.at<Pose>(first->poseKey()).equals(eePose()));
        EXPECT_TRUE(new_values.at<Vector6>(first->velocityKey()).isZero());
        values.insert(new_values);
        values.update(first->velocityKey(), someVelocity());

        new_values.clear();
        const auto factors = ee->endEffectorPrior()->extendState(km, kPeriod, values, new_values);
        ASSERT_EQ(factors.size(), 1u);
        EXPECT_NE(std::dynamic_pointer_cast<gtsam::WnoaMotionFactor<Pose>>(factors[0]), nullptr);
        // Initial guess: the constant-velocity prediction from the previous row.
        const auto second = ee->state()->estimation_nodes[1];
        EXPECT_TRUE(new_values.at<Pose>(second->poseKey()).equals(eePose() * Pose::Expmap(kPeriod * someVelocity()), 1e-12));
        EXPECT_TRUE(new_values.at<Vector6>(second->velocityKey()).isApprox(someVelocity()));
        values.insert(new_values);
        EXPECT_NEAR(factors[0]->error(values), 0.0, 1e-18);

        new_values.clear();
        EXPECT_TRUE(ee->endEffectorPrior()->extendState(km, kPeriod, values, new_values).empty()); // idempotent
        EXPECT_TRUE(new_values.empty());
    }

    // ---------------------------------------------------------------------------------
    // Interpolation

    TEST(EndEffectorInterpolator, OnNodeIsTheNodeItself)
    {
        EERig rig;
        Eigen::MatrixXd conditional;
        const EEValues at_node = query(rig, kPeriod, &conditional);
        EXPECT_TRUE(at_node.pose().equals(rig.node(1).pose(), 1e-12));
        EXPECT_TRUE(at_node.varpi().isApprox(rig.node(1).varpi(), 1e-12));
        EXPECT_LT(conditional.norm(), 1e-12);
    }

    TEST(EndEffectorInterpolator, MatchesGtsamsWnoaInterpolator)
    {
        EERig rig;
        const double t = 0.37 * kPeriod;
        Eigen::MatrixXd conditional;
        const EEValues interpolated = query(rig, t, &conditional);

        const gtsam::Interpolator<Pose> reference(qPsd());
        const gtsam::TimestampedPoseVelocity<Pose> k(rig.node(0).pose(), rig.node(0).varpi(), 0.0), kp1(rig.node(1).pose(), rig.node(1).varpi(), kPeriod);
        const auto expected = reference.interpolatePoseAndVelocity(k, kp1, t);
        EXPECT_TRUE(interpolated.pose().equals(expected.pose, 1e-12));
        EXPECT_TRUE(interpolated.varpi().isApprox(expected.vel, 1e-12));

        const gtsam::TimestampedPoseVelocity<Pose> tau(expected.pose, expected.vel, t);
        EXPECT_TRUE(conditional.isApprox(reference.computeConditionalCov(k, kp1, tau), 1e-12));
    }

    TEST(EndEffectorInterpolator, JacobiansMatchNumerical)
    {
        EERig rig;
        for (bool on_velocity : {false, true})
        {
            expectJacobiansMatchNumerical(interpolatedFactor(rig, kPeriod, on_velocity), rig.values); // on-node: exact
            expectJacobiansMatchNumerical(interpolatedFactor(rig, 0.37 * kPeriod, on_velocity), rig.values, kInterpolationTolerance);
        }
    }

    TEST(EndEffectorInterpolator, ExtrapolationIsConstantVelocity)
    {
        EERig rig;
        const double dt = 0.04;
        Eigen::MatrixXd conditional;
        const EEValues extrapolated = query(rig, kPeriod + dt, &conditional);
        EXPECT_TRUE(extrapolated.pose().equals(rig.node(1).pose() * Pose::Expmap(dt * rig.node(1).varpi()), 1e-12));
        EXPECT_TRUE(extrapolated.varpi().isApprox(rig.node(1).varpi(), 1e-12));
        EXPECT_GT(conditional.norm(), 0.0);
        for (bool on_velocity : {false, true})
            expectJacobiansMatchNumerical(interpolatedFactor(rig, kPeriod + dt, on_velocity), rig.values, 1e-5);
    }

    // ---------------------------------------------------------------------------------
    // Attaching robots

    TEST(EndEffector, EstimatingWithoutAttachedRobotsThrows)
    {
        auto ee = std::make_shared<EndEffector>(eePose(), qPsd());
        KeyManager km;
        gtsam::Values new_values;
        EXPECT_THROW(ee->extendState(km, 0.0, gtsam::Values(), new_values), std::runtime_error);
    }

    TEST(EndEffector, AttachValidatesThroughTheJoint)
    {
        auto ee = std::make_shared<EndEffector>(eePose(), qPsd());
        UprightRobot a(Eigen::Vector3d::Zero());
        EXPECT_THROW(ee->attach(std::shared_ptr<RobotPrior>(), kLength, Matrix6::Identity() * 1e-6), std::invalid_argument);
        EXPECT_THROW(ee->attach(a.prior, kLength + 0.1, Matrix6::Identity() * 1e-6), std::invalid_argument);
        EXPECT_THROW(ee->attach(a.prior, kLength, Matrix6::Zero()), std::invalid_argument);
        EXPECT_THROW(ee->attach(ee->endEffectorPrior(), 0.0, Matrix6::Identity() * 1e-6), std::invalid_argument); // to itself
        EXPECT_TRUE(ee->joints().empty()); // nothing kept from a failed attach
    }

    TEST(EndEffector, AddsItsPriorAndEveryJointsFactors)
    {
        auto ee = std::make_shared<EndEffector>(eePose(), qPsd());
        UprightRobot a(Eigen::Vector3d::Zero()), b(Eigen::Vector3d(0.2, 0, 0));
        ee->attach(a.prior, kLength, Matrix6::Identity() * 1e-6);
        ee->attach(b.prior, kLength, Matrix6::Identity() * 1e-6);

        KeyManager km;
        gtsam::Values values;
        for (double t : {0.0, kPeriod})
        {
            for (const std::shared_ptr<Types::Model> &model : std::vector<std::shared_ptr<Types::Model>>{a.prior, b.prior, ee})
            {
                gtsam::Values new_values;
                const auto prior_factors = model->extendState(km, t, values, new_values);
                if (model == ee)
                    EXPECT_EQ(prior_factors.size(), t == 0.0 ? 0u : 1u); // its WNOA motion factor
                values.insert(new_values);
            }
            gtsam::Values none;
            EXPECT_EQ(ee->getNewFactors(t, none).size(), 4u) << "t = " << t; // pose + velocity per joint
        }
    }

    // ---------------------------------------------------------------------------------
    // Estimation

    namespace
    {
        /// Two upright robots whose tips are rigidly attached to an end effector between
        /// them; robot A's tip is measured moving sideways, robot B is only held by the joints.
        struct AttachedProblem
        {
            UprightRobot a{Eigen::Vector3d(-0.05, 0, 0)}, b{Eigen::Vector3d(0.05, 0, 0)};
            Pose ee0 = Pose(a.tip().rotation(), gtsam::Point3(0, 0, kLength));
            std::shared_ptr<EndEffector> ee = std::make_shared<EndEffector>(ee0, qPsd());

            AttachedProblem(Estimator &estimator, int ticks)
            {
                estimator.addPrior(a.prior);
                estimator.addPrior(b.prior);
                for (const UprightRobot *robot : {&a, &b})
                {
                    Joints::JointOptions options;
                    options.offset = robot->tip().between(ee0);
                    ee->attach(robot->prior, kLength, Matrix6::Identity() * 1e-6, options);
                }
                estimator.addModel(ee);

                const Matrix6 covariance = Matrix6::Identity() * 1e-6;
                auto sensor = std::make_shared<Sensors::PoseSensor<RobotValues>>(a.prior);
                for (int k = 0; k < ticks; k++)
                {
                    const Pose measured = a.tip() * Pose(gtsam::Rot3(), gtsam::Point3(0, 0.01 * k, 0));
                    sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(kLength, k * kPeriod, measured.matrix(), covariance, Eigen::MatrixXi::Ones(6, 1)));
                }
                estimator.addSensor(sensor);
            }

            Pose tip(const UprightRobot &robot, double t, const gtsam::Values &values) const
            {
                return std::dynamic_pointer_cast<RobotValues>(robot.prior->query(t, kLength, values))->pose();
            }
            Pose endEffector(double t, const gtsam::Values &values) const
            {
                return std::dynamic_pointer_cast<EEValues>(ee->endEffectorPrior()->query(t, 0.0, values))->pose();
            }
        };
    } // namespace

    TEST(EndEffector, BatchEstimateHoldsEveryAttachedRobot)
    {
        constexpr int ticks = 5;
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        AttachedProblem problem(estimator, ticks);
        for (int k = 0; k < ticks; k++)
            estimator.estimate(k * kPeriod);
        estimator.finalize();

        const gtsam::Values &values = estimator.getCurrentValues();
        const double t = (ticks - 1) * kPeriod;
        const Pose ee = problem.endEffector(t, values);
        for (const UprightRobot *robot : {&problem.a, &problem.b})
            EXPECT_LT(Pose::Logmap(robot->tip().between(problem.ee0).between(problem.tip(*robot, t, values).between(ee))).norm(), 5e-3);
        // A's measured motion moves the end effector, and through it B's tip.
        EXPECT_GT((ee.translation() - problem.ee0.translation()).norm(), 5e-3);
        EXPECT_GT((problem.tip(problem.b, t, values).translation() - problem.b.tip().translation()).norm(), 5e-3);
    }

    TEST(EndEffector, HoldsAnAttachedQuasiStaticRobot)
    {
        // Robot A (WNOA) is measured moving sideways; robot B (quasi-static) is only held by the end effector.
        using QSPrior = Priors::WNOAQuasiStatic::Prior<Pose>;
        constexpr int ticks = 4;
        UprightRobot a(Eigen::Vector3d(-0.05, 0, 0));
        const Pose base_b(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0.05, 0, 0));
        const Pose tip_b0 = base_b * Pose(gtsam::Rot3(), gtsam::Point3(kLength, 0, 0));
        auto state_b = std::make_shared<QSPrior::StateType>(kLength, 5);
        auto prior_b = std::make_shared<QSPrior>(state_b, QSPrior::ValuesType(base_b, (Vector6() << 0, 0, 0, 1, 0, 0).finished()));
        state_b->lock("pose", 0.0, base_b);

        const Pose ee0(a.tip().rotation(), gtsam::Point3(0, 0, kLength));
        auto ee = std::make_shared<EndEffector>(ee0, qPsd());
        Joints::JointOptions options_a, options_b;
        options_a.offset = a.tip().between(ee0);
        options_b.offset = tip_b0.between(ee0);
        ee->attach(a.prior, kLength, Matrix6::Identity() * 1e-6, options_a);
        const auto joint_b = ee->attach(prior_b, kLength, Matrix6::Identity() * 1e-6, options_b);
        static_assert(!decltype(joint_b)::element_type::joins_velocity);
        ASSERT_EQ(ee->joints().size(), 2u);
        EXPECT_EQ(ee->joints()[1]->stateA().get(), static_cast<const Types::State *>(state_b.get()));

        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        estimator.addPrior(a.prior);
        estimator.addPrior(prior_b);
        estimator.addModel(ee);
        auto sensor = std::make_shared<Sensors::PoseSensor<RobotValues>>(a.prior);
        for (int k = 0; k < ticks; k++)
        {
            const Pose measured = a.tip() * Pose(gtsam::Rot3(), gtsam::Point3(0, 0.01 * k, 0));
            sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(kLength, k * kPeriod, measured.matrix(), Matrix6::Identity() * 1e-6,
                                                                            Eigen::MatrixXi::Ones(6, 1)));
        }
        estimator.addSensor(sensor);
        for (int k = 0; k < ticks; k++)
            estimator.estimate(k * kPeriod);
        estimator.finalize();

        const gtsam::Values &values = estimator.getCurrentValues();
        const double t = (ticks - 1) * kPeriod;
        const Pose ee_t = std::dynamic_pointer_cast<EEValues>(ee->endEffectorPrior()->query(t, 0.0, values))->pose();
        const Pose tip_b = std::dynamic_pointer_cast<QSPrior::ValuesType>(prior_b->query(t, kLength, values))->pose();
        EXPECT_LT(Pose::Logmap(options_b.offset.between(tip_b.between(ee_t))).norm(), 5e-3);
        EXPECT_GT((tip_b.translation() - tip_b0.translation()).norm(), 5e-3); // dragged along through the end effector
    }

    TEST(EndEffector, PoseMeasurementsOnTheEndEffectorMoveEveryAttachedRobot)
    {
        // Nothing on the robots is measured: only the end effector's own pose, moving sideways.
        constexpr int ticks = 5;
        Estimator estimator(Testing::estimatorOptions(Estimator::BATCH_LAG));
        UprightRobot a(Eigen::Vector3d(-0.05, 0, 0)), b(Eigen::Vector3d(0.05, 0, 0));
        const Pose ee0(a.tip().rotation(), gtsam::Point3(0, 0, kLength));
        auto ee = std::make_shared<EndEffector>(ee0, qPsd());
        estimator.addPrior(a.prior);
        estimator.addPrior(b.prior);
        for (const UprightRobot *robot : {&a, &b})
        {
            Joints::JointOptions options;
            options.offset = robot->tip().between(ee0);
            ee->attach(robot->prior, kLength, Matrix6::Identity() * 1e-6, options);
        }
        estimator.addModel(ee);

        const Matrix6 covariance = Matrix6::Identity() * 1e-6;
        const auto measured = [&](int k)
        { return ee0 * Pose(gtsam::Rot3(), gtsam::Point3(0, 0.01 * k, 0)); };
        auto sensor = std::make_shared<Sensors::PoseSensor<EEValues>>(ee); // measures the end effector itself
        for (int k = 0; k < ticks; k++)
            sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(0.0, k * kPeriod, measured(k).matrix(), covariance, Eigen::MatrixXi::Ones(6, 1)));
        estimator.addSensor(sensor);

        for (int k = 0; k < ticks; k++)
            estimator.estimate(k * kPeriod);
        estimator.finalize();

        const gtsam::Values &values = estimator.getCurrentValues();
        const double t = (ticks - 1) * kPeriod;
        const Pose estimated_ee = std::dynamic_pointer_cast<EEValues>(ee->endEffectorPrior()->query(t, 0.0, values))->pose();
        EXPECT_LT(Pose::Logmap(measured(ticks - 1).between(estimated_ee)).norm(), 5e-3);
        for (const UprightRobot *robot : {&a, &b})
        {
            const Pose tip = std::dynamic_pointer_cast<RobotValues>(robot->prior->query(t, kLength, values))->pose();
            EXPECT_LT(Pose::Logmap(robot->tip().between(ee0).between(tip.between(estimated_ee))).norm(), 5e-3);
            EXPECT_GT((tip.translation() - robot->tip().translation()).norm(), 2e-2); // followed the 4 cm sideways move
        }
    }

    TEST(EndEffector, SlidingWindowEstimateRuns)
    {
        constexpr int ticks = 8;
        Estimator estimator(Testing::estimatorOptions(0.25));
        AttachedProblem problem(estimator, ticks);
        for (int k = 0; k < ticks; k++)
        {
            ASSERT_NO_THROW(estimator.estimate(k * kPeriod));
            const gtsam::Values &values = estimator.getCurrentValues();
            const Pose ee = problem.endEffector(k * kPeriod, values);
            EXPECT_LT(Pose::Logmap(problem.a.tip().between(problem.ee0).between(problem.tip(problem.a, k * kPeriod, values).between(ee))).norm(), 5e-3)
                << "tick " << k;
        }
    }
} // namespace Spacetime
