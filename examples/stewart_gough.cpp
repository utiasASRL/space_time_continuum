/**
 * @file stewart_gough.cpp
 * @brief Six continuum robots as the legs of a Stewart-Gough platform, each tip rigidly
 *        attached to the platform: an end effector whose frame sits at the platform's centre,
 *        equidistant from every tip. The bases are locked; each leg is actuated by
 *        elongation, measured at its base by a strain sensor that sees only the elongation
 *        component. Nothing on the platform is measured: its motion comes from the six leg
 *        elongations through the joints, and the end effector has no prior of its own in space.
 *
 * Geometry (world z up): base anchors at 0/120/240 deg +- 12 deg on a 15 cm circle,
 * platform anchors on an 8 cm circle rotated by 60 deg, each leg joining a base anchor to
 * a neighbouring platform anchor (the classic 6-6 layout). The platform height makes every
 * straight leg exactly its length, so the initial guess satisfies the joints.
 *
 * Actuation: each leg's commanded elongation is the distance from its anchor to its tip on a
 * large target platform motion (+-5 cm in each direction, +-20 deg tilt, +-15 deg twist),
 * over its rest length. With the base orientations locked, the legs also bend to carry the
 * platform. Each leg's prior keeps its elongation uniform along the backbone (stiff in the
 * arclength PSD) but free at its base (loose in the unary factor), where the tight elongation
 * measurement sets it. Everything is hard-coded; no configuration file.
 *
 * Usage: stewart_gough [--headless] [--output <dir>]
 */

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>

#include <spacetime/estimator/sensor_models/FBGSensor.hpp>

#include "JointDemo.hpp"

using namespace JointDemo;
using StrainSensor = Spacetime::Sensors::FBGSensor<Values>;

int main(int argc, char *argv[])
{
    constexpr double kPeriod = 0.1;
    constexpr double kEndTime = 8.0;
    constexpr double kLegLength = 0.3;
    constexpr unsigned int kNodes = 3;
    constexpr double kBaseRadius = 0.15, kPlatformRadius = 0.08;
    constexpr double kTranslation = 0.05, kTilt = 50.0, kTwist = 15.0; // platform motion amplitudes [m], [deg]
    constexpr double kDeg = M_PI / 180.0;
    constexpr double kElongationMeasurementVariance = 1e-8;

    const double base_angles[6] = {-12, 12, 108, 132, 228, 252};
    const double platform_angles[6] = {-48, 48, 72, 168, 192, 288};
    const auto onCircle = [](double radius, double degrees, double z)
    { return Eigen::Vector3d(radius * std::cos(degrees * kDeg), radius * std::sin(degrees * kDeg), z); };

    // Every leg spans the same horizontal distance; pick the height that makes it straight.
    const double span = (onCircle(kPlatformRadius, platform_angles[0], 0) - onCircle(kBaseRadius, base_angles[0], 0)).norm();
    const double height = std::sqrt(kLegLength * kLegLength - span * span);

    // Locked bases. Each leg is actuated by elongation: uniform along its backbone (stiff in
    // the arclength prior), free at its base (loose in the unary factor), and set there by a
    // tight elongation measurement.
    StrainPrior elongation_prior;
    elongation_prior.unary = 1e2;
    elongation_prior.arclength = 1e-6;
    std::vector<DemoRobot> robots;
    std::vector<Eigen::Vector3d> anchors;
    std::vector<Pose> nominal_tips;
    for (int i = 0; i < 6; i++)
    {
        const Eigen::Vector3d base = onCircle(kBaseRadius, base_angles[i], 0), tip = onCircle(kPlatformRadius, platform_angles[i], height);
        const Pose base_pose(rotationWithXAxis(tip - base), gtsam::Point3(base));
        robots.push_back(makeRobot(kLegLength, kNodes, base_pose, (Vector6() << 0, 0, 0, 1, 0, 0).finished(), true, elongation_prior));
        anchors.push_back(base);
        nominal_tips.push_back(base_pose * Pose(gtsam::Rot3(), gtsam::Point3(kLegLength, 0, 0)));
    }

    auto estimator_options = Spacetime::EstimatorOptions();
    estimator_options.smootherLag = Spacetime::Estimator::BATCH_LAG;
    estimator_options.compute_covariance = true;
    Spacetime::Estimator estimator(estimator_options);
    for (const auto &robot : robots)
        estimator.addPrior(robot.prior);

    // The platform: an end effector framed at its centre, every tip rigidly attached to it at
    // its nominal pose, held to 1 mm / 1 mrad.
    const Pose platform0(gtsam::Rot3(), gtsam::Point3(0, 0, height));
    auto platform_ee = std::make_shared<EndEffector>(platform0);
    for (int i = 0; i < 6; i++)
    {
        Spacetime::Joints::JointOptions options;
        options.offset = nominal_tips[i].between(platform0);
        platform_ee->attach(robots[i].prior, kLegLength, Matrix6::Identity() * 1e-6, options);
    }
    estimator.addModel(platform_ee);

    // The target platform motion, relative to its nominal pose; it starts there at t = 0.
    const auto platform = [&](double t)
    {
        const double phase = 2.0 * M_PI * t / kEndTime;
        const gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(kTilt * kDeg * std::sin(phase), kTilt * kDeg * std::sin(2.0 * phase),
                                                         kTwist * kDeg * std::sin(3.0 * phase));
        const gtsam::Point3 translation(kTranslation * std::sin(2.0 * phase), kTranslation * (1.0 - std::cos(phase)),
                                        height + kTranslation * std::sin(phase));
        return Pose(rotation, translation);
    };
    // Each leg's commanded elongation (stretch, 1 = rest length): the distance from its anchor
    // to its target tip over its rest length, as a prismatic leg would extend.
    const auto elongation = [&](int leg, double t)
    {
        const Pose tip = platform(t) * platform0.between(nominal_tips[leg]);
        return (Eigen::Vector3d(tip.translation()) - anchors[leg]).norm() / kLegLength;
    };

    // The actuators: each base's elongation strain alone (the other components masked out), measured tightly.
    const Matrix6 covariance = (Vector6() << 1, 1, 1, kElongationMeasurementVariance, 1, 1).finished().asDiagonal();
    Eigen::MatrixXi elongation_only = Eigen::MatrixXi::Zero(6, 1);
    elongation_only(3, 0) = 1;
    for (int i = 0; i < 6; i++)
    {
        auto sensor = std::make_shared<StrainSensor>(robots[i].prior);
        for (int k = 0; k * kPeriod <= kEndTime + 1e-9; k++)
        {
            const double t = k * kPeriod;
            const Vector6 strain = (Vector6() << 0, 0, 0, elongation(i, t), 0, 0).finished();
            sensor->addMeasurement(std::make_shared<Spacetime::Types::MeasurementBase>(0.0, t, strain, covariance, elongation_only));
        }
        estimator.addSensor(sensor);
    }

    return run(estimator, robots, {}, {platform_ee}, 0.0, kEndTime, kPeriod, headlessRequested(argc, argv), argc, argv);
}
