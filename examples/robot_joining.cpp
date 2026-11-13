/**
 * @file robot_joining.cpp
 * @brief Two continuum robots with offset bases, rigidly joined from the tip of robot A to
 *        the mid-point of robot B through a short (3 cm) link. Robot B, whose tip is free,
 *        is measured at its mid-point and its tip as it is driven into a growing "S" shape,
 *        one way and then the other, sideways (orthogonal to robot A's bending plane); the
 *        joint drags robot A along. Robot A's prior is stiff in elongation, so it bends
 *        rather than stretches to follow.
 *
 * Geometry (world z up): robot A stands at the origin and bends over by a quarter turn,
 * so its tip points along +x. Robot B stands upright further along +x, its mid-point 3 cm
 * past A's tip along A's tip tangent. Everything is hard-coded; no configuration file.
 *
 * Usage: robot_joining [--headless] [--output <dir>]
 */

#include <cmath>
#include <memory>
#include <vector>

#include <gtsam/geometry/Pose3.h>

#include "JointDemo.hpp"

using namespace JointDemo;

int main(int argc, char *argv[])
{
    constexpr double kPeriod = 1.0 / 30.0;
    constexpr double kEndTime = 2.0;
    constexpr double kLengthA = 0.4;
    constexpr unsigned int kNodes = 5;
    constexpr double kLink = 0.03;
    constexpr double kMaxSCurvature = 3.0;     // [rad/m] of each half of robot B's S
    constexpr double kElongationVariance = 1e-4; // robot A's elongation prior, relative to its other strains (identity)

    const Pose upright(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0, 0, 0)); // body x along world z
    const auto strain = [](double curvature)
    { return (Vector6() << 0, curvature, 0, 1, 0, 0).finished(); }; // bending in the world x-z plane (about body y)
    const auto sideways_strain = [](double curvature)
    { return (Vector6() << 0, 0, curvature, 1, 0, 0).finished(); }; // bending in the world y-z plane (about body z)

    // Robot A: a quarter circle in the x-z plane, from upright to pointing along +x.
    const double curvature_a = (M_PI / 2) / kLengthA;
    const Pose tip_a = upright * Pose::Expmap(kLengthA * strain(curvature_a));

    // Robot B: upright and straight, its mid-point kLink past A's tip along A's tip tangent.
    const gtsam::Point3 joint_point = tip_a.transformFrom(gtsam::Point3(kLink, 0, 0));
    const double length_b = 2.0 * joint_point.z(), mid_b = joint_point.z();
    const Pose base_b(upright.rotation(), gtsam::Point3(joint_point.x(), joint_point.y(), 0.0));

    // Robot B's S shape, in the world y-z plane -- orthogonal to robot A's bending plane, so A
    // follows by swinging sideways rather than stretching: curvature +k over B's lower half
    // and -k over its upper half, so its tip ends up parallel to its base. k sweeps
    // 0 -> +max -> -max -> 0 over the run.
    const auto s_curvature = [&](double t) { return kMaxSCurvature * std::sin(2.0 * M_PI * t / kEndTime); };
    const auto pose_b = [&](double s, double t)
    {
        const double k = s_curvature(t);
        if (s <= mid_b)
            return base_b * Pose::Expmap(s * sideways_strain(k));
        return base_b * Pose::Expmap(mid_b * sideways_strain(k)) * Pose::Expmap((s - mid_b) * sideways_strain(-k));
    };

    const std::vector<DemoRobot> robots = {makeRobot(kLengthA, kNodes, upright, strain(curvature_a), true, StrainPrior::uniform(kElongationVariance)),
                                           makeRobot(length_b, kNodes, base_b, strain(0.0))};

    auto estimator_options = Spacetime::EstimatorOptions();
    estimator_options.smootherLag = Spacetime::Estimator::BATCH_LAG;
    estimator_options.compute_covariance = true;
    Spacetime::Estimator estimator(estimator_options);
    for (const auto &robot : robots)
        estimator.addPrior(robot.prior);

    // The rigid link from A's tip to B's mid-point, at their nominal relative pose, held to 1 mm / 1 mrad.
    Spacetime::Joints::JointOptions options;
    options.offset = tip_a.between(pose_b(mid_b, 0.0));
    const std::vector<std::shared_ptr<Joint>> joints = {
        std::make_shared<Joint>(robots[0].prior, kLengthA, robots[1].prior, mid_b, Matrix6::Identity() * 1e-6, options)};
    for (const auto &joint : joints)
        estimator.addModel(joint);

    // Robot B's mid-point and tip, measured as it goes through the S.
    const Matrix6 covariance = (Vector6() << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6).finished().asDiagonal();
    auto sensor = std::make_shared<PoseSensor>(covariance, robots[1].prior);
    for (int k = 0; k * kPeriod <= kEndTime + 1e-9; k++)
    {
        const double t = k * kPeriod;
        for (const double s : {mid_b, length_b})
            measure(*sensor, s, t, pose_b(s, t), covariance);
    }
    estimator.addSensor(sensor);

    return run(estimator, robots, joints, {}, 0.0, kEndTime, kPeriod, headlessRequested(argc, argv), argc, argv);
}
