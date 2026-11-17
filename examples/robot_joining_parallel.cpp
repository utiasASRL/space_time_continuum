/**
 * @file robot_joining_parallel.cpp
 * @brief A two-leg parallel continuum robot, in the spirit of the parallel robot of Lilge
 *        et al., "State Estimation for Continuum Multi-Robot Systems on SE(3)": two
 *        equal-length robots with fixed (locked) bases, their tips rigidly attached to a
 *        shared end effector -- a platform with a width, so the tips never touch. Only the
 *        end effector is measured: its pose, as it is moved around. Nothing on the robots is
 *        measured; their shapes come from the joints and their priors. Both priors are stiff
 *        in elongation and shear, so the legs bend rather than stretch or shear to follow.
 *
 * Geometry (world z up): the bases stand upright 20 cm apart along x and each leg bows
 * inward in the x-z plane (constant curvature), so its tip sits 6 cm in from its base, tilted
 * inward; the platform is level, centred between the two tips (about 8 cm apart). This
 * initial guess satisfies the joints exactly. Everything is hard-coded; no configuration file.
 *
 * Motion: the platform swings 27 cm out of the legs' plane (along y), dipping (by up to 17 cm)
 * and rolling (by up to 79 deg) with the swing as a pair of bending legs would carry it, and
 * yaws -- motions the legs make by bending alone. In their own plane the two clamped,
 * inextensible, unshearable legs are close to a rigid truss: lowering or swaying the platform
 * there would take very sharp S-bends (e.g. about 18 rad/m to lower it by 5 cm), so it does not.
 *
 * Usage: robot_joining_parallel [--headless] [--output <dir>]
 */

#include <cmath>
#include <memory>
#include <vector>

#include <gtsam/geometry/Pose3.h>

#include "JointDemo.hpp"

using namespace JointDemo;
using EndEffectorPoseSensor = Spacetime::Sensors::PoseSensor<Spacetime::Joints::EndEffectorValues<Pose>>;

int main(int argc, char *argv[])
{
    constexpr double kPeriod = 0.1;
    constexpr double kEndTime = 5.0;
    constexpr double kLength = 0.35;
    constexpr unsigned int kNodes = 5;
    constexpr double kBaseHalfSpacing = 0.1;    // [m] each base's distance from the centre
    constexpr double kBowCurvature = 1.0;       // [rad/m] of each leg's initial inward bow
    constexpr double kStretchVariance = 1e-2;   // each leg's elongation and shear priors, relative to its torsion (identity)
    constexpr double kBendingVariance = 1e3;    // each leg's bending arclength PSD: compliant, so its large bends are cheap
    constexpr double kSwing = 0.27;             // platform swing amplitude [m] (30 cm needs ~17 rad/m bends, the legs' reach limit)
    constexpr double kYaw = 10.0;               // [deg]
    constexpr double kDeg = M_PI / 180.0;

    const Pose upright(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0, 0, 0)); // body x along world z
    const auto strain = [](double curvature)
    { return (Vector6() << 0, curvature, 0, 1, 0, 0).finished(); }; // bending in the world x-z plane (about body y)

    // The legs, mirror images in x: the right one (+x) bows towards -x (positive curvature about
    // body y turns body x towards world +x, so it takes the negative curvature) and vice versa.
    StrainPrior bending;
    bending.arclength = kBendingVariance;
    std::vector<DemoRobot> robots;
    std::vector<Pose> tips;
    for (const double side : {-1.0, 1.0})
    {
        const Pose base(upright.rotation(), gtsam::Point3(side * kBaseHalfSpacing, 0, 0));
        const Vector6 bow = strain(-side * kBowCurvature);
        robots.push_back(makeRobot(kLength, kNodes, base, bow, true, StrainPrior::uniform(kStretchVariance), StrainPrior::uniform(kStretchVariance), bending));
        tips.push_back(base * Pose::Expmap(kLength * bow));
    }

    auto estimator_options = Spacetime::EstimatorOptions();
    estimator_options.smootherLag = Spacetime::Estimator::BATCH_LAG;
    estimator_options.compute_covariance = true;
    Spacetime::Estimator estimator(estimator_options);
    for (const auto &robot : robots)
        estimator.addPrior(robot.prior);

    // The platform: level, centred between the tips, each tip rigidly attached at its nominal
    // pose and held to 1 mm / 1 mrad.
    const Pose platform0(gtsam::Rot3(), gtsam::Point3(0.5 * (tips[0].translation() + tips[1].translation())));
    auto platform_ee = std::make_shared<EndEffector>(platform0);
    for (std::size_t i = 0; i < robots.size(); i++)
    {
        Spacetime::Joints::JointOptions options;
        options.offset = tips[i].between(platform0);
        platform_ee->attach(robots[i].prior, kLength, Matrix6::Identity() * 1e-6, options);
    }
    estimator.addModel(platform_ee);

    // The platform's motion, relative to its nominal pose; it starts and ends there. It swings
    // out of the legs' plane by y, and -- as the legs bend sideways under it -- dips by dz(y) and
    // rolls by roll(y). These are fits (to 0.2 mm and 0.8 deg, up to a 30 cm swing) of the
    // least-bending platform pose for each y, found offline for inextensible, unshearable legs
    // whose bending costs the integral of their squared curvature change from the initial bow.
    const auto platform = [&](double t)
    {
        const double phase = 2.0 * M_PI * t / kEndTime;
        const double y = kSwing * std::sin(phase);
        const double y2 = y * y;
        const double dz = y2 * (-1.787 + y2 * (-2.295 - 81.98 * y2));
        const double roll = y * (-4.273 + y2 * (-13.18 + 37.31 * y2));
        const gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(roll, 0.0, kYaw * kDeg * std::sin(2.0 * phase));
        return Pose(rotation, platform0.translation() + gtsam::Point3(0.0, y, dz));
    };

    // The end effector's pose, measured directly (e.g. a tracker on the platform).
    const Matrix6 covariance = (Vector6() << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6).finished().asDiagonal();
    auto sensor = std::make_shared<EndEffectorPoseSensor>(platform_ee->endEffectorPrior());
    for (int k = 0; k * kPeriod <= kEndTime + 1e-9; k++)
    {
        const double t = k * kPeriod;
        sensor->addMeasurement(std::make_shared<Spacetime::Types::MeasurementBase>(0.0, t, platform(t).matrix(), covariance, Eigen::MatrixXi::Ones(6, 1)));
    }
    estimator.addSensor(sensor);

    return run(estimator, robots, {}, {platform_ee}, 0.0, kEndTime, kPeriod, headlessRequested(argc, argv), argc, argv);
}
