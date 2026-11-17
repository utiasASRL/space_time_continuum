/**
 * @file bucky_ball.cpp
 * @brief A bucky ball pressed from the top: a stress test of joining many robots through end
 *        effectors.
 *
 * The ball is a truncated icosahedron (12 pentagons, 20 hexagons): each of its 90 edges is a
 * straight continuum robot with 2 nodes (its ends), and each of its 60 vertices an end effector rigidly
 * joined, with no translation between them, to the ends of its 3 edges. The joint offsets are
 * the rotations between each edge's end frame and its vertex's frame, so the priors' zero-error
 * solution is the perfect ball. No edge's base is locked; its unary pose and velocity priors are
 * loose, while its unary strain prior pulls it back to straight.
 *
 * The ball rests on the ground (z = 0) with its bottom hexagon's 6 end effectors locked there.
 * One vertex of its top hexagon is measured, by position only: after kLeadIn with no measurement,
 * it is pushed straight down kPushDepth over kPushRamp, held there for kPushHold, then released (no
 * more measurements), and the ball springs back over kRecovery. The measurement-free lead-in and
 * recovery keep the start and end estimates close to the undeformed ball (the batch smoother would
 * otherwise bend the start towards the push), so a looping replay doesn't jump. Everything is
 * hard-coded; no configuration file.
 *
 * It is one large batch problem (every vertex couples three edges at every time step), so its
 * size is kept down: 2 nodes per edge, a 0.1 s period, and a loose LM convergence tolerance.
 *
 * The C++ counterpart of python/examples/bucky_ball.py.
 *
 * Usage: bucky_ball [--headless] [--output <dir>]
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include <gtsam/geometry/Pose3.h>

#include "JointDemo.hpp"

using namespace JointDemo;
using EndEffectorPoseSensor = Spacetime::Sensors::PoseSensor<Spacetime::Joints::EndEffectorValues<Pose>>;

namespace
{
    constexpr double kPeriod = 0.1;
    constexpr double kEdgeLength = 0.1;        // [m] every edge (ball diameter ~0.5 m)
    constexpr unsigned int kNodes = 2;         // per edge robot
    constexpr double kLeadIn = 0.5;            // [s] before the push, with no measurement
    constexpr double kPushDepth = 0.3;         // [m] how far the top vertex is pushed down
    constexpr double kPushRamp = 1.0;          // [s] to push it down
    constexpr double kPushHold = 1.0;          // [s] held down, then released
    constexpr double kRecovery = 1.0;          // [s] after the release
    constexpr double kEndTime = kLeadIn + kPushRamp + kPushHold + kRecovery;
    constexpr double kStretchVariance = 1e-2;  // each edge's elongation and shear priors, relative to its torsion and bending (identity)
    constexpr double kFreeBaseVariance = 1e4;  // each edge's unary base pose and velocity variances: its base is free to move

    /// The ball: its vertices, edges ((i, j) pairs, i < j), and bottom and top hexagons (vertex indices).
    struct Ball
    {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::pair<std::size_t, std::size_t>> edges;
        std::vector<std::size_t> bottom, top;
    };

    /// The truncated icosahedron with edges of `edge_length`, its bottom hexagon level on the ground (z = 0).
    Ball truncatedIcosahedron(double edge_length)
    {
        // The even (cyclic) permutations of these, with every sign, have edge length 2.
        const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
        const std::array<std::array<double, 3>, 3> generators{{{0.0, 1.0, 3.0 * phi}, {1.0, 2.0 + phi, 2.0 * phi}, {phi, 2.0, 2.0 * phi + 1.0}}};
        std::set<std::array<double, 3>> points; // sorted, so the vertex order matches the Python demo's
        for (const auto &generator : generators)
            for (int signs = 0; signs < 8; signs++)
            {
                std::array<double, 3> signed_point;
                for (int i = 0; i < 3; i++)
                    signed_point[i] = generator[i] * (((signs >> (2 - i)) & 1) ? 1.0 : -1.0);
                for (int shift = 0; shift < 3; shift++) // rotated right by `shift`
                    points.insert({signed_point[(3 - shift) % 3], signed_point[(4 - shift) % 3], signed_point[(5 - shift) % 3]});
            }

        // A hexagon is centred on (1, 1, 1): turn it to face down (-z).
        const Eigen::Vector3d down = -Eigen::Vector3d::UnitZ(), centre = Eigen::Vector3d::Ones().normalized();
        const gtsam::Rot3 rotation = gtsam::Rot3::AxisAngle(gtsam::Point3(centre.cross(down).normalized()), std::acos(centre.dot(down)));
        Ball ball;
        for (const auto &point : points)
            ball.vertices.push_back(rotation.matrix() * (Eigen::Vector3d(point[0], point[1], point[2]) * (edge_length / 2.0)));
        double min_z = ball.vertices.front().z(), max_z = min_z;
        for (const auto &v : ball.vertices)
        {
            min_z = std::min(min_z, v.z());
            max_z = std::max(max_z, v.z());
        }
        for (auto &v : ball.vertices)
            v.z() -= min_z;
        max_z -= min_z;

        const double tolerance = 1e-6 * edge_length;
        for (std::size_t i = 0; i < ball.vertices.size(); i++)
        {
            for (std::size_t j = i + 1; j < ball.vertices.size(); j++)
                if (std::abs((ball.vertices[j] - ball.vertices[i]).norm() - edge_length) < tolerance)
                    ball.edges.emplace_back(i, j);
            if (ball.vertices[i].z() < tolerance)
                ball.bottom.push_back(i);
            if (ball.vertices[i].z() > max_z - tolerance)
                ball.top.push_back(i);
        }
        return ball;
    }

    /// The base and tip poses of the straight edge robot from vertex `start` to vertex `end`, its backbone along body x.
    std::pair<Pose, Pose> edgeFrames(const Eigen::Vector3d &start, const Eigen::Vector3d &end)
    {
        const Pose base(rotationWithXAxis(end - start), start);
        return {base, base * Pose(gtsam::Rot3(), gtsam::Point3((end - start).norm(), 0.0, 0.0))};
    }

    /// The frame of the end effector at `vertex`: world-aligned.
    Pose vertexPose(const Eigen::Vector3d &vertex) { return Pose(gtsam::Rot3(), vertex); }

    /// The pushed vertex's measured drop (<= 0) at every `period`, as (time, drop) pairs: from
    /// `start`, pushed down `depth` at a constant rate over `ramp`, held for `hold`, then released
    /// (no more pairs).
    std::vector<std::pair<double, double>> pushProfile(double start, double depth, double ramp, double hold, double period)
    {
        std::vector<std::pair<double, double>> profile;
        for (int k = 0; k * period <= ramp + hold + 1e-9; k++)
        {
            const double t = k * period;
            profile.emplace_back(start + t, -depth * std::min(t / ramp, 1.0));
        }
        return profile;
    }

    /// The furthest any vertex's estimated position is from the undeformed `ball` at `time`.
    double deformation(const Ball &ball, const std::vector<std::shared_ptr<JointDemo::EndEffector>> &end_effectors, double time, const gtsam::Values &values)
    {
        double furthest = 0.0;
        for (std::size_t v = 0; v < ball.vertices.size(); v++)
            furthest = std::max(furthest, (poseOf(*end_effectors[v]->endEffectorPrior(), 0.0, time, values).translation() - ball.vertices[v]).norm());
        return furthest;
    }
} // namespace

int main(int argc, char *argv[])
{
    const Ball ball = truncatedIcosahedron(kEdgeLength);
    const std::size_t pushed = ball.top.front();

    auto estimator_options = Spacetime::EstimatorOptions();
    estimator_options.smootherLag = Spacetime::Estimator::BATCH_LAG;
    estimator_options.compute_covariance = false;
    estimator_options.lm_params.relativeErrorTol = 1e-3; // later iterations change the error by < 1e-3
    Spacetime::Estimator estimator(estimator_options);

    // The edges: straight, stiff in elongation and shear so the ball keeps its size; free bases.
    const Vector6 straight = (Vector6() << 0, 0, 0, 1, 0, 0).finished();
    const StrainPrior stretch = StrainPrior::uniform(kStretchVariance);
    std::vector<DemoRobot> robots;
    std::vector<std::vector<std::tuple<std::shared_ptr<Prior>, double, Pose>>> ends(ball.vertices.size()); // per vertex: (edge prior, arclength, end pose)
    for (const auto &[i, j] : ball.edges)
    {
        const auto [base, tip] = edgeFrames(ball.vertices[i], ball.vertices[j]);
        robots.push_back(makeRobot(kEdgeLength, kNodes, base, straight, false, stretch, stretch, StrainPrior(), kFreeBaseVariance, kFreeBaseVariance));
        estimator.addPrior(robots.back().prior);
        ends[i].emplace_back(robots.back().prior, 0.0, base);
        ends[j].emplace_back(robots.back().prior, kEdgeLength, tip);
    }

    // The vertices: each end rigidly joined to its vertex with no translation, only the rotation
    // between the edge's end frame and the vertex's frame. The bottom hexagon is fixed to the ground.
    std::vector<std::shared_ptr<EndEffector>> end_effectors;
    for (std::size_t v = 0; v < ball.vertices.size(); v++)
    {
        const Pose pose = vertexPose(ball.vertices[v]);
        auto end_effector = std::make_shared<EndEffector>(pose);
        for (const auto &[edge_prior, s, end_pose] : ends[v])
        {
            Spacetime::Joints::JointOptions options;
            options.offset = end_pose.between(pose);
            end_effector->attach(edge_prior, s, Matrix6::Identity() * 1e-6, options);
        }
        if (std::find(ball.bottom.begin(), ball.bottom.end(), v) != ball.bottom.end())
            end_effector->state()->lock("pose", 0.0, pose);
        estimator.addModel(end_effector);
        end_effectors.push_back(end_effector);
    }

    // The pushed vertex's position (e.g. a tracked probe pressing on it).
    const Matrix6 covariance = (Vector6() << 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6).finished().asDiagonal();
    const Eigen::MatrixXi position_only = (Eigen::MatrixXi(6, 1) << 0, 0, 0, 1, 1, 1).finished(); // [rot, trans]
    auto sensor = std::make_shared<EndEffectorPoseSensor>(end_effectors[pushed]->endEffectorPrior());
    for (const auto &[t, drop] : pushProfile(kLeadIn, kPushDepth, kPushRamp, kPushHold, kPeriod))
    {
        const Pose measured = vertexPose(ball.vertices[pushed] + Eigen::Vector3d(0.0, 0.0, drop));
        sensor->addMeasurement(std::make_shared<Spacetime::Types::MeasurementBase>(0.0, t, measured.matrix(), covariance, position_only));
    }
    estimator.addSensor(sensor);

    // Every edge's two nodes sit at vertices, whose end-effector frames already mark them.
    const auto report = [&](const gtsam::Values &values)
    {
        std::cout << "Furthest vertex from the undeformed ball: " << deformation(ball, end_effectors, 0.0, values) << " m at the start, "
                  << deformation(ball, end_effectors, kEndTime, values) << " m at the end." << std::endl;
    };
    return run(estimator, robots, {}, end_effectors, 0.0, kEndTime, kPeriod, headlessRequested(argc, argv), argc, argv, true,
               /*show_node_frames=*/false, report);
}
