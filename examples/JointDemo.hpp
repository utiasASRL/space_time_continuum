#ifndef EXAMPLES_JOINTDEMO_HPP
#define EXAMPLES_JOINTDEMO_HPP

/**
 * @file JointDemo.hpp
 * @brief Shared plumbing of the multi-robot joint demos (robot_joining.cpp,
 *        stewart_gough.cpp): building upright WNOA robots with locked bases, running the
 *        estimator through the standalone interface, drawing every robot and joint, and
 *        reporting how well each joint holds.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime>

#ifdef VISUALIZER
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);
#endif

namespace JointDemo
{
    using Pose = gtsam::Pose3;
    using Prior = Spacetime::Priors::WNOA::Prior<Pose>;
    using Node = Prior::NodeType;
    using Values = Prior::ValuesType;
    using State = Prior::StateType;
    using Joint = Spacetime::Joints::Joint<Prior, Prior>;
    using EndEffector = Spacetime::Joints::EndEffector<Pose>;
    using EndEffectorJoint = Spacetime::Joints::Joint<Prior, EndEffector::PriorType>;
    using PoseSensor = Spacetime::Sensors::PoseSensor<Values>;
    using Vector6 = Eigen::Matrix<double, 6, 1>;
    using Matrix6 = Eigen::Matrix<double, 6, 6>;

    /// A WNOA robot: its state, the prior extending it, and its initial base pose.
    struct DemoRobot
    {
        std::shared_ptr<State> state;
        std::shared_ptr<Prior> prior;
        Pose base;
    };

    /**
     * @brief How a robot's prior treats a group of its strains -- elongation (translational x),
     *        shear (translational y and z) or bending (rotational y and z), in GTSAM
     *        [rot, trans] order -- as scales of the corresponding identity variances.
     */
    struct StrainPrior
    {
        double unary = 1.0;     ///< In the base unary factor (Q0): large lets the base strain be set by measurements.
        double arclength = 1.0; ///< In the arclength PSD (Q2): small keeps the strain uniform along the backbone.
        double time = 1.0;      ///< In the strain's time PSD (Q3).

        /// The same scale everywhere: e.g. small elongation and shear make a backbone that bends rather than stretches or shears.
        static StrainPrior uniform(double variance) { return StrainPrior{variance, variance, variance}; }
    };

    /**
     * @brief A robot of `length` with `nodes` evenly spaced nodes, its base at `base` (T_ib,
     *        the backbone along the body x axis) -- locked there unless `lock_base` is false,
     *        e.g. for a base driven by measurements -- and whose prior mean has the constant
     *        strain `strain` ([rot, trans]; (0, 0, 0, 1, 0, 0) is straight).
     * @param elongation, shear, bending See StrainPrior; every other prior variance (torsion's
     *        included) is the identity.
     * @param unary_pose, unary_velocity Scales of the base's unary pose and velocity variances
     *        (Q0): large lets a free base move.
     */
    inline DemoRobot makeRobot(double length, unsigned int nodes, const Pose &base, const Vector6 &strain, bool lock_base = true,
                               const StrainPrior &elongation = StrainPrior(), const StrainPrior &shear = StrainPrior(),
                               const StrainPrior &bending = StrainPrior(), double unary_pose = 1.0, double unary_velocity = 1.0)
    {
        Prior::Matrix3N Q0 = Prior::Matrix3N::Identity();
        Q0.topLeftCorner<6, 6>() *= unary_pose;
        Q0.bottomRightCorner<6, 6>() *= unary_velocity;
        Prior::MatrixN Q2 = Prior::MatrixN::Identity(), Q3 = Prior::MatrixN::Identity();
        // Rotational y and z (bending), translational x (elongation), y and z (shear), in GTSAM [rot, trans] order.
        const std::pair<int, const StrainPrior *> scaled[] = {{1, &bending}, {2, &bending}, {3, &elongation}, {4, &shear}, {5, &shear}};
        for (const auto &[i, scale] : scaled)
        {
            Q0(6 + i, 6 + i) = scale->unary; // [pose, strain, velocity] blocks
            Q2(i, i) = scale->arclength;
            Q3(i, i) = scale->time;
        }

        DemoRobot out;
        out.base = base;
        out.state = std::make_shared<State>(length, nodes);
        Values reference;
        reference.pose(base);
        reference.epsilon(strain);
        out.prior = std::make_shared<Prior>(out.state, reference, Q0, Prior::MatrixN::Identity(), Q2, Q3, Prior::MatrixN::Identity());
        if (lock_base)
            out.state->lock("pose", 0.0, base);
        return out;
    }

    /// Rotation whose x axis is `x` (normalized); y is horizontal where possible.
    inline gtsam::Rot3 rotationWithXAxis(const Eigen::Vector3d &x)
    {
        const Eigen::Vector3d ex = x.normalized();
        Eigen::Vector3d ey = Eigen::Vector3d::UnitZ().cross(ex);
        if (ey.norm() < 1e-9)
            ey = Eigen::Vector3d::UnitY();
        ey.normalize();
        const Eigen::Vector3d ez = ex.cross(ey);
        Eigen::Matrix3d R;
        R << ex, ey, ez;
        return gtsam::Rot3(R);
    }

    /// Adds a 6-DoF pose measurement of `robot` at arclength `s` and `time`, with covariance `covariance` ([rot, trans]).
    inline void measure(PoseSensor &sensor, double s, double time, const Pose &pose, const Matrix6 &covariance)
    {
        sensor.addMeasurement(std::make_shared<Spacetime::Types::MeasurementBase>(s, time, pose.matrix(), covariance, Eigen::MatrixXi::Ones(6, 1)));
    }

    /// The estimated pose at arclength `s` and `time` of the robot (a WNOA robot or an end effector) extended by `prior`.
    inline Pose poseOf(const Spacetime::Types::StatefulModel<> &prior, double s, double time, const gtsam::Values &values)
    {
        const auto node = prior.query(time, s, values);
        if (const auto wnoa = std::dynamic_pointer_cast<Values>(node))
            return wnoa->pose();
        return std::dynamic_pointer_cast<Spacetime::Joints::EndEffectorNodeValues<Pose>>(node)->pose();
    }

    /// How far a joint is from holding at `time`: |Log(T_off^-1 T_A^-1 T_B)| ([rad, m] mixed), from the estimator's current values.
    template <typename JointT>
    double jointResidual(const JointT &joint, double time, const gtsam::Values &values)
    {
        const Pose T_a = poseOf(*joint.priorA(), joint.arclengthA(), time, values), T_b = poseOf(*joint.priorB(), joint.arclengthB(), time, values);
        return Pose::Logmap(joint.offset().between(T_a.between(T_b))).norm();
    }

    /// Furthest the point at arclength `s` of the robot extended by `prior` moves from its start_time position, sampled every `period`.
    inline double travel(const Spacetime::Types::StatefulModel<> &prior, double s, double start_time, double end_time, double period, const gtsam::Values &values)
    {
        const auto position = [&](double time)
        { return poseOf(prior, s, time, values).translation(); };
        const gtsam::Point3 start = position(start_time);
        double furthest = 0.0;
        for (double time = start_time; time <= end_time + 1e-9; time += period)
            furthest = std::max(furthest, (position(time) - start).norm());
        return furthest;
    }

    /// The joints attached to `end_effector`, typed (they all join a WNOA robot to it).
    inline std::vector<std::shared_ptr<EndEffectorJoint>> jointsOf(const EndEffector &end_effector)
    {
        std::vector<std::shared_ptr<EndEffectorJoint>> joints;
        for (const auto &model : end_effector.joints())
            joints.push_back(std::dynamic_pointer_cast<EndEffectorJoint>(model));
        return joints;
    }

    /// How well a joint holds at the end time: see jointResidual and travel.
    struct JointReport
    {
        std::string id; ///< "<i>" for a robot-robot joint, "ee<e>.<i>" for end effector e's joint i.
        double residual, travel_a, travel_b;
    };

    /// The directory after `--output` among the arguments, or "" if there is none.
    inline std::string outputRequested(int argc, char *argv[])
    {
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--output")
                return argv[i + 1];
        return "";
    }

    /**
     * @brief Writes the estimate for regression testing (python/tests/test_demo_regression.py) to `directory`:
     *        estimates.csv, every entry of every estimated variable as `key,value` rows (key "<symbol>[<i>]",
     *        a pose as its row-major 4x4 matrix), and joints.csv, each joint's JointReport.
     */
    inline void writeOutput(const std::string &directory, const gtsam::Values &values, const std::vector<JointReport> &reports)
    {
        std::filesystem::create_directories(directory);
        std::ofstream estimates(directory + "/estimates.csv"), joints(directory + "/joints.csv");
        if (!estimates || !joints)
            throw std::runtime_error("JointDemo: cannot write to " + directory + ".");
        estimates << std::setprecision(17) << "key,value\n";
        for (const auto &[key, value] : values)
        {
            Eigen::MatrixXd entries;
            if (const auto *pose = dynamic_cast<const gtsam::GenericValue<Pose> *>(&value))
                entries = pose->value().matrix().transpose(); // column-major storage of the transpose: row-major order
            else if (const auto *vector6 = dynamic_cast<const gtsam::GenericValue<Vector6> *>(&value))
                entries = vector6->value();
            else
                entries = values.at<gtsam::Vector>(key);
            const std::string symbol = gtsam::Symbol(key).string();
            for (Eigen::Index i = 0; i < entries.size(); i++)
                estimates << symbol << "[" << i << "]," << entries.data()[i] << "\n";
        }
        joints << std::setprecision(17) << "joint,residual,travel_a,travel_b\n";
        for (const auto &r : reports)
            joints << r.id << "," << r.residual << "," << r.travel_a << "," << r.travel_b << "\n";
    }

#ifdef VISUALIZER
    /**
     * @brief The floor the robots stand on: a horizontal plane (normal +z) at z = 0, centred
     *        under the robots' bases and sized to their horizontal spread, with a margin.
     *        Every demo's robots start on (bucky_ball: around) z = 0.
     */
    inline Spacetime::Visualization::GroundPlaneOptions groundPlaneUnder(const std::vector<DemoRobot> &robots)
    {
        Spacetime::Visualization::GroundPlaneOptions plane;
        if (robots.empty())
            return plane;
        Eigen::Vector2d centre = Eigen::Vector2d::Zero();
        for (const auto &robot : robots)
            centre += robot.base.translation().head<2>();
        centre /= static_cast<double>(robots.size());
        double spread = 0.0;
        for (const auto &robot : robots)
            spread = std::max(spread, (robot.base.translation().head<2>() - centre).norm());
        plane.position = Eigen::Vector3d(centre.x(), centre.y(), 0.0);
        plane.size = std::max(0.3, 3.0 * spread);
        return plane;
    }
#endif

    /**
     * @brief Estimates [start_time, end_time] with a batch `estimator` already holding every
     *        robot, joint, end effector and sensor, then (unless `headless`) replays it in the
     *        visualizer, and prints each joint's residual at the end time -- or, if
     *        `summary_only`, just their largest and mean residuals and travels. The visualizer
     *        draws the floor under the robots (groundPlaneUnder), toggled with P.
     * @param show_node_frames Whether the robots' node frames start shown (toggle A), e.g. not where
     *        end effectors' frames already mark them.
     * @param on_estimated Called with the estimate once the run is solved (before the replay), e.g.
     *        for a demo's own report.
     */
    inline int run(Spacetime::Estimator &estimator, const std::vector<DemoRobot> &robots, const std::vector<std::shared_ptr<Joint>> &joints,
                   const std::vector<std::shared_ptr<EndEffector>> &end_effectors, double start_time, double end_time, double period, bool headless,
                   int &argc, char *argv[], bool summary_only = false, bool show_node_frames = true,
                   const std::function<void(const gtsam::Values &)> &on_estimated = nullptr)
    {
        Spacetime::SessionOptions session_options;
        session_options.start_time = start_time;
        session_options.end_time = end_time;
        session_options.period = period;
        session_options.real_time = false;
        Spacetime::Session session(session_options);

        const auto initial_wall_time = std::chrono::steady_clock::now();
#ifdef VISUALIZER
        std::unique_ptr<Spacetime::Visualization::Default::VisualizerHost> visualizer;
        if (!headless)
        {
            Spacetime::Visualization::VisualizerOptions vis_options;
            vis_options.online = false;
            vis_options.estimator_window = estimator.isBatch() ? Spacetime::Estimator::BATCH_LAG : estimator.smootherLag();
            vis_options.estimation_period = period; // one video frame per estimation tick (V)
            auto build = [&](Spacetime::Visualization::Default::Visualizer &vis)
            {
                using RobotView = Spacetime::Visualization::Default::ContinuumRobot<Node>;
                // Every robot view shares the backbone toggle (B), every sensor view the sensor toggle (S).
                std::map<const Spacetime::Types::State *, std::shared_ptr<RobotView>> views;
                for (const auto &demo_robot : robots)
                {
                    auto view = std::make_shared<RobotView>(demo_robot.state);
                    view->setToggleKey(Qt::Key_B);
                    view->show_node_frames = show_node_frames;
                    vis.addVisualizable(view);
                    view->bindPriorInterpolator(demo_robot.prior);
                    view->bindSnapshotSource([&estimator](double)
                                             { return estimator.getSnapshot(); });
                    views[demo_robot.state.get()] = view;
                }
                for (const auto &queue : estimator.getMeasurementQueues())
                    if (const auto sensor = std::dynamic_pointer_cast<Spacetime::Types::SensorBase>(queue))
                    {
                        auto sensor_view = Spacetime::Visualization::Default::makeSensorView<Values>(sensor);
                        sensor_view->setToggleKey(Qt::Key_S);
                        vis.addVisualizable(sensor_view);
                    }
                for (const auto &joint : joints)
                {
                    auto joint_view = Spacetime::Visualization::Default::Joint::of(*joint, views.at(joint->priorA()->state().get()), views.at(joint->priorB()->state().get()));
                    joint_view->setToggleKey(Qt::Key_N);
                    vis.addVisualizable(joint_view);
                }
                // Each end effector's frame (toggle E), and its joints like any other (N).
                for (const auto &end_effector : end_effectors)
                {
                    auto frame = std::make_shared<Spacetime::Visualization::Default::EndEffector<Pose>>(end_effector);
                    frame->setToggleKey(Qt::Key_E);
                    frame->bindSnapshotSource([&estimator](double)
                                              { return estimator.getSnapshot(); });
                    vis.addVisualizable(frame);
                    for (const auto &joint : jointsOf(*end_effector))
                    {
                        auto joint_view = Spacetime::Visualization::Default::Joint::of(*joint, views.at(joint->priorA()->state().get()), frame);
                        joint_view->setToggleKey(Qt::Key_N);
                        vis.addVisualizable(joint_view);
                    }
                }
                // The floor under the robots (toggle P).
                vis.addVisualizable(std::make_shared<Spacetime::Visualization::Default::GroundPlane>(groundPlaneUnder(robots)));
                // Estimator phase timings along the bottom of the window, toggled with H (the run has ended before playback: its average and total, or total only in batch mode).
                vis.addVisualizable(std::make_shared<Spacetime::Visualization::Default::RuntimeWidget>([&estimator]()
                                                                                                        { return estimator.getSnapshot(); }));
            };
            visualizer = std::make_unique<Spacetime::Visualization::Default::VisualizerHost>(argc, argv, vis_options, "", build);
        }
#else
        (void)headless;
        (void)show_node_frames;
        (void)argc;
        (void)argv;
#endif

        // Offline: estimate everything first (reported below), then replay it in the window, if
        // any (session.join()). Closing the window stops the session.
#ifdef VISUALIZER
        if (visualizer)
            session.start(estimator, *visualizer);
        else
#endif
            session.start(estimator);
        session.joinEstimation();
        const double total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - initial_wall_time).count();
        std::cout << "Estimation complete in " << total_time << " s." << std::endl;

        const double last_time = session.endTime();
        const gtsam::Values &values = estimator.getCurrentValues();
        std::vector<JointReport> reports;
        const auto report = [&](const std::string &name, const std::string &id, const auto &joint)
        {
            const JointReport r{id, jointResidual(joint, last_time, values),
                                travel(*joint.priorA(), joint.arclengthA(), start_time, last_time, period, values),
                                travel(*joint.priorB(), joint.arclengthB(), start_time, last_time, period, values)};
            if (!summary_only)
                std::cout << name << ": residual at t = " << last_time << " is " << r.residual << "; its point on robot A travels up to "
                          << r.travel_a << " m, and on side B up to " << r.travel_b << " m." << std::endl;
            reports.push_back(r);
        };
        for (std::size_t i = 0; i < joints.size(); i++)
            report("Joint " + std::to_string(i), std::to_string(i), *joints[i]);
        for (std::size_t e = 0; e < end_effectors.size(); e++)
        {
            const auto ee_joints = jointsOf(*end_effectors[e]);
            for (std::size_t i = 0; i < ee_joints.size(); i++)
                report("End effector " + std::to_string(e) + ", joint " + std::to_string(i), "ee" + std::to_string(e) + "." + std::to_string(i), *ee_joints[i]);
        }
        if (summary_only && !reports.empty())
        {
            double max_residual = 0.0, sum_residual = 0.0, min_travel = reports.front().travel_a, max_travel = 0.0;
            for (const auto &r : reports)
            {
                max_residual = std::max(max_residual, r.residual);
                sum_residual += r.residual;
                min_travel = std::min({min_travel, r.travel_a, r.travel_b});
                max_travel = std::max({max_travel, r.travel_a, r.travel_b});
            }
            std::cout << reports.size() << " joints at t = " << last_time << ": residual max " << max_residual << ", mean "
                      << sum_residual / reports.size() << "; their points travel " << min_travel << " to " << max_travel << " m." << std::endl;
        }
        if (on_estimated)
            on_estimated(values);
        if (const std::string output_path = outputRequested(argc, argv); !output_path.empty())
            writeOutput(output_path, values, reports);

        session.join(); // runs the window, if any, until it is closed
        return session.viewerExitCode();
    }

    /// Whether `--headless` is among the arguments.
    inline bool headlessRequested(int argc, char *argv[])
    {
        for (int i = 1; i < argc; i++)
            if (std::string(argv[i]) == "--headless")
                return true;
        return false;
    }
} // namespace JointDemo

#endif // EXAMPLES_JOINTDEMO_HPP
