/**
 * @file bind_visualizer.cpp
 * @brief spacetime.visualizer: the Qt/VTK visualizer, a Viewer a session runs
 *        (session.start(estimator, visualizer); session.join()). Python registers the robots'
 *        priors, joints and end effectors to draw; their views are built in C++ (as in
 *        examples/JointDemo.hpp) once the window exists, on its GUI thread (see VisualizerHost).
 */

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <optional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <QObject>
#include <QTimer>
#include <vtkActor.h>
#include <vtkAutoInit.h>
#include <vtkProperty.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/interface/Viewer.hpp>
#include <spacetime/visualizer/default/VisualizerHost.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/objects/EndEffector.hpp>
#include <spacetime/visualizer/default/objects/GroundPlane.hpp>
#include <spacetime/visualizer/default/objects/Joint.hpp>
#include <spacetime/visualizer/default/objects/SensorViews.hpp>
#include <spacetime/visualizer/default/utilities.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        namespace Vis = Spacetime::Visualization;
        using Spacetime::Types::State;

        /**
         * The visualizer of a set of robots, given by their priors (B toggles their backbones,
         * S the estimator's pose sensors on them), joints (N) and end effectors (E), replaying the
         * estimator's snapshots, and optional ground planes (P). A Viewer: a session opens it
         * (start(estimator, visualizer)) and runs it (join()), through a VisualizerHost.
         */
        class Visualizer : public Spacetime::Interface::Viewer
        {
        public:
            Visualizer(Vis::VisualizerOptions options, std::string output_path, Vis::Default::HostMode mode)
                : m_options(std::move(options)), m_output_path(std::move(output_path)), m_mode(mode)
            {
                constexpr unsigned int max_points = Vis::Default::ContinuumRobot<WnoaNode>::kMaxSpatialInterpolationPoints;
                if (m_options.robot.spatial_interpolation_points > max_points)
                    throw py::value_error("Visualizer: options.robot.spatial_interpolation_points must be at most " + std::to_string(max_points) + ".");
            }

            ~Visualizer() override
            {
                if (!m_host)
                    return;
                py::gil_scoped_release release;
                m_host.reset(); // closes the window and, in WorkerThread mode, joins its thread
            }

            /// Viewer: builds the window on its GUI thread (see VisualizerHost::open). Called by Session.start, without the GIL.
            void open(Spacetime::Estimator &estimator, ClosedCallback on_closed) override
            {
                requireNotStarted();
                if (m_options.estimator_window == -1.0 && !estimator.isBatch())
                    m_options.estimator_window = estimator.smootherLag();
                auto build = [this, &estimator](Vis::Default::Visualizer &vis)
                {
                    buildViews(vis, estimator);
                    if (m_mode == Vis::Default::HostMode::CallingThread)
                        watchForInterrupts(vis);
                };
                m_host = std::make_unique<Vis::Default::VisualizerHost>(m_argc, m_argv, m_options, m_output_path, build, m_mode);
                m_host->open(std::move(on_closed));
            }

            bool isOnline() const override { return m_options.online; }

            void beginPlayback(double start_time, double end_time, LiveTimeSource live_time_source) override
            {
                requireStarted();
                if (m_options.online && !live_time_source)
                    throw py::value_error("Visualizer.beginPlayback: online mode needs a live_time_source (e.g. a session's now).");
                m_host->beginPlayback(start_time, end_time, std::move(live_time_source));
            }

            /// Viewer: shows the window until it is closed (see VisualizerHost::run). Called by Session.join, without the GIL.
            int run() override
            {
                requireStarted();
                const int code = m_host->run();
                if (m_interrupt)
                {
                    py::gil_scoped_acquire gil; // the KeyboardInterrupt holds Python objects
                    std::rethrow_exception(std::exchange(m_interrupt, nullptr));
                }
                return code;
            }

            void requestClose() override
            {
                if (m_host)
                    m_host->requestClose();
            }

            Vis::Default::HostMode mode() const { return m_mode; }

            /// The robot of `prior` (any of RobotPriors).
            void addPrior(const pStateful &prior)
            {
                if (!visitPrior(RobotPriors{}, prior, [this](auto typed)
                                { addRobotView<typename decltype(typed)::element_type::NodeType>(typed); }))
                    throw py::type_error("Visualizer.addPrior: the prior must be a wnoa, wnov or wnoa_quasi_static Prior.");
            }

            /// A joint between two robots added already.
            void addJoint(const std::shared_ptr<JointBase> &joint)
            {
                requireNotStarted();
                if (std::dynamic_pointer_cast<EndEffectorPrior>(joint->priorModelB()))
                    throw py::value_error("Visualizer.addJoint: this joint is an end effector's; add the end effector (addEndEffector), which draws its joints.");
                requireRobot(joint->stateA().get());
                requireRobot(joint->stateB().get());
                m_joints.push_back(joint);
            }

            /// An end effector, whose joints' robots are added already.
            void addEndEffector(const std::shared_ptr<EndEffector> &end_effector)
            {
                requireNotStarted();
                for (const auto &joint : end_effector->joints())
                    requireRobot(joint->stateA().get());
                m_end_effectors.push_back(end_effector);
            }

            /// An STL mesh drawn rigidly attached to the robot of `prior` (added already) at `arclength`; see Default::ContinuumRobot::addSTL.
            void addSTL(const pStateful &prior, const std::string &stl_path, double arclength, double scale, const gtsam::Pose3 &offset,
                        std::optional<Eigen::Vector3d> color)
            {
                requireNotStarted();
                const State *state = nullptr;
                double length = 0.0;
                if (!visitPrior(RobotPriors{}, prior, [&](auto typed)
                                { state = typed->state().get(), length = typed->state()->length; }))
                    throw py::type_error("Visualizer.addSTL: the prior must be a wnoa, wnov or wnoa_quasi_static Prior.");
                if (std::find(m_robot_order.begin(), m_robot_order.end(), state) == m_robot_order.end())
                    throw py::value_error("Visualizer.addSTL: add the robot (addPrior) before its meshes.");
                // Checked here rather than when the view is built on the visualizer thread, so the error points at this call.
                if (!(arclength >= 0.0 && arclength <= length))
                    throw py::value_error("Visualizer.addSTL: arclength " + std::to_string(arclength) + " is outside the robot [0, " +
                                          std::to_string(length) + "].");
                if (!(scale > 0.0 && std::isfinite(scale)))
                    throw py::value_error("Visualizer.addSTL: the scale must be positive and finite.");
                // Read once here to reject a missing or unreadable file now: a view that fails to build fails Session.start.
                try
                {
                    Vis::Default::Utilities::loadSTLMesh(stl_path);
                }
                catch (const std::runtime_error &error)
                {
                    throw py::value_error(std::string("Visualizer.addSTL: ") + error.what());
                }
                m_meshes[state].push_back({stl_path, arclength, scale, offset, std::move(color)});
            }

            /// A ground plane (toggle P); see Default::GroundPlane.
            void addGroundPlane(const Vis::GroundPlaneOptions &options)
            {
                requireNotStarted();
                try
                {
                    Vis::Default::GroundPlane check(options); // validates now, so the error points at this call
                }
                catch (const std::invalid_argument &error)
                {
                    throw py::value_error(std::string("Visualizer.addGroundPlane: ") + error.what());
                }
                m_ground_planes.push_back(options);
            }

            std::size_t robotCount() const { return m_robot_order.size(); }

        private:
            Vis::VisualizerOptions m_options;
            std::string m_output_path;
            Vis::Default::HostMode m_mode;
            int m_argc = 1;
            char m_arg0[10] = "spacetime";
            char *m_argv[2] = {m_arg0, nullptr};

            /// Builds each robot's view in the visualizer: it is called on the visualizer thread.
            std::vector<std::function<void(Vis::Default::Visualizer &, Spacetime::Estimator &)>> m_robot_builders;
            std::vector<const State *> m_robot_order;
            std::vector<std::shared_ptr<JointBase>> m_joints;
            std::vector<std::shared_ptr<EndEffector>> m_end_effectors;
            std::vector<Vis::GroundPlaneOptions> m_ground_planes;
            /// Per robot (built on the visualizer thread): the source of its drawn pose at an arclength, for its joints' views.
            std::map<const State *, std::function<Vis::Default::Joint::PointPoseFn(double)>> m_point_sources;
            std::unique_ptr<Vis::Default::VisualizerHost> m_host;
            /// The KeyboardInterrupt (Ctrl-C) that closed the window, rethrown by run().
            std::exception_ptr m_interrupt;

            struct Mesh
            {
                std::string path;
                double arclength;
                double scale;
                gtsam::Pose3 offset;
                std::optional<Eigen::Vector3d> color;
            };
            /// The meshes added with addSTL, per robot; attached when the robot's view is built.
            std::map<const State *, std::vector<Mesh>> m_meshes;

            template <typename NodeT, typename PriorT>
            void addRobotView(const std::shared_ptr<PriorT> &prior)
            {
                requireNotStarted();
                m_robot_order.push_back(prior->state().get());
                m_robot_builders.push_back(
                    [this, prior](Vis::Default::Visualizer &vis, Spacetime::Estimator &estimator)
                    {
                        auto view = std::make_shared<Vis::Default::ContinuumRobot<NodeT>>(prior->state());
                        view->radius = m_options.robot.radius;
                        view->setToggleKey(Qt::Key_B);
                        vis.addVisualizable(view);
                        view->bindPriorInterpolator(prior);
                        view->setSpatialInterpolationPoints(m_options.robot.spatial_interpolation_points); // after the interpolator, without which it is a no-op
                        view->bindSnapshotSource([&estimator](double) { return estimator.getSnapshot(); });
                        if (const auto meshes = m_meshes.find(prior->state().get()); meshes != m_meshes.end())
                            for (const auto &mesh : meshes->second)
                            {
                                const auto actor = view->addSTL(mesh.path, mesh.arclength, mesh.scale, mesh.offset);
                                if (mesh.color)
                                    actor->GetProperty()->SetColor(mesh.color->x(), mesh.color->y(), mesh.color->z());
                            }
                        for (const auto &queue : estimator.getMeasurementQueues())
                            if (const auto sensor = std::dynamic_pointer_cast<Spacetime::Types::SensorBase>(queue); sensor && sensor->stateful() == prior)
                            {
                                auto sensor_view = Vis::Default::makeSensorView<typename NodeT::ValuesType>(sensor);
                                sensor_view->setToggleKey(Qt::Key_S);
                                vis.addVisualizable(sensor_view);
                            }
                        m_point_sources[prior->state().get()] = [view](double s) { return Vis::Default::Joint::pointOn(view, s); };
                    });
            }

            void buildViews(Vis::Default::Visualizer &vis, Spacetime::Estimator &estimator)
            {
                for (const auto &plane : m_ground_planes)
                    vis.addVisualizable(std::make_shared<Vis::Default::GroundPlane>(plane));
                for (const auto &builder : m_robot_builders)
                    builder(vis, estimator);
                const auto add_joint = [&vis](Vis::Default::Joint::PointPoseFn point_a, Vis::Default::Joint::PointPoseFn point_b, bool has_offset)
                {
                    auto view = std::make_shared<Vis::Default::Joint>(std::move(point_a), std::move(point_b), has_offset);
                    view->setToggleKey(Qt::Key_N);
                    vis.addVisualizable(view);
                };
                const auto point_on = [this](const JointBase &joint) { return m_point_sources.at(joint.stateA().get())(joint.arclengthA()); };
                for (const auto &joint : m_joints)
                    add_joint(point_on(*joint), m_point_sources.at(joint->stateB().get())(joint->arclengthB()), joint->hasOffset());
                for (const auto &end_effector : m_end_effectors)
                {
                    auto frame = std::make_shared<Vis::Default::EndEffector<Pose>>(end_effector);
                    frame->setToggleKey(Qt::Key_E);
                    frame->bindSnapshotSource([&estimator](double) { return estimator.getSnapshot(); });
                    vis.addVisualizable(frame);
                    for (const auto &joint : end_effector->joints())
                        add_joint(point_on(*joint), frame->pointPose(), joint->hasOffset());
                }
            }

            /// CallingThread mode: the event loop runs on Python's main thread, whose signals (Ctrl-C) are only
            /// handled by Python code; so check for them now and then, and close the window on one.
            void watchForInterrupts(Vis::Default::Visualizer &vis)
            {
                auto *timer = new QTimer(&vis);
                QObject::connect(timer, &QTimer::timeout, &vis, [this, timer]()
                                 {
                    py::gil_scoped_acquire gil;
                    if (PyErr_CheckSignals() == 0)
                        return;
                    m_interrupt = std::make_exception_ptr(py::error_already_set());
                    timer->stop();
                    m_host->requestClose(); });
                timer->start(100);
            }

            void requireRobot(const State *state) const
            {
                if (std::find(m_robot_order.begin(), m_robot_order.end(), state) == m_robot_order.end())
                    throw py::value_error("Visualizer: add the joint's robots (addPrior) before the joint.");
            }

            void requireNotStarted() const
            {
                if (m_host)
                    throw std::runtime_error("Visualizer: already opened (by Session.start); add everything before.");
            }

            void requireStarted() const
            {
                if (!m_host)
                    throw std::runtime_error("Visualizer: not opened; give it to Session.start(estimator, visualizer).");
            }
        };
    } // namespace

    void bindVisualizer(py::module_ &m)
    {
        using Vis::VisualizerOptions;
        using CameraOptions = VisualizerOptions::CameraOptions;
        using RobotOptions = VisualizerOptions::RobotOptions;

        py::class_<CameraOptions>(m, "CameraOptions", "A camera viewpoint, in the world frame.")
            .def(py::init<>())
            .def_readwrite("position", &CameraOptions::position, "Camera position [m].")
            .def_readwrite("focal_point", &CameraOptions::focal_point, "The point looked at [m].")
            .def_readwrite("view_up", &CameraOptions::view_up, "The up direction.");

        py::class_<RobotOptions>(m, "RobotOptions", "How the robot views are drawn.")
            .def(py::init<>())
            .def_readwrite("radius", &RobotOptions::radius, "Backbone tube radius [m]; also scales the node frames.")
            .def_readwrite("spatial_interpolation_points", &RobotOptions::spatial_interpolation_points,
                           "Extra backbone points drawn between each pair of adjacent nodes at startup (0 to 10; [ and ] adjust it at runtime). "
                           "Visualizer() raises ValueError above 10.");

        using GroundPlaneOptions = Vis::GroundPlaneOptions;
        py::class_<GroundPlaneOptions>(m, "GroundPlaneOptions",
                                       "A square ground plane, drawn only when added (Visualizer.addGroundPlane; "
                                       "see utilities.ConfigLoader.loadGroundPlaneOptions).")
            .def(py::init<>())
            .def(py::init([](const Eigen::Vector3d &normal, const Eigen::Vector3d &position, double size)
                          { return GroundPlaneOptions{normal, position, size}; }),
                 py::arg("normal") = Eigen::Vector3d::UnitZ().eval(), py::arg("position") = Eigen::Vector3d::Zero().eval(), py::arg("size") = 1.0)
            .def_readwrite("normal", &GroundPlaneOptions::normal, "Plane normal (world frame); nonzero, need not be unit length.")
            .def_readwrite("position", &GroundPlaneOptions::position, "Centre of the plane [m].")
            .def_readwrite("size", &GroundPlaneOptions::size, "Side length of the square [m]; positive.");

        py::class_<VisualizerOptions>(m, "VisualizerOptions", "Display options of the visualizer (see utilities.ConfigLoader.loadVisualizerOptions).")
            .def(py::init<>())
            .def_readwrite("cameras", &VisualizerOptions::cameras,
                           "The first is the interactive view, the rest only extra recorded screenshots; empty: the camera is fitted "
                           "to the scene. A copy: assign the whole list.")
            .def_readwrite("robot", &VisualizerOptions::robot, "Applied to every robot added with addPrior.")
            .def_readwrite("show_world_frame", &VisualizerOptions::show_world_frame, "Draw the world coordinate frame.")
            .def_readwrite("show_info_label", &VisualizerOptions::show_info_label, "Currently unused.")
            .def_readwrite("show_diagnostics_label", &VisualizerOptions::show_diagnostics_label, "Show the diagnostics overlay at startup (H toggles it).")
            .def_readwrite("show_timeline_widget", &VisualizerOptions::show_timeline_widget, "Show the timeline at startup (Y toggles it).")
            .def_readwrite("stale_timeout", &VisualizerOptions::stale_timeout, "Seconds after its last data before a non-static view is hidden.")
            .def_readwrite("online", &VisualizerOptions::online,
                           "Online: playback follows wall-clock time and views read the live estimate. Offline (default): replays the run.")
            .def_readwrite("estimator_window", &VisualizerOptions::estimator_window,
                           "The estimator's fixed-lag window [s], bounding the display delay; <= 0 means unknown. "
                           "Session.start(estimator, visualizer) fills it in from a sliding-window estimator when left at -1.")
            .def_readwrite("estimation_period", &VisualizerOptions::estimation_period,
                           "The estimator's tick period [s]: offline, the arrow keys step between ticks and V exports one video frame per tick. "
                           "<= 0 (the default) means unknown.");

        py::enum_<Vis::Default::HostMode>(m, "HostMode", "Where the visualizer runs Qt (see Visualizer).")
            .value("WorkerThread", Vis::Default::HostMode::WorkerThread, "A thread of its own, lowered in priority (Linux's default).")
            .value("CallingThread", Vis::Default::HostMode::CallingThread,
                   "The thread calling Session.start and join (macOS and Windows' default; on macOS the main thread).");
        m.def("defaultHostMode", &Vis::Default::defaultHostMode, "This platform's default HostMode.");

        py::classh<Visualizer, Spacetime::Interface::Viewer>(m, "Visualizer",
                               "The Qt/VTK visualizer window, a Viewer: add what to draw, then session.start(estimator, visualizer) and "
                               "session.join(), which shows it until it is closed (closing it stops the session). Keys B, S, N and E toggle "
                               "the robots, pose sensors, joints and end effectors. On macOS, start and join the session from the main thread. "
                               "Dropping it closes the window.")
            .def(py::init<VisualizerOptions, std::string, Vis::Default::HostMode>(), py::arg("options"), py::arg("output_path") = "",
                 py::arg("host_mode") = Vis::Default::defaultHostMode(),
                 "Screenshots and videos go to <output_path>/screenshots and <output_path>/videos (empty: the working directory). "
                 "`host_mode`: where Qt runs (see HostMode); the platform's default unless testing.")
            .def("hostMode", &Visualizer::mode, "Where Qt runs (see HostMode).")
            .def("addPrior", &Visualizer::addPrior, py::arg("prior"), py::keep_alive<1, 2>(),
                 "Draws the robot of `prior` (wnoa, wnov or wnoa_quasi_static; else TypeError), and its pose sensors.")
            .def("addJoint", &Visualizer::addJoint, py::arg("joint"), py::keep_alive<1, 2>(),
                 "Draws a joint between two robots added already (else ValueError). An end effector's joints are drawn by addEndEffector.")
            .def("addEndEffector", &Visualizer::addEndEffector, py::arg("end_effector"), py::keep_alive<1, 2>(),
                 "Draws an end effector's frame and joints; its robots must be added already (else ValueError).")
            .def("addSTL", &Visualizer::addSTL, py::arg("prior"), py::arg("stl_path"), py::arg("arclength"), py::arg("scale") = 1.0,
                 py::arg("offset") = gtsam::Pose3(), py::arg("color") = py::none(),
                 "Draws the STL mesh at `stl_path` rigidly attached to the robot of `prior` (WNOA, WNOV or WNOA quasi-static, added already with addPrior) at "
                 "`arclength` [m]: at world_T_point * offset * Scale(scale), i.e. scaled by `scale` in its own frame (e.g. 0.001 for a mesh "
                 "in millimetres), then placed by `offset` (gtsam.Pose3, point_T_mesh) relative to the robot's frame there. `color` is an "
                 "optional RGB triple in [0, 1]. It is shown with the robot's backbone (B). ValueError if the robot was not added, "
                 "`arclength` is outside [0, length], `scale` is not positive and finite, or the file is missing or holds no STL geometry. "
                 "Add meshes before Session.start.")
            .def("addGroundPlane", &Visualizer::addGroundPlane, py::arg("options") = GroundPlaneOptions(),
                 "Draws a square ground plane (P toggles it). Raises ValueError for a zero normal or a non-positive size.")
            .def("robotCount", &Visualizer::robotCount, "The number of robots added.");
    }
} // namespace SpacetimePython
