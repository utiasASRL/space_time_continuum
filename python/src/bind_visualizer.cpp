/**
 * @file bind_visualizer.cpp
 * @brief spacetime.visualizer: the Qt/VTK visualizer, on its own thread. Python registers the
 *        robots' priors, joints and end effectors to draw; their views are built in C++ (as in
 *        examples/JointDemo.hpp) once the window exists.
 */

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/VisualizerThread.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/objects/EndEffector.hpp>
#include <spacetime/visualizer/default/objects/Joint.hpp>
#include <spacetime/visualizer/default/objects/SensorViews.hpp>

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
         * estimator's snapshots.
         */
        class Visualizer
        {
        public:
            Visualizer(Vis::VisualizerOptions options, std::string output_path) : m_options(std::move(options)), m_output_path(std::move(output_path)) {}

            ~Visualizer()
            {
                if (!m_thread)
                    return;
                py::gil_scoped_release release;
                m_thread->requestQuit();
                m_thread->join();
                m_thread.reset();
            }

            void addPrior(const std::shared_ptr<WnoaPrior> &prior) { addRobotView<WnoaNode>(prior); }
            void addWnovPrior(const std::shared_ptr<WnovPrior> &prior) { addRobotView<WnovNode>(prior); }

            /// A joint between two robots added already.
            void addJoint(const std::shared_ptr<RobotJoint> &joint)
            {
                requireNotStarted();
                requireRobot(joint->priorA()->state().get());
                requireRobot(joint->priorB()->state().get());
                m_joints.push_back(joint);
            }

            /// An end effector, whose joints' robots are added already.
            void addEndEffector(const std::shared_ptr<EndEffector> &end_effector)
            {
                requireNotStarted();
                for (const auto &model : end_effector->joints())
                    requireRobot(std::dynamic_pointer_cast<EndEffectorJoint>(model)->priorA()->state().get());
                m_end_effectors.push_back(end_effector);
            }

            std::size_t robotCount() const { return m_robot_order.size(); }

            /// Opens the window, drawing `estimator`'s snapshots; `on_closed` (e.g. a session's requestStop) runs when it is closed.
            void start(Spacetime::Estimator &estimator, std::function<void()> on_closed)
            {
                requireNotStarted();
                if (m_options.estimator_window == -1.0 && !estimator.isBatch())
                    m_options.estimator_window = estimator.smootherLag();
                auto build = [this, &estimator](Vis::Default::Visualizer &vis) { buildViews(vis, estimator); };
                py::gil_scoped_release release;
                m_thread = std::make_unique<Vis::Default::VisualizerThread>(m_argc, m_argv, m_options, m_output_path, build, std::move(on_closed));
                m_thread->waitReady();
            }

            void beginPlayback(double start_time, double end_time)
            {
                requireStarted();
                m_thread->beginPlayback(start_time, end_time);
            }

            /// Waits for the window to be closed; returns the Qt exit code.
            int join()
            {
                requireStarted();
                py::gil_scoped_release release;
                const int code = m_thread->join();
                m_thread.reset();
                return code;
            }

        private:
            using RobotView = Vis::Default::ContinuumRobot<WnoaNode>;

            Vis::VisualizerOptions m_options;
            std::string m_output_path;
            int m_argc = 1;
            char m_arg0[10] = "spacetime";
            char *m_argv[2] = {m_arg0, nullptr};

            /// Builds each robot's view in the visualizer: it is called on the visualizer thread.
            std::vector<std::function<void(Vis::Default::Visualizer &, Spacetime::Estimator &)>> m_robot_builders;
            std::vector<const State *> m_robot_order;
            std::vector<std::shared_ptr<RobotJoint>> m_joints;
            std::vector<std::shared_ptr<EndEffector>> m_end_effectors;
            std::map<const State *, std::shared_ptr<RobotView>> m_views; // the WNOA robots' views, for their joints
            std::unique_ptr<Vis::Default::VisualizerThread> m_thread;

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
                        view->bindSnapshotSource([&estimator](double) { return estimator.getSnapshot(); });
                        for (const auto &queue : estimator.getMeasurementQueues())
                            if (const auto sensor = std::dynamic_pointer_cast<Spacetime::Types::SensorBase>(queue); sensor && sensor->stateful() == prior)
                            {
                                auto sensor_view = Vis::Default::makeSensorView<typename NodeT::ValuesType>(sensor);
                                sensor_view->setToggleKey(Qt::Key_S);
                                vis.addVisualizable(sensor_view);
                            }
                        if constexpr (std::is_same_v<NodeT, WnoaNode>)
                            m_views[prior->state().get()] = view;
                    });
            }

            void buildViews(Vis::Default::Visualizer &vis, Spacetime::Estimator &estimator)
            {
                for (const auto &builder : m_robot_builders)
                    builder(vis, estimator);
                for (const auto &joint : m_joints)
                {
                    auto view = Vis::Default::Joint::of(*joint, m_views.at(joint->priorA()->state().get()), m_views.at(joint->priorB()->state().get()));
                    view->setToggleKey(Qt::Key_N);
                    vis.addVisualizable(view);
                }
                for (const auto &end_effector : m_end_effectors)
                {
                    auto frame = std::make_shared<Vis::Default::EndEffector<Pose>>(end_effector);
                    frame->setToggleKey(Qt::Key_E);
                    frame->bindSnapshotSource([&estimator](double) { return estimator.getSnapshot(); });
                    vis.addVisualizable(frame);
                    for (const auto &model : end_effector->joints())
                    {
                        const auto joint = std::dynamic_pointer_cast<EndEffectorJoint>(model);
                        auto view = Vis::Default::Joint::of(*joint, m_views.at(joint->priorA()->state().get()), frame);
                        view->setToggleKey(Qt::Key_N);
                        vis.addVisualizable(view);
                    }
                }
            }

            void requireRobot(const State *state) const
            {
                if (std::find(m_robot_order.begin(), m_robot_order.end(), state) == m_robot_order.end())
                    throw py::value_error("Visualizer: add the joint's robots (addPrior) before the joint.");
            }

            void requireNotStarted() const
            {
                if (m_thread)
                    throw std::runtime_error("Visualizer: already started; add everything before start().");
            }

            void requireStarted() const
            {
                if (!m_thread)
                    throw std::runtime_error("Visualizer: not started.");
            }
        };
    } // namespace

    void bindVisualizer(py::module_ &m)
    {
        using Vis::VisualizerOptions;
        using CameraOptions = VisualizerOptions::CameraOptions;
        using RobotOptions = VisualizerOptions::RobotOptions;

        py::class_<CameraOptions>(m, "CameraOptions")
            .def(py::init<>())
            .def_readwrite("position", &CameraOptions::position)
            .def_readwrite("focal_point", &CameraOptions::focal_point)
            .def_readwrite("view_up", &CameraOptions::view_up);

        py::class_<RobotOptions>(m, "RobotOptions", "How the robot views are drawn.")
            .def(py::init<>())
            .def_readwrite("radius", &RobotOptions::radius, "Backbone tube radius [m]; also scales the node frames.");

        py::class_<VisualizerOptions>(m, "VisualizerOptions")
            .def(py::init<>())
            .def_readwrite("cameras", &VisualizerOptions::cameras, "One view per camera; empty for a single default view. A copy: assign the whole list.")
            .def_readwrite("robot", &VisualizerOptions::robot, "Applied to every robot added with addPrior.")
            .def_readwrite("show_floor", &VisualizerOptions::show_floor)
            .def_readwrite("show_world_frame", &VisualizerOptions::show_world_frame)
            .def_readwrite("show_info_label", &VisualizerOptions::show_info_label)
            .def_readwrite("show_diagnostics_label", &VisualizerOptions::show_diagnostics_label)
            .def_readwrite("show_timeline_widget", &VisualizerOptions::show_timeline_widget)
            .def_readwrite("stale_timeout", &VisualizerOptions::stale_timeout)
            .def_readwrite("online", &VisualizerOptions::online)
            .def_readwrite("estimator_window", &VisualizerOptions::estimator_window)
            .def_readwrite("estimation_period", &VisualizerOptions::estimation_period,
                           "The estimator's tick period [s]: offline, the arrow keys step between ticks and V exports one video frame per tick. "
                           "<= 0 (the default) means unknown.");

        py::class_<Visualizer>(m, "Visualizer", "The visualizer window, on its own thread; add what to draw, then start().")
            .def(py::init<VisualizerOptions, std::string>(), py::arg("options"), py::arg("output_path") = "")
            .def("addPrior", &Visualizer::addPrior, py::arg("prior"), py::keep_alive<1, 2>(), "Draws the robot of `prior` (WNOA or WNOV), and its pose sensors.")
            .def("addPrior", &Visualizer::addWnovPrior, py::arg("prior"), py::keep_alive<1, 2>())
            .def("addJoint", &Visualizer::addJoint, py::arg("joint"), py::keep_alive<1, 2>())
            .def("addEndEffector", &Visualizer::addEndEffector, py::arg("end_effector"), py::keep_alive<1, 2>())
            .def("robotCount", &Visualizer::robotCount)
            .def("start", &Visualizer::start, py::arg("estimator"), py::arg("on_closed") = nullptr, py::keep_alive<1, 2>())
            .def("beginPlayback", &Visualizer::beginPlayback, py::arg("start_time"), py::arg("end_time"))
            .def("join", &Visualizer::join);
    }
} // namespace SpacetimePython
