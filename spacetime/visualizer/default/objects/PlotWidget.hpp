#ifndef DEFAULT_PLOTWIDGET_HPP
#define DEFAULT_PLOTWIDGET_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include <QColor>
#include <QRect>
#include <QSize>
#include <QString>
#include <gtsam/geometry/Pose3.h>
#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkMatrix4x4.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkPropPicker.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/visualizer/MeasurementHistory.hpp>
#include <spacetime/visualizer/PlotData.hpp>
#include <spacetime/visualizer/default/PlotRendering.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/utilities.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/objects/PlotPanel.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief State plots of a robot (the legacy visualizer's PlotWidget): a panel on
     *        the right of the window plotting every variable of the robot's nodes --
     *        whatever the node type holds (see NodeHistory), e.g. pose, strain and
     *        velocity for WNOA, pose only for WNOV -- with +/- 3 sigma bounds when the
     *        snapshot carries a posterior covariance, and pose ground truth overlaid.
     *        Two modes (M): against time at the selected node's arclength, or against
     *        arclength at the displayed time. The selected node is stepped with J/K or
     *        picked by clicking its frame, and highlighted in the scene. Each plot can be
     *        panned and zoomed (see PlotView); hiding the panel or switching mode resets
     *        them. The panel is toggled with H, together with the diagnostics panel.
     *
     *        Alternatively, built from named states with no robot view (e.g. the bias
     *        states of a robot's IMUs, see Biases::BiasModel), it plots the last spatial
     *        node of each state against time, in one panel, its groups prefixed with the
     *        state's name. It then binds only H: no mode, selection or highlight.
     *
     *        Reads the robot view's state and snapshot source (see
     *        ContinuumRobot::bindSnapshotSource), keeping every node's last plotted
     *        value, so the plots cover the whole run rather than only the estimator's
     *        window. Plots are only re-rendered while shown, and only when what they
     *        show changed.
     */
    template <Types::NodeType NodeT>
    class PlotWidget : public Visualizable
    {
    public:
        using StateType = Types::ContinuumRobotState<NodeT>;
        /// The snapshot to plot for an estimate time (see ContinuumRobot::bindSnapshotSource).
        using SnapshotSource = std::function<std::shared_ptr<const EstimatorSnapshot>(double)>;

        /// A state plotted under a name (see the named-states constructor).
        struct NamedState
        {
            std::string name;
            std::shared_ptr<const StateType> state;
        };

        /**
         * @param robot_view           The robot view whose state, snapshots and node frames are used. Add it to the visualizer too.
         * @param ground_truth_sensors Ground-truth sensors; pose sensors among them are overlaid on the pose plots.
         *                             Subscribed to here (see MeasurementHistory), so construct this before their data is consumed.
         * @throws std::invalid_argument if `robot_view` is null.
         */
        explicit PlotWidget(std::shared_ptr<const ContinuumRobot<NodeT>> robot_view,
                            const std::vector<std::shared_ptr<Types::SensorBase>> &ground_truth_sensors = {})
            : m_selection(requireView(robot_view)->state()->spatial_dim)
        {
            m_sources.push_back(Source{"", robot_view->state(), {}});
            // Read lazily: the view's snapshot source may be bound after this is built.
            m_snapshot_source = [view = robot_view.get()](double estimate_time)
            { return view->mp_snapshot_source ? view->mp_snapshot_source(estimate_time) : nullptr; };
            mp_robot_view = robot_view;
            mp_node_frames = &robot_view->mp_axes;
            m_highlight_scale = std::max(0.015, 3.0 * robot_view->radius);
            if constexpr (requires { typename Sensors::PoseSensor<typename NodeT::ValuesType>; })
                for (const auto &sensor : ground_truth_sensors)
                    if (std::dynamic_pointer_cast<Sensors::PoseSensor<typename NodeT::ValuesType>>(sensor))
                        m_ground_truth.emplace_back(*sensor, kGroundTruthCapacity);
        }

        /**
         * @param states          The states to plot, in panel order; each one's last spatial node is plotted against time.
         * @param snapshot_source The snapshots holding their values.
         * @throws std::invalid_argument if `states` is empty, holds a null state, or `snapshot_source` is empty.
         */
        PlotWidget(const std::vector<NamedState> &states, SnapshotSource snapshot_source)
            : m_snapshot_source(std::move(snapshot_source)), m_selection(0)
        {
            if (states.empty())
                throw std::invalid_argument("Default::PlotWidget requires at least one state.");
            if (!m_snapshot_source)
                throw std::invalid_argument("Default::PlotWidget requires a snapshot source.");
            for (const auto &named : states)
            {
                if (!named.state)
                    throw std::invalid_argument("Default::PlotWidget requires non-null states.");
                m_sources.push_back(Source{named.name, named.state, {}});
            }
        }

        /// Short name if `brief`, otherwise a one-line description.
        std::string info(bool brief = true) const override { return brief ? "PlotWidget" : "PlotWidget: state plots against time or arclength"; }

        /// Plotters for the (first plotted state's) node variables; register one here for a variable type with no built-in plotter (see PlotterRegistry).
        PlotterRegistry &plotters() { return m_sources.front().history.plotters(); }

        /// Creates the panel (and, with a robot view, the selection highlight), and binds H (panel) plus, with a robot view, M (mode), J/K and left-click (selected node).
        void visualize(Visualizer &visualizer) const override
        {
            const bool with_robot = hasRobotView();
            mp_panel = new PlotPanel(&visualizer, [this]()
                                     { toggleMode(); });
            mp_panel->setModeButtonVisible(with_robot);
            visualizer.addOverlayWidget(mp_panel, [](QSize window)
                                        {
                // Right of the diagnostics panel, above the timeline (as in the legacy visualizer).
                const int width = std::clamp(window.width() - 370, 320, 910);
                return QRect(window.width() - width - 10, 10, width, std::max(100, window.height() - 70 - Visualizer::bottomBarHeight())); });
            visible_ = visualizer.diagnosticsVisible(); // H toggles both, so start in sync
            mp_panel->setVisible(visible_);

            visualizer.bindShortcut(Qt::Key_H, [this]()
                                    { toggleVisibility(); });
            visualizer.addControlInfo("Toggle plots [h]");
            bindVisibilityToggle(visualizer, info(true));
            if (!with_robot)
                return;

            mp_highlight = vtkSmartPointer<vtkActor>::New();
            mp_highlight->SetMapper(Utilities::createUnitSphereMapper());
            mp_highlight->GetProperty()->SetColor(1.0, 0.55, 0.0);
            mp_highlight->GetProperty()->SetOpacity(0.5);
            mp_highlight->SetScale(m_highlight_scale);
            mp_highlight->SetVisibility(false);
            visualizer.addActor(mp_highlight);

            visualizer.bindShortcut(Qt::Key_M, [this]()
                                    { toggleMode(); });
            visualizer.bindShortcut(Qt::Key_J, [this]()
                                    { stepSelection(-1); });
            visualizer.bindShortcut(Qt::Key_K, [this]()
                                    { stepSelection(+1); });
            visualizer.addControlInfo("Switch plot mode [m]");
            visualizer.addControlInfo("Select plotted node [j, k, click]");

            mp_renderer = visualizer.getRenderer();
            mp_picker = vtkSmartPointer<vtkPropPicker>::New();
            mp_click_callback = vtkSmartPointer<vtkCallbackCommand>::New();
            mp_click_callback->SetClientData(const_cast<PlotWidget *>(this));
            mp_click_callback->SetCallback([](vtkObject *caller, unsigned long, void *client_data, void *)
                                           { static_cast<const PlotWidget *>(client_data)->onLeftClick(static_cast<vtkRenderWindowInteractor *>(caller)); });
            visualizer.getInteractor()->AddObserver(vtkCommand::LeftButtonPressEvent, mp_click_callback);
        }

        /**
         * @brief Records the snapshot for `estimate_time` if it is new (also while hidden,
         *        so the history keeps accumulating), moves the selection highlight, and
         *        re-renders the plots if shown and anything they show changed. The
         *        displayed row is the latest at or before min(time, snapshot time), as
         *        for the robot view.
         */
        void updateVisualization(double time, double estimate_time) const override
        {
            if (!mp_panel)
                return; // not visualized yet
            ingest(estimate_time);

            updateHighlight();
            if (!visible_)
                return;

            const double display_time = displayTime(time);
            RenderKey key{{}, m_selection.revision(), {}, mp_panel->plotSize()};
            for (const auto &source : m_sources)
            {
                key.history_revisions.push_back(source.history.revision());
                key.rows.push_back(source.history.latestRowAtOrBefore(display_time));
            }
            if (m_last_render && *m_last_render == key)
                return;
            m_last_render = key;
            render(display_time);
        }

        /**
         * @brief Records the snapshot for `estimate_time` into every state's history, unless
         *        it is the one recorded last (or there is none).
         * @return Whether a new snapshot was recorded.
         */
        bool ingest(double estimate_time) const
        {
            auto snapshot = m_snapshot_source(estimate_time);
            if (!snapshot || snapshot == mp_last_snapshot)
                return false;
            mp_last_snapshot = snapshot;
            for (auto &source : m_sources)
                source.history.ingest(source.state->snapshotEstimationNodes(), source.state->spatial_dim, *snapshot);
            return true;
        }

        /**
         * @brief The plot groups shown at display `time` (clamped to the last recorded
         *        snapshot's time), without ground truth: per state, its groups for its latest
         *        row at or before then, named "<state name> <variable>" for named states.
         */
        std::vector<PlotGroup> plotGroups(double time) const
        {
            const double display_time = displayTime(time);
            const bool by_time = m_selection.mode() == PlotMode::TimeAtArclength;
            std::vector<PlotGroup> groups;
            for (const auto &source : m_sources)
            {
                const auto row = source.history.latestRowAtOrBefore(display_time);
                if (!row)
                    continue;
                auto slice = by_time ? source.history.sliceAtArclength(plottedSpatial(source), *source.history.rowTime(*row))
                                     : source.history.sliceAtTime(*row);
                for (auto &group : slice)
                {
                    if (!source.name.empty())
                        group.name = source.name + " " + group.name;
                    groups.push_back(std::move(group));
                }
            }
            return groups;
        }

        /// Shows or hides the panel (and the selection highlight). Hiding resets every plot's pan/zoom.
        void toggleVisibility() const override
        {
            visible_ = !visible_;
            if (mp_panel)
            {
                mp_panel->setVisible(visible_);
                if (!visible_)
                    mp_panel->resetViews();
            }
            updateHighlight();
        }

        /// Switches between plotting against time and against arclength, resetting every plot's pan/zoom (the x-axis changes meaning). Only with a robot view.
        void toggleMode() const
        {
            if (!hasRobotView())
                return;
            m_selection.toggleMode();
            if (mp_panel)
                mp_panel->resetViews();
        }
        /// Moves the selected node `delta` nodes along the robot (towards the tip if positive).
        void stepSelection(int delta) const { m_selection.step(delta); }
        /// The selected spatial node and plot mode.
        const PlotSelection &selection() const { return m_selection; }

    private:
        /// One plotted state and the history of its nodes.
        struct Source
        {
            std::string name;
            std::shared_ptr<const StateType> state;
            NodeHistory history;
        };

        /// What the plots were last rendered for.
        struct RenderKey
        {
            std::vector<unsigned long> history_revisions;
            unsigned long selection_revision;
            std::vector<std::optional<std::size_t>> rows;
            QSize size;
            bool operator==(const RenderKey &other) const
            {
                return std::tie(history_revisions, selection_revision, rows) == std::tie(other.history_revisions, other.selection_revision, other.rows) &&
                       size == other.size;
            }
        };

        static std::shared_ptr<const ContinuumRobot<NodeT>> requireView(std::shared_ptr<const ContinuumRobot<NodeT>> view)
        {
            if (!view)
                throw std::invalid_argument("Default::PlotWidget requires a non-null robot view.");
            return view;
        }

        bool hasRobotView() const { return mp_robot_view != nullptr; }

        /// min(time, the last recorded snapshot's time), as the robot view displays.
        double displayTime(double time) const { return mp_last_snapshot ? std::min(time, mp_last_snapshot->time) : time; }

        /// The spatial node of `source` plotted against time: the selected one with a robot view, else its last.
        unsigned int plottedSpatial(const Source &source) const
        {
            if (hasRobotView())
                return m_selection.spatialIndex();
            return source.state->spatial_dim ? source.state->spatial_dim - 1 : 0;
        }

        const NodeHistory &robotHistory() const { return m_sources.front().history; }

        /// Half the arclength gap from the selected node to its nearest neighbour (infinite for a single node).
        double arclengthTolerance(unsigned int spatial) const
        {
            const auto &arclengths = m_sources.front().state->arclengths;
            double gap = std::numeric_limits<double>::infinity();
            if (spatial > 0)
                gap = std::min(gap, arclengths[spatial] - arclengths[spatial - 1]);
            if (spatial + 1 < arclengths.size())
                gap = std::min(gap, arclengths[spatial + 1] - arclengths[spatial]);
            return 0.5 * gap;
        }

        /// Half the time gap from `row` to its nearest cached neighbour, or the stale timeout if it has none.
        double timeTolerance(std::size_t row) const
        {
            const NodeHistory &history = robotHistory();
            const double row_time = *history.rowTime(row);
            double gap = std::numeric_limits<double>::infinity();
            if (row > 0)
                if (const auto t = history.rowTime(row - 1))
                    gap = std::min(gap, row_time - *t);
            if (const auto t = history.rowTime(row + 1))
                gap = std::min(gap, *t - row_time);
            return std::isfinite(gap) ? 0.5 * gap : stale_timeout_;
        }

        /// Re-renders every plot for `display_time`.
        void render(double display_time) const
        {
            const bool by_time = m_selection.mode() == PlotMode::TimeAtArclength;
            const bool with_robot = hasRobotView();
            mp_panel->setModeButtonText(by_time ? "Switch to: against arclength [m]" : "Switch to: against time [m]");

            std::optional<double> row_time;
            for (const auto &source : m_sources)
                if (const auto row = source.history.latestRowAtOrBefore(display_time))
                    row_time = std::max(row_time.value_or(*source.history.rowTime(*row)), *source.history.rowTime(*row));
            if (!row_time)
            {
                mp_panel->setFilterText("No estimate yet");
                mp_panel->setGroups({});
                return;
            }

            const std::vector<PlotGroup> groups = plotGroups(display_time);
            GroundTruthPoints ground_truth;
            const unsigned int spatial = m_selection.spatialIndex();
            if (!with_robot)
                mp_panel->setFilterText(QString("Against time, up to %1 s").arg(*row_time, 0, 'f', 4));
            else
            {
                const auto &arclengths = m_sources.front().state->arclengths;
                std::vector<std::vector<Types::MeasurementQueue::pMeasurementType>> measurements;
                if (by_time)
                {
                    mp_panel->setFilterText(QString("Node %1 at arclength %2 m").arg(spatial).arg(arclengths[spatial], 0, 'f', 4));
                    for (const auto &history : m_ground_truth)
                        measurements.push_back(history.measurementsBetween(-std::numeric_limits<double>::infinity(), *row_time));
                    ground_truth = poseGroundTruthAtArclength(measurements, arclengths[spatial], arclengthTolerance(spatial), *row_time);
                }
                else
                {
                    mp_panel->setFilterText(QString("At time %1 s").arg(*row_time, 0, 'f', 4));
                    const double tolerance = timeTolerance(*robotHistory().latestRowAtOrBefore(display_time));
                    for (const auto &history : m_ground_truth)
                        measurements.push_back(history.measurementsBetween(*row_time - tolerance, *row_time + tolerance));
                    ground_truth = poseGroundTruthAtTime(measurements, *row_time, tolerance);
                }
            }

            const QString x_name = by_time ? "Time" : "Arclength";
            std::vector<PlotPanel::GroupLayout> layout;
            for (const auto &group : groups)
                layout.push_back({group.supported ? QString("<b>%1</b> \xe2\x80\x94 vs %2").arg(QString::fromStdString(group.name), x_name)
                                                  : QString("<b>%1</b> \xe2\x80\x94 no plotter for this variable type").arg(QString::fromStdString(group.name)),
                                  static_cast<int>(group.series.size())});
            mp_panel->setGroups(layout);

            // Ground truth is a pose: overlay it on the first pose variable.
            std::size_t ground_truth_group = groups.size();
            for (std::size_t g = 0; g < groups.size() && ground_truth_group == groups.size(); g++)
                if (groups[g].type == typeid(gtsam::Pose3))
                    ground_truth_group = g;

            const QSize size = mp_panel->plotSize();
            const double marker = by_time || !with_robot ? std::numeric_limits<double>::quiet_NaN() : m_sources.front().state->arclengths[spatial];
            static const std::vector<double> none;
            for (std::size_t g = 0; g < groups.size(); g++)
                for (std::size_t c = 0; c < groups[g].series.size(); c++)
                {
                    const std::string &label = groups[g].labels[c];
                    const std::string title = label.rfind(groups[g].name, 0) == 0 ? label : groups[g].name + " " + label;
                    const bool with_ground_truth = g == ground_truth_group && c < ground_truth.components.size();
                    mp_panel->setPlot(g, c, PlotView::Content{groups[g].series[c], with_ground_truth ? ground_truth.x : none,
                                                              with_ground_truth ? ground_truth.components[c] : none,
                                                              QString::fromStdString(title), kColors[c % kColors.size()], size, marker});
                }
        }

        /// Places the highlight on the selected node's frame; hidden while the panel or that frame is (and absent without a robot view).
        void updateHighlight() const
        {
            if (!mp_highlight || !mp_node_frames)
                return;
            const auto &axes = *mp_node_frames;
            const unsigned int spatial = m_selection.spatialIndex();
            if (!visible_ || spatial >= axes.size() || !axes[spatial]->GetVisibility() || !axes[spatial]->GetUserMatrix())
            {
                mp_highlight->SetVisibility(false);
                return;
            }
            vtkMatrix4x4 *matrix = axes[spatial]->GetUserMatrix();
            mp_highlight->SetPosition(matrix->GetElement(0, 3), matrix->GetElement(1, 3), matrix->GetElement(2, 3));
            mp_highlight->SetVisibility(true);
        }

        /// Selects the node whose frame was clicked, if any (only while the panel is shown). Camera interaction is unaffected.
        void onLeftClick(vtkRenderWindowInteractor *interactor) const
        {
            if (!visible_ || !mp_renderer || !mp_node_frames)
                return;
            const int *position = interactor->GetEventPosition();
            if (!mp_picker->Pick(position[0], position[1], 0, mp_renderer))
                return;
            const vtkProp *picked = mp_picker->GetViewProp();
            const auto &axes = *mp_node_frames;
            for (unsigned int i = 0; i < axes.size(); i++)
                if (axes[i].GetPointer() == picked)
                {
                    m_selection.select(i);
                    updateHighlight();
                    return;
                }
        }

        /// Ground-truth measurements kept per sensor; above MeasurementHistory's default, as plots span the whole run.
        static constexpr std::size_t kGroundTruthCapacity = 100000;
        /// Component colours (as in the legacy plots), cycled per component.
        inline static const std::vector<QColor> kColors = {QColor(255, 0, 0), QColor(0, 160, 0), QColor(0, 0, 255),
                                                           QColor(255, 140, 0), QColor(128, 0, 128), QColor(255, 105, 180)};

        mutable std::vector<Source> m_sources;
        SnapshotSource m_snapshot_source;
        /// The robot view, kept alive for its node frames (null for named states).
        std::shared_ptr<const Visualizable> mp_robot_view;
        /// The robot view's node frames (see ContinuumRobot::mp_axes), which the highlight and click selection use; null without a robot view.
        const std::vector<vtkSmartPointer<vtkActor>> *mp_node_frames = nullptr;
        double m_highlight_scale = 0.015;
        std::vector<MeasurementHistory> m_ground_truth;
        mutable PlotSelection m_selection;
        mutable std::shared_ptr<const EstimatorSnapshot> mp_last_snapshot;
        mutable std::optional<RenderKey> m_last_render;

        /// Owned by the visualizer (its Qt child); created by visualize().
        mutable PlotPanel *mp_panel = nullptr;
        mutable vtkSmartPointer<vtkActor> mp_highlight;
        mutable vtkSmartPointer<vtkRenderer> mp_renderer;
        mutable vtkSmartPointer<vtkPropPicker> mp_picker;
        mutable vtkSmartPointer<vtkCallbackCommand> mp_click_callback;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_PLOTWIDGET_HPP
