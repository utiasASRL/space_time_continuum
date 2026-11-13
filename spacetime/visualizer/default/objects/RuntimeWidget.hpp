#ifndef DEFAULT_RUNTIMEWIDGET_HPP
#define DEFAULT_RUNTIMEWIDGET_HPP

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <QLabel>
#include <QRect>
#include <QSize>
#include <QString>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/RuntimeStatistics.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>

namespace Spacetime::Visualization::Default
{
    namespace RuntimeFormatting
    {
        /// `seconds` in milliseconds, to one decimal place.
        inline std::string milliseconds(double seconds)
        {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.1f ms", seconds * 1e3);
            return buffer;
        }

        /// What each field shows (see formatRuntime).
        enum class Mode
        {
            Online,  ///< The last duration and running average.
            Offline, ///< The average and total.
            Batch,   ///< The total only.
        };

        /// One quantity's last, average and total durations, formatted for `mode`.
        inline std::string value(double last, double average, double total, Mode mode)
        {
            switch (mode)
            {
            case Mode::Online:
                return milliseconds(last) + " (avg " + milliseconds(average) + ")";
            case Mode::Offline:
                return "avg " + milliseconds(average) + ", total " + milliseconds(total);
            case Mode::Batch:
            default:
                return milliseconds(total);
            }
        }

        /// One "<name>: <value>" field, formatted for `mode`.
        inline std::string field(const char *name, const RuntimeStatistics &stats, RuntimePhase phase, double PhaseTimings::*member, Mode mode)
        {
            std::string text = std::string(name) + ": ";
            if (stats.count(phase) == 0)
                return text + "--";
            return text + value(stats.last().*member, stats.average().*member, stats.total().*member, mode);
        }

        /// The summed duration of all phases.
        inline double sum(const PhaseTimings &timings) { return timings.graph_construction + timings.solve + timings.covariance; }

        /// The "Total: <value>" field, summed over the phases, formatted as in field().
        inline std::string total(const RuntimeStatistics &stats, Mode mode)
        {
            return "Total: " + value(sum(stats.last()), sum(stats.average()), sum(stats.total()), mode);
        }
    } // namespace RuntimeFormatting

    /**
     * @brief The runtime bar's text for `snapshot`'s phase timings (see RuntimeStatistics),
     *        in ms to one decimal place. Online: each phase's last duration and running
     *        average; offline: its average and total; for a batch estimator (whose graph is
     *        built once per estimate() but solved once by finalize(), so the per-phase
     *        averages are not comparable): its total only. A phase not timed yet shows "--",
     *        and covariance extraction shows "off" when the estimator does not extract it.
     */
    inline std::string formatRuntime(const EstimatorSnapshot *snapshot, bool online)
    {
        using RuntimeFormatting::field;
        using RuntimeFormatting::Mode;
        if (!snapshot)
            return "Graph Construction: --   |   Solve: --   |   Covariance Extraction: --   |   Total: --";
        const Mode mode = snapshot->batch ? Mode::Batch : (online ? Mode::Online : Mode::Offline);
        const RuntimeStatistics &stats = snapshot->runtime;
        return field("Graph Construction", stats, RuntimePhase::GraphConstruction, &PhaseTimings::graph_construction, mode) + "   |   " +
               field("Solve", stats, RuntimePhase::Solve, &PhaseTimings::solve, mode) + "   |   " +
               (snapshot->covariance_enabled ? field("Covariance Extraction", stats, RuntimePhase::Covariance, &PhaseTimings::covariance, mode)
                                             : std::string("Covariance Extraction: off")) + "   |   " +
               RuntimeFormatting::total(stats, mode);
    }

    /**
     * @brief A bar at the bottom of the window showing the estimator's phase timings
     *        (see formatRuntime), read from the latest published snapshot. Online it
     *        tracks every estimate; offline the estimator has finished before playback,
     *        so it shows the run's constant average and total (batch: total only).
     *        Hidden by default: H toggles it together with the diagnostics panel.
     */
    class RuntimeWidget : public Visualizable
    {
    public:
        /// Object name of the bar's label (e.g. to find it among the visualizer's children).
        static constexpr const char *kObjectName = "runtime_bar";
        /// Height of the bar, in pixels (see Visualizer::bottomBarHeight).
        static constexpr int kHeight = Visualizer::bottomBarHeight();

        using SnapshotSource = std::function<std::shared_ptr<const EstimatorSnapshot>()>;

        /**
         * @param source Returns the estimator's latest snapshot (e.g. Estimator::getSnapshot), or nullptr before the first.
         * @throws std::invalid_argument if `source` is empty.
         */
        explicit RuntimeWidget(SnapshotSource source) : Visualizable(true), m_source(std::move(source))
        {
            if (!m_source)
                throw std::invalid_argument("RuntimeWidget: the snapshot source is empty.");
        }

        /// Short name if `brief`, otherwise a one-line description.
        std::string info(bool brief = true) const override { return brief ? "RuntimeWidget" : "RuntimeWidget: estimator phase timings"; }

        /// Creates the bar along the bottom of the window, shown only while the diagnostics panel is, and binds H to toggle it.
        void visualize(Visualizer &visualizer) const override
        {
            m_online = visualizer.isOnline();
            mp_label = new QLabel(&visualizer);
            mp_label->setObjectName(kObjectName);
            mp_label->setAlignment(Qt::AlignCenter);
            mp_label->setStyleSheet("QLabel { background-color : white; color : black; font-size: 13px; }");
            visualizer.addOverlayWidget(mp_label, [](QSize window)
                                        { return QRect(150, window.height() - kHeight, std::max(0, window.width() - 300), kHeight); });
            setText(formatRuntime(nullptr, m_online));
            visible_ = visualizer.diagnosticsVisible(); // H toggles both, so start in sync
            mp_label->setVisible(visible_);

            visualizer.bindShortcut(Qt::Key_H, [this]()
                                    { toggleVisibility(); });
            visualizer.addControlInfo("Toggle runtime [h]");
        }

        /// Shows or hides the bar.
        void toggleVisibility() const override
        {
            visible_ = !visible_;
            if (mp_label)
                mp_label->setVisible(visible_);
        }

        /// Shows the latest snapshot's timings, if it changed.
        void updateVisualization(double, double) const override
        {
            if (!mp_label)
                return; // not visualized yet
            auto snapshot = m_source();
            if (snapshot == mp_last_snapshot)
                return;
            mp_last_snapshot = snapshot;
            setText(formatRuntime(snapshot.get(), m_online));
        }

    private:
        void setText(const std::string &text) const { mp_label->setText(QString::fromStdString(text)); }

        SnapshotSource m_source;
        mutable QLabel *mp_label = nullptr; // owned by the visualizer
        mutable std::shared_ptr<const EstimatorSnapshot> mp_last_snapshot;
        mutable bool m_online = false;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_RUNTIMEWIDGET_HPP
