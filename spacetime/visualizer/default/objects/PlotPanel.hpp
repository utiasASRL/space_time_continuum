#ifndef DEFAULT_PLOTPANEL_HPP
#define DEFAULT_PLOTPANEL_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <QFrame>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QString>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>

#include <spacetime/visualizer/PlotData.hpp>
#include <spacetime/visualizer/default/PlotRendering.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief One interactive plot: shows renderPlot() of its content and lets the user
     *        pan (left-drag) and zoom (wheel, anchored at the cursor). Where the cursor
     *        is decides the axes affected (see regionAt): on the y-axis only the vertical
     *        range changes, on the x-axis only the horizontal one, inside the axes both.
     *        Wheel events elsewhere (e.g. the title) are ignored, so they scroll the
     *        enclosing panel. The view persists across setContent() until resetView().
     */
    class PlotView : public QWidget
    {
    public:
        /// What is plotted (see renderPlot).
        struct Content
        {
            PlotSeries series;              ///< Mean and +/- 3 sigma bounds.
            std::vector<double> gt_x, gt_y; ///< Ground-truth samples (drawn as dots); may be empty.
            QString title;                  ///< Drawn above the axes.
            QColor color = Qt::black;       ///< Colour of the mean and bounds.
            QSize size;                     ///< Rendered size [px]; at least 100 x 60.
            double marker_x = std::numeric_limits<double>::quiet_NaN(); ///< x of the dashed vertical marker; NaN for none.
        };

        /// An empty plot (until setContent()).
        explicit PlotView(QWidget *parent = nullptr) : QWidget(parent) {}

        /// Replaces what is plotted, keeping the current view, and re-renders.
        void setContent(Content content)
        {
            m_content = std::move(content);
            setFixedSize(renderedSize());
            render();
        }

        /// Returns to the view fitted to the data.
        void resetView()
        {
            if (!m_view)
                return;
            m_view.reset();
            m_drag.reset();
            render();
        }

        /// The user's view, or nullopt while fitted to the data.
        const std::optional<PlotRange> &view() const { return m_view; }
        /// The last rendered plot.
        const QImage &image() const { return m_image; }

    protected:
        /// Draws the last rendered image.
        void paintEvent(QPaintEvent *) override
        {
            QPainter painter(this);
            painter.drawImage(0, 0, m_image);
        }

        /// One notch (120 eighths of a degree) scales the affected spans by kZoomPerNotch; wheel up zooms in.
        void wheelEvent(QWheelEvent *event) override
        {
            const QPoint position = event->position().toPoint();
            const PlotRegion region = regionAt(position, renderedSize());
            const std::optional<PlotRange> current = shownRange();
            if (region == PlotRegion::Outside || !current || event->angleDelta().y() == 0)
            {
                event->ignore();
                return;
            }
            const double scale = std::pow(kZoomPerNotch, event->angleDelta().y() / 120.0);
            const auto [anchor_x, anchor_y] = toData(*current, position);
            m_view = zoomRange(*current, anchor_x, anchor_y,
                               region == PlotRegion::YAxis ? 1.0 : scale,
                               region == PlotRegion::XAxis ? 1.0 : scale);
            render();
            event->accept();
        }

        /// A left press on the axes or an axis strip starts a pan (see mouseMoveEvent); other presses are ignored.
        void mousePressEvent(QMouseEvent *event) override
        {
            const QPoint position = event->position().toPoint();
            const PlotRegion region = regionAt(position, renderedSize());
            const std::optional<PlotRange> current = shownRange();
            if (event->button() != Qt::LeftButton || region == PlotRegion::Outside || !current)
            {
                event->ignore();
                return;
            }
            m_drag = Drag{position, *current, region};
            setCursor(Qt::ClosedHandCursor);
            event->accept();
        }

        /// The data under the cursor follows it; on an axis only that axis moves.
        void mouseMoveEvent(QMouseEvent *event) override
        {
            if (!m_drag)
            {
                event->ignore();
                return;
            }
            const QRect area = plotArea(renderedSize());
            const QPoint delta = event->position().toPoint() - m_drag->start;
            const PlotRange &start = m_drag->start_range;
            const double dx = m_drag->region == PlotRegion::YAxis ? 0.0 : -delta.x() * (start.x_max - start.x_min) / area.width();
            const double dy = m_drag->region == PlotRegion::XAxis ? 0.0 : delta.y() * (start.y_max - start.y_min) / area.height();
            m_view = panRange(start, dx, dy);
            render();
            event->accept();
        }

        /// Ends a pan, keeping the panned view.
        void mouseReleaseEvent(QMouseEvent *event) override
        {
            if (!m_drag || event->button() != Qt::LeftButton)
            {
                event->ignore();
                return;
            }
            m_drag.reset();
            unsetCursor();
            event->accept();
        }

    private:
        /// An in-progress drag: where it started and the range shown then.
        struct Drag
        {
            QPoint start;          ///< Press position [px].
            PlotRange start_range; ///< Range shown at the press.
            PlotRegion region;     ///< Region pressed, which decides the axes panned.
        };

        /// Span scale per wheel notch (< 1 zooms in on wheel up).
        static constexpr double kZoomPerNotch = 0.8;

        /// Content::size clamped as renderPlot clamps it.
        QSize renderedSize() const { return m_content.size.expandedTo(QSize(100, 60)); }

        /// The range currently drawn: the user's view, or the fit (nullopt without data).
        std::optional<PlotRange> shownRange() const { return m_view ? m_view : fitRange(m_content.series, m_content.gt_x, m_content.gt_y); }

        /// Data coordinates of pixel `position` under `range` (the inverse of renderPlot's mapping).
        std::pair<double, double> toData(const PlotRange &range, QPoint position) const
        {
            const QRect area = plotArea(renderedSize());
            const double x = range.x_min + (position.x() - area.left()) * (range.x_max - range.x_min) / area.width();
            const double y = range.y_min + (area.bottom() - position.y()) * (range.y_max - range.y_min) / area.height();
            return {x, y};
        }

        /// Re-renders m_image for the current content and view, and schedules a repaint.
        void render()
        {
            m_image = renderPlot(m_content.series, m_content.gt_x, m_content.gt_y, m_content.title, m_content.color,
                                 m_content.size, m_content.marker_x, m_view);
            update();
        }

        Content m_content;
        QImage m_image;
        /// The user's panned/zoomed range, or nullopt to fit the data.
        std::optional<PlotRange> m_view;
        /// The pan in progress, if any.
        std::optional<Drag> m_drag;
    };

    /**
     * @brief The scrollable overlay a PlotWidget draws into: a mode button, a line
     *        saying where the plots are sliced, then per plotted variable a header and
     *        one PlotView per component. Holds no data of its own: setGroups() lays it out
     *        and setPlot() fills it. Child widgets are owned by Qt. GUI thread only.
     */
    class PlotPanel : public QScrollArea
    {
    public:
        /// Header and plot count of one variable group. `count` is 0 for a group with nothing to plot.
        struct GroupLayout
        {
            QString header; ///< Header text (rich text allowed).
            int count = 0;  ///< Number of component plots.
        };

        /// Creates the (hidden) panel; `on_mode_clicked` runs when the mode button is pressed.
        PlotPanel(QWidget *parent, std::function<void()> on_mode_clicked) : QScrollArea(parent)
        {
            setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
            setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
            setWidgetResizable(true);
            setFrameShape(QFrame::StyledPanel);

            auto *container = new QWidget();
            m_layout = new QVBoxLayout(container);
            m_layout->setContentsMargins(kPadding / 2, kPadding / 2, kPadding / 2, kPadding / 2);

            m_mode_button = new QPushButton();
            QObject::connect(m_mode_button, &QPushButton::clicked, [fn = std::move(on_mode_clicked)]()
                             { if (fn) fn(); });
            m_layout->addWidget(m_mode_button);

            m_filter_label = new QLabel();
            m_filter_label->setAlignment(Qt::AlignCenter);
            m_filter_label->setStyleSheet("font-style: italic; color: #444;");
            m_layout->addWidget(m_filter_label);

            m_groups_container = new QWidget();
            m_groups_layout = new QVBoxLayout(m_groups_container);
            m_groups_layout->setContentsMargins(0, 0, 0, 0);
            m_layout->addWidget(m_groups_container);
            m_layout->addStretch();

            setWidget(container);
            setVisible(false);
        }

        /// Sets the mode button's label.
        void setModeButtonText(const QString &text) { m_mode_button->setText(text); }
        /// Shows or hides the mode button (e.g. for plots with a single mode).
        void setModeButtonVisible(bool visible) { m_mode_button->setVisible(visible); }
        /// Sets the line under the mode button saying where the plots are sliced.
        void setFilterText(const QString &text) { m_filter_label->setText(text); }

        /// Lays the panel out for `groups`, rebuilding the plots only if their counts changed (headers are always updated).
        void setGroups(const std::vector<GroupLayout> &groups)
        {
            bool same = groups.size() == m_headers.size();
            for (std::size_t g = 0; same && g < groups.size(); g++)
                same = static_cast<std::size_t>(groups[g].count) == m_plots[g].size();
            if (!same)
                rebuild(groups);
            for (std::size_t g = 0; g < groups.size(); g++)
                m_headers[g]->setText(groups[g].header);
        }

        /// Shows `content` as plot `component` of group `group`, keeping that plot's view (ignored if out of range).
        void setPlot(std::size_t group, std::size_t component, PlotView::Content content)
        {
            if (PlotView *view = plot(group, component))
                view->setContent(std::move(content));
        }

        /// Plot `component` of group `group`, or nullptr if out of range.
        PlotView *plot(std::size_t group, std::size_t component) const
        {
            if (group >= m_plots.size() || component >= m_plots[group].size())
                return nullptr;
            return m_plots[group][component];
        }

        /// Returns every plot to the view fitted to its data.
        void resetViews()
        {
            for (const auto &plots : m_plots)
                for (PlotView *view : plots)
                    view->resetView();
        }

        /// Size each plot is rendered at: the viewport width minus padding, about a third as tall.
        QSize plotSize() const
        {
            const int width = std::max(300, viewport()->width() - kPadding);
            return QSize(width, std::clamp(width / 3, 150, 300));
        }

    protected:
        /// Scrolls (for wheel events a plot did not use to zoom), then consumes the event so it never reaches the VTK view beneath (where it would zoom the camera).
        void wheelEvent(QWheelEvent *event) override
        {
            QScrollArea::wheelEvent(event);
            event->accept();
        }

        /// Consumes clicks and drags not used by a plot, for the same reason (they would rotate the camera).
        void mousePressEvent(QMouseEvent *event) override { event->accept(); }
        /// See mousePressEvent.
        void mouseMoveEvent(QMouseEvent *event) override { event->accept(); }
        /// See mousePressEvent.
        void mouseReleaseEvent(QMouseEvent *event) override { event->accept(); }

    private:
        /// Replaces every header and plot label with ones for `groups`.
        void rebuild(const std::vector<GroupLayout> &groups)
        {
            while (QLayoutItem *item = m_groups_layout->takeAt(0))
            {
                delete item->widget();
                delete item;
            }
            m_headers.clear();
            m_plots.clear();
            for (const auto &group : groups)
            {
                auto *header = new QLabel();
                header->setAlignment(Qt::AlignCenter);
                header->setMinimumHeight(kHeaderHeight);
                m_groups_layout->addWidget(header);
                m_headers.push_back(header);

                std::vector<PlotView *> plots;
                for (int c = 0; c < group.count; c++)
                {
                    auto *plot = new PlotView();
                    m_groups_layout->addWidget(plot, 0, Qt::AlignHCenter);
                    plots.push_back(plot);
                }
                m_plots.push_back(std::move(plots));
            }
        }

        /// Margin around the plots [px].
        static constexpr int kPadding = 20;
        /// Minimum height of a group header [px].
        static constexpr int kHeaderHeight = 30;

        QVBoxLayout *m_layout;
        QPushButton *m_mode_button;
        QLabel *m_filter_label;
        QWidget *m_groups_container;
        QVBoxLayout *m_groups_layout;
        /// One header per group.
        std::vector<QLabel *> m_headers;
        /// Per group, one plot per component.
        std::vector<std::vector<PlotView *>> m_plots;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_PLOTPANEL_HPP
