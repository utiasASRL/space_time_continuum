#include <algorithm>
#include <cmath>

#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPen>

#include "PlotRendering.hpp"

namespace Spacetime::Visualization::Default
{
    namespace
    {
        /// Whether xs[i] and ys[i] are both finite.
        bool finitePair(const std::vector<double> &xs, const std::vector<double> &ys, std::size_t i)
        {
            return std::isfinite(xs[i]) && std::isfinite(ys[i]);
        }

        /// Whether `xs` and `ys` have the same, non-zero length.
        bool matched(const std::vector<double> &xs, const std::vector<double> &ys)
        {
            return !xs.empty() && xs.size() == ys.size();
        }

        /// `size` clamped to the smallest plot drawn.
        QSize clampedSize(QSize size) { return QSize(std::max(100, size.width()), std::max(60, size.height())); }

        /// Bold title font for a plot `height` pixels tall.
        QFont titleFont(int height)
        {
            QFont font;
            font.setBold(true);
            font.setPointSize(std::clamp(height / 25, 8, 12));
            return font;
        }
    } // namespace

    QRect plotArea(QSize size)
    {
        size = clampedSize(size);
        const int width = size.width(), height = size.height();
        const int left_margin = std::max(50, width / 8);
        const int margin = std::max(25, std::min(width, height) / 15);
        const int top_margin = std::max(margin, QFontMetrics(titleFont(height)).height() + 10);
        return QRect(left_margin, top_margin, width - left_margin - margin, height - top_margin - margin);
    }

    PlotRegion regionAt(QPoint position, QSize size)
    {
        const QRect area = plotArea(size);
        const bool in_x = position.x() >= area.left() && position.x() <= area.right();
        const bool in_y = position.y() >= area.top() && position.y() <= area.bottom();
        if (in_x && in_y)
            return PlotRegion::Plot;
        if (in_y && position.x() >= 0 && position.x() < area.left())
            return PlotRegion::YAxis;
        if (in_x && position.y() > area.bottom() && position.y() < clampedSize(size).height())
            return PlotRegion::XAxis;
        return PlotRegion::Outside;
    }

    QImage renderPlot(const PlotSeries &series, const std::vector<double> &gt_x, const std::vector<double> &gt_y,
                      const QString &title, const QColor &color, QSize size, double marker_x, const std::optional<PlotRange> &range)
    {
        size = clampedSize(size);
        const int width = size.width();
        const int height = size.height();
        QImage image(width, height, QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::white);
        QPainter painter(&image);
        painter.setRenderHint(QPainter::Antialiasing);

        const bool has_mean = matched(series.x, series.mean);
        const bool has_bounds = has_mean && series.lower.size() == series.x.size() && series.upper.size() == series.x.size();
        const bool has_gt = matched(gt_x, gt_y);

        const std::optional<PlotRange> fitted = fitRange(series, gt_x, gt_y);
        if (!fitted)
        {
            painter.drawText(image.rect(), Qt::AlignCenter, "No data available");
            return image;
        }
        const PlotRange shown = range ? *range : *fitted;
        const double x_min = shown.x_min, x_max = shown.x_max;
        const double y_min = shown.y_min, y_max = shown.y_max;
        const double y_range = y_max - y_min;

        // Common power-of-ten factor for very small or large values.
        const double max_abs = std::max(std::abs(y_min), std::abs(y_max));
        int exponent = 0;
        double scale = 1.0;
        if (max_abs > 0 && (max_abs < 0.001 || max_abs >= 10000))
        {
            exponent = static_cast<int>(std::floor(std::log10(max_abs)));
            scale = std::pow(10.0, -exponent);
        }

        const QFont title_font = titleFont(height);
        const int title_height = QFontMetrics(title_font).height();
        const QRect plot_rect = plotArea(size);
        const int left_margin = plot_rect.left();
        auto px = [&](double x)
        { return plot_rect.left() + (x - x_min) / (x_max - x_min) * plot_rect.width(); };
        auto py = [&](double y)
        { return plot_rect.bottom() - (y - y_min) / (y_max - y_min) * plot_rect.height(); };

        painter.setPen(QPen(Qt::black, 1));
        painter.drawRect(plot_rect);

        painter.setFont(title_font);
        painter.drawText(QRect(0, 4, width, title_height), Qt::AlignCenter, title);

        QFont label_font = painter.font();
        label_font.setBold(false);
        label_font.setPointSize(std::max(6, height / 30));
        painter.setFont(label_font);
        if (exponent != 0)
            painter.drawText(QRect(plot_rect.left() + 5, plot_rect.top() - 18, 80, 15), Qt::AlignLeft, QString("\xc3\x97" "10^%1").arg(exponent));

        // Grid with tick labels.
        const int x_grids = std::max(3, std::min(6, width / 120));
        for (int i = 0; i <= x_grids; i++)
        {
            const double x = x_min + (x_max - x_min) * i / x_grids;
            const int sx = static_cast<int>(px(x));
            painter.setPen(QPen(Qt::lightGray, 1));
            painter.drawLine(sx, plot_rect.top(), sx, plot_rect.bottom());
            painter.setPen(QPen(Qt::black, 1));
            painter.drawText(QRect(sx - 25, plot_rect.bottom() + 5, 50, 15), Qt::AlignCenter, QString::number(x, 'g', 3));
        }
        const double scaled_range = y_range * scale;
        const int decimals = scaled_range < 0.1 ? 3 : (scaled_range > 100 ? 1 : 2);
        const int y_grids = std::max(3, std::min(6, height / 60));
        for (int i = 0; i <= y_grids; i++)
        {
            const double y = y_min + (y_max - y_min) * i / y_grids;
            const int sy = static_cast<int>(py(y));
            painter.setPen(QPen(Qt::lightGray, 1));
            painter.drawLine(plot_rect.left(), sy, plot_rect.right(), sy);
            painter.setPen(QPen(Qt::black, 1));
            const double display = y * scale;
            const QString label = std::abs(display) < 1000 ? QString::number(display, 'f', decimals) : QString::number(display, 'e', 1);
            painter.drawText(QRect(5, sy - 8, left_margin - 10, 16), Qt::AlignRight | Qt::AlignVCenter, label);
        }

        if (std::isfinite(marker_x) && marker_x >= x_min && marker_x <= x_max)
        {
            QPen marker_pen(Qt::darkGray, 1);
            marker_pen.setStyle(Qt::DashLine);
            painter.setPen(marker_pen);
            painter.drawLine(QPointF(px(marker_x), plot_rect.top()), QPointF(px(marker_x), plot_rect.bottom()));
        }

        painter.setClipRect(plot_rect);

        // A polyline through the finite samples of `ys`, broken at non-finite ones.
        auto draw_line = [&](const std::vector<double> &ys)
        {
            bool has_previous = false;
            QPointF previous;
            for (std::size_t i = 0; i < series.x.size(); i++)
            {
                if (!finitePair(series.x, ys, i))
                {
                    has_previous = false;
                    continue;
                }
                const QPointF point(px(series.x[i]), py(ys[i]));
                if (has_previous)
                    painter.drawLine(previous, point);
                previous = point;
                has_previous = true;
            }
        };

        if (has_mean)
        {
            painter.setPen(QPen(color, 2));
            draw_line(series.mean);
            painter.setBrush(color);
            for (std::size_t i = 0; i < series.x.size(); i++)
                if (finitePair(series.x, series.mean, i))
                    painter.drawEllipse(QPointF(px(series.x[i]), py(series.mean[i])), 2.0, 2.0);
            painter.setBrush(Qt::NoBrush);
        }
        if (has_bounds)
        {
            QPen bound_pen(color, 1);
            bound_pen.setStyle(Qt::DashLine);
            painter.setPen(bound_pen);
            draw_line(series.lower);
            draw_line(series.upper);
        }
        if (has_gt)
        {
            painter.setPen(QPen(Qt::black, 1));
            painter.setBrush(Qt::black);
            for (std::size_t i = 0; i < gt_x.size(); i++)
                if (finitePair(gt_x, gt_y, i))
                    painter.drawEllipse(QPointF(px(gt_x[i]), py(gt_y[i])), 1.5, 1.5);
        }
        return image;
    }
} // namespace Spacetime::Visualization::Default
