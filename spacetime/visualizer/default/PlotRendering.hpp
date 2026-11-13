#ifndef DEFAULT_PLOTRENDERING_HPP
#define DEFAULT_PLOTRENDERING_HPP

#include <limits>
#include <optional>
#include <vector>

#include <QColor>
#include <QImage>
#include <QPoint>
#include <QRect>
#include <QSize>
#include <QString>

#include <spacetime/visualizer/PlotData.hpp>

namespace Spacetime::Visualization::Default
{
    /// Where a point on a rendered plot lies: inside the axes, on the x-axis strip below them, on the y-axis strip left of them, or elsewhere (title, corners).
    enum class PlotRegion
    {
        Outside,
        Plot,
        XAxis,
        YAxis
    };

    /// The axes rectangle renderPlot() draws into for an image of `size` (after the same clamping).
    QRect plotArea(QSize size);

    /// The region of a plot of `size` that `position` (in image pixels) falls in.
    PlotRegion regionAt(QPoint position, QSize size);

    /**
     * @brief Draws one plot: `series`' mean as a solid line with a marker per sample,
     *        its finite +/- 3 sigma bounds as dashed lines, and ground truth
     *        (`gt_x`, `gt_y`) as black dots. Non-finite samples break the lines instead
     *        of being drawn. Shows "No data available" if nothing is finite. Drawn into a
     *        QImage (needs a QGuiApplication for text).
     * @param size     Image size; clamped to at least 100 x 60.
     * @param marker_x If finite and within the x range, a dashed vertical marker is drawn there.
     * @param range    Data window to show; fitted to the data (see fitRange) if nullopt.
     */
    QImage renderPlot(const PlotSeries &series, const std::vector<double> &gt_x, const std::vector<double> &gt_y,
                      const QString &title, const QColor &color, QSize size,
                      double marker_x = std::numeric_limits<double>::quiet_NaN(),
                      const std::optional<PlotRange> &range = std::nullopt);
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_PLOTRENDERING_HPP
