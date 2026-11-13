#pragma once

#include <cmath>

#include <QObject>
#include <QVTKOpenGLNativeWidget.h>
#include <QTimer>
#include <QEvent>
#include <QPaintEvent>
#include <QDomDocument>
#include <QDomElement>
#include <QDomNodeList>
#include <QDomNode>
#include <QMouseEvent>
#include <QDebug>
#include <QString>
#include <QCoreApplication>
#include <QDateTime>
#include <QLabel>
#include <QSlider>
#include <QPainter>
#include <QPixmap>
#include <QVBoxLayout>
#include <QScrollArea>

#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief Timeline overlay: a thin QSlider marking the current time, with
     *        labelled major/minor tick marks underneath and an optional highlighted
     *        window (e.g. the estimator's fixed-lag window) behind the slider.
     */
    class TimelineWidget : public QWidget
    {
        Q_OBJECT

        const static int _element_padding = 2;
        const static int _font_px_size = 15;
        const static int _tick_height = 10;
        const static int _slider_height = 10;
        const static int _handle_width = 1;
        const static int _x_padding = _font_px_size;

        int m_y_padding = 0;
        double m_start_time = 0.0;
        double m_end_time = 10.0;
        double m_window_size = -1.0;

        /// Draws the ticks and labels below the slider; the major interval is the smallest preset giving <= 20 major ticks, with 10 minor ticks each.
        void paintTimeline()
        {
            QFont font = QFont("Arial");
            font.setPixelSize(_font_px_size);

            QPainter painter(this);
            painter.setPen(QPen(Qt::black, 1));
            painter.setFont(font);

            double duration = m_end_time - m_start_time;
        
            // Determine appropriate tick spacing to limit major ticks to ~20 or fewer
            double major_tick_interval = 1.0; // Start with 1 second
            double intervals[] = {1.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0}; // 1s, 2s, 4s, 10s, 20s, 30s, 1m, 2m, 5m, 10m, 20m, 30m, 1h
        
            for (double interval : intervals) {
                if (duration / interval <= 20.0) {
                    major_tick_interval = interval;
                    break;
                }
                major_tick_interval = interval;
            }
        
            // Minor tick interval is 1/10th of major tick interval
            double minor_tick_interval = major_tick_interval / 10.0;
        
            // Calculate tick positions
            int start_major_tick = (int)std::ceil(m_start_time / major_tick_interval);
            int end_major_tick = (int)std::floor(m_end_time / major_tick_interval);
        
            int start_minor_tick = (int)std::ceil(m_start_time / minor_tick_interval);
            int end_minor_tick = (int)std::floor(m_end_time / minor_tick_interval);

            // Draw minor ticks
            for (int i = start_minor_tick; i <= end_minor_tick; i++)
            {
                double tick_time = i * minor_tick_interval;
                if (tick_time < m_start_time || tick_time > m_end_time)
                    continue;
                
                int x = std::round((double)_x_padding + ((tick_time - m_start_time) / duration) * (slider->width() - _handle_width));
                int y = m_y_padding + _slider_height + _element_padding;
            
                // Skip if this is a major tick position
                if (std::abs(std::fmod(tick_time, major_tick_interval)) < minor_tick_interval * 0.1)
                    continue;
                
                painter.drawLine(x, y, x, y + _tick_height - 4); // minor ticks
            }

            // Draw major ticks and labels
            for (int i = start_major_tick; i <= end_major_tick; i++)
            {
                double tick_time = i * major_tick_interval;
                if (tick_time < m_start_time || tick_time > m_end_time)
                    continue;
                
                int x = std::round((double)_x_padding + ((tick_time - m_start_time) / duration) * (slider->width() - _handle_width));
                int y = m_y_padding + _slider_height + _element_padding;
            
                painter.drawLine(x, y, x, y + _tick_height); // major ticks

                // Draw label - format based on interval size
                QString label;
                if (major_tick_interval >= 60.0) {
                    // For intervals >= 1 minute, show minutes:seconds or hours:minutes
                    int total_seconds = (int)tick_time;
                    if (major_tick_interval >= 3600.0) {
                        // Hours:minutes format
                        int hours = total_seconds / 3600;
                        int minutes = (total_seconds % 3600) / 60;
                        label = QString("%1:%2").arg(hours).arg(minutes, 2, 10, QChar('0'));
                    } else {
                        // Minutes:seconds format
                        int minutes = total_seconds / 60;
                        int seconds = total_seconds % 60;
                        label = QString("%1:%2").arg(minutes).arg(seconds, 2, 10, QChar('0'));
                    }
                } else {
                    // For intervals < 1 minute, show seconds
                    label = QString::number(tick_time, 'f', (major_tick_interval < 1.0) ? 1 : 0);
                }
            
                QRect textRect(x - _font_px_size, y + _tick_height + _element_padding, 2 * _font_px_size, _font_px_size);
                painter.drawText(textRect, Qt::AlignCenter, label);
            }
        };

        /// Fills the highlighted window starting at the slider position, or the whole slider if no window is set.
        void paintWindow()
        {
            double px_per_second = (slider->width() - _handle_width) / (m_end_time - m_start_time);
            int window_pixels = (m_window_size < 0) ? slider->width() : m_window_size * px_per_second;

            int current_x = (m_window_size < 0) ? _x_padding : slider->value() * (slider->width() - _handle_width) / (slider->maximum() - slider->minimum()) + _x_padding;
            int current_y = m_y_padding;

            QRect windowRect(current_x, current_y, window_pixels, _slider_height);

            QPainter painter(this);
            painter.setPen(QPen(Qt::black, 0));
            painter.fillRect(windowRect, QColor(255, 120, 120));
        };

        /// Repaints the slider, then the ticks and the window highlight.
        void paintEvent(QPaintEvent *event) override
        {
            slider->update(event->region());
            paintTimeline();
            paintWindow();
        }

    public:
        /// Child slider whose value marks the current time.
        QSlider *slider;

        /// Creates the slider on a transparent background. Call setRange() and setGeometry() before use.
        TimelineWidget(QWidget *parent = nullptr) : QWidget(parent)
        {
            slider = new QSlider(Qt::Horizontal, this);

            this->setStyleSheet("background-color: transparent;");
        };

        /// Moves the slider to `value` (in the units passed to setRange as min/max).
        void setValue(int value)
        {
            slider->setValue(value);
        };
        /**
         * @brief Sets the slider's integer range and the time span it represents.
         * @param min, max     Slider value range (mapped linearly to [start_time, end_time]).
         * @param window_size  Width [s] of the highlighted window; negative, or not
         *                     shorter than the span, highlights everything up to the slider.
         */
        void setRange(int min, int max, double start_time, double end_time, double window_size = -1)
        {
            slider->setMinimum(min);
            slider->setMaximum(max);
            m_start_time = start_time;
            m_end_time = end_time;
            m_window_size = (window_size < m_end_time - m_start_time) ? window_size : -1.0;

            QString color = (m_window_size < 0) ? QString("rgb(255, 120, 120)") : QString("rgb(255, 201, 201)");
            slider->setStyleSheet(
                "QSlider::groove:horizontal {"
                "height: " +
                QString::number(_slider_height) + "px;"
                                                  "border: 0px;"
                                                  "background: transparent;"
                                                  "}"
                                                  "QSlider::handle:horizontal {"
                                                  "background: rgb(0, 0, 0);"
                                                  "width: " +
                QString::number(_handle_width) + "px;"
                                                 "}"
                                                 "QSlider::sub-page:horizontal {"
                                                 "background: " +
                color + ";"
                        "}");
        };
        /// Shadows QWidget::setGeometry to also vertically centre the slider and ticks; warns if `h` is too small to fit them.
        void setGeometry(int x, int y, int w, int h)
        {
            QWidget::setGeometry(x, y, w, h);
            m_y_padding = (h - _slider_height - _font_px_size - _tick_height - 2 * _element_padding) / 2;
            if (m_y_padding < 0)
                warning() << "TimelineWidget: Not enough height to display timeline properly. Minimum: " << (_slider_height + _font_px_size + _tick_height + 2 * _element_padding) << "px required.";
            slider->setGeometry(_x_padding, m_y_padding, this->width() - 2 * _x_padding, _slider_height);
        };
    };
} // namespace Spacetime::Visualization::Default
