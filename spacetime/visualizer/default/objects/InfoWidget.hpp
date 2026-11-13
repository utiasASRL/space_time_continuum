#pragma once

#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include <type_traits>
#include <utility>

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
#include <QWheelEvent>
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
#include <QFrame>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief A translucent, scrollable overlay panel: a QLabel (holding the actual
     *        rich-text content, sized to fit it exactly) inside a QScrollArea (whose
     *        own geometry the owner controls and keeps window-bounded). Content
     *        that overflows the panel's fixed viewport scrolls instead of being
     *        cropped or forcing the panel to keep growing.
     */
    class InfoWidget : public QScrollArea
    {
        Q_OBJECT

    private:
        /// One panel row: a bold header, or a label with an optional value.
        struct DisplayItem
        {
            enum Type
            {
                VALUE,
                HEADER
            };
            Type type;
            std::string label;
            void *value;
            std::function<std::string()> stringifier; // Function to convert value to string

            /// A value row: label `l`, optional pointer `v` (only checked for null), and stringifier `s` (may be empty for a label-only row).
            DisplayItem(const std::string &l, void *v, std::function<std::string()> s)
                : type(VALUE), label(l), value(v), stringifier(s) {}

            /// A header row.
            DisplayItem(const std::string &headerText)
                : type(HEADER), label(headerText), value(nullptr) {}
        };

        std::vector<DisplayItem> m_displayItems;
        QLabel *m_content;
        /// The text last shown, so an unchanged panel is not re-laid out (see updateDisplay).
        QString m_displayedText;

        /// Re-renders every row as rich text (an exception from a stringifier shows "Error displaying value") and, if it changed, resizes the content to fit.
        void updateDisplay()
        {
            QString displayText;
            for (const auto &item : m_displayItems)
            {
                if (item.type == DisplayItem::HEADER)
                {
                    if (!displayText.isEmpty())
                        displayText += "<br><br>";
                    displayText += "<b>" + QString::fromStdString(item.label) + "</b>";
                }
                else // VALUE
                {
                    if (!displayText.isEmpty())
                        displayText += "<br>";

                    displayText += QString::fromStdString(item.label);
                    if (item.value || item.stringifier)
                    {
                        displayText += ": ";

                        try
                        {
                            displayText += QString::fromStdString(item.stringifier());
                        }
                        catch (...)
                        {
                            displayText += "Error displaying value";
                        }
                    }
                }
            }
            if (displayText == m_displayedText)
                return; // refreshed every frame: skip the rich-text re-layout when nothing changed
            m_displayedText = displayText;
            m_content->setText(displayText);
            m_content->adjustSize(); // grows the content to fit; the surrounding QScrollArea clips/scrolls to it rather than the other way around
        }

        /// SFINAE trait: whether `std::ostream << T` is well-formed.
        template <typename T>
        struct is_streamable
        {
        private:
            /// Selected when `std::ostream << U` is well-formed.
            template <typename U>
            static auto test(int) -> decltype(std::declval<std::ostream &>() << std::declval<U>(), std::true_type{});

            /// Fallback overload.
            template <typename>
            static std::false_type test(...);

        public:
            static constexpr bool value = decltype(test<T>(0))::value;
        };

    public:
        /// Creates the (initially hidden) panel with scrollbars as needed; the owner sets its geometry.
        InfoWidget(QWidget *parent = nullptr) : QScrollArea(parent)
        {
            this->setFrameShape(QFrame::NoFrame);
            this->setStyleSheet("QScrollArea { background: transparent; border: none; }");
            this->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
            this->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
            this->setWidgetResizable(false); // let m_content size to its own content; the scroll area's viewport (not the content) is what stays window-bounded

            m_content = new QLabel(this);
            m_content->setStyleSheet(
                "QLabel {"
                "background-color: rgba(0, 0, 0, 150);"
                "color: white;"
                "padding: 5px;"
                "border-radius: 5px;"
                "line-height: 1.2;"
                "}");
            m_content->setAlignment(Qt::AlignTop | Qt::AlignLeft);
            m_content->setTextFormat(Qt::RichText); // Enable HTML formatting
            this->setWidget(m_content);
            this->setVisible(false);
        }

        /// Appends a bold header row.
        void addHeader(const std::string &label)
        {
            m_displayItems.emplace_back(label);
            updateDisplay();
        }

        /// Appends a `label: value` row whose value is re-evaluated by `valueFunction` on every refresh().
        void addLabel(const std::string &label, std::function<std::string()> valueFunction)
        {
            m_displayItems.emplace_back(label, nullptr, valueFunction);
            updateDisplay();
        }

        /// Appends a `label: *value` row, streamed on every refresh() ("null" if `value` is null). `value` must outlive the row.
        template <typename T>
        typename std::enable_if<is_streamable<T>::value, void>::type
        addLabel(const std::string &label, T *value)
        {
            // Create a lambda that captures the pointer and converts to string
            auto stringifier = [value]() -> std::string
            {
                if (value == nullptr)
                {
                    return "null";
                }

                std::ostringstream oss;
                oss << *value;
                return oss.str();
            };

            m_displayItems.emplace_back(label, static_cast<void *>(value), stringifier);
            updateDisplay();
        }

        /// Appends a label-only row (e.g. a control description).
        void addLabel(const std::string &label)
        {
            m_displayItems.emplace_back(label, nullptr, nullptr);
            updateDisplay();
        }

        /// Overload for non-streamable types: fails to compile with a descriptive static_assert.
        template <typename T>
        typename std::enable_if<!is_streamable<T>::value, void>::type
        addLabel(const std::string &label, T *value)
        {
            static_assert(is_streamable<T>::value,
                          "Type T must be streamable (support std::cout << value). "
                          "Please define operator<< for your type or use a streamable type.");
        }

        /// Removes every value row labelled `label`.
        void removeDiagnosticValue(const std::string &label)
        {
            m_displayItems.erase(
                std::remove_if(m_displayItems.begin(), m_displayItems.end(),
                               [&label](const DisplayItem &item)
                               {
                                   return item.type == DisplayItem::VALUE && item.label == label;
                               }),
                m_displayItems.end());
            updateDisplay();
        }

        /// Removes every header row labelled `label`.
        void removeHeader(const std::string &label)
        {
            m_displayItems.erase(
                std::remove_if(m_displayItems.begin(), m_displayItems.end(),
                               [&label](const DisplayItem &item)
                               {
                                   return item.type == DisplayItem::HEADER && item.label == label;
                               }),
                m_displayItems.end());
            updateDisplay();
        }

        /// Removes every row, headers included.
        void clearAll()
        {
            m_displayItems.clear();
            m_displayedText.clear();
            m_content->setText("");
            m_content->adjustSize();
        }

        /// Removes every value row, keeping headers.
        void clearDiagnosticValues()
        {
            m_displayItems.erase(
                std::remove_if(m_displayItems.begin(), m_displayItems.end(),
                               [](const DisplayItem &item)
                               {
                                   return item.type == DisplayItem::VALUE;
                               }),
                m_displayItems.end());
            updateDisplay();
        }

        /// Re-evaluates every value and redraws; call when values change (Visualizer does so every frame).
        void refresh()
        {
            updateDisplay();
        }

    protected:
        /// Scrolls, then consumes the wheel event even at the scroll bounds -- otherwise it
        /// propagates to the VTK render window beneath, which interprets it as a camera zoom.
        void wheelEvent(QWheelEvent *event) override
        {
            QScrollArea::wheelEvent(event);
            event->accept();
        }
    };
} // namespace Spacetime::Visualization::Default
