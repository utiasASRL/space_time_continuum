#include <cmath>
#include <limits>
#include <memory>

#include <QColor>
#include <QApplication>
#include <QCoreApplication>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QImage>
#include <QLabel>
#include <QSize>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/bias_models/Bias.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/visualizer/default/PlotRendering.hpp>
#include <spacetime/visualizer/default/objects/InfoWidget.hpp>
#include <spacetime/visualizer/default/objects/PlotPanel.hpp>
#include <spacetime/visualizer/default/objects/PlotWidget.hpp>

namespace Spacetime
{
    using namespace Visualization;
    using namespace Visualization::Default;

    static_assert(IsVisualizable<PlotWidget<Priors::WNOA::Node<gtsam::Pose3>>>);
    static_assert(IsVisualizableWithToggle<PlotWidget<Priors::WNOA::Node<gtsam::Pose3>>>);
    static_assert(IsVisualizable<PlotWidget<Priors::WNOV::Node<gtsam::Pose3>>>);
    static_assert(IsVisualizable<PlotWidget<Biases::BiasNode<gtsam::Vector3>>>);

    namespace
    {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        const QSize kSize(400, 200);

        /// Whether any pixel of `image` differs from white.
        bool hasInk(const QImage &image)
        {
            for (int y = 0; y < image.height(); y++)
                for (int x = 0; x < image.width(); x++)
                    if (image.pixelColor(x, y) != QColor(Qt::white))
                        return true;
            return false;
        }

        /// Whether `image` contains a pixel close to `color`.
        bool hasColor(const QImage &image, const QColor &color)
        {
            for (int y = 0; y < image.height(); y++)
                for (int x = 0; x < image.width(); x++)
                {
                    const QColor c = image.pixelColor(x, y);
                    if (std::abs(c.red() - color.red()) < 30 && std::abs(c.green() - color.green()) < 30 && std::abs(c.blue() - color.blue()) < 30)
                        return true;
                }
            return false;
        }

        /**
         * @brief Owns the one offscreen QApplication the widget tests share, and destroys it
         *        once every test has run, while main() is still running. A function-static
         *        QApplication would instead be destroyed by the exit handlers, after Qt's own
         *        global state (e.g. its input-device registry), and crash the process on exit.
         */
        class QtApplicationEnvironment : public ::testing::Environment
        {
        public:
            void TearDown() override { s_app.reset(); }

            /// Creates the application on first use (only the tests needing it pay for it).
            static void ensure()
            {
                if (QCoreApplication::instance())
                    return;
                qputenv("QT_QPA_PLATFORM", "offscreen");
                s_app = std::make_unique<QApplication>(s_argc, s_argv);
            }

        private:
            static inline int s_argc = 1;
            static inline char s_name[] = "tests";
            static inline char *s_argv[] = {s_name, nullptr};
            static inline std::unique_ptr<QApplication> s_app;
        };
        // Registered before main(), so gtest tears it down after the last test.
        [[maybe_unused]] ::testing::Environment *const qt_application_environment =
            ::testing::AddGlobalTestEnvironment(new QtApplicationEnvironment);

        /// Text rendering and widgets need a QApplication; one offscreen instance serves every test.
        void ensureApplication() { QtApplicationEnvironment::ensure(); }

        class PlotRendering : public ::testing::Test
        {
        protected:
            static void SetUpTestSuite() { ensureApplication(); }
        };

        class PlotViewTest : public ::testing::Test
        {
        protected:
            static void SetUpTestSuite() { ensureApplication(); }

            void SetUp() override
            {
                PlotView::Content content;
                content.series = PlotSeries{{0.0, 1.0, 2.0}, {0.0, 1.0, 0.5}, {kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
                content.title = "test";
                content.color = Qt::red;
                content.size = kSize;
                view.setContent(content);
                fit = *fitRange(content.series, {}, {});
                area = plotArea(kSize);
            }

            /// Sends a one-notch wheel event (up = zoom in) at `position`; returns whether it was accepted.
            bool wheel(QPoint position, int notches = 1)
            {
                QWheelEvent event(position, position, QPoint(), QPoint(0, 120 * notches), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
                QCoreApplication::sendEvent(&view, &event);
                return event.isAccepted();
            }

            void mouse(QEvent::Type type, QPoint position, Qt::MouseButton button, Qt::MouseButtons buttons)
            {
                QMouseEvent event(type, position, position, button, buttons, Qt::NoModifier);
                QCoreApplication::sendEvent(&view, &event);
            }

            static double xSpan(const PlotRange &r) { return r.x_max - r.x_min; }
            static double ySpan(const PlotRange &r) { return r.y_max - r.y_min; }

            PlotView view;
            PlotRange fit{};
            QRect area;
        };
    } // namespace

    TEST_F(PlotRendering, EmptySeriesRendersPlaceholderAtTheRequestedSize)
    {
        const QImage image = renderPlot(PlotSeries{}, {}, {}, "empty", Qt::red, kSize);
        EXPECT_EQ(image.size(), kSize);
        EXPECT_TRUE(hasInk(image)); // the "No data available" text
        EXPECT_FALSE(hasColor(image, Qt::red));
    }

    TEST_F(PlotRendering, ConstantSeriesIsDrawnInItsColor)
    {
        PlotSeries series{{0.0, 1.0, 2.0}, {5.0, 5.0, 5.0}, {kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
        const QImage image = renderPlot(series, {}, {}, "constant", QColor(0, 0, 255), kSize);
        EXPECT_TRUE(hasColor(image, QColor(0, 0, 255)));
    }

    TEST_F(PlotRendering, PartialNaNBoundsAndGroundTruthAreHandled)
    {
        PlotSeries series{{0.0, 1.0, 2.0}, {0.0, 1.0, 0.5}, {-1.0, kNaN, 0.0}, {1.0, kNaN, 1.0}};
        const std::vector<double> gt_x{0.5, 1.5, 3.0};
        const std::vector<double> gt_y{0.2, kNaN, 0.4};
        const QImage image = renderPlot(series, gt_x, gt_y, "bounds", QColor(255, 0, 0), kSize, /*marker_x=*/1.0);
        EXPECT_EQ(image.size(), kSize);
        EXPECT_TRUE(hasColor(image, QColor(255, 0, 0)));
    }

    TEST_F(PlotRendering, GroundTruthOnlyIsPlotted)
    {
        const QImage image = renderPlot(PlotSeries{}, {0.0, 1.0}, {1.0, 2.0}, "gt", Qt::red, kSize);
        const QImage placeholder = renderPlot(PlotSeries{}, {}, {}, "gt", Qt::red, kSize);
        EXPECT_NE(image, placeholder); // axes and dots, not the "No data available" placeholder
    }

    TEST_F(PlotRendering, DegenerateSizeIsClampedToAMinimum)
    {
        const QImage image = renderPlot(PlotSeries{}, {}, {}, "tiny", Qt::red, QSize(0, -5));
        EXPECT_GT(image.width(), 0);
        EXPECT_GT(image.height(), 0);
    }

    TEST_F(PlotRendering, PlotAreaRegionsAreTheAxesAndTheInterior)
    {
        const QRect area = plotArea(kSize);
        EXPECT_TRUE(QRect(QPoint(0, 0), kSize).contains(area));
        EXPECT_EQ(regionAt(area.center(), kSize), PlotRegion::Plot);
        EXPECT_EQ(regionAt(QPoint(area.left() - 5, area.center().y()), kSize), PlotRegion::YAxis);
        EXPECT_EQ(regionAt(QPoint(area.center().x(), area.bottom() + 5), kSize), PlotRegion::XAxis);
        EXPECT_EQ(regionAt(QPoint(area.center().x(), 1), kSize), PlotRegion::Outside); // title strip
        EXPECT_EQ(regionAt(QPoint(area.left() - 5, area.bottom() + 5), kSize), PlotRegion::Outside); // corner
    }

    TEST_F(PlotRendering, ExplicitRangeReplacesTheFittedOne)
    {
        PlotSeries series{{0.0, 1.0, 2.0}, {0.0, 1.0, 0.5}, {kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
        const QImage fitted = renderPlot(series, {}, {}, "range", QColor(0, 0, 255), kSize);
        const QImage away = renderPlot(series, {}, {}, "range", QColor(0, 0, 255), kSize, kNaN, PlotRange{100.0, 101.0, 100.0, 101.0});
        EXPECT_TRUE(hasColor(fitted, QColor(0, 0, 255)));
        EXPECT_FALSE(hasColor(away, QColor(0, 0, 255))); // the data lies outside the view
    }

    TEST_F(PlotViewTest, WheelInsideZoomsBothAxes)
    {
        EXPECT_TRUE(wheel(area.center()));
        ASSERT_TRUE(view.view().has_value());
        EXPECT_LT(xSpan(*view.view()), xSpan(fit));
        EXPECT_LT(ySpan(*view.view()), ySpan(fit));
        EXPECT_NEAR(xSpan(*view.view()) / xSpan(fit), ySpan(*view.view()) / ySpan(fit), 1e-12);
    }

    TEST_F(PlotViewTest, WheelOnTheYAxisZoomsOnlyVertically)
    {
        EXPECT_TRUE(wheel(QPoint(area.left() - 5, area.center().y())));
        ASSERT_TRUE(view.view().has_value());
        EXPECT_DOUBLE_EQ(view.view()->x_min, fit.x_min);
        EXPECT_DOUBLE_EQ(view.view()->x_max, fit.x_max);
        EXPECT_LT(ySpan(*view.view()), ySpan(fit));
    }

    TEST_F(PlotViewTest, WheelOnTheXAxisZoomsOnlyHorizontally)
    {
        EXPECT_TRUE(wheel(QPoint(area.center().x(), area.bottom() + 5), -1)); // zoom out
        ASSERT_TRUE(view.view().has_value());
        EXPECT_GT(xSpan(*view.view()), xSpan(fit));
        EXPECT_DOUBLE_EQ(view.view()->y_min, fit.y_min);
        EXPECT_DOUBLE_EQ(view.view()->y_max, fit.y_max);
    }

    TEST_F(PlotViewTest, WheelOutsideTheAxesIsLeftForThePanelToScroll)
    {
        EXPECT_FALSE(wheel(QPoint(area.center().x(), 1)));
        EXPECT_FALSE(view.view().has_value());
    }

    TEST_F(PlotViewTest, DragInsidePansBothAxes)
    {
        const QPoint start = area.center();
        mouse(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
        mouse(QEvent::MouseMove, start + QPoint(40, 20), Qt::NoButton, Qt::LeftButton);
        mouse(QEvent::MouseButtonRelease, start + QPoint(40, 20), Qt::LeftButton, Qt::NoButton);
        ASSERT_TRUE(view.view().has_value());
        // Content follows the cursor: dragging right/down shows earlier x and larger y.
        EXPECT_NEAR(view.view()->x_min, fit.x_min - 40.0 * xSpan(fit) / area.width(), 1e-9);
        EXPECT_NEAR(view.view()->y_min, fit.y_min + 20.0 * ySpan(fit) / area.height(), 1e-9);
        EXPECT_NEAR(xSpan(*view.view()), xSpan(fit), 1e-12);
    }

    TEST_F(PlotViewTest, DragOnAnAxisPansOnlyThatAxis)
    {
        const QPoint start(area.center().x(), area.bottom() + 5);
        mouse(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
        mouse(QEvent::MouseMove, start + QPoint(30, 30), Qt::NoButton, Qt::LeftButton);
        mouse(QEvent::MouseButtonRelease, start + QPoint(30, 30), Qt::LeftButton, Qt::NoButton);
        ASSERT_TRUE(view.view().has_value());
        EXPECT_LT(view.view()->x_min, fit.x_min);
        EXPECT_DOUBLE_EQ(view.view()->y_min, fit.y_min);
    }

    TEST_F(PlotViewTest, ViewSurvivesNewContentUntilReset)
    {
        wheel(area.center());
        const PlotRange zoomed = *view.view();
        PlotView::Content content;
        content.series = PlotSeries{{0.0, 5.0}, {3.0, 4.0}, {kNaN, kNaN}, {kNaN, kNaN}};
        content.size = kSize;
        view.setContent(content);
        ASSERT_TRUE(view.view().has_value());
        EXPECT_DOUBLE_EQ(view.view()->x_min, zoomed.x_min);

        view.resetView();
        EXPECT_FALSE(view.view().has_value());
    }

    TEST_F(PlotViewTest, WheelWithoutDataDoesNothing)
    {
        PlotView empty;
        PlotView::Content content;
        content.size = kSize;
        empty.setContent(content);
        QWheelEvent event(area.center(), area.center(), QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(&empty, &event);
        EXPECT_FALSE(empty.view().has_value());
    }

    TEST_F(PlotViewTest, PanelResetsEveryPlotsView)
    {
        PlotPanel panel(nullptr, {});
        panel.setGroups({{"a", 2}, {"b", 1}});
        PlotView::Content content;
        content.series = PlotSeries{{0.0, 1.0}, {0.0, 1.0}, {kNaN, kNaN}, {kNaN, kNaN}};
        content.size = kSize;
        for (auto [g, c] : {std::pair{0, 0}, std::pair{0, 1}, std::pair{1, 0}})
            panel.setPlot(g, c, content);
        ASSERT_NE(panel.plot(1, 0), nullptr);
        EXPECT_EQ(panel.plot(2, 0), nullptr);

        QWheelEvent event(area.center(), area.center(), QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(panel.plot(1, 0), &event);
        ASSERT_TRUE(panel.plot(1, 0)->view().has_value());

        panel.resetViews();
        EXPECT_FALSE(panel.plot(0, 0)->view().has_value());
        EXPECT_FALSE(panel.plot(1, 0)->view().has_value());
    }

    // The InfoWidget tests live here to share this file's QApplication (one per process).
    class InfoWidgetTest : public ::testing::Test
    {
    protected:
        static void SetUpTestSuite() { ensureApplication(); }

        QLabel *content() const { return qobject_cast<QLabel *>(widget.widget()); }

        InfoWidget widget;
        int value = 1;
    };

    TEST_F(InfoWidgetTest, RefreshShowsChangedValues)
    {
        widget.addLabel("Value", [this]() { return std::to_string(value); });
        ASSERT_NE(content(), nullptr);
        EXPECT_TRUE(content()->text().contains("Value: 1"));
        value = 2;
        widget.refresh();
        EXPECT_TRUE(content()->text().contains("Value: 2"));
    }

    TEST_F(InfoWidgetTest, RefreshWithUnchangedValuesLeavesTheContentAlone)
    {
        widget.addLabel("Value", [this]() { return std::to_string(value); });
        const QSize fitted = content()->size();
        content()->resize(1, 1); // a marker: only a re-layout (adjustSize) would undo it
        widget.refresh();
        EXPECT_EQ(content()->size(), QSize(1, 1)) << "an unchanged panel must not be re-laid out every frame";

        value = 2;
        widget.refresh();
        EXPECT_EQ(content()->size(), fitted) << "a changed panel is re-laid out to fit";
    }

    namespace
    {
        using BiasNode3 = Biases::BiasNode<gtsam::Vector3>;
        using BiasPlotWidget = PlotWidget<BiasNode3>;

        /// Two accelerometer biases sharing a key manager (as the IMUs of one estimator do), with rows at t = 0 and 0.1.
        struct TwoBiases
        {
            KeyManager km;
            std::shared_ptr<Biases::BiasModel<gtsam::Vector3>> accelerometer0 = std::make_shared<Biases::BiasModel<gtsam::Vector3>>();
            std::shared_ptr<Biases::BiasModel<gtsam::Vector3>> accelerometer1 = std::make_shared<Biases::BiasModel<gtsam::Vector3>>();
            gtsam::Values values;

            TwoBiases()
            {
                for (double time : {0.0, 0.1})
                    for (const auto &model : {accelerometer0, accelerometer1})
                    {
                        gtsam::Values new_values;
                        model->extendState(km, time, values, new_values);
                        values.insert(new_values);
                    }
                // accelerometer0's bias is (t, 2t, 3t), accelerometer1's is -(t, 2t, 3t).
                for (const auto &node : accelerometer0->state()->estimation_nodes)
                    values.update(node->biasKey(), gtsam::Vector3(node->time, 2 * node->time, 3 * node->time));
                for (const auto &node : accelerometer1->state()->estimation_nodes)
                    values.update(node->biasKey(), gtsam::Vector3(-node->time, -2 * node->time, -3 * node->time));
            }

            std::vector<BiasPlotWidget::NamedState> states() const { return {{"accelerometer0", accelerometer0->state()}, {"accelerometer1", accelerometer1->state()}}; }
        };

        BiasPlotWidget::SnapshotSource fixedSnapshot(double time, const gtsam::Values &values)
        {
            auto snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{time, values});
            return [snapshot](double) { return snapshot; };
        }
    } // namespace

    TEST(PlotWidgetStates, PlotsEveryStateAgainstTimeUnderItsName)
    {
        TwoBiases biases;
        BiasPlotWidget widget(biases.states(), fixedSnapshot(0.1, biases.values));
        EXPECT_TRUE(widget.ingest(0.1));
        EXPECT_FALSE(widget.ingest(0.1)); // the same snapshot again

        const auto groups = widget.plotGroups(0.1);
        ASSERT_EQ(groups.size(), 2u);
        EXPECT_EQ(groups[0].name, "accelerometer0 bias");
        EXPECT_EQ(groups[1].name, "accelerometer1 bias");
        EXPECT_EQ(groups[0].labels, (std::vector<std::string>{"bias[0]", "bias[1]", "bias[2]"}));
        for (std::size_t g = 0; g < groups.size(); g++)
        {
            ASSERT_EQ(groups[g].series.size(), 3u);
            const double sign = g == 0 ? 1.0 : -1.0;
            for (std::size_t c = 0; c < 3; c++)
            {
                EXPECT_EQ(groups[g].series[c].x, (std::vector<double>{0.0, 0.1}));
                ASSERT_EQ(groups[g].series[c].mean.size(), 2u);
                EXPECT_DOUBLE_EQ(groups[g].series[c].mean[0], 0.0);
                EXPECT_DOUBLE_EQ(groups[g].series[c].mean[1], sign * 0.1 * static_cast<double>(c + 1));
            }
        }

        // Earlier display times show only the rows estimated by then.
        const auto earlier = widget.plotGroups(0.05);
        ASSERT_EQ(earlier.size(), 2u);
        EXPECT_EQ(earlier[0].series[0].x, (std::vector<double>{0.0}));
    }

    TEST(PlotWidgetStates, NothingToPlotBeforeASnapshot)
    {
        TwoBiases biases;
        BiasPlotWidget widget(biases.states(), [](double) { return std::shared_ptr<const EstimatorSnapshot>(); });
        EXPECT_FALSE(widget.ingest(0.1));
        EXPECT_TRUE(widget.plotGroups(0.1).empty());
    }

    TEST(PlotWidgetStates, RequiresStatesAndASnapshotSource)
    {
        TwoBiases biases;
        const auto source = fixedSnapshot(0.1, biases.values);
        EXPECT_THROW(BiasPlotWidget({}, source), std::invalid_argument);
        EXPECT_THROW(BiasPlotWidget({{"accelerometer0", nullptr}}, source), std::invalid_argument);
        EXPECT_THROW(BiasPlotWidget(biases.states(), nullptr), std::invalid_argument);
    }

    TEST(PlotWidgetStates, RobotViewGroupsAreNotPrefixed)
    {
        using Node = Priors::WNOA::Node<gtsam::Pose3>;
        auto state = std::make_shared<Types::ContinuumRobotState<Node>>(1.0, 2);
        Priors::WNOA::Values<gtsam::Pose3> reference;
        reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 1, 0, 0).finished());
        auto prior = std::make_shared<Priors::WNOA::Prior<gtsam::Pose3>>(state, reference);
        KeyManager km;
        gtsam::Values values;
        prior->extendState(km, 0.0, gtsam::Values(), values);

        auto robot_view = std::make_shared<ContinuumRobot<Node>>(state);
        robot_view->bindSnapshotSource(PlotWidget<Node>::SnapshotSource(fixedSnapshot(0.0, values)));
        PlotWidget<Node> widget(robot_view);
        EXPECT_TRUE(widget.ingest(0.0));
        const auto groups = widget.plotGroups(0.0);
        ASSERT_EQ(groups.size(), 3u);
        EXPECT_EQ(groups[0].name, "pose");
        EXPECT_EQ(groups[1].name, "strain");
        EXPECT_EQ(groups[2].name, "velocity");
        EXPECT_EQ(groups[0].series[0].x, (std::vector<double>{0.0}));
    }
} // namespace Spacetime
