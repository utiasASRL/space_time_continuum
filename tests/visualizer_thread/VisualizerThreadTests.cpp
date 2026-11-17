/**
 * Built as its own executable (see tests/CMakeLists.txt): a process can only have one
 * QApplication at a time, and the main test executable already keeps one alive on its
 * main thread for the widget tests, whereas these tests create theirs on a worker thread.
 */
#include <atomic>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <thread>

#include <sys/resource.h>

#include <QApplication>
#include <QLabel>
#include <QRect>
#include <QImage>
#include <QShortcut>
#include <QSlider>
#include <QTimer>
#include <gtest/gtest.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkRenderer.h>

#include <spacetime/visualizer/default/VisualizerThread.hpp>
#include <spacetime/visualizer/default/objects/InfoWidget.hpp>
#include <spacetime/visualizer/default/objects/RuntimeWidget.hpp>

#include "../visualizer/VideoProbe.hpp"

namespace Spacetime::Visualization::Default
{
    namespace
    {
        int argc = 1;
        char name[] = "visualizer_thread_tests";
        char *argv[] = {name, nullptr};

        class VisualizerThreadTest : public ::testing::Test
        {
        protected:
            void SetUp() override { qputenv("QT_QPA_PLATFORM", "offscreen"); }
        };

        /// Runs `key`'s action, as a key press would (offscreen, a QShortcut never sees real key events).
        void press(Visualizer &visualizer, Qt::Key key)
        {
            for (QShortcut *shortcut : visualizer.findChildren<QShortcut *>())
                if (shortcut->key() == QKeySequence(key))
                    Q_EMIT shortcut->activated();
        }

        /// Whether any of the visualizer's shortcuts is bound to `key`.
        bool isBound(Visualizer &visualizer, Qt::Key key)
        {
            for (QShortcut *shortcut : visualizer.findChildren<QShortcut *>())
                if (shortcut->key() == QKeySequence(key))
                    return true;
            return false;
        }

        /// Presses X once, then quits.
        void screenshotOnce(Visualizer &visualizer, std::atomic<bool> &pressed)
        {
            QTimer::singleShot(50, &visualizer, [&visualizer, &pressed]()
                               {
                press(visualizer, Qt::Key_X);
                pressed = true;
                QApplication::quit(); });
        }

        /// A scratch folder, emptied now and removed when destroyed.
        struct ScratchFolder
        {
            explicit ScratchFolder(const std::string &name) : path(std::filesystem::temp_directory_path() / name)
            {
                std::filesystem::remove_all(path);
                std::filesystem::create_directories(path);
            }
            ~ScratchFolder() { std::filesystem::remove_all(path); }
            std::filesystem::path path;
        };

        double secondsSince(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } // namespace

    TEST_F(VisualizerThreadTest, BuildsOnItsOwnLowerPriorityThread)
    {
        std::thread::id build_thread;
        int build_nice = 0;
        const int control_nice = getpriority(PRIO_PROCESS, 0);

        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [&](Visualizer &)
                             {
            build_thread = std::this_thread::get_id();
            build_nice = getpriority(PRIO_PROCESS, 0); });
        vis.waitReady();

        EXPECT_NE(build_thread, std::this_thread::get_id());
        EXPECT_GT(build_nice, control_nice);
        EXPECT_EQ(getpriority(PRIO_PROCESS, 0), control_nice); // the control thread is untouched

        vis.requestQuit();
        vis.join();
    }

    TEST_F(VisualizerThreadTest, QuitBeforePlaybackEndsTheThreadAndReportsClosed)
    {
        std::atomic<int> closed{0};
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [](Visualizer &) {}, [&]()
                             { closed++; });
        vis.waitReady();
        const auto start = std::chrono::steady_clock::now();
        vis.requestQuit();
        vis.join();
        EXPECT_LT(secondsSince(start), 2.0);
        EXPECT_EQ(closed.load(), 1);
    }

    TEST_F(VisualizerThreadTest, QuitDuringPlaybackEndsTheEventLoop)
    {
        std::atomic<int> closed{0};
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [](Visualizer &) {}, [&]()
                             { closed++; });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        const auto start = std::chrono::steady_clock::now();
        vis.requestQuit();
        EXPECT_EQ(vis.join(), 0);
        EXPECT_LT(secondsSince(start), 2.0);
        EXPECT_EQ(closed.load(), 1);
    }

    // Quitting from the GUI itself (e.g. the q shortcut, or closing the window) tells control.
    TEST_F(VisualizerThreadTest, QuitFromTheGuiReportsClosed)
    {
        std::atomic<int> closed{0};
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [](Visualizer &)
                             { QTimer::singleShot(50, []()
                                                  { QApplication::quit(); }); },
                             [&]()
                             { closed++; });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);

        const auto start = std::chrono::steady_clock::now();
        while (closed.load() == 0 && secondsSince(start) < 5.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        EXPECT_EQ(closed.load(), 1);
        vis.join();
    }

    TEST_F(VisualizerThreadTest, BuildExceptionIsRethrownByWaitReady)
    {
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [](Visualizer &)
                             { throw std::runtime_error("build failure"); });
        EXPECT_THROW(vis.waitReady(), std::runtime_error);
        vis.join();
    }

    TEST_F(VisualizerThreadTest, LiveTimeSourceIsReadFromTheVisualizerThread)
    {
        VisualizerOptions options;
        options.online = true;
        std::atomic<bool> read{false};
        std::thread::id reader;
        VisualizerThread vis(argc, argv, options, "", [](Visualizer &) {});
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0, [&]()
                          {
            reader = std::this_thread::get_id();
            read = true;
            return 0.0; });

        const auto start = std::chrono::steady_clock::now();
        while (!read && secondsSince(start) < 5.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        vis.requestQuit();
        vis.join();
        ASSERT_TRUE(read.load());
        EXPECT_NE(reader, std::this_thread::get_id());
    }

    TEST_F(VisualizerThreadTest, DestructorQuitsAndJoins)
    {
        const auto start = std::chrono::steady_clock::now();
        {
            VisualizerThread vis(argc, argv, VisualizerOptions(), "", [](Visualizer &) {});
            vis.waitReady();
            vis.beginPlayback(0.0, 1.0);
        }
        EXPECT_LT(secondsSince(start), 5.0);
    }

    // The runtime bar is flush with the bottom of the window, with the timeline directly above it.
    TEST_F(VisualizerThreadTest, RuntimeBarIsAtTheBottomBelowTheTimeline)
    {
        VisualizerOptions options;
        options.show_timeline_widget = true;
        std::atomic<bool> measured{false};
        QRect bar, timeline;
        VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                             {
            visualizer.addVisualizable(std::make_shared<RuntimeWidget>([]()
                                                                       { return std::shared_ptr<const EstimatorSnapshot>(); }));
            // Once shown, so the resize is delivered right away.
            QTimer::singleShot(50, &visualizer, [&]()
                               {
                visualizer.resize(1000, 600);
                bar = visualizer.findChild<QLabel *>(RuntimeWidget::kObjectName)->geometry();
                timeline = visualizer.findChild<TimelineWidget *>()->geometry();
                measured = true;
                QApplication::quit(); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();

        ASSERT_TRUE(measured.load());
        EXPECT_EQ(bar.y() + bar.height(), 600);
        EXPECT_GT(bar.width(), 0);
        EXPECT_EQ(timeline.y() + timeline.height(), bar.y());
    }

    // The runtime bar starts in sync with the (hidden by default) diagnostics panel, and H toggles both.
    TEST_F(VisualizerThreadTest, RuntimeBarStartsHiddenAndTogglesLikeTheDiagnosticsPanel)
    {
        for (const bool diagnostics : {false, true})
        {
            VisualizerOptions options;
            options.show_diagnostics_label = diagnostics;
            std::atomic<bool> measured{false};
            bool initially = !diagnostics, toggled = diagnostics, restored = !diagnostics;
            VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                                 {
                auto widget = std::make_shared<RuntimeWidget>([]()
                                                              { return std::shared_ptr<const EstimatorSnapshot>(); });
                visualizer.addVisualizable(widget);
                QTimer::singleShot(50, &visualizer, [&, widget]()
                                   {
                    const QLabel *bar = visualizer.findChild<QLabel *>(RuntimeWidget::kObjectName);
                    initially = bar->isVisible();
                    widget->toggleVisibility(); // what H runs
                    toggled = bar->isVisible();
                    widget->toggleVisibility();
                    restored = bar->isVisible();
                    measured = true;
                    QApplication::quit(); }); });
            vis.waitReady();
            vis.beginPlayback(0.0, 1.0);
            vis.join();

            ASSERT_TRUE(measured.load());
            EXPECT_EQ(initially, diagnostics);
            EXPECT_EQ(toggled, !diagnostics);
            EXPECT_EQ(restored, diagnostics);
        }
    }

    // X saves exactly one screenshot per press, under <output_path>/screenshots/sample_<index>/, however many frames render in between.
    TEST_F(VisualizerThreadTest, ScreenshotKeyWritesOneFramePerPress)
    {
        ScratchFolder output("spacetime_screenshot_test");
        std::atomic<bool> pressed{false};
        VisualizerThread vis(argc, argv, VisualizerOptions(), output.path.string(), [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                press(visualizer, Qt::Key_X);
                QTimer::singleShot(300, &visualizer, [&]()
                                   {
                    press(visualizer, Qt::Key_X);
                    QTimer::singleShot(300, &visualizer, [&]()
                                       { pressed = true; QApplication::quit(); }); }); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();

        ASSERT_TRUE(pressed.load());
        const std::filesystem::path frames = output.path / "screenshots" / "sample_0";
        EXPECT_TRUE(std::filesystem::exists(frames / "0.png"));
        EXPECT_TRUE(std::filesystem::exists(frames / "1.png"));
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(frames), std::filesystem::directory_iterator()), 2);
    }

    // With no output path (the demos' default), screenshots go under the working directory instead of the filesystem root.
    TEST_F(VisualizerThreadTest, ScreenshotWithoutAnOutputPathWritesUnderTheWorkingDirectory)
    {
        ScratchFolder working("spacetime_screenshot_cwd_test");
        const std::filesystem::path previous = std::filesystem::current_path();
        std::filesystem::current_path(working.path);

        std::atomic<bool> pressed{false};
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [&](Visualizer &visualizer)
                             { screenshotOnce(visualizer, pressed); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();
        std::filesystem::current_path(previous);

        ASSERT_TRUE(pressed.load());
        EXPECT_TRUE(std::filesystem::exists(working.path / "screenshots" / "sample_0" / "0.png"));
    }

    // A screenshot that cannot be written is reported instead of crashing the visualizer.
    TEST_F(VisualizerThreadTest, UnwritableOutputPathDoesNotCrash)
    {
        // A regular file where the output folder should be: its screenshots/ subfolder cannot be created.
        ScratchFolder scratch("spacetime_screenshot_blocker_test");
        const std::filesystem::path blocker = scratch.path / "blocker";
        std::ofstream(blocker) << "not a folder";

        std::atomic<bool> pressed{false};
        VisualizerThread vis(argc, argv, VisualizerOptions(), blocker.string(), [&](Visualizer &visualizer)
                             { screenshotOnce(visualizer, pressed); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();

        EXPECT_TRUE(pressed.load());
        EXPECT_TRUE(std::filesystem::is_regular_file(blocker));
    }

    // Info messages ("Recording video...", "Screenshot saved", ...) are left out of exported frames, but stay on screen.
    TEST_F(VisualizerThreadTest, ExportedFramesLeaveOutTheInfoMessage)
    {
        std::atomic<bool> measured{false};
        bool identical = false, still_shown = false;
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                visualizer.playback().pause(); // a still scene, so frames only differ by the message
                visualizer.update(0.0);
                const QImage without = visualizer.grabFrame();
                visualizer.sendMessage("Recording video...");
                const QImage with = visualizer.grabFrame();
                identical = with == without;
                still_shown = visualizer.findChild<QLabel *>("info_label") && visualizer.findChild<QLabel *>("info_label")->isVisible();
                measured = true;
                QApplication::quit(); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();

        ASSERT_TRUE(measured.load());
        EXPECT_TRUE(identical);
        EXPECT_TRUE(still_shown);
    }

    // V (offline) steps through every estimation tick of the range and encodes one frame per tick, at 1 / period fps.
    TEST_F(VisualizerThreadTest, VideoKeyExportsOneFramePerEstimationTick)
    {
        if (!Spacetime::Tests::onPath("ffmpeg") || !Spacetime::Tests::onPath("ffprobe"))
            GTEST_SKIP() << "ffmpeg/ffprobe not on PATH";
        ScratchFolder output("spacetime_video_test");
        VisualizerOptions options;
        options.estimation_period = 0.1;
        std::atomic<bool> started{false}, finished{false};
        VisualizerThread vis(argc, argv, options, output.path.string(), [&](Visualizer &visualizer)
                             {
            QTimer::singleShot(50, &visualizer, [&]()
                               {
                press(visualizer, Qt::Key_V);
                started = visualizer.exportingVideo();
                // Poll until the export ends (a guard quits regardless after 30 s).
                auto *poll = new QTimer(&visualizer);
                QObject::connect(poll, &QTimer::timeout, &visualizer, [&visualizer, &finished, poll]()
                                 {
                    if (visualizer.exportingVideo())
                        return;
                    poll->stop();
                    finished = true;
                    QApplication::quit(); });
                poll->start(20); });
            QTimer::singleShot(30000, &visualizer, []()
                               { QApplication::quit(); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();

        ASSERT_TRUE(started.load());
        ASSERT_TRUE(finished.load());
        const std::filesystem::path video = output.path / "videos" / "video_0.mp4";
        ASSERT_TRUE(std::filesystem::exists(video));
        const auto info = Spacetime::Tests::probeVideo(video);
        ASSERT_TRUE(info.has_value());
        EXPECT_EQ(info->frames, 11); // 0, 0.1, ..., 1.0
        EXPECT_DOUBLE_EQ(info->fps, 10.0);
    }

    // A live stream has no end to step to, so V is offline only.
    TEST_F(VisualizerThreadTest, VideoKeyIsOfflineOnly)
    {
        for (const bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            options.estimation_period = 0.1;
            bool bound = online;
            QString text;
            VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                                 {
                bound = isBound(visualizer, Qt::Key_V);
                text = qobject_cast<QLabel *>(visualizer.findChild<InfoWidget *>()->widget())->text(); });
            vis.waitReady();
            vis.requestQuit();
            vis.join();

            EXPECT_EQ(bound, !online);
            EXPECT_EQ(text.contains("Record video [V]: 0"), !online) << text.toStdString();
            EXPECT_TRUE(text.contains("Screenshot [x]")) << text.toStdString();
        }
    }

    // Offline, the arrow keys step between estimation ticks; a live stream has nothing to step through, so they are unbound online.
    TEST_F(VisualizerThreadTest, ArrowKeysStepBetweenEstimationTicksOfflineOnly)
    {
        for (const bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            options.estimation_period = 0.1;
            std::atomic<bool> measured{false};
            bool left = online, right = online;
            double after_right = -1.0, after_left = -1.0;
            VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                                 { QTimer::singleShot(50, &visualizer, [&]()
                                                      {
                left = isBound(visualizer, Qt::Key_Left);
                right = isBound(visualizer, Qt::Key_Right);
                if (!online)
                {
                    visualizer.playback().pause();
                    visualizer.playback().seek(0.23); // between ticks
                    press(visualizer, Qt::Key_Right);
                    after_right = visualizer.playback().displayedTime();
                    press(visualizer, Qt::Key_Left);
                    press(visualizer, Qt::Key_Left);
                    after_left = visualizer.playback().displayedTime();
                }
                measured = true;
                QApplication::quit(); }); });
            vis.waitReady();
            vis.beginPlayback(0.0, 1.0);
            vis.join();

            ASSERT_TRUE(measured.load());
            EXPECT_EQ(left, !online);
            EXPECT_EQ(right, !online);
            if (!online)
            {
                EXPECT_NEAR(after_right, 0.3, 1e-9);
                EXPECT_NEAR(after_left, 0.1, 1e-9);
            }
        }
    }

    // Without the estimation period, the arrow keys keep stepping 1% of the range.
    // Offline, moving the timeline slider seeks playback; online, the slider is display only.
    TEST_F(VisualizerThreadTest, TimelineSeeksOfflineOnly)
    {
        for (const bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            options.show_timeline_widget = true;
            std::atomic<bool> measured{false};
            bool ignores_mouse = !online;
            double before = -1.0, after = -1.0;
            VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                                 { QTimer::singleShot(50, &visualizer, [&]()
                                                      {
                visualizer.playback().pause();
                QSlider *slider = visualizer.findChild<TimelineWidget *>()->slider;
                ignores_mouse = slider->testAttribute(Qt::WA_TransparentForMouseEvents);
                before = visualizer.playback().displayedTime();
                slider->setSliderPosition(750); // as a drag does, then the action it triggers
                slider->triggerAction(QAbstractSlider::SliderMove);
                after = visualizer.playback().displayedTime();
                measured = true;
                QApplication::quit(); }); });
            vis.waitReady();
            vis.beginPlayback(0.0, 2.0);
            vis.join();

            ASSERT_TRUE(measured.load());
            EXPECT_EQ(ignores_mouse, online);
            if (online)
                EXPECT_EQ(after, before);
            else
                EXPECT_NEAR(after, 1.5, 1e-9); // 750 / 1000 of [0, 2]
        }
    }

    // With a sliding window, the timeline highlights [t - window, t] behind the current time t.
    TEST_F(VisualizerThreadTest, TimelineHighlightsTheWindowBehindTheCurrentTime)
    {
        VisualizerOptions options;
        options.show_timeline_widget = true;
        options.estimator_window = 2.0;
        std::atomic<bool> measured{false};
        QColor in_window, before_window, after_time;
        VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                visualizer.resize(1000, 600);
                visualizer.playback().pause();
                visualizer.playback().seek(5.0);
                visualizer.update(5.0);
                TimelineWidget *timeline = visualizer.findChild<TimelineWidget *>();
                const QImage image = timeline->grab().toImage();
                const QRect groove = timeline->slider->geometry();
                const auto x = [&](double t) { return groove.x() + static_cast<int>(t / 10.0 * groove.width()); };
                const int y = groove.y() + groove.height() / 2;
                in_window = image.pixelColor(x(4.0), y);
                before_window = image.pixelColor(x(1.0), y);
                after_time = image.pixelColor(x(8.0), y);
                measured = true;
                QApplication::quit(); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 10.0);
        vis.join();

        ASSERT_TRUE(measured.load());
        EXPECT_EQ(in_window, QColor(255, 120, 120));
        EXPECT_EQ(before_window, QColor(255, 201, 201));
        EXPECT_NE(after_time, QColor(255, 120, 120));
        EXPECT_NE(after_time, QColor(255, 201, 201));
    }

    TEST_F(VisualizerThreadTest, ArrowKeysStepAPercentOfTheRangeWithoutAPeriod)
    {
        std::atomic<bool> measured{false};
        double after_right = -1.0;
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                visualizer.playback().pause();
                visualizer.playback().seek(2.0);
                press(visualizer, Qt::Key_Right);
                after_right = visualizer.playback().displayedTime();
                measured = true;
                QApplication::quit(); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 10.0);
        vis.join();

        ASSERT_TRUE(measured.load());
        EXPECT_NEAR(after_right, 2.1, 1e-9);
    }

    // Without the estimation period the ticks are unknown: V writes nothing (and does not crash).
    TEST_F(VisualizerThreadTest, VideoWithoutAPeriodWritesNothing)
    {
        ScratchFolder output("spacetime_video_no_period_test");
        std::atomic<bool> pressed{false};
        bool exporting = true;
        VisualizerThread vis(argc, argv, VisualizerOptions(), output.path.string(), [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                press(visualizer, Qt::Key_V);
                exporting = visualizer.exportingVideo();
                pressed = true;
                QApplication::quit(); }); });
        vis.waitReady();
        vis.beginPlayback(0.0, 1.0);
        vis.join();

        ASSERT_TRUE(pressed.load());
        EXPECT_FALSE(exporting);
        EXPECT_FALSE(std::filesystem::exists(output.path / "videos"));
    }

    // Many instances of one view (e.g. one robot view per robot) describe their shared controls once.
    TEST_F(VisualizerThreadTest, RepeatedControlInfoIsListedOnce)
    {
        QString text;
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [&](Visualizer &visualizer)
                             {
            for (int i = 0; i < 3; i++)
                visualizer.addControlInfo("Adjust things [[, ]]");
            visualizer.addControlInfo("Other control [o]");
            text = qobject_cast<QLabel *>(visualizer.findChild<InfoWidget *>()->widget())->text(); });
        vis.waitReady();
        vis.requestQuit();
        vis.join();

        EXPECT_EQ(text.count("Adjust things [[, ]]"), 1);
        EXPECT_EQ(text.count("Other control [o]"), 1);
    }

    // A light kit: every light moves with the camera, one of them a headlight, so whichever side
    // of the scene the camera looks at is lit.
    TEST_F(VisualizerThreadTest, SceneIsLitFromTheCamera)
    {
        int lights = 0, headlights = 0, camera_lights = 0;
        VisualizerThread vis(argc, argv, VisualizerOptions(), "", [&](Visualizer &visualizer)
                             {
            vtkLightCollection *collection = visualizer.getRenderer()->GetLights();
            collection->InitTraversal();
            while (vtkLight *light = collection->GetNextItem())
            {
                lights++;
                headlights += light->LightTypeIsHeadlight() ? 1 : 0;
                camera_lights += (light->LightTypeIsHeadlight() || light->LightTypeIsCameraLight()) ? 1 : 0;
            } });
        vis.waitReady();
        vis.requestQuit();
        vis.join();

        EXPECT_GT(lights, 1);
        EXPECT_EQ(headlights, 1);
        EXPECT_EQ(camera_lights, lights) << "no light stays fixed in the world";
    }

    // L loops an offline replay (off by default); a live stream has nothing to loop.
    TEST_F(VisualizerThreadTest, ReplayLoopIsOfflineOnlyAndStartsOff)
    {
        for (bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            QString text;
            VisualizerThread vis(argc, argv, options, "", [&](Visualizer &visualizer)
                                 { text = qobject_cast<QLabel *>(visualizer.findChild<InfoWidget *>()->widget())->text(); });
            vis.waitReady();
            vis.requestQuit();
            vis.join();

            if (online)
                EXPECT_FALSE(text.contains("Loop replay")) << text.toStdString();
            else
                EXPECT_TRUE(text.contains("Loop replay [L]: 0")) << text.toStdString();
        }
    }
} // namespace Spacetime::Visualization::Default
