/**
 * Built as its own executable (see tests/CMakeLists.txt): a process can only have one
 * QApplication at a time, and the main test executable already keeps one alive for the
 * widget tests, whereas these create (and destroy) one per test, on the host's GUI thread:
 * its own thread (HostMode::WorkerThread) or the test's (HostMode::CallingThread). Every
 * test runs in each mode this platform supports.
 */
#include <atomic>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <thread>

#include <QApplication>
#include <QLabel>
#include <QRect>
#include <QImage>
#include <QShortcut>
#include <QSlider>
#include <QTimer>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkRenderer.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/interface/Interfaces.hpp>
#include <spacetime/interface/standalone/ThreadPriority.hpp>
#include <spacetime/visualizer/default/VisualizerHost.hpp>
#include <spacetime/visualizer/default/objects/InfoWidget.hpp>
#include <spacetime/visualizer/default/objects/RuntimeWidget.hpp>

#include "../ProcessId.hpp"
#include "../TestEstimatorOptions.hpp"
#include "../visualizer/VideoProbe.hpp"

namespace Spacetime::Visualization::Default
{
    namespace
    {
        int argc = 1;
        char name[] = "visualizer_thread_tests";
        char *argv[] = {name, nullptr};

        using Interface::Standalone::currentThreadPriority;

        /// The mode the running test hosts the visualizer in (its parameter).
        HostMode current_mode = HostMode::WorkerThread;

        /// A VisualizerHost in the running test's mode.
        struct Host : VisualizerHost
        {
            Host(VisualizerOptions options, std::string output_path, BuildFn build)
                : VisualizerHost(argc, argv, std::move(options), std::move(output_path), std::move(build), current_mode) {}
        };

        class VisualizerHostTest : public ::testing::TestWithParam<HostMode>
        {
        public:
            bool callingThread() const { return GetParam() == HostMode::CallingThread; }

        protected:
            void SetUp() override
            {
                qputenv("QT_QPA_PLATFORM", "offscreen");
                current_mode = GetParam();
            }
        };

        /**
         * Runs a test's body as the control thread. On Linux in CallingThread mode, that is a
         * fresh thread: run() lowers its calling thread, which an unprivileged thread cannot raise
         * back, and threads inherit their creator's priority; so lowering the test's own thread
         * would leave every later test's threads lowered too. Elsewhere, the test's own thread
         * (on macOS it must be: the GUI runs on the main thread).
         */
        template <typename Fn>
        void asControlThread(const VisualizerHostTest &test, Fn fn)
        {
#if !defined(__APPLE__) && !defined(_WIN32)
            if (test.callingThread())
            {
                std::thread control(fn);
                control.join();
                return;
            }
#endif
            (void)test;
            fn();
        }

/// A test run in each host mode, its body as the control thread (see asControlThread); `test` is the fixture.
#define HOST_TEST(name)                                                       \
    void name##Body(const VisualizerHostTest &test);                          \
    TEST_P(VisualizerHostTest, name)                                          \
    {                                                                         \
        asControlThread(*this, [this]() { name##Body(*this); });              \
    }                                                                         \
    void name##Body([[maybe_unused]] const VisualizerHostTest &test)

        /// Calls `fn` from another thread after `delay`, e.g. to quit a host whose run() blocks this one; joined on destruction.
        struct Later
        {
            template <typename Fn>
            Later(std::chrono::milliseconds delay, Fn fn) : thread([delay, fn]()
                                                                   { std::this_thread::sleep_for(delay); fn(); }) {}
            ~Later() { thread.join(); }
            std::thread thread;
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

        /// A scratch folder, emptied now and removed when destroyed; named apart per host mode and process, as ctest runs tests concurrently.
        struct ScratchFolder
        {
            explicit ScratchFolder(const std::string &name)
                : path(std::filesystem::temp_directory_path() /
                       (name + (current_mode == HostMode::WorkerThread ? "_worker_" : "_calling_") + std::to_string(Testing::processId())))
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

    // WorkerThread: built on the host's own thread, lowered for good; CallingThread: built on the
    // calling thread, which is not lowered until run() shows the window.
    HOST_TEST(BuildsOnTheGuiThreadAndOnlyLowersIt)
    {
        std::thread::id build_thread;
        int build_priority = 0;
        const int control_priority = currentThreadPriority();

        Host vis(VisualizerOptions(), "", [&](Visualizer &)
                 {
            build_thread = std::this_thread::get_id();
            build_priority = currentThreadPriority(); });
        vis.open();

        if (test.callingThread())
        {
            EXPECT_EQ(build_thread, std::this_thread::get_id());
            EXPECT_EQ(build_priority, control_priority);
        }
        else
        {
            EXPECT_NE(build_thread, std::this_thread::get_id());
            EXPECT_LT(build_priority, control_priority);
        }
        EXPECT_EQ(currentThreadPriority(), control_priority);

        vis.requestClose();
        vis.run();
    }

    // CallingThread: the calling thread is lowered while the window runs, and restored after where the
    // OS allows (not on Linux, where an unprivileged thread cannot raise its priority again).
    HOST_TEST(CallingThreadIsLoweredOnlyWhileTheWindowRuns)
    {
        if (!test.callingThread())
            GTEST_SKIP() << "CallingThread mode only";
        const int control_priority = currentThreadPriority();
        std::atomic<bool> measured{false};
        int running_priority = control_priority;
        Host vis(VisualizerOptions(), "", [&](Visualizer &visualizer)
                 { QTimer::singleShot(50, &visualizer, [&]()
                                      {
                running_priority = currentThreadPriority();
                measured = true;
                QApplication::quit(); }); });
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

        ASSERT_TRUE(measured.load());
        EXPECT_LT(running_priority, control_priority);
#if defined(__APPLE__) || defined(_WIN32)
        EXPECT_EQ(currentThreadPriority(), control_priority);
#endif
    }

    HOST_TEST(QuitBeforePlaybackEndsTheHostAndReportsClosed)
    {
        std::atomic<int> closed{0};
        Host vis(VisualizerOptions(), "", [](Visualizer &) {});
        vis.open([&]()
                 { closed++; });
        const auto start = std::chrono::steady_clock::now();
        vis.requestClose();
        EXPECT_EQ(vis.run(), 0);
        EXPECT_LT(secondsSince(start), 2.0);
        EXPECT_EQ(closed.load(), 1);
    }

    HOST_TEST(QuitDuringPlaybackEndsTheEventLoop)
    {
        std::atomic<int> closed{0};
        Host vis(VisualizerOptions(), "", [](Visualizer &) {});
        vis.open([&]()
                 { closed++; });
        vis.beginPlayback(0.0, 1.0);

        const auto start = std::chrono::steady_clock::now();
        {
            Later quit(std::chrono::milliseconds(100), [&]()
                       { vis.requestClose(); });
            EXPECT_EQ(vis.run(), 0);
        }
        EXPECT_LT(secondsSince(start), 2.0);
        EXPECT_EQ(closed.load(), 1);
    }

    // Quitting from the GUI itself (e.g. the q shortcut, or closing the window) tells control.
    HOST_TEST(QuitFromTheGuiReportsClosed)
    {
        std::atomic<int> closed{0};
        Host vis(VisualizerOptions(), "", [](Visualizer &)
                 { QTimer::singleShot(50, []()
                                      { QApplication::quit(); }); });
        vis.open([&]()
                 { closed++; });
        vis.beginPlayback(0.0, 1.0);
        vis.run();
        EXPECT_EQ(closed.load(), 1);
    }

    HOST_TEST(BuildExceptionIsRethrownByOpen)
    {
        std::atomic<int> closed{0};
        Host vis(VisualizerOptions(), "", [](Visualizer &)
                 { throw std::runtime_error("build failure"); });
        EXPECT_THROW(vis.open([&]()
                              { closed++; }),
                     std::runtime_error);
        EXPECT_EQ(vis.run(), 0);
        EXPECT_EQ(closed.load(), 1);
    }

    HOST_TEST(RunWithoutPlaybackIsAnErrorOnTheCallingThread)
    {
        if (!test.callingThread())
            GTEST_SKIP() << "CallingThread mode only (a worker waits for playback)";
        Host vis(VisualizerOptions(), "", [](Visualizer &) {});
        vis.open();
        EXPECT_THROW(vis.run(), std::logic_error);
        vis.requestClose();
        vis.run();
    }

    HOST_TEST(LiveTimeSourceIsReadFromTheGuiThread)
    {
        VisualizerOptions options;
        options.online = true;
        std::atomic<bool> read{false};
        std::thread::id reader;
        Host vis(options, "", [](Visualizer &) {});
        vis.open();
        vis.beginPlayback(0.0, 1.0, [&]()
                          {
            reader = std::this_thread::get_id();
            read = true;
            return 0.0; });

        {
            Later quit(std::chrono::milliseconds(0), [&]()
                       {
                const auto start = std::chrono::steady_clock::now();
                while (!read && secondsSince(start) < 5.0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                vis.requestClose(); });
            vis.run();
        }
        ASSERT_TRUE(read.load());
        if (test.callingThread())
            EXPECT_EQ(reader, std::this_thread::get_id());
        else
            EXPECT_NE(reader, std::this_thread::get_id());
    }

    HOST_TEST(DestructorQuitsAndJoins)
    {
        const auto start = std::chrono::steady_clock::now();
        {
            Host vis(VisualizerOptions(), "", [](Visualizer &) {});
            vis.open();
            vis.beginPlayback(0.0, 1.0);
        }
        EXPECT_LT(secondsSince(start), 5.0);
    }

    // The runtime bar is flush with the bottom of the window, with the timeline directly above it.
    HOST_TEST(RuntimeBarIsAtTheBottomBelowTheTimeline)
    {
        VisualizerOptions options;
        options.show_timeline_widget = true;
        std::atomic<bool> measured{false};
        QRect bar, timeline;
        Host vis(options, "", [&](Visualizer &visualizer)
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
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

        ASSERT_TRUE(measured.load());
        EXPECT_EQ(bar.y() + bar.height(), 600);
        EXPECT_GT(bar.width(), 0);
        EXPECT_EQ(timeline.y() + timeline.height(), bar.y());
    }

    // The runtime bar starts in sync with the (hidden by default) diagnostics panel, and H toggles both.
    HOST_TEST(RuntimeBarStartsHiddenAndTogglesLikeTheDiagnosticsPanel)
    {
        for (const bool diagnostics : {false, true})
        {
            VisualizerOptions options;
            options.show_diagnostics_label = diagnostics;
            std::atomic<bool> measured{false};
            bool initially = !diagnostics, toggled = diagnostics, restored = !diagnostics;
            Host vis(options, "", [&](Visualizer &visualizer)
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
            vis.open();
            vis.beginPlayback(0.0, 1.0);
            vis.run();

            ASSERT_TRUE(measured.load());
            EXPECT_EQ(initially, diagnostics);
            EXPECT_EQ(toggled, !diagnostics);
            EXPECT_EQ(restored, diagnostics);
        }
    }

    // X saves exactly one screenshot per press, under <output_path>/screenshots/sample_<index>/, however many frames render in between.
    HOST_TEST(ScreenshotKeyWritesOneFramePerPress)
    {
        ScratchFolder output("spacetime_screenshot_test");
        std::atomic<bool> pressed{false};
        Host vis(VisualizerOptions(), output.path.string(), [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                press(visualizer, Qt::Key_X);
                QTimer::singleShot(300, &visualizer, [&]()
                                   {
                    press(visualizer, Qt::Key_X);
                    QTimer::singleShot(300, &visualizer, [&]()
                                       { pressed = true; QApplication::quit(); }); }); }); });
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

        ASSERT_TRUE(pressed.load());
        const std::filesystem::path frames = output.path / "screenshots" / "sample_0";
        EXPECT_TRUE(std::filesystem::exists(frames / "0.png"));
        EXPECT_TRUE(std::filesystem::exists(frames / "1.png"));
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(frames), std::filesystem::directory_iterator()), 2);
    }

    // With no output path (the demos' default), screenshots go under the working directory instead of the filesystem root.
    HOST_TEST(ScreenshotWithoutAnOutputPathWritesUnderTheWorkingDirectory)
    {
        ScratchFolder working("spacetime_screenshot_cwd_test");
        const std::filesystem::path previous = std::filesystem::current_path();
        std::filesystem::current_path(working.path);

        std::atomic<bool> pressed{false};
        Host vis(VisualizerOptions(), "", [&](Visualizer &visualizer)
                             { screenshotOnce(visualizer, pressed); });
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();
        std::filesystem::current_path(previous);

        ASSERT_TRUE(pressed.load());
        EXPECT_TRUE(std::filesystem::exists(working.path / "screenshots" / "sample_0" / "0.png"));
    }

    // A screenshot that cannot be written is reported instead of crashing the visualizer.
    HOST_TEST(UnwritableOutputPathDoesNotCrash)
    {
        // A regular file where the output folder should be: its screenshots/ subfolder cannot be created.
        ScratchFolder scratch("spacetime_screenshot_blocker_test");
        const std::filesystem::path blocker = scratch.path / "blocker";
        std::ofstream(blocker) << "not a folder";

        std::atomic<bool> pressed{false};
        Host vis(VisualizerOptions(), blocker.string(), [&](Visualizer &visualizer)
                             { screenshotOnce(visualizer, pressed); });
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

        EXPECT_TRUE(pressed.load());
        EXPECT_TRUE(std::filesystem::is_regular_file(blocker));
    }

    // Info messages ("Recording video...", "Screenshot saved", ...) are left out of exported frames, but stay on screen.
    HOST_TEST(ExportedFramesLeaveOutTheInfoMessage)
    {
        std::atomic<bool> measured{false};
        bool identical = false, still_shown = false;
        Host vis(VisualizerOptions(), "", [&](Visualizer &visualizer)
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
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

        ASSERT_TRUE(measured.load());
        EXPECT_TRUE(identical);
        EXPECT_TRUE(still_shown);
    }

    // V (offline) steps through every estimation tick of the range and encodes one frame per tick, at 1 / period fps.
    HOST_TEST(VideoKeyExportsOneFramePerEstimationTick)
    {
        if (!Spacetime::Tests::onPath("ffmpeg") || !Spacetime::Tests::onPath("ffprobe"))
            GTEST_SKIP() << "ffmpeg/ffprobe not on PATH";
        ScratchFolder output("spacetime_video_test");
        VisualizerOptions options;
        options.estimation_period = 0.1;
        std::atomic<bool> started{false}, finished{false};
        Host vis(options, output.path.string(), [&](Visualizer &visualizer)
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
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

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
    HOST_TEST(VideoKeyIsOfflineOnly)
    {
        for (const bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            options.estimation_period = 0.1;
            bool bound = online;
            QString text;
            Host vis(options, "", [&](Visualizer &visualizer)
                                 {
                bound = isBound(visualizer, Qt::Key_V);
                text = qobject_cast<QLabel *>(visualizer.findChild<InfoWidget *>()->widget())->text(); });
            vis.open();
            vis.requestClose();
            vis.run();

            EXPECT_EQ(bound, !online);
            EXPECT_EQ(text.contains("Record video [V]: 0"), !online) << text.toStdString();
            EXPECT_TRUE(text.contains("Screenshot [x]")) << text.toStdString();
        }
    }

    // Offline, the arrow keys step between estimation ticks; a live stream has nothing to step through, so they are unbound online.
    HOST_TEST(ArrowKeysStepBetweenEstimationTicksOfflineOnly)
    {
        for (const bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            options.estimation_period = 0.1;
            std::atomic<bool> measured{false};
            bool left = online, right = online;
            double after_right = -1.0, after_left = -1.0;
            Host vis(options, "", [&](Visualizer &visualizer)
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
            vis.open();
            vis.beginPlayback(0.0, 1.0);
            vis.run();

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
    HOST_TEST(TimelineSeeksOfflineOnly)
    {
        for (const bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            options.show_timeline_widget = true;
            std::atomic<bool> measured{false};
            bool ignores_mouse = !online;
            double before = -1.0, after = -1.0;
            Host vis(options, "", [&](Visualizer &visualizer)
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
            vis.open();
            vis.beginPlayback(0.0, 2.0);
            vis.run();

            ASSERT_TRUE(measured.load());
            EXPECT_EQ(ignores_mouse, online);
            if (online)
                EXPECT_EQ(after, before);
            else
                EXPECT_NEAR(after, 1.5, 1e-9); // 750 / 1000 of [0, 2]
        }
    }

    // With a sliding window, the timeline highlights [t - window, t] behind the current time t.
    HOST_TEST(TimelineHighlightsTheWindowBehindTheCurrentTime)
    {
        VisualizerOptions options;
        options.show_timeline_widget = true;
        options.estimator_window = 2.0;
        std::atomic<bool> measured{false};
        QColor in_window, before_window, after_time;
        Host vis(options, "", [&](Visualizer &visualizer)
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
        vis.open();
        vis.beginPlayback(0.0, 10.0);
        vis.run();

        ASSERT_TRUE(measured.load());
        EXPECT_EQ(in_window, QColor(255, 120, 120));
        EXPECT_EQ(before_window, QColor(255, 201, 201));
        EXPECT_NE(after_time, QColor(255, 120, 120));
        EXPECT_NE(after_time, QColor(255, 201, 201));
    }

    HOST_TEST(ArrowKeysStepAPercentOfTheRangeWithoutAPeriod)
    {
        std::atomic<bool> measured{false};
        double after_right = -1.0;
        Host vis(VisualizerOptions(), "", [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                visualizer.playback().pause();
                visualizer.playback().seek(2.0);
                press(visualizer, Qt::Key_Right);
                after_right = visualizer.playback().displayedTime();
                measured = true;
                QApplication::quit(); }); });
        vis.open();
        vis.beginPlayback(0.0, 10.0);
        vis.run();

        ASSERT_TRUE(measured.load());
        EXPECT_NEAR(after_right, 2.1, 1e-9);
    }

    // Without the estimation period the ticks are unknown: V writes nothing (and does not crash).
    HOST_TEST(VideoWithoutAPeriodWritesNothing)
    {
        ScratchFolder output("spacetime_video_no_period_test");
        std::atomic<bool> pressed{false};
        bool exporting = true;
        Host vis(VisualizerOptions(), output.path.string(), [&](Visualizer &visualizer)
                             { QTimer::singleShot(50, &visualizer, [&]()
                                                  {
                press(visualizer, Qt::Key_V);
                exporting = visualizer.exportingVideo();
                pressed = true;
                QApplication::quit(); }); });
        vis.open();
        vis.beginPlayback(0.0, 1.0);
        vis.run();

        ASSERT_TRUE(pressed.load());
        EXPECT_FALSE(exporting);
        EXPECT_FALSE(std::filesystem::exists(output.path / "videos"));
    }

    // Many instances of one view (e.g. one robot view per robot) describe their shared controls once.
    HOST_TEST(RepeatedControlInfoIsListedOnce)
    {
        QString text;
        Host vis(VisualizerOptions(), "", [&](Visualizer &visualizer)
                             {
            for (int i = 0; i < 3; i++)
                visualizer.addControlInfo("Adjust things [[, ]]");
            visualizer.addControlInfo("Other control [o]");
            text = qobject_cast<QLabel *>(visualizer.findChild<InfoWidget *>()->widget())->text(); });
        vis.open();
        vis.requestClose();
        vis.run();

        EXPECT_EQ(text.count("Adjust things [[, ]]"), 1);
        EXPECT_EQ(text.count("Other control [o]"), 1);
    }

    // A light kit: every light moves with the camera, one of them a headlight, so whichever side
    // of the scene the camera looks at is lit.
    HOST_TEST(SceneIsLitFromTheCamera)
    {
        int lights = 0, headlights = 0, camera_lights = 0;
        Host vis(VisualizerOptions(), "", [&](Visualizer &visualizer)
                             {
            vtkLightCollection *collection = visualizer.getRenderer()->GetLights();
            collection->InitTraversal();
            while (vtkLight *light = collection->GetNextItem())
            {
                lights++;
                headlights += light->LightTypeIsHeadlight() ? 1 : 0;
                camera_lights += (light->LightTypeIsHeadlight() || light->LightTypeIsCameraLight()) ? 1 : 0;
            } });
        vis.open();
        vis.requestClose();
        vis.run();

        EXPECT_GT(lights, 1);
        EXPECT_EQ(headlights, 1);
        EXPECT_EQ(camera_lights, lights) << "no light stays fixed in the world";
    }

    // L loops an offline replay (off by default); a live stream has nothing to loop.
    HOST_TEST(ReplayLoopIsOfflineOnlyAndStartsOff)
    {
        for (bool online : {false, true})
        {
            VisualizerOptions options;
            options.online = online;
            QString text;
            Host vis(options, "", [&](Visualizer &visualizer)
                                 { text = qobject_cast<QLabel *>(visualizer.findChild<InfoWidget *>()->widget())->text(); });
            vis.open();
            vis.requestClose();
            vis.run();

            if (online)
                EXPECT_FALSE(text.contains("Loop replay")) << text.toStdString();
            else
                EXPECT_TRUE(text.contains("Loop replay [L]: 0")) << text.toStdString();
        }
    }
    // A session running the window (online, so both run at once) keeps its estimator thread
    // above the GUI thread, in either mode.
    HOST_TEST(EstimatorThreadStaysAboveTheGuiThread)
    {
        using Pose = gtsam::Pose3;
        using ValueType = Priors::WNOA::NodeValues<Pose>;
        auto state = std::make_shared<Types::ContinuumRobotState<Priors::WNOA::Node<Pose>>>(1.0, 2);
        auto prior = std::make_shared<Priors::WNOA::Prior<Pose>>(state, ValueType());
        Estimator estimator(Testing::estimatorOptions(0.35));
        estimator.addPrior(prior);
        auto sensor = std::make_shared<Sensors::PoseSensor<ValueType>>(prior);
        estimator.addSensor(sensor);
        for (int i = 0; i <= 20; i++)
            sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(0.0, 0.05 * i, Eigen::Matrix4d::Identity().eval(),
                                                                            (Eigen::Matrix<double, 6, 6>::Identity() * 0.01).eval(),
                                                                            Eigen::MatrixXi::Ones(6, 1).eval()));

        Interface::SessionOptions session_options;
        session_options.start_time = 0.0;
        session_options.end_time = 0.6;
        session_options.period = 0.05;
        session_options.real_time = true;
        Interface::Session<> session(session_options);
        std::atomic<int> estimator_priority{INT_MIN}, gui_priority{INT_MAX};
        session.setTickCallback([&](double)
                                { estimator_priority = currentThreadPriority(); });

        VisualizerOptions options;
        options.online = true;
        Host vis(options, "", [&](Visualizer &visualizer)
                 {
            auto *poll = new QTimer(&visualizer);
            QObject::connect(poll, &QTimer::timeout, &visualizer, [&]()
                             {
                gui_priority = currentThreadPriority();
                if (session.finished())
                    QApplication::quit(); });
            poll->start(20);
            QTimer::singleShot(10000, &visualizer, []()
                               { QApplication::quit(); }); });
        session.start(estimator, vis);
        session.join();

        ASSERT_NE(estimator_priority.load(), INT_MIN) << "no estimate";
        ASSERT_NE(gui_priority.load(), INT_MAX) << "the window never ran";
        EXPECT_GT(estimator_priority.load(), gui_priority.load());
        EXPECT_TRUE(session.finished());
    }

    std::string modeName(const ::testing::TestParamInfo<HostMode> &info)
    {
        return info.param == HostMode::WorkerThread ? "WorkerThread" : "CallingThread";
    }

#if defined(__APPLE__)
    // AppKit only runs on the main thread: no worker-thread GUI.
    INSTANTIATE_TEST_SUITE_P(Modes, VisualizerHostTest, ::testing::Values(HostMode::CallingThread), modeName);
#else
    INSTANTIATE_TEST_SUITE_P(Modes, VisualizerHostTest, ::testing::Values(HostMode::WorkerThread, HostMode::CallingThread), modeName);
#endif
} // namespace Spacetime::Visualization::Default
