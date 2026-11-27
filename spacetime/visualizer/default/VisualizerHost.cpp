#include <stdexcept>

#include <QApplication>
#include <QMetaObject>

#if defined(__APPLE__)
#include <pthread.h>
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>
#endif

#include "VisualizerHost.hpp"
#include "Visualizer.hpp"
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/interface/standalone/ThreadPriority.hpp>

namespace Spacetime::Visualization::Default
{
    HostMode defaultHostMode()
    {
#if defined(__APPLE__) || defined(_WIN32)
        return HostMode::CallingThread;
#else
        return HostMode::WorkerThread;
#endif
    }

    VisualizerHost::VisualizerHost(int &argc, char **argv, VisualizerOptions options, std::string output_path, BuildFn build,
                                   HostMode mode, int nice_delta)
        : m_argc(argc), m_argv(argv), m_options(std::move(options)), m_output_path(std::move(output_path)), m_build(std::move(build)),
          m_mode(mode), m_nice_delta(nice_delta) {}

    VisualizerHost::~VisualizerHost()
    {
        requestClose();
        if (m_thread.joinable())
            m_thread.join();
        if (m_mode == HostMode::CallingThread)
            teardown(); // no-op unless opened and never run
    }

    void VisualizerHost::open(ClosedCallback on_closed)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_opened)
                throw std::logic_error("VisualizerHost::open: already opened.");
#if defined(__APPLE__)
            // AppKit only runs on the process's main thread.
            if (m_mode == HostMode::WorkerThread)
                throw std::logic_error("VisualizerHost::open: macOS only allows a GUI on the main thread: use HostMode::CallingThread.");
            if (pthread_main_np() == 0)
                throw std::logic_error("VisualizerHost::open: on macOS the visualizer must run on the main thread "
                                       "(start the session with the visualizer, and join it, from the main thread).");
#endif
            m_opened = true;
            m_on_closed = std::move(on_closed);
        }

        if (m_mode == HostMode::WorkerThread)
        {
            m_thread = std::thread([this]()
                                   { workerMain(); });
            std::exception_ptr error;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_changed.wait(lock, [this]()
                               { return m_ready; });
                error = m_build_error;
            }
            if (error)
            {
                m_thread.join(); // it tears itself down
                std::rethrow_exception(error);
            }
            return;
        }

        construct();
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            error = m_build_error;
        }
        if (error)
        {
            teardown();
            std::rethrow_exception(error);
        }
    }

    void VisualizerHost::open(Estimator &, ClosedCallback on_closed)
    {
        open(std::move(on_closed));
    }

    bool VisualizerHost::isOnline() const
    {
        return m_options.online;
    }

    void VisualizerHost::beginPlayback(double start_time, double end_time, LiveTimeSource live_time_source)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_playback = Playback{start_time, end_time, std::move(live_time_source)};
        }
        m_changed.notify_all();
    }

    void VisualizerHost::requestClose()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_quit_requested = true;
            // Queued, so it also lands if the event loop has not started yet: QCoreApplication::quit()
            // does nothing outside a running loop, but a posted call runs once exec() does.
            if (mp_app)
                QMetaObject::invokeMethod(mp_app, []()
                                          { QCoreApplication::quit(); }, Qt::QueuedConnection);
        }
        m_changed.notify_all();
    }

    int VisualizerHost::run()
    {
        if (m_mode == HostMode::WorkerThread)
        {
            if (m_thread.joinable())
                m_thread.join();
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_exit_code;
        }

        std::optional<Playback> playback;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_opened)
                throw std::logic_error("VisualizerHost::run: call open() first.");
            if (m_done)
                return m_exit_code;
            if (!m_quit_requested)
            {
                if (!m_playback)
                    throw std::logic_error("VisualizerHost::run: call beginPlayback() (or requestClose()) first.");
                playback = m_playback;
            }
        }

        int exit_code = 0;
        if (playback)
        {
            // Lowered only while the window runs: the session's estimator thread exists by now.
            Interface::Standalone::ScopedLowerThreadPriority lowered(m_nice_delta);
            exit_code = runEventLoop(std::move(*playback));
            if (!lowered.restore())
                warning() << "VisualizerHost: could not restore this thread's priority after the window closed (on Linux that "
                             "needs CAP_SYS_NICE or RLIMIT_NICE), so threads it creates from now on (e.g. a later session's "
                             "estimator) start lowered too. HostMode::WorkerThread, Linux's default, avoids this.";
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_exit_code = exit_code;
        }
        teardown();
        return exit_code;
    }

    void VisualizerHost::construct()
    {
        try
        {
#if defined(__APPLE__)
            // VTK renders through an OpenGL 3.2 core profile, which macOS only provides when asked for
            // up front (elsewhere the default context already offers it).
            QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
#endif
            mp_application = std::make_unique<QApplication>(m_argc, m_argv);
            mp_visualizer = std::make_unique<Visualizer>(m_options, m_output_path);
            m_build(*mp_visualizer);

            std::lock_guard<std::mutex> lock(m_mutex);
            m_ready = true;
            mp_app = mp_application.get();
            if (m_quit_requested) // asked before mp_app could receive it
                QMetaObject::invokeMethod(mp_app, []()
                                          { QCoreApplication::quit(); }, Qt::QueuedConnection);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_build_error = std::current_exception();
            m_ready = true;
        }
        m_changed.notify_all();
    }

    int VisualizerHost::runEventLoop(Playback playback)
    {
        mp_visualizer->setTimeRange(playback.start_time, playback.end_time);
        if (playback.live_time_source)
            mp_visualizer->setLiveTimeSource(std::move(playback.live_time_source));
        mp_visualizer->show();
        return mp_application->exec();
    }

    void VisualizerHost::teardown()
    {
        ClosedCallback on_closed;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_opened || m_done)
                return;
            m_done = true;
            mp_app = nullptr; // the application is about to be destroyed
            on_closed = std::move(m_on_closed);
        }
        m_changed.notify_all();
        mp_visualizer.reset();
        mp_application.reset();
        if (on_closed)
            on_closed();
    }

    void VisualizerHost::workerMain()
    {
        Interface::Standalone::lowerCurrentThreadPriority(m_nice_delta);
        construct();

        std::optional<Playback> playback;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_build_error)
            {
                m_changed.wait(lock, [this]()
                               { return m_quit_requested || m_playback; });
                if (!m_quit_requested)
                    playback = m_playback;
            }
        }

        if (playback)
        {
            const int exit_code = runEventLoop(std::move(*playback));
            std::lock_guard<std::mutex> lock(m_mutex);
            m_exit_code = exit_code;
        }
        teardown();
    }
} // namespace Spacetime::Visualization::Default
