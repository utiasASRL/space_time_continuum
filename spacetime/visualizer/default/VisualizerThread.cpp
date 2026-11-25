#include <QApplication>
#include <QMetaObject>

#include "VisualizerThread.hpp"
#include "Visualizer.hpp"
#include <spacetime/interface/standalone/ThreadPriority.hpp>

namespace Spacetime::Visualization::Default
{
    VisualizerThread::VisualizerThread(int &argc, char **argv, VisualizerOptions options, std::string output_path, BuildFn build,
                                       std::function<void()> on_closed, int nice_delta)
        : m_argc(argc), m_argv(argv), m_options(std::move(options)), m_output_path(std::move(output_path)), m_build(std::move(build)),
          m_on_closed(std::move(on_closed)), m_nice_delta(nice_delta)
    {
        m_thread = std::thread([this]()
                               { threadMain(); });
    }

    VisualizerThread::~VisualizerThread()
    {
        requestQuit();
        join();
    }

    void VisualizerThread::waitReady()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_changed.wait(lock, [this]()
                       { return m_ready; });
        if (m_build_error)
            std::rethrow_exception(m_build_error);
    }

    void VisualizerThread::beginPlayback(double start_time, double end_time, std::function<double()> live_time_source)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_playback = Playback{start_time, end_time, std::move(live_time_source)};
        }
        m_changed.notify_all();
    }

    void VisualizerThread::requestQuit()
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

    int VisualizerThread::join()
    {
        if (m_thread.joinable())
            m_thread.join();
        return m_exit_code;
    }

    void VisualizerThread::threadMain()
    {
        Interface::Standalone::lowerCurrentThreadPriority(m_nice_delta);
        try
        {
            QApplication app(m_argc, m_argv);
            Visualizer visualizer(m_options, m_output_path);
            m_build(visualizer);

            std::optional<Playback> playback;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_ready = true;
                mp_app = &app;
                m_changed.notify_all();
                m_changed.wait(lock, [this]()
                               { return m_quit_requested || m_playback; });
                if (!m_quit_requested)
                    playback = m_playback;
            }

            if (playback)
            {
                visualizer.setTimeRange(playback->start_time, playback->end_time);
                if (playback->live_time_source)
                    visualizer.setLiveTimeSource(std::move(playback->live_time_source));
                visualizer.show();
                const int exit_code = app.exec();
                std::lock_guard<std::mutex> lock(m_mutex);
                m_exit_code = exit_code;
            }

            std::lock_guard<std::mutex> lock(m_mutex);
            mp_app = nullptr; // the application is about to be destroyed
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_ready)
            {
                m_build_error = std::current_exception();
                m_ready = true;
            }
            mp_app = nullptr;
            m_changed.notify_all();
        }

        if (m_on_closed)
            m_on_closed();
    }
} // namespace Spacetime::Visualization::Default
