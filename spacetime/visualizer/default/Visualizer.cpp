#include <filesystem>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

#include <QShortcut>
#include <QApplication>
#include <QResizeEvent>
#include <QPixmap>

#include <vtkLightKit.h>
#include <vtkCamera.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkImageData.h>

#include "Visualizer.hpp"
#include "objects/GroundPlane.hpp"
#include "utilities.hpp"
#include <spacetime/visualizer/BackendConcept.hpp>

namespace Spacetime::Visualization::Default
{
    Visualizer::Visualizer(VisualizerOptions options, std::string output_path)
        : QVTKOpenGLNativeWidget(), m_options(options), m_playback(options.online), m_output_path(output_path)
    {
        vtkNew<vtkGenericOpenGLRenderWindow> window;
        this->setRenderWindow(window);
        mp_renWin = this->renderWindow();

        mp_ren = vtkSmartPointer<vtkRenderer>::New();
        mp_renWin->AddRenderer(mp_ren);
        this->setGeometry(100, 100, 1280, 720);

        m_infoLabel = new QLabel(this);
        m_infoLabel->setObjectName("info_label");
        m_infoLabel->setAlignment(Qt::AlignCenter);
        m_infoLabel->setStyleSheet("QLabel { background-color : white; color : black; font-size: 16px; font-weight: bold; }");
        m_infoLabel->setGeometry(this->width() / 2 - 150, 10, 300, 50);
        m_infoLabel->setVisible(false);

        m_timelineWidget = new TimelineWidget(this);
        m_timelineWidget->setVisible(m_options.show_timeline_widget);

        m_diagnosticsLabel = new InfoWidget(this);
        m_diagnosticsLabel->addHeader("Information");
        m_diagnosticsLabel->addLabel("Mode", [this]()
                                     { return std::string(m_options.online ? "online" : "offline"); });
        m_diagnosticsLabel->addLabel("Time", [this]()
                                     { std::ostringstream oss; oss << m_playback.displayedTime(); return oss.str(); });
        m_diagnosticsLabel->addLabel("Delay [,, .]", [this]()
                                     { std::ostringstream oss; oss << m_playback.delay(); return oss.str(); });
        m_diagnosticsLabel->addLabel("Shown time", [this]()
                                     { std::ostringstream oss; oss << m_playback.queryTime(); return oss.str(); });
        m_diagnosticsLabel->addLabel("Speed [\xe2\x86\x91, \xe2\x86\x93]", [this]()
                                     { std::ostringstream oss; oss << m_playback.speed(); return oss.str(); });
        m_diagnosticsLabel->addLabel("Playing [Space]", [this]()
                                     { return std::string(m_playback.state() == PlaybackState::Playing ? "1" : "0"); });
        m_diagnosticsLabel->addLabel("Update [ms]", [this]()
                                     { std::ostringstream oss; oss << std::fixed << std::setprecision(1) << m_update_seconds * 1e3; return oss.str(); });
        m_diagnosticsLabel->addLabel("Render [ms]", [this]()
                                     { std::ostringstream oss; oss << std::fixed << std::setprecision(1) << m_render_seconds * 1e3; return oss.str(); });
        m_diagnosticsLabel->addLabel("Camera position", [this]()
                                     {
            double pos[3];
            this->mp_ren->GetActiveCamera()->GetPosition(pos);
            std::ostringstream oss;
            oss << "<br>" << pos[0] << ",<br>" << pos[1] << ",<br>" << pos[2];
            return oss.str(); });
        m_diagnosticsLabel->addLabel("Camera focal", [this]()
                                     {
            double fp[3];
            this->mp_ren->GetActiveCamera()->GetFocalPoint(fp);
            std::ostringstream oss;
            oss << "<br>" << fp[0] << ",<br>" << fp[1] << ",<br>" << fp[2];
            return oss.str(); });
        m_diagnosticsLabel->addHeader("Controls");
        m_diagnosticsLabel->addLabel("Quit [q]");
        m_diagnosticsLabel->addLabel("Screenshot [x]");
        m_diagnosticsLabel->addLabel("Reset camera [r]");
        m_diagnosticsLabel->addLabel("Toggle diagnostics [h]");
        m_diagnosticsLabel->addLabel("Toggle timeline [y]", [this]()
                                     { return std::string(m_timelineWidget->isVisible() ? "1" : "0"); });
        // A fixed, window-bounded viewport: content that grows past it (e.g. as
        // more Visualizables register their own keybindings) scrolls instead of
        // being cropped or endlessly growing the panel itself.
        m_diagnosticsLabel->setGeometry(10, 10, 340, diagnosticsLabelHeight());
        m_diagnosticsLabel->setVisible(m_options.show_diagnostics_label);

        m_animation_timer.setInterval(33); // ~30 FPS
        connect(&m_animation_timer, &QTimer::timeout, this, &Visualizer::slotAnimationTimer);
        m_last_tick_time = std::chrono::steady_clock::now(); // otherwise the first tick's wall_dt is measured against the epoch (e.g. machine uptime), instantly blowing displayedTime() far past any valid range
        m_animation_timer.start();

        m_infoTimer.setSingleShot(true);
        connect(&m_infoTimer, &QTimer::timeout, this, &Visualizer::hideMessage);

        // Playback controls
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Space), this), &QShortcut::activated, [this]()
                         {
            if (exportingVideo()) return; // the export steps through the ticks itself
            if (m_playback.state() == PlaybackState::Playing) m_playback.pause();
            else m_playback.play(); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Up), this), &QShortcut::activated, [this]()
                         { m_playback.setSpeed(m_playback.speed() / 1.5); });
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Down), this), &QShortcut::activated, [this]()
                         { m_playback.setSpeed(m_playback.speed() * 1.5); });

        // Stepping through a replay (a live stream has nothing to step through).
        if (!m_options.online)
        {
            QObject::connect(new QShortcut(QKeySequence(Qt::Key_Left), this), &QShortcut::activated, [this]()
                             { stepTime(-1); });
            QObject::connect(new QShortcut(QKeySequence(Qt::Key_Right), this), &QShortcut::activated, [this]()
                             { stepTime(+1); });
        }

        // Phase delay: draw objects at (playback time - delay) to inspect earlier points in the estimator window
        if (m_options.estimator_window > 0)
            m_playback.setMaxDelay(m_options.estimator_window);
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Comma), this), &QShortcut::activated, [this]()
                         {
            m_playback.setDelay(m_playback.delay() - delayStep());
            update(m_playback.displayedTime()); });
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Period), this), &QShortcut::activated, [this]()
                         {
            m_playback.setDelay(m_playback.delay() + delayStep());
            update(m_playback.displayedTime()); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_R), this), &QShortcut::activated, [this]()
                         { this->resetCameraView(); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_X), this), &QShortcut::activated, [this]()
                         { takeScreenshot(); });

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Q), this), &QShortcut::activated, []()
                         { QApplication::quit(); });

        // Bound through the key-binding registry so other objects (e.g. PlotWidget) can share H.
        bindShortcut(Qt::Key_H, [this]()
                     { m_diagnosticsLabel->setVisible(m_diagnosticsLabel->isHidden()); });
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Y), this), &QShortcut::activated, [this]()
                         { m_timelineWidget->setVisible(m_timelineWidget->isHidden()); });

        // Replaying from the start once the end is reached (a live stream has no end).
        if (!m_options.online)
            bindToggleShortcut(Qt::Key_L, [this]()
                               { m_playback.setLooping(!m_playback.looping()); }, [this]()
                               { return m_playback.looping(); }, "Loop replay");

        // Exporting every estimation tick to a video, then cancelling it (a live stream has no end).
        if (!m_options.online)
        {
            m_video_timer.setInterval(0);
            connect(&m_video_timer, &QTimer::timeout, this, &Visualizer::exportNextVideoFrame);
            bindToggleShortcut(Qt::Key_V, [this]()
                               { exportingVideo() ? finishVideoExport(true) : startVideoExport(); }, [this]()
                               { return exportingVideo(); }, "Record video");
        }

        bindToggleShortcut(Qt::Key_O, [this]()
                           { m_orbiting = !m_orbiting; }, [this]()
                           { return m_orbiting; }, "Orbit camera");

        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Equal), this), &QShortcut::activated, [this]()
                         { this->mp_ren->GetActiveCamera()->Zoom(1.1); this->mp_renWin->Render(); });
        QObject::connect(new QShortcut(QKeySequence(Qt::Key_Minus), this), &QShortcut::activated, [this]()
                         { this->mp_ren->GetActiveCamera()->Zoom(0.9); this->mp_renWin->Render(); });

        m_playback.play(); // default to playing in both modes

        initScene();
    }

    static_assert(VisualizerBackend<Visualizer>);

    Visualizer::~Visualizer() {}

    void Visualizer::setTimeRange(double start_time, double end_time)
    {
        m_time_range_start = start_time;
        m_time_range_end = end_time;
        m_timelineWidget->setRange(0, 1000, start_time, end_time);
        m_playback.setTimeBounds(start_time, end_time); // offline only; harmless no-op influence in online mode since tick()'s online branch never checks these bounds
        m_playback.seek(start_time); // offline only; no-op (ignored) in online mode
    }

    void Visualizer::sendMessage(QString message)
    {
        QMetaObject::invokeMethod(&m_infoTimer, "stop", Qt::QueuedConnection);
        m_infoLabel->setText(message);
        m_infoLabel->setVisible(true);
    }
    void Visualizer::sendMessage(QString message, int duration)
    {
        QMetaObject::invokeMethod(&m_infoTimer, "stop", Qt::QueuedConnection);
        m_infoLabel->setText(message);
        m_infoLabel->setVisible(true);
        QMetaObject::invokeMethod(&m_infoTimer, "start", Qt::QueuedConnection, Q_ARG(int, duration));
    }
    void Visualizer::hideMessage()
    {
        m_infoLabel->setVisible(false);
    }

    vtkSmartPointer<vtkPolyDataMapper> Visualizer::unitSphereMapper()
    {
        if (!mp_unit_sphere_mapper)
            mp_unit_sphere_mapper = Utilities::createUnitSphereMapper();
        return mp_unit_sphere_mapper;
    }

    vtkSmartPointer<vtkPolyDataMapper> Visualizer::unitCylinderMapper()
    {
        if (!mp_unit_cylinder_mapper)
            mp_unit_cylinder_mapper = Utilities::createUnitCylinderMapper();
        return mp_unit_cylinder_mapper;
    }

    vtkSmartPointer<vtkPolyDataMapper> Visualizer::frameMapper(double cylinderRadius, double totalLength)
    {
        auto &mapper = m_frame_mappers[{cylinderRadius, totalLength}];
        if (!mapper)
            mapper = Utilities::createFrameMapper(cylinderRadius, totalLength);
        return mapper;
    }

    void Visualizer::update(double time)
    {
        const auto smooth = [](double &average, std::chrono::steady_clock::time_point start)
        {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            average += kFrameTimeSmoothing * (seconds - average);
        };

        const auto update_start = std::chrono::steady_clock::now();
        const double query_time = time - m_playback.delay();
        for (const auto &fn : m_visualizables)
            fn(query_time, time);
        smooth(m_update_seconds, update_start);

        if (m_diagnosticsLabel)
            m_diagnosticsLabel->refresh();

        if (m_timelineWidget && m_time_range_end > m_time_range_start)
        {
            double fraction = std::clamp((time - m_time_range_start) / (m_time_range_end - m_time_range_start), 0.0, 1.0);
            m_timelineWidget->setValue(static_cast<int>(fraction * 1000));
        }

        const auto render_start = std::chrono::steady_clock::now();
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();
        smooth(m_render_seconds, render_start);
    }

    void Visualizer::slotAnimationTimer()
    {
        auto now = std::chrono::steady_clock::now();
        double wall_dt = std::chrono::duration<double>(now - m_last_tick_time).count();
        m_last_tick_time = now;
        if (exportingVideo())
            return; // exportNextVideoFrame drives the frames

        const double previous_time = m_playback.displayedTime();
        double displayed_time = m_playback.tick(wall_dt);
        if (m_orbiting) // follows playback: still while paused or at the end
            Utilities::orbitAboutVertical(mp_ren->GetActiveCamera(), kOrbitDegreesPerSecond * M_PI / 180.0 * std::max(0.0, displayed_time - previous_time));
        update(displayed_time);
    }

    void Visualizer::resizeEvent(QResizeEvent *event)
    {
        QVTKOpenGLNativeWidget::resizeEvent(event);
        m_infoLabel->setGeometry(this->width() / 2 - 150, 10, 300, 50);
        if (m_timelineWidget)
            m_timelineWidget->setGeometry(150, this->height() - bottomBarHeight() - 50, this->width() - 300, 50);
        if (m_diagnosticsLabel)
            m_diagnosticsLabel->setGeometry(10, 10, 340, diagnosticsLabelHeight());
        for (const auto &[widget, layout] : m_overlays)
            widget->setGeometry(layout(size()));
    }

    int Visualizer::diagnosticsLabelHeight() const
    {
        return std::max(100, std::min(500, this->height() - 20));
    }

    double Visualizer::delayStep() const
    {
        return 0.01; // m_options.estimator_window > 0 ? m_options.estimator_window / 10.0 : 0.05;
    }

    void Visualizer::initScene()
    {
        info() << "Visualizer::initScene()";
        mp_ren->SetBackground(1., 1., 1.);

        initLights();
        if (m_options.show_floor)
            addVisualizable(std::make_shared<GroundPlane>());
        if (m_options.show_world_frame)
            initWorldFrame();

        resetCameraView();

        mp_renWin->Render();
    }

    void Visualizer::initLights()
    {
        // Key, fill, back and head lights, all moving with the camera: whichever side of the
        // scene the camera looks at is lit, and the headlight keeps nearer surfaces readable.
        vtkNew<vtkLightKit> light_kit;
        light_kit->AddLightsToRenderer(mp_ren);
    }

    void Visualizer::initWorldFrame()
    {
        mp_ren->AddActor(Utilities::createFrameActor(0.05, 0.01));
    }

    void Visualizer::resetCameraView()
    {
        if (m_options.cameras.empty())
        {
            mp_ren->ResetCamera();
            mp_renWin->Render();
            return;
        }
        mp_ren->GetActiveCamera()->SetPosition(m_options.cameras[0].position(0), m_options.cameras[0].position(1), m_options.cameras[0].position(2));
        mp_ren->GetActiveCamera()->SetFocalPoint(m_options.cameras[0].focal_point(0), m_options.cameras[0].focal_point(1), m_options.cameras[0].focal_point(2));
        mp_ren->GetActiveCamera()->SetViewUp(m_options.cameras[0].view_up(0), m_options.cameras[0].view_up(1), m_options.cameras[0].view_up(2));
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();
    }

    void Visualizer::screenshot()
    {
        const std::filesystem::path file_path = screenshotsFolder() / ("sample_" + std::to_string(m_sample_index)) /
                                                (std::to_string(m_screenshot_frame_idx) + ".png");
        writeFrame(file_path);

        if (m_verbose)
            info() << "Screenshot saved for current view at " << file_path;

        for (size_t i = 1; i < m_options.cameras.size(); i++)
            screenshotFromCamera(m_options.cameras[i], "camera_" + std::to_string(i));

        m_screenshot_frame_idx++;
    }

    void Visualizer::screenshotFromCamera(const VisualizerOptions::CameraOptions &camera_config, const std::string &subfolder_name)
    {
        double current_pos[3], current_focal[3], current_up[3];
        mp_ren->GetActiveCamera()->GetPosition(current_pos);
        mp_ren->GetActiveCamera()->GetFocalPoint(current_focal);
        mp_ren->GetActiveCamera()->GetViewUp(current_up);

        mp_ren->GetActiveCamera()->SetPosition(camera_config.position(0), camera_config.position(1), camera_config.position(2));
        mp_ren->GetActiveCamera()->SetFocalPoint(camera_config.focal_point(0), camera_config.focal_point(1), camera_config.focal_point(2));
        mp_ren->GetActiveCamera()->SetViewUp(camera_config.view_up(0), camera_config.view_up(1), camera_config.view_up(2));
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();

        const std::filesystem::path file_path = screenshotsFolder() / subfolder_name /
                                                (std::to_string(m_screenshot_frame_idx) + ".png");
        writeFrame(file_path);

        mp_ren->GetActiveCamera()->SetPosition(current_pos);
        mp_ren->GetActiveCamera()->SetFocalPoint(current_focal);
        mp_ren->GetActiveCamera()->SetViewUp(current_up);
        mp_ren->ResetCameraClippingRange();
        mp_renWin->Render();

        if (m_verbose)
            info() << "Screenshot saved for camera: " << subfolder_name << " at " << file_path;
    }

    std::filesystem::path Visualizer::screenshotsFolder() const
    {
        // No output path (the demos' default) means the working directory, not the filesystem root.
        return (m_output_path.empty() ? std::filesystem::path(".") : std::filesystem::path(m_output_path)) / "screenshots";
    }

    std::filesystem::path Visualizer::videosFolder() const
    {
        return screenshotsFolder().parent_path() / "videos";
    }

    QImage Visualizer::grabFrame()
    {
        // The info message (e.g. "Recording video...") is for the viewer, not the exported frame.
        const bool message_shown = m_infoLabel->isVisible();
        m_infoLabel->setVisible(false);
        QImage frame = this->grab().toImage();
        m_infoLabel->setVisible(message_shown);
        return frame;
    }

    void Visualizer::writeFrame(const std::filesystem::path &file_path)
    {
        std::filesystem::create_directories(file_path.parent_path());

        QImage image = grabFrame().convertToFormat(QImage::Format_RGBA8888);
        QImage flippedImage = image.mirrored(false, true); // Flip vertically to correct VTK coordinate system

        vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
        imageData->SetDimensions(flippedImage.width(), flippedImage.height(), 1);
        imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 4);

        unsigned char *vtkPixels = static_cast<unsigned char *>(imageData->GetScalarPointer());
        memcpy(vtkPixels, flippedImage.constBits(), flippedImage.sizeInBytes());

        m_screenshot_writer->SetInputData(imageData);
        m_screenshot_writer->SetFileName(file_path.c_str());
        m_screenshot_writer->Write();
    }

    void Visualizer::stepTime(int direction)
    {
        if (exportingVideo())
            return; // the export steps through the ticks itself
        if (m_options.estimation_period > 0.0)
            m_playback.stepTick(m_options.estimation_period, direction);
        else // ticks unknown: 1% of the range
            m_playback.seek(m_playback.displayedTime() + direction * std::max(0.01, (m_time_range_end - m_time_range_start) * 0.01));
        update(m_playback.displayedTime());
    }

    void Visualizer::takeScreenshot()
    {
        try
        {
            screenshot();
            sendMessage("Screenshot saved", 1000);
        }
        catch (const std::exception &e)
        {
            error() << "Screenshot failed: " << e.what();
            sendMessage("Screenshot failed: could not write the file", 3000);
        }
    }

    void Visualizer::startVideoExport()
    {
        const double period = m_options.estimation_period;
        if (!(period > 0.0) || !(m_time_range_end >= m_time_range_start))
        {
            warning() << "Video export needs the estimation period (VisualizerOptions::estimation_period) and a time range.";
            sendMessage("Video export unavailable: the estimation period is unknown", 3000);
            return;
        }
        const std::filesystem::path path = videosFolder() / ("video_" + std::to_string(m_video_index) + ".mp4");
        try
        {
            m_video = std::make_unique<VideoWriter>(path, size() * devicePixelRatio(), 1.0 / period);
        }
        catch (const std::exception &e)
        {
            error() << "Video export failed: " << e.what();
            sendMessage("Video export failed: is ffmpeg installed?", 3000);
            return;
        }
        m_video_index++;
        m_video_tick = 0;
        m_playback.pause();
        sendMessage("Recording video...");
        if (m_verbose)
            info() << "Exporting the estimation ticks to " << path;
        m_video_timer.start();
    }

    void Visualizer::exportNextVideoFrame()
    {
        if (!m_video)
            return;
        // The k-th tick, computed (not accumulated) so rounding does not drift; the last one may sit a rounding error past the end.
        const double time = m_time_range_start + static_cast<double>(m_video_tick) * m_options.estimation_period;
        if (time > m_time_range_end + 1e-9 * std::max(1.0, std::abs(m_time_range_end)))
        {
            finishVideoExport(false);
            return;
        }
        m_playback.seek(time);
        update(m_playback.displayedTime());
        try
        {
            m_video->write(grabFrame());
        }
        catch (const std::exception &e)
        {
            abortVideoExport(e.what());
            return;
        }
        m_video_tick++;
    }

    void Visualizer::finishVideoExport(bool cancelled)
    {
        m_video_timer.stop();
        if (!m_video)
            return;
        // Released first, so the export is over even if finishing throws.
        const std::unique_ptr<VideoWriter> video = std::move(m_video);
        try
        {
            video->finish();
        }
        catch (const std::exception &e)
        {
            error() << "Video export failed: " << e.what();
            sendMessage("Video export failed", 3000);
            return;
        }
        const std::string summary = video->path().string() + " (" + std::to_string(video->frames()) + " frames)";
        if (m_verbose)
            info() << (cancelled ? "Video export cancelled; saved " : "Video saved: ") << summary;
        sendMessage(QString::fromStdString((cancelled ? "Video cancelled, saved: " : "Video saved: ") + summary), 4000);
    }

    void Visualizer::abortVideoExport(const std::string &what)
    {
        m_video_timer.stop();
        m_video.reset(); // finishes the partial file, logging any error
        error() << "Video export failed: " << what;
        sendMessage("Video export failed", 3000);
    }
} // namespace Spacetime::Visualization::Default
