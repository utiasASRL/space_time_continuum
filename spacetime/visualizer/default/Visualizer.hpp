#pragma once

#include <chrono>
#include <filesystem>
#include <sstream>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include <concepts>
#include <iostream>
#include <map>
#include <set>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <QObject>
#include <QRect>
#include <QSize>
#include <QWidget>
#include <QTimer>
#include <QString>
#include <QImage>
#include <QLabel>
#include <QShortcut>
#include <QKeySequence>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkAssembly.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkPNGWriter.h>
#include <vtkPolyDataMapper.h>

#include <spacetime/visualizer/PlaybackController.hpp>
#include <spacetime/visualizer/VisualizerOptions.hpp>
#include <spacetime/visualizer/default/KeyBindingRegistry.hpp>
#include <spacetime/visualizer/default/VideoWriter.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/objects/InfoWidget.hpp>
#include <spacetime/visualizer/default/objects/TimelineWidget.hpp>
// The state plot panel is a Visualizable of its own: see objects/PlotWidget.hpp.

namespace Spacetime::Visualization::Default
{
    class Visualizer;

    /// An object Visualizer::addVisualizable accepts: drawn once by visualize(), then updated every frame.
    template <typename T>
    concept IsVisualizable = requires(T a, Visualizer &visualizer, double time) {
        { a.visualize(visualizer) } -> std::same_as<void>;
        { a.updateVisualization(time, time) } -> std::same_as<void>;
    };

    /// An IsVisualizable that also reports a name and can toggle its own visibility.
    template <typename T>
    concept IsVisualizableWithToggle = IsVisualizable<T> && requires(T a) {
        { a.info(true) } -> std::same_as<std::string>;
        { a.toggleVisibility() } -> std::same_as<void>;
    };

    /**
     * @brief The default (Qt/VTK) backend's visualizer: a render widget that, on a
     *        ~30 FPS timer, advances its PlaybackController and updates every added
     *        Visualizable, plus the playback/camera/screenshot/video shortcuts, the
     *        diagnostics panel and the timeline overlay. Starts playing.
     */
    class Visualizer : public QVTKOpenGLNativeWidget
    {
        Q_OBJECT

        /// Per-frame update callbacks of the added Visualizables, called as (query time, estimate time).
        std::vector<std::function<void(double, double)>> m_visualizables;

    public:
        /**
         * @param options     Display options; `options.online` fixes the playback mode.
         * @param output_path Root folder for recorded screenshots (written under `<output_path>/screenshots/`); empty means the working directory.
         */
        Visualizer(VisualizerOptions options, std::string output_path = "");
        /// Default destructor; child widgets are owned by Qt.
        ~Visualizer();
        /// Shows `message` in the top-centre info label until replaced or hideMessage() is called.
        void sendMessage(QString message);
        /// Shows `message` in the top-centre info label for `duration` ms.
        void sendMessage(QString message, int duration);
        /// Starts a new screenshot subfolder (`sample_<index>`) for subsequent recorded frames.
        void incrementSampleIndex() { m_sample_index++; }

        /// Establishes the range the timeline widget displays and (offline mode) that playback and seeking clamp to; offline, also seeks to `start_time`.
        void setTimeRange(double start_time, double end_time);

        /// Online only: the time PlaybackController tracks while playing (see PlaybackController::setLiveTimeSource). Must be set for online mode to show anything, otherwise the displayed time never leaves 0.
        void setLiveTimeSource(std::function<double()> now_fn) { m_playback.setLiveTimeSource(std::move(now_fn)); }

        /// Applies the configured stale timeout (for Visualization::Visualizable subclasses), calls `visualizable->visualize(*this)` and registers it for per-frame updates. Keeps it alive for the visualizer's lifetime.
        template <IsVisualizable T>
        void addVisualizable(std::shared_ptr<T> visualizable)
        {
            if constexpr (std::derived_from<T, Visualization::Visualizable>)
                visualizable->setStaleTimeout(m_options.stale_timeout);
            visualizable->visualize(*this);
            m_visualizables.push_back([visualizable](double time, double estimate_time)
                                      { visualizable->updateVisualization(time, estimate_time); });
        }

        /// Lets external code (e.g. a Visualizable's own visualize() override) add a matching line to the diagnostics/info panel.
        /// A label already added is not repeated, so every instance of a view can describe the controls they share.
        void addControlInfo(const std::string &label)
        {
            if (m_diagnosticsLabel && m_control_infos.insert(label).second)
                m_diagnosticsLabel->addLabel(label);
        }

        /**
         * @brief Registers a keyboard shortcut with no diagnostics-panel entry of
         *        its own (e.g. a second key for a control already described by
         *        another addControlInfo/bindToggleShortcut line). Meant to be
         *        called directly from a Visualizable's own visualize() override --
         *        see Default::Visualizable::bindVisibilityToggle for the pattern this
         *        exists to support -- so that key-binding "plumbing" lives with the
         *        object rather than in the script that constructs it. Binding
         *        the same key more than once runs every bound fn on each press
         *        (see KeyBindingRegistry for why this can't just be one
         *        QShortcut per call).
         */
        void bindShortcut(Qt::Key key, std::function<void()> fn)
        {
            if (m_key_bindings.addHandler(key, std::move(fn)))
                QObject::connect(new QShortcut(QKeySequence(key), this), &QShortcut::activated, [this, key]()
                                 { m_key_bindings.dispatch(key); });
        }

        /**
         * @brief Registers a keyboard shortcut for a boolean toggle plus a
         *        diagnostics-panel row showing the live value as 1/0. Toggles
         *        sharing a key (e.g. several sensors of one type) share that
         *        key's single row -- labelled by the first binding -- which
         *        reads 1 only while every one of them is on.
         */
        void bindToggleShortcut(Qt::Key key, std::function<void()> toggleFn, std::function<bool()> stateFn, const std::string &label)
        {
            bindShortcut(key, std::move(toggleFn));
            if (m_key_bindings.addToggleState(key, std::move(stateFn)) && m_diagnosticsLabel)
                m_diagnosticsLabel->addLabel(label + " [" + QKeySequence(key).toString().toStdString() + "]", [this, key]()
                                             { return std::string(m_key_bindings.allOn(key) ? "1" : "0"); });
        }

        /**
         * @brief Adds `widget` (a child of this visualizer) as an overlay whose geometry is
         *        `layout(window size)`, applied now and on every resize. Meant for a
         *        Visualizable's own visualize() override (e.g. PlotWidget's panel).
         */
        void addOverlayWidget(QWidget *widget, std::function<QRect(QSize)> layout)
        {
            widget->setGeometry(layout(size()));
            m_overlays.push_back({widget, std::move(layout)});
        }

        /// Rate the camera orbits at while orbiting is on (toggle O), per second of playback time (see slotAnimationTimer).
        static constexpr double kOrbitDegreesPerSecond = 20.0;

        /// Height (pixels) reserved along the bottom edge for the runtime bar (see RuntimeWidget); the timeline sits directly above it.
        static constexpr int bottomBarHeight() { return 25; }

        /// Whether the diagnostics panel is shown (toggled with H), so an overlay toggled by the same key can start in sync with it.
        bool diagnosticsVisible() const { return m_diagnosticsLabel && !m_diagnosticsLabel->isHidden(); }

        /// Whether the visualizer is tracking live estimator state (true) or scrubbing a fixed offline recording (false). Lets a Visualizable's own visualize() decide whether to expose controls that only make sense offline (e.g. temporal interpolation).
        bool isOnline() const { return m_options.online; }

        /// Adds a VTK actor (any type vtkRenderer::AddActor accepts) to the scene.
        template <typename T>
        void addActor(const T &actor)
        {
            mp_ren->AddActor(actor);
        }
        /// The render window's interactor.
        vtkRenderWindowInteractor *getInteractor()
        {
            return mp_renWin->GetInteractor();
        }
        /// The scene's renderer.
        vtkRenderer *getRenderer() { return mp_ren; }

        /// One unit-sphere mapper (Utilities::createUnitSphereMapper) for every view to share, created on first use, so many spheres don't each upload their own copy of it.
        vtkSmartPointer<vtkPolyDataMapper> unitSphereMapper();
        /// One unit-cylinder mapper (Utilities::createUnitCylinderMapper) for every view to share, created on first use.
        vtkSmartPointer<vtkPolyDataMapper> unitCylinderMapper();
        /// One frame mapper (Utilities::createFrameMapper) per size for every view to share, created on first use.
        vtkSmartPointer<vtkPolyDataMapper> frameMapper(double cylinderRadius, double totalLength);

        /// The widget's current frame (top row first) as screenshots and videos export it: without the info message (see sendMessage), which stays on screen.
        QImage grabFrame();

        /// Whether a video export (V, see startVideoExport) is in progress.
        bool exportingVideo() const { return m_video != nullptr; }

        /// The play/pause/seek/speed/delay state driving each frame.
        PlaybackController &playback() { return m_playback; }

        /**
         * @brief Updates every Visualizable at `time - delay` (estimate time `time`),
         *        refreshes the diagnostics panel and timeline, and renders.
         * @param time Playback (displayed) time, before the phase delay.
         */
        void update(double time);

    public Q_SLOTS:
        /// Animation-timer tick: advances playback by the elapsed wall time, orbits the camera by the playback time advanced (if orbiting), and calls update().
        void slotAnimationTimer();
        /// Hides the info label shown by sendMessage().
        void hideMessage();

    protected:
        /// Re-lays out the info label, timeline and diagnostics panel for the new size.
        void resizeEvent(QResizeEvent *event) override;

    private:
        VisualizerOptions m_options;
        /// Sets the background, lights, optional floor and world frame, and the initial camera.
        void initScene();
        /// Saves the current view to `<screenshotsFolder()>/sample_<index>/<frame>.png`, then one frame per extra camera (cameras[1..]).
        void screenshot();
        /// Temporarily moves the camera to `camera_config`, saves `<screenshotsFolder()>/<subfolder_name>/<frame>.png`, then restores the camera.
        void screenshotFromCamera(const VisualizerOptions::CameraOptions &camera_config, const std::string &subfolder_name);
        /// `<output_path>/screenshots`, or `./screenshots` (the working directory) when no output path was given.
        std::filesystem::path screenshotsFolder() const;
        /// `<output_path>/videos`, or `./videos` when no output path was given (see screenshotsFolder).
        std::filesystem::path videosFolder() const;
        /// Writes grabFrame() as a PNG to `file_path`, creating its parent folders if needed.
        void writeFrame(const std::filesystem::path &file_path);
        /// Left/right arrows (offline only): steps to the adjacent estimation tick in `direction` (see PlaybackController::stepTick), or 1% of the range if VisualizerOptions::estimation_period is unknown, then renders.
        void stepTime(int direction);
        /// X: saves one screenshot (see screenshot()); a failure is logged and shown instead of thrown, since Qt aborts on an exception leaving an event handler.
        void takeScreenshot();

        /**
         * @brief V (offline only): pauses playback and, one frame per event-loop pass,
         *        renders every estimation tick (range start + k * estimation_period, up to
         *        the range end) into `<videosFolder()>/video_<index>.mp4` at
         *        1 / estimation_period fps, so it plays in real time. Shows a message, and
         *        records nothing, if the period is unknown or the encoder cannot start.
         */
        void startVideoExport();
        /// Renders and encodes the next tick (see startVideoExport), finishing the video after the last.
        void exportNextVideoFrame();
        /// Ends the export: finishes the file and reports it (`cancelled` if stopped early with V).
        void finishVideoExport(bool cancelled);
        /// Ends the export after `what` went wrong, logging and showing it.
        void abortVideoExport(const std::string &what);

        // initialization functions
        /// Lights the scene with a vtkLightKit (key, fill, back and head lights, following the camera).
        void initLights();
        /// Adds axes at the world origin.
        void initWorldFrame();
        /// Moves the camera to the first configured camera, or auto-fits the scene if none is configured.
        void resetCameraView();

        /// Height for the diagnostics panel's fixed viewport: bounded by the window so content that overflows scrolls instead of being cropped.
        int diagnosticsLabelHeight() const;

        /// Amount one delay hotkey press changes the phase delay by: currently a fixed 0.01 s (a window-relative step is commented out in the implementation).
        double delayStep() const;

        vtkSmartPointer<vtkRenderWindow> mp_renWin;
        vtkSmartPointer<vtkRenderer> mp_ren;

        PlaybackController m_playback;
        QTimer m_animation_timer;
        std::chrono::steady_clock::time_point m_last_tick_time;

        vtkNew<vtkPNGWriter> m_screenshot_writer;
        /// The video being exported (see startVideoExport), or nullptr.
        std::unique_ptr<VideoWriter> m_video;
        /// Drives exportNextVideoFrame while exporting (0 ms: once per event-loop pass).
        QTimer m_video_timer;
        /// Index of the next tick to export.
        std::size_t m_video_tick = 0;
        /// Numbers the exported videos (video_<index>.mp4).
        unsigned int m_video_index = 0;
        /// Whether the camera orbits its focal point as playback advances (toggle O; see kOrbitDegreesPerSecond).
        bool m_orbiting = false;

        std::string m_output_path;

        QLabel *m_infoLabel;
        QTimer m_infoTimer;
        InfoWidget *m_diagnosticsLabel = nullptr;
        KeyBindingRegistry m_key_bindings;
        /// Labels added by addControlInfo, so each is listed once.
        std::set<std::string> m_control_infos;
        TimelineWidget *m_timelineWidget = nullptr;
        /// See addOverlayWidget.
        std::vector<std::pair<QWidget *, std::function<QRect(QSize)>>> m_overlays;
        double m_time_range_start = 0.0, m_time_range_end = 1.0;

        /// See unitSphereMapper / unitCylinderMapper.
        vtkSmartPointer<vtkPolyDataMapper> mp_unit_sphere_mapper, mp_unit_cylinder_mapper;
        /// See frameMapper; keyed by (cylinderRadius, totalLength).
        std::map<std::pair<double, double>, vtkSmartPointer<vtkPolyDataMapper>> m_frame_mappers;

        /// Running averages (see kFrameTimeSmoothing) of update()'s view-update and render wall times [s], shown in the diagnostics panel.
        double m_update_seconds = 0.0, m_render_seconds = 0.0;
        /// Weight of the newest frame in m_update_seconds / m_render_seconds.
        static constexpr double kFrameTimeSmoothing = 0.1;

        bool m_verbose = true;
        int m_sample_index = 0;
        unsigned int m_screenshot_frame_idx = 0;
    };
} // namespace Spacetime::Visualization::Default