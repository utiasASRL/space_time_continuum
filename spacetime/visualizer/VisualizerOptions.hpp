#ifndef VISUALIZEROPTIONS_HPP
#define VISUALIZEROPTIONS_HPP

#include <vector>

#include <Eigen/Core>

namespace Spacetime::Visualization
{
    /// Runtime options shared by every visualizer backend. Backends ignore the fields that don't apply to them.
    struct VisualizerOptions
    {
        /// A camera viewpoint (world frame). The first configured camera is the interactive view; the rest are only used for extra recorded screenshots.
        struct CameraOptions
        {
            Eigen::Vector3d position = Eigen::Vector3d(0, -1, 0.5);
            Eigen::Vector3d focal_point = Eigen::Vector3d(0, 0, 0);
            Eigen::Vector3d view_up = Eigen::Vector3d(0, 0, 1);
        };
        std::vector<CameraOptions> cameras;

        /// How the robot views are drawn. Programs copy these onto each robot view (e.g. Default::ContinuumRobot::radius).
        struct RobotOptions
        {
            /// Backbone tube radius [m]; also scales the node frames.
            double radius = 0.01;
        };
        RobotOptions robot;

        bool show_floor = true;
        bool show_world_frame = true;
        bool show_info_label = false;
        bool show_diagnostics_label = false;
        bool show_timeline_widget = false;

        /// Seconds since a non-static Visualizable's last data before it's hidden entirely; static objects are always shown.
        double stale_timeout = 0.2;

        /// Offline: data is loaded once up front and PlaybackController scrubs/replays it.
        /// Online: PlaybackController tracks wall-clock time and each visualizable reads live estimator state.
        bool online = false;

        /// Estimator fixed-lag window [s]; the phase delay is clamped to [0, estimator_window]. <= 0 means unknown: delay is only floored at 0. (The delay step is currently a fixed 0.01 s, see Default::Visualizer::delayStep.)
        double estimator_window = -1.0;

        /// Estimator tick period [s] (the session's period): offline, the arrow keys step between ticks and video export (V) renders one frame per tick, at 1 / period fps. <= 0 means unknown: the arrows step 1% of the range and video export is disabled.
        double estimation_period = -1.0;
    };
} // namespace Spacetime::Visualization

#endif // VISUALIZEROPTIONS_HPP
