#ifndef DEFAULT_CONTINUUMROBOT_HPP
#define DEFAULT_CONTINUUMROBOT_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkTubeFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkMatrix4x4.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/utilities/RobotShape.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/utilities.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief Default-backend view of a Types::ContinuumRobotState<NodeT>: its backbone
     *        tube, node frames, posterior-covariance ellipsoids and any STL meshes attached
     *        along it (see addSTL). The state itself
     *        holds no visualization state; this view reads its nodes through
     *        ContinuumRobotState::snapshotEstimationNodes and the values to draw them at
     *        from a bound EstimatorSnapshot source. The robot's sensors are drawn by
     *        views of their own (see makeSensorView).
     *
     *        Like every default-backend view, it is used on the visualizer thread only;
     *        the estimator may extend the state concurrently, which is safe because the
     *        nodes are read through the state's mutex-guarded snapshotEstimationNodes().
     * @tparam NodeT Node type of the drawn ContinuumRobotState.
     */
    template <Types::NodeType NodeT>
    class ContinuumRobot : public Visualizable
    {
        /// The state whose node grid is drawn.
        std::shared_ptr<const Types::ContinuumRobotState<NodeT>> mp_state;
        /// Spatial nodes per temporal row.
        const unsigned int spatial_dim;

    public:
        /// Throws std::invalid_argument if `state` is null.
        explicit ContinuumRobot(std::shared_ptr<const Types::ContinuumRobotState<NodeT>> state)
            : mp_state(std::move(state)), spatial_dim(mp_state ? mp_state->spatial_dim : 0), m_shape(nonNull(mp_state)) {}

        /**
         * @brief Bind a source of EstimatorSnapshots this view should read from.
         *        Online mode binds a function that always returns the estimator's
         *        latest snapshot (ignoring its argument, the estimate time --
         *        see Visualizable::updateVisualization); offline mode binds a lookup
         *        over a recorded history so that rewinding past the fixed-lag
         *        smoother's marginalization window still resolves to the correct
         *        historical values -- the view itself doesn't know or care which.
         */
        void bindSnapshotSource(std::function<std::shared_ptr<const EstimatorSnapshot>(double)> source) { mp_snapshot_source = std::move(source); }

        /// The state whose node grid this view draws.
        const std::shared_ptr<const Types::ContinuumRobotState<NodeT>> &state() const { return mp_state; }

        /// Backbone polyline points (true nodes plus spatial interpolation points), rewritten every frame.
        mutable vtkSmartPointer<vtkPoints> mp_backbone_points;
        /// Backbone polyline (points, line cells and per-segment grey shades); see rebuildBackboneGeometry.
        mutable vtkSmartPointer<vtkPolyData> mp_backbone_polydata;
        /// Turns the backbone polyline into a tube of `radius`.
        mutable vtkSmartPointer<vtkTubeFilter> mp_backbone_tube;
        /// The backbone tube's actor; created hidden by initBackbone.
        mutable vtkSmartPointer<vtkActor> mp_backbone_actor;
        /// One frame per spatial node (see Utilities::createFrameActor).
        mutable std::vector<vtkSmartPointer<vtkActor>> mp_axes;

        /// Posterior position-covariance ellipsoids, one per backbone point (true nodes and
        /// spatial interpolation points, in backbone order). Sized for the maximum number of
        /// interpolation points; the ones not in use are hidden. Empty until the covariance
        /// is first toggled on (see toggleCovariance).
        mutable std::vector<vtkSmartPointer<vtkActor>> mp_covariance_actors;
        /// The unit sphere the ellipsoids draw (the visualizer's shared one once visualized).
        mutable vtkSmartPointer<vtkPolyDataMapper> mp_covariance_mapper;
        /// Where the ellipsoids are added once created; set by visualize().
        mutable vtkSmartPointer<vtkRenderer> mp_covariance_renderer;
        /// Whether the covariance ellipsoids are on (toggle C, see toggleCovariance).
        mutable bool m_show_covariance = false;
        /// Set once the missing-covariance warning was logged, so it is logged only once.
        mutable bool m_warned_missing_covariance = false;

        /// Backbone tube radius; also scales the node frames. Read in visualize().
        double radius = 0.01;

        /// Whether the node frames are drawn (with the backbone). Set it before visualize() to
        /// start them hidden, e.g. where other frames already mark the nodes; toggleNodeFrames()
        /// flips it.
        mutable bool show_node_frames = true;

        /// Source of the EstimatorSnapshots this view draws, see bindSnapshotSource.
        std::function<std::shared_ptr<const EstimatorSnapshot>(double)> mp_snapshot_source;

        /// Reuses the prior's own interpolation scheme (e.g. WNOA::Interpolator::interpolate)
        /// for visualization, without this view needing to know which prior model it is
        /// (see Utilities::RobotShape::InterpolateFn).
        using InterpolateFn = typename ::Spacetime::Utilities::RobotShape<NodeT>::InterpolateFn;
        /// Resolves the drawn poses from a snapshot (see bindInterpolator).
        mutable ::Spacetime::Utilities::RobotShape<NodeT> m_shape;

        /// Upper bound for adjustSpatialInterpolationPoints.
        static constexpr unsigned int kMaxSpatialInterpolationPoints = 10;
        /// Extra points per backbone segment.
        mutable unsigned int m_spatial_interp_points = 0;
        /// See toggleTemporalInterpolation.
        mutable bool m_temporal_interp_enabled = false;

        /// Sets the interpolator used for spatial subdivision, temporal interpolation and poseAt(); without one, only the true nodes are drawn.
        void bindInterpolator(InterpolateFn fn) const { m_shape.bindInterpolator(std::move(fn)); }

        /**
         * @brief Binds the interpolator of `prior` (e.g. a Priors::WNOA::Prior<Pose3>), the
         *        prior extending the drawn state, with its posterior covariance propagation.
         *        The view keeps the prior alive.
         * @throws std::invalid_argument if `prior` is null or extends another state.
         */
        template <typename PriorT>
        void bindPriorInterpolator(std::shared_ptr<PriorT> prior) const { m_shape.bindPriorInterpolator(std::move(prior)); }

        /**
         * @brief The estimated pose (world_T_point) at (time, arclength), read from the
         *        snapshot bound for estimate_time (see bindSnapshotSource) through the bound
         *        interpolator, exactly as the backbone is drawn: at the latest row at or
         *        before min(time, snapshot time), interpolating towards the next row only
         *        with temporal interpolation on (see toggleTemporalInterpolation) and that
         *        row in the snapshot, and otherwise at the row's own time (never
         *        extrapolating). Spatially, it interpolates towards the next node. Anything
         *        placed with it (e.g. a Joint view) thus moves in step with the drawn robot.
         *        Safe to call while the estimator extends the state (see snapshotEstimationNodes).
         * @return nullopt without a snapshot or interpolator, outside the robot's extent, or
         *         when a bordering node is missing from the snapshot (e.g. marginalized).
         */
        std::optional<Eigen::Matrix4d> poseAt(double time, double arclength, double estimate_time) const
        {
            if (!mp_snapshot_source || !m_shape.hasInterpolator())
                return std::nullopt;
            auto snapshot = mp_snapshot_source(estimate_time);
            if (!snapshot)
                return std::nullopt;
            return poseAt(time, arclength, *snapshot, mp_state->snapshotEstimationNodes());
        }

        /// poseAt(time, arclength, estimate_time) from an already resolved `snapshot` and
        /// `nodes` (a snapshotEstimationNodes() copy), e.g. for several points in one frame.
        std::optional<Eigen::Matrix4d> poseAt(double time, double arclength, const EstimatorSnapshot &snapshot,
                                              const std::vector<std::shared_ptr<NodeT>> &nodes) const
        {
            return m_shape.poseAt(time, arclength, snapshot, nodes, m_temporal_interp_enabled);
        }

        /// An STL mesh drawn rigidly attached to the robot at an arclength (see addSTL).
        struct AttachedMesh
        {
            double arclength;                ///< Where along the robot it is attached [m].
            vtkSmartPointer<vtkActor> actor; ///< The mesh's actor (mesh already scaled and offset).
            /// Whether the latest frame placed it (a pose resolved with the robot shown), i.e. it may be shown.
            bool drawn = false;
        };
        /// The meshes attached by addSTL, in the order added.
        mutable std::vector<AttachedMesh> mp_meshes;
        /// Where the meshes are added; set by initMeshes() (from visualize()).
        mutable vtkSmartPointer<vtkRenderer> mp_mesh_renderer;

        /// Whether the attached meshes are drawn (with the backbone). Set it before visualize() to
        /// start them hidden; toggleMeshes() flips it.
        mutable bool show_meshes = true;

        /// Shows or hides the attached meshes (see show_meshes), immediately: shown, they reappear where
        /// the latest frame placed them, and only if it did.
        void toggleMeshes() const
        {
            show_meshes = !show_meshes;
            for (const auto &mesh : mp_meshes)
                mesh.actor->SetVisibility(show_meshes && mesh.drawn);
        }

        /// See show_meshes.
        bool meshesShown() const { return show_meshes; }

        /**
         * @brief Attaches the STL mesh at `stl_path` to the robot at `arclength`: it is drawn
         *        at the estimated pose there (see poseAt) as world_T_point * offset *
         *        Scale(scale), i.e. scaled by `scale` in its own frame, then placed by `offset`
         *        (point_T_mesh) relative to the robot's frame at `arclength`. The mesh is read,
         *        scaled and offset once, here; each frame only moves its origin. It is shown
         *        with the backbone (and its visibility toggle), unless the meshes are toggled off
         *        (see toggleMeshes), and only while an interpolator is bound (see bindInterpolator). Added to the visualizer at once if the view is
         *        already visualized, otherwise by visualize().
         * @return The mesh's actor, e.g. to colour it.
         * @throws std::invalid_argument if `arclength` is outside [0, length] or `scale` is not
         *         positive and finite.
         * @throws std::runtime_error if the mesh cannot be read (see Utilities::loadSTLMesh).
         */
        vtkSmartPointer<vtkActor> addSTL(const std::string &stl_path, double arclength, double scale = 1.0,
                                         const gtsam::Pose3 &offset = gtsam::Pose3()) const
        {
            if (!(arclength >= 0.0 && arclength <= mp_state->length))
                throw std::invalid_argument("Default::ContinuumRobot::addSTL: arclength " + std::to_string(arclength) + " is outside the robot [0, " +
                                            std::to_string(mp_state->length) + "].");
            if (!(scale > 0.0 && std::isfinite(scale)))
                throw std::invalid_argument("Default::ContinuumRobot::addSTL: the scale must be positive and finite.");

            vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
            transform->SetMatrix(Utilities::toVtkMatrix(offset.matrix()));
            vtkSmartPointer<vtkTransformPolyDataFilter> filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
            filter->SetInputData(Utilities::loadSTLMesh(stl_path, scale));
            filter->SetTransform(transform);
            filter->Update();
            vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
            mesh->ShallowCopy(filter->GetOutput());

            vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            mapper->SetInputData(mesh);
            vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
            actor->SetMapper(mapper);
            Utilities::applySurfaceMaterial(actor->GetProperty());
            actor->SetPickable(false);
            actor->SetVisibility(false);

            mp_meshes.push_back({arclength, actor, false});
            if (mp_mesh_renderer)
                mp_mesh_renderer->AddActor(actor);
            return actor;
        }

        /// Adds the meshes attached so far to `renderer`, and those attached later as they are (see addSTL).
        void initMeshes(vtkRenderer *renderer) const
        {
            mp_mesh_renderer = renderer;
            for (const auto &mesh : mp_meshes)
                mp_mesh_renderer->AddActor(mesh.actor);
        }

        /// Extra rendering-subdivision points inserted between each pair of adjacent spatial nodes along the backbone. Available in both online and offline mode. No-op if no interpolator is bound.
        void adjustSpatialInterpolationPoints(int delta) const
        {
            if (!m_shape.hasInterpolator())
                return;
            const int new_value = std::clamp(static_cast<int>(m_spatial_interp_points) + delta, 0, static_cast<int>(kMaxSpatialInterpolationPoints));
            if (static_cast<unsigned int>(new_value) == m_spatial_interp_points)
                return;
            m_spatial_interp_points = static_cast<unsigned int>(new_value);
            rebuildBackboneGeometry();
        }

        /// Whether to draw the posterior position-covariance ellipsoids. They are only shown
        /// when the snapshot carries a covariance (EstimatorOptions::compute_covariance).
        /// The ellipsoid actors are created the first time it is turned on.
        void toggleCovariance() const
        {
            m_show_covariance = !m_show_covariance;
            if (m_show_covariance && mp_covariance_actors.empty())
                initCovarianceEllipsoids();
        }

        /// Whether to interpolate between the two temporal rows bordering the display time (offline only -- callers should not expose this control in online mode, since the "next" row may not exist yet).
        void toggleTemporalInterpolation() const
        {
            if (m_shape.hasInterpolator())
                m_temporal_interp_enabled = !m_temporal_interp_enabled;
        }

        /// Whether the backbone and poseAt() interpolate between the two temporal rows bordering the display time (off by default).
        bool temporalInterpolationEnabled() const { return m_temporal_interp_enabled; }

        /// Shows or hides the node frames (see show_node_frames), immediately: shown, they follow the backbone.
        void toggleNodeFrames() const
        {
            show_node_frames = !show_node_frames;
            const bool shown = show_node_frames && mp_backbone_actor && mp_backbone_actor->GetVisibility();
            for (const auto &axes : mp_axes)
                axes->SetVisibility(shown);
        }

        /// See show_node_frames.
        bool nodeFramesShown() const { return show_node_frames; }

        /// "ContinuumRobot", or with `brief` false, also its node count.
        std::string info(bool brief = true) const override
        {
            if (brief)
                return "ContinuumRobot";
            return "ContinuumRobot (" + std::to_string(spatial_dim) + " nodes per row)";
        };

        /**
         * @brief Draws the backbone, node frames and attached meshes (the covariance ellipsoids are
         *        created once first shown, see toggleCovariance), and binds the controls: the visibility toggle (if set),
         *        [ / ] spatial interpolation points, T temporal interpolation (offline
         *        only), C posterior covariance, A node frames and G attached meshes (geometry).
         */
        void visualize(Visualizer &visualizer) const override
        {
            initBackbone(visualizer);
            initRobotFrames(visualizer);
            initMeshes(visualizer.getRenderer());
            // The ellipsoids are only created once shown (see toggleCovariance).
            mp_covariance_mapper = visualizer.unitSphereMapper();
            mp_covariance_renderer = visualizer.getRenderer();

            bindVisibilityToggle(visualizer, info(true));

            // Spatial interpolation-point-count adjustment is available in both
            // online and offline mode (adjustSpatialInterpolationPoints no-ops
            // until an interpolator is bound); temporal interpolation needs a
            // "next" temporal row bordering the display time, which isn't
            // available while data is still streaming in live, so that control is
            // offline-only.
            visualizer.bindShortcut(Qt::Key_BracketLeft, [this]()
                                    { adjustSpatialInterpolationPoints(-1); });
            visualizer.bindShortcut(Qt::Key_BracketRight, [this]()
                                    { adjustSpatialInterpolationPoints(+1); });
            visualizer.addControlInfo("Adjust " + info(true) + " spatial interpolation points [[, ]]");

            if (!visualizer.isOnline())
                visualizer.bindToggleShortcut(Qt::Key_T, [this]()
                                              { toggleTemporalInterpolation(); }, [this]()
                                              { return temporalInterpolationEnabled(); }, "Toggle " + info(true) + " temporal interpolation");

            visualizer.bindToggleShortcut(Qt::Key_C, [this]()
                                          { toggleCovariance(); }, [this]()
                                          { return m_show_covariance; }, "Toggle " + info(true) + " posterior covariance");

            visualizer.bindToggleShortcut(Qt::Key_A, [this]()
                                          { toggleNodeFrames(); }, [this]()
                                          { return nodeFramesShown(); }, "Toggle " + info(true) + " node frames");

            visualizer.bindToggleShortcut(Qt::Key_G, [this]()
                                          { toggleMeshes(); }, [this]()
                                          { return meshesShown(); }, "Toggle " + info(true) + " meshes");
        };

        /**
         * @brief Draws the latest temporal row at or before
         *        min(time, snapshot time) from the snapshot for `estimate_time`. Hides the
         *        robot if there is no snapshot or row, or the row is not fully in the
         *        snapshot (e.g. marginalized); with temporal interpolation enabled and the
         *        next row available, interpolates to exactly `time`.
         */
        void updateVisualization(double time, double estimate_time) const override
        {
            // The snapshot is chosen by estimate_time (what the estimator knew "now")
            // while rows are chosen by the possibly-delayed display time, so a phase
            // delay shows an earlier point inside the current estimation window.
            auto snapshot = mp_snapshot_source ? mp_snapshot_source(estimate_time) : nullptr;
            auto nodes = mp_state->snapshotEstimationNodes(); // thread-safe copy: see ContinuumRobotState::m_nodes_mutex
            const std::optional<double> row_time = snapshot ? m_shape.rowTime(time, *snapshot, nodes) : std::nullopt;

            if (!row_time)
            {
                mp_backbone_actor->SetVisibility(false);
                for (const auto &axes : mp_axes)
                    axes->SetVisibility(false);
                for (auto &mesh : mp_meshes)
                {
                    mesh.drawn = false;
                    mesh.actor->SetVisibility(false);
                }
                hideCovarianceEllipsoids(0);
                return;
            }

            const bool fresh = isFresh(time, *row_time);

            // Posterior covariance is only resolved while the ellipsoids are shown, so the
            // extra Jacobian work is skipped otherwise.
            const bool show_covariance = m_show_covariance && visible_ && fresh;
            if (show_covariance && !snapshot->covariance && !m_warned_missing_covariance)
            {
                warning() << "ContinuumRobot: no posterior covariance to show; set estimator.compute_covariance to true in the config.";
                m_warned_missing_covariance = true;
            }
            const PosteriorCovariance *posterior = show_covariance ? snapshot->covariance.get() : nullptr;

            // Backbone: the true nodes plus m_spatial_interp_points extra points per segment
            // (time-interpolated to the display time between the two bordering rows if
            // enabled, see toggleTemporalInterpolation), for a smoother rendered curve.
            const auto backbone = m_shape.backbone(time, *snapshot, nodes, m_spatial_interp_points, m_temporal_interp_enabled, posterior);
            for (std::size_t k = 0; k < backbone->points.size(); k++)
            {
                const auto &point = backbone->points[k];
                mp_backbone_points->SetPoint(k, point.pose(0, 3), point.pose(1, 3), point.pose(2, 3));
                updateCovarianceEllipsoid(k, point.pose, point.covariance);
                if (k % backbone->points_per_segment == 0)
                {
                    const auto &axes = mp_axes[k / backbone->points_per_segment];
                    Utilities::setUserMatrix(axes, point.pose);
                    axes->SetVisibility(visible_ && fresh && show_node_frames);
                }
            }
            mp_backbone_points->Modified();
            mp_backbone_actor->SetVisibility(visible_ && fresh);
            hideCovarianceEllipsoids(backbone->points.size());

            // Each mesh at the pose its arclength is drawn at (the backbone's own row and interpolation).
            for (auto &mesh : mp_meshes)
            {
                const auto pose = (visible_ && fresh) ? poseAt(time, mesh.arclength, *snapshot, nodes) : std::nullopt;
                if (pose)
                    Utilities::setUserMatrix(mesh.actor, *pose);
                mesh.drawn = pose.has_value();
                mesh.actor->SetVisibility(show_meshes && mesh.drawn);
            }
        };

        /// Flips visibility and applies it to the backbone and frames immediately (ellipsoids and meshes
        /// are hidden when turning off; meshes are shown again by the next updateVisualization).
        void toggleVisibility() const override
        {
            visible_ = !visible_;
            mp_backbone_actor->SetVisibility(visible_);
            for (const auto &axes : mp_axes)
                axes->SetVisibility(visible_ && show_node_frames);
            if (!visible_)
                for (auto &mesh : mp_meshes)
                {
                    mesh.drawn = false;
                    mesh.actor->SetVisibility(false);
                }
            if (!visible_)
                hideCovarianceEllipsoids(0);
        };

        /// Creates the (initially hidden) backbone tube actor of `radius` and adds it to `visualizer`.
        void initBackbone(Visualizer &visualizer) const
        {
            mp_backbone_tube = vtkSmartPointer<vtkTubeFilter>::New();
            mp_backbone_tube->SetRadius(radius);
            mp_backbone_tube->SetNumberOfSides(Utilities::kTubeSides);

            rebuildBackboneGeometry();

            vtkSmartPointer<vtkPolyDataMapper> backboneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            backboneMapper->SetInputConnection(mp_backbone_tube->GetOutputPort());
            backboneMapper->SetScalarModeToUseCellData();

            mp_backbone_actor = vtkSmartPointer<vtkActor>::New();
            mp_backbone_actor->SetMapper(backboneMapper);
            Utilities::applySurfaceMaterial(mp_backbone_actor->GetProperty());
            mp_backbone_actor->SetVisibility(false);

            visualizer.addActor(mp_backbone_actor);
        }

        /// (Re)builds the backbone's points/cells/colors for the current spatial_dim +
        /// m_spatial_interp_points point count. Called once from initBackbone and again
        /// whenever adjustSpatialInterpolationPoints() changes the subdivision count.
        void rebuildBackboneGeometry() const
        {
            const unsigned int total_points = spatial_dim + (spatial_dim - 1) * m_spatial_interp_points;
            const unsigned int points_per_segment = m_spatial_interp_points + 1;

            mp_backbone_points = vtkSmartPointer<vtkPoints>::New();
            for (unsigned int i = 0; i < total_points; i++)
                mp_backbone_points->InsertPoint(i, 0, 0, 0);

            mp_backbone_polydata = vtkSmartPointer<vtkPolyData>::New();
            mp_backbone_polydata->SetPoints(mp_backbone_points);
            mp_backbone_polydata->Allocate();

            for (unsigned int m = 0; m + 1 < total_points; m++)
            {
                vtkIdType connectivity[2] = {m, m + 1};
                mp_backbone_polydata->InsertNextCell(VTK_LINE, 2, connectivity);
            }

            // Alternate the shades per true spatial segment, so subdividing a segment
            // doesn't change its color.
            vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
            colors->SetName("Colors");
            colors->SetNumberOfComponents(3);
            colors->SetNumberOfTuples(mp_backbone_polydata->GetNumberOfCells());
            for (vtkIdType i = 0; i < mp_backbone_polydata->GetNumberOfCells(); i++)
            {
                const unsigned int true_segment = static_cast<unsigned int>(i) / points_per_segment;
                const unsigned char shade = Utilities::kBackboneShades[true_segment % 2];
                colors->SetTuple3(i, shade, shade, shade);
            }
            mp_backbone_polydata->GetCellData()->SetScalars(colors);

            if (mp_backbone_tube)
                mp_backbone_tube->SetInputData(mp_backbone_polydata);
        }

        /// Creates one (initially hidden) frame per spatial node, sharing the visualizer's frame mapper, and adds them to `visualizer`.
        void initRobotFrames(Visualizer &visualizer) const
        {
            mp_axes.clear();
            for (unsigned int i = 0; i < spatial_dim; i++)
            {
                vtkSmartPointer<vtkActor> axes = Utilities::createFrameActor(visualizer.frameMapper(std::max(0.025, 0.25 * radius), std::max(0.015, 1.5 * radius)));
                axes->SetVisibility(false);
                mp_axes.push_back(axes);
                visualizer.addActor(axes);
            }
        }

        /// Creates (hidden) ellipsoid actors for the maximum backbone point count, all drawing
        /// mp_covariance_mapper (a new unit sphere if unset), and adds them to
        /// mp_covariance_renderer if set.
        void initCovarianceEllipsoids() const
        {
            if (!mp_covariance_mapper)
                mp_covariance_mapper = Utilities::createUnitSphereMapper();
            mp_covariance_actors.clear();
            const unsigned int max_points = spatial_dim + (spatial_dim - 1) * kMaxSpatialInterpolationPoints;
            for (unsigned int i = 0; i < max_points; i++)
            {
                vtkSmartPointer<vtkActor> actor = Utilities::createEllipsoidActor(mp_covariance_mapper);
                mp_covariance_actors.push_back(actor);
                if (mp_covariance_renderer)
                    mp_covariance_renderer->AddActor(actor);
            }
        }

        /**
         * @brief Place backbone point `index`'s ellipsoid from its posterior covariance, or
         *        hide it if none was resolved.
         * @param pose The point's pose (world_T_point).
         * @param covariance Posterior covariance whose leading 6 x 6 block is the pose's,
         *        in gtsam's Pose3 tangent order [rotation; translation] with the right
         *        (body-frame) perturbation T = T_hat * Exp(delta). Empty if unavailable.
         */
        void updateCovarianceEllipsoid(unsigned int index, const Eigen::Matrix4d &pose, const Eigen::MatrixXd &covariance) const
        {
            if (index >= mp_covariance_actors.size())
                return;
            const auto &actor = mp_covariance_actors[index];
            if (covariance.rows() < 6 || covariance.cols() < 6)
            {
                actor->SetVisibility(false);
                return;
            }
            // To first order the body-frame translation perturbation moves the world
            // position by R * delta_t, so its world covariance is R Sigma_tt R^T.
            const Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
            const Eigen::Matrix3d position_covariance = R * covariance.block<3, 3>(3, 3) * R.transpose();
            Utilities::setUserMatrix(actor, Utilities::ellipsoidTransform(pose.block<3, 1>(0, 3), position_covariance));
            actor->SetVisibility(true);
        }

        /// Hide every ellipsoid from backbone point `first` onwards.
        void hideCovarianceEllipsoids(unsigned int first) const
        {
            for (unsigned int i = first; i < mp_covariance_actors.size(); i++)
                mp_covariance_actors[i]->SetVisibility(false);
        }

    private:
        /// `state`, or throws std::invalid_argument if it is null (for the constructor's initializers).
        static const std::shared_ptr<const Types::ContinuumRobotState<NodeT>> &nonNull(const std::shared_ptr<const Types::ContinuumRobotState<NodeT>> &state)
        {
            if (!state)
                throw std::invalid_argument("Default::ContinuumRobot requires a non-null state.");
            return state;
        }
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_CONTINUUMROBOT_HPP
