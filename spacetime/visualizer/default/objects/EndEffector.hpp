#ifndef DEFAULT_ENDEFFECTOR_HPP
#define DEFAULT_ENDEFFECTOR_HPP

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/joint_models/EndEffector.hpp>
#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/utilities.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief Default-backend view of a Joints::EndEffector: a coordinate frame at its
     *        estimated pose, hidden while that pose is unavailable. Its joints are drawn by
     *        Default::Joint views (see Joint::of, which reads this view's pointPose()).
     *
     * The pose is resolved exactly as a robot view's poseAt() (at s = 0), through an
     * internal, undrawn ContinuumRobot view of the end effector's state bound to its own
     * prior's interpolator, so the two never disagree about rows, snapshots or interpolation. Like a
     * robot view, it snaps to the latest row at or before the display time unless temporal
     * interpolation is on (T, offline only; see toggleTemporalInterpolation).
     *
     * @tparam Pose The end effector's pose type.
     */
    template <typename Pose>
    class EndEffector : public Visualizable
    {
    public:
        using NodeType = Joints::EndEffectorNode<Pose>;
        /// The estimated world_T_ee at (time, estimate_time), or nullopt.
        using PoseFn = std::function<std::optional<Eigen::Matrix4d>(double time, double estimate_time)>;

        explicit EndEffector(std::shared_ptr<const Joints::EndEffector<Pose>> end_effector)
            : mp_poser(std::make_shared<ContinuumRobot<NodeType>>(end_effector ? end_effector->state() : nullptr))
        {
            mp_poser->bindPriorInterpolator(end_effector->endEffectorPrior());
        }

        /// Source of the EstimatorSnapshots this view draws (see ContinuumRobot::bindSnapshotSource).
        void bindSnapshotSource(std::function<std::shared_ptr<const EstimatorSnapshot>(double)> source) const
        {
            mp_poser->bindSnapshotSource(std::move(source));
        }

        /// The estimated end-effector pose (world_T_ee), see ContinuumRobot::poseAt.
        std::optional<Eigen::Matrix4d> poseAt(double time, double estimate_time) const { return mp_poser->poseAt(time, 0.0, estimate_time); }

        /// A pose source for the end effector's frame, holding this view's resolver weakly (see Joint::of).
        PoseFn pointPose() const
        {
            std::weak_ptr<ContinuumRobot<NodeType>> weak = mp_poser;
            return [weak](double time, double estimate_time) -> std::optional<Eigen::Matrix4d>
            {
                const auto locked = weak.lock();
                return locked ? locked->poseAt(time, 0.0, estimate_time) : std::nullopt;
            };
        }

        /// Whether to interpolate between the two temporal rows bordering the display time (see ContinuumRobot::toggleTemporalInterpolation).
        void toggleTemporalInterpolation() const { mp_poser->toggleTemporalInterpolation(); }

        /// See toggleTemporalInterpolation (off by default).
        bool temporalInterpolationEnabled() const { return mp_poser->temporalInterpolationEnabled(); }

        std::string info(bool brief = true) const override { return brief ? "EndEffector" : "EndEffector frame"; }

        /// Length of each drawn axis [m]. Set before visualize().
        double axis_length = 0.06;

        /// Creates the frame actor without adding it to a visualizer (visualize() adds it), drawing
        /// `mapper` (a frame of axis_length, see Utilities::createFrameMapper) -- one of its own if null.
        void initActors(vtkSmartPointer<vtkPolyDataMapper> mapper = nullptr) const
        {
            mp_axes = mapper ? Utilities::createFrameActor(mapper) : Utilities::createFrameActor(kCylinderRadius * axis_length, axis_length);
            mp_axes->SetVisibility(false);
        }

        void visualize(Visualizer &visualizer) const override
        {
            initActors(visualizer.frameMapper(kCylinderRadius * axis_length, axis_length));
            visualizer.addActor(mp_axes);
            bindVisibilityToggle(visualizer, info(true));
            // Temporal interpolation, as the robot views bind it (offline only; T toggles them together).
            if (!visualizer.isOnline())
                visualizer.bindToggleShortcut(Qt::Key_T, [this]()
                                              { toggleTemporalInterpolation(); }, [this]()
                                              { return temporalInterpolationEnabled(); }, "Toggle " + info(true) + " temporal interpolation");
        }

        void updateVisualization(double time, double estimate_time) const override
        {
            const std::optional<Eigen::Matrix4d> pose = poseAt(time, estimate_time);
            if (!pose)
            {
                mp_axes->SetVisibility(false);
                return;
            }
            Utilities::setUserMatrix(mp_axes, *pose);
            mp_axes->SetVisibility(visible_);
        }

        mutable vtkSmartPointer<vtkActor> mp_axes;

    private:
        /// The frame's shaft radius relative to axis_length (see Utilities::createFrameMapper).
        static constexpr double kCylinderRadius = 0.04;

        std::shared_ptr<ContinuumRobot<NodeType>> mp_poser;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_ENDEFFECTOR_HPP
