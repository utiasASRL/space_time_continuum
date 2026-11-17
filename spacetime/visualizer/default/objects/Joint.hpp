#ifndef DEFAULT_JOINT_HPP
#define DEFAULT_JOINT_HPP

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>

#include <spacetime/visualizer/default/Visualizable.hpp>
#include <spacetime/visualizer/default/Visualizer.hpp>
#include <spacetime/visualizer/default/objects/ContinuumRobot.hpp>
#include <spacetime/visualizer/default/utilities.hpp>

namespace Spacetime::Visualization::Default
{
    /**
     * @brief Default-backend view of a joint between two robots (see Joints::Joint): a solid
     *        dark grey cylinder between the two joined points when the joint has an offset, or
     *        a dark grey sphere at them when it does not. Drawn at the estimated poses of both
     *        points, and hidden while either is unavailable.
     */
    class Joint : public Visualizable
    {
    public:
        /// The estimated world_T_point of one joined point at (time, estimate_time), or nullopt (see ContinuumRobot::poseAt).
        using PointPoseFn = std::function<std::optional<Eigen::Matrix4d>(double time, double estimate_time)>;

        /**
         * @param point_a Source of joint point A's pose.
         * @param point_b Source of joint point B's pose.
         * @param has_offset Draw a cylinder between the points (true) or a sphere at them (false).
         * @throws std::invalid_argument for an empty pose source.
         */
        Joint(PointPoseFn point_a, PointPoseFn point_b, bool has_offset)
            : m_point_a(std::move(point_a)), m_point_b(std::move(point_b)), m_has_offset(has_offset)
        {
            if (!m_point_a || !m_point_b)
                throw std::invalid_argument("Default::Joint requires two pose sources.");
        }

        /**
         * @brief The view of `joint` (a Joints::Joint) between the objects drawn by `view_a`
         *        and `view_b`: robot views (ContinuumRobot, read through poseAt() at the joint's
         *        arclengths) or any view providing a pointPose() source of its own (e.g.
         *        Default::EndEffector). Holds the views weakly.
         */
        template <typename JointT, typename ViewA, typename ViewB>
        static std::shared_ptr<Joint> of(const JointT &joint, const std::shared_ptr<ViewA> &view_a, const std::shared_ptr<ViewB> &view_b)
        {
            return std::make_shared<Joint>(pointOn(view_a, joint.arclengthA()), pointOn(view_b, joint.arclengthB()), joint.hasOffset());
        }

        /// A pose source reading arclength `s` of the object drawn by `view` (held weakly; see of()).
        template <typename ViewT>
        static PointPoseFn pointOn(const std::shared_ptr<ViewT> &view, double s)
        {
            if constexpr (requires { view->pointPose(); })
            {
                (void)s; // the view's own point, e.g. an end effector's frame
                return view->pointPose();
            }
            else
            {
                std::weak_ptr<ViewT> weak = view;
                return [weak, s](double time, double estimate_time) -> std::optional<Eigen::Matrix4d>
                {
                    const auto locked = weak.lock();
                    return locked ? locked->poseAt(time, s, estimate_time) : std::nullopt;
                };
            }
        }

        /// "Joint", or with `brief` false, "Joint (link)" / "Joint (ball)".
        std::string info(bool brief = true) const override { return brief ? "Joint" : std::string("Joint (") + (m_has_offset ? "link" : "ball") + ")"; }

        /// Whether a cylinder (link) is drawn between the points rather than a sphere at them.
        bool drawsLink() const { return m_has_offset; }

        /// Radius of the cylinder or sphere [m]. Set before updateVisualization().
        double radius = 0.015;

        /// Creates the actor without adding it to a visualizer (visualize() adds it), drawing `mapper`
        /// (a unit cylinder with an offset, otherwise a unit sphere) -- one of its own if null.
        void initActors(vtkSmartPointer<vtkPolyDataMapper> mapper = nullptr) const
        {
            if (!mapper)
                mapper = m_has_offset ? Utilities::createUnitCylinderMapper() : Utilities::createUnitSphereMapper();
            mp_actor = vtkSmartPointer<vtkActor>::New();
            mp_actor->SetMapper(mapper);
            mp_actor->GetProperty()->SetColor(Utilities::kJointGrey, Utilities::kJointGrey, Utilities::kJointGrey);
            Utilities::applySurfaceMaterial(mp_actor->GetProperty());
            mp_actor->SetVisibility(false);
        }

        /// Adds the (initially hidden) actor, sharing the visualizer's unit mapper, and binds its visibility toggle (if set).
        void visualize(Visualizer &visualizer) const override
        {
            initActors(m_has_offset ? visualizer.unitCylinderMapper() : visualizer.unitSphereMapper());
            visualizer.addActor(mp_actor);
            bindVisibilityToggle(visualizer, info(true));
        }

        /**
         * @brief Places the cylinder between the two point positions (with an offset), or a
         *        sphere of `radius` at their midpoint (without one); hidden if either pose is
         *        unavailable or the view is toggled off. Only the points' positions are used.
         */
        void updateVisualization(double time, double estimate_time) const override
        {
            const std::optional<Eigen::Matrix4d> pose_a = m_point_a(time, estimate_time), pose_b = m_point_b(time, estimate_time);
            if (!pose_a || !pose_b)
            {
                mp_actor->SetVisibility(false);
                return;
            }
            const Eigen::Vector3d p_a = pose_a->block<3, 1>(0, 3), p_b = pose_b->block<3, 1>(0, 3);
            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            if (m_has_offset)
                transform = Utilities::cylinderMatrixBetween(p_a, p_b, radius);
            else
            {
                // Between the two estimates, which a soft joint only holds approximately together.
                transform.block<3, 3>(0, 0) *= radius;
                transform.block<3, 1>(0, 3) = 0.5 * (p_a + p_b);
            }
            Utilities::setUserMatrix(mp_actor, transform);
            mp_actor->SetVisibility(visible_);
        }

        /// The cylinder or sphere; created by initActors().
        mutable vtkSmartPointer<vtkActor> mp_actor;

    private:
        /// Pose sources of the two joined points.
        PointPoseFn m_point_a, m_point_b;
        /// See drawsLink.
        bool m_has_offset;
    };
} // namespace Spacetime::Visualization::Default

#endif // DEFAULT_JOINT_HPP
