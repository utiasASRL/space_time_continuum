#ifndef JOINTBASE_HPP
#define JOINTBASE_HPP

#include <memory>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

#include <spacetime/estimator/types/Model.hpp>
#include <spacetime/estimator/types/StatefulModel.hpp>
#include <spacetime/estimator/types/States.hpp>

namespace Spacetime::Joints
{
    /// What a Joint holds together.
    enum class JointType
    {
        Rigid,    ///< The full pose, and the full body twist.
        Spherical ///< The point's position, and its linear velocity; rotations are free.
    };

    /// Optional settings of a Joint.
    struct JointOptions
    {
        JointType type = JointType::Rigid; ///< What the joint holds together.
        /// T_AB: joint point B relative to joint point A, in A's frame (T_B = T_A T_AB).
        /// A spherical joint only uses its translation.
        gtsam::Pose3 offset;
        /// Standard deviation of the velocity residual (rad/s and m/s, per component) on top
        /// of the propagated offset uncertainty, which vanishes at zero velocity. Must be > 0.
        double velocity_sigma_floor = 1e-3;
    };

    /**
     * @class JointBase
     * @brief The prior-independent interface of every Joint<PriorA, PriorB>: which robots it
     *        joins, where, and how. Code that handles joints of any prior pair (e.g. a
     *        visualizer, an EndEffector's joints(), the Python bindings) uses it instead of a
     *        concrete Joint type.
     */
    class JointBase : public Types::Model
    {
    public:
        /// The prior of robot A (Joint::priorA, untyped).
        virtual std::shared_ptr<Types::StatefulModel<>> priorModelA() const = 0;
        /// The prior of robot B (Joint::priorB, untyped).
        virtual std::shared_ptr<Types::StatefulModel<>> priorModelB() const = 0;
        /// Robot A's state, identifying the robot (e.g. to find its view).
        virtual std::shared_ptr<const Types::State> stateA() const = 0;
        /// Robot B's state.
        virtual std::shared_ptr<const Types::State> stateB() const = 0;
        /// Joint arclength s_a on robot A.
        virtual double arclengthA() const = 0;
        /// Joint arclength s_b on robot B.
        virtual double arclengthB() const = 0;
        /// The offset T_AB (see JointOptions::offset).
        virtual const gtsam::Pose3 &offset() const = 0;
        /// The options the joint was built with.
        virtual const JointOptions &options() const = 0;
        /// The offset's covariance, as given to the constructor (6x6 rigid, 3x3 spherical).
        virtual const Eigen::MatrixXd &offsetCovariance() const = 0;
        /// Whether the joined points are apart (the offset has a translation), e.g. to draw a link rather than a ball.
        virtual bool hasOffset() const = 0;
        /// Whether both robots' nodes have a velocity, so the velocities are joined too.
        virtual bool joinsVelocity() const = 0;
    };
} // namespace Spacetime::Joints

#endif // JOINTBASE_HPP
