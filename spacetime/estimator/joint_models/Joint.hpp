#ifndef JOINT_HPP
#define JOINT_HPP

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/estimator/joint_models/JointFactors.hpp>
#include <spacetime/estimator/prior_models/TwoSidedInterpolationFactor.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/Model.hpp>
#include <spacetime/estimator/utilities/functions.hpp>

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
     * @class Joint
     * @brief A model joining two robots: the point at arclength s_a on robot A and the one
     *        at s_b on robot B are held together, through an optional offset, by soft
     *        factors on their poses (or positions) and, when both robots' nodes have
     *        velocities, their velocities.
     *
     * Each robot is given by the prior extending its state. At every new node row, getNewFactors adds a TwoSidedInterpolationFactor per residual (see
     * JointFactors.hpp): the joint points may lie between nodes. The offset's uncertainty
     * enters the pose/position residual directly and the velocity residual to first order
     * (plus JointOptions::velocity_sigma_floor). A zero (hard-constraint) covariance is not
     * supported.
     *
     * @tparam PriorA Robot A's prior type (a Priors::GridPrior, e.g. WNOA::Prior<Pose3>).
     * @tparam PriorB Robot B's prior type.
     */
    template <typename PriorA, typename PriorB>
    class Joint : public Types::Model
    {
    public:
        using InterpolatorA = typename PriorA::InterpolatorType;                             ///< Robot A's interpolator.
        using InterpolatorB = typename PriorB::InterpolatorType;                             ///< Robot B's interpolator.
        using KeysA = typename InterpolatorA::InterpolatedKeys;                              ///< Placeholder keys of robot A's interpolated node.
        using KeysB = typename InterpolatorB::InterpolatedKeys;                              ///< Placeholder keys of robot B's interpolated node.
        using StateA = Types::ContinuumRobotState<typename PriorA::NodeType>;                ///< Robot A's state.
        using StateB = Types::ContinuumRobotState<typename PriorB::NodeType>;                ///< Robot B's state.
        using FactorType = Priors::TwoSidedInterpolationFactor<InterpolatorA, InterpolatorB>; ///< Outer factor wrapping each residual.

        static_assert(std::is_same_v<std::decay_t<decltype(std::declval<typename PriorA::ValuesType>().pose())>, gtsam::Pose3> &&
                          std::is_same_v<std::decay_t<decltype(std::declval<typename PriorB::ValuesType>().pose())>, gtsam::Pose3>,
                      "Joint requires both robots' poses to be gtsam::Pose3.");

        /// Whether both robots' nodes have a velocity, so the velocities are joined too.
        static constexpr bool joins_velocity = requires { KeysA::Velocity(); } && requires { KeysB::Velocity(); };

        /**
         * @param prior_a The prior of robot A.
         * @param s_a Joint arclength on robot A, within its nodes' extent.
         * @param prior_b The prior of robot B (not extending robot A's state).
         * @param s_b Joint arclength on robot B, within its nodes' extent.
         * @param offset_covariance Uncertainty of the offset: 6x6 in its tangent space
         *        ([rot, trans]) for a rigid joint, 3x3 of its translation for a spherical
         *        one. Must be symmetric positive-definite.
         * @param options See JointOptions.
         * @throws std::invalid_argument for invalid inputs (see the parameters).
         */
        Joint(std::shared_ptr<PriorA> prior_a, double s_a, std::shared_ptr<PriorB> prior_b, double s_b,
              const Eigen::MatrixXd &offset_covariance, const JointOptions &options = JointOptions())
            : m_prior_a(std::move(prior_a)), m_prior_b(std::move(prior_b)), m_s_a(s_a), m_s_b(s_b),
              m_offset_covariance(offset_covariance), m_options(options)
        {
            if (!m_prior_a || !m_prior_b)
                throw std::invalid_argument("Joint requires two non-null priors.");
            m_state_a = m_prior_a->state();
            m_state_b = m_prior_b->state();
            if (static_cast<const Types::State *>(m_state_a.get()) == static_cast<const Types::State *>(m_state_b.get()))
                throw std::invalid_argument("Joint requires two different robots.");
            checkArclength(*m_state_a, m_s_a, "A");
            checkArclength(*m_state_b, m_s_b, "B");

            const Eigen::Index dim = m_options.type == JointType::Rigid ? 6 : 3;
            if (m_offset_covariance.rows() != dim || m_offset_covariance.cols() != dim)
                throw std::invalid_argument("Joint: the offset covariance must be " + std::to_string(dim) + "x" + std::to_string(dim) + " for this joint type.");
            if (!m_offset_covariance.isApprox(m_offset_covariance.transpose(), 1e-12) ||
                m_offset_covariance.llt().info() != Eigen::Success)
                throw std::invalid_argument("Joint: the offset covariance must be symmetric positive-definite (hard constraints are not supported).");
            if (!(m_options.velocity_sigma_floor > 0.0))
                throw std::invalid_argument("Joint: velocity_sigma_floor must be positive.");
        }

        /**
         * @brief Joins the newest node row, once: both robots must have a row at the same
         *        time (e.g. both priors extended by the same Estimator::estimate call).
         * @throws std::runtime_error if the robots' newest rows are at different times.
         */
        Factors getNewFactors(double time, gtsam::Values &new_values) override
        {
            (void)time;
            (void)new_values;
            const std::optional<double> t_a = newestRowTime(*m_state_a), t_b = newestRowTime(*m_state_b);
            if (!t_a && !t_b)
                return {};
            const auto joined = [this](const std::optional<double> &t)
            { return t && m_last_joined_time && *t <= *m_last_joined_time + TOLERANCE; };
            if (joined(t_a) && joined(t_b))
                return {};
            if (!t_a || !t_b || std::abs(*t_a - *t_b) > TOLERANCE)
                throw std::runtime_error("Joint: the joined robots' newest node rows are at different times; extend both priors at every estimation time.");

            const double row_time = *t_a;
            m_last_joined_time = row_time;
            const auto node_a = m_prior_a->interpolatedNode(row_time, m_s_a);
            const auto node_b = m_prior_b->interpolatedNode(row_time, m_s_b, sideBKeys());
            if (!node_a || !node_b)
                throw std::runtime_error("Joint: a joint point could not be resolved on its robot's nodes.");

            Factors factors;
            if (m_options.type == JointType::Rigid)
                addRigidFactors(*node_a, *node_b, factors);
            else
                addSphericalFactors(*node_a, *node_b, factors);
            return factors;
        }

        /// The prior of robot A.
        std::shared_ptr<PriorA> priorA() const { return m_prior_a; }
        /// The prior of robot B.
        std::shared_ptr<PriorB> priorB() const { return m_prior_b; }
        /// Joint arclength s_a on robot A.
        double arclengthA() const { return m_s_a; }
        /// Joint arclength s_b on robot B.
        double arclengthB() const { return m_s_b; }
        /// The offset T_AB (see JointOptions::offset).
        const gtsam::Pose3 &offset() const { return m_options.offset; }
        /// The options the joint was built with.
        const JointOptions &options() const { return m_options; }
        /// The offset's covariance, as given to the constructor (6x6 rigid, 3x3 spherical).
        const Eigen::MatrixXd &offsetCovariance() const { return m_offset_covariance; }
        /// Whether the joined points are apart (the offset has a translation), e.g. to draw a link rather than a ball.
        bool hasOffset() const { return m_options.offset.translation().norm() > 1e-9; }

    private:
        /// @throws std::invalid_argument if `s` lies outside the state's node arclengths (or it has none).
        template <typename StateT>
        static void checkArclength(const StateT &state, double s, const std::string &name)
        {
            if (state.arclengths.empty() || s < state.arclengths.front() - TOLERANCE || s > state.arclengths.back() + TOLERANCE)
                throw std::invalid_argument("Joint: arclength " + std::to_string(s) + " lies outside robot " + name + "'s nodes.");
        }

        /// Time of the state's newest node row, or nullopt if it has none yet.
        template <typename StateT>
        static std::optional<double> newestRowTime(const StateT &state)
        {
            if (state.estimation_nodes.empty())
                return std::nullopt;
            return state.estimation_nodes.back()->time;
        }

        /// Side B's placeholders, moved to the 'r' character so they never collide with side A's.
        static typename Priors::InterpolatedNode<InterpolatorB>::KeyMap sideBKeys()
        {
            typename Priors::InterpolatedNode<InterpolatorB>::KeyMap map;
            for (const gtsam::Key key : KeysB::keys())
                map[key] = gtsam::Symbol('r', gtsam::Symbol(key).index());
            return map;
        }

        void addRigidFactors(const Priors::InterpolatedNode<InterpolatorA> &node_a, const Priors::InterpolatedNode<InterpolatorB> &node_b, Factors &factors) const
        {
            const gtsam::Key pose_a = node_a.innerKey(KeysA::Pose()), pose_b = node_b.innerKey(KeysB::Pose());
            factors.push_back(std::make_shared<FactorType>(
                std::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(pose_a, pose_b, m_options.offset, gtsam::noiseModel::Gaussian::Covariance(m_offset_covariance)),
                node_a, node_b));

            if constexpr (joins_velocity)
            {
                const gtsam::Key velocity_a = node_a.innerKey(KeysA::Velocity()), velocity_b = node_b.innerKey(KeysB::Velocity());
                const gtsam::Pose3 offset = m_options.offset;
                const Matrix6 sigma = m_offset_covariance;
                factors.push_back(std::make_shared<FactorType>(
                    std::make_shared<RigidVelocityFactor>(velocity_a, velocity_b, offset, gtsam::noiseModel::Isotropic::Sigma(6, m_options.velocity_sigma_floor)),
                    node_a, node_b,
                    [velocity_a, offset, sigma](const gtsam::Values &inner_values)
                    { return Eigen::MatrixXd(RigidVelocityFactor::offsetCovariance(inner_values.at<Vector6>(velocity_a), offset, sigma)); }));
            }
        }

        void addSphericalFactors(const Priors::InterpolatedNode<InterpolatorA> &node_a, const Priors::InterpolatedNode<InterpolatorB> &node_b, Factors &factors) const
        {
            const gtsam::Key pose_a = node_a.innerKey(KeysA::Pose()), pose_b = node_b.innerKey(KeysB::Pose());
            const Vector3 r = m_options.offset.translation();
            factors.push_back(std::make_shared<FactorType>(
                std::make_shared<SphericalPositionFactor>(pose_a, pose_b, r, gtsam::noiseModel::Gaussian::Covariance(m_offset_covariance)),
                node_a, node_b));

            if constexpr (joins_velocity)
            {
                const gtsam::Key velocity_a = node_a.innerKey(KeysA::Velocity()), velocity_b = node_b.innerKey(KeysB::Velocity());
                const Matrix3 sigma = m_offset_covariance;
                factors.push_back(std::make_shared<FactorType>(
                    std::make_shared<SphericalVelocityFactor>(pose_a, velocity_a, pose_b, velocity_b, r,
                                                              gtsam::noiseModel::Isotropic::Sigma(3, m_options.velocity_sigma_floor)),
                    node_a, node_b,
                    [velocity_a, sigma](const gtsam::Values &inner_values)
                    { return Eigen::MatrixXd(SphericalVelocityFactor::offsetCovariance(inner_values.at<Vector6>(velocity_a), sigma)); }));
            }
        }

        std::shared_ptr<PriorA> m_prior_a;
        std::shared_ptr<PriorB> m_prior_b;
        double m_s_a, m_s_b;
        Eigen::MatrixXd m_offset_covariance;
        JointOptions m_options;
        std::shared_ptr<StateA> m_state_a;
        std::shared_ptr<StateB> m_state_b;
        std::optional<double> m_last_joined_time; // time of the last row joined, so each row is joined once
    };
} // namespace Spacetime::Joints

#endif // JOINT_HPP
