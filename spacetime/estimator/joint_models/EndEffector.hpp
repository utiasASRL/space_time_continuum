#ifndef ENDEFFECTOR_HPP
#define ENDEFFECTOR_HPP

#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/joint_models/EndEffectorPrior.hpp>
#include <spacetime/estimator/joint_models/EndEffectorState.hpp>
#include <spacetime/estimator/joint_models/Joint.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/Model.hpp>

namespace Spacetime::Joints
{
    /**
     * @class EndEffector
     * @brief A rigid body that robots attach to: a zero-length robot (one node per time
     *        row, at s = 0) holding only a pose and a velocity, with a WNOA prior in time
     *        (see EndEffectorPrior) and no anchor of its own -- the attached robots position it.
     *
     * It is a self-contained, stateful model: it owns its state (see state()), the prior
     * extending it and every Joint attached to it with attach(), so one
     * Estimator::addModel registers all of them (do not add its prior separately). Its
     * outer factors and queries are its prior's, so a sensor can measure it directly. Each
     * attached robot is joint side A, at its own arclength, and the end effector side B, so
     * a joint's offset is T_point_ee (the end effector's frame seen from the robot's point:
     * T_ee = T_point T_point_ee).
     *
     * @tparam Pose Pose type (gtsam::Pose3, which Joint requires).
     */
    template <typename Pose = gtsam::Pose3>
    class EndEffector : public Types::StatefulModel<Types::ContinuumRobotState<EndEffectorNode<Pose>>>
    {
    public:
        using NodeType = EndEffectorNode<Pose>;
        using PriorType = EndEffectorPrior<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using Factors = Types::Model::Factors;
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;

        /**
         * @param initial_pose Initial guess of the end effector's pose (T_ib).
         * @param q_psd_diag Diagonal of the WNOA power spectral density of its motion.
         */
        explicit EndEffector(const Pose &initial_pose, const TangentVector &q_psd_diag = TangentVector::Ones())
            : Types::StatefulModel<StateType>(std::make_shared<StateType>(std::vector<double>{0.0})),
              m_prior(std::make_shared<PriorType>(this->state(), initial_pose, q_psd_diag)) {}

        /**
         * @brief Rigidly (or, per `options`, spherically) attach the point at arclength `s`
         *        of the robot extended by `robot_prior` to this end effector.
         * @param offset_covariance See Joint (6x6 rigid, 3x3 spherical, positive-definite).
         * @param options See JointOptions; its offset is T_point_ee.
         * @return The new joint, which this end effector keeps (see joints()).
         * @throws std::invalid_argument for invalid inputs (see Joint), keeping nothing.
         */
        template <typename RobotPriorT = Priors::WNOA::Prior<Pose>>
        std::shared_ptr<Joint<RobotPriorT, PriorType>> attach(std::shared_ptr<RobotPriorT> robot_prior, double s, const Eigen::MatrixXd &offset_covariance,
                                                               const JointOptions &options = JointOptions())
        {
            auto joint = std::make_shared<Joint<RobotPriorT, PriorType>>(std::move(robot_prior), s, m_prior, 0.0, offset_covariance, options);
            m_joints.push_back(joint);
            return joint;
        }

        /// The joints attached so far, in attach() order.
        const std::vector<std::shared_ptr<Types::Model>> &joints() const { return m_joints; }

        /// The prior extending the end effector's state.
        std::shared_ptr<PriorType> endEffectorPrior() const { return m_prior; }

        /**
         * @brief Phase 1: extends the end effector's prior.
         * @throws std::runtime_error if no robot is attached: nothing would position it.
         */
        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            if (m_joints.empty())
                throw std::runtime_error("EndEffector: no robot is attached; attach at least one before estimating, since nothing else positions it.");
            return m_prior->extendState(km, time, current_values, new_values);
        }

        /// Phase 2: every attached joint's factors.
        Factors getNewFactors(double time, gtsam::Values &new_values) override
        {
            Factors factors = m_prior->getNewFactors(time, new_values);
            for (const auto &joint : m_joints)
            {
                const Factors joint_factors = joint->getNewFactors(time, new_values);
                factors.insert(factors.end(), joint_factors.begin(), joint_factors.end());
            }
            return factors;
        }

        void onMarginalization(double cutoff, const gtsam::Values &current_values) override
        {
            m_prior->onMarginalization(cutoff, current_values);
            for (const auto &joint : m_joints)
                joint->onMarginalization(cutoff, current_values);
        }

        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override { return m_prior->covarianceKeyGroups(values); }

        /// Its prior's outer factor (see StatefulModel::getOuterFactor).
        std::shared_ptr<gtsam::NonlinearFactor> getOuterFactor(double time, double arclength,
                                                               const gtsam::NoiseModelFactor::shared_ptr &inner_factor) const override
        {
            return m_prior->getOuterFactor(time, arclength, inner_factor);
        }

        /// Its prior's query (see StatefulModel::query).
        std::shared_ptr<Types::NodeValues> query(double time, double arclength, const gtsam::Values &values,
                                                 Eigen::MatrixXd *conditionalCovariance = nullptr,
                                                 const PosteriorCovariance *posterior = nullptr,
                                                 Eigen::MatrixXd *posteriorCovariance = nullptr) const override
        {
            return m_prior->query(time, arclength, values, conditionalCovariance, posterior, posteriorCovariance);
        }

    private:
        std::shared_ptr<PriorType> m_prior;
        std::vector<std::shared_ptr<Types::Model>> m_joints;
    };
} // namespace Spacetime::Joints

#endif // ENDEFFECTOR_HPP
