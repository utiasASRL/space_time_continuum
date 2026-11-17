#ifndef ENDEFFECTORPRIOR_HPP
#define ENDEFFECTORPRIOR_HPP

#include <memory>
#include <vector>

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WnoaFactor.h>

#include <spacetime/estimator/joint_models/EndEffectorInterpolator.hpp>
#include <spacetime/estimator/joint_models/EndEffectorState.hpp>
#include <spacetime/estimator/prior_models/GridPrior.hpp>

namespace Spacetime::Joints
{
    /**
     * @class EndEffectorPrior
     * @brief The WNOA prior in time of an end effector (a zero-length grid with one node per
     *        row): a gtsam::WnoaMotionFactor between consecutive rows, and nothing else. It
     *        is deliberately not anchored in time; the robots attached to the end effector
     *        (see Joints::EndEffector) position it.
     *
     * A new row's initial guess is the constant-velocity prediction from the previous row's
     * current estimate, and the first row's is the given initial pose, at rest.
     *
     * @tparam Pose Pose type (e.g. gtsam::Pose3).
     */
    template <typename Pose>
    class EndEffectorPrior : public Priors::GridPrior<EndEffectorNode<Pose>, EndEffectorInterpolator<Pose>>
    {
        using Base = Priors::GridPrior<EndEffectorNode<Pose>, EndEffectorInterpolator<Pose>>;

    public:
        using NodeType = EndEffectorNode<Pose>;                            ///< Node: a pose and a body twist.
        using ValuesType = typename NodeType::ValuesType;                  ///< A node's (pose, twist) values.
        using StateType = Types::ContinuumRobotState<NodeType>;            ///< The zero-length grid state.
        using TangentVector = typename gtsam::traits<Pose>::TangentVector; ///< Tangent vector of Pose ([rot, trans]).

        /**
         * @param state The end effector's zero-length state.
         * @param initial_pose Initial guess of the first row's pose (T_ib).
         * @param q_psd_diag Diagonal of the WNOA power spectral density.
         */
        EndEffectorPrior(std::shared_ptr<StateType> state, const Pose &initial_pose, const TangentVector &q_psd_diag)
            : Base(state, EndEffectorInterpolator<Pose>(q_psd_diag), false, true, "EndEffectorPrior"),
              m_initial_pose(initial_pose), m_q_psd_diag(q_psd_diag) {}

        /**
         * @brief Phase 1: adds one row at `time` (if strictly after the last, see
         *        GridPrior::beginRow) and its WnoaMotionFactor to the previous row. The
         *        initial guess is the previous row's pose propagated by its twist over dt,
         *        with the same twist; the first row gets the initial pose at rest.
         */
        Types::Model::Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            Types::Model::Factors new_factors;
            if (!this->beginRow(km, time))
                return new_factors;
            const auto state = this->state();
            const std::shared_ptr<NodeType> &node = state->estimation_nodes.back();
            const std::shared_ptr<NodeType> previous = state->getPreviousTemporalNode(node);
            if (!previous)
            {
                node->setValues(new_values, ValuesType(m_initial_pose, TangentVector::Zero()));
                return new_factors;
            }

            const double dt = node->time - previous->time;
            new_factors.push_back(std::make_shared<gtsam::WnoaMotionFactor<Pose>>(
                previous->poseKey(), previous->velocityKey(), node->poseKey(), node->velocityKey(), dt, m_q_psd_diag));
            const ValuesType last = previous->getValues(current_values.exists(previous->poseKey()) ? current_values : new_values);
            node->setValues(new_values, ValuesType(last.pose() * gtsam::traits<Pose>::Expmap(dt * last.varpi()), last.varpi()));
            return new_factors;
        }

    private:
        Pose m_initial_pose;
        TangentVector m_q_psd_diag;
    };
} // namespace Spacetime::Joints

#endif // ENDEFFECTORPRIOR_HPP
