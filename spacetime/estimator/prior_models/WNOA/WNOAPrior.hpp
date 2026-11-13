#ifndef WNOAPRIOR_HPP
#define WNOAPRIOR_HPP

#include <vector>
#include <concepts>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <cassert>

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/nonlinear/CustomFactor.h>
#include <gtsam/nonlinear/WnoaFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>
#include <spacetime/estimator/prior_models/WNOA/legacy/BinarySpaceFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/legacy/BinaryTimeFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/legacy/UnaryFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/Interpolator.hpp>
#include <spacetime/estimator/prior_models/InterpolationFactor.hpp>
#include <spacetime/estimator/prior_models/GridPrior.hpp>

namespace Spacetime::Priors::WNOA
{
    /** @brief Which GP prior factors connect neighbouring nodes. */
    enum class FactorType
    {
        /// gtsam::WnoaMotionFactor + BetweenFactor: the GP defined on the Lie algebra.
        WNOA,
        /// Legacy::BinaryTimeFactor / Legacy::BinarySpaceFactor: the GP defined on the group, with
        /// the legacy code's ad(eps)*varpi strain-velocity coupling in the space factor (the
        /// form behind the TRO paper's published results).
        Legacy,
        /// The same group factors without the coupling term: the space factor of the TRO paper, Eq. (20).
        Paper
    };

    /** @brief Options of the WNOA prior, loaded from the config's "prior.wnoa" section (see Utilities::ConfigLoader::loadWNOAPriorOptions). */
    struct PriorOptions
    {
        /// Which GP prior factors connect neighbouring nodes.
        FactorType factor_type = FactorType::WNOA;
        /// Use each measurement's own noise model for its interpolation factor, without
        /// adding the interpolation's conditional covariance (see InterpolationFactor).
        bool fixed_noise_model = false;
        /// Let each interpolation factor's error() reuse the augmented noise model of its
        /// latest linearization instead of rebuilding it (see InterpolationFactor).
        bool cache_noise_model = true;
        /// Add the unary factor to the first row's base node only, not to every row's.
        bool remove_unary = false;
    };


    /**
     * @class Prior
     * @brief White-noise-on-acceleration prior on the space-time grid: in time the pose
     *        integrates the velocity and the strain is a random walk; in arclength the pose
     *        integrates the strain and the velocity is a random walk (see Functions).
     *
     * Each new row gets time factors to the previous row, a unary factor on its base node
     * (first row only with PriorOptions::remove_unary), and space factors along the row, of
     * the kind selected by PriorOptions::factor_type. Q1..Q4 are assumed diagonal:
     * FactorType::WNOA passes only Q1's and Q2's diagonals to gtsam::WnoaMotionFactor, and
     * the Legacy / Paper factors and the Interpolator invert all four as diagonal.
     */
    template <typename Pose>
    class Prior : public GridPrior<Node<Pose>, Interpolator<Pose>>
    {
    public:
        using NodeType = Node<Pose>;
        using ValuesType = typename NodeType::ValuesType;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using InterpolatorType = Interpolator<Pose>;
        using Base = GridPrior<NodeType, InterpolatorType>;

        // Check that Pose type is a testable Lie group
        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        // We currently support vector spaces and Lie groups
        static_assert(std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                     gtsam::lie_group_tag> ||
                          std::is_same_v<typename gtsam::traits<Pose>::structure_category,
                                         gtsam::vector_space_tag>,
                      "Pose type must be either a Lie group or vector space");

        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

        static constexpr int dim = gtsam::traits<Pose>::dimension;
        static constexpr int node_dim = NodeType::dim;

        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using Matrix3N = Eigen::Matrix<double, 3 * dim, 3 * dim>;
        using MatrixNode = Eigen::Matrix<double, node_dim, node_dim>;
        using VectorNode = Eigen::Matrix<double, node_dim, 1>;

        /**
         * @param state The grid state this prior extends.
         * @param base_start_point Reference base-node values for the unary factor (pose T_ib;
         *        strain and velocity in GTSAM [rotational, translational] order).
         * @param Q0 Unary factor covariance over [pose, strain, velocity]; FactorType::WNOA
         *        uses only its three diagonal blocks.
         * @param Q1 Time-axis (pose, velocity) WNOA process PSD.
         * @param Q2 Arclength-axis (pose, strain) WNOA process PSD.
         * @param Q3 Strain random-walk-in-time PSD.
         * @param Q4 Velocity random-walk-in-arclength PSD.
         * @param options See PriorOptions.
         */
        Prior(
            std::shared_ptr<StateType> state,
            const Values<Pose> &base_start_point,
            const Matrix3N &Q0 = Matrix3N::Identity(),
            const MatrixN &Q1 = MatrixN::Identity(),
            const MatrixN &Q2 = MatrixN::Identity(),
            const MatrixN &Q3 = MatrixN::Identity(),
            const MatrixN &Q4 = MatrixN::Identity(),
            const PriorOptions &options = PriorOptions())
            : Base(state, InterpolatorType(Q1, Q2, Q3, Q4), options.fixed_noise_model, options.cache_noise_model, "WNOA::Prior"),
              m_node_reference(base_start_point), m_Q0(Q0), m_Q1(Q1), m_Q2(Q2), m_Q3(Q3), m_Q4(Q4), m_options(options) {}

        /// The options this prior was built with.
        const PriorOptions &options() const { return m_options; }

        /**
         * @brief Append a node row at `time` (see GridPrior::beginRow) and return its prior
         *        factors. Initial values average the zero-error predictions from the
         *        previous row and from the unary reference / previous spatial node.
         * @return No factors if `time` is not after the last row.
         */
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors;
            if (!this->beginRow(km, time))
                return new_factors;
            auto state = this->state();

            // Add relevant factors for new nodes
            for (unsigned int i = state->estimation_nodes.size() - state->spatial_dim; i < state->estimation_nodes.size(); i++)
            {
                auto current_node = std::dynamic_pointer_cast<NodeType>(state->estimation_nodes.at(i));
                if (!current_node)
                    throw std::runtime_error("WNOA::Prior::extendState expected a WNOA node in the state.");

                // Add time factor if applicable
                typename NodeType::ValuesType temporal_reference_values;
                typename NodeType::ValuesType spatial_reference_values;

                // Add time factor if there is a previous temporal node to connect to
                std::shared_ptr<NodeType> previous_temporal_node = state->getPreviousTemporalNode(current_node);
                if (previous_temporal_node != nullptr)
                {
                    double dt = current_node->time - previous_temporal_node->time;
                    if (m_options.factor_type != FactorType::WNOA)
                        new_factors.push_back(Legacy::BinaryTimeFactor::createFactor<Pose>(previous_temporal_node, current_node, m_Q1, m_Q3));
                    else
                    {
                        new_factors.push_back(
                            std::make_shared<gtsam::WnoaMotionFactor<Pose>>(
                                previous_temporal_node->poseKey(),     // pose
                                previous_temporal_node->velocityKey(), // varpi
                                current_node->poseKey(),               // pose
                                current_node->velocityKey(),           // varpi
                                dt,
                                m_Q1.diagonal()));
                        new_factors.push_back(
                            std::make_shared<gtsam::BetweenFactor<typename gtsam::traits<Pose>::TangentVector>>(
                                previous_temporal_node->strainKey(), // epsilon
                                current_node->strainKey(),           // epsilon
                                gtsam::traits<Pose>::TangentVector::Zero(),
                                gtsam::noiseModel::Gaussian::Covariance(dt * m_Q3)));
                    }

                    typename NodeType::ValuesType previous_temporal_values = previous_temporal_node->getValues(current_values);
                    temporal_reference_values = Legacy::BinaryTimeFactor::Factor<Pose>::getZeroErrorNode(previous_temporal_values, dt);
                }

                // Add unary factor to the base node if applicable
                if (i == state->estimation_nodes.size() - state->spatial_dim)
                {
                    // Boundary (unary) factors on the base node of every row, or of the first
                    // row only with remove_unary
                    if (!m_options.remove_unary || previous_temporal_node == nullptr)
                    {
                        if (m_options.factor_type != FactorType::WNOA)
                        {
                            // Exact Jacobian (gtsam::PriorFactor<Pose> uses an identity one, which is
                            // inaccurate far from the reference).
                            new_factors.push_back(Legacy::UnaryFactor::createFactor<Pose>(current_node, m_node_reference, m_Q0));
                        }
                        else
                        {
                            new_factors.push_back(
                                std::make_shared<gtsam::PriorFactor<Pose>>(
                                    current_node->poseKey(), // pose
                                    m_node_reference.pose(),
                                    gtsam::noiseModel::Gaussian::Covariance(m_Q0.template block<dim, dim>(0, 0))));
                            new_factors.push_back(
                                std::make_shared<gtsam::PriorFactor<typename gtsam::traits<Pose>::TangentVector>>(
                                    current_node->strainKey(), // epsilon
                                    m_node_reference.epsilon(),
                                    gtsam::noiseModel::Gaussian::Covariance(m_Q0.template block<dim, dim>(dim, dim))));
                            new_factors.push_back(
                                std::make_shared<gtsam::PriorFactor<typename gtsam::traits<Pose>::TangentVector>>(
                                    current_node->velocityKey(), // varpi
                                    m_node_reference.varpi(),
                                    gtsam::noiseModel::Gaussian::Covariance(m_Q0.template block<dim, dim>(dim * 2, dim * 2))));
                        }
                    }

                    // typename NodeType::ValuesType unary_reference_values = unary_factor->getZeroErrorNode();

                    typename NodeType::ValuesType initial_values = (previous_temporal_node != nullptr)
                                                                       ? Functions<Pose>::average(temporal_reference_values, m_node_reference)
                                                                       : m_node_reference;
                    current_node->setValues(new_values, initial_values);
                }
                // Add spatial factors if applicable
                else if (i > state->estimation_nodes.size() - state->spatial_dim)
                {
                    auto previous_spatial_node = std::dynamic_pointer_cast<NodeType>(state->estimation_nodes.at(i - 1));
                    if (!previous_spatial_node)
                        throw std::runtime_error("WNOA::Prior::extendState expected a WNOA node in the state.");
                    double ds = current_node->arclength - previous_spatial_node->arclength;
                    if (m_options.factor_type != FactorType::WNOA)
                        new_factors.push_back(Legacy::BinarySpaceFactor::createFactor<Pose>(previous_spatial_node, current_node, m_Q2, m_Q4, m_options.factor_type == FactorType::Legacy));
                    else
                    {
                        new_factors.push_back(
                            std::make_shared<gtsam::WnoaMotionFactor<Pose>>(
                                previous_spatial_node->poseKey(),   // pose
                                previous_spatial_node->strainKey(), // epsilon
                                current_node->poseKey(),            // pose
                                current_node->strainKey(),          // epsilon
                                ds,
                                m_Q2.diagonal()));
                        new_factors.push_back(
                            std::make_shared<gtsam::BetweenFactor<typename gtsam::traits<Pose>::TangentVector>>(
                                previous_spatial_node->velocityKey(), // varpi
                                current_node->velocityKey(),          // varpi
                                gtsam::traits<Pose>::TangentVector::Zero(),
                                gtsam::noiseModel::Gaussian::Covariance(ds * m_Q4)));
                    }

                    typename NodeType::ValuesType previous_spatial_values = previous_spatial_node->getValues(new_values);
                    // FactorType::WNOA keeps the coupled guess it has always used.
                    spatial_reference_values = Legacy::BinarySpaceFactor::Factor<Pose>::getZeroErrorNode(previous_spatial_values, ds, m_options.factor_type != FactorType::Paper);

                    if (previous_temporal_node != nullptr)
                        current_node->setValues(new_values, Functions<Pose>::average(temporal_reference_values, spatial_reference_values));
                    else
                        current_node->setValues(new_values, spatial_reference_values);
                }
            }

            return new_factors;
        }

    private:
        Values<Pose> m_node_reference;
        Matrix3N m_Q0;
        MatrixN m_Q1;
        MatrixN m_Q2;
        MatrixN m_Q3;
        MatrixN m_Q4;
        PriorOptions m_options;
    };
} // namespace Spacetime::Priors::WNOA

#endif // WNOAPRIOR_HPP
