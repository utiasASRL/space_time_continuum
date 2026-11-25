#ifndef WNOAPRIOR_HPP
#define WNOAPRIOR_HPP

#include <vector>
#include <concepts>
#include <array>
#include <cmath>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <iterator>
#include <cassert>

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/nonlinear/CustomFactor.h>

#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAState.hpp>
#include <spacetime/estimator/prior_models/WNOA/BinarySpaceFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/BinaryTimeFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/UnaryFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/experimental/BinarySpaceFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/experimental/BinaryTimeFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/experimental/UnaryFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA/functions.hpp>
#include <spacetime/estimator/prior_models/WNOA/Interpolator.hpp>
#include <spacetime/estimator/prior_models/WNOA/Covariances.hpp>
#include <spacetime/estimator/prior_models/InterpolationFactor.hpp>
#include <spacetime/estimator/prior_models/GridPrior.hpp>
#include <spacetime/estimator/prior_models/StraightStrain.hpp>

namespace Spacetime::Priors::WNOA
{
    /** @brief Which GP prior factors connect neighbouring nodes. */
    enum class FactorType
    {
        /// BinaryTimeFactor / BinarySpaceFactor / UnaryFactor: the GP defined on the group, as in
        /// the TRO paper (space factor of Eq. (20)).
        WNOA,
        /// The same group factors with the legacy code's ad(eps)*varpi strain-velocity coupling
        /// in the space factor (the form behind the TRO paper's published results).
        Legacy,
        /// Experimental::BinaryTimeFactor / BinarySpaceFactor / UnaryFactor (gtsam::WnoaMotionFactor,
        /// BetweenFactor and PriorFactor): the GP defined on the Lie algebra.
        Experimental
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
     * FactorType::Experimental passes only Q1's and Q2's diagonals to gtsam::WnoaMotionFactor, and
     * the WNOA / Legacy factors and the Interpolator invert all four as diagonal.
     *
     * Q1..Q4 may vary along the robot (see CovarianceOverrides, given at construction):
     * the time factors at node n use node n's Q1/Q3, the space factors between nodes n and
     * n+1 use segment n's Q2/Q4, and measurement interpolation in the cell [s_n, s_{n+1}]
     * uses both of these.
     *
     * @tparam Pose The pose Lie group (e.g. gtsam::Pose3).
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

        static constexpr int dim = gtsam::traits<Pose>::dimension; ///< Pose tangent dimension.
        static constexpr int node_dim = NodeType::dim;              ///< Node tangent dimension (3 * dim).

        using MatrixN = Eigen::Matrix<double, dim, dim>;            ///< PSD block (Q1..Q4).
        using Matrix3N = Eigen::Matrix<double, 3 * dim, 3 * dim>;   ///< Whole-node block (Q0).
        using MatrixNode = Eigen::Matrix<double, node_dim, node_dim>;
        using VectorNode = Eigen::Matrix<double, node_dim, 1>;

        /**
         * @param state The grid state this prior extends.
         * @param base_start_point Reference base-node values for the unary factor (pose T_ib;
         *        strain and velocity in GTSAM [rotational, translational] order).
         * @param Q0 Unary factor covariance over [pose, strain, velocity]; FactorType::Experimental
         *        uses only its three diagonal blocks.
         * @param Q1 Time-axis (pose, velocity) WNOA process PSD.
         * @param Q2 Arclength-axis (pose, strain) WNOA process PSD.
         * @param Q3 Strain random-walk-in-time PSD.
         * @param Q4 Velocity random-walk-in-arclength PSD.
         * @param options See PriorOptions.
         * @param overrides Per-arclength values of Q1..Q4 replacing the nominal ones above
         *        (see CovarianceOverrides); none by default.
         * Warns if base_start_point's strain does not have the body x axis as its tangent
         * axis (see Priors::warnIfTangentNotX). DefaultCovariances() gives tuned Q0..Q4.
         * @throws std::invalid_argument for an override key that is not a node arclength
         *         (see Priors::CovarianceProfile).
         */
        Prior(
            std::shared_ptr<StateType> state,
            const NodeValues<Pose> &base_start_point,
            const Matrix3N &Q0,
            const MatrixN &Q1,
            const MatrixN &Q2,
            const MatrixN &Q3,
            const MatrixN &Q4,
            const PriorOptions &options = PriorOptions(),
            const CovarianceOverrides<Pose> &overrides = CovarianceOverrides<Pose>())
            : Prior(state, base_start_point, Q0,
                    std::make_shared<const CovarianceProfiles<Pose>>(Q1, Q2, Q3, Q4, stateArclengths(state), overrides),
                    options) {}

        /**
         * @brief A prior with the default tuned covariances (see DefaultCovariances).
         * @param state The grid state this prior extends.
         * @param base_start_point Reference base-node values (see the full constructor).
         * @param options See PriorOptions.
         * @param overrides Per-arclength values of Q1..Q4 (see CovarianceOverrides).
         */
        Prior(std::shared_ptr<StateType> state, const NodeValues<Pose> &base_start_point, const PriorOptions &options = PriorOptions(),
              const CovarianceOverrides<Pose> &overrides = CovarianceOverrides<Pose>())
            : Prior(std::move(state), base_start_point, std::get<0>(DefaultCovariances()), std::get<1>(DefaultCovariances()),
                    std::get<2>(DefaultCovariances()), std::get<3>(DefaultCovariances()), std::get<4>(DefaultCovariances()), options,
                    overrides) {}

    private:
        Prior(std::shared_ptr<StateType> state, const NodeValues<Pose> &base_start_point, const Matrix3N &Q0,
              std::shared_ptr<const CovarianceProfiles<Pose>> profiles, const PriorOptions &options)
            : Base(state, InterpolatorType(profiles), options.fixed_noise_model, options.cache_noise_model, "WNOA::Prior"),
              m_node_reference(base_start_point), m_Q0(Q0), m_Q(std::move(profiles)), m_options(options)
        {
            warnIfTangentNotX<Pose>(base_start_point.epsilon(), "WNOA::Prior");
        }

    public:

        /**
         * @brief The default tuned covariances (Q0, Q1, Q2, Q3, Q4), in constructor order.
         *        For gtsam::Pose3, the experiment tuning of "A Stochastic Framework for
         *        Continuous-Time Estimation of Continuum Robots"
         *        (papers/.../configs/experiment_base.json), scaled by its prior_factor 0.5,
         *        with Q0's rotational-strain diagonal permuted from the paper's z tangent
         *        axis to the library's x (see Priors::straightStrain): torsion (about x) is
         *        held tight and bending (about y and z) is free. Identity for other poses.
         */
        static std::tuple<Matrix3N, MatrixN, MatrixN, MatrixN, MatrixN> DefaultCovariances()
        {
            if constexpr (dim == 6)
            {
                constexpr double prior_factor = 0.5;
                Eigen::Matrix<double, 18, 1> q0;
                q0 << 1e-2, 1e-2, 1e-2, 8e-5, 8e-5, 8e-5, // pose [rot, trans]
                    1e-2, 1e2, 1e2, 1e-2, 1e-2, 1e-2,     // strain [torsion, bending y, bending z, trans]
                    1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2;   // velocity
                const auto diag = [](double r, double t)
                { return MatrixN((Eigen::Matrix<double, 6, 1>() << r, r, r, t, t, t).finished().asDiagonal()); };
                return {prior_factor * Matrix3N(q0.asDiagonal()), prior_factor * diag(3e2, 3e0), prior_factor * diag(5e2, 8e-1),
                        prior_factor * diag(3e2, 3e1), prior_factor * diag(5e1, 3e1)};
            }
            else
                return {Matrix3N::Identity(), MatrixN::Identity(), MatrixN::Identity(), MatrixN::Identity(), MatrixN::Identity()};
        }

        /// The options this prior was built with.
        const PriorOptions &options() const { return m_options; }

        /// Q1..Q4 resolved over the state's nodes and segments (see CovarianceOverrides).
        const CovarianceProfiles<Pose> &covarianceProfiles() const { return *m_Q; }

        /**
         * @brief Append a node row at `time` (see GridPrior::beginRow) and return its prior
         *        factors. Initial values average the zero-error predictions from the
         *        previous row and from the unary reference / previous spatial node.
         * @return No factors if `time` is not after the last row.
         * @throws std::runtime_error if the state holds a node that is not a WNOA::Node<Pose>.
         */
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors;
            if (!this->beginRow(km, time))
                return new_factors;
            auto state = this->state();

            // Add relevant factors for new nodes
            const unsigned int row_start = state->estimation_nodes.size() - state->spatial_dim;
            for (unsigned int i = row_start; i < state->estimation_nodes.size(); i++)
            {
                const unsigned int n = i - row_start; // spatial index
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
                    const MatrixN &Q1 = m_Q->Q1.at(n);
                    const MatrixN &Q3 = m_Q->Q3.at(n);
                    if (m_options.factor_type == FactorType::Experimental)
                        appendFactors(new_factors, Experimental::BinaryTimeFactor::createFactors<Pose>(previous_temporal_node, current_node, Q1, Q3));
                    else
                        new_factors.push_back(BinaryTimeFactor::createFactor<Pose>(previous_temporal_node, current_node, Q1, Q3));

                    typename NodeType::ValuesType previous_temporal_values = previous_temporal_node->getValues(current_values);
                    temporal_reference_values = BinaryTimeFactor::Factor<Pose>::getZeroErrorNode(previous_temporal_values, dt);
                }

                // Add unary factor to the base node if applicable
                if (i == state->estimation_nodes.size() - state->spatial_dim)
                {
                    // Boundary (unary) factors on the base node of every row, or of the first
                    // row only with remove_unary
                    if (!m_options.remove_unary || previous_temporal_node == nullptr)
                    {
                        if (m_options.factor_type == FactorType::Experimental)
                            appendFactors(new_factors, Experimental::UnaryFactor::createFactors<Pose>(current_node, m_node_reference, m_Q0));
                        else
                            new_factors.push_back(UnaryFactor::createFactor<Pose>(current_node, m_node_reference, m_Q0));
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
                    const MatrixN &Q2 = m_Q->Q2.at(n - 1); // the segment [s_{n-1}, s_n]
                    const MatrixN &Q4 = m_Q->Q4.at(n - 1);
                    if (m_options.factor_type == FactorType::Experimental)
                        appendFactors(new_factors, Experimental::BinarySpaceFactor::createFactors<Pose>(previous_spatial_node, current_node, Q2, Q4));
                    else
                        new_factors.push_back(BinarySpaceFactor::createFactor<Pose>(previous_spatial_node, current_node, Q2, Q4, m_options.factor_type == FactorType::Legacy));

                    typename NodeType::ValuesType previous_spatial_values = previous_spatial_node->getValues(new_values);
                    // FactorType::Experimental keeps the coupled guess it has always used.
                    spatial_reference_values = BinarySpaceFactor::Factor<Pose>::getZeroErrorNode(previous_spatial_values, ds, m_options.factor_type != FactorType::WNOA);

                    if (previous_temporal_node != nullptr)
                        current_node->setValues(new_values, Functions<Pose>::average(temporal_reference_values, spatial_reference_values));
                    else
                        current_node->setValues(new_values, spatial_reference_values);
                }
            }

            return new_factors;
        }

    private:
        static void appendFactors(std::vector<std::shared_ptr<gtsam::NonlinearFactor>> &factors, std::vector<std::shared_ptr<gtsam::NonlinearFactor>> &&more)
        {
            factors.insert(factors.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
        }

        NodeValues<Pose> m_node_reference;
        Matrix3N m_Q0;
        std::shared_ptr<const CovarianceProfiles<Pose>> m_Q; // Q1..Q4, shared with the interpolator
        PriorOptions m_options;
    };
} // namespace Spacetime::Priors::WNOA

#endif // WNOAPRIOR_HPP
