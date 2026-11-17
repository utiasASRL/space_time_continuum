#ifndef WNOAQUASISTATICPRIOR_HPP
#define WNOAQUASISTATICPRIOR_HPP

#include <memory>
#include <vector>

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/WnoaFactor.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/GridPrior.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/WNOAQuasiStaticState.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/Interpolator.hpp>

namespace Spacetime::Priors::WNOAQuasiStatic
{
    /** @brief Options of the quasi-static prior, loaded from the config's "prior.wnoa" section (see Utilities::ConfigLoader::loadWNOAQuasiStaticPriorOptions). */
    struct PriorOptions
    {
        /// Use each measurement's own noise model for its interpolation factor, without
        /// adding the interpolation's conditional covariance (see InterpolationFactor).
        bool fixed_noise_model = false;
        /// Let each interpolation factor's error() reuse the augmented noise model of its
        /// latest linearization instead of rebuilding it (see InterpolationFactor).
        bool cache_noise_model = true;
    };

    /**
     * @class Prior
     * @brief Quasi-static prior: the arclength dimension of the WNOA prior alone, i.e. the
     *        quasi-static GP prior of Lilge et al. (2022) that the spatio-temporal WNOA
     *        prior reduces to with K = 1 and no velocity.
     *
     * Every node row is an independent shape estimate, with gtsam's own factors (the same
     * ones as WNOA::Prior's FactorType::Experimental):
     *  - unary gtsam::PriorFactor<Pose> and gtsam::PriorFactor<TangentVector> on its base
     *    node's pose and strain, from the pose and strain blocks of Q0;
     *  - a gtsam::WnoaMotionFactor on (pose, strain) between arclength neighbours, with PSD Q2.
     * No factor connects two rows. Rows are meant to be placed at measurement times (see
     * EstimatorOptions::quasi_static); a query between two rows takes the nearer one (see
     * WNOAQuasiStatic::Interpolator).
     *
     * @tparam Pose Pose type for states (e.g., Pose3).
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
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;

        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

        static constexpr int dim = gtsam::traits<Pose>::dimension; ///< Pose tangent dimension.
        using MatrixN = Eigen::Matrix<double, dim, dim>;            ///< PSD block (Q2).
        using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;   ///< [pose, strain] block (Q0).

        /**
         * @param state The grid state this prior extends.
         * @param base_start_point Reference base-node values for the unary factors (pose
         *        T_ib; strain in GTSAM [rotational, translational] order). Also the strain
         *        of the initial guess.
         * @param Q0 Unary factor covariance over [pose, strain]; only its two diagonal blocks are used.
         * @param Q2 Arclength-axis (pose, strain) WNOA process PSD.
         * @param options See PriorOptions.
         */
        Prior(
            std::shared_ptr<StateType> state,
            const ValuesType &base_start_point,
            const Matrix2N &Q0 = Matrix2N::Identity(),
            const MatrixN &Q2 = MatrixN::Identity(),
            const PriorOptions &options = PriorOptions())
            : Base(state, InterpolatorType(Q2), options.fixed_noise_model, options.cache_noise_model, "WNOAQuasiStatic::Prior"),
              m_node_reference(base_start_point), m_Q0(Q0), m_Q2(Q2), m_options(options) {}

        /// The options this prior was built with.
        const PriorOptions &options() const { return m_options; }

        /**
         * @brief Append a node row at `time` (see GridPrior::beginRow) and return its prior
         *        factors. The row is initialized as the zero-error shape: constant strain
         *        from the reference base node.
         * @return No factors if `time` is not after the last row.
         */
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            (void)current_values; // rows are independent
            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors;
            if (!this->beginRow(km, time))
                return new_factors;
            auto state = this->state();

            const unsigned int row_start = state->estimation_nodes.size() - state->spatial_dim;
            const std::shared_ptr<NodeType> &base_node = state->estimation_nodes.at(row_start);
            new_factors.push_back(std::make_shared<gtsam::PriorFactor<Pose>>(
                base_node->poseKey(), m_node_reference.pose(),
                gtsam::noiseModel::Gaussian::Covariance(m_Q0.template topLeftCorner<dim, dim>())));
            new_factors.push_back(std::make_shared<gtsam::PriorFactor<TangentVector>>(
                base_node->strainKey(), m_node_reference.epsilon(),
                gtsam::noiseModel::Gaussian::Covariance(m_Q0.template bottomRightCorner<dim, dim>())));
            base_node->setValues(new_values, m_node_reference);

            for (unsigned int i = row_start + 1; i < state->estimation_nodes.size(); i++)
            {
                const std::shared_ptr<NodeType> &previous_node = state->estimation_nodes.at(i - 1);
                const std::shared_ptr<NodeType> &current_node = state->estimation_nodes.at(i);
                const double ds = current_node->arclength - previous_node->arclength;
                new_factors.push_back(std::make_shared<gtsam::WnoaMotionFactor<Pose>>(
                    previous_node->poseKey(), previous_node->strainKey(),
                    current_node->poseKey(), current_node->strainKey(),
                    ds, m_Q2.diagonal()));

                const ValuesType previous = previous_node->getValues(new_values);
                const ValuesType initial(previous.pose() * gtsam::traits<Pose>::Expmap(ds * previous.epsilon()), previous.epsilon());
                current_node->setValues(new_values, initial);
            }

            return new_factors;
        }

    private:
        ValuesType m_node_reference;
        Matrix2N m_Q0;
        MatrixN m_Q2;
        PriorOptions m_options;
    };
} // namespace Spacetime::Priors::WNOAQuasiStatic

#endif // WNOAQUASISTATICPRIOR_HPP
