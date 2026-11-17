#ifndef WNOVPRIOR_HPP
#define WNOVPRIOR_HPP

#include <memory>
#include <optional>
#include <vector>

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/prior_models/GridPrior.hpp>
#include <spacetime/estimator/prior_models/WNOV/WNOVState.hpp>
#include <spacetime/estimator/prior_models/WNOV/Interpolator.hpp>

namespace Spacetime::Priors::WNOV
{
    /** @brief Options of the WNOV prior, loaded from the config's "prior.wnov" section (see Utilities::ConfigLoader::loadWNOVPriorOptions). */
    struct PriorOptions
    {
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
     * @brief White-noise-on-velocity prior: the pose is a random walk (Wiener process) in
     *        both arclength and time, so the estimate between nodes is piecewise linear.
     *
     * Every node row gets, from gtsam's own factors:
     *  - a unary gtsam::PriorFactor<Pose> on its base node, at T0 with covariance Q0 (on the
     *    first row only with PriorOptions::remove_unary);
     *  - a gtsam::BetweenFactor<Pose> between arclength neighbours, measuring the nominal
     *    strain Exp(ds * epsilon0) with covariance ds * Q1 -- e = ln(Exp(ds * epsilon0)^-1
     *    T_1^-1 T_2), about a constant-strain shape of strain epsilon0;
     *  - a gtsam::BetweenFactor<Pose> to the previous row's node at the same arclength,
     *    measuring the identity with covariance dt * Q2 -- e = ln(T_1^-1 T_2).
     * Scaling the covariances by the node spacing is what makes the prior a Wiener
     * process, so that GP interpolation between nodes is exactly geodesic interpolation
     * (see WNOV::Interpolator).
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
        using TangentVector = typename gtsam::traits<Pose>::TangentVector;

        GTSAM_CONCEPT_ASSERT(gtsam::IsTestable<Pose>);
        GTSAM_CONCEPT_ASSERT(gtsam::IsLieGroup<Pose>);

        static constexpr int dim = gtsam::traits<Pose>::dimension; ///< Pose tangent dimension.
        using MatrixN = Eigen::Matrix<double, dim, dim>;            ///< Covariance / PSD block (Q0..Q2).

        /**
         * @param state The grid state this prior extends.
         * @param T0 Base pose (T_ib) the unary factor holds each row's base node to (the
         *        first row's only with PriorOptions::remove_unary).
         * @param epsilon0 Nominal strain, in GTSAM [rotational, translational] order.
         * @param Q0 Unary factor covariance, in GTSAM [rotational, translational] order.
         * @param Q1 Arclength random-walk PSD (covariance per unit arclength).
         * @param Q2 Time random-walk PSD (covariance per unit time).
         * @param options See PriorOptions.
         */
        Prior(
            std::shared_ptr<StateType> state,
            const Pose &T0,
            const TangentVector &epsilon0,
            const MatrixN &Q0 = MatrixN::Identity(),
            const MatrixN &Q1 = MatrixN::Identity(),
            const MatrixN &Q2 = MatrixN::Identity(),
            const PriorOptions &options = PriorOptions())
            : Base(state, InterpolatorType(Q1, Q2, epsilon0), options.fixed_noise_model, options.cache_noise_model, "WNOV::Prior"),
              m_T0(T0), m_epsilon0(epsilon0), m_Q0(Q0), m_Q1(Q1), m_Q2(Q2), m_options(options) {}

        /// The options this prior was built with.
        const PriorOptions &options() const { return m_options; }

        /**
         * @brief Append a node row at `time` (see GridPrior::beginRow) and return its prior
         *        factors. Each node's initial pose is the geodesic midpoint of its
         *        previous-row pose and its spatial (or unary) zero-error prediction.
         * @return No factors if `time` is not after the last row.
         */
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors;
            if (!this->beginRow(km, time))
                return new_factors;
            auto state = this->state();

            const unsigned int row_start = state->estimation_nodes.size() - state->spatial_dim;
            for (unsigned int i = row_start; i < state->estimation_nodes.size(); i++)
            {
                const std::shared_ptr<NodeType> &current_node = state->estimation_nodes.at(i);

                // Time factor to the previous row, whose pose is the zero-error prediction.
                std::optional<Pose> temporal_reference;
                const auto previous_temporal_node = state->getPreviousTemporalNode(current_node);
                if (previous_temporal_node)
                {
                    const double dt = current_node->time - previous_temporal_node->time;
                    new_factors.push_back(std::make_shared<gtsam::BetweenFactor<Pose>>(
                        previous_temporal_node->poseKey(), current_node->poseKey(), Pose(),
                        gtsam::noiseModel::Gaussian::Covariance(dt * m_Q2)));
                    temporal_reference = current_values.at<Pose>(previous_temporal_node->poseKey());
                }

                // Unary factor on the base node (the first row's only, with remove_unary),
                // spatial factor on every other node, each with its zero-error prediction.
                Pose reference;
                if (i == row_start)
                {
                    if (!m_options.remove_unary || !previous_temporal_node)
                        new_factors.push_back(std::make_shared<gtsam::PriorFactor<Pose>>(
                            current_node->poseKey(), m_T0, gtsam::noiseModel::Gaussian::Covariance(m_Q0)));
                    reference = m_T0;
                }
                else
                {
                    const std::shared_ptr<NodeType> &previous_spatial_node = state->estimation_nodes.at(i - 1);
                    const double ds = current_node->arclength - previous_spatial_node->arclength;
                    const Pose nominal_strain = gtsam::traits<Pose>::Expmap(ds * m_epsilon0);
                    new_factors.push_back(std::make_shared<gtsam::BetweenFactor<Pose>>(
                        previous_spatial_node->poseKey(), current_node->poseKey(), nominal_strain,
                        gtsam::noiseModel::Gaussian::Covariance(ds * m_Q1)));
                    reference = new_values.at<Pose>(previous_spatial_node->poseKey()) * nominal_strain;
                }

                const Pose initial = temporal_reference ? gtsam::interpolate<Pose>(*temporal_reference, reference, 0.5) : reference;
                current_node->setValues(new_values, ValuesType(initial));
            }

            return new_factors;
        }

    private:
        Pose m_T0;
        TangentVector m_epsilon0;
        MatrixN m_Q0;
        MatrixN m_Q1;
        MatrixN m_Q2;
        PriorOptions m_options;
    };
} // namespace Spacetime::Priors::WNOV

#endif // WNOVPRIOR_HPP
