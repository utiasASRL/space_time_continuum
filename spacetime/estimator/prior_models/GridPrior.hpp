#ifndef GRIDPRIOR_HPP
#define GRIDPRIOR_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/prior_models/InterpolatedNode.hpp>
#include <spacetime/estimator/prior_models/InterpolationFactor.hpp>

namespace Spacetime::Priors
{
    /**
     * @class GridPrior
     * @brief Prior-agnostic base for priors defined on the ContinuumRobotState
     *        space-time grid (e.g. WNOA::Prior, WNOV::Prior).
     *
     * Everything here depends only on the grid's topology and on an interpolator
     * satisfying the Priors::InterpolationFactor contract (e.g. a
     * Priors::GridInterpolator): selecting a query's bordering nodes, wrapping a sensor's
     * inner factor in an InterpolationFactor, resolving read-only queries, and choosing
     * which posterior covariance blocks to extract. A derived prior only supplies its
     * factors and initial values, in extendState.
     *
     * @tparam NodeT The prior's node type.
     * @tparam InterpolatorT The prior's interpolator, whose NodeType is NodeT.
     */
    template <typename NodeT, typename InterpolatorT>
    class GridPrior : public Types::StatefulModel<Types::ContinuumRobotState<NodeT>>
    {
    public:
        using NodeType = NodeT;
        using ValuesType = typename NodeT::ValuesType;
        using StateType = Types::ContinuumRobotState<NodeT>;
        using InterpolatorType = InterpolatorT;

        static_assert(std::is_same_v<typename InterpolatorT::NodeType, NodeT>,
                      "GridPrior's interpolator must operate on the prior's own node type.");

        /**
         * @param state The grid state this prior extends.
         * @param interpolator The prior's interpolator, used for measurement factors and queries.
         * @param fixed_noise_model Use each measurement's own noise model for its
         *        interpolation factor, without adding the interpolation's conditional
         *        covariance (see InterpolationFactor).
         * @param cache_noise_model Let each interpolation factor reuse the augmented noise
         *        model of its latest linearization when evaluating its error (see
         *        InterpolationFactor).
         * @param name Prefix of this prior's warnings (e.g. "WNOA::Prior").
         */
        GridPrior(std::shared_ptr<StateType> state, const InterpolatorT &interpolator,
                  bool fixed_noise_model, bool cache_noise_model, std::string name)
            : Types::StatefulModel<StateType>(std::move(state)), m_interpolator(interpolator),
              m_fixed_noise_model(fixed_noise_model), m_cache_noise_model(cache_noise_model), m_name(std::move(name)) {}

        /// The interpolator this prior builds its measurement factors and queries with.
        const InterpolatorT &interpolator() const { return m_interpolator; }

        /// The earliest row time the smoother still keeps (see onMarginalization); -inf before any marginalization.
        double windowStart() const { return m_window_start; }

        /**
         * @brief The query point (time, arclength) on its bordering nodes (selected exactly as
         *        for getOuterFactor and query), its placeholder keys renamed by
         *        `key_map` (see InterpolatedNode), for factors that interpolate more than one
         *        node (e.g. TwoSidedInterpolationFactor).
         * @return nullopt if the query precedes the earliest node.
         */
        std::optional<InterpolatedNode<InterpolatorT>> interpolatedNode(
            double time, double arclength, const typename InterpolatedNode<InterpolatorT>::KeyMap &key_map = {}) const
        {
            std::shared_ptr<NodeType> node00, node01, node10, node11;
            if (!selectCorners(time, arclength, node00, node01, node10, node11))
                return std::nullopt;
            return InterpolatedNode<InterpolatorT>(m_interpolator, time, arclength, node00, node01, node10, node11, key_map);
        }

        /**
         * @brief Wrap `inner_factor` in an InterpolationFactor on the bordering nodes of
         *        (time, arclength) (see StatefulModel::getOuterFactor).
         * @return nullptr, with a warning, if the query precedes the earliest node or its
         *         bordering nodes were already marginalized (see onMarginalization).
         */
        std::shared_ptr<gtsam::NonlinearFactor> getOuterFactor(
            double time,
            double arclength,
            const gtsam::NoiseModelFactor::shared_ptr &inner_factor) const override
        {
            std::shared_ptr<NodeType> node00, node01, node10, node11;
            if (!selectCorners(time, arclength, node00, node01, node10, node11))
            {
                warning() << m_name << ": cannot resolve (t=" << time << ", s=" << arclength
                          << "): it lies before the estimated state's earliest node "
                             "(extrapolation is forward-only).";
                return nullptr;
            }

            // Corner selection comes from the state's topology, which never prunes
            // marginalized rows, so a measurement that arrives late (e.g. a real-time
            // dataloader thread pushing it after the estimate at a later time already
            // drained its queue) can land on a row the smoother no longer holds. Its factor
            // would reference marginalized keys and throw in the smoother update, so drop it.
            // Checking node00 suffices: every other corner is at or after it.
            if (node00->time < m_window_start - TOLERANCE)
            {
                warning() << m_name << ": cannot attach (t=" << time << ", s=" << arclength
                          << "): its bordering nodes were already marginalized (the window starts at t="
                          << m_window_start << ").";
                return nullptr;
            }

            // With one node per estimate() at the current time, a measurement drained at
            // time <= current always has a node at or after it, so temporal extrapolation
            // should never happen in practice (spatial extrapolation only for a sensor past
            // the last arclength node). It is still supported -- but a factor built this way
            // is permanent: nothing replaces it later except the smoother's marginalization.
            warnIfExtrapolated(time, arclength, node00, node01, node10);

            return std::make_shared<InterpolationFactor<InterpolatorT>>(
                inner_factor, m_interpolator, time, arclength, node00, node01, node10, node11, m_fixed_noise_model,
                m_cache_noise_model);
        }

        /**
         * @brief Interpolate (or extrapolate) the node state at (time, arclength) from
         *        `values` (see StatefulModel::query for the outputs).
         * @return nullptr if the query precedes the earliest node or a bordering node's
         *         variables are absent from `values` (e.g. marginalized).
         */
        std::shared_ptr<Types::NodeValues> query(
            double time,
            double arclength,
            const gtsam::Values &values,
            Eigen::MatrixXd *conditionalCovariance = nullptr,
            const PosteriorCovariance *posterior = nullptr,
            Eigen::MatrixXd *posteriorCovariance = nullptr) const override
        {
            return resolveQuery(time, arclength, values, nullptr, conditionalCovariance, posterior, posteriorCovariance);
        }

        /**
         * @brief query() that also outputs the interpolated node's Jacobian w.r.t. each of its
         *        bordering nodes' keys (see GridInterpolator::interpolate's `jacobians`).
         * @param jacobians Output, cleared first: corner key -> node_dim x (that key's dim) block.
         * @return nullptr (with `jacobians` left empty) when query() would return nullptr.
         */
        std::shared_ptr<Types::NodeValues> queryJacobians(
            double time,
            double arclength,
            const gtsam::Values &values,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> &jacobians) const
        {
            jacobians.clear();
            return resolveQuery(time, arclength, values, &jacobians, nullptr, nullptr, nullptr);
        }

        /**
         * @brief Records the start of the window the smoother keeps, so that
         *        getOuterFactor refuses measurements on rows it marginalizes. Only
         *        factor creation is gated: query() checks liveness against its Values.
         */
        void onMarginalization(double cutoff, const gtsam::Values &values) override
        {
            (void)values;
            m_window_start = std::max(m_window_start, cutoff);
        }

        /**
         * @brief Extract, for every node still in `values`, its own variance and its
         *        covariance with its neighbours in time and in arclength.
         *
         * Wherever two consecutive time rows are both live, one group per grid cell (its
         * four corners) is emitted. A cell's joint marginal contains the corners' variances,
         * both time- and both arclength-neighbour blocks, and the two diagonal blocks, which
         * is exactly what 2D (space-then-time) posterior interpolation needs (for WNOA, the
         * 72 x 72 P_{n,k}^{(s,t)} of the paper's eq. (38)). A live row with no live temporal
         * neighbour falls back to arclength-neighbour pairs, and a lone node to itself.
         */
        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
        {
            auto state = this->state();
            const unsigned int spatial_dim = state->spatial_dim;
            const unsigned int num_rows = state->estimation_nodes.size() / spatial_dim;

            auto node_at = [&](unsigned int row, unsigned int n)
            { return state->estimation_nodes.at(row * spatial_dim + n); };
            auto row_is_live = [&](unsigned int row)
            {
                for (unsigned int n = 0; n < spatial_dim; n++)
                    for (const gtsam::Key key : node_at(row, n)->keys())
                        if (!values.exists(key))
                            return false;
                return true;
            };
            auto append_keys = [](gtsam::KeyVector &group, const auto &node)
            {
                const gtsam::KeyVector keys = node->keys();
                group.insert(group.end(), keys.begin(), keys.end());
            };

            std::vector<bool> live(num_rows);
            for (unsigned int row = 0; row < num_rows; row++)
                live[row] = row_is_live(row);

            std::vector<gtsam::KeyVector> groups;
            for (unsigned int row = 0; row < num_rows; row++)
            {
                if (!live[row])
                    continue;

                if (row + 1 < num_rows && live[row + 1])
                {
                    // Cells between this row and the next (time pairs if there is only one
                    // spatial node, so no cells exist).
                    const unsigned int span = spatial_dim > 1 ? 2 : 1;
                    for (unsigned int n = 0; n + span <= spatial_dim; n++)
                    {
                        gtsam::KeyVector group;
                        for (unsigned int dn = 0; dn < span; dn++)
                        {
                            append_keys(group, node_at(row, n + dn));
                            append_keys(group, node_at(row + 1, n + dn));
                        }
                        groups.push_back(std::move(group));
                    }
                }
                else if (row == 0 || !live[row - 1])
                {
                    // An isolated row: covered by no cell, so extract its arclength pairs
                    // (or the lone node).
                    const unsigned int span = spatial_dim > 1 ? 2 : 1;
                    for (unsigned int n = 0; n + span <= spatial_dim; n++)
                    {
                        gtsam::KeyVector group;
                        for (unsigned int dn = 0; dn < span; dn++)
                            append_keys(group, node_at(row, n + dn));
                        groups.push_back(std::move(group));
                    }
                }
            }
            return groups;
        }

    protected:
        /**
         * @brief Append a new node row at `time`, unless one already exists there or later.
         *
         * Exactly one row per call, at the requested time. The estimation loop, not the
         * prior, owns the spacing (see Interface::EstimationSchedule), and warns when it
         * runs too slowly.
         *
         * @return false (and no row added) if `time` is not strictly after the last row --
         *         extendState is idempotent and never creates a zero-length or backwards
         *         interval.
         */
        bool beginRow(KeyManager &km, double time) const
        {
            auto state = this->state();
            if (!state->estimation_nodes.empty() && time <= state->estimation_nodes.back()->time + TOLERANCE)
                return false;

            state->addStateAtTime(km, time);
            return true;
        }

    private:
        /// query()'s body, with the interpolator's optional per-key Jacobians output.
        std::shared_ptr<Types::NodeValues> resolveQuery(
            double time,
            double arclength,
            const gtsam::Values &values,
            std::unordered_map<gtsam::Key, Eigen::MatrixXd> *jacobians,
            Eigen::MatrixXd *conditionalCovariance,
            const PosteriorCovariance *posterior,
            Eigen::MatrixXd *posteriorCovariance) const
        {
            std::shared_ptr<NodeType> node00, node01, node10, node11;
            if (!selectCorners(time, arclength, node00, node01, node10, node11))
                return nullptr;

            // The bordering nodes exist in the state's topology but their variables may
            // already have been marginalized out of `values` by a fixed-lag smoother (the
            // state never prunes estimation_nodes). Node::getValues would throw on that,
            // so report "unavailable" instead.
            for (const auto &corner : {node00, node01, node10, node11})
            {
                if (!corner)
                    continue;
                for (const gtsam::Key key : corner->keys())
                    if (!values.exists(key))
                        return nullptr;
            }

            // No warning here: extrapolating a state QUERY (e.g. predicting past the newest
            // node) is a legitimate use, unlike attaching a measurement that way.
            return m_interpolator.interpolate(
                time, arclength, values, node00, node01, node10, node11,
                jacobians, conditionalCovariance, posterior, posteriorCovariance);
        }

        /**
         * @brief Select the bordering nodes for a space-time query, shared by
         *        getOuterFactor and query so the two can never disagree.
         *
         * A bordering node is fetched for an axis only when that axis genuinely
         * interpolates: the query is off node00's coordinate AND a next node exists on
         * that axis. Leaving it null is exactly how the interpolator is told to
         * extrapolate along that axis instead (see GridInterpolator's dispatch
         * documentation), so a query past the end of the grid resolves against fewer
         * corners rather than being rejected.
         *
         * @return false when no anchor node exists at all -- i.e. the query precedes the
         *         estimated state's earliest node in arclength or time. Extrapolation is
         *         forward-only, so such queries cannot be served.
         */
        bool selectCorners(
            double time, double arclength,
            std::shared_ptr<NodeType> &node00,
            std::shared_ptr<NodeType> &node01,
            std::shared_ptr<NodeType> &node10,
            std::shared_ptr<NodeType> &node11) const
        {
            auto state = this->state();

            node00 = state->getClosestShiftedNode(arclength, time, -1, -1);
            node01 = nullptr;
            node10 = nullptr;
            node11 = nullptr;
            if (!node00)
                return false;

            if (std::abs(time - node00->time) >= TOLERANCE)
                node01 = state->getNextTemporalNode(node00);
            if (std::abs(arclength - node00->arclength) >= TOLERANCE)
                node10 = state->getNextSpatialNode(node00);
            if (node01 && node10)
                node11 = state->getNextSpatialNode(node01);

            return true;
        }

        /**
         * @brief Warn that a measurement factor is being built by extrapolation rather than
         *        interpolation, naming each extrapolated axis and how far past the anchor
         *        node it reaches.
         *
         * An extrapolated factor conditions on one bordering node instead of two, so it is
         * less accurate, and once in the graph it is never upgraded.
         */
        void warnIfExtrapolated(
            double time, double arclength,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10) const
        {
            const double time_offset = time - node00->time;
            const double arclength_offset = arclength - node00->arclength;
            const bool time_extrapolated = std::abs(time_offset) >= TOLERANCE && !node01;
            const bool space_extrapolated = std::abs(arclength_offset) >= TOLERANCE && !node10;
            if (!time_extrapolated && !space_extrapolated)
                return;

            auto log = warning();
            log << m_name << ": adding an EXTRAPOLATED measurement factor at (t=" << time
                << ", s=" << arclength << "):";
            if (time_extrapolated)
                log << " " << time_offset << " past the newest node in time";
            if (time_extrapolated && space_extrapolated)
                log << " and";
            if (space_extrapolated)
                log << " " << arclength_offset << " past the last node in arclength";
            log << ". It is less accurate than an interpolated factor and will not be upgraded.";
        }

        InterpolatorT m_interpolator;
        bool m_fixed_noise_model;
        bool m_cache_noise_model;
        std::string m_name;
        double m_window_start = -std::numeric_limits<double>::infinity(); // earliest row time the smoother keeps, see onMarginalization
    };
} // namespace Spacetime::Priors

#endif // GRIDPRIOR_HPP
