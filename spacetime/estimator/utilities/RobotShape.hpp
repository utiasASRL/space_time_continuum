/**
 * @file RobotShape.hpp
 * @brief The estimated shape of a continuum robot (Types::ContinuumRobotState) at a time,
 *        read from an EstimatorSnapshot through the robot prior's own interpolator: the
 *        backend-agnostic part of drawing a robot, shared by the visualizer backends and
 *        interfaces that publish it (e.g. ROS2).
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_UTILITIES_ROBOTSHAPE_HPP
#define SPACETIME_UTILITIES_ROBOTSHAPE_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>

namespace Spacetime::Utilities
{
    /**
     * @brief Resolves poses along a Types::ContinuumRobotState<NodeT> from a snapshot.
     *        Rows are picked as the latest at or before min(time, snapshot time): estimate()
     *        appends its row before solving but publishes the snapshot holding it only
     *        afterwards, so a newer row may not be in the snapshot yet. Every query takes
     *        the nodes as a ContinuumRobotState::snapshotEstimationNodes() copy, so it is
     *        safe while the estimator extends the state.
     */
    template <Types::NodeType NodeT>
    class RobotShape
    {
    public:
        /// The prior's interpolation scheme (e.g. WNOA::Interpolator::interpolate), see bindInterpolator.
        /// When `posterior` is non-null, the interpolated node's posterior covariance is
        /// written to `posteriorCovariance` (left empty if it cannot be resolved).
        using InterpolateFn = std::function<std::shared_ptr<Types::NodeValues>(
            double time, double arclength, const gtsam::Values &values,
            const std::shared_ptr<NodeT> &node00, const std::shared_ptr<NodeT> &node01,
            const std::shared_ptr<NodeT> &node10, const std::shared_ptr<NodeT> &node11,
            const PosteriorCovariance *posterior, Eigen::MatrixXd *posteriorCovariance)>;

        /// One resolved point along the backbone.
        struct Point
        {
            double arclength;
            /// world_T_point.
            Eigen::Matrix4d pose;
            /// Posterior covariance whose leading 6 x 6 block is the pose's ([rot, trans], body
            /// frame), or empty if not requested or unavailable.
            Eigen::MatrixXd covariance;
        };

        /// The backbone of one row, see backbone().
        struct Backbone
        {
            /// Time of the row drawn (before any temporal interpolation).
            double row_time;
            /// Points per node-to-node segment: node i is points[i * points_per_segment].
            unsigned int points_per_segment;
            /// Every node and the extra points between them, base to tip.
            std::vector<Point> points;
        };

        /// Throws std::invalid_argument if `state` is null.
        explicit RobotShape(std::shared_ptr<const Types::ContinuumRobotState<NodeT>> state)
            : mp_state(std::move(state)), m_spatial_dim(mp_state ? mp_state->spatial_dim : 0)
        {
            if (!mp_state)
                throw std::invalid_argument("RobotShape requires a non-null state.");
        }

        /// The state whose shape is resolved.
        const std::shared_ptr<const Types::ContinuumRobotState<NodeT>> &state() const { return mp_state; }

        /// Sets the interpolator used between nodes and rows; without one, only the true nodes resolve.
        void bindInterpolator(InterpolateFn fn) { m_interpolate = std::move(fn); }

        /**
         * @brief Binds the interpolator of `prior` (e.g. a Priors::WNOA::Prior<Pose3>), the
         *        prior extending the state, with its posterior covariance propagation. Keeps
         *        the prior alive.
         * @throws std::invalid_argument if `prior` is null or extends another state.
         */
        template <typename PriorT>
        void bindPriorInterpolator(std::shared_ptr<PriorT> prior)
        {
            if (!prior || prior->state() != mp_state)
                throw std::invalid_argument("RobotShape::bindPriorInterpolator: the prior must be the one extending the state.");
            bindInterpolator([prior](double time, double arclength, const gtsam::Values &values,
                                     const std::shared_ptr<NodeT> &node00, const std::shared_ptr<NodeT> &node01,
                                     const std::shared_ptr<NodeT> &node10, const std::shared_ptr<NodeT> &node11,
                                     const PosteriorCovariance *posterior, Eigen::MatrixXd *posteriorCovariance)
                             { return prior->interpolator().interpolate(time, arclength, values, node00, node01, node10, node11,
                                                                        nullptr, nullptr, posterior, posteriorCovariance); });
        }

        bool hasInterpolator() const { return static_cast<bool>(m_interpolate); }

        /**
         * @brief Time of the row backbone() resolves for `time` (see the class comment), or
         *        nullopt if there is none or it is not fully in the snapshot (e.g. marginalized).
         */
        std::optional<double> rowTime(double time, const EstimatorSnapshot &snapshot, const std::vector<std::shared_ptr<NodeT>> &nodes) const
        {
            const auto row = latestRow(time, snapshot, nodes);
            if (!row)
                return std::nullopt;
            return nodes[*row * m_spatial_dim]->time;
        }

        /**
         * @brief The estimated pose (world_T_point) at (time, arclength): at the latest row at
         *        or before min(time, snapshot time), interpolating towards the next row only
         *        with `temporal_interpolation` and that row in the snapshot, and otherwise at
         *        the row's own time (never extrapolating). Spatially, it interpolates towards
         *        the next node.
         * @return nullopt without an interpolator, before the robot's base or first row, or
         *         when a bordering node is missing from the snapshot (e.g. marginalized).
         */
        std::optional<Eigen::Matrix4d> poseAt(double time, double arclength, const EstimatorSnapshot &snapshot,
                                              const std::vector<std::shared_ptr<NodeT>> &nodes, bool temporal_interpolation) const
        {
            if (!m_interpolate || m_spatial_dim == 0)
                return std::nullopt;
            const unsigned int num_rows = nodes.size() / m_spatial_dim;

            const double search_time = std::min(time, snapshot.time);
            unsigned int row = 0;
            while (row < num_rows && nodes[row * m_spatial_dim]->time <= search_time)
                row++;
            if (row == 0)
                return std::nullopt;
            row--; // latest row at or before search_time

            constexpr double tolerance = 1e-9;
            unsigned int spatial = 0;
            while (spatial + 1 < m_spatial_dim && nodes[row * m_spatial_dim + spatial + 1]->arclength <= arclength + tolerance)
                spatial++;
            if (arclength < nodes[row * m_spatial_dim]->arclength - tolerance)
                return std::nullopt;

            auto inSnapshot = [&](const std::shared_ptr<NodeT> &node)
            {
                for (const gtsam::Key key : node->keys())
                    if (!snapshot.values.exists(key))
                        return false;
                return true;
            };
            auto at = [&](unsigned int r, unsigned int i)
            { return nodes[r * m_spatial_dim + i]; };

            std::shared_ptr<NodeT> node00 = at(row, spatial), node01, node10, node11;
            if (!inSnapshot(node00))
                return std::nullopt;
            if (arclength - node00->arclength > tolerance && spatial + 1 < m_spatial_dim)
            {
                node10 = at(row, spatial + 1);
                if (!inSnapshot(node10))
                    return std::nullopt;
            }
            if (temporal_interpolation && time - node00->time > tolerance && row + 1 < num_rows && inSnapshot(at(row + 1, spatial)) &&
                (!node10 || inSnapshot(at(row + 1, spatial + 1))))
            {
                node01 = at(row + 1, spatial);
                if (node10)
                    node11 = at(row + 1, spatial + 1);
            }
            const double query_time = node01 ? std::min(time, node01->time) : node00->time;

            auto interpolated = std::static_pointer_cast<typename NodeT::ValuesType>(
                m_interpolate(query_time, arclength, snapshot.values, node00, node01, node10, node11, nullptr, nullptr));
            if (!interpolated)
                return std::nullopt;
            return interpolated->pose().matrix();
        }

        /**
         * @brief The backbone at `time`: the row rowTime() resolves, each node plus
         *        `extra_points` evenly spaced (in arclength) points per node-to-node segment.
         *        With `temporal_interpolation` and the next row fully in the snapshot, it is
         *        interpolated to `time` between the two rows. Without an interpolator, only
         *        the true nodes are resolved (`extra_points` and `temporal_interpolation`
         *        are ignored).
         * @param posterior If non-null, each point's posterior covariance is resolved from it
         *        (extra Jacobian work: pass it only when needed).
         * @return nullopt if rowTime() finds no row.
         */
        std::optional<Backbone> backbone(double time, const EstimatorSnapshot &snapshot, const std::vector<std::shared_ptr<NodeT>> &nodes,
                                         unsigned int extra_points, bool temporal_interpolation,
                                         const PosteriorCovariance *posterior = nullptr) const
        {
            const auto row = latestRow(time, snapshot, nodes);
            if (!row)
                return std::nullopt;
            if (!m_interpolate)
            {
                extra_points = 0;
                temporal_interpolation = false;
            }
            const unsigned int num_rows = nodes.size() / m_spatial_dim;
            const unsigned int next_row = *row + 1;
            const double row_time = nodes[*row * m_spatial_dim]->time;

            // Interpolate between the two rows bordering `time` only if the next one is fully
            // in the snapshot, instead of snapping to row_time.
            const bool next_row_valid = temporal_interpolation && next_row < num_rows && rowInSnapshot(next_row, snapshot, nodes);
            const double next_row_time = next_row_valid ? nodes[next_row * m_spatial_dim]->time : row_time;
            const double query_time = next_row_valid ? std::clamp(time, row_time, next_row_time) : row_time;

            Backbone result{row_time, extra_points + 1, {}};
            result.points.resize(m_spatial_dim + (m_spatial_dim - 1) * extra_points);
            for (unsigned int i = 0; i < m_spatial_dim; i++)
            {
                const auto &node00 = nodes[*row * m_spatial_dim + i];
                Point &point = result.points[i * result.points_per_segment];
                point.arclength = node00->arclength;
                if (next_row_valid)
                {
                    const auto &node01 = nodes[next_row * m_spatial_dim + i];
                    auto interpolated = m_interpolate(query_time, node00->arclength, snapshot.values, node00, node01, nullptr, nullptr,
                                                      posterior, posterior ? &point.covariance : nullptr);
                    point.pose = std::static_pointer_cast<typename NodeT::ValuesType>(interpolated)->pose().matrix();
                }
                else
                {
                    point.pose = node00->getValues(snapshot.values).pose().matrix();
                    if (posterior && posterior->has(node00->poseKey()))
                        point.covariance = posterior->covariance(node00->poseKey());
                }
            }

            // The extra points (full 2D space+time interpolation if next_row_valid, otherwise space-only).
            for (unsigned int i = 0; i + 1 < m_spatial_dim && extra_points > 0; i++)
            {
                const auto &node00 = nodes[*row * m_spatial_dim + i];
                const auto &node10 = nodes[*row * m_spatial_dim + i + 1];
                std::shared_ptr<NodeT> node01 = next_row_valid ? nodes[next_row * m_spatial_dim + i] : nullptr;
                std::shared_ptr<NodeT> node11 = next_row_valid ? nodes[next_row * m_spatial_dim + i + 1] : nullptr;

                for (unsigned int k = 1; k < result.points_per_segment; k++)
                {
                    const double frac = static_cast<double>(k) / result.points_per_segment;
                    Point &point = result.points[i * result.points_per_segment + k];
                    point.arclength = (1.0 - frac) * node00->arclength + frac * node10->arclength;
                    auto interpolated = m_interpolate(query_time, point.arclength, snapshot.values, node00, node01, node10, node11,
                                                      posterior, posterior ? &point.covariance : nullptr);
                    point.pose = std::static_pointer_cast<typename NodeT::ValuesType>(interpolated)->pose().matrix();
                }
            }
            return result;
        }

    private:
        std::shared_ptr<const Types::ContinuumRobotState<NodeT>> mp_state;
        /// Spatial nodes per temporal row.
        unsigned int m_spatial_dim;
        InterpolateFn m_interpolate;

        /// Whether every node of row `row` has its pose in the snapshot.
        bool rowInSnapshot(unsigned int row, const EstimatorSnapshot &snapshot, const std::vector<std::shared_ptr<NodeT>> &nodes) const
        {
            for (unsigned int i = 0; i < m_spatial_dim; i++)
                if (!snapshot.values.exists(nodes[row * m_spatial_dim + i]->poseKey()))
                    return false; // the fixed-lag smoother may have marginalized it out already
            return true;
        }

        /// Index of the latest row at or before min(time, snapshot time), if fully in the snapshot.
        std::optional<unsigned int> latestRow(double time, const EstimatorSnapshot &snapshot, const std::vector<std::shared_ptr<NodeT>> &nodes) const
        {
            if (m_spatial_dim == 0)
                return std::nullopt;
            const unsigned int num_rows = nodes.size() / m_spatial_dim;
            // Binary search: rows are appended in increasing time order (see ContinuumRobotState::addStateAtTime).
            const double search_time = std::min(time, snapshot.time);
            unsigned int lo = 0, hi = num_rows; // hi is exclusive, search over [lo, hi)
            while (lo < hi)
            {
                const unsigned int mid = lo + (hi - lo) / 2;
                if (nodes[mid * m_spatial_dim]->time <= search_time)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo == 0 || !rowInSnapshot(lo - 1, snapshot, nodes))
                return std::nullopt;
            return lo - 1;
        }
    };
} // namespace Spacetime::Utilities

#endif // SPACETIME_UTILITIES_ROBOTSHAPE_HPP
