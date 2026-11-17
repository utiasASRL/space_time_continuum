#ifndef CONTINUUMROBOT_HPP
#define CONTINUUMROBOT_HPP

#include <iostream>
#include <concepts>
#include <map>
#include <limits>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <mutex>
#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <typeindex>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/types/States.hpp>

namespace Spacetime::Types
{
    /**
     * @brief The state of a continuum robot: a grid of NodeT nodes, one row (at fixed
     *        arclengths) per addStateAtTime call. Nodes are indexed by a flat index (order
     *        of creation) and by their (spatial, temporal) grid index.
     *
     * The estimator thread alone extends and reads the state; a visualizer thread reads
     * it only through snapshotEstimationNodes.
     *
     * @tparam NodeT The node type of every grid point (e.g. WNOA::Node<gtsam::Pose3>).
     */
    template <NodeType NodeT>
    struct ContinuumRobotState : public State
    {
        double length;                                     ///< Robot length [m].
        unsigned int spatial_dim;                          ///< Nodes per row.
        std::vector<double> arclengths;                    ///< Node arclengths [m], ascending.
        std::map<double, unsigned int> arclength_to_index; ///< Spatial index of each node arclength.
        
        std::vector<std::shared_ptr<NodeT>> estimation_nodes; ///< Every node, by flat index.

        /// `spatial_dim` nodes evenly spaced over [0, length] [m] (requires spatial_dim >= 2; not checked).
        ContinuumRobotState(double length, unsigned int spatial_dim)
            : length(length), spatial_dim(spatial_dim)
        {
            for (unsigned int n = 0; n < spatial_dim; n++)
            {
                arclengths.push_back(n * length / (spatial_dim - 1));
                arclength_to_index[arclengths.back()] = n;
                // debug() << "Added arclength: " << arclengths.back() << " at index: " << n;
            }
        }
        /// Nodes at the given ascending `arclengths` [m] (non-empty; not checked). `length` is the last one.
        ContinuumRobotState(std::vector<double> arclengths)
            : length(arclengths.back()), spatial_dim(arclengths.size()), arclengths(arclengths)
        {
            for (unsigned int n = 0; n < arclengths.size(); n++)
            {
                arclength_to_index[arclengths[n]] = n;
            }
        } // Constructor for custom node locations on the robot

        /// Flat index of the node at (spatial index, temporal index), or unsigned(-1) if there is none.
        unsigned int getNodeIndex(unsigned int spatial_index, unsigned int temporal_index) const
        {
            NodeGridIndex key{spatial_index, temporal_index};
            return m_node_mapping.contains(key) ? m_node_mapping.at(key) : static_cast<unsigned int>(-1);
        };
        /// (arclength, time) of the node at flat `index`, or (-1, -1) if there is none.
        std::tuple<double, double> getNodeArclengthTime(unsigned int index) const
        {
            if (!m_node_mapping_inverse.contains(index))
                return std::make_tuple(-1.0, -1.0);

            const NodeGridIndex node_grid_index = m_node_mapping_inverse.at(index);
            const unsigned int temporal_index = std::get<1>(node_grid_index);
            if (temporal_index >= m_time_values.size())
                return std::make_tuple(-1.0, -1.0);

            return std::make_tuple(
                arclengths.at(std::get<0>(node_grid_index)),
                m_time_values.at(temporal_index));
        };

        /**
         * @brief Hold the variable `type` (e.g. "pose", "strain", "velocity") of the nodes
         *        at arclength `s` fixed at `value`, instead of estimating it.
         * @param s Arclength of a node (within TOLERANCE), or -1 for the tip.
         * @param t If given, only the node row created at this time (within TOLERANCE) is
         *        locked; otherwise every row is. Applies to rows created after the call.
         * @throws std::invalid_argument if `s` is not a node arclength, `type` is not a
         *         variable of the node type, or `value` has the wrong type for it.
         */
        template <typename T>
        void lock(const std::string &type, double s, const T &value, std::optional<double> t = std::nullopt)
        {
            const double arclength = (s == -1.0) ? arclengths.back() : s;
            auto it = std::find_if(arclengths.begin(), arclengths.end(), [&](double a)
                                   { return std::abs(a - arclength) < TOLERANCE; });
            if (it == arclengths.end())
                throw std::invalid_argument("ContinuumRobotState::lock: s = " + std::to_string(s) + " is not a node arclength (or -1 for the tip).");

            KeyManager probe_keys; // only used to read the node type's variables
            const NodeT probe(probe_keys, 0.0, 0.0);
            const auto names = probe.names();
            if (std::find(names.begin(), names.end(), type) == names.end())
                throw std::invalid_argument("ContinuumRobotState::lock: the node type has no variable \"" + type + "\".");
            if (probe.type(type) != std::type_index(typeid(T)))
                throw std::invalid_argument("ContinuumRobotState::lock: the value's type does not match variable \"" + type + "\".");

            m_locks.push_back(Lock{type, static_cast<unsigned int>(std::distance(arclengths.begin(), it)), t,
                                   [type, value](const NodeT &node, gtsam::Values &values)
                                   { node.insert(type, value, values); }});
        }

        /**
         * @brief Adds a row of nodes at `time` [s], at every arclength, registering their keys
         *        with `km`, and inserts the matching locks' values into lockedValues(). Rows
         *        are expected in increasing time. Warns once for a time-specific lock whose
         *        time was skipped.
         */
        void addStateAtTime(KeyManager &km, double time) override
        {
            const unsigned int temporal_index = m_time_values.size();
            m_time_values.push_back(time);

            for (auto &lock : m_locks)
            {
                if (lock.time && !lock.matched && *lock.time < time - TOLERANCE)
                {
                    warning() << "ContinuumRobotState: no node row was created at t = " << *lock.time << " (next row at t = "
                              << time << "), so the \"" << lock.type << "\" lock at that time was not applied.";
                    lock.matched = true; // warn once
                }
            }

            for (unsigned int spatial_index = 0; spatial_index < arclengths.size(); spatial_index++)
            {
                double arclength = arclengths.at(spatial_index);
                auto new_node = std::make_shared<NodeT>(km, arclength, time);
                {
                    // Locked because in online-visualization mode a visualizer thread
                    // concurrently calls snapshotEstimationNodes() while this (the
                    // estimator thread) appends here -- every other reader of
                    // estimation_nodes/m_node_mapping* below is only ever called
                    // synchronously from this same estimator thread, so it needs no
                    // lock of its own.
                    std::lock_guard<std::mutex> lock(m_nodes_mutex);
                    this->estimation_nodes.push_back(new_node);
                }
                // debug() << "Added node at arclength: " << arclength << ", time: " << time;

                const unsigned int flat_node_index = this->estimation_nodes.size() - 1;
                const NodeGridIndex node_grid_index{spatial_index, temporal_index};
                m_node_mapping[node_grid_index] = flat_node_index;
                m_node_mapping_inverse[flat_node_index] = node_grid_index;
                m_node_pointer_to_flat_index[new_node.get()] = flat_node_index;

                for (auto &lock : m_locks)
                {
                    if (lock.spatial_index != spatial_index || (lock.time && std::abs(*lock.time - time) >= TOLERANCE))
                        continue;
                    if (!this->locked_values_.exists(new_node->key(lock.type)))
                        lock.insert(*new_node, this->locked_values_);
                    lock.matched = true;
                }
            }
        }

        /// Times of the node rows, ascending. Like estimation_nodes, only for the estimator thread.
        const std::vector<double> &times() const override { return m_time_values; }

        /// Thread-safe copy of estimation_nodes for a visualizer thread to read concurrently with the estimator thread's addStateAtTime.
        std::vector<std::shared_ptr<NodeT>> snapshotEstimationNodes() const
        {
            std::lock_guard<std::mutex> lock(m_nodes_mutex);
            return estimation_nodes;
        }

        /// The next node along the robot, or nullptr at the tip. @throws std::runtime_error if `node` is not in the state.
        std::shared_ptr<NodeT> getNextSpatialNode(const std::shared_ptr<NodeBase> &node) const { return getShiftedNode(node, 1, 0); }
        /// The previous node along the robot, or nullptr at the base. See getNextSpatialNode.
        std::shared_ptr<NodeT> getPreviousSpatialNode(const std::shared_ptr<NodeBase> &node) const { return getShiftedNode(node, -1, 0); }
        /// The node at the same arclength in the next row, or nullptr in the latest row. See getNextSpatialNode.
        std::shared_ptr<NodeT> getNextTemporalNode(const std::shared_ptr<NodeBase> &node) const { return getShiftedNode(node, 0, 1); }
        /// The node at the same arclength in the previous row, or nullptr in the first row. See getNextSpatialNode.
        std::shared_ptr<NodeT> getPreviousTemporalNode(const std::shared_ptr<NodeBase> &node) const { return getShiftedNode(node, 0, -1); }

        /**
         * @brief The node nearest (L1 in arclength + time) to (arclength, time) on the side
         *        each direction selects: < 0 at or below, > 0 at or above, 0 either side.
         *        nullptr if there is none. Linear in the number of nodes.
         */
        std::shared_ptr<NodeT> getClosestShiftedNode(double arclength, double time, int spatial_direction, int temporal_direction) const
        {
            // This is similar to getShiftedNode but does not require the input node to already be in the state. The spatial_direction and temporal_direction parameters specify the direction in which to search for the closest node (e.g. spatial_direction = -1 and temporal_direction = 0 would search for the closest node with arclength less than the input arclength and any time). This can be useful for finding the appropriate nodes to connect measurement factors to when the measurement does not correspond exactly to an existing node in the state.

            std::shared_ptr<NodeT> closest_node = nullptr;
            double closest_distance = std::numeric_limits<double>::infinity();

            for (const auto &node : this->estimation_nodes)
            {
                const double spatial_delta = node->arclength - arclength;
                const double temporal_delta = node->time - time;

                if (spatial_direction < 0 && spatial_delta > 0.0)
                    continue;
                if (spatial_direction > 0 && spatial_delta < 0.0)
                    continue;
                if (temporal_direction < 0 && temporal_delta > 0.0)
                    continue;
                if (temporal_direction > 0 && temporal_delta < 0.0)
                    continue;

                const double distance = std::abs(spatial_delta) + std::abs(temporal_delta); // this logic is fine on a grid, will break with more complex state shapes. Fine for now.
                if (distance < closest_distance)
                {
                    closest_distance = distance;
                    closest_node = std::dynamic_pointer_cast<NodeT>(node);
                }
            }

            return closest_node;
        }

        /**
         * @brief The node at the arclength nearest to `arclength`, in the row nearest to `time`
         *        (arclength resolved first, then time), with exact midpoints going to the smaller
         *        arclength / earlier time.
         * @return nullptr if the state has no nodes yet, or the query lies outside the grid's
         *         extent (before the first or past the last node, within TOLERANCE) in either axis.
         */
        std::shared_ptr<NodeBase> getNearestNode(double arclength, double time) const override
        {
            if (m_time_values.empty() || arclengths.empty())
                return nullptr;
            if (arclength < arclengths.front() - TOLERANCE || arclength > arclengths.back() + TOLERANCE)
                return nullptr;
            if (time < m_time_values.front() - TOLERANCE || time > m_time_values.back() + TOLERANCE)
                return nullptr;

            const unsigned int node_index = getNodeIndex(nearestIndex(arclengths, arclength), nearestIndex(m_time_values, time));
            if (node_index == static_cast<unsigned int>(-1))
                throw std::runtime_error("ContinuumRobotState::getNearestNode: node index not found in state.");
            return this->estimation_nodes.at(node_index);
        }

    private:
        /// Index of the entry of the sorted `sorted_values` nearest to `value`, ties (within TOLERANCE) going to the lower one.
        static unsigned int nearestIndex(const std::vector<double> &sorted_values, double value)
        {
            const auto upper = std::lower_bound(sorted_values.begin(), sorted_values.end(), value);
            if (upper == sorted_values.begin())
                return 0;
            if (upper == sorted_values.end())
                return sorted_values.size() - 1;
            const auto lower = std::prev(upper);
            const auto nearest = (value - *lower <= *upper - value + TOLERANCE) ? lower : upper;
            return static_cast<unsigned int>(std::distance(sorted_values.begin(), nearest));
        }

        /// The node shifted from `node` by the given grid steps, or nullptr past the grid's edge. @throws std::runtime_error if `node` is not in the state.
        std::shared_ptr<NodeT> getShiftedNode(const std::shared_ptr<NodeBase> &node, int spatial_shift, int temporal_shift) const
        {
            const unsigned int current_node_index = getNodeFlatIndex(node);
            if (current_node_index == static_cast<unsigned int>(-1))
                throw std::runtime_error("Something went wrong in getShiftedNode: node index not found in state.");

            const NodeGridIndex current_grid_index = m_node_mapping_inverse.at(current_node_index);
            const int shifted_spatial_index = static_cast<int>(std::get<0>(current_grid_index)) + spatial_shift;
            const int shifted_temporal_index = static_cast<int>(std::get<1>(current_grid_index)) + temporal_shift;

            if (shifted_spatial_index < 0 || shifted_spatial_index >= static_cast<int>(spatial_dim))
                return nullptr; // Shifted spatial node does not exist in robot topology
            if (shifted_temporal_index < 0 || shifted_temporal_index >= static_cast<int>(m_time_values.size()))
                return nullptr; // Shifted temporal node does not exist in robot topology

            const unsigned int node_index = this->getNodeIndex(static_cast<unsigned int>(shifted_spatial_index), static_cast<unsigned int>(shifted_temporal_index));
            if (node_index == static_cast<unsigned int>(-1))
                throw std::runtime_error("Something went wrong in getShiftedNode: node index not found in state. This should not happen if the state is properly extended with addStateAtTime.");

            return std::dynamic_pointer_cast<NodeT>(this->estimation_nodes.at(node_index));
        }

        /// Flat index of `node` (by pointer), or unsigned(-1) if it is not in the state.
        unsigned int getNodeFlatIndex(const std::shared_ptr<NodeBase> &node) const
        {
            auto it = m_node_pointer_to_flat_index.find(node.get());
            return it != m_node_pointer_to_flat_index.end() ? it->second : static_cast<unsigned int>(-1);
        }

        using NodeGridIndex = std::tuple<unsigned int, unsigned int>; // (spatial index, temporal index)

        /// A pending lock(): which variable, at which spatial index and (optionally) time.
        struct Lock
        {
            std::string type;
            unsigned int spatial_index;
            std::optional<double> time;
            std::function<void(const NodeT &, gtsam::Values &)> insert;
            bool matched = false; ///< a node row matched it (or it was warned about)
        };
        std::vector<Lock> m_locks;

        std::map<NodeGridIndex, unsigned int> m_node_mapping;
        std::map<unsigned int, NodeGridIndex> m_node_mapping_inverse;
        std::map<const NodeBase *, unsigned int> m_node_pointer_to_flat_index;
        std::vector<double> m_time_values;
        mutable std::mutex m_nodes_mutex;
    };
} // namespace Spacetime::Types
#endif // CONTINUUMROBOT_HPP