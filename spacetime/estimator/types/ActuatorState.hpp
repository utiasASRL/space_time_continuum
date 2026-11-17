#ifndef ACTUATORSTATE_HPP
#define ACTUATORSTATE_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/types/States.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime::Types
{
    /// The well-known interpolated key standing in for an input node's input (see NodeBase::interpolatedKeys).
    struct InputKeys
    {
        /// Interpolated input key. Only needs to be distinct from the other keys of an inner factor (see InterpolatedKeyMap).
        static gtsam::Key Input() { return gtsam::Symbol('i', 0); }
    };

    /// The values of one input node: its input vector, the node's single component.
    struct InputValues : public NodeValues
    {
        /// A zero input of `dim` entries.
        explicit InputValues(unsigned int dim) : NodeValues()
        {
            values.resize(1); // one component, the input
            input(Eigen::VectorXd::Zero(dim));
        }

        /// The input vector (stored as a column matrix).
        const Eigen::MatrixXd &input() const { return values[0]; }
        /// Sets the input vector.
        void input(const Eigen::VectorXd &value) { values[0] = value; }
    };

    /// One actuator input u_k: a node, at the measurement's time and arclength 0, holding the input vector of `dim()` entries.
    struct InputNode : public NodeBase
    {
        /// Registers the input with `km`, keyed with `symbol` and timestamped at `time`.
        InputNode(KeyManager &km, double time, unsigned int dim, unsigned char symbol) : NodeBase(km, 0.0, time), m_dim(dim)
        {
            m_input_key = registerVariable<gtsam::Vector>(km, "input", symbol);
        }

        /// Key of the input variable (a gtsam::Vector of dim() entries).
        gtsam::Key inputKey() const { return m_input_key; }

        /// Size of the input.
        unsigned int dim() const { return m_dim; }

        /// The single pair (InputKeys::Input(), inputKey()).
        std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const override { return {{InputKeys::Input(), m_input_key}}; }

        /// The input read from `values`. @throws std::runtime_error if it is not dim() entries (gtsam's error if it is absent).
        std::shared_ptr<NodeValues> getValuesBase(const gtsam::Values &values) const override
        {
            const gtsam::Vector &input = values.at<gtsam::Vector>(m_input_key);
            if (input.size() != static_cast<Eigen::Index>(m_dim))
                throw std::runtime_error("InputNode: the input " + gtsam::DefaultKeyFormatter(m_input_key) + " has " + std::to_string(input.size()) +
                                         " entries, not " + std::to_string(m_dim) + ".");
            auto result = std::make_shared<InputValues>(m_dim);
            result->input(input);
            return result;
        }

    private:
        unsigned int m_dim;
        gtsam::Key m_input_key;
    };

    /**
     * @brief The state of an actuator (see Types::ActuatorBase): one input node per measured
     *        input, in time order. Off-input queries snap to the nearest input in time (see
     *        getNearestNode), so a sensor can measure an actuator's inputs.
     */
    struct ActuatorState : public State
    {
        const unsigned int input_dim;     ///< Size of every input.
        const unsigned char input_symbol; ///< Symbol character of the input keys.

        /// @throws std::invalid_argument for a zero `input_dim`.
        explicit ActuatorState(unsigned int input_dim, unsigned char input_symbol = 'u') : input_dim(input_dim), input_symbol(input_symbol)
        {
            if (input_dim == 0)
                throw std::invalid_argument("An actuator's input needs at least one entry.");
        }

        /// Adds an input at `time` (registering its key with `km`), after any input at or before it, and returns it.
        std::shared_ptr<InputNode> addInput(KeyManager &km, double time)
        {
            const auto position = std::upper_bound(m_times.begin(), m_times.end(), time);
            const auto index = std::distance(m_times.begin(), position);
            auto node = std::make_shared<InputNode>(km, time, input_dim, input_symbol);
            m_times.insert(position, time);
            m_inputs.insert(m_inputs.begin() + index, node);
            return node;
        }

        /// Every input added so far, in time order, including marginalized ones.
        const std::vector<std::shared_ptr<InputNode>> &inputs() const { return m_inputs; }

        /// The inputs' times, ascending.
        const std::vector<double> &times() const override { return m_times; }

        /**
         * @brief The input nearest to `time` (`arclength` is unused), the earlier of two
         *        equally near (within TOLERANCE); nullptr outside the inputs' extent, or
         *        without inputs.
         */
        std::shared_ptr<NodeBase> getNearestNode(double arclength, double time) const override
        {
            (void)arclength;
            if (m_times.empty() || time < m_times.front() - TOLERANCE || time > m_times.back() + TOLERANCE)
                return nullptr;
            // Ties within TOLERANCE go to the earlier input, as ContinuumRobotState's do.
            auto nearest = std::lower_bound(m_times.begin(), m_times.end(), time);
            if (nearest == m_times.end() || (nearest != m_times.begin() && time - *std::prev(nearest) <= *nearest - time + TOLERANCE))
                --nearest;
            return m_inputs[std::distance(m_times.begin(), nearest)];
        }

    private:
        std::vector<double> m_times;
        std::vector<std::shared_ptr<InputNode>> m_inputs;
    };
} // namespace Spacetime::Types

#endif // ACTUATORSTATE_HPP
