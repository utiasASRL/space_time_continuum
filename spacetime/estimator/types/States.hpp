#ifndef STATES_HPP
#define STATES_HPP

#include <iostream>
#include <concepts>
#include <map>
#include <limits>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <typeindex>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime::Types
{
    /// Type-erased values of one node; derived types give `values` its layout.
    struct NodeValues
    {
        virtual ~NodeValues() = default;
        std::vector<Eigen::MatrixXd> values; ///< One matrix per node component.
        /// Fills every component with random entries.
        virtual void setRandom()
        {
            for (auto &value : values)
                value.setRandom();
        };
        /// Copies `other`'s values.
        virtual NodeValues &operator=(const NodeValues &other)
        {
            const auto *typed_other = dynamic_cast<const NodeValues *>(&other);
            if (!typed_other)
                throw std::runtime_error("NodeValues::operator= received incompatible node values type.");

            this->values = typed_other->values;
            return *this;
        }
    };
    /// NodeValues types that declare a static `dim` (the node's tangent dimension). @tparam T The candidate values type.
    template <typename T>
    concept ValuesType = std::derived_from<T, NodeValues> && requires(T n) { T::dim; };

    /// A registered variable: its key and C++ type.
    struct VariableInfo
    {
        gtsam::Key key;        ///< The variable's key.
        std::type_index type;  ///< C++ type of the variable's value (e.g. gtsam::Pose3).

        /// Same key and type.
        bool operator==(const VariableInfo &other) const
        {
            return key == other.key && type == other.type;
        }
        /// Lexicographic by (key, type).
        bool operator<(const VariableInfo &other) const
        {
            if (key != other.key)
                return key < other.key;
            return type < other.type;
        }
    };

    /**
     * @brief Base of a node: a set of named variables estimated at one (arclength, time).
     *        Derived node types register their variables in their constructor (see
     *        registerVariable) and, to support sensors and queries, override
     *        interpolatedKeys and getValuesBase.
     */
    struct NodeBase
    {
        /// Derived types register their variables with `km` (see registerVariable); unused here.
        NodeBase(KeyManager &km, double arclength, double time)
            : arclength(arclength), time(time) {}
        virtual ~NodeBase() = default;

        /// Keys of this node's variables, ordered by variable name.
        gtsam::KeyVector keys() const
        {
            gtsam::KeyVector out;
            for (const auto &[name, variable] : variables_)
                out.push_back(variable.key);
            return out;
        }
        /// Names of this node's variables, in alphabetical order.
        std::vector<std::string> names() const
        {
            std::vector<std::string> out;
            for (const auto &[name, variable] : variables_)
                out.push_back(name);
            return out;
        }

        /// Key of the variable `name`. @throws std::runtime_error if there is none.
        gtsam::Key key(const std::string &name) const
        {
            auto it = variables_.find(name);
            if (it == variables_.end())
                throw std::runtime_error("NodeBase::key: variable name not found.");
            return it->second.key;
        }
        /// C++ type of the variable `name`. @throws std::runtime_error if there is none.
        std::type_index type(const std::string &name) const
        {
            auto it = variables_.find(name);
            if (it == variables_.end())
                throw std::runtime_error("NodeBase::type: variable name not found.");
            return it->second.type;
        }

        /**
         * @brief Inserts `value` into `values` at the key of the variable `name`.
         * @tparam T The variable's value type; must match the registered one.
         * @throws std::out_of_range for an unknown name; std::runtime_error if T is not its type.
         */
        template <typename T>
        void insert(const std::string &name, const T &value, gtsam::Values &values) const
        {
            const auto &entry = variables_.at(name);
            if (entry.type != std::type_index(typeid(T)))
                throw std::runtime_error("Node::insert type mismatch for variable '" + name + "'");
            values.insert(entry.key, value);
        }

        /**
         * @brief (well-known interpolated key, this node's own key) pairs, in the order the
         *        node's components appear in its tangent vector (see Types::InterpolatedKeyMap).
         *
         * Lets a prior map a sensor's inner factor straight onto this node without knowing the
         * node type (see Types::StatefulModel's default nearest-node getOuterFactor). Empty by
         * default, meaning the node type does not support that.
         */
        virtual std::vector<std::pair<gtsam::Key, gtsam::Key>> interpolatedKeys() const { return {}; }

        /// This node's values read from `values`, type-erased. nullptr if the node type does not support it.
        virtual std::shared_ptr<NodeValues> getValuesBase(const gtsam::Values &values) const
        {
            (void)values;
            return nullptr;
        }

        /// Prints the node's location and variables (gtsam Testable).
        virtual void print(const std::string &s = std::string()) const
        {
            std::cout << s << "Node at arclength: " << arclength << ", time: " << time << std::endl;
            for (const auto &var : variables_)
            {
                std::cout << "- Variable Name: " << var.first << ", Key: " << gtsam::Symbol(var.second.key) << ", Type: " << var.second.type.name() << std::endl;
            }
        }
        /// Same as print(s), also printing each variable's value from `values`.
        virtual void print(const gtsam::Values &values, const std::string &s = std::string()) const
        {
            std::cout << s << "Node at arclength: " << arclength << ", time: " << time << std::endl;
            for (const auto &var : variables_)
            {
                std::cout << "- Variable Name: " << var.first << ", Key: " << gtsam::Symbol(var.second.key) << ", Type: " << var.second.type.name() << " with value: ";
                values.at(var.second.key).print();
            }
        }

        /// Strict-weak ordering: by time, then arclength, then variables.
        bool operator<(const NodeBase &other) const
        {
            if (this->time < other.time)
                return true;
            if (this->time > other.time)
                return false;
            if (this->arclength < other.arclength)
                return true;
            if (this->arclength > other.arclength)
                return false;
            if (this->variables_.size() != other.variables_.size())
                return this->variables_.size() < other.variables_.size();
            if (this->variables_ != other.variables_)
                return this->variables_ < other.variables_;
            return false; // they are equal
        }

        /**
         * @brief Strict-weak ordering: (time, arclength, variables) descending.
         *
         * Mirror of `operator<` with all comparisons reversed.
         */
        bool operator>(const NodeBase &other) const
        {
            if (this->time > other.time)
                return true;
            if (this->time < other.time)
                return false;
            if (this->arclength > other.arclength)
                return true;
            if (this->arclength < other.arclength)
                return false;
            if (this->variables_.size() != other.variables_.size())
                return this->variables_.size() > other.variables_.size();
            if (this->variables_ != other.variables_)
                return this->variables_ > other.variables_;
            return false; // they are equal
        }

        /// Returns true if this state's timestamp is before \p time.
        bool operator<(double time) const { return this->time < time; }

        /// Returns true if this state's timestamp is after \p time.
        bool operator>(double time) const { return this->time > time; }

        /// Same arclength, time and variables.
        bool operator==(const NodeBase &other) const
        {
            return (this->arclength == other.arclength) && (this->time == other.time) && (this->variables_ == other.variables_);
        }

    protected:
        /**
         * @brief Adds (or replaces) the variable `name` of type T, with a new `chr` key from
         *        `km` timestamped at this node's time, and returns that key.
         * @tparam T The variable's value type (e.g. gtsam::Pose3, gtsam::Vector).
         */
        template <typename T>
        gtsam::Key registerVariable(KeyManager &km, const std::string &name, unsigned char chr)
        {
            gtsam::Key key = km.nextKey(chr, time);
            variables_.insert_or_assign(name, VariableInfo{key, std::type_index(typeid(T))});
            return key;
        }

    public:
        const double arclength; ///< [m]
        const double time;      ///< [s]

    private:
        std::map<std::string, VariableInfo> variables_;
    };
    /// Types derived from NodeBase. @tparam T The candidate node type.
    template <typename T>
    concept NodeType = std::derived_from<T, NodeBase>;

    /**
     * @brief The estimated state: plain data, extended by the one model that owns it (e.g.
     *        a prior, see Types::Model). It holds no reference back to that model.
     */
    struct State
    {
        virtual ~State() = default;

        /// Adds the state's nodes at `time` (e.g. a node row), registering their keys with `km`. Does nothing by default.
        virtual void addStateAtTime(KeyManager &km, double time) {};

        /**
         * @brief The node nearest to (arclength, time), for priors that snap off-node queries
         *        to a node (see Types::StatefulModel's default getOuterFactor). nullptr if the state
         *        has no such node, e.g. it is not node-based or the query is outside its extent.
         */
        virtual std::shared_ptr<NodeBase> getNearestNode(double arclength, double time) const
        {
            (void)arclength;
            (void)time;
            return nullptr;
        }

        /// Times of the state's node rows, ascending (see addStateAtTime); empty if the state has no rows.
        virtual const std::vector<double> &times() const
        {
            static const std::vector<double> none;
            return none;
        }

        /**
         * @brief Values of the state variables held fixed (locked), by key. A locked
         *        variable is never estimated: the Estimator treats it as a known constant
         *        (see e.g. ContinuumRobotState::lock).
         */
        const gtsam::Values &lockedValues() const { return locked_values_; }

    protected:
        gtsam::Values locked_values_; ///< Locked variables by key; filled by derived states (see lockedValues).
    };
    /// Types derived from State. @tparam T The candidate state type.
    template <typename T>
    concept StateType = std::derived_from<T, State>;
} // namespace Spacetime::Types

namespace std
{
    /**
     * @brief Hash `NodeBase` by its variable keys (consistent with operator==, since equal
     *        nodes have the same variables).
     */
    template <>
    struct hash<Spacetime::Types::NodeBase>
    {
        /// Boost-style combination of the hashes of `p`'s keys.
        std::size_t operator()(const Spacetime::Types::NodeBase &p) const noexcept
        {
            std::size_t seed = p.keys().size();
            for (const auto &key : p.keys())
            {
                // Hash combination algorithm (inspired by Boost)
                seed ^= std::hash<gtsam::Key>{}(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
} // namespace std

#endif // STATES_HPP