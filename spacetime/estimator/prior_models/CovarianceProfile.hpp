#ifndef COVARIANCEPROFILE_HPP
#define COVARIANCEPROFILE_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/utilities/functions.hpp>

namespace Spacetime::Priors
{
    /**
     * @brief Which grid edges a prior covariance (e.g. WNOA's Q1..Q4) belongs to, which
     *        fixes how an arclength override is keyed (see CovarianceOverride).
     */
    enum class CovarianceAxis
    {
        /// Time-axis covariance: one factor per node (between consecutive rows at that node's
        /// arclength), keyed by the node's arclength.
        Time,
        /// Arclength-axis covariance: one factor per segment [s_n, s_{n+1}], keyed by the
        /// segment's lower arclength s_n.
        Space
    };

    /**
     * @brief A grid state's node arclengths, for resolving a prior's covariance overrides
     *        before its base StatefulModel is constructed.
     * @throws std::invalid_argument for a null state, as StatefulModel does.
     */
    template <typename StateT>
    const std::vector<double> &stateArclengths(const std::shared_ptr<StateT> &state)
    {
        if (!state)
            throw std::invalid_argument("StatefulModel requires a non-null state.");
        return state->arclengths;
    }

    /**
     * @class CovarianceOverride
     * @brief Arclength-dependent overrides of one prior covariance, given to a prior at
     *        construction and applied at every time step.
     *
     * Keys are node arclengths of the prior's state (within TOLERANCE), or -1 for the tip.
     * For a time-axis covariance (CovarianceAxis::Time, e.g. WNOA's Q1) the key is the node
     * whose time factors take the value; for an arclength-axis one (CovarianceAxis::Space,
     * e.g. WNOA's Q2) it is the LOWER node of the segment, so `set(0.0, Q)` sets the segment
     * between 0 and the next node, and -1 the last segment.
     *
     * The prior resolves each node (or segment) as: the nominal value, then the function
     * (see set(Function)) if one is given, then every set / scale call at that key, in call
     * order, so a later call wins. Measurement interpolation follows the same values: a
     * query in the cell [s_n, s_{n+1}] uses node n's time-axis and segment n's
     * arclength-axis covariances (see Priors::GridInterpolator).
     *
     * @tparam MatrixT The covariance's matrix type.
     */
    template <typename MatrixT>
    class CovarianceOverride
    {
    public:
        /// Profile over the robot: (key arclength, nominal value) -> value.
        using Function = std::function<MatrixT(double arclength, const MatrixT &nominal)>;

        /// One set or scale call.
        struct Entry
        {
            double arclength;             ///< Key arclength (-1 for the tip).
            std::optional<MatrixT> value; ///< Replacement, for set.
            double scale = 1.0;           ///< Factor applied after the replacement, for scale.
        };

        /// Replace the covariance at node (or segment) `arclength` by `value`.
        CovarianceOverride &set(double arclength, const MatrixT &value)
        {
            m_entries.push_back(Entry{arclength, value, 1.0});
            return *this;
        }

        /// Multiply the covariance resolved so far at node (or segment) `arclength` by `factor`.
        CovarianceOverride &scale(double arclength, double factor)
        {
            m_entries.push_back(Entry{arclength, std::nullopt, factor});
            return *this;
        }

        /// Set the covariance at every node (or segment) to `function(key arclength, nominal)`, before the point entries.
        CovarianceOverride &set(Function function)
        {
            m_function = std::move(function);
            return *this;
        }

        /// True if nothing overrides the nominal value.
        bool empty() const { return m_entries.empty() && !m_function; }

        /// The set / scale calls, in call order.
        const std::vector<Entry> &entries() const { return m_entries; }

        /// The profile function (empty if none was set).
        const Function &function() const { return m_function; }

    private:
        std::vector<Entry> m_entries;
        Function m_function;
    };

    /**
     * @class CovarianceProfile
     * @brief One prior covariance resolved over the robot's nodes (or segments): the
     *        immutable table the prior's factors and its interpolator read (see
     *        CovarianceOverride for the keying and resolution order).
     * @tparam MatrixT The covariance's matrix type.
     */
    template <typename MatrixT>
    class CovarianceProfile
    {
    public:
        /// A constant covariance: `nominal` everywhere.
        explicit CovarianceProfile(const MatrixT &nominal = MatrixT::Identity(), CovarianceAxis axis = CovarianceAxis::Time)
            : m_nominal(nominal), m_axis(axis) {}

        /**
         * @param nominal The value wherever nothing overrides it.
         * @param axis Whether the covariance is per node (Time) or per segment (Space).
         * @param arclengths The state's ascending node arclengths.
         * @param overrides The overrides to resolve (an empty one gives a constant profile).
         * @param name The covariance's name, for error messages (e.g. "WNOA::Prior Q2").
         * @throws std::invalid_argument if a key is not a node arclength (or -1), or an
         *         arclength-axis key is the last node (no segment starts there).
         */
        CovarianceProfile(const MatrixT &nominal, CovarianceAxis axis, const std::vector<double> &arclengths,
                          const CovarianceOverride<MatrixT> &overrides, const std::string &name = "covariance")
            : m_nominal(nominal), m_axis(axis)
        {
            if (overrides.empty())
                return;

            const std::size_t nodes = arclengths.size();
            const std::size_t count = (axis == CovarianceAxis::Time) ? nodes : (nodes > 0 ? nodes - 1 : 0);
            if (count == 0)
                throw std::invalid_argument(name + ": cannot override an arclength-axis covariance on a state with fewer than two nodes.");

            m_arclengths = arclengths;
            m_values.assign(count, nominal);
            if (overrides.function())
                for (std::size_t n = 0; n < count; n++)
                    m_values[n] = overrides.function()(arclengths[n], nominal);

            for (const auto &entry : overrides.entries())
            {
                const std::optional<unsigned int> node = Types::nodeIndexAt(arclengths, entry.arclength);
                if (!node)
                    throw std::invalid_argument(name + ": s = " + std::to_string(entry.arclength) +
                                                " is not a node arclength (or -1 for the tip).");
                unsigned int index = *node;
                if (axis == CovarianceAxis::Space && index == nodes - 1)
                {
                    if (entry.arclength != -1.0)
                        throw std::invalid_argument(name + ": s = " + std::to_string(entry.arclength) +
                                                    " is the last node, where no segment starts (an arclength-axis "
                                                    "override is keyed by its segment's lower node; -1 is the last segment).");
                    index = static_cast<unsigned int>(count - 1);
                }
                if (entry.value)
                    m_values[index] = *entry.value;
                m_values[index] *= entry.scale;
            }
        }

        /// The value wherever nothing overrides it.
        const MatrixT &nominal() const { return m_nominal; }

        /// Whether the covariance is per node (Time) or per segment (Space).
        CovarianceAxis axis() const { return m_axis; }

        /// True if the covariance is the nominal one everywhere (no overrides were given).
        bool isConstant() const { return m_values.empty(); }

        /// Number of nodes (Time) or segments (Space) resolved; 0 for a constant profile.
        std::size_t size() const { return m_values.size(); }

        /// The value at node (Time) or segment (Space) `index`; the nominal one for a constant profile.
        /// @throws std::out_of_range for an index past the resolved table.
        const MatrixT &at(unsigned int index) const
        {
            if (isConstant())
                return m_nominal;
            return m_values.at(index);
        }

        /**
         * @brief The value of the grid cell anchored at `arclength`: that of the last node at
         *        or before it (Time), or of the segment starting there (Space, clamped to the
         *        last segment past the tip). nullopt gives the nominal value.
         */
        const MatrixT &atArclength(std::optional<double> arclength) const
        {
            if (isConstant() || !arclength)
                return m_nominal;
            const auto upper = std::upper_bound(m_arclengths.begin(), m_arclengths.end(), *arclength + TOLERANCE);
            const std::size_t node = upper == m_arclengths.begin() ? 0 : static_cast<std::size_t>(std::distance(m_arclengths.begin(), upper)) - 1;
            return m_values[std::min(node, m_values.size() - 1)];
        }

    private:
        MatrixT m_nominal;
        CovarianceAxis m_axis;
        std::vector<double> m_arclengths; // the state's node arclengths (empty when constant)
        std::vector<MatrixT> m_values;    // per node (Time) or segment (Space); empty when constant
    };
} // namespace Spacetime::Priors

#endif // COVARIANCEPROFILE_HPP
