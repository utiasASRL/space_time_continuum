#ifndef INTERPOLATEDNODE_HPP
#define INTERPOLATEDNODE_HPP

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/States.hpp>

namespace Spacetime::Priors
{
    /**
     * @class InterpolatedNode
     * @brief One space-time query point of a grid prior (its plan and up to four bordering
     *        nodes), evaluated on the outer gtsam::Values and exposed to an inner factor
     *        under caller-chosen "inner" keys: the building block of interpolation factors.
     *
     * The inner factor sees the interpolated node's components (e.g. WNOA's pose, strain
     * and velocity) at `InterpolatorT::InterpolatedKeys`' well-known placeholder keys,
     * optionally renamed through a key map, so that one factor can hold several
     * interpolated nodes whose placeholders would otherwise collide (see
     * TwoSidedInterpolationFactor). The interpolation's Jacobians w.r.t. the bordering
     * nodes' keys are chained onto the inner factor's Jacobians with chain().
     *
     * @tparam InterpolatorT Prior-specific interpolator (see InterpolationFactor for its contract).
     */
    template <typename InterpolatorT>
    class InterpolatedNode
    {
    public:
        using NodeType = typename InterpolatorT::NodeType;
        using ValuesType = typename InterpolatorT::ValuesType;
        using InterpolatedKeysT = typename InterpolatorT::InterpolatedKeys;
        /// Placeholder key (one of InterpolatedKeysT::keys()) -> inner key.
        using KeyMap = std::unordered_map<gtsam::Key, gtsam::Key>;
        /// Bordering-node key -> node_dim x (that key's dim) Jacobian of the interpolated node.
        using Jacobians = std::unordered_map<gtsam::Key, Eigen::MatrixXd>;

        static_assert(Types::InterpolatedKeyMap<InterpolatedKeysT, ValuesType>,
                      "InterpolatorT::InterpolatedKeys must satisfy Types::InterpolatedKeyMap.");

        static constexpr int node_dim = InterpolatorT::node_dim;

        /**
         * @param interpolator Prior-specific interpolator instance.
         * @param time Query time.
         * @param arclength Query arclength.
         * @param node00..node11 Bordering nodes, with GridInterpolator's nullptr conventions
         *        (see InterpolationFactor's constructor).
         * @param key_map Inner key of each placeholder; placeholders it omits keep their own key.
         * @throws std::invalid_argument for a null node00, a key_map entry that is not a
         *         placeholder, or two placeholders mapped to the same inner key.
         */
        InterpolatedNode(
            const InterpolatorT &interpolator,
            double time, double arclength,
            const std::shared_ptr<NodeType> &node00,
            const std::shared_ptr<NodeType> &node01,
            const std::shared_ptr<NodeType> &node10,
            const std::shared_ptr<NodeType> &node11,
            const KeyMap &key_map = {})
            : m_interpolator(interpolator), m_time(time), m_arclength(arclength),
              m_node00(node00), m_node01(node01), m_node10(node10), m_node11(node11)
        {
            if (!node00)
                throw std::invalid_argument("InterpolatedNode requires node00 to be non-null.");
            // The query and its corners are fixed, so its value-independent part is planned once.
            m_plan = m_interpolator.makePlan(m_time, m_arclength, node00, node01, node10, node11);

            const gtsam::KeyVector placeholders = InterpolatedKeysT::keys();
            for (const auto &[placeholder, inner] : key_map)
                if (std::find(placeholders.begin(), placeholders.end(), placeholder) == placeholders.end())
                    throw std::invalid_argument("InterpolatedNode: the key map renames a key that is not an interpolated key.");
            for (const gtsam::Key placeholder : placeholders)
            {
                const auto it = key_map.find(placeholder);
                const gtsam::Key inner = it == key_map.end() ? placeholder : it->second;
                if (!m_inner_to_placeholder.emplace(inner, placeholder).second)
                    throw std::invalid_argument("InterpolatedNode: two interpolated keys are mapped to the same inner key.");
                m_placeholder_to_inner[placeholder] = inner;
            }

            // Kept in the set's own order: InterpolationFactor's keys (and so the regression
            // results) have always been ordered this way.
            std::unordered_set<gtsam::Key> key_set;
            for (const auto &node : {node00, node01, node10, node11})
                if (node)
                    for (const gtsam::Key key : node->keys())
                        key_set.insert(key);
            m_corner_keys = gtsam::KeyVector(key_set.begin(), key_set.end());
        }

        double time() const { return m_time; }
        double arclength() const { return m_arclength; }

        /// Every key of the bordering nodes, each once.
        const gtsam::KeyVector &cornerKeys() const { return m_corner_keys; }

        /// The inner key standing in for `placeholder` (one of InterpolatedKeysT::keys()).
        gtsam::Key innerKey(gtsam::Key placeholder) const { return m_placeholder_to_inner.at(placeholder); }

        /// Whether `inner` is one of this node's inner keys.
        bool ownsInnerKey(gtsam::Key inner) const { return m_inner_to_placeholder.count(inner) > 0; }

        /// The inner keys, in InterpolatedKeysT::keys() order.
        gtsam::KeyVector innerKeys() const
        {
            gtsam::KeyVector keys;
            for (const gtsam::Key placeholder : InterpolatedKeysT::keys())
                keys.push_back(m_placeholder_to_inner.at(placeholder));
            return keys;
        }

        /// The (offset, dim) rows of inner key `inner`'s component within the node_dim tangent vector.
        std::pair<int, int> rowRange(gtsam::Key inner) const { return InterpolatedKeysT::rowRange(m_inner_to_placeholder.at(inner)); }

        /**
         * @brief Interpolate the node from `values`.
         * @param jacobians Optional output, filled for every corner key.
         * @param conditionalCovariance Optional output, node_dim x node_dim.
         * @throws std::runtime_error if the interpolator returns another node values type.
         */
        std::shared_ptr<ValuesType> evaluate(const gtsam::Values &values, Jacobians *jacobians, Eigen::MatrixXd *conditionalCovariance) const
        {
            auto interpolated = std::dynamic_pointer_cast<ValuesType>(m_interpolator.interpolate(
                m_plan, values, m_node00, m_node01, m_node10, m_node11, jacobians, conditionalCovariance));
            if (!interpolated)
                throw std::runtime_error("InterpolatedNode: interpolator returned an unexpected node values type.");
            return interpolated;
        }

        /// Insert the component of `node` that inner key `inner` stands in for into `inner_values`, at `inner`.
        void insert(gtsam::Key inner, const ValuesType &node, gtsam::Values &inner_values) const
        {
            const gtsam::Key placeholder = m_inner_to_placeholder.at(inner);
            if (placeholder == inner)
            {
                InterpolatedKeysT::insert(placeholder, node, inner_values);
                return;
            }
            gtsam::Values scratch;
            InterpolatedKeysT::insert(placeholder, node, scratch);
            inner_values.insert(inner, scratch.at(placeholder));
        }

        /**
         * @brief Chain the inner factor's Jacobian `J_inner` w.r.t. inner key `inner` through
         *        the interpolation: adds J_inner * (its rows of each corner's Jacobian) to
         *        H[index_of(corner key)].
         * @param G Optional error_dim x node_dim inner Jacobian w.r.t. the whole node; this key's columns are set.
         */
        template <typename IndexOf>
        void chain(gtsam::Key inner, const Eigen::MatrixXd &J_inner, const Jacobians &jacobians, const IndexOf &index_of,
                   std::vector<gtsam::Matrix> *H, Eigen::MatrixXd *G = nullptr) const
        {
            const auto [rowOffset, rowDim] = rowRange(inner);
            if (G)
                G->block(0, rowOffset, J_inner.rows(), rowDim) = J_inner;
            if (!H)
                return;
            for (const auto &[cornerKey, cornerJac] : jacobians)
            {
                const Eigen::MatrixXd contribution = J_inner * cornerJac.block(rowOffset, 0, rowDim, cornerJac.cols());
                gtsam::Matrix &block = (*H)[index_of(cornerKey)];
                if (block.size() == 0)
                    block = contribution;
                else
                    block += contribution;
            }
        }

        /// Give every corner key the inner factor never touched a properly-sized zero block in H.
        template <typename IndexOf>
        static void fillUntouched(const Jacobians &jacobians, const IndexOf &index_of, int error_dim, std::vector<gtsam::Matrix> &H)
        {
            // The width is read from `jacobians` (populated for every corner key whenever it
            // is requested) rather than assumed to be some fixed, uniform per-key width.
            for (const auto &[cornerKey, cornerJac] : jacobians)
            {
                gtsam::Matrix &block = H[index_of(cornerKey)];
                if (block.size() == 0)
                    block = Eigen::MatrixXd::Zero(error_dim, cornerJac.cols());
            }
        }

    private:
        InterpolatorT m_interpolator;
        typename InterpolatorT::Plan m_plan;
        double m_time;
        double m_arclength;
        std::shared_ptr<NodeType> m_node00, m_node01, m_node10, m_node11;
        gtsam::KeyVector m_corner_keys;
        std::unordered_map<gtsam::Key, gtsam::Key> m_placeholder_to_inner;
        std::unordered_map<gtsam::Key, gtsam::Key> m_inner_to_placeholder;
    };
} // namespace Spacetime::Priors

#endif // INTERPOLATEDNODE_HPP
