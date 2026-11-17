#ifndef NEARESTNODEFACTOR_HPP
#define NEARESTNODEFACTOR_HPP

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

namespace Spacetime::Types
{
    /**
     * @class NearestNodeFactor
     * @brief Evaluates an inner factor, built on a node type's well-known "interpolated" keys
     *        (see Types::InterpolatedKeyMap), directly on one estimated node, i.e. an off-node
     *        query snapped to the nearest node (see StatefulModel::getOuterFactor's default).
     *
     * Snapping is the identity map from the node to the "interpolated" one, so the inner
     * factor's error and Jacobian blocks are this factor's own, key for key, and no conditional
     * covariance is added to its noise model (unlike Priors::InterpolationFactor). Inner keys
     * that are not interpolated keys (e.g. a landmark) are passed through unchanged.
     *
     * This does the same job as gtsam's NonlinearFactor::rekey, which is not used because it
     * requires the inner factor to implement clone().
     */
    class NearestNodeFactor : public gtsam::NoiseModelFactor
    {
    public:
        using Base = gtsam::NoiseModelFactor; ///< gtsam base class.
        using This = NearestNodeFactor;       ///< This factor's type.

        using Base::error;
        using Base::unwhitenedError;

        /**
         * @param inner_factor Factor on (a subset of) the interpolated keys in `key_map`, plus
         *        optionally other keys, which are passed through.
         * @param key_map (interpolated key, node key) pairs of the snapped node (see
         *        NodeBase::interpolatedKeys).
         * @throws std::runtime_error if `inner_factor` is null, or if remapping would give this
         *         factor the same key twice.
         */
        NearestNodeFactor(
            const gtsam::NoiseModelFactor::shared_ptr &inner_factor,
            const std::vector<std::pair<gtsam::Key, gtsam::Key>> &key_map)
            : Base(inner_factor ? inner_factor->noiseModel() : gtsam::SharedNoiseModel()),
              m_inner_factor(inner_factor)
        {
            if (!inner_factor)
                throw std::runtime_error("NearestNodeFactor requires a non-null inner factor.");

            std::unordered_map<gtsam::Key, gtsam::Key> to_outer(key_map.begin(), key_map.end());
            std::unordered_set<gtsam::Key> seen;
            for (const gtsam::Key key : inner_factor->keys())
            {
                auto it = to_outer.find(key);
                const gtsam::Key outer = (it != to_outer.end()) ? it->second : key;
                if (!seen.insert(outer).second)
                    throw std::runtime_error(
                        "NearestNodeFactor: the inner factor references the same node variable twice "
                        "(once by its interpolated key and once directly).");
                this->keys_.push_back(outer);
            }
        }

        ~NearestNodeFactor() override = default;

        /// Prints this factor's keys, then the inner factor.
        void print(
            const std::string &s = "",
            const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
        {
            std::cout << s << "NearestNodeFactor on ";
            for (const auto &k : this->keys())
                std::cout << keyFormatter(k) << " ";
            std::cout << std::endl;
            m_inner_factor->print("Inner Factor: ", keyFormatter);
        }

        /// Whether `expected` is a NearestNodeFactor with the same keys, noise model and inner factor.
        bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override
        {
            const This *e = dynamic_cast<const This *>(&expected);
            return e != nullptr && Base::equals(*e, tol) && m_inner_factor->equals(*e->m_inner_factor, tol);
        }

        /// The inner factor's unwhitened error (and Jacobians), on the node's values.
        gtsam::Vector unwhitenedError(const gtsam::Values &values, gtsam::OptionalMatrixVecType H = nullptr) const override
        {
            // keys_ is the inner factor's keys, remapped in place, so position i of both matches.
            const gtsam::KeyVector &inner_keys = m_inner_factor->keys();
            gtsam::Values values_inner;
            for (size_t i = 0; i < inner_keys.size(); ++i)
                values_inner.insert(inner_keys[i], values.at(this->keys_[i]));

            return H ? m_inner_factor->unwhitenedError(values_inner, H)
                     : m_inner_factor->unwhitenedError(values_inner);
        }

        /// The wrapped factor.
        const gtsam::NoiseModelFactor::shared_ptr &innerFactor() const { return m_inner_factor; }

    private:
        gtsam::NoiseModelFactor::shared_ptr m_inner_factor;
    };
} // namespace Spacetime::Types

#endif // NEARESTNODEFACTOR_HPP
