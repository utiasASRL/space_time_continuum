/**
 * @file LockedFactor.hpp
 * @brief Wraps a factor so that some of its variables are known constants rather than
 *        estimated variables (see Types::State::lockedValues).
 * @author Spencer Teetaert
 */

#ifndef LOCKEDFACTOR_HPP
#define LOCKEDFACTOR_HPP

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

namespace Spacetime
{
    /**
     * @brief A factor over the free keys of `inner`, evaluating `inner` with its locked
     *        keys held at their constant values.
     *
     * The locked keys are not variables of the graph. The linearization is that of
     * `inner` with the locked keys' columns dropped (their perturbation is zero), so
     * the result is exact.
     */
    class LockedFactor : public gtsam::NonlinearFactor
    {
    public:
        using Base = gtsam::NonlinearFactor;

        /**
         * @param inner The factor to wrap.
         * @param locked Values of (at least) the keys of `inner` that are locked.
         * @throws std::invalid_argument if `inner` is null.
         */
        LockedFactor(const gtsam::NonlinearFactor::shared_ptr &inner, const gtsam::Values &locked)
            : Base(freeKeys(inner, locked)), m_inner(inner)
        {
            for (const gtsam::Key key : inner->keys())
                if (locked.exists(key))
                    m_constants.insert(key, locked.at(key));
        }

        /// Whether any key of `factor` is in `locked` (i.e. whether it needs wrapping).
        static bool touches(const gtsam::NonlinearFactor &factor, const gtsam::Values &locked)
        {
            for (const gtsam::Key key : factor.keys())
                if (locked.exists(key))
                    return true;
            return false;
        }

        /// Error of `inner`, with the locked keys at their constants.
        double error(const gtsam::Values &values) const override { return m_inner->error(withConstants(values)); }

        /// Dimension of `inner`'s error.
        size_t dim() const override { return m_inner->dim(); }

        /**
         * @brief Linearization of `inner` with the locked keys' columns dropped.
         * @throws std::runtime_error if `inner` does not linearize to a JacobianFactor.
         */
        std::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &values) const override
        {
            auto linear = m_inner->linearize(withConstants(values));
            if (!linear)
                return linear;

            if (auto jacobian = std::dynamic_pointer_cast<gtsam::JacobianFactor>(linear))
            {
                std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms;
                for (auto it = jacobian->begin(); it != jacobian->end(); ++it)
                    if (!m_constants.exists(*it))
                        terms.emplace_back(*it, jacobian->getA(it));
                return std::make_shared<gtsam::JacobianFactor>(terms, jacobian->getb(), jacobian->get_model());
            }
            if (std::dynamic_pointer_cast<gtsam::HessianFactor>(linear))
                throw std::runtime_error("LockedFactor: Hessian linearizations of a locked factor are not supported.");
            throw std::runtime_error("LockedFactor: unsupported linearization type.");
        }

        /// Shallow copy (shares `inner`).
        gtsam::NonlinearFactor::shared_ptr clone() const override { return std::make_shared<LockedFactor>(*this); }

        /// The wrapped factor.
        const gtsam::NonlinearFactor::shared_ptr &inner() const { return m_inner; }
        /// Values of the locked keys of `inner`.
        const gtsam::Values &constants() const { return m_constants; }

    private:
        /// The keys of `inner` not in `locked`. @throws std::invalid_argument if `inner` is null.
        static gtsam::KeyVector freeKeys(const gtsam::NonlinearFactor::shared_ptr &inner, const gtsam::Values &locked)
        {
            if (!inner)
                throw std::invalid_argument("LockedFactor requires a non-null inner factor.");
            gtsam::KeyVector keys;
            for (const gtsam::Key key : inner->keys())
                if (!locked.exists(key))
                    keys.push_back(key);
            return keys;
        }

        /// `values` restricted to this factor's free keys, plus the constants.
        gtsam::Values withConstants(const gtsam::Values &values) const
        {
            gtsam::Values all = m_constants;
            for (const gtsam::Key key : keys())
                all.insert(key, values.at(key));
            return all;
        }

        gtsam::NonlinearFactor::shared_ptr m_inner;
        gtsam::Values m_constants;
    };
}

#endif // LOCKEDFACTOR_HPP
