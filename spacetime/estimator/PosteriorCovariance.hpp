/**
 * @file PosteriorCovariance.hpp
 * @brief Sparse store of posterior covariance blocks extracted from a solved factor graph.
 * @author Spencer Teetaert
 */

#ifndef POSTERIORCOVARIANCE_HPP
#define POSTERIORCOVARIANCE_HPP

#include <algorithm>
#include <map>
#include <set>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/BayesTreeCovariance.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime
{
    /**
     * @brief An immutable collection of posterior covariance blocks Sigma_ab between
     *        pairs of state keys.
     *
     * Only the blocks that were requested at extraction time are stored: the full dense
     * posterior covariance is never formed. Extraction is driven by a list of key GROUPS
     * (see compute()), and every pairwise block inside each group is stored. Which groups
     * to extract is a policy decision left to the caller (see Types::Model::covarianceKeyGroups), so other states can
     * be extracted later without changing this class.
     */
    class PosteriorCovariance
    {
    public:
        /**
         * @brief Extract the joint marginal of every group from a solved graph's marginals.
         * @param marginals Marginals of the solved graph (see gtsam::Marginals).
         * @param groups Key groups whose joint marginals are extracted. Every pairwise
         *        block (including each key with itself) inside a group is stored. A block
         *        appearing in several groups is identical in each and is stored once.
         * @param known Values of keys held fixed (locked, see Types::State::lockedValues).
         *        They are not variables of the graph: their blocks are stored as zeros.
         * @param tree The Bayes tree `marginals` was factorized into (see
         *        BayesTreeMarginals). If given, the blocks are recovered in one pass over it
         *        (see BayesTreeCovariance::recover), and only the pairs it cannot recover
         *        are queried from `marginals`. If null, every group is queried.
         * @return The extracted blocks (never null).
         */
        static std::shared_ptr<const PosteriorCovariance> compute(
            const gtsam::Marginals &marginals, const std::vector<gtsam::KeyVector> &groups,
            const gtsam::Values &known = gtsam::Values(), const gtsam::GaussianBayesTree *tree = nullptr)
        {
            using KeyPair = BayesTreeCovariance::KeyPair;
            auto result = std::make_shared<PosteriorCovariance>();

            // Every requested pair of estimated keys, once, with the first group requesting it.
            std::vector<KeyPair> pairs;
            std::vector<size_t> pair_group;
            std::set<KeyPair> seen;
            for (size_t g = 0; g < groups.size(); g++)
            {
                gtsam::KeyVector estimated;
                for (const gtsam::Key key : groups[g])
                    if (!known.exists(key))
                        estimated.push_back(key);
                for (size_t i = 0; i < estimated.size(); i++)
                    for (size_t j = i; j < estimated.size(); j++)
                        if (const KeyPair pair = ordered(estimated[i], estimated[j]); seen.insert(pair).second)
                        {
                            pairs.push_back(pair);
                            pair_group.push_back(g);
                        }
            }

            std::vector<KeyPair> unresolved = pairs;
            if (tree)
            {
                BayesTreeCovariance::Result recovered = BayesTreeCovariance::recover(*tree, pairs);
                for (auto &[pair, block] : recovered.blocks)
                    result->insert(pair.first, pair.second, block);
                unresolved = std::move(recovered.unresolved);
            }

            // Query the rest from the marginals: per group, the joint of the keys its unresolved pairs involve.
            std::map<KeyPair, size_t> group_of;
            for (size_t p = 0; p < pairs.size(); p++)
                group_of[pairs[p]] = pair_group[p];
            std::map<size_t, std::vector<KeyPair>> by_group;
            for (const KeyPair &pair : unresolved)
                by_group[group_of.at(pair)].push_back(pair);
            std::vector<std::vector<KeyPair>> queries;
            for (auto &[g, group_pairs] : by_group)
                queries.push_back(std::move(group_pairs));

            for (const std::vector<KeyPair> &query : queries)
            {
                gtsam::KeyVector keys;
                for (const auto &[a, b] : query)
                    for (const gtsam::Key key : {a, b})
                        if (std::find(keys.begin(), keys.end(), key) == keys.end())
                            keys.push_back(key);
                // Blocks are read per key pair (JointMarginal::at), which avoids depending on
                // the key order fullMatrix() would impose.
                const gtsam::JointMarginal joint = marginals.jointMarginalCovariance(keys);
                for (const auto &[a, b] : query)
                    result->insert(a, b, joint.at(a, b));
            }

            debug() << "PosteriorCovariance: " << pairs.size() - unresolved.size() << " of " << pairs.size()
                    << " blocks recovered from the Bayes tree, " << unresolved.size() << " queried in "
                    << queries.size() << " joint marginals.";

            // Blocks involving known keys are zero.
            for (const gtsam::KeyVector &group : groups)
            {
                auto dim = [&](gtsam::Key key)
                { return known.exists(key) ? known.at(key).dim() : static_cast<size_t>(result->covariance(key).rows()); };
                for (size_t i = 0; i < group.size(); i++)
                    for (size_t j = i; j < group.size(); j++)
                        if (known.exists(group[i]) || known.exists(group[j]))
                            result->insert(group[i], group[j], gtsam::Matrix::Zero(dim(group[i]), dim(group[j])));
            }
            return result;
        }

        /// Store (or overwrite) the block Sigma_ab. Sigma_ba is then served as its transpose.
        void insert(gtsam::Key a, gtsam::Key b, const gtsam::Matrix &block)
        {
            if (a <= b)
                m_blocks[{a, b}] = block;
            else
                m_blocks[{b, a}] = block.transpose();
        }

        /// Whether the block Sigma_ab (equivalently Sigma_ba) was extracted.
        bool has(gtsam::Key a, gtsam::Key b) const { return m_blocks.contains(ordered(a, b)); }

        /// Whether the marginal variance Sigma_aa was extracted.
        bool has(gtsam::Key a) const { return has(a, a); }

        /**
         * @brief Posterior cross-covariance Sigma_ab = E[da db^T] between two keys.
         * @throws std::out_of_range if the block was not extracted.
         */
        gtsam::Matrix covariance(gtsam::Key a, gtsam::Key b) const
        {
            auto it = m_blocks.find(ordered(a, b));
            if (it == m_blocks.end())
                throw std::out_of_range("PosteriorCovariance: requested block was not extracted.");
            return a <= b ? it->second : gtsam::Matrix(it->second.transpose());
        }

        /**
         * @brief Posterior marginal covariance (variance) Sigma_aa of a single key.
         * @throws std::out_of_range if it was not extracted.
         */
        gtsam::Matrix covariance(gtsam::Key a) const { return covariance(a, a); }

        /**
         * @brief Assemble the joint covariance of several keys, in the given order.
         * @param out Set to the dense block matrix [Sigma_{keys[i], keys[j]}].
         * @return false (leaving `out` untouched) if any required block was not extracted.
         */
        bool joint(const gtsam::KeyVector &keys, Eigen::MatrixXd &out) const
        {
            std::vector<Eigen::Index> offsets(keys.size() + 1, 0);
            for (size_t i = 0; i < keys.size(); i++)
            {
                auto it = m_blocks.find(ordered(keys[i], keys[i]));
                if (it == m_blocks.end())
                    return false;
                offsets[i + 1] = offsets[i] + it->second.rows();
            }
            for (size_t i = 0; i < keys.size(); i++)
                for (size_t j = i + 1; j < keys.size(); j++)
                    if (!has(keys[i], keys[j]))
                        return false;

            out.resize(offsets.back(), offsets.back());
            for (size_t i = 0; i < keys.size(); i++)
            {
                for (size_t j = i; j < keys.size(); j++)
                {
                    const gtsam::Matrix block = covariance(keys[i], keys[j]);
                    out.block(offsets[i], offsets[j], block.rows(), block.cols()) = block;
                    if (j != i)
                        out.block(offsets[j], offsets[i], block.cols(), block.rows()) = block.transpose();
                }
            }
            return true;
        }

        /// Number of stored (unordered) blocks.
        size_t size() const { return m_blocks.size(); }
        /// Whether no block is stored.
        bool empty() const { return m_blocks.empty(); }

    private:
        /// (a, b) sorted so that first <= second, the storage key of Sigma_ab.
        static std::pair<gtsam::Key, gtsam::Key> ordered(gtsam::Key a, gtsam::Key b)
        {
            return a <= b ? std::make_pair(a, b) : std::make_pair(b, a);
        }

        std::map<std::pair<gtsam::Key, gtsam::Key>, gtsam::Matrix> m_blocks; // keyed with first <= second
    };
} // namespace Spacetime

#endif // POSTERIORCOVARIANCE_HPP
