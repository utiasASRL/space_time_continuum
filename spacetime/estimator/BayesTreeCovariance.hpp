/**
 * @file BayesTreeCovariance.hpp
 * @brief Posterior covariance blocks recovered in one top-down pass over a Gaussian
 *        Bayes tree, for any variable ordering or state layout.
 * @author Spencer Teetaert
 */

#ifndef BAYESTREECOVARIANCE_HPP
#define BAYESTREECOVARIANCE_HPP

#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

namespace Spacetime
{
    /// gtsam::Marginals that also exposes the Bayes tree it factorized, so the tree can be reused.
    class BayesTreeMarginals : public gtsam::Marginals
    {
    public:
        using gtsam::Marginals::Marginals;

        /// The Bayes tree the marginals were computed from.
        const gtsam::GaussianBayesTree &bayesTree() const { return bayesTree_; }
    };

    /**
     * @brief Covariance recovery on a Bayes tree (the tree form of Barfoot's eq. 3.73).
     *
     * Every clique holds a conditional R x_F + S x_S = d over its frontal variables F given
     * its separator S. Given the separator's joint covariance Sigma_SS,
     *     Sigma_FS = G Sigma_SS,    Sigma_FF = R^-1 R^-T + G Sigma_SS G^T,    with G = -R^-1 S.
     * A root has no separator, and every child's separator lies inside its parent's
     * F u S (the running-intersection property), so one pass from the roots down gives the
     * joint covariance of every clique's F u S, using only dense products and a triangular
     * solve per clique.
     *
     * A pair (a, b) is recovered when b lies in the F u S of the clique where a is a frontal
     * variable, or vice versa. That covers every pair linked by a factor (plus fill-in), so
     * adding linkingFactors() for the requested groups to the graph before factorizing it
     * makes every pair of every group recoverable. Other pairs are reported unresolved
     * (use gtsam::Marginals for those).
     */
    namespace BayesTreeCovariance
    {
        /// A requested (a, b) pair of keys.
        using KeyPair = std::pair<gtsam::Key, gtsam::Key>;

        /// Output of recover().
        struct Result
        {
            std::map<KeyPair, gtsam::Matrix> blocks; ///< Sigma_ab for each recovered pair (a, b)
            std::vector<KeyPair> unresolved;         ///< requested pairs not in any single clique
        };

        namespace detail
        {
            /// Parent index of a root clique.
            constexpr std::size_t NO_PARENT = std::numeric_limits<std::size_t>::max();

            /// A clique of the tree, with the layout of the joint covariance over its F u S.
            struct Clique
            {
                gtsam::GaussianBayesTree::sharedClique clique; ///< the tree's clique
                std::size_t parent = NO_PARENT;    ///< index of the parent clique in the previous level
                gtsam::KeyVector keys;             ///< frontals, then separator (the conditional's key order)
                std::vector<Eigen::Index> offsets; ///< offsets[i] is the row of keys[i]; offsets.back() the total dimension
                Eigen::Index frontal_dim = 0;      ///< total dimension of the frontals

                /// Reads the key layout from the clique's conditional.
                explicit Clique(const gtsam::GaussianBayesTree::sharedClique &c, std::size_t parent_index)
                    : clique(c), parent(parent_index)
                {
                    const gtsam::GaussianConditional &conditional = *c->conditional();
                    offsets.push_back(0);
                    for (auto it = conditional.begin(); it != conditional.end(); ++it)
                    {
                        keys.push_back(*it);
                        offsets.push_back(offsets.back() + conditional.getDim(it));
                    }
                    frontal_dim = offsets[conditional.nrFrontals()];
                }

                /// Position of `key` in keys, or nullopt if it is not in this clique's F u S.
                std::optional<std::size_t> index(gtsam::Key key) const
                {
                    for (std::size_t i = 0; i < keys.size(); i++)
                        if (keys[i] == key)
                            return i;
                    return std::nullopt;
                }

                /// Dimension of keys[i].
                Eigen::Index dim(std::size_t i) const { return offsets[i + 1] - offsets[i]; }

                /// The (keys[i], keys[j]) block of a joint covariance laid out like this clique.
                gtsam::Matrix block(const gtsam::Matrix &joint, std::size_t i, std::size_t j) const
                {
                    return joint.block(offsets[i], offsets[j], dim(i), dim(j));
                }
            };

            /// Joint covariance over the clique's F u S, given its parent's joint (both unused at a root).
            inline gtsam::Matrix cliqueJoint(const Clique &clique, const Clique *parent, const gtsam::Matrix *parent_joint)
            {
                const gtsam::GaussianConditional &conditional = *clique.clique->conditional();
                gtsam::Matrix R = conditional.R();
                gtsam::Matrix S = conditional.S();
                if (const auto &model = conditional.get_model())
                {
                    // Whiten, so the conditional has unit noise (constrained models are rejected by recover).
                    R = model->invsigmas().asDiagonal() * R;
                    S = model->invsigmas().asDiagonal() * S;
                }
                const Eigen::Index nF = clique.frontal_dim, n = clique.offsets.back(), nS = n - nF;
                gtsam::Matrix R_inv = gtsam::Matrix::Identity(nF, nF);
                R.triangularView<Eigen::Upper>().solveInPlace(R_inv);
                if (nS == 0)
                    return R_inv * R_inv.transpose();

                // The separator's joint covariance, gathered from the parent's joint.
                const std::size_t nf_keys = conditional.nrFrontals();
                std::vector<std::size_t> in_parent(clique.keys.size());
                for (std::size_t i = nf_keys; i < clique.keys.size(); i++)
                    in_parent[i] = parent->index(clique.keys[i]).value(); // running intersection: always present
                gtsam::Matrix Sigma_SS(nS, nS);
                for (std::size_t i = nf_keys; i < clique.keys.size(); i++)
                    for (std::size_t j = nf_keys; j < clique.keys.size(); j++)
                        Sigma_SS.block(clique.offsets[i] - nF, clique.offsets[j] - nF, clique.dim(i), clique.dim(j)) =
                            parent->block(*parent_joint, in_parent[i], in_parent[j]);

                const gtsam::Matrix gain = -R_inv * S; // G = -R^-1 S
                const gtsam::Matrix Sigma_FS = gain * Sigma_SS;
                gtsam::Matrix Sigma_FF = R_inv * R_inv.transpose() + Sigma_FS * gain.transpose();
                Sigma_FF = 0.5 * (Sigma_FF + Sigma_FF.transpose());

                gtsam::Matrix joint(n, n);
                joint.topLeftCorner(nF, nF) = Sigma_FF;
                joint.topRightCorner(nF, nS) = Sigma_FS;
                joint.bottomLeftCorner(nS, nF) = Sigma_FS.transpose();
                joint.bottomRightCorner(nS, nS) = Sigma_SS;
                return joint;
            }

            /// A requested block Sigma_{frontal, other}, served by the clique where `frontal` is a frontal key.
            struct Request
            {
                std::size_t pair;          ///< index of the requested pair
                gtsam::Key frontal, other; ///< the pair, in the order the block is computed
            };
        } // namespace detail

        /**
         * @brief Zero-information factors, one over each group's keys that are in `values`.
         *
         * They leave the linearized problem (and so every covariance) unchanged, but make
         * the elimination treat each group's keys as linked, so that every pair of a group
         * shares a clique of the resulting Bayes tree and is recovered by recover(). Keys
         * missing from `values` (e.g. locked states) are skipped.
         */
        inline gtsam::GaussianFactorGraph linkingFactors(const std::vector<gtsam::KeyVector> &groups, const gtsam::Values &values)
        {
            gtsam::GaussianFactorGraph factors;
            for (const gtsam::KeyVector &group : groups)
            {
                std::vector<std::pair<gtsam::Key, gtsam::Matrix>> terms;
                for (const gtsam::Key key : group)
                    if (values.exists(key))
                        terms.emplace_back(key, gtsam::Matrix::Zero(1, values.at(key).dim()));
                if (terms.size() > 1)
                    factors.emplace_shared<gtsam::JacobianFactor>(terms, gtsam::Vector::Zero(1), gtsam::noiseModel::Unit::Create(1));
            }
            return factors;
        }

        /**
         * @brief Recover the covariance blocks Sigma_ab of the requested pairs.
         * @param tree A Bayes tree of the linearized problem (e.g. BayesTreeMarginals::bayesTree).
         * @param pairs Requested (a, b) pairs; a == b requests a variance.
         * @return The recovered blocks, and the pairs that no single clique contains. If a
         *         conditional has a constrained (hard) noise model, nothing is recovered and
         *         every pair is unresolved.
         */
        inline Result recover(const gtsam::GaussianBayesTree &tree, const std::vector<KeyPair> &pairs)
        {
            Result result;

            // The cliques, grouped by depth from the roots.
            std::vector<std::vector<detail::Clique>> levels(1);
            std::unordered_map<const gtsam::GaussianBayesTreeClique *, std::pair<std::size_t, std::size_t>> location;
            for (const auto &root : tree.roots())
                levels[0].emplace_back(root, detail::NO_PARENT);
            for (std::size_t level = 0; !levels[level].empty(); level++)
            {
                levels.emplace_back();
                for (std::size_t c = 0; c < levels[level].size(); c++)
                {
                    const detail::Clique &clique = levels[level][c];
                    const auto &model = clique.clique->conditional()->get_model();
                    if (model && model->isConstrained())
                    {
                        result.unresolved = pairs;
                        return result;
                    }
                    location[clique.clique.get()] = {level, c};
                    for (const auto &child : clique.clique->children)
                        levels[level + 1].emplace_back(child, c);
                }
            }
            levels.pop_back();

            // Assign each pair to the clique of one key's frontal, if the other key is in that clique's F u S.
            std::vector<std::vector<std::vector<detail::Request>>> assigned(levels.size());
            for (std::size_t level = 0; level < levels.size(); level++)
                assigned[level].resize(levels[level].size());
            auto assign = [&](std::size_t p, gtsam::Key frontal, gtsam::Key other)
            {
                const auto node = tree.nodes().find(frontal);
                if (node == tree.nodes().end())
                    return false;
                const auto [level, c] = location.at(node->second.get());
                if (!levels[level][c].index(other))
                    return false;
                assigned[level][c].push_back({p, frontal, other});
                return true;
            };
            for (std::size_t p = 0; p < pairs.size(); p++)
            {
                const auto [a, b] = pairs[p];
                if (!assign(p, a, b) && !assign(p, b, a))
                    result.unresolved.push_back(pairs[p]);
            }

            // Top-down pass, keeping only the previous level's joints.
            std::vector<gtsam::Matrix> parent_joints;
            for (std::size_t level = 0; level < levels.size(); level++)
            {
                const std::vector<detail::Clique> &cliques = levels[level];
                std::vector<gtsam::Matrix> joints(cliques.size());
                for (std::size_t c = 0; c < cliques.size(); c++)
                {
                    const detail::Clique &clique = cliques[c];
                    if (clique.clique->children.empty() && assigned[level][c].empty())
                        continue; // nothing below it and nothing requested from it
                    const detail::Clique *parent = level == 0 ? nullptr : &levels[level - 1][clique.parent];
                    joints[c] = detail::cliqueJoint(clique, parent, parent ? &parent_joints[clique.parent] : nullptr);
                    for (const detail::Request &request : assigned[level][c])
                    {
                        gtsam::Matrix block = clique.block(joints[c], *clique.index(request.frontal), *clique.index(request.other));
                        if (request.frontal != pairs[request.pair].first)
                            block.transposeInPlace();
                        result.blocks[pairs[request.pair]] = std::move(block);
                    }
                }
                parent_joints = std::move(joints);
            }
            return result;
        }
    } // namespace BayesTreeCovariance
}

#endif // BAYESTREECOVARIANCE_HPP
