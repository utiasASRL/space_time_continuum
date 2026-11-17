#include <random>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/estimator/BayesTreeCovariance.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>

namespace Spacetime
{
    namespace
    {
        using KeyPair = BayesTreeCovariance::KeyPair;

        /**
         * A random linear-Gaussian problem on a K x N grid of variables of mixed dimension,
         * with a unary factor on every variable and binary factors between grid neighbours
         * (in time and in arclength) and, optionally, across each cell's diagonal. Optionally, a "global"
         * variable linked to every node (a state that breaks the banded structure), and a
         * second, disconnected grid (a forest with several roots).
         */
        struct RandomProblem
        {
            gtsam::GaussianFactorGraph graph;
            gtsam::VectorValues solution;
            std::vector<std::vector<gtsam::Key>> grid;
            std::vector<gtsam::Key> all_keys;
            std::mt19937 rng{7};

            bool diagonals;

            RandomProblem(int K, int N, bool global = false, bool disconnected = false, bool with_diagonals = true)
                : diagonals(with_diagonals)
            {
                addGrid('x', K, N);
                if (disconnected)
                    addGrid('y', 3, 2);
                if (global)
                {
                    const gtsam::Key g = gtsam::Symbol('g', 0);
                    addVariable(g, 4);
                    for (const auto &row : grid)
                        for (const gtsam::Key key : row)
                            addBinary(key, g);
                }
            }

            size_t dim(gtsam::Key key) const { return solution.at(key).size(); }

            gtsam::Matrix random(size_t rows, size_t cols)
            {
                std::normal_distribution<double> normal;
                gtsam::Matrix A(rows, cols);
                for (Eigen::Index i = 0; i < A.size(); i++)
                    A.data()[i] = normal(rng);
                return A;
            }

            void addVariable(gtsam::Key key, size_t d)
            {
                solution.insert(key, gtsam::Vector::Zero(d));
                all_keys.push_back(key);
                graph.add(key, random(d, d) + 3.0 * gtsam::Matrix::Identity(d, d), gtsam::Vector::Zero(d),
                          gtsam::noiseModel::Unit::Create(d));
            }

            void addBinary(gtsam::Key a, gtsam::Key b)
            {
                const size_t rows = std::max(dim(a), dim(b));
                std::uniform_real_distribution<double> sigma(0.5, 2.0);
                gtsam::Vector sigmas(rows);
                for (Eigen::Index i = 0; i < sigmas.size(); i++)
                    sigmas(i) = sigma(rng);
                graph.add(a, random(rows, dim(a)), b, random(rows, dim(b)), gtsam::Vector::Zero(rows),
                          gtsam::noiseModel::Diagonal::Sigmas(sigmas));
            }

            void addGrid(char symbol, int K, int N)
            {
                const size_t dims[] = {6, 3, 2};
                const size_t first_row = grid.size();
                for (int k = 0; k < K; k++)
                {
                    grid.emplace_back();
                    for (int n = 0; n < N; n++)
                    {
                        const gtsam::Key key = gtsam::Symbol(symbol, k * N + n);
                        addVariable(key, dims[(k + n) % 3]);
                        grid.back().push_back(key);
                    }
                }
                for (int k = 0; k < K; k++)
                    for (int n = 0; n < N; n++)
                    {
                        const auto &rows = grid;
                        const gtsam::Key key = rows[first_row + k][n];
                        if (n + 1 < N)
                            addBinary(key, rows[first_row + k][n + 1]);
                        if (k + 1 < K)
                            addBinary(key, rows[first_row + k + 1][n]);
                        if (diagonals && k + 1 < K && n + 1 < N)
                            addBinary(key, rows[first_row + k + 1][n + 1]);
                    }
            }

            /// One group per grid cell (as WNOA::Prior::covarianceKeyGroups), plus `extra` groups.
            std::vector<gtsam::KeyVector> cellGroups() const
            {
                std::vector<gtsam::KeyVector> groups;
                for (size_t k = 0; k + 1 < grid.size(); k++)
                    for (size_t n = 0; n + 1 < grid[k].size() && n + 1 < grid[k + 1].size(); n++)
                        groups.push_back({grid[k][n], grid[k][n + 1], grid[k + 1][n], grid[k + 1][n + 1]});
                return groups;
            }

            std::vector<KeyPair> allPairs() const
            {
                std::vector<KeyPair> pairs;
                for (size_t i = 0; i < all_keys.size(); i++)
                    for (size_t j = i; j < all_keys.size(); j++)
                        pairs.emplace_back(std::min(all_keys[i], all_keys[j]), std::max(all_keys[i], all_keys[j]));
                return pairs;
            }
        };

        void expectMatchesMarginals(const BayesTreeCovariance::Result &result, const gtsam::Marginals &marginals)
        {
            for (const auto &[pair, block] : result.blocks)
            {
                const gtsam::Matrix reference = pair.first == pair.second
                                                    ? marginals.marginalCovariance(pair.first)
                                                    : marginals.jointMarginalCovariance({pair.first, pair.second}).at(pair.first, pair.second);
                ASSERT_EQ(block.rows(), reference.rows());
                ASSERT_EQ(block.cols(), reference.cols());
                EXPECT_LT((block - reference).norm(), 1e-10 * (1.0 + reference.norm()))
                    << gtsam::DefaultKeyFormatter(pair.first) << ", " << gtsam::DefaultKeyFormatter(pair.second);
            }
        }

        void expectSameStore(const PosteriorCovariance &a, const PosteriorCovariance &b, const std::vector<gtsam::KeyVector> &groups)
        {
            EXPECT_EQ(a.size(), b.size());
            for (const auto &group : groups)
                for (const gtsam::Key i : group)
                    for (const gtsam::Key j : group)
                    {
                        ASSERT_TRUE(a.has(i, j));
                        ASSERT_TRUE(b.has(i, j));
                        EXPECT_LT((a.covariance(i, j) - b.covariance(i, j)).norm(), 1e-10 * (1.0 + b.covariance(i, j).norm()));
                    }
        }
    } // namespace

    TEST(BayesTreeCovariance, MatchesTheClosedFormOnAScalarChain)
    {
        // x0 ~ N(0, s0^2), x1 = x0 + N(0, s1^2): var(x1) = s0^2 + s1^2, cov(x0, x1) = s0^2.
        const gtsam::Key x0 = gtsam::Symbol('x', 0), x1 = gtsam::Symbol('x', 1);
        const double s0 = 0.3, s1 = 0.7;
        gtsam::GaussianFactorGraph graph;
        graph.add(x0, gtsam::I_1x1, gtsam::Vector1(0.0), gtsam::noiseModel::Isotropic::Sigma(1, s0));
        graph.add(x0, -gtsam::I_1x1, x1, gtsam::I_1x1, gtsam::Vector1(0.0), gtsam::noiseModel::Isotropic::Sigma(1, s1));
        gtsam::VectorValues solution;
        solution.insert(x0, gtsam::Vector1(0.0));
        solution.insert(x1, gtsam::Vector1(0.0));
        const BayesTreeMarginals marginals(graph, solution);

        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), {{x0, x0}, {x0, x1}, {x1, x1}});
        EXPECT_TRUE(result.unresolved.empty());
        EXPECT_NEAR(result.blocks.at({x0, x0})(0, 0), s0 * s0, 1e-12);
        EXPECT_NEAR(result.blocks.at({x0, x1})(0, 0), s0 * s0, 1e-12);
        EXPECT_NEAR(result.blocks.at({x1, x1})(0, 0), s0 * s0 + s1 * s1, 1e-12);
    }

    TEST(BayesTreeCovariance, EveryVarianceAndFactorLinkedPairIsRecovered)
    {
        RandomProblem problem(12, 5);
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        std::vector<KeyPair> linked;
        for (const gtsam::Key key : problem.all_keys)
            linked.emplace_back(key, key);
        for (const auto &factor : problem.graph)
            if (factor->size() == 2)
                linked.emplace_back(std::min(factor->keys()[0], factor->keys()[1]), std::max(factor->keys()[0], factor->keys()[1]));

        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), linked);
        EXPECT_TRUE(result.unresolved.empty()) << "a factor's keys always share a clique";
        expectMatchesMarginals(result, marginals);
    }

    TEST(BayesTreeCovariance, HandlesAStateLinkedToEveryNode)
    {
        // A global state breaks the banded (block-tridiagonal) structure the legacy recursion needs.
        RandomProblem problem(10, 4, true);
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), problem.allPairs());
        EXPECT_FALSE(result.blocks.empty());
        expectMatchesMarginals(result, marginals);
        const gtsam::Key g = gtsam::Symbol('g', 0);
        for (const gtsam::Key key : problem.all_keys)
            EXPECT_TRUE(result.blocks.count({std::min(key, g), std::max(key, g)})) << "g is linked to every node";
    }

    TEST(BayesTreeCovariance, HandlesAForestWithSeveralRoots)
    {
        RandomProblem problem(6, 3, false, true);
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        ASSERT_GE(marginals.bayesTree().roots().size(), 2u);
        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), problem.allPairs());
        expectMatchesMarginals(result, marginals);
        for (const gtsam::Key key : problem.all_keys)
            EXPECT_TRUE(result.blocks.count({key, key}));
    }

    TEST(BayesTreeCovariance, ReportsPairsOutsideEveryCliqueAsUnresolved)
    {
        RandomProblem problem(20, 1); // a chain: its ends never share a clique
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        const gtsam::Key first = problem.grid.front()[0], last = problem.grid.back()[0];
        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), {{first, last}, {first, first}});
        ASSERT_EQ(result.unresolved.size(), 1u);
        EXPECT_EQ(result.unresolved[0], KeyPair(first, last));
        EXPECT_TRUE(result.blocks.count({first, first}));
    }

    TEST(BayesTreeCovariance, LinkingFactorsMakeEveryGroupRecoverableWithoutChangingIt)
    {
        // Without diagonal factors (the WNOA prior's structure), a cell's diagonal corners need not share a clique.
        RandomProblem problem(15, 5, true, false, false);
        const auto groups = problem.cellGroups();
        std::vector<KeyPair> pairs;
        for (const auto &group : groups)
            for (size_t i = 0; i < group.size(); i++)
                for (size_t j = i; j < group.size(); j++)
                    pairs.emplace_back(std::min(group[i], group[j]), std::max(group[i], group[j]));

        gtsam::Values values;
        for (const gtsam::Key key : problem.all_keys)
            values.insert(key, gtsam::Vector(problem.solution.at(key)));
        const gtsam::GaussianFactorGraph links = BayesTreeCovariance::linkingFactors(groups, values);
        EXPECT_EQ(links.size(), groups.size());
        gtsam::GaussianFactorGraph linked = problem.graph;
        linked.push_back(links);

        const BayesTreeMarginals reference(problem.graph, problem.solution);
        const BayesTreeMarginals marginals(linked, problem.solution);
        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), pairs);
        EXPECT_TRUE(result.unresolved.empty());
        expectMatchesMarginals(result, reference); // the links carry no information

        // Keys missing from the values (locked states) are left out of the links.
        gtsam::Values partial = values;
        partial.erase(problem.grid[0][0]);
        const auto partial_links = BayesTreeCovariance::linkingFactors({{problem.grid[0][0], problem.grid[0][1]}, groups[0]}, partial);
        ASSERT_EQ(partial_links.size(), 1u) << "a group with a single estimated key needs no link";
        EXPECT_EQ(partial_links[0]->size(), 3u);
    }

    TEST(BayesTreeCovariance, UnknownKeysAreUnresolved)
    {
        RandomProblem problem(3, 2);
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        const gtsam::Key missing = gtsam::Symbol('z', 0);
        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), {{missing, missing}});
        EXPECT_TRUE(result.blocks.empty());
        EXPECT_EQ(result.unresolved.size(), 1u);
    }

    TEST(BayesTreeCovariance, ConstrainedConditionalsLeaveEverythingUnresolved)
    {
        RandomProblem problem(4, 3);
        const gtsam::Key key = problem.grid[0][0];
        problem.graph.add(key, gtsam::Matrix::Identity(problem.dim(key), problem.dim(key)), gtsam::Vector::Zero(problem.dim(key)),
                          gtsam::noiseModel::Constrained::All(problem.dim(key)));
        const BayesTreeMarginals marginals(problem.graph, problem.solution, gtsam::Marginals::QR);
        const auto pairs = problem.allPairs();
        const auto result = BayesTreeCovariance::recover(marginals.bayesTree(), pairs);
        EXPECT_TRUE(result.blocks.empty());
        EXPECT_EQ(result.unresolved.size(), pairs.size());

        // PosteriorCovariance then queries everything from the marginals.
        const auto groups = problem.cellGroups();
        const auto with_tree = PosteriorCovariance::compute(marginals, groups, gtsam::Values(), &marginals.bayesTree());
        const auto without_tree = PosteriorCovariance::compute(marginals, groups);
        expectSameStore(*with_tree, *without_tree, groups);
    }

    TEST(BayesTreeCovariance, PosteriorCovarianceIsTheSameWithAndWithoutTheTree)
    {
        RandomProblem problem(10, 4, true);
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        auto groups = problem.cellGroups();
        groups.push_back({problem.grid.front()[0], problem.grid.back()[3]}); // far apart: needs the fallback
        groups.push_back({gtsam::Symbol('g', 0), problem.grid[4][2]});

        const auto with_tree = PosteriorCovariance::compute(marginals, groups, gtsam::Values(), &marginals.bayesTree());
        const auto without_tree = PosteriorCovariance::compute(marginals, groups);
        expectSameStore(*with_tree, *without_tree, groups);
    }

    TEST(BayesTreeCovariance, PosteriorCovarianceKeepsKnownKeysAsZeroBlocks)
    {
        RandomProblem problem(5, 3);
        const BayesTreeMarginals marginals(problem.graph, problem.solution);
        const gtsam::Key locked = gtsam::Symbol('l', 0);
        gtsam::Values known;
        known.insert(locked, gtsam::Vector3(1.0, 2.0, 3.0));
        std::vector<gtsam::KeyVector> groups = problem.cellGroups();
        groups.push_back({locked, problem.grid[0][0], problem.grid[0][1]});

        const auto with_tree = PosteriorCovariance::compute(marginals, groups, known, &marginals.bayesTree());
        const auto without_tree = PosteriorCovariance::compute(marginals, groups, known);
        expectSameStore(*with_tree, *without_tree, groups);
        EXPECT_TRUE(with_tree->covariance(locked).isZero());
        EXPECT_EQ(with_tree->covariance(locked, problem.grid[0][0]).rows(), 3);
        EXPECT_EQ(with_tree->covariance(locked, problem.grid[0][0]).cols(), static_cast<Eigen::Index>(problem.dim(problem.grid[0][0])));
        EXPECT_TRUE(with_tree->covariance(locked, problem.grid[0][0]).isZero());
    }
}
