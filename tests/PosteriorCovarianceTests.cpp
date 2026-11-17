#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>

#include "NumericalDifferentiation.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using ValuesType = Priors::WNOA::Values<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using PriorType = Priors::WNOA::Prior<Pose>;
        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        using NumDiff = Testing::NumericalDifferentiation<Pose>;

        constexpr int dim = gtsam::traits<Pose>::dimension;
        constexpr int node_dim = NodeType::dim;
        using MatrixN = Eigen::Matrix<double, dim, dim>;
        using VectorNode = Eigen::Matrix<double, node_dim, 1>;

        /// Two scalar-ish (2D vector) variables x0 -- x1 with a prior on x0 and a between
        /// factor, whose posterior covariance is known in closed form:
        /// Sigma_00 = s0^2 I, Sigma_01 = s0^2 I, Sigma_11 = (s0^2 + s1^2) I.
        struct LinearChain
        {
            const double s0 = 0.5, s1 = 2.0;
            const gtsam::Key x0 = gtsam::Symbol('x', 0), x1 = gtsam::Symbol('x', 1), x2 = gtsam::Symbol('x', 2);
            gtsam::NonlinearFactorGraph graph;
            gtsam::Values values;

            LinearChain()
            {
                graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector2>>(x0, gtsam::Vector2::Zero(), gtsam::noiseModel::Isotropic::Sigma(2, s0));
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Vector2>>(x0, x1, gtsam::Vector2::Zero(), gtsam::noiseModel::Isotropic::Sigma(2, s1));
                graph.emplace_shared<gtsam::BetweenFactor<gtsam::Vector2>>(x1, x2, gtsam::Vector2::Zero(), gtsam::noiseModel::Isotropic::Sigma(2, s1));
                values.insert(x0, gtsam::Vector2(0.1, -0.2));
                values.insert(x1, gtsam::Vector2(0.3, 0.4));
                values.insert(x2, gtsam::Vector2(-0.5, 0.2));
            }
        };

        std::shared_ptr<PriorType> makePrior(unsigned int spatial_dim, double length = 1.0)
        {
            auto state = std::make_shared<StateType>(length, spatial_dim);
            ValuesType node_reference;
            node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
            return std::make_shared<PriorType>(state, node_reference);
        }

        /// Extend `prior` directly (no estimator) with one row per requested time.
        gtsam::Values extendRows(const std::shared_ptr<PriorType> &prior, KeyManager &km, std::initializer_list<double> times)
        {
            gtsam::Values values;
            for (const double t : times)
            {
                gtsam::Values new_values;
                prior->extendState(km, t, values, new_values);
                values.insert(new_values);
            }
            return values;
        }

        struct PriorWithState
        {
            std::shared_ptr<StateType> state;
            std::shared_ptr<PriorType> prior;
        };

        PriorWithState makePriorWithState(unsigned int spatial_dim, double length = 1.0)
        {
            auto state = std::make_shared<StateType>(length, spatial_dim);
            ValuesType node_reference;
            node_reference.epsilon((Eigen::Vector<double, 6>() << 0, 0, 0, 0, 0, 1.0).finished());
            return {state, std::make_shared<PriorType>(state, node_reference)};
        }

        gtsam::KeyVector keysOf(std::initializer_list<std::shared_ptr<NodeType>> nodes)
        {
            gtsam::KeyVector keys;
            for (const auto &node : nodes)
            {
                const gtsam::KeyVector node_keys = node->keys();
                keys.insert(keys.end(), node_keys.begin(), node_keys.end());
            }
            return keys;
        }

        void insertNode(const std::shared_ptr<NodeType> &node, const ValuesType &v, gtsam::Values &values)
        {
            node->insert("pose", v.pose(), values);
            node->insert("strain", v.epsilon(), values);
            node->insert("velocity", v.varpi(), values);
        }

        /// Deterministic small-angle node values (as in WNOAInterpolationTests), varied by index.
        ValuesType smallValues(int index)
        {
            const double scale = 0.02;
            Eigen::Matrix<double, 6, 1> pose_vec, eps_vec, varpi_vec;
            const double sign = (index % 2 == 0) ? 1.0 : -1.0;
            pose_vec << 0.3 * sign, -0.2, 0.5, 1.0 + 0.1 * index, -0.5, 0.2 * sign;
            eps_vec << 0.5, -0.3 * sign, 0.2, -0.1, 0.4, 0.3 + 0.1 * index;
            varpi_vec << 0.1 * index, -0.4, 0.3 * sign, 0.2, -0.1, 0.5;
            ValuesType v;
            v.pose(gtsam::traits<Pose>::Expmap(pose_vec * scale));
            v.epsilon(eps_vec * scale);
            v.varpi(varpi_vec * scale);
            return v;
        }

        InterpolatorType makeInterpolator()
        {
            return InterpolatorType(MatrixN::Identity() * 0.7, MatrixN::Identity() * 0.5,
                                    MatrixN::Identity() * 0.3, MatrixN::Identity() * 0.2);
        }

        /// A random SPD matrix of size n, scaled by `scale`, from a fixed seed.
        Eigen::MatrixXd randomSPD(Eigen::Index n, double scale, unsigned int seed)
        {
            std::mt19937 rng(seed);
            std::normal_distribution<double> normal(0.0, 1.0);
            Eigen::MatrixXd A(n, n);
            for (Eigen::Index i = 0; i < n; i++)
                for (Eigen::Index j = 0; j < n; j++)
                    A(i, j) = normal(rng);
            return scale * (A * A.transpose() / static_cast<double>(n) + 0.1 * Eigen::MatrixXd::Identity(n, n));
        }

        /// A PosteriorCovariance holding every block of the dense joint `P` over `keys` (each of dimension dim).
        PosteriorCovariance posteriorFromDense(const gtsam::KeyVector &keys, const Eigen::MatrixXd &P)
        {
            PosteriorCovariance posterior;
            for (size_t i = 0; i < keys.size(); i++)
                for (size_t j = i; j < keys.size(); j++)
                    posterior.insert(keys[i], keys[j], P.block(i * dim, j * dim, dim, dim));
            return posterior;
        }

        void expectSymmetricPSD(const Eigen::MatrixXd &P, double tol = 1e-10)
        {
            ASSERT_EQ(P.rows(), P.cols());
            EXPECT_TRUE(P.isApprox(P.transpose(), 1e-12)) << "not symmetric";
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(P);
            EXPECT_GE(solver.eigenvalues().minCoeff(), -tol) << "not PSD";
        }

        /// A 2 x 2 (space x time) grid of corners at s in {0, 3}, t in {0, 2}.
        struct CornerGrid
        {
            KeyManager km;
            std::shared_ptr<NodeType> node00, node01, node10, node11;
            gtsam::Values values;

            CornerGrid()
            {
                node00 = std::make_shared<NodeType>(km, 0.0, 0.0);
                node01 = std::make_shared<NodeType>(km, 0.0, 2.0);
                node10 = std::make_shared<NodeType>(km, 3.0, 0.0);
                node11 = std::make_shared<NodeType>(km, 3.0, 2.0);
                insertNode(node00, smallValues(0), values);
                insertNode(node01, smallValues(1), values);
                insertNode(node10, smallValues(2), values);
                insertNode(node11, smallValues(3), values);
            }

            /// The corner keys in (pose, strain, velocity) order per node, matching
            /// the Jacobian layout the tests assemble.
            gtsam::KeyVector keys(std::initializer_list<std::shared_ptr<NodeType>> nodes) const
            {
                gtsam::KeyVector out;
                for (const auto &n : nodes)
                    for (const gtsam::Key k : {n->poseKey(), n->strainKey(), n->velocityKey()})
                        out.push_back(k);
                return out;
            }
        };

        std::string writeTempConfig(const std::string &name, const std::string &contents)
        {
            const std::string path = testing::TempDir() + name;
            std::ofstream(path) << contents;
            return path;
        }
    } // namespace

    // ------------------------------------------------------------------
    // PosteriorCovariance container
    // ------------------------------------------------------------------

    TEST(PosteriorCovariance, MatchesClosedFormOnALinearChain)
    {
        LinearChain chain;
        gtsam::Marginals marginals(chain.graph, chain.values);
        auto posterior = PosteriorCovariance::compute(marginals, {{chain.x0, chain.x1}});

        const double v0 = chain.s0 * chain.s0, v1 = v0 + chain.s1 * chain.s1;
        EXPECT_TRUE(posterior->covariance(chain.x0).isApprox(v0 * Eigen::Matrix2d::Identity(), 1e-9));
        EXPECT_TRUE(posterior->covariance(chain.x1).isApprox(v1 * Eigen::Matrix2d::Identity(), 1e-9));
        EXPECT_TRUE(posterior->covariance(chain.x0, chain.x1).isApprox(v0 * Eigen::Matrix2d::Identity(), 1e-9));
        EXPECT_EQ(posterior->size(), 3u); // (x0,x0), (x0,x1), (x1,x1)
    }

    TEST(PosteriorCovariance, CrossCovarianceIsTransposeWhenKeysAreSwapped)
    {
        PosteriorCovariance posterior;
        const gtsam::Key a = gtsam::Symbol('a', 0), b = gtsam::Symbol('b', 0);
        Eigen::MatrixXd block(2, 3);
        block << 1, 2, 3, 4, 5, 6;
        posterior.insert(b, a, block); // stored under (a, b) as its transpose
        EXPECT_TRUE(posterior.covariance(b, a).isApprox(block));
        EXPECT_TRUE(posterior.covariance(a, b).isApprox(block.transpose()));
        EXPECT_TRUE(posterior.has(a, b));
        EXPECT_TRUE(posterior.has(b, a));
        EXPECT_FALSE(posterior.has(a));
    }

    TEST(PosteriorCovariance, MissingBlocksThrowAndJointReportsFailure)
    {
        LinearChain chain;
        gtsam::Marginals marginals(chain.graph, chain.values);
        auto posterior = PosteriorCovariance::compute(marginals, {{chain.x0, chain.x1}, {chain.x2}});

        EXPECT_NO_THROW(posterior->covariance(chain.x2));
        EXPECT_THROW(posterior->covariance(chain.x0, chain.x2), std::out_of_range);
        EXPECT_THROW(posterior->covariance(gtsam::Symbol('y', 0)), std::out_of_range);

        Eigen::MatrixXd out = Eigen::MatrixXd::Constant(1, 1, 42.0);
        EXPECT_FALSE(posterior->joint({chain.x0, chain.x2}, out)); // cross block not extracted
        EXPECT_EQ(out(0, 0), 42.0) << "joint must leave its output untouched on failure";
    }

    TEST(PosteriorCovariance, JointMatchesGtsamInAnyKeyOrder)
    {
        LinearChain chain;
        gtsam::Marginals marginals(chain.graph, chain.values);
        auto posterior = PosteriorCovariance::compute(marginals, {{chain.x0, chain.x1, chain.x2}});

        const gtsam::KeyVector order = {chain.x2, chain.x0, chain.x1}; // deliberately not sorted
        Eigen::MatrixXd assembled;
        ASSERT_TRUE(posterior->joint(order, assembled));
        ASSERT_EQ(assembled.rows(), 6);

        const gtsam::JointMarginal reference = marginals.jointMarginalCovariance(order);
        for (size_t i = 0; i < order.size(); i++)
            for (size_t j = 0; j < order.size(); j++)
                EXPECT_TRUE(assembled.block(2 * i, 2 * j, 2, 2).isApprox(reference.at(order[i], order[j]), 1e-9));
    }

    TEST(PosteriorCovariance, EmptyGroupsGiveAnEmptyStore)
    {
        LinearChain chain;
        gtsam::Marginals marginals(chain.graph, chain.values);
        auto posterior = PosteriorCovariance::compute(marginals, {});
        EXPECT_TRUE(posterior->empty());
        auto with_empty_group = PosteriorCovariance::compute(marginals, {gtsam::KeyVector{}});
        EXPECT_TRUE(with_empty_group->empty());

        Eigen::MatrixXd out;
        EXPECT_TRUE(posterior->joint({}, out)); // the joint of no keys is the empty matrix
        EXPECT_EQ(out.size(), 0);
    }

    // ------------------------------------------------------------------
    // WNOA::Prior::covarianceKeyGroups (the extraction policy)
    // ------------------------------------------------------------------

    TEST(PosteriorCovariance, WnoaGroupsAreOneCellPerGridSquare)
    {
        auto [state, prior] = makePriorWithState(3);
        KeyManager km;
        const gtsam::Values values = extendRows(prior, km, {0.0, 0.1});
        const auto &n = state->estimation_nodes; // row-major: n[row * 3 + s]

        const auto groups = prior->covarianceKeyGroups(values);
        ASSERT_EQ(groups.size(), 2u);
        EXPECT_EQ(groups[0], keysOf({n[0], n[3], n[1], n[4]}));
        EXPECT_EQ(groups[1], keysOf({n[1], n[4], n[2], n[5]}));
    }

    TEST(PosteriorCovariance, WnoaGroupsFallBackToArclengthPairsForASingleRow)
    {
        auto [state, prior] = makePriorWithState(3);
        KeyManager km;
        const gtsam::Values values = extendRows(prior, km, {0.0});
        const auto &n = state->estimation_nodes;

        const auto groups = prior->covarianceKeyGroups(values);
        ASSERT_EQ(groups.size(), 2u);
        EXPECT_EQ(groups[0], keysOf({n[0], n[1]}));
        EXPECT_EQ(groups[1], keysOf({n[1], n[2]}));
    }

    TEST(PosteriorCovariance, WnoaGroupsAreTimePairsForASingleSpatialNode)
    {
        auto [state, prior] = makePriorWithState(1);
        KeyManager km;
        const gtsam::Values values = extendRows(prior, km, {0.0, 0.1, 0.2});
        const auto &n = state->estimation_nodes;

        const auto groups = prior->covarianceKeyGroups(values);
        ASSERT_EQ(groups.size(), 2u);
        EXPECT_EQ(groups[0], keysOf({n[0], n[1]}));
        EXPECT_EQ(groups[1], keysOf({n[1], n[2]}));
    }

    TEST(PosteriorCovariance, WnoaGroupsSkipMarginalizedRows)
    {
        auto [state, prior] = makePriorWithState(2);
        KeyManager km;
        gtsam::Values values = extendRows(prior, km, {0.0, 0.1, 0.2});
        const auto &n = state->estimation_nodes;
        for (const gtsam::Key key : keysOf({n[0], n[1]})) // row 0 marginalized out
            values.erase(key);

        const auto groups = prior->covarianceKeyGroups(values);
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_EQ(groups[0], keysOf({n[2], n[4], n[3], n[5]}));

        EXPECT_TRUE(prior->covarianceKeyGroups(gtsam::Values()).empty());
    }

    // ------------------------------------------------------------------
    // Estimator extraction API
    // ------------------------------------------------------------------

    TEST(PosteriorCovariance, EstimatorDoesNotExtractByDefault)
    {
        auto prior = makePrior(3);
        Estimator estimator(/*smootherLag=*/1.0);
        estimator.addPrior(prior);
        estimator.estimate(0.0);
        estimator.estimate(0.1);
        EXPECT_EQ(estimator.getCurrentCovariance(), nullptr);
        ASSERT_NE(estimator.getSnapshot(), nullptr);
        EXPECT_EQ(estimator.getSnapshot()->covariance, nullptr);
    }

    TEST(PosteriorCovariance, EstimatorExtractsVariancesAndNeighbourBlocks)
    {
        auto [state, prior] = makePriorWithState(3);
        EstimatorOptions options;
        options.smootherLag = 1.0;
        options.compute_covariance = true;
        Estimator estimator(options);
        estimator.addPrior(prior);
        estimator.estimate(0.0);
        estimator.estimate(0.1);

        auto posterior = estimator.getCurrentCovariance();
        ASSERT_NE(posterior, nullptr);
        ASSERT_NE(estimator.getSnapshot(), nullptr);
        EXPECT_EQ(estimator.getSnapshot()->covariance, posterior);

        const gtsam::Marginals reference(estimator.getFactors(), estimator.getCurrentValues());
        const auto &n = state->estimation_nodes; // row-major, 3 per row
        auto expect_block = [&](const std::shared_ptr<NodeType> &a, const std::shared_ptr<NodeType> &b)
        {
            for (const gtsam::Key ka : a->keys())
                for (const gtsam::Key kb : b->keys())
                {
                    ASSERT_TRUE(posterior->has(ka, kb));
                    const gtsam::Matrix expected = (ka == kb) ? reference.marginalCovariance(ka)
                                                              : reference.jointMarginalCovariance({ka, kb}).at(ka, kb);
                    EXPECT_TRUE(posterior->covariance(ka, kb).isApprox(expected, 1e-8));
                    EXPECT_TRUE(estimator.covariance(ka, kb).isApprox(expected, 1e-8));
                }
        };
        for (const auto &node : n)
            expect_block(node, node);          // variances
        expect_block(n[0], n[3]);              // time neighbours
        expect_block(n[0], n[1]);              // arclength neighbours
        expect_block(n[1], n[3]);              // cell diagonal (needed by 2D interpolation)
        EXPECT_FALSE(posterior->has(n[0]->poseKey(), n[2]->poseKey())) << "non-neighbours are not extracted";

        // The on-demand API still serves blocks the policy did not extract.
        EXPECT_TRUE(estimator.covariance(n[0]->poseKey(), n[2]->poseKey())
                        .isApprox(reference.jointMarginalCovariance({n[0]->poseKey(), n[2]->poseKey()}).at(n[0]->poseKey(), n[2]->poseKey()), 1e-8));
        EXPECT_TRUE(estimator.covariance(n[4]->poseKey()).isApprox(posterior->covariance(n[4]->poseKey()), 1e-8));
    }

    TEST(PosteriorCovariance, EstimatorPosteriorIsNoMoreUncertainThanTheBasePrior)
    {
        // The base node carries a unary prior with covariance Q0 (identity by default);
        // conditioning on anything more can only shrink it.
        auto [state, prior] = makePriorWithState(3);
        Estimator estimator(/*smootherLag=*/1.0);
        estimator.addPrior(prior);
        estimator.estimate(0.0);

        const gtsam::Matrix base_pose_covariance = estimator.covariance(state->estimation_nodes[0]->poseKey());
        expectSymmetricPSD(base_pose_covariance);
        expectSymmetricPSD(MatrixN::Identity() - base_pose_covariance, 1e-9);
    }

    TEST(PosteriorCovariance, EstimatorCovarianceRejectsUnavailableKeys)
    {
        auto [state, prior] = makePriorWithState(2);
        Estimator estimator(/*smootherLag=*/0.15);
        EXPECT_THROW(estimator.covariance(gtsam::Symbol('p', 0)), std::runtime_error); // before the first estimate

        estimator.addPrior(prior);
        for (const double t : {0.0, 0.1, 0.2, 0.3})
            estimator.estimate(t);

        const gtsam::Key marginalized = state->estimation_nodes.front()->poseKey();
        ASSERT_FALSE(estimator.getCurrentValues().exists(marginalized));
        EXPECT_THROW(estimator.covariance(marginalized), std::out_of_range);
        EXPECT_THROW(estimator.covariance(state->estimation_nodes.back()->poseKey(), marginalized), std::out_of_range);
        EXPECT_NO_THROW(estimator.covariance(state->estimation_nodes.back()->poseKey()));
    }

    TEST(PosteriorCovariance, EstimatorExtractionSurvivesMarginalization)
    {
        auto [state, prior] = makePriorWithState(2);
        EstimatorOptions options;
        options.smootherLag = 0.15;
        options.compute_covariance = true;
        Estimator estimator(options);
        estimator.addPrior(prior);
        for (const double t : {0.0, 0.1, 0.2, 0.3})
            estimator.estimate(t);

        auto posterior = estimator.getCurrentCovariance();
        ASSERT_NE(posterior, nullptr);
        EXPECT_FALSE(posterior->has(state->estimation_nodes.front()->poseKey()));
        const auto &newest = state->estimation_nodes.back();
        ASSERT_TRUE(posterior->has(newest->poseKey()));
        EXPECT_TRUE(posterior->covariance(newest->poseKey()).isApprox(estimator.covariance(newest->poseKey()), 1e-8));
    }

    TEST(PosteriorCovariance, EstimatorKeyGroupsCanBeOverridden)
    {
        auto [state, prior] = makePriorWithState(3);
        EstimatorOptions options;
        options.smootherLag = 1.0;
        options.compute_covariance = true;
        Estimator estimator(options);
        estimator.addPrior(prior);
        const gtsam::Key far_a = gtsam::Symbol('p', 0), far_b = gtsam::Symbol('p', 2);
        estimator.setCovarianceKeyGroupsFn([&](const gtsam::Values &)
                                           { return std::vector<gtsam::KeyVector>{{far_a, far_b}}; });
        estimator.estimate(0.0);

        auto posterior = estimator.getCurrentCovariance();
        ASSERT_NE(posterior, nullptr);
        EXPECT_EQ(posterior->size(), 3u);
        EXPECT_TRUE(posterior->has(far_a, far_b));
        EXPECT_FALSE(posterior->has(gtsam::Symbol('s', 0)));
    }

    // ------------------------------------------------------------------
    // Interpolator posterior propagation
    // ------------------------------------------------------------------

    TEST(PosteriorCovariance, InterpolatedPosteriorAtANodeIsThatNodesMarginal)
    {
        CornerGrid g;
        InterpolatorType interp = makeInterpolator();
        const gtsam::KeyVector keys = g.keys({g.node00, g.node01, g.node10, g.node11});
        const Eigen::MatrixXd P = randomSPD(4 * node_dim, 1e-3, 1);
        const PosteriorCovariance posterior = posteriorFromDense(keys, P);

        Eigen::MatrixXd result;
        interp.interpolate(0.0, 0.0, g.values, g.node00, g.node01, g.node10, g.node11, nullptr, nullptr, &posterior, &result);
        ASSERT_EQ(result.rows(), node_dim);
        EXPECT_TRUE(result.isApprox(P.topLeftCorner(node_dim, node_dim), 1e-9));
    }

    TEST(PosteriorCovariance, InterpolatedPosteriorIsConditionalPlusPropagatedCorners)
    {
        CornerGrid g;
        InterpolatorType interp = makeInterpolator();
        const gtsam::KeyVector keys = g.keys({g.node00, g.node01, g.node10, g.node11});
        const Eigen::MatrixXd P = randomSPD(4 * node_dim, 1e-3, 2);
        const PosteriorCovariance posterior = posteriorFromDense(keys, P);

        std::unordered_map<gtsam::Key, Eigen::MatrixXd> jacobians;
        Eigen::MatrixXd conditional, result;
        interp.interpolate(0.8, 1.1, g.values, g.node00, g.node01, g.node10, g.node11, &jacobians, &conditional, &posterior, &result);

        Eigen::MatrixXd J(node_dim, 4 * node_dim);
        for (size_t i = 0; i < keys.size(); i++)
            J.middleCols(i * dim, dim) = jacobians.at(keys[i]);
        EXPECT_TRUE(result.isApprox(conditional + J * P * J.transpose(), 1e-10));
        expectSymmetricPSD(result);
        // More uncertain than the conditional alone, since the corners are uncertain.
        expectSymmetricPSD(result - conditional);
    }

    TEST(PosteriorCovariance, InterpolatedPosteriorIsSymmetricPSDInEveryShape)
    {
        CornerGrid g;
        InterpolatorType interp = makeInterpolator();
        const gtsam::KeyVector keys = g.keys({g.node00, g.node01, g.node10, g.node11});
        const PosteriorCovariance posterior = posteriorFromDense(keys, randomSPD(4 * node_dim, 1e-3, 3));

        struct Shape
        {
            const char *name;
            double time, arclength;
            std::shared_ptr<NodeType> n01, n10, n11;
        };
        const Shape shapes[] = {
            {"time only", 0.8, 0.0, g.node01, nullptr, nullptr},
            {"space only", 0.0, 1.1, nullptr, g.node10, nullptr},
            {"space and time", 0.8, 1.1, g.node01, g.node10, g.node11},
            {"time extrapolation", 0.5, 0.0, nullptr, nullptr, nullptr},
            {"space extrapolation, time interpolation", 0.8, 1.1, g.node01, nullptr, nullptr},
            {"double extrapolation", 0.5, 1.1, nullptr, nullptr, nullptr},
        };
        for (const Shape &shape : shapes)
        {
            SCOPED_TRACE(shape.name);
            Eigen::MatrixXd result;
            interp.interpolate(shape.time, shape.arclength, g.values, g.node00, shape.n01, shape.n10, shape.n11,
                               nullptr, nullptr, &posterior, &result);
            ASSERT_EQ(result.rows(), node_dim);
            expectSymmetricPSD(result);
        }
    }

    TEST(PosteriorCovariance, InterpolatedPosteriorIsEmptyWhenBlocksAreMissing)
    {
        CornerGrid g;
        InterpolatorType interp = makeInterpolator();
        // Only the two time neighbours at s = 0 are available -- not enough for a 2D query.
        const gtsam::KeyVector keys = g.keys({g.node00, g.node01});
        const PosteriorCovariance partial = posteriorFromDense(keys, randomSPD(2 * node_dim, 1e-3, 4));

        Eigen::MatrixXd result = Eigen::MatrixXd::Identity(3, 3);
        auto with_cov = interp.interpolate(0.8, 1.1, g.values, g.node00, g.node01, g.node10, g.node11, nullptr, nullptr, &partial, &result);
        EXPECT_EQ(result.size(), 0);
        auto mean_only = interp.interpolate(0.8, 1.1, g.values, g.node00, g.node01, g.node10, g.node11);
        const auto a = std::dynamic_pointer_cast<ValuesType>(with_cov), b = std::dynamic_pointer_cast<ValuesType>(mean_only);
        ASSERT_TRUE(a && b);
        EXPECT_TRUE(a->pose().equals(b->pose(), 1e-12));

        // ... but a time-only query at s = 0 only needs those two nodes.
        interp.interpolate(0.8, 0.0, g.values, g.node00, g.node01, nullptr, nullptr, nullptr, nullptr, &partial, &result);
        EXPECT_EQ(result.rows(), node_dim);

        // A null posterior yields an empty result too.
        interp.interpolate(0.8, 0.0, g.values, g.node00, g.node01, nullptr, nullptr, nullptr, nullptr, nullptr, &result);
        EXPECT_EQ(result.size(), 0);
    }

    TEST(PosteriorCovariance, InterpolatedPosteriorMatchesMonteCarlo)
    {
        // Sample corner perturbations from P_c, push each through the (nonlinear)
        // interpolated mean, and compare the sample covariance of the result against the
        // linearized J P_c J^T. P_c is small, so linearization error is negligible and the
        // comparison is limited by Monte Carlo noise (~sqrt(2 / N) relative).
        CornerGrid g;
        InterpolatorType interp = makeInterpolator();
        const std::vector<std::shared_ptr<NodeType>> corners = {g.node00, g.node01, g.node10, g.node11};
        const gtsam::KeyVector keys = g.keys({g.node00, g.node01, g.node10, g.node11});
        const Eigen::MatrixXd P = randomSPD(4 * node_dim, 1e-6, 5);
        const PosteriorCovariance posterior = posteriorFromDense(keys, P);

        const double query_time = 0.8, query_arclength = 1.1;
        Eigen::MatrixXd conditional, result;
        auto mean = std::dynamic_pointer_cast<ValuesType>(
            interp.interpolate(query_time, query_arclength, g.values, g.node00, g.node01, g.node10, g.node11, nullptr, &conditional, &posterior, &result));
        ASSERT_TRUE(mean);
        const Eigen::MatrixXd propagated = result - conditional; // J P_c J^T

        const Eigen::MatrixXd L = P.llt().matrixL();
        std::mt19937 rng(7);
        std::normal_distribution<double> normal(0.0, 1.0);
        const int samples = 20000;
        Eigen::MatrixXd sample_covariance = Eigen::MatrixXd::Zero(node_dim, node_dim);
        for (int s = 0; s < samples; s++)
        {
            Eigen::VectorXd z(4 * node_dim);
            for (Eigen::Index i = 0; i < z.size(); i++)
                z(i) = normal(rng);
            const Eigen::VectorXd delta = L * z;

            gtsam::Values perturbed = g.values;
            for (size_t c = 0; c < corners.size(); c++)
            {
                const ValuesType v = NumDiff::retractNode(corners[c]->getValues(g.values), delta.segment<node_dim>(c * node_dim));
                perturbed.update(corners[c]->poseKey(), v.pose());
                perturbed.update(corners[c]->strainKey(), v.epsilon());
                perturbed.update(corners[c]->velocityKey(), v.varpi());
            }
            auto sample = std::dynamic_pointer_cast<ValuesType>(
                interp.interpolate(query_time, query_arclength, perturbed, g.node00, g.node01, g.node10, g.node11));
            const VectorNode e = NumDiff::localCoordinatesNode(*mean, *sample);
            sample_covariance += e * e.transpose();
        }
        sample_covariance /= samples;

        const double relative_error = (sample_covariance - propagated).norm() / propagated.norm();
        EXPECT_LT(relative_error, 0.05) << "Monte Carlo covariance disagrees with J P_c J^T";
    }

    TEST(PosteriorCovariance, PriorQueryForwardsThePosterior)
    {
        auto [state, prior] = makePriorWithState(3);
        EstimatorOptions options;
        options.smootherLag = 1.0;
        options.compute_covariance = true;
        Estimator estimator(options);
        estimator.addPrior(prior);
        estimator.estimate(0.0);
        estimator.estimate(0.1);
        auto posterior = estimator.getCurrentCovariance();
        ASSERT_NE(posterior, nullptr);

        Eigen::MatrixXd at_node, interior;
        ASSERT_NE(prior->query(0.1, 0.5, estimator.getCurrentValues(), nullptr, posterior.get(), &at_node), nullptr);
        Eigen::MatrixXd expected;
        ASSERT_TRUE(posterior->joint(state->estimation_nodes[4]->keys(), expected));
        // Node keys() are name-ordered (pose, strain, velocity), matching the interpolator's layout.
        EXPECT_TRUE(at_node.isApprox(expected, 1e-9));

        ASSERT_NE(prior->query(0.05, 0.25, estimator.getCurrentValues(), nullptr, posterior.get(), &interior), nullptr);
        ASSERT_EQ(interior.rows(), node_dim);
        expectSymmetricPSD(interior);
    }

    // ------------------------------------------------------------------
    // ConfigLoader::loadEstimatorOptions
    // ------------------------------------------------------------------

    TEST(PosteriorCovariance, ConfigLoaderReadsTheEstimatorSection)
    {
        const std::string path = writeTempConfig("estimator_options_full.json", R"({
            "estimator": {
                "smootherLag": 0.25,
                "compute_covariance": true,
                "lm_params": { "maxIterations": 12, "lambdaInitial": 0.5 }
            }
        })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        const EstimatorOptions options = loader.loadEstimatorOptions();
        EXPECT_DOUBLE_EQ(options.smootherLag, 0.25);
        EXPECT_TRUE(options.compute_covariance);
        EXPECT_EQ(options.lm_params.maxIterations, 12u);
        EXPECT_DOUBLE_EQ(options.lm_params.lambdaInitial, 0.5);
        std::remove(path.c_str());
    }

    TEST(PosteriorCovariance, ConfigLoaderDefaultsWithoutTheEstimatorSection)
    {
        const std::string missing = writeTempConfig("estimator_options_missing.json", R"({ "other": 1 })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(missing));
        const EstimatorOptions defaults;
        const EstimatorOptions options = loader.loadEstimatorOptions();
        EXPECT_DOUBLE_EQ(options.smootherLag, defaults.smootherLag);
        EXPECT_FALSE(options.compute_covariance);
        EXPECT_EQ(options.lm_params.maxIterations, defaults.lm_params.maxIterations);
        std::remove(missing.c_str());

        const std::string partial = writeTempConfig("estimator_options_partial.json", R"({ "estimator": { "smootherLag": 0.5 } })");
        Utilities::ConfigLoader partial_loader;
        ASSERT_TRUE(partial_loader.loadConfig(partial));
        const EstimatorOptions partial_options = partial_loader.loadEstimatorOptions();
        EXPECT_DOUBLE_EQ(partial_options.smootherLag, 0.5);
        EXPECT_FALSE(partial_options.compute_covariance);
        std::remove(partial.c_str());
    }

    TEST(PosteriorCovariance, ConfigLoaderReadsTheSmootherAndOrdering)
    {
        const std::string path = writeTempConfig("estimator_options_smoother.json", R"({
            "estimator": {
                "smootherLag": 0.25,
                "smoother": "incremental",
                "isam2_params": { "relinearizeThreshold": 0.05, "relinearizeSkip": 3 },
                "lm_params": { "orderingType": "COLAMD" }
            }
        })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        const EstimatorOptions options = loader.loadEstimatorOptions();
        EXPECT_EQ(options.smoother, EstimatorOptions::Smoother::Incremental);
        EXPECT_DOUBLE_EQ(std::get<double>(options.isam2_params.relinearizeThreshold), 0.05);
        EXPECT_EQ(options.isam2_params.relinearizeSkip, 3);
        EXPECT_EQ(options.lm_params.orderingType, gtsam::Ordering::COLAMD);
        std::remove(path.c_str());

        const std::string batch = writeTempConfig("estimator_options_batch.json", R"({
            "estimator": { "smoother": "batch", "lm_params": { "orderingType": "METIS", "maxIterations": 5 } }
        })");
        Utilities::ConfigLoader batch_loader;
        ASSERT_TRUE(batch_loader.loadConfig(batch));
        const EstimatorOptions batch_options = batch_loader.loadEstimatorOptions();
        EXPECT_EQ(batch_options.smoother, EstimatorOptions::Smoother::Batch);
        EXPECT_EQ(batch_options.lm_params.orderingType, gtsam::Ordering::METIS);
        std::remove(batch.c_str());
    }

    TEST(PosteriorCovariance, ConfigLoaderKeepsTheSmootherAndOrderingDefaults)
    {
        // lm_params without an orderingType, and no smoother: the EstimatorOptions defaults.
        const std::string path = writeTempConfig("estimator_options_no_smoother.json", R"({
            "estimator": { "lm_params": { "maxIterations": 5 } }
        })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        const EstimatorOptions defaults;
        const EstimatorOptions options = loader.loadEstimatorOptions();
        EXPECT_EQ(options.smoother, defaults.smoother);
        EXPECT_EQ(options.lm_params.orderingType, defaults.lm_params.orderingType);
        EXPECT_EQ(options.isam2_params.relinearizeSkip, defaults.isam2_params.relinearizeSkip);
        std::remove(path.c_str());
    }

    TEST(PosteriorCovariance, ConfigLoaderRejectsUnknownSmootherOrOrdering)
    {
        const std::string smoother = writeTempConfig("estimator_options_bad_smoother.json", R"({ "estimator": { "smoother": "isam" } })");
        Utilities::ConfigLoader smoother_loader;
        ASSERT_TRUE(smoother_loader.loadConfig(smoother));
        EXPECT_THROW(smoother_loader.loadEstimatorOptions(), std::invalid_argument);
        std::remove(smoother.c_str());

        const std::string ordering = writeTempConfig("estimator_options_bad_ordering.json", R"({ "estimator": { "lm_params": { "orderingType": "AMD" } } })");
        Utilities::ConfigLoader ordering_loader;
        ASSERT_TRUE(ordering_loader.loadConfig(ordering));
        EXPECT_THROW(ordering_loader.loadEstimatorOptions(), std::invalid_argument);
        std::remove(ordering.c_str());
    }
} // namespace Spacetime
