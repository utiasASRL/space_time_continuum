#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>
#include <json/json.h>

#include <spacetime/estimator/prior_models/CovarianceProfile.hpp>
#include <spacetime/estimator/prior_models/WNOA/WNOAPrior.hpp>
#include <spacetime/estimator/prior_models/WNOV/WNOVPrior.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic/WNOAQuasiStaticPrior.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

// Per-arclength covariance overrides (Priors::CovarianceOverride / CovarianceProfile): how they
// resolve, and that every prior's factors and interpolation use exactly the value of their
// node, segment or cell -- i.e. agree bit for bit with a constant-covariance prior built with
// that value.
namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Factors = std::vector<std::shared_ptr<gtsam::NonlinearFactor>>;

        const std::vector<double> kArclengths = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
        constexpr unsigned int kNodes = 4;
        const std::vector<double> kTimes = {0.0, 0.1, 0.2};

        Matrix6 diag(double r, double t) { return (Vector6() << r, r, r, t, t, t).finished().asDiagonal(); }

        /// Extend `prior` by a row at each of kTimes, collecting every row's factors, and
        /// perturb the poses so that interpolation is not at a zero-error point.
        template <typename PriorT>
        std::vector<Factors> extend(PriorT &prior, gtsam::Values &values)
        {
            KeyManager km;
            std::vector<Factors> rows;
            for (const double t : kTimes)
            {
                gtsam::Values new_values;
                rows.push_back(prior.extendState(km, t, values, new_values));
                values.insert(new_values);
            }
            int i = 0;
            for (const gtsam::Key key : values.keys())
                if (gtsam::Symbol(key).chr() == 'p')
                {
                    const Vector6 d = (Vector6() << 0.01, -0.02, 0.015, 0.01, -0.01, 0.005).finished() * (1.0 + 0.1 * i++);
                    values.update(key, values.at<Pose>(key).retract(d));
                }
            return rows;
        }

        /// The spatial index of the newest-row node a prior factor belongs to: the largest
        /// spatial index among its keys in its newest row (time factors: their node; space
        /// factors: their upper node; unary factors: 0).
        template <typename StateT>
        unsigned int factorNode(const StateT &state, const gtsam::NonlinearFactor &factor)
        {
            std::map<gtsam::Key, std::pair<double, unsigned int>> key_node; // key -> (time, spatial index)
            for (unsigned int flat = 0; flat < state.estimation_nodes.size(); flat++)
                for (const gtsam::Key key : state.estimation_nodes[flat]->keys())
                    key_node[key] = {state.estimation_nodes[flat]->time, flat % state.spatial_dim};
            double newest = -1.0;
            for (const gtsam::Key key : factor.keys())
                newest = std::max(newest, key_node.at(key).first);
            unsigned int n = 0;
            for (const gtsam::Key key : factor.keys())
                if (key_node.at(key).first == newest)
                    n = std::max(n, key_node.at(key).second);
            return n;
        }

        /// Two factors with the same keys, noise model and error.
        void expectSameFactor(const gtsam::NonlinearFactor &a, const gtsam::NonlinearFactor &b, const gtsam::Values &values, const std::string &where)
        {
            ASSERT_EQ(a.keys(), b.keys()) << where;
            EXPECT_EQ(a.error(values), b.error(values)) << where;
            const auto *na = dynamic_cast<const gtsam::NoiseModelFactor *>(&a);
            const auto *nb = dynamic_cast<const gtsam::NoiseModelFactor *>(&b);
            ASSERT_EQ(na == nullptr, nb == nullptr) << where;
            if (na)
                EXPECT_TRUE(na->noiseModel()->equals(*nb->noiseModel(), 0.0)) << where;
        }

        /**
         * @brief Check every factor of `overridden`'s rows against a constant prior's, and
         *        its queries against a constant prior's.
         * @param make_constant (time-axis node index, arclength-axis segment index) ->
         *        constant prior with those values.
         */
        template <typename PriorT, typename MakeConstant>
        void expectMatchesConstantPriors(PriorT &overridden, MakeConstant make_constant)
        {
            gtsam::Values values;
            const std::vector<Factors> rows = extend(overridden, values);
            const auto &state = *overridden.state();

            // Factors: node n's take node n's time-axis value and segment n-1's arclength-axis one.
            std::map<unsigned int, std::vector<Factors>> reference_rows;
            for (unsigned int n = 0; n < kNodes; n++)
            {
                auto reference = make_constant(n, n == 0 ? 0u : n - 1);
                gtsam::Values reference_values;
                reference_rows[n] = extend(*reference, reference_values);
            }
            for (std::size_t row = 0; row < rows.size(); row++)
                for (std::size_t j = 0; j < rows[row].size(); j++)
                {
                    const unsigned int n = factorNode(state, *rows[row][j]);
                    ASSERT_EQ(reference_rows[n][row].size(), rows[row].size());
                    expectSameFactor(*rows[row][j], *reference_rows[n][row][j], values,
                                     "row " + std::to_string(row) + " factor " + std::to_string(j) + " node " + std::to_string(n));
                }

            // Queries in the cell [s_n, s_{n+1}]: node n's time-axis and segment n's
            // arclength-axis values (past the tip, the last segment's).
            struct Query
            {
                double t, s;
                unsigned int node, segment;
            };
            const std::vector<Query> queries = {
                {0.05, 0.0, 0, 0},       // time only, at the base
                {0.15, 2.0 / 3.0, 2, 2}, // time only, at a middle node
                {0.1, 0.5, 1, 1},        // arclength only
                {0.05, 0.5, 1, 1},       // both
                {0.15, 0.9, 2, 2},       // both, last cell
                {0.1, 1.2, 3, 2},        // arclength extrapolated past the tip
                {0.05, 1.2, 3, 2},       // time interpolated, arclength extrapolated
                {0.3, 0.5, 1, 1},        // time extrapolated
            };
            for (const auto &q : queries)
            {
                auto reference = make_constant(q.node, q.segment);
                gtsam::Values reference_values;
                extend(*reference, reference_values);
                Eigen::MatrixXd cov, reference_cov;
                std::unordered_map<gtsam::Key, Eigen::MatrixXd> jac, reference_jac;
                const auto node = overridden.query(q.t, q.s, values, &cov);
                const auto reference_node = reference->query(q.t, q.s, values, &reference_cov);
                overridden.queryJacobians(q.t, q.s, values, jac);
                reference->queryJacobians(q.t, q.s, values, reference_jac);
                const std::string where = "query t " + std::to_string(q.t) + " s " + std::to_string(q.s);
                ASSERT_NE(node, nullptr) << where;
                ASSERT_NE(reference_node, nullptr) << where;
                EXPECT_TRUE(cov == reference_cov) << where << "\n"
                                                  << cov - reference_cov;
                ASSERT_EQ(jac.size(), reference_jac.size()) << where;
                for (const auto &[key, block] : jac)
                    EXPECT_TRUE(block == reference_jac.at(key)) << where;
            }
        }
    } // namespace

    // ---- Resolution ------------------------------------------------------------------------

    TEST(CovarianceProfile, ResolvesNominalThenFunctionThenEntriesInOrder)
    {
        Priors::CovarianceOverride<Matrix6> overrides;
        overrides.set([](double s, const Matrix6 &nominal)
                      { return (1.0 + s) * nominal; })
            .set(1.0 / 3.0, 5.0 * Matrix6::Identity())
            .scale(1.0 / 3.0, 2.0)
            .set(-1, 7.0 * Matrix6::Identity());
        const Priors::CovarianceProfile<Matrix6> time(Matrix6::Identity(), Priors::CovarianceAxis::Time, kArclengths, overrides);
        ASSERT_FALSE(time.isConstant());
        ASSERT_EQ(time.size(), kNodes);
        EXPECT_TRUE(time.at(0) == Matrix6::Identity());
        EXPECT_TRUE(time.at(1) == 10.0 * Matrix6::Identity()); // set, then scaled
        EXPECT_TRUE(time.at(2) == (1.0 + 2.0 / 3.0) * Matrix6::Identity());
        EXPECT_TRUE(time.at(3) == 7.0 * Matrix6::Identity()); // -1: the tip

        const Priors::CovarianceProfile<Matrix6> space(Matrix6::Identity(), Priors::CovarianceAxis::Space, kArclengths, overrides);
        ASSERT_EQ(space.size(), kNodes - 1);
        EXPECT_TRUE(space.at(1) == 10.0 * Matrix6::Identity());
        EXPECT_TRUE(space.at(2) == 7.0 * Matrix6::Identity()); // -1: the last segment

        // Cell lookup: the last node at or before the arclength; past the tip, the last segment.
        EXPECT_TRUE(time.atArclength(0.5) == time.at(1));
        EXPECT_TRUE(time.atArclength(1.0 / 3.0 - 1e-12) == time.at(1));
        EXPECT_TRUE(time.atArclength(1.5) == time.at(3));
        EXPECT_TRUE(space.atArclength(1.0) == space.at(2));
        EXPECT_TRUE(space.atArclength(std::nullopt) == Matrix6::Identity());
    }

    TEST(CovarianceProfile, EmptyOverridesAreConstant)
    {
        const Priors::CovarianceProfile<Matrix6> profile(2.0 * Matrix6::Identity(), Priors::CovarianceAxis::Space, kArclengths, {});
        EXPECT_TRUE(profile.isConstant());
        EXPECT_TRUE(profile.at(5) == 2.0 * Matrix6::Identity());
        EXPECT_TRUE(profile.atArclength(0.5) == 2.0 * Matrix6::Identity());
    }

    TEST(CovarianceProfile, RejectsKeysThatAreNotNodes)
    {
        Priors::CovarianceOverride<Matrix6> off_node;
        off_node.set(0.5, Matrix6::Identity());
        EXPECT_THROW(Priors::CovarianceProfile<Matrix6>(Matrix6::Identity(), Priors::CovarianceAxis::Time, kArclengths, off_node),
                     std::invalid_argument);

        // An arclength-axis key on the last node names no segment (-1 does: the last one).
        Priors::CovarianceOverride<Matrix6> last_node;
        last_node.set(1.0, Matrix6::Identity());
        EXPECT_THROW(Priors::CovarianceProfile<Matrix6>(Matrix6::Identity(), Priors::CovarianceAxis::Space, kArclengths, last_node),
                     std::invalid_argument);
        EXPECT_NO_THROW(Priors::CovarianceProfile<Matrix6>(Matrix6::Identity(), Priors::CovarianceAxis::Time, kArclengths, last_node));
    }

    // ---- WNOA ------------------------------------------------------------------------------

    namespace
    {
        using WnoaPrior = Priors::WNOA::Prior<Pose>;

        WnoaPrior::ValuesType reference()
        {
            WnoaPrior::ValuesType values;
            values.epsilon((Vector6() << 0.1, 0.2, -0.1, 1.0, 0.0, 0.0).finished());
            return values;
        }

        Priors::WNOA::CovarianceOverrides<Pose> wnoaOverrides()
        {
            Priors::WNOA::CovarianceOverrides<Pose> overrides;
            overrides.overrideQ1(1.0 / 3.0, diag(5e2, 4.0))
                .overrideQ1(0.0, diag(1e1, 2e-1))
                .overrideQ2(0.0, diag(1e3, 2.0))
                .overrideQ3([](double s, const WnoaPrior::MatrixN &nominal)
                            { return (1.0 + 3.0 * s) * nominal; })
                .overrideQ4(-1, diag(7.0, 2e2));
            overrides.Q2.scale(2.0 / 3.0, 0.25);
            return overrides;
        }
    } // namespace

    class WNOACovarianceOverrides : public testing::TestWithParam<Priors::WNOA::FactorType>
    {
    };

    TEST_P(WNOACovarianceOverrides, FactorsAndQueriesUseTheirNodeSegmentAndCell)
    {
        Priors::WNOA::PriorOptions options;
        options.factor_type = GetParam();
        const auto [Q0, Q1, Q2, Q3, Q4] = WnoaPrior::DefaultCovariances();
        const auto overrides = wnoaOverrides();

        WnoaPrior prior(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Q0, Q1, Q2, Q3, Q4, options, overrides);
        const auto &profiles = prior.covarianceProfiles();
        EXPECT_TRUE(profiles.Q1.at(0) == diag(1e1, 2e-1));
        EXPECT_TRUE(profiles.Q1.at(2) == Q1);
        EXPECT_TRUE(profiles.Q2.at(2) == 0.25 * Q2); // keyed by the segment [2/3, 1]
        EXPECT_TRUE(profiles.Q3.at(3) == 4.0 * Q3);
        EXPECT_TRUE(profiles.Q4.at(2) == diag(7.0, 2e2));

        expectMatchesConstantPriors(prior, [&](unsigned int node, unsigned int segment)
                                    { return std::make_shared<WnoaPrior>(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Q0,
                                                                         profiles.Q1.at(node), profiles.Q2.at(segment), profiles.Q3.at(node),
                                                                         profiles.Q4.at(segment), options); });
    }

    INSTANTIATE_TEST_SUITE_P(FactorTypes, WNOACovarianceOverrides,
                             testing::Values(Priors::WNOA::FactorType::WNOA, Priors::WNOA::FactorType::Legacy,
                                             Priors::WNOA::FactorType::Experimental));

    TEST(WNOACovarianceOverridesCompat, OverridesEqualToTheNominalChangeNothing)
    {
        const auto [Q0, Q1, Q2, Q3, Q4] = WnoaPrior::DefaultCovariances();
        Priors::WNOA::CovarianceOverrides<Pose> identity;
        const auto keep = [](double, const WnoaPrior::MatrixN &nominal)
        { return nominal; };
        identity.overrideQ1(keep).overrideQ2(keep).overrideQ3(keep).overrideQ4(keep);

        WnoaPrior overridden(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Q0, Q1, Q2, Q3, Q4, {}, identity);
        ASSERT_FALSE(overridden.covarianceProfiles().Q1.isConstant());
        expectMatchesConstantPriors(overridden, [&](unsigned int, unsigned int)
                                    { return std::make_shared<WnoaPrior>(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Q0,
                                                                         Q1, Q2, Q3, Q4); });
    }

    TEST(WNOACovarianceOverridesCompat, DefaultCovariancesConstructorTakesOverrides)
    {
        Priors::WNOA::CovarianceOverrides<Pose> overrides;
        overrides.overrideQ2(1.0 / 3.0, diag(1.0, 1.0));
        WnoaPrior prior(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Priors::WNOA::PriorOptions(), overrides);
        EXPECT_TRUE(prior.covarianceProfiles().Q2.at(1) == diag(1.0, 1.0));
        EXPECT_TRUE(prior.covarianceProfiles().Q2.at(0) == std::get<2>(WnoaPrior::DefaultCovariances()));
        EXPECT_TRUE(prior.covarianceProfiles().Q1.isConstant());
    }

    TEST(WNOACovarianceOverridesCompat, InvalidKeyThrowsAtConstruction)
    {
        Priors::WNOA::CovarianceOverrides<Pose> overrides;
        overrides.overrideQ4(1.0, diag(1.0, 1.0)); // the tip starts no segment
        EXPECT_THROW(WnoaPrior(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Priors::WNOA::PriorOptions(), overrides),
                     std::invalid_argument);
        EXPECT_THROW(WnoaPrior(nullptr, reference(), Priors::WNOA::PriorOptions(), overrides), std::invalid_argument);
    }

    // ---- WNOV ------------------------------------------------------------------------------

    namespace
    {
        using WnovPrior = Priors::WNOV::Prior<Pose>;
        const Vector6 kEpsilon0 = (Vector6() << 0.1, 0.2, -0.1, 1.0, 0.0, 0.0).finished();
    } // namespace

    TEST(WNOVCovarianceOverrides, FactorsAndQueriesUseTheirNodeSegmentAndCell)
    {
        const Matrix6 Q0 = diag(1e-2, 1e-3), Q1 = diag(2.0, 0.5), Q2 = diag(3.0, 0.2);
        Priors::WNOV::CovarianceOverrides<Pose> overrides;
        overrides.overrideQ1(1.0 / 3.0, diag(9.0, 0.1))
            .overrideQ2([](double s, const Matrix6 &nominal)
                        { return (1.0 + s) * nominal; })
            .overrideQ2(0.0, diag(0.5, 0.05));

        WnovPrior prior(std::make_shared<WnovPrior::StateType>(kArclengths), Pose(), kEpsilon0, Q0, Q1, Q2, {}, overrides);
        const auto &profiles = prior.covarianceProfiles();
        EXPECT_TRUE(profiles.Q1.at(1) == diag(9.0, 0.1));
        EXPECT_TRUE(profiles.Q2.at(0) == diag(0.5, 0.05));
        EXPECT_TRUE(profiles.Q2.at(3) == 2.0 * Q2);

        expectMatchesConstantPriors(prior, [&](unsigned int node, unsigned int segment)
                                    { return std::make_shared<WnovPrior>(std::make_shared<WnovPrior::StateType>(kArclengths), Pose(), kEpsilon0, Q0,
                                                                         profiles.Q1.at(segment), profiles.Q2.at(node)); });
    }

    // ---- Quasi-static -----------------------------------------------------------------------

    namespace
    {
        using QuasiStaticPrior = Priors::WNOAQuasiStatic::Prior<Pose>;
    } // namespace

    TEST(WNOAQuasiStaticCovarianceOverrides, FactorsAndQueriesUseTheirSegment)
    {
        const auto [Q0, Q2] = QuasiStaticPrior::DefaultCovariances();
        Priors::WNOAQuasiStatic::CovarianceOverrides<Pose> overrides;
        overrides.overrideQ2(1.0 / 3.0, diag(5.0, 0.3));
        overrides.Q2.scale(-1, 4.0);
        const QuasiStaticPrior::ValuesType base(Pose(), kEpsilon0);

        QuasiStaticPrior prior(std::make_shared<QuasiStaticPrior::StateType>(kArclengths), base, Q0, Q2, {}, overrides);
        EXPECT_TRUE(prior.covarianceProfiles().Q2.at(1) == diag(5.0, 0.3));
        EXPECT_TRUE(prior.covarianceProfiles().Q2.at(2) == 4.0 * Q2);

        expectMatchesConstantPriors(prior, [&](unsigned int, unsigned int segment)
                                    { return std::make_shared<QuasiStaticPrior>(std::make_shared<QuasiStaticPrior::StateType>(kArclengths), base, Q0,
                                                                                prior.covarianceProfiles().Q2.at(segment)); });
    }

    // ---- Config -----------------------------------------------------------------------------

    namespace
    {
        Utilities::ConfigLoader loaderFor(const std::string &json)
        {
            Json::CharReaderBuilder builder;
            Json::Value root;
            std::string errors;
            std::istringstream stream(json);
            EXPECT_TRUE(Json::parseFromStream(builder, stream, &root, &errors)) << errors;
            Utilities::ConfigLoader loader;
            loader.addConfig(root);
            return loader;
        }
    } // namespace

    TEST(CovarianceOverridesConfig, LoadsDiagonalsScaledByThePriorFactorAndScales)
    {
        const auto loader = loaderFor(R"({"prior": {
            "wnoa": {"weights": {"prior_factor": 0.5},
                     "overrides": {"Q1": [{"arclength": 0.0, "diagonal": [2, 2, 2, 4, 4, 4]}],
                                   "Q2": [{"arclength": -1, "scale": 3.0}]}},
            "wnov": {"overrides": {"Q2": [{"arclength": 1.0, "diagonal": [1, 1, 1, 1, 1, 1]}, {"arclength": 1.0, "scale": 2}]}}}})");

        const auto wnoa = loader.loadWNOACovarianceOverrides<Pose>();
        const auto [Q0, Q1, Q2, Q3, Q4] = WnoaPrior::DefaultCovariances();
        WnoaPrior prior(std::make_shared<WnoaPrior::StateType>(kArclengths), reference(), Q0, Q1, Q2, Q3, Q4, {}, wnoa);
        EXPECT_TRUE(prior.covarianceProfiles().Q1.at(0) == diag(1.0, 2.0));
        EXPECT_TRUE(prior.covarianceProfiles().Q2.at(2) == 3.0 * Q2);
        EXPECT_TRUE(wnoa.Q3.empty() && wnoa.Q4.empty());

        const auto quasi_static = loader.loadWNOAQuasiStaticCovarianceOverrides<Pose>();
        ASSERT_EQ(quasi_static.Q2.entries().size(), 1u);
        EXPECT_EQ(quasi_static.Q2.entries()[0].scale, 3.0);

        const auto wnov = loader.loadWNOVCovarianceOverrides<Pose>();
        WnovPrior wnov_prior(std::make_shared<WnovPrior::StateType>(kArclengths), Pose(), kEpsilon0, Matrix6::Identity(), Matrix6::Identity(),
                             Matrix6::Identity(), {}, wnov);
        EXPECT_TRUE(wnov_prior.covarianceProfiles().Q2.at(3) == 2.0 * Matrix6::Identity());
        EXPECT_TRUE(wnov.Q1.empty());
    }

    TEST(CovarianceOverridesConfig, AbsentSectionGivesNoOverridesAndMalformedEntriesThrow)
    {
        EXPECT_TRUE(loaderFor(R"({"prior": {"wnoa": {}}})").loadWNOACovarianceOverrides<Pose>().empty());
        EXPECT_TRUE(Utilities::ConfigLoader().loadWNOVCovarianceOverrides<Pose>().empty());
        EXPECT_THROW(loaderFor(R"({"prior": {"wnoa": {"overrides": {"Q1": [{"diagonal": [1, 1, 1, 1, 1, 1]}]}}}})").loadWNOACovarianceOverrides<Pose>(),
                     std::invalid_argument);
        EXPECT_THROW(loaderFor(R"({"prior": {"wnoa": {"overrides": {"Q1": [{"arclength": 0}]}}}})").loadWNOACovarianceOverrides<Pose>(),
                     std::invalid_argument);
        EXPECT_THROW(loaderFor(R"({"prior": {"wnoa": {"overrides": {"Q1": [{"arclength": 0, "diagonal": [1, 1]}]}}}})").loadWNOACovarianceOverrides<Pose>(),
                     std::invalid_argument);
    }
} // namespace Spacetime
