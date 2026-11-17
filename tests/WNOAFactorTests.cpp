#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/WnoaFactor.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

namespace Spacetime::Priors::WNOA
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = WNOA::Values<Pose>;
        using Matrix6 = Eigen::Matrix<double, 6, 6>;
        using Vector6 = Eigen::Matrix<double, 6, 1>;

        Matrix6 diag(std::initializer_list<double> values)
        {
            return Vector6(Eigen::Map<const Vector6>(std::data(values))).asDiagonal();
        }

        const Matrix6 kQa = diag({0.2, 0.3, 0.4, 2.0, 3.0, 5.0});
        const Matrix6 kQb = diag({7.0, 1.5, 0.5, 0.1, 0.05, 9.0});

        gtsam::Matrix covarianceOf(const gtsam::NonlinearFactor &factor)
        {
            auto noise = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
                dynamic_cast<const gtsam::NoiseModelFactor &>(factor).noiseModel());
            return noise->covariance();
        }

        void insertValues(const std::shared_ptr<NodeType> &node, const ValueType &v, gtsam::Values &values)
        {
            node->insert("pose", v.pose(), values);
            node->insert("strain", v.epsilon(), values);
            node->insert("velocity", v.varpi(), values);
        }

        std::string writeTempConfig(const std::string &name, const std::string &contents)
        {
            // Per-test (and per-process) name: ctest runs tests concurrently.
            const std::string path = testing::TempDir() + testing::UnitTest::GetInstance()->current_test_info()->name() + "_" + std::to_string(::getpid()) + "_" + name;
            std::ofstream(path) << contents;
            return path;
        }

        template <typename T>
        std::size_t countFactors(const std::vector<std::shared_ptr<gtsam::NonlinearFactor>> &factors)
        {
            std::size_t n = 0;
            for (const auto &f : factors)
                n += std::dynamic_pointer_cast<T>(f) != nullptr;
            return n;
        }
    } // namespace

    // ------------------------------------------------------------------
    // BinarySpaceFactor / BinaryTimeFactor
    // ------------------------------------------------------------------

    TEST(WNOAFactors, EachFactorUsesItsOwnSpacing)
    {
        // Factors built back to back with different spacings must not share a noise model.
        KeyManager km;
        auto n0 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto n1 = std::make_shared<NodeType>(km, 0.1, 0.0);
        auto n2 = std::make_shared<NodeType>(km, 0.4, 0.0);
        auto short_factor = BinarySpaceFactor::createFactor<Pose>(n0, n1, kQa, kQb);
        auto long_factor = BinarySpaceFactor::createFactor<Pose>(n1, n2, kQa, kQb);
        EXPECT_TRUE(covarianceOf(*short_factor).isApprox(Functions<Pose>::getQs(0.1, kQa, kQb), 1e-9));
        EXPECT_TRUE(covarianceOf(*long_factor).isApprox(Functions<Pose>::getQs(0.3, kQa, kQb), 1e-9));

        auto t0 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto t1 = std::make_shared<NodeType>(km, 0.0, 0.05);
        auto t2 = std::make_shared<NodeType>(km, 0.0, 0.25);
        auto short_time = BinaryTimeFactor::createFactor<Pose>(t0, t1, kQa, kQb);
        auto long_time = BinaryTimeFactor::createFactor<Pose>(t1, t2, kQa, kQb);
        EXPECT_TRUE(covarianceOf(*short_time).isApprox(Functions<Pose>::getQt(0.05, kQa, kQb), 1e-9));
        EXPECT_TRUE(covarianceOf(*long_time).isApprox(Functions<Pose>::getQt(0.2, kQa, kQb), 1e-9));
    }

    TEST(WNOAFactors, SpaceJacobiansWithNonzeroRates)
    {
        KeyManager km;
        auto node0 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node1 = std::make_shared<NodeType>(km, 0.3, 0.0);
        for (bool coupling : {true, false})
        {
            auto factor = BinarySpaceFactor::createFactor<Pose>(node0, node1, kQa, kQb, coupling);
            EXPECT_EQ(factor->couplesStrainAndVelocity(), coupling);
            for (int trial = 0; trial < 5; trial++)
            {
                ValueType v0, v1;
                v0.setRandom();
                v1.setRandom();
                gtsam::Values values;
                insertValues(node0, v0, values);
                insertValues(node1, v1, values);
                EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOASpaceFactor", *factor, values, 1e-6, 1e-5)) << "coupling " << coupling;
            }
        }
    }

    TEST(WNOAFactors, TimeJacobiansWithNonzeroRates)
    {
        KeyManager km;
        auto node0 = std::make_shared<NodeType>(km, 0.0, 0.0);
        auto node1 = std::make_shared<NodeType>(km, 0.0, 0.2);
        auto factor = BinaryTimeFactor::createFactor<Pose>(node0, node1, kQa, kQb);
        for (int trial = 0; trial < 5; trial++)
        {
            ValueType v0, v1;
            v0.setRandom();
            v1.setRandom();
            gtsam::Values values;
            insertValues(node0, v0, values);
            insertValues(node1, v1, values);
            EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOATimeFactor", *factor, values, 1e-6, 1e-5));
        }
    }

    TEST(WNOAFactors, ZeroErrorNodesHaveZeroError)
    {
        ValueType v0;
        v0.setRandom();
        const double ds = 0.2, dt = 0.05;

        for (bool coupling : {true, false})
        {
            const ValueType space_next = BinarySpaceFactor::Factor<Pose>::getZeroErrorNode(v0, ds, coupling);
            BinarySpaceFactor::Factor<Pose> space(0, 1, 2, 3, 4, 5, BinarySpaceFactor::noiseModel<Pose>(ds, kQa, kQb), ds, coupling);
            EXPECT_LT(space.evaluateError(v0.pose(), v0.epsilon(), v0.varpi(), space_next.pose(), space_next.epsilon(), space_next.varpi()).norm(), 1e-12) << "coupling " << coupling;
        }
        // Without the coupling, the velocity is carried along unchanged.
        EXPECT_TRUE(BinarySpaceFactor::Factor<Pose>::getZeroErrorNode(v0, ds, false).varpi().isApprox(v0.varpi(), 1e-15));

        const ValueType time_next = BinaryTimeFactor::Factor<Pose>::getZeroErrorNode(v0, dt);
        BinaryTimeFactor::Factor<Pose> time(0, 1, 2, 3, 4, 5, BinaryTimeFactor::noiseModel<Pose>(dt, kQa, kQb), dt);
        EXPECT_LT(time.evaluateError(v0.pose(), v0.epsilon(), v0.varpi(), time_next.pose(), time_next.epsilon(), time_next.varpi()).norm(), 1e-12);
    }

    TEST(WNOAFactors, SpaceVelocityRowMatchesTheLegacyForm)
    {
        // Legacy (T_bi): e = varpi1 - varpi0 - ds * eps0^curlywedge * varpi0, with eps_bi = -eps_ib,
        // varpi_bi = -varpi_ib; its error is the negative of the T_ib one.
        ValueType v0, v1;
        v0.setRandom();
        v1.setRandom();
        const double ds = 0.2;
        BinarySpaceFactor::Factor<Pose> factor(0, 1, 2, 3, 4, 5, BinarySpaceFactor::noiseModel<Pose>(ds, kQa, kQb), ds);
        const gtsam::Vector e = factor.evaluateError(v0.pose(), v0.epsilon(), v0.varpi(), v1.pose(), v1.epsilon(), v1.varpi());
        const Vector6 eps0_bi = -v0.epsilon(), varpi0_bi = -v0.varpi(), varpi1_bi = -v1.varpi();
        const Vector6 legacy = varpi1_bi - varpi0_bi - ds * Pose::adjointMap(eps0_bi) * varpi0_bi;
        EXPECT_TRUE(e.tail<6>().isApprox(-legacy, 1e-12));
        EXPECT_TRUE(e.segment<6>(6).isApprox(v1.epsilon() - v0.epsilon()));
    }

    TEST(WNOAFactors, UncoupledSpaceVelocityRowIsTheDifference)
    {
        // TRO paper, Eq. (20): e_varpi = varpi1 - varpi0, with the same pose and strain rows
        // as the coupled (legacy) factor.
        const double ds = 0.2;
        BinarySpaceFactor::Factor<Pose> coupled(0, 1, 2, 3, 4, 5, BinarySpaceFactor::noiseModel<Pose>(ds, kQa, kQb), ds, true);
        BinarySpaceFactor::Factor<Pose> uncoupled(0, 1, 2, 3, 4, 5, BinarySpaceFactor::noiseModel<Pose>(ds, kQa, kQb), ds, false);
        for (int trial = 0; trial < 5; trial++)
        {
            ValueType v0, v1;
            v0.setRandom();
            v1.setRandom();
            const gtsam::Vector e = uncoupled.evaluateError(v0.pose(), v0.epsilon(), v0.varpi(), v1.pose(), v1.epsilon(), v1.varpi());
            const gtsam::Vector e_coupled = coupled.evaluateError(v0.pose(), v0.epsilon(), v0.varpi(), v1.pose(), v1.epsilon(), v1.varpi());
            EXPECT_TRUE(e.tail<6>().isApprox(v1.varpi() - v0.varpi(), 1e-15));
            EXPECT_TRUE(e.head<12>().isApprox(e_coupled.head<12>(), 1e-15));
            EXPECT_GT((e - e_coupled).norm(), 1e-6); // random rates: the coupling term is nonzero
        }
    }

    TEST(WNOAFactors, SpaceErrorVanishesOnACompatibleField)
    {
        // T(s,t) = Exp(t c) Exp(s a) has a constant strain eps = a and a velocity
        // varpi(s) = Ad(Exp(-s a)) c, so d(eps)/dt = 0 and d(varpi)/ds = -ad(eps) varpi.
        // The space factor's error must then be second order in ds; with the wrong sign on
        // the ad() term it would be first order.
        const Vector6 a = (Vector6() << 0.3, -0.2, 0.5, 0.1, 0.0, 1.0).finished();
        const Vector6 c = (Vector6() << -0.4, 0.7, 0.2, 0.3, -0.1, 0.2).finished();
        const double t = 0.7, s0 = 0.1;
        auto field = [&](double s)
        {
            ValueType v;
            v.pose(Pose::Expmap(t * c) * Pose::Expmap(s * a));
            v.epsilon(a);
            v.varpi(Pose::Expmap(-s * a).AdjointMap() * c);
            return v;
        };

        double previous_error = 0.0;
        for (double ds : {0.02, 0.01})
        {
            const ValueType v0 = field(s0), v1 = field(s0 + ds);
            BinarySpaceFactor::Factor<Pose> factor(0, 1, 2, 3, 4, 5, BinarySpaceFactor::noiseModel<Pose>(ds, kQa, kQb), ds);
            const gtsam::Vector e = factor.evaluateError(v0.pose(), v0.epsilon(), v0.varpi(), v1.pose(), v1.epsilon(), v1.varpi());
            EXPECT_LT(e.head<12>().norm(), 1e-12); // pose and strain rows are exact
            const double varpi_error = e.tail<6>().norm();
            EXPECT_LT(varpi_error, ds * ds * (Pose::adjointMap(a) * Pose::adjointMap(a) * v0.varpi()).norm()); // leading term is ds^2/2 of it
            if (previous_error > 0.0)
                EXPECT_NEAR(previous_error / varpi_error, 4.0, 0.2); // halving ds quarters the error
            previous_error = varpi_error;
        }
    }

    // ------------------------------------------------------------------
    // Prior options
    // ------------------------------------------------------------------

    namespace
    {
        struct PriorFixture
        {
            std::shared_ptr<StateType> state = std::make_shared<StateType>(1.0, 4);
            ValueType reference;
            std::shared_ptr<Prior<Pose>> prior;

            explicit PriorFixture(const PriorOptions &options)
            {
                reference.epsilon((Vector6() << 0, 0, 0, 0, 0, 1.0).finished());
                reference.varpi((Vector6() << 0, 0, 0.5, 0, 0.1, 0).finished());
                reference.pose(Pose(gtsam::Rot3::Ry(-M_PI / 2), gtsam::Point3(0.1, 0.2, 0.3)));
                prior = std::make_shared<Prior<Pose>>(state, reference, Eigen::Matrix<double, 18, 18>::Identity(), kQa, kQa, kQb, kQb, options);
            }
        };
    } // namespace

    TEST(WNOAFactors, PriorUsesTheGroupFactorsWhenSelected)
    {
        for (FactorType type : {FactorType::Legacy, FactorType::WNOA})
        {
            PriorOptions options;
            options.factor_type = type;
            PriorFixture f(options);
            KeyManager km;
            gtsam::Values current, fresh;

            auto row0 = f.prior->extendState(km, 0.0, current, fresh);
            EXPECT_EQ(countFactors<BinarySpaceFactor::Factor<Pose>>(row0), 3u);
            EXPECT_EQ(countFactors<BinaryTimeFactor::Factor<Pose>>(row0), 0u);
            EXPECT_EQ(countFactors<gtsam::WnoaMotionFactor<Pose>>(row0), 0u);
            EXPECT_EQ(countFactors<UnaryFactor::Factor<Pose>>(row0), 1u);
            EXPECT_EQ(countFactors<gtsam::PriorFactor<Pose>>(row0), 0u);
            EXPECT_EQ(row0.size(), 3u + 1u); // space factors + the base's unary factor
            current.insert(fresh);
            fresh.clear();

            auto row1 = f.prior->extendState(km, 0.1, current, fresh);
            EXPECT_EQ(countFactors<BinarySpaceFactor::Factor<Pose>>(row1), 3u);
            EXPECT_EQ(countFactors<BinaryTimeFactor::Factor<Pose>>(row1), 4u);
            EXPECT_EQ(countFactors<gtsam::WnoaMotionFactor<Pose>>(row1), 0u);
            EXPECT_EQ(row1.size(), 3u + 4u + 1u);

            // Only the legacy space factors couple strain and velocity.
            for (const auto &factor : row1)
                if (auto space = std::dynamic_pointer_cast<BinarySpaceFactor::Factor<Pose>>(factor))
                    EXPECT_EQ(space->couplesStrainAndVelocity(), type == FactorType::Legacy);
        }
    }

    TEST(WNOAFactors, InitialRowHasZeroSpaceError)
    {
        // The first row's initial values come from the space factors' zero-error nodes, so
        // each factor type must use the zero-error node of its own space factor.
        for (FactorType type : {FactorType::Legacy, FactorType::WNOA})
        {
            PriorOptions options;
            options.factor_type = type;
            PriorFixture f(options);
            KeyManager km;
            gtsam::Values current, fresh;
            auto row0 = f.prior->extendState(km, 0.0, current, fresh);
            for (const auto &factor : row0)
                if (auto space = std::dynamic_pointer_cast<BinarySpaceFactor::Factor<Pose>>(factor))
                    EXPECT_LT(space->unwhitenedError(fresh).norm(), 1e-12) << "type " << static_cast<int>(type);
        }
    }

    TEST(WNOAFactors, WNOAUnaryFactorUsesTheFullQ0AndAnExactJacobian)
    {
        // The WNOA and legacy factor types share the unary factor.
        PriorOptions options;
        options.factor_type = FactorType::WNOA;
        PriorFixture f(options);
        KeyManager km;
        gtsam::Values current, fresh;
        auto row0 = f.prior->extendState(km, 0.0, current, fresh);
        std::shared_ptr<UnaryFactor::Factor<Pose>> unary;
        for (const auto &factor : row0)
            if (auto u = std::dynamic_pointer_cast<UnaryFactor::Factor<Pose>>(factor))
                unary = u;
        ASSERT_NE(unary, nullptr);
        EXPECT_TRUE(covarianceOf(*unary).isApprox(Eigen::Matrix<double, 18, 18>::Identity()));

        // Far from the reference, where gtsam::PriorFactor's identity Jacobian is inexact.
        auto base = std::dynamic_pointer_cast<NodeType>(f.state->estimation_nodes.front());
        ValueType far;
        far.pose(f.reference.pose() * Pose::Expmap((Vector6() << 1.0, -0.5, 0.8, 0.2, 0.1, -0.3).finished()));
        far.epsilon((Vector6() << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6).finished());
        far.varpi((Vector6() << -0.1, 0.0, 0.2, 0.3, -0.4, 0.5).finished());
        gtsam::Values values;
        insertValues(base, far, values);
        EXPECT_TRUE(gtsam::internal::testFactorJacobians("WNOAUnaryFactor", *unary, values, 1e-6, 1e-5));
    }

    TEST(WNOAFactors, PriorDefaultsToTheUncoupledGroupFactors)
    {
        EXPECT_EQ(PriorOptions().factor_type, FactorType::WNOA);
        PriorFixture f{PriorOptions()};
        KeyManager km;
        gtsam::Values current, fresh;
        auto row0 = f.prior->extendState(km, 0.0, current, fresh);
        EXPECT_EQ(countFactors<gtsam::WnoaMotionFactor<Pose>>(row0), 0u);
        EXPECT_EQ(countFactors<BinarySpaceFactor::Factor<Pose>>(row0), 3u);
        for (const auto &factor : row0)
            if (auto space = std::dynamic_pointer_cast<BinarySpaceFactor::Factor<Pose>>(factor))
                EXPECT_FALSE(space->couplesStrainAndVelocity());
    }

    TEST(WNOAFactors, PriorUsesTheLieAlgebraFactorsWhenExperimental)
    {
        PriorOptions options;
        options.factor_type = FactorType::Experimental;
        PriorFixture f(options);
        KeyManager km;
        gtsam::Values current, fresh;
        auto row0 = f.prior->extendState(km, 0.0, current, fresh);
        EXPECT_EQ(countFactors<gtsam::WnoaMotionFactor<Pose>>(row0), 3u);
        EXPECT_EQ(countFactors<BinarySpaceFactor::Factor<Pose>>(row0), 0u);
        EXPECT_EQ(countFactors<UnaryFactor::Factor<Pose>>(row0), 0u);
        EXPECT_EQ(countFactors<gtsam::PriorFactor<Pose>>(row0), 1u);
        current.insert(fresh);
        fresh.clear();
        auto row1 = f.prior->extendState(km, 0.1, current, fresh);
        EXPECT_EQ(countFactors<gtsam::WnoaMotionFactor<Pose>>(row1), 3u + 4u);
        EXPECT_EQ(countFactors<BinaryTimeFactor::Factor<Pose>>(row1), 0u);
    }

    TEST(WNOAFactors, FixedNoiseModelSkipsTheConditionalCovariance)
    {
        // An off-node measurement: the inflated factor weighs the same residual less.
        auto errorFor = [](bool fixed)
        {
            PriorOptions options;
            options.fixed_noise_model = fixed;
            PriorFixture f(options);
            KeyManager km;
            gtsam::Values values, fresh;
            for (double t : {0.0, 0.1})
            {
                f.prior->extendState(km, t, values, fresh);
                values.insert(fresh);
                fresh.clear();
            }
            auto inner = std::make_shared<gtsam::PriorFactor<Pose>>(
                Interpolator<Pose>::InterpolatedKeys::Pose(), Pose(gtsam::Rot3(), gtsam::Point3(1, 1, 1)),
                gtsam::noiseModel::Gaussian::Covariance(Matrix6::Identity() * 1e-2));
            auto factor = f.prior->getOuterFactor(0.05, 0.4, inner);
            return factor->error(values);
        };
        const double fixed_error = errorFor(true), inflated_error = errorFor(false);
        EXPECT_GT(fixed_error, 0.0);
        EXPECT_GT(fixed_error, inflated_error);
    }

    namespace
    {
        using FactorVector = std::vector<std::shared_ptr<gtsam::NonlinearFactor>>;

        /// The factors of three consecutive rows, and every value they initialized.
        struct ExtendedRows
        {
            std::vector<FactorVector> rows;
            gtsam::Values values;
        };

        ExtendedRows extendThreeRows(const PriorOptions &options)
        {
            PriorFixture f(options);
            KeyManager km;
            ExtendedRows out;
            for (double t : {0.0, 0.1, 0.2})
            {
                gtsam::Values fresh;
                out.rows.push_back(f.prior->extendState(km, t, out.values, fresh));
                out.values.insert(fresh);
            }
            return out;
        }

        /// Unary factors of either factor type: the paper's UnaryFactor, or the three gtsam priors.
        std::size_t countUnaryFactors(const FactorVector &factors)
        {
            return countFactors<UnaryFactor::Factor<Pose>>(factors) + countFactors<gtsam::PriorFactor<Pose>>(factors) +
                   countFactors<gtsam::PriorFactor<Vector6>>(factors);
        }
    } // namespace

    TEST(WNOAFactors, EveryRowHasAUnaryFactorByDefault)
    {
        for (FactorType type : {FactorType::Experimental, FactorType::Legacy, FactorType::WNOA})
        {
            PriorOptions options;
            options.factor_type = type;
            EXPECT_FALSE(options.remove_unary);
            const std::size_t per_row = type == FactorType::Experimental ? 3u : 1u;
            const ExtendedRows extended = extendThreeRows(options);
            for (const FactorVector &row : extended.rows)
                EXPECT_EQ(countUnaryFactors(row), per_row);
        }
    }

    TEST(WNOAFactors, RemoveUnaryKeepsOnlyTheFirstRowsUnaryFactor)
    {
        for (FactorType type : {FactorType::Experimental, FactorType::Legacy, FactorType::WNOA})
        {
            PriorOptions with, without;
            with.factor_type = without.factor_type = type;
            without.remove_unary = true;
            const ExtendedRows kept = extendThreeRows(with), removed = extendThreeRows(without);
            const std::size_t per_row = type == FactorType::Experimental ? 3u : 1u;

            EXPECT_EQ(countUnaryFactors(removed.rows[0]), per_row);
            for (std::size_t r = 0; r < removed.rows.size(); r++)
            {
                if (r > 0)
                    EXPECT_EQ(countUnaryFactors(removed.rows[r]), 0u) << "row " << r;
                // Only the unary factors are dropped.
                EXPECT_EQ(removed.rows[r].size(), kept.rows[r].size() - (r > 0 ? per_row : 0u)) << "row " << r;
            }
        }
    }

    TEST(WNOAFactors, RemoveUnaryLeavesTheInitialValuesUnchanged)
    {
        for (FactorType type : {FactorType::Experimental, FactorType::Legacy, FactorType::WNOA})
        {
            PriorOptions with, without;
            with.factor_type = without.factor_type = type;
            without.remove_unary = true;
            EXPECT_TRUE(extendThreeRows(with).values.equals(extendThreeRows(without).values, 1e-12));
        }
    }

    // ------------------------------------------------------------------
    // ConfigLoader
    // ------------------------------------------------------------------

    TEST(WNOAFactors, ConfigLoaderReadsThePriorOptions)
    {
        const std::string path = writeTempConfig("wnoa_prior_options.json", R"({
            "prior": { "type": "wnoa", "wnoa": { "factor_type": "legacy", "fixed_noise_model": true, "remove_unary": true } }
        })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        const PriorOptions options = loader.loadWNOAPriorOptions();
        EXPECT_EQ(options.factor_type, FactorType::Legacy);
        EXPECT_TRUE(options.fixed_noise_model);
        EXPECT_TRUE(options.remove_unary);
        std::remove(path.c_str());
    }

    TEST(WNOAFactors, ConfigLoaderReadsEachFactorType)
    {
        const std::pair<const char *, FactorType> cases[] = {{"wnoa", FactorType::WNOA}, {"legacy", FactorType::Legacy}, {"experimental", FactorType::Experimental}};
        for (const auto &[name, type] : cases)
        {
            const std::string path = writeTempConfig("wnoa_prior_factor_type.json", std::string(R"({ "prior": { "type": "wnoa", "wnoa": { "factor_type": ")") + name + R"(" } } })");
            Utilities::ConfigLoader loader;
            ASSERT_TRUE(loader.loadConfig(path));
            EXPECT_EQ(loader.loadWNOAPriorOptions().factor_type, type) << name;
            std::remove(path.c_str());
        }
    }

    TEST(WNOAFactors, ConfigLoaderDefaultsThePriorOptions)
    {
        const std::string path = writeTempConfig("wnoa_prior_options_missing.json", R"({ "prior": { "type": "wnoa", "time_period": 0.1, "wnoa": {} } })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        const PriorOptions options = loader.loadWNOAPriorOptions();
        EXPECT_EQ(options.factor_type, FactorType::WNOA);
        EXPECT_FALSE(options.fixed_noise_model);
        EXPECT_FALSE(options.remove_unary);
        std::remove(path.c_str());
    }

    TEST(WNOAFactors, ConfigLoaderRejectsAnUnknownFactorType)
    {
        const std::string path = writeTempConfig("wnoa_prior_options_bad.json", R"({ "prior": { "type": "wnoa", "wnoa": { "factor_type": "group" } } })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        try
        {
            loader.loadWNOAPriorOptions();
            ADD_FAILURE() << "expected std::invalid_argument";
        }
        catch (const std::invalid_argument &e)
        {
            const std::string message = e.what();
            for (const char *name : {"\"wnoa\"", "\"legacy\"", "\"experimental\""})
                EXPECT_NE(message.find(name), std::string::npos) << message;
        }
        std::remove(path.c_str());
    }

    TEST(WNOAFactors, ConfigLoaderRejectsTheRenamedPaperFactorType)
    {
        const std::string path = writeTempConfig("wnoa_prior_options_paper.json", R"({ "prior": { "type": "wnoa", "wnoa": { "factor_type": "paper" } } })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        try
        {
            loader.loadWNOAPriorOptions();
            ADD_FAILURE() << "expected std::invalid_argument";
        }
        catch (const std::invalid_argument &e)
        {
            EXPECT_NE(std::string(e.what()).find("renamed to \"wnoa\""), std::string::npos) << e.what();
        }
        std::remove(path.c_str());
    }

    TEST(WNOAFactors, ConfigLoaderReadsTheDemoTimeWindow)
    {
        const std::string path = writeTempConfig("demo_time_window.json", R"({ "demo_data": { "start_time": 2.5 } })");
        Utilities::ConfigLoader loader;
        ASSERT_TRUE(loader.loadConfig(path));
        double start = 10.0, end = 20.0;
        loader.loadDemoTimeWindow(start, end);
        EXPECT_DOUBLE_EQ(start, 2.5);
        EXPECT_DOUBLE_EQ(end, 20.0); // missing entries keep the value passed in
        std::remove(path.c_str());
    }
} // namespace Spacetime::Priors::WNOA
