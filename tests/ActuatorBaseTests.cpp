#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using NodeType = Priors::WNOA::Node<Pose>;
        using StateType = Types::ContinuumRobotState<NodeType>;
        using ValueType = Priors::WNOA::Values<Pose>;
        using Keys = ValueType::InterpolatedKeys;
        using Vector6 = gtsam::Vector6;

        /// error = strain - u: the input sets the strain at every arclength it acts at.
        class StrainInputFactor : public gtsam::NoiseModelFactorN<Vector6, gtsam::Vector>
        {
        public:
            using Base = gtsam::NoiseModelFactorN<Vector6, gtsam::Vector>;
            StrainInputFactor(const gtsam::SharedNoiseModel &noise_model, gtsam::Key input_key)
                : Base(noise_model, Keys::Strain(), input_key) {}

            gtsam::Vector evaluateError(const Vector6 &strain, const gtsam::Vector &input,
                                        gtsam::OptionalMatrixType H_strain, gtsam::OptionalMatrixType H_input) const override
            {
                if (H_strain)
                    *H_strain = gtsam::Matrix66::Identity();
                if (H_input)
                    *H_input = -gtsam::Matrix66::Identity();
                return strain - input;
            }
        };

        /// A toy actuator whose 6-dim input is the strain it imposes.
        class StrainActuator : public Types::ActuatorBase
        {
        public:
            StrainActuator(pStatefulType prior, std::vector<double> arclengths, double sigma = 0.1)
                : Types::ActuatorBase(std::move(prior), std::move(arclengths), 6), m_sigma(sigma) {}

            gtsam::NoiseModelFactor::shared_ptr getFactor(const Types::MeasurementBase &, gtsam::Key input_key, double) const override
            {
                return std::make_shared<StrainInputFactor>(gtsam::noiseModel::Isotropic::Sigma(6, m_sigma), input_key);
            }

        private:
            double m_sigma;
        };

        std::shared_ptr<Priors::WNOA::Prior<Pose>> makePrior(std::shared_ptr<StateType> &state)
        {
            state = std::make_shared<StateType>(1.0, 4);
            ValueType node_reference;
            node_reference.epsilon((Vector6() << 0, 0, 0, 0, 0, 1.0).finished());
            return std::make_shared<Priors::WNOA::Prior<Pose>>(state, node_reference);
        }

        const Vector6 kInput = (Vector6() << 0.5, -0.2, 0.1, 0.0, 0.0, 1.0).finished();

        std::shared_ptr<Types::MeasurementBase> makeInput(double time, const Vector6 &value = kInput, double variance = 1e-2)
        {
            return std::make_shared<Types::MeasurementBase>(0.0, time, value, Eigen::Matrix<double, 6, 6>::Identity() * variance);
        }

        void extendPrior(Types::Model &prior, KeyManager &km, std::initializer_list<double> times, gtsam::Values &values)
        {
            for (double t : times)
            {
                gtsam::Values new_values;
                prior.extendState(km, t, values, new_values);
                values.insert(new_values);
            }
        }

        /// A sensor measuring an actuator's input directly, through the actuator's own outer factor.
        class InputSensor : public Types::SensorBase
        {
        public:
            using Types::SensorBase::SensorBase;

            gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
            {
                return std::make_shared<gtsam::PriorFactor<gtsam::Vector>>(Types::InputKeys::Input(), gtsam::Vector(measurement->value.reshaped()),
                                                                           gtsam::noiseModel::Gaussian::Covariance(measurement->covariance));
            }
        };

        bool hasKey(const gtsam::NonlinearFactor &factor, gtsam::Key key)
        {
            return std::find(factor.keys().begin(), factor.keys().end(), key) != factor.keys().end();
        }
    } // namespace

    // Its inputs are variables of the estimation problem, so it is a stateful model, whose state holds them.
    TEST(ActuatorBase, IsAStatefulModelWithAMeasurementQueue)
    {
        EXPECT_TRUE((std::derived_from<Types::ActuatorBase, Types::StatefulModel<Types::ActuatorState>>));
        EXPECT_TRUE(Types::StatefulType<Types::ActuatorBase>);
        EXPECT_TRUE((std::derived_from<Types::ActuatorBase, Types::MeasurementQueue>));

        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        StrainActuator actuator(prior, {0.5});
        ASSERT_NE(actuator.state(), nullptr);
        EXPECT_EQ(actuator.state()->input_dim, 6u);
        EXPECT_EQ(actuator.state()->input_symbol, 'u');
        EXPECT_EQ(actuator.inputDim(), 6u);
        EXPECT_EQ(actuator.stateful(), prior); // the state it acts on, not its own
    }

    TEST(ActuatorBase, RejectsANullPriorOrNoArclengths)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        EXPECT_THROW(StrainActuator(nullptr, {0.5}), std::runtime_error);
        EXPECT_THROW(StrainActuator(prior, {}), std::invalid_argument);
        struct Dimensionless : Types::ActuatorBase
        {
            using Types::ActuatorBase::ActuatorBase;
            gtsam::NoiseModelFactor::shared_ptr getFactor(const Types::MeasurementBase &, gtsam::Key, double) const override { return nullptr; }
        };
        EXPECT_THROW(Dimensionless(prior, {0.5}, 0), std::invalid_argument);
    }

    // Each due measurement becomes one input variable, timestamped at the measurement for the
    // smoother, initialized at the measured value and tied to it by its covariance.
    TEST(ActuatorBase, ExtendStateAddsOneInputPerDueMeasurement)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        StrainActuator actuator(prior, {0.5});
        actuator.addMeasurement(makeInput(0.0, kInput, 1e-2));
        actuator.addMeasurement(makeInput(0.05, 2.0 * kInput, 4e-2));
        actuator.addMeasurement(makeInput(0.2));

        KeyManager km;
        gtsam::Values values, new_values;
        const auto factors = actuator.extendState(km, 0.1, values, new_values);

        ASSERT_EQ(actuator.inputKeys().size(), 2u);
        const gtsam::Key u0 = gtsam::Symbol('u', 0), u1 = gtsam::Symbol('u', 1);
        EXPECT_EQ(actuator.inputKeys()[0], std::make_pair(0.0, u0));
        EXPECT_EQ(actuator.inputKeys()[1], std::make_pair(0.05, u1));
        ASSERT_EQ(actuator.state()->inputs().size(), 2u); // one input node each
        EXPECT_EQ(actuator.state()->inputs()[1]->inputKey(), u1);
        ASSERT_TRUE(new_values.exists(u0) && new_values.exists(u1));
        EXPECT_TRUE(new_values.at<gtsam::Vector>(u0).isApprox(kInput));
        EXPECT_TRUE(new_values.at<gtsam::Vector>(u1).isApprox(2.0 * kInput));

        const auto timestamps = km.drainPending();
        EXPECT_DOUBLE_EQ(timestamps.at(u0), 0.0);
        EXPECT_DOUBLE_EQ(timestamps.at(u1), 0.05);

        ASSERT_EQ(factors.size(), 2u);
        for (size_t i = 0; i < 2; i++)
        {
            auto prior_factor = std::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Vector>>(factors[i]);
            ASSERT_NE(prior_factor, nullptr);
            EXPECT_EQ(prior_factor->key(), i == 0 ? u0 : u1);
            auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(prior_factor->noiseModel());
            ASSERT_NE(gaussian, nullptr);
            EXPECT_TRUE(gaussian->covariance().isApprox(Eigen::Matrix<double, 6, 6>::Identity() * (i == 0 ? 1e-2 : 4e-2)));
        }
        EXPECT_EQ(actuator.size(), 1u); // t = 0.2 is not due yet
    }

    TEST(ActuatorBase, RejectsMalformedMeasurements)
    {
        std::shared_ptr<StateType> state;
        StrainActuator actuator(makePrior(state), {0.5});
        KeyManager km;
        gtsam::Values values, new_values;
        // A covariance of the wrong size, then a value of the wrong size.
        actuator.addMeasurement(std::make_shared<Types::MeasurementBase>(0.0, 0.0, kInput, Eigen::Matrix3d::Identity()));
        EXPECT_THROW(actuator.extendState(km, 0.1, values, new_values), std::invalid_argument);
        actuator.addMeasurement(std::make_shared<Types::MeasurementBase>(0.0, 0.0, Eigen::Vector3d::Ones(), Eigen::Matrix3d::Identity()));
        EXPECT_THROW(actuator.extendState(km, 0.1, values, new_values), std::invalid_argument);
        EXPECT_TRUE(new_values.empty());
    }

    // One factor per arclength per input, on the bordering nodes and the shared input.
    TEST(ActuatorBase, GetNewFactorsAddsOneFactorPerArclengthOnTheSharedInput)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        StrainActuator actuator(prior, {0.2, 0.5, 0.8});
        actuator.addMeasurement(makeInput(0.05));

        KeyManager km;
        gtsam::Values values;
        extendPrior(*prior, km, {0.0, 0.1}, values);
        gtsam::Values new_values;
        actuator.extendState(km, 0.1, values, new_values);
        const auto factors = actuator.getNewFactors(0.1, new_values);

        ASSERT_EQ(factors.size(), 3u);
        const gtsam::Key u0 = gtsam::Symbol('u', 0);
        for (const auto &factor : factors)
        {
            ASSERT_NE(factor, nullptr);
            EXPECT_EQ(factor->keys().size(), 13u); // four corners x (pose, strain, velocity), and the input
            EXPECT_TRUE(hasKey(*factor, u0));
        }
        EXPECT_TRUE(actuator.getNewFactors(0.1, new_values).empty()); // each input's factors are added once
    }

    TEST(ActuatorBase, InterpolatedFactorJacobiansMatchNumerical)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        StrainActuator actuator(prior, {0.5});

        KeyManager km;
        gtsam::Values values;
        extendPrior(*prior, km, {0.0, 0.1}, values);
        int i = 0;
        for (const gtsam::Key key : values.keys())
            if (gtsam::Symbol(key).chr() == 's')
                values.update(key, Vector6(values.at<Vector6>(key) + (Vector6() << 0.05, -0.04, 0.03, 0.02, -0.01, 0.01).finished() * (1.0 + 0.1 * i++)));
        const gtsam::Key input_key = gtsam::Symbol('u', 0);
        values.insert(input_key, gtsam::Vector(kInput));

        using InterpolatorType = Priors::WNOA::Interpolator<Pose>;
        auto node00 = state->getClosestShiftedNode(0.5, 0.05, -1, -1);
        ASSERT_NE(node00, nullptr);
        auto node01 = state->getNextTemporalNode(node00);
        auto node10 = state->getNextSpatialNode(node00);
        auto node11 = node01 ? state->getNextSpatialNode(node01) : nullptr;
        ASSERT_TRUE(node00 && node01 && node10 && node11);
        auto interpolator = InterpolatorType(Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity(),
                                             Eigen::Matrix<double, 6, 6>::Identity(), Eigen::Matrix<double, 6, 6>::Identity());
        Priors::InterpolationFactor<InterpolatorType> factor(actuator.getFactor(*makeInput(0.05), input_key, 0.5),
                                                             interpolator, 0.05, 0.5, node00, node01, node10, node11, true);
        // The pass-through input is one of the factor's own keys.
        EXPECT_TRUE(hasKey(factor, input_key));
        EXPECT_EQ(factor.keys().size(), 13u);
        // The pose-key columns carry the interpolator's small-angle
        // approximation error, so each block is checked relative to its size.
        auto analytic = std::dynamic_pointer_cast<gtsam::JacobianFactor>(factor.linearize(values));
        ASSERT_NE(analytic, nullptr);
        const gtsam::JacobianFactor numerical = gtsam::linearizeNumerically(factor, values, 1e-6);
        ASSERT_EQ(analytic->size(), numerical.size());
        for (size_t k = 0; k < analytic->size(); k++)
        {
            ASSERT_EQ(analytic->keys()[k], numerical.keys()[k]);
            const gtsam::Matrix A = analytic->getA(analytic->begin() + k);
            const gtsam::Matrix N = numerical.getA(numerical.begin() + k);
            EXPECT_LT((A - N).cwiseAbs().maxCoeff(), 1e-3 * std::max(1.0, N.cwiseAbs().maxCoeff()))
                << "key " << gtsam::DefaultKeyFormatter(analytic->keys()[k]);
        }
        EXPECT_TRUE(analytic->getb().isApprox(numerical.getb(), 1e-9));
        // The input's block is exact: -I, whitened.
        const gtsam::Matrix A_input = analytic->getA(analytic->find(input_key));
        const gtsam::Matrix N_input = numerical.getA(numerical.find(input_key));
        EXPECT_LT((A_input - N_input).cwiseAbs().maxCoeff(), 1e-6);
    }

    // As a stateful model, the actuator resolves its input at an arbitrary time to the nearest
    // input: queries read it, and outer factors put an inner factor's input key on it.
    TEST(ActuatorBase, QueriesAndOuterFactorsResolveTheNearestInput)
    {
        std::shared_ptr<StateType> state;
        StrainActuator actuator(makePrior(state), {0.5});
        actuator.addMeasurement(makeInput(0.0, kInput));
        actuator.addMeasurement(makeInput(0.1, 2.0 * kInput));
        KeyManager km;
        gtsam::Values values;
        actuator.extendState(km, 0.1, gtsam::Values(), values);
        const gtsam::Key u0 = gtsam::Symbol('u', 0), u1 = gtsam::Symbol('u', 1);

        const auto at = [&](double time)
        { return std::dynamic_pointer_cast<Types::InputValues>(actuator.query(time, 0.0, values)); };
        ASSERT_NE(at(0.02), nullptr);
        EXPECT_TRUE(at(0.02)->input().isApprox(kInput));
        EXPECT_TRUE(at(0.08)->input().isApprox(2.0 * kInput));
        EXPECT_EQ(actuator.query(0.2, 0.0, values), nullptr); // after the last input

        auto inner = std::make_shared<gtsam::PriorFactor<gtsam::Vector>>(Types::InputKeys::Input(), gtsam::Vector(kInput),
                                                                         gtsam::noiseModel::Isotropic::Sigma(6, 1.0));
        const auto factor = actuator.getOuterFactor(0.08, 0.0, inner);
        ASSERT_NE(factor, nullptr);
        EXPECT_EQ(factor->keys(), gtsam::KeyVector{u1});
        EXPECT_NEAR(factor->error(values), 0.5 * kInput.squaredNorm(), 1e-12); // |2u - u|^2 / 2
        EXPECT_EQ(actuator.getOuterFactor(0.02, 0.0, inner)->keys(), gtsam::KeyVector{u0});
    }

    // A sensor can measure the actuator's input directly: a precise one pulls the estimate to it.
    TEST(ActuatorBase, SensorsCanMeasureItsInputs)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        auto actuator = std::make_shared<StrainActuator>(prior, std::vector<double>{0.5}, 1.0);
        auto sensor = std::make_shared<InputSensor>(actuator);
        for (int k = 0; k <= 2; k++)
        {
            actuator->addMeasurement(makeInput(0.1 * k, kInput, 1.0));
            sensor->addMeasurement(makeInput(0.1 * k, 2.0 * kInput, 1e-6));
        }

        Estimator estimator(Estimator::BATCH_LAG);
        estimator.addPrior(prior);
        estimator.addModel(actuator);
        estimator.addSensor(sensor);
        for (int k = 0; k <= 2; k++)
            estimator.estimate(0.1 * k);
        estimator.finalize();

        const gtsam::Key latest = actuator->inputKeys().back().second;
        EXPECT_TRUE(estimator.getCurrentValues().at<gtsam::Vector>(latest).isApprox(2.0 * kInput, 1e-3))
            << estimator.getCurrentValues().at<gtsam::Vector>(latest).transpose();
    }

    TEST(ActuatorBase, FutureMeasurementsWaitForTheirTime)
    {
        std::shared_ptr<StateType> state;
        StrainActuator actuator(makePrior(state), {0.5});
        actuator.addMeasurement(makeInput(0.3));
        KeyManager km;
        gtsam::Values values, new_values;
        EXPECT_TRUE(actuator.extendState(km, 0.1, values, new_values).empty());
        EXPECT_TRUE(new_values.empty());
        EXPECT_EQ(actuator.extendState(km, 0.3, values, new_values).size(), 1u);
    }

    // A measurement before the estimated state's first node cannot be attached: its strain
    // factors are dropped, and its input keeps only its measurement factor.
    TEST(ActuatorBase, MeasurementsOutsideTheStateAreDropped)
    {
        std::shared_ptr<StateType> state;
        auto prior = makePrior(state);
        StrainActuator actuator(prior, {0.5});
        actuator.addMeasurement(makeInput(-1.0));

        KeyManager km;
        gtsam::Values values;
        extendPrior(*prior, km, {0.0}, values);
        gtsam::Values new_values;
        EXPECT_EQ(actuator.extendState(km, 0.0, values, new_values).size(), 1u);
        std::vector<std::shared_ptr<gtsam::NonlinearFactor>> factors;
        EXPECT_NO_THROW(factors = actuator.getNewFactors(0.0, new_values));
        EXPECT_TRUE(factors.empty());
    }

    // With a near-exact input, every arclength's strain follows it; the strain factors in turn
    // inform the input, so its posterior is tighter than its measurement.
    TEST(ActuatorBase, EstimatorEstimatesTheSharedInput)
    {
        for (const double lag : {0.3, Estimator::BATCH_LAG})
        {
            std::shared_ptr<StateType> state;
            auto prior = makePrior(state);
            auto actuator = std::make_shared<StrainActuator>(prior, std::vector<double>{0.2, 0.5, 0.8}, 1e-2);

            Estimator estimator(lag);
            estimator.addModel(prior);
            estimator.addModel(actuator);
            for (int k = 0; k <= 10; k++)
                actuator->addMeasurement(makeInput(0.1 * k, kInput, 1e-2));
            for (int k = 0; k <= 10; k++)
                estimator.estimate(0.1 * k);
            estimator.finalize();

            const gtsam::Values values = estimator.getCurrentValues();
            const gtsam::Key latest = actuator->inputKeys().back().second;
            ASSERT_TRUE(values.exists(latest)) << "lag " << lag;
            EXPECT_TRUE(values.at<gtsam::Vector>(latest).isApprox(kInput, 1e-2)) << values.at<gtsam::Vector>(latest).transpose();
            auto queried = std::dynamic_pointer_cast<ValueType>(prior->query(1.0, 0.5, values));
            ASSERT_NE(queried, nullptr);
            EXPECT_TRUE(queried->epsilon().isApprox(kInput, 5e-2)) << "lag " << lag << ": " << queried->epsilon().transpose();

            const gtsam::Matrix covariance = estimator.covariance(latest);
            EXPECT_LT(covariance.trace(), 6 * 1e-2) << "lag " << lag;

            if (lag > 0) // the sliding window marginalized the oldest inputs
                EXPECT_FALSE(values.exists(actuator->inputKeys().front().second));
        }
    }
} // namespace Spacetime
