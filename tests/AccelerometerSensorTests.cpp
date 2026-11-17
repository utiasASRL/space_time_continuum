#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtest/gtest.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/joint_models/EndEffectorPrior.hpp>
#include <spacetime/estimator/prior_models/TwoSidedInterpolationFactor.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/sensor_models/AccelerometerSensor.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>

#include "NumericalDifferentiation.hpp"

namespace Spacetime
{
    namespace
    {
        using Pose = gtsam::Pose3;
        using Rot = gtsam::Rot3;
        using Vector3 = gtsam::Vector3;
        using Matrix3 = gtsam::Matrix3;
        using Vector6 = Eigen::Matrix<double, 6, 1>;
        using Preintegration = Sensors::AccelerometerPreintegration;
        using Options = Sensors::AccelerometerOptions;

        using WNOAPrior = Priors::WNOA::Prior<Pose>;
        using WNOAState = Types::ContinuumRobotState<Priors::WNOA::Node<Pose>>;
        using WNOAAccelerometer = Sensors::AccelerometerSensor<WNOAPrior>;
        using WNOAInterpolator = WNOAPrior::InterpolatorType;
        using EEPrior = Joints::EndEffectorPrior<Pose>;
        using EEState = Types::ContinuumRobotState<EEPrior::NodeType>;
        using EEAccelerometer = Sensors::AccelerometerSensor<EEPrior>;

        const gtsam::Key X0 = gtsam::Symbol('x', 0), V0 = gtsam::Symbol('w', 0);
        const gtsam::Key X1 = gtsam::Symbol('x', 1), V1 = gtsam::Symbol('w', 1);
        const gtsam::Key B0 = gtsam::Symbol('b', 0), B1 = gtsam::Symbol('b', 1);

        const Matrix3 kAccelCovariance = Vector3(1e-4, 2e-4, 3e-4).asDiagonal();

        /// Linear bias weights between B0 at t0 and B1 at t1.
        Preintegration::BiasWeightsFn linearBias(double t0, double t1)
        {
            return [t0, t1](double t)
            {
                const double tau = (t - t0) / (t1 - t0);
                return Preintegration::BiasWeights{{B0, 1.0 - tau}, {B1, tau}};
            };
        }

        /// n samples held over [t0 + i h, t0 + (i+1) h), each reading `reading` at its hold
        /// interval's midpoint (the time the preintegration evaluates it at).
        Preintegration preintegrate(double t0, double t1, int n, const std::function<Vector3(double)> &reading,
                                    Preintegration::BiasWeightsFn bias = nullptr)
        {
            Preintegration pim(t0, t1, std::move(bias));
            const double h = (t1 - t0) / n;
            for (int i = 0; i < n; i++)
                pim.addSample(t0 + i * h, reading(t0 + (i + 0.5) * h), kAccelCovariance);
            pim.close();
            return pim;
        }

        Vector6 twist(const Vector3 &omega, const Vector3 &nu) { return (Vector6() << omega, nu).finished(); }

        gtsam::Values makeValues(const Pose &T0, const Vector6 &varpi0, const Pose &T1, const Vector6 &varpi1,
                                 const Vector3 &b0 = Vector3::Zero(), const Vector3 &b1 = Vector3::Zero())
        {
            gtsam::Values values;
            values.insert(X0, T0);
            values.insert(V0, varpi0);
            values.insert(X1, T1);
            values.insert(V1, varpi1);
            values.insert(B0, b0);
            values.insert(B1, b1);
            return values;
        }

        /// The residual, sample by sample, straight from the measurement model (independent of
        /// the factor's preintegrated sums).
        Vector3 referenceError(const Preintegration &pim, const Options &options, const gtsam::Values &values)
        {
            const Pose T0 = values.at<Pose>(X0), T1 = values.at<Pose>(X1);
            const Vector6 varpi0 = values.at<Vector6>(V0), varpi1 = values.at<Vector6>(V1);
            const Rot R0 = T0.rotation(), R1 = T1.rotation();
            const Vector3 w0 = varpi0.head<3>(), w1 = varpi1.head<3>();
            const double dt = pim.deltaT();
            const Vector3 alpha = (w1 - w0) / dt;
            const Vector3 r = options.offset.translation();
            const Rot R_bs = options.offset.rotation();

            Vector3 sum = Vector3::Zero();
            for (const auto &sample : pim.samples())
            {
                Vector3 b = Vector3::Zero();
                for (const auto &[key, weight] : sample.bias_weights)
                    b += weight * values.at<Vector3>(key);
                const Vector3 wj = (1.0 - sample.tau) * w0 + sample.tau * w1;
                const Vector3 f = R_bs * Vector3(sample.measured - b) - (alpha.cross(r) + wj.cross(wj.cross(r)));
                const Rot dR = options.use_slerp ? Rot::Expmap(sample.tau * Rot::Logmap(R0.between(R1))) : Rot();
                sum += dR * f * sample.dt;
            }
            return R0.unrotate(R1.rotate(Vector3(varpi1.tail<3>()))) - Vector3(varpi0.tail<3>()) - R0.unrotate(options.gravity) * dt - sum;
        }

        /// Sets the translational velocity of V1 so that the reference residual vanishes.
        void solveForZeroError(const Preintegration &pim, const Options &options, gtsam::Values &values)
        {
            Vector6 varpi1 = values.at<Vector6>(V1);
            varpi1.tail<3>().setZero();
            values.update(V1, varpi1);
            const Vector3 c = referenceError(pim, options, values);
            const Rot R0 = values.at<Pose>(X0).rotation(), R1 = values.at<Pose>(X1).rotation();
            varpi1.tail<3>() = -(R1.inverse() * R0).matrix() * c;
            values.update(V1, varpi1);
        }

        Sensors::AccelerometerFactor makeFactor(const Preintegration &pim, const Options &options)
        {
            return Sensors::AccelerometerFactor(X0, V0, X1, V1, pim, options);
        }

        Options withOffset(Options options, const Pose &offset)
        {
            options.offset = offset;
            return options;
        }

        Options withSlerp(Options options, bool use_slerp)
        {
            options.use_slerp = use_slerp;
            return options;
        }

        void expectJacobiansMatchNumerical(const gtsam::NoiseModelFactor &factor, const gtsam::Values &values, double tolerance = 1e-6)
        {
            std::vector<gtsam::Matrix> analytic(factor.size()); // gtsam indexes H without resizing it
            factor.unwhitenedError(values, analytic);
            const std::vector<gtsam::Matrix> numerical = Testing::numericalFactorJacobians(factor, values);
            ASSERT_EQ(analytic.size(), numerical.size());
            for (size_t i = 0; i < analytic.size(); i++)
                EXPECT_TRUE(gtsam::assert_equal(numerical[i], analytic[i], tolerance)) << "key " << gtsam::Symbol(factor.keys()[i]);
        }

        /// A generic, rotating and accelerating state over [0, 0.1].
        gtsam::Values genericValues()
        {
            const Pose T0(Rot::RzRyRx(0.2, -0.3, 0.4), gtsam::Point3(0.1, -0.2, 0.3));
            const Pose T1 = T0 * Pose(Rot::RzRyRx(0.05, 0.08, -0.04), gtsam::Point3(0.02, 0.01, -0.03));
            return makeValues(T0, twist(Vector3(0.5, -0.8, 0.4), Vector3(0.2, 0.1, -0.3)),
                              T1, twist(Vector3(0.9, -0.2, 0.1), Vector3(0.3, -0.1, -0.2)),
                              Vector3(0.05, -0.02, 0.03), Vector3(0.06, -0.01, 0.02));
        }

        const Pose kOffset(Rot::RzRyRx(0.3, -0.5, 1.2), gtsam::Point3(0.05, -0.1, 0.08));

        Vector3 genericReading(double t) { return Vector3(0.3 + t, -0.2 + 2.0 * t * t, 9.7 - t); }

        // ---- Sensor rig -------------------------------------------------------------------

        std::shared_ptr<WNOAPrior> makeWNOAPrior(std::shared_ptr<WNOAState> &state)
        {
            state = std::make_shared<WNOAState>(1.0, 4); // nodes at s = 0, 1/3, 2/3, 1
            Priors::WNOA::Values<Pose> reference;
            reference.epsilon((Vector6() << 0, 0, 0, 1.0, 0, 0).finished());
            return std::make_shared<WNOAPrior>(state, reference);
        }

        std::shared_ptr<Types::MeasurementBase> reading(double s, double t, const Vector3 &value,
                                                        const Eigen::MatrixXi &mask = Eigen::MatrixXi::Ones(3, 1))
        {
            return std::make_shared<Types::MeasurementBase>(s, t, value, kAccelCovariance, mask);
        }

        /// Adds samples every 0.01 s over [t0, t1] (both included) at arclength s.
        void addSamples(Types::SensorBase &sensor, double s, double t0, double t1)
        {
            for (int i = 0; t0 + 0.01 * i <= t1 + 1e-9; i++)
                sensor.addMeasurement(reading(s, t0 + 0.01 * i, genericReading(t0 + 0.01 * i)));
        }

        /// One Estimator::estimate's phases at `time`, by hand: every model's extendState,
        /// then every sensor's getNewFactors, whose factors (concatenated) are returned.
        Types::Model::Factors step(const std::vector<Types::Model *> &models, const std::vector<Types::Model *> &sensors,
                                   KeyManager &km, gtsam::Values &values, double time)
        {
            gtsam::Values new_values;
            for (auto *model : models)
                model->extendState(km, time, values, new_values);
            for (auto *sensor : sensors)
                sensor->extendState(km, time, values, new_values);
            values.insert(new_values);
            new_values.clear();
            Types::Model::Factors factors;
            for (auto *sensor : sensors)
            {
                auto new_factors = sensor->getNewFactors(time, new_values);
                factors.insert(factors.end(), new_factors.begin(), new_factors.end());
            }
            return factors;
        }

        /// Distinct, non-trivial perturbations of every WNOA variable.
        void perturb(gtsam::Values &values)
        {
            int i = 0;
            for (const gtsam::Key key : values.keys())
            {
                const double scale = 1.0 + 0.1 * i++;
                const Vector6 d = (Vector6() << 0.01, -0.02, 0.015, 0.01, -0.01, 0.005).finished() * scale;
                const unsigned char chr = gtsam::Symbol(key).chr();
                if (chr == 'p')
                    values.update(key, values.at<Pose>(key).retract(d));
                else if (chr == 's' || chr == 'v')
                    values.update(key, Vector6(values.at<Vector6>(key) + 5.0 * d.reverse()));
                else if (chr == 'b')
                    values.update(key, Vector3(values.at<Vector3>(key) + d.head<3>()));
            }
        }

        Options noBias()
        {
            Options options;
            options.estimate_bias = false;
            return options;
        }

        std::string writeFile(const std::filesystem::path &path, const std::string &contents)
        {
            std::ofstream(path) << contents;
            return path.string();
        }
    } // namespace

    // ---- Preintegration -----------------------------------------------------------------

    TEST(AccelerometerPreintegration, HoldsEachSampleUntilTheNextAndClipsToTheInterval)
    {
        Preintegration pim(0.1, 0.2);
        const Vector3 a0(9.0, 9.0, 9.0), a1(1.0, 2.0, 3.0), a2(-1.0, 0.5, 2.0), a3(0.3, -0.4, 1.0);
        pim.addSample(0.05, a0, kAccelCovariance); // superseded before the interval starts
        pim.addSample(0.08, a1, kAccelCovariance); // carried over: holds from t_start
        pim.addSample(0.13, a2, kAccelCovariance);
        pim.addSample(0.17, a3, kAccelCovariance);
        EXPECT_FALSE(pim.complete()); // the last sample is still open
        pim.close();
        ASSERT_TRUE(pim.complete());

        ASSERT_EQ(pim.samples().size(), 3u);
        const std::vector<double> dts = {0.03, 0.04, 0.03}, taus = {0.15, 0.5, 0.85};
        const std::vector<Vector3> readings = {a1, a2, a3};
        Vector3 sum = Vector3::Zero();
        double w0 = 0, w01 = 0, w1 = 0;
        for (size_t j = 0; j < 3; j++)
        {
            EXPECT_NEAR(pim.samples()[j].dt, dts[j], 1e-12);
            EXPECT_NEAR(pim.samples()[j].tau, taus[j], 1e-12);
            EXPECT_TRUE(gtsam::assert_equal(readings[j], pim.samples()[j].measured));
            sum += dts[j] * readings[j];
            w0 += dts[j] * (1 - taus[j]) * (1 - taus[j]);
            w01 += dts[j] * taus[j] * (1 - taus[j]);
            w1 += dts[j] * taus[j] * taus[j];
        }
        EXPECT_NEAR(pim.deltaT(), 0.1, 1e-12);
        EXPECT_NEAR(pim.coveredTime(), 0.1, 1e-12);
        EXPECT_TRUE(gtsam::assert_equal(sum, pim.sumMeasured(), 1e-12));
        EXPECT_NEAR(pim.w0(), w0, 1e-12);
        EXPECT_NEAR(pim.w01(), w01, 1e-12);
        EXPECT_NEAR(pim.w1(), w1, 1e-12);
        EXPECT_TRUE(pim.biasWeights().empty());
    }

    TEST(AccelerometerPreintegration, AnUncoveredStartIsIncomplete)
    {
        Preintegration late(0.1, 0.2);
        late.addSample(0.13, Vector3::Zero(), kAccelCovariance);
        late.close();
        EXPECT_FALSE(late.complete());
        EXPECT_NEAR(late.coveredTime(), 0.07, 1e-12);

        Preintegration empty(0.1, 0.2);
        empty.close();
        EXPECT_FALSE(empty.complete());
    }

    TEST(AccelerometerPreintegration, RejectsSamplesOutOfOrderOrPastTheEnd)
    {
        Preintegration pim(0.1, 0.2);
        pim.addSample(0.15, Vector3::Zero(), kAccelCovariance);
        EXPECT_THROW(pim.addSample(0.12, Vector3::Zero(), kAccelCovariance), std::invalid_argument);
        EXPECT_THROW(pim.addSample(0.2, Vector3::Zero(), kAccelCovariance), std::invalid_argument);
        EXPECT_THROW(Preintegration(0.2, 0.1), std::invalid_argument);
    }

    TEST(AccelerometerPreintegration, CovarianceIsTheSumOfHeldSampleCovariances)
    {
        Preintegration pim(0.0, 0.1);
        pim.addSample(0.0, Vector3::Zero(), kAccelCovariance);
        pim.addSample(0.04, Vector3::Zero(), 2.0 * kAccelCovariance);
        pim.close();
        const Matrix3 expected = 0.04 * 0.04 * kAccelCovariance + 0.06 * 0.06 * 2.0 * kAccelCovariance;
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(expected), gtsam::Matrix(pim.covariance()), 1e-15));
    }

    TEST(AccelerometerPreintegration, BiasWeightsAreAggregatedPerKey)
    {
        const Preintegration pim = preintegrate(0.1, 0.2, 7, genericReading, linearBias(0.1, 0.2));
        ASSERT_EQ(pim.biasKeys(), (gtsam::KeyVector{B0, B1}));
        double w_b0 = 0, w_b1 = 0;
        for (const auto &sample : pim.samples())
        {
            ASSERT_EQ(sample.bias_weights.size(), 2u);
            w_b0 += sample.dt * (1 - sample.tau);
            w_b1 += sample.dt * sample.tau;
        }
        ASSERT_EQ(pim.biasWeights().size(), 2u);
        EXPECT_EQ(pim.biasWeights()[0].first, B0);
        EXPECT_NEAR(pim.biasWeights()[0].second, w_b0, 1e-12);
        EXPECT_EQ(pim.biasWeights()[1].first, B1);
        EXPECT_NEAR(pim.biasWeights()[1].second, w_b1, 1e-12);
        EXPECT_NEAR(w_b0 + w_b1, pim.coveredTime(), 1e-12);
    }

    // ---- Factor -------------------------------------------------------------------------

    TEST(AccelerometerFactor, IncompletePreintegrationThrows)
    {
        Preintegration pim(0.0, 0.1);
        pim.addSample(0.05, Vector3::Zero(), kAccelCovariance);
        pim.close();
        EXPECT_THROW(makeFactor(pim, Options()), std::invalid_argument);
    }

    TEST(AccelerometerFactor, MatchesTheSampleBySampleModel)
    {
        gtsam::Values values = genericValues();
        for (const bool slerp : {false, true})
            for (const Pose &offset : {Pose(), kOffset})
                for (const bool bias : {false, true})
                {
                    const Options options = withOffset(withSlerp(Options(), slerp), offset);
                    const Preintegration pim = preintegrate(0.0, 0.1, 9, genericReading, bias ? linearBias(0.0, 0.1) : nullptr);
                    EXPECT_TRUE(gtsam::assert_equal(referenceError(pim, options, values), Vector3(makeFactor(pim, options).unwhitenedError(values)), 1e-10))
                        << "slerp " << slerp << ", offset " << (offset.translation().norm() > 0) << ", bias " << bias;

                    gtsam::Values consistent = values;
                    solveForZeroError(pim, options, consistent);
                    EXPECT_LT(makeFactor(pim, options).unwhitenedError(consistent).norm(), 1e-10);
                }
    }

    TEST(AccelerometerFactor, ErrorSignAndScale)
    {
        // At rest and level, an accelerometer reads +g up; an extra 0.1 m/s^2 along x is
        // unexplained, so the velocity increment it predicts (0.1 dt along x) is the error's negative.
        const double dt = 0.1;
        const Preintegration pim = preintegrate(0.0, dt, 4, [](double) { return Vector3(0.1, 0.0, 9.81); });
        const gtsam::Values values = makeValues(Pose(), Vector6::Zero(), Pose(), Vector6::Zero());
        EXPECT_TRUE(gtsam::assert_equal(Vector3(-0.1 * dt, 0.0, 0.0), Vector3(makeFactor(pim, Options()).unwhitenedError(values)), 1e-12));
    }

    TEST(AccelerometerFactor, ConstantWorldAccelerationGivesZeroError)
    {
        const Rot R = Rot::RzRyRx(0.2, 0.1, -0.3);
        const Vector3 a_world(0.5, -0.3, 0.2), v_world(0.1, 0.2, -0.1);
        const double dt = 0.1;
        const Options options;
        const Vector3 specific_force = R.unrotate(a_world - options.gravity);
        const Preintegration pim = preintegrate(0.0, dt, 5, [&](double) { return specific_force; });
        const gtsam::Values values = makeValues(Pose(R, gtsam::Point3()), twist(Vector3::Zero(), R.unrotate(v_world)),
                                                Pose(R, gtsam::Point3()), twist(Vector3::Zero(), R.unrotate(v_world + a_world * dt)));
        EXPECT_LT(makeFactor(pim, options).unwhitenedError(values).norm(), 1e-12);
    }

    TEST(AccelerometerFactor, TheLeverArmAccountsForCentripetalAndAngularAcceleration)
    {
        // A body spinning up about z around its fixed origin, with gravity off: an offset
        // accelerometer reads only alpha x r + w x (w x r), which the factor must explain.
        const double dt = 0.1;
        Options options = withOffset(Options(), Pose(Rot::RzRyRx(0.2, -0.4, 0.9), gtsam::Point3(0.1, 0.05, -0.02)));
        options.gravity.setZero();
        const Vector3 w0(0.0, 0.0, 2.0), w1(0.0, 0.0, 3.0), r = options.offset.translation();
        const Vector3 alpha = (w1 - w0) / dt;
        const Preintegration pim = preintegrate(0.0, dt, 6, [&](double t)
                                                {
                                                    const Vector3 w = w0 + (t / dt) * (w1 - w0);
                                                    return options.offset.rotation().unrotate(Vector3(alpha.cross(r) + w.cross(w.cross(r))));
                                                });
        const gtsam::Values values = makeValues(Pose(), twist(w0, Vector3::Zero()), Pose(Rot::Rz(0.25), gtsam::Point3()), twist(w1, Vector3::Zero()));
        for (const bool slerp : {false, true})
            EXPECT_LT(makeFactor(pim, withSlerp(options, slerp)).unwhitenedError(values).norm(), 1e-12) << "slerp " << slerp;
    }

    TEST(AccelerometerFactor, SlerpFollowsTheRotationWithinTheInterval)
    {
        // Spinning about x at a constant rate around a fixed origin: gravity sweeps through the
        // body frame. SLERP between the two node rotations is the true rotation; holding R_k is not.
        const double dt = 0.2, rate = 3.0;
        const Options options;
        const Preintegration pim = preintegrate(0.0, dt, 8, [&](double t) { return Rot::Rx(rate * t).unrotate(-options.gravity); });
        const Vector6 varpi = twist(Vector3(rate, 0.0, 0.0), Vector3::Zero());
        const gtsam::Values values = makeValues(Pose(), varpi, Pose(Rot::Rx(rate * dt), gtsam::Point3()), varpi);
        EXPECT_LT(makeFactor(pim, withSlerp(options, true)).unwhitenedError(values).norm(), 1e-12);
        EXPECT_GT(makeFactor(pim, withSlerp(options, false)).unwhitenedError(values).norm(), 1e-2);
    }

    TEST(AccelerometerFactor, ABiasIsRemovedFromTheReadings)
    {
        const Pose T(Rot::RzRyRx(0.4, -0.6, 0.3), gtsam::Point3());
        const Vector3 bias(0.2, -0.1, 0.05);
        const double dt = 0.1;
        for (const bool slerp : {false, true})
        {
            const Options options = withOffset(withSlerp(Options(), slerp), Pose(Rot::RzRyRx(0.5, 0.2, -0.3), gtsam::Point3()));
            const Vector3 specific_force = options.offset.rotation().unrotate(T.rotation().unrotate(-options.gravity));
            const Preintegration pim = preintegrate(0.0, dt, 5, [&](double) { return Vector3(specific_force + bias); }, linearBias(0.0, dt));
            const auto factor = makeFactor(pim, options);
            EXPECT_LT(factor.unwhitenedError(makeValues(T, Vector6::Zero(), T, Vector6::Zero(), bias, bias)).norm(), 1e-12);
            // An unestimated bias shows up as the velocity increment it fakes.
            EXPECT_TRUE(gtsam::assert_equal(Vector3(-dt * (options.offset.rotation() * bias)),
                                            Vector3(factor.unwhitenedError(makeValues(T, Vector6::Zero(), T, Vector6::Zero()))), 1e-12));
        }
    }

    TEST(AccelerometerFactor, JacobiansMatchNumerical)
    {
        const gtsam::Values values = genericValues();
        for (const bool slerp : {false, true})
            for (const Pose &offset : {Pose(), kOffset})
                for (const bool bias : {false, true})
                {
                    SCOPED_TRACE(testing::Message() << "slerp " << slerp << ", offset " << (offset.translation().norm() > 0) << ", bias " << bias);
                    const Preintegration pim = preintegrate(0.0, 0.1, 9, genericReading, bias ? linearBias(0.0, 0.1) : nullptr);
                    expectJacobiansMatchNumerical(makeFactor(pim, withOffset(withSlerp(Options(), slerp), offset)), values);
                }
    }

    TEST(AccelerometerFactor, NoiseIsThePreintegratedCovarianceInTheBodyFrame)
    {
        const Preintegration pim = preintegrate(0.0, 0.1, 5, genericReading);
        const auto factor = makeFactor(pim, withOffset(Options(), kOffset));
        auto gaussian = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(factor.noiseModel());
        ASSERT_NE(gaussian, nullptr);
        const Matrix3 R_bs = kOffset.rotation().matrix();
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(R_bs * pim.covariance() * R_bs.transpose()), gaussian->covariance(), 1e-15));
    }

    // ---- Sensor -------------------------------------------------------------------------

    TEST(AccelerometerSensor, OneFactorPerCompletedRowPair)
    {
        for (const double s : {1.0 / 3.0, 0.5})
            for (const bool bias : {true, false})
            {
                SCOPED_TRACE(testing::Message() << "s " << s << ", bias " << bias);
                std::shared_ptr<WNOAState> state;
                auto prior = makeWNOAPrior(state);
                Options options;
                options.estimate_bias = bias;
                WNOAAccelerometer accelerometer(prior, s, options);
                addSamples(accelerometer, s, 0.0, 0.2);

                KeyManager km;
                gtsam::Values values;
                EXPECT_TRUE(step({prior.get()}, {&accelerometer}, km, values, 0.0).empty());
                for (const double t : {0.1, 0.2})
                {
                    const auto factors = step({prior.get()}, {&accelerometer}, km, values, t);
                    ASSERT_EQ(factors.size(), 1u);
                    auto factor = std::dynamic_pointer_cast<Priors::TwoSidedInterpolationFactor<WNOAInterpolator, WNOAInterpolator>>(factors.front());
                    ASSERT_NE(factor, nullptr);
                    const bool on_grid = std::abs(s - 1.0 / 3.0) < 1e-9;
                    EXPECT_EQ(factor->keys().size(), (on_grid ? 6u : 12u) + (bias ? 2u : 0u));
                    for (const gtsam::Key key : factor->keys())
                        EXPECT_TRUE(values.exists(key));
                    EXPECT_EQ(factor->dim(), 3u);
                }
                EXPECT_TRUE(accelerometer.empty());
            }
    }

    TEST(AccelerometerSensor, ExtendStateAddsTheBiasStates)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAAccelerometer accelerometer(prior, 0.5);
        ASSERT_NE(accelerometer.biasModel(), nullptr);
        KeyManager km;
        gtsam::Values values;
        for (const double t : {0.0, 0.1, 0.2})
            step({prior.get()}, {&accelerometer}, km, values, t);
        EXPECT_EQ(accelerometer.biasModel()->state()->times(), (std::vector<double>{0.0, 0.1, 0.2}));
        for (const auto &node : accelerometer.biasModel()->state()->estimation_nodes)
            EXPECT_TRUE(values.exists(node->biasKey()));
        EXPECT_EQ(accelerometer.covarianceKeyGroups(values).size(), 3u);

        WNOAAccelerometer unbiased(prior, 0.5, noBias());
        EXPECT_EQ(unbiased.biasModel(), nullptr);
        gtsam::Values new_values;
        EXPECT_TRUE(unbiased.extendState(km, 0.3, values, new_values).empty());
        EXPECT_TRUE(new_values.empty());
        EXPECT_TRUE(unbiased.covarianceKeyGroups(values).empty());
    }

    TEST(AccelerometerSensor, AnUncoveredFirstIntervalIsSkipped)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAAccelerometer accelerometer(prior, 0.5);
        addSamples(accelerometer, 0.5, 0.03, 0.2);
        KeyManager km;
        gtsam::Values values;
        EXPECT_TRUE(step({prior.get()}, {&accelerometer}, km, values, 0.0).empty());
        EXPECT_TRUE(step({prior.get()}, {&accelerometer}, km, values, 0.1).empty());
        EXPECT_EQ(step({prior.get()}, {&accelerometer}, km, values, 0.2).size(), 1u);
    }

    TEST(AccelerometerSensor, SamplesBeforeTheFirstRowCarryIntoTheFirstInterval)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAAccelerometer accelerometer(prior, 0.5);
        accelerometer.addMeasurement(reading(0.5, -0.5, genericReading(0.0)));
        accelerometer.addMeasurement(reading(0.5, -0.02, genericReading(0.0)));
        addSamples(accelerometer, 0.5, 0.05, 0.1);
        KeyManager km;
        gtsam::Values values;
        EXPECT_TRUE(step({prior.get()}, {&accelerometer}, km, values, 0.0).empty());
        EXPECT_EQ(step({prior.get()}, {&accelerometer}, km, values, 0.1).size(), 1u);
        EXPECT_TRUE(accelerometer.empty());
    }

    TEST(AccelerometerSensor, LateOffArclengthAndMaskedSamplesAreDropped)
    {
        // Each sensor gets the same samples as a reference one, plus samples that must be
        // ignored; their factors must then agree exactly.
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAAccelerometer reference(prior, 0.5, noBias()), late(prior, 0.5, noBias()), off_arclength(prior, 0.5, noBias()), masked(prior, 0.5, noBias());
        for (auto *accelerometer : {&reference, &late, &off_arclength, &masked})
            addSamples(*accelerometer, 0.5, 0.0, 0.2);
        const Vector3 wild(50.0, -40.0, 30.0);
        off_arclength.addMeasurement(reading(0.9, 0.155, wild));
        Eigen::MatrixXi mask = Eigen::MatrixXi::Ones(3, 1);
        mask(1, 0) = 0;
        masked.addMeasurement(reading(0.5, 0.155, wild, mask));

        KeyManager km;
        gtsam::Values values;
        const std::vector<Types::Model *> sensors = {&reference, &late, &off_arclength, &masked};
        step({prior.get()}, sensors, km, values, 0.0);
        step({prior.get()}, sensors, km, values, 0.1);
        late.addMeasurement(reading(0.5, 0.05, wild)); // its interval [0, 0.1] is closed
        const auto factors = step({prior.get()}, sensors, km, values, 0.2);
        ASSERT_EQ(factors.size(), 4u);
        perturb(values);
        const gtsam::Vector expected = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factors[0])->unwhitenedError(values);
        for (size_t i = 1; i < factors.size(); i++)
            EXPECT_TRUE(gtsam::assert_equal(expected, std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factors[i])->unwhitenedError(values), 1e-12)) << i;
        for (auto *accelerometer : {&reference, &late, &off_arclength, &masked})
            EXPECT_TRUE(accelerometer->empty());
    }

    TEST(AccelerometerSensor, IntervalsStartingBeforeTheSmootherWindowAreDropped)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAAccelerometer accelerometer(prior, 0.5);
        addSamples(accelerometer, 0.5, 0.0, 0.2);
        KeyManager km;
        gtsam::Values values;
        step({prior.get()}, {&accelerometer}, km, values, 0.0);
        EXPECT_EQ(step({prior.get()}, {&accelerometer}, km, values, 0.1).size(), 1u);
        prior->onMarginalization(0.15, values); // the row at t = 0.1 is gone
        EXPECT_NEAR(prior->windowStart(), 0.15, 1e-12);
        EXPECT_TRUE(step({prior.get()}, {&accelerometer}, km, values, 0.2).empty());
    }

    TEST(AccelerometerSensor, InterpolatedFactorJacobiansMatchNumerical)
    {
        for (const double s : {1.0 / 3.0, 0.5})
            for (const bool slerp : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "s " << s << ", slerp " << slerp);
                std::shared_ptr<WNOAState> state;
                auto prior = makeWNOAPrior(state);
                Options options = withOffset(withSlerp(Options(), slerp), kOffset);
                WNOAAccelerometer accelerometer(prior, s, options);
                addSamples(accelerometer, s, 0.0, 0.1);
                KeyManager km;
                gtsam::Values values;
                step({prior.get()}, {&accelerometer}, km, values, 0.0);
                const auto factors = step({prior.get()}, {&accelerometer}, km, values, 0.1);
                ASSERT_EQ(factors.size(), 1u);
                perturb(values);
                auto factor = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factors.front());
                // Off-grid, WNOA::Interpolator's Jacobians w.r.t. the bordering poses are a
                // documented small-angle approximation (see TwoSidedInterpolationFactorTests).
                const bool on_grid = std::abs(s - 1.0 / 3.0) < 1e-9;
                expectJacobiansMatchNumerical(*factor, values, on_grid ? 1e-5 : 5e-3);
            }
    }

    TEST(AccelerometerSensor, WorksWithAnEndEffector)
    {
        auto state = std::make_shared<EEState>(std::vector<double>{0.0});
        auto prior = std::make_shared<EEPrior>(state, Pose(), Vector6::Ones());
        EEAccelerometer accelerometer(prior, 0.0);
        addSamples(accelerometer, 0.0, 0.0, 0.1);
        KeyManager km;
        gtsam::Values values;
        step({prior.get()}, {&accelerometer}, km, values, 0.0);
        const auto factors = step({prior.get()}, {&accelerometer}, km, values, 0.1);
        ASSERT_EQ(factors.size(), 1u);
        EXPECT_EQ(factors.front()->keys().size(), 6u); // two (pose, velocity) nodes and two biases
    }

    TEST(AccelerometerSensor, HasNoPerMeasurementFactorAndValidatesItsInputs)
    {
        std::shared_ptr<WNOAState> state;
        auto prior = makeWNOAPrior(state);
        WNOAAccelerometer accelerometer(prior, 0.5);
        EXPECT_EQ(accelerometer.getFactor(reading(0.5, 0.0, Vector3::Zero())), nullptr);
        EXPECT_NEAR(accelerometer.arclength(), 0.5, 1e-15);
        EXPECT_THROW(WNOAAccelerometer(nullptr, 0.5), std::invalid_argument);
        EXPECT_THROW(WNOAAccelerometer(prior, 1.5), std::invalid_argument);
        EXPECT_THROW(WNOAAccelerometer(prior, -0.1), std::invalid_argument);
    }

    // ---- End to end ---------------------------------------------------------------------

    namespace
    {
        /// A rigid body yawing at a constant rate while accelerating uniformly in the world.
        struct Trajectory
        {
            double yaw_rate = 0.2;
            Vector3 v0 = Vector3(0.1, 0.0, 0.0), a = Vector3(0.5, -0.3, 0.2);

            Rot R(double t) const { return Rot::Rz(yaw_rate * t); }
            Pose T(double t) const { return Pose(R(t), gtsam::Point3(v0 * t + 0.5 * a * t * t)); }
            Vector3 nu(double t) const { return R(t).unrotate(v0 + a * t); }
            Vector3 specificForce(double t, const Vector3 &gravity) const { return R(t).unrotate(a - gravity); }
        };

        /// Estimates the trajectory over [0, duration] with rows every 0.1 s, a pose sensor
        /// every `pose_period`, and optionally an accelerometer at 100 Hz (readings offset by `bias`);
        /// returns the end effector's final state, and the estimated bias if any.
        struct EstimateResult
        {
            Vector6 varpi;
            std::optional<Vector3> bias;
        };

        EstimateResult estimate(const Trajectory &trajectory, double duration, double pose_period, std::optional<Options> accelerometer_options,
                     const Vector3 &bias = Vector3::Zero())
        {
            auto state = std::make_shared<EEState>(std::vector<double>{0.0});
            auto prior = std::make_shared<EEPrior>(state, Pose(), Vector6::Constant(1.0));
            using PoseSensor = Sensors::PoseSensor<EEPrior::ValuesType>;
            const Eigen::Matrix<double, 6, 6> pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
            auto pose_sensor = std::make_shared<PoseSensor>(prior);

            // Batch: the end effector is unanchored between sparse poses, so the window
            // solves of a sliding-window smoother would be underconstrained at first.
            Estimator estimator(Estimator::BATCH_LAG);
            estimator.addPrior(prior);
            estimator.addSensor(pose_sensor);
            std::shared_ptr<EEAccelerometer> accelerometer;
            if (accelerometer_options)
            {
                accelerometer = std::make_shared<EEAccelerometer>(prior, 0.0, *accelerometer_options);
                estimator.addSensor(accelerometer);
            }

            const int rows = static_cast<int>(std::round(duration / 0.1));
            const int pose_every = static_cast<int>(std::round(pose_period / 0.1));
            for (int k = 0; k <= rows; k++)
            {
                const double t = 0.1 * k;
                if (k % pose_every == 0)
                    pose_sensor->addMeasurement(std::make_shared<Types::MeasurementBase>(0.0, t, trajectory.T(t).matrix(), pose_covariance));
                if (accelerometer)
                    for (int i = 0; i < 10 && k < rows; i++)
                    {
                        const double ti = t + 0.01 * i;
                        accelerometer->addMeasurement(reading(0.0, ti, Vector3(trajectory.specificForce(ti + 0.005, accelerometer_options->gravity) + bias)));
                    }
                estimator.estimate(t);
            }
            estimator.finalize();

            EstimateResult run;
            auto queried = std::dynamic_pointer_cast<EEPrior::ValuesType>(prior->query(duration, 0.0, estimator.getCurrentValues()));
            EXPECT_NE(queried, nullptr);
            run.varpi = queried ? queried->varpi() : Vector6::Zero();
            if (accelerometer && accelerometer->biasModel())
            {
                auto bias_values = std::dynamic_pointer_cast<Biases::BiasValues<Vector3>>(accelerometer->biasModel()->query(duration, 0.0, estimator.getCurrentValues()));
                EXPECT_NE(bias_values, nullptr);
                if (bias_values)
                    run.bias = bias_values->bias();
            }
            return run;
        }
    } // namespace

    TEST(AccelerometerSensor, ImprovesTheVelocityEstimateBetweenSparsePoses)
    {
        const Trajectory trajectory;
        const double duration = 2.0;
        const Vector3 truth = trajectory.nu(duration);
        const double pose_only = (estimate(trajectory, duration, 0.5, std::nullopt).varpi.tail<3>() - truth).norm();
        for (const bool slerp : {false, true})
        {
            Options options = withSlerp(noBias(), slerp);
            const double with_accelerometer = (estimate(trajectory, duration, 0.5, options).varpi.tail<3>() - truth).norm();
            EXPECT_LT(with_accelerometer, 0.5 * pose_only) << "slerp " << slerp << ": " << with_accelerometer << " vs " << pose_only;
            EXPECT_LT(with_accelerometer, 0.05) << "slerp " << slerp;
        }
    }

    TEST(AccelerometerSensor, EstimatesAConstantAccelerometerBias)
    {
        const Trajectory trajectory;
        const Vector3 bias(0.2, -0.1, 0.05);
        for (const bool slerp : {false, true})
        {
            const EstimateResult run = estimate(trajectory, 3.0, 0.1, withSlerp(Options(), slerp), bias);
            ASSERT_TRUE(run.bias.has_value());
            EXPECT_TRUE(gtsam::assert_equal(bias, *run.bias, 3e-2)) << "slerp " << slerp << ": " << run.bias->transpose();
        }
    }

    // ---- ConfigLoader -------------------------------------------------------------------

    namespace
    {
        struct AccelerometerDemoData
        {
            // One folder per test: ctest -j runs each test in its own process, concurrently.
            std::filesystem::path dir = std::filesystem::path(testing::TempDir()) /
                                        (std::string("accelerometer_sensor_demo_data_") + testing::UnitTest::GetInstance()->current_test_info()->name());

            AccelerometerDemoData()
            {
                std::filesystem::create_directories(dir);
                writeFile(dir / "accelerometer_ee.csv",
                          "sensor type:,accelerometer\nground truth:,0\ntimestamp (s),arclength (m),data\n"
                          "0.0,0.5,0.1,0.2,9.8\n0.01,0.5,0.4,0.5,9.7\n2.0,0.5,0.7,0.8,9.9\n");
                writeFile(dir / "aurora1.csv",
                          "sensor type:,aurora\nground truth:,0\ntimestamp (s),arclength (m),data\n"
                          "0.0,1.0,0,0,1,0,0,0\n");
            }
            ~AccelerometerDemoData() { std::filesystem::remove_all(dir); }

            Utilities::ConfigLoader loader(const std::string &demo_data_extra) const
            {
                const std::string config = writeFile(dir / "config.json", std::string(R"({ "demo_data": {
                    "folder_path": ")") + dir.string() + R"(",
                    "start_time": 0, "end_time": 1, "include_aurora": true, )" + demo_data_extra + R"( } })");
                Utilities::ConfigLoader loader;
                EXPECT_TRUE(loader.loadConfig(config));
                return loader;
            }

            template <typename PriorT>
            std::vector<std::shared_ptr<Types::SensorBase>> load(const std::string &demo_data_extra, const std::shared_ptr<PriorT> &prior) const
            {
                std::vector<std::shared_ptr<Types::SensorBase>> sensors, gt_sensors;
                loader(demo_data_extra).loadDemoData(sensors, gt_sensors, prior);
                EXPECT_TRUE(gt_sensors.empty());
                return sensors;
            }
        };

        std::shared_ptr<WNOAAccelerometer> findAccelerometer(const std::vector<std::shared_ptr<Types::SensorBase>> &sensors)
        {
            std::shared_ptr<WNOAAccelerometer> found;
            for (const auto &sensor : sensors)
                if (auto accelerometer = std::dynamic_pointer_cast<WNOAAccelerometer>(sensor))
                {
                    EXPECT_EQ(found, nullptr) << "more than one accelerometer";
                    found = accelerometer;
                }
            return found;
        }
    } // namespace

    TEST(AccelerometerSensor, ConfigLoaderLoadsAccelerometerSensorsAndTheirOptions)
    {
        AccelerometerDemoData data;
        std::shared_ptr<WNOAState> state;
        const auto sensors = data.load(R"(
            "include_accelerometer": true,
            "weights": { "R_accelerometer": [1e-3, 2e-3, 3e-3], "measurement_factor": 2 },
            "accelerometer": {
                "use_slerp": true,
                "gravity": [0, 0, -9.8],
                "offset": [0.1, 0.2, 0.3, 0, 0, 0.5],
                "bias": { "estimate": true, "initial": [0.01, 0.02, 0.03], "initial_sigma": [0.1, 0.2, 0.3], "random_walk_sigma": [1e-3, 2e-3, 3e-3] }
            })", makeWNOAPrior(state));
        ASSERT_EQ(sensors.size(), 2u);
        const auto accelerometer = findAccelerometer(sensors);
        ASSERT_NE(accelerometer, nullptr);

        EXPECT_NEAR(accelerometer->arclength(), 0.5, 1e-15);
        EXPECT_EQ(accelerometer->size(), 2u); // t = 2.0 is past end_time
        const auto measurement = accelerometer->peekAll().front();
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(2.0 * Vector3(1e-3, 2e-3, 3e-3).asDiagonal().toDenseMatrix()), gtsam::Matrix(measurement->covariance), 1e-15));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector(Vector3(0.1, 0.2, 9.8)), gtsam::Vector(measurement->value)));

        const Options &options = accelerometer->options();
        EXPECT_TRUE(options.use_slerp);
        EXPECT_TRUE(gtsam::assert_equal(Vector3(0, 0, -9.8), options.gravity));
        // The offset is a legacy [translational, rotational] Log, like the pose data.
        EXPECT_TRUE(gtsam::assert_equal(Pose::Expmap((Vector6() << 0, 0, 0.5, 0.1, 0.2, 0.3).finished()), options.offset, 1e-12));
        EXPECT_TRUE(options.estimate_bias);
        EXPECT_TRUE(gtsam::assert_equal(Vector3(0.01, 0.02, 0.03), options.bias.initial));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(Vector3(0.01, 0.04, 0.09).asDiagonal().toDenseMatrix()), gtsam::Matrix(options.bias.initial_covariance), 1e-15));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(Vector3(1e-6, 4e-6, 9e-6).asDiagonal().toDenseMatrix()), gtsam::Matrix(options.bias.random_walk_psd), 1e-18));
        EXPECT_NE(accelerometer->biasModel(), nullptr);
    }

    TEST(AccelerometerSensor, ConfigLoaderDefaultsWithoutAnAccelerometerBlock)
    {
        AccelerometerDemoData data;
        std::shared_ptr<WNOAState> state;
        const auto accelerometer = findAccelerometer(data.load(R"("include_accelerometer": true)", makeWNOAPrior(state)));
        ASSERT_NE(accelerometer, nullptr);
        const Options defaults;
        const Options &options = accelerometer->options();
        EXPECT_EQ(options.use_slerp, defaults.use_slerp);
        EXPECT_TRUE(gtsam::assert_equal(defaults.gravity, options.gravity));
        EXPECT_TRUE(gtsam::assert_equal(defaults.offset, options.offset));
        EXPECT_EQ(options.estimate_bias, defaults.estimate_bias);
        EXPECT_TRUE(gtsam::assert_equal(defaults.bias.initial, options.bias.initial));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(defaults.bias.initial_covariance), gtsam::Matrix(options.bias.initial_covariance)));
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(defaults.bias.random_walk_psd), gtsam::Matrix(options.bias.random_walk_psd)));
        // Without weights, the identity covariance.
        EXPECT_TRUE(gtsam::assert_equal(gtsam::Matrix(Matrix3::Identity()), gtsam::Matrix(accelerometer->peekAll().front()->covariance)));
    }

    TEST(AccelerometerSensor, ConfigLoaderCanDisableTheBias)
    {
        AccelerometerDemoData data;
        std::shared_ptr<WNOAState> state;
        const auto accelerometer = findAccelerometer(data.load(R"("include_accelerometer": true, "accelerometer": { "bias": { "estimate": false } })", makeWNOAPrior(state)));
        ASSERT_NE(accelerometer, nullptr);
        EXPECT_FALSE(accelerometer->options().estimate_bias);
        EXPECT_EQ(accelerometer->biasModel(), nullptr);
    }

    TEST(AccelerometerSensor, ConfigLoaderSkipsDisabledAccelerometerFiles)
    {
        AccelerometerDemoData data;
        std::shared_ptr<WNOAState> state;
        for (const std::string extra : {R"("include_accelerometer": false)", R"("include_gyro": false)"})
        {
            const auto sensors = data.load(extra, makeWNOAPrior(state));
            ASSERT_EQ(sensors.size(), 1u);
            EXPECT_EQ(findAccelerometer(sensors), nullptr);
        }
    }

    TEST(AccelerometerSensor, ConfigLoaderSkipsAccelerometerFilesForAPriorWithoutVelocity)
    {
        AccelerometerDemoData data;
        using WNOVPrior = Priors::WNOV::Prior<Pose>;
        auto state = std::make_shared<Types::ContinuumRobotState<Priors::WNOV::Node<Pose>>>(1.0, 3);
        auto prior = std::make_shared<WNOVPrior>(state, Pose(), Vector6::Zero());
        const auto sensors = data.load(R"("include_accelerometer": true)", prior);
        ASSERT_EQ(sensors.size(), 1u); // the aurora only
        EXPECT_NE(std::dynamic_pointer_cast<Sensors::PoseSensor<Priors::WNOV::Values<Pose>>>(sensors.front()), nullptr);
    }
} // namespace Spacetime
