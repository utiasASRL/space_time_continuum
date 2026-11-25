#ifndef GyroSensor_HPP
#define GyroSensor_HPP

#include <algorithm>
#include <cmath>
#include <concepts>

#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/types/ActuatorBase.hpp>
#include <spacetime/estimator/types/ContinuumRobot.hpp>
#include <spacetime/estimator/types/InterpolatedKeys.hpp>
#include <spacetime/estimator/types/Robot.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/bias_models/Bias.hpp>
#include <spacetime/estimator/utilities/functions.hpp>

namespace Spacetime::Sensors
{
    namespace
    {
        /// Node values with a velocity varpi() and an interpolated Velocity() key.
        template <typename ValuesT>
        concept GyroSensorValuesType = Types::ValuesType<ValuesT> && requires(ValuesT n) {
            typename ValuesT::PoseType;
            typename ValuesT::TangentVector;
            typename ValuesT::InterpolatedKeys;
            { n.varpi() } -> std::convertible_to<typename ValuesT::TangentVector>;
            { ValuesT::InterpolatedKeys::Velocity() } -> std::convertible_to<gtsam::Key>;
        };
    };

    /// Parameters of a GyroSensor.
    struct GyroOptions
    {
        /// Pose T_bs of the gyroscope in the robot's body frame. Only its rotation R_bs is used: a
        /// rigid body's angular velocity is the same everywhere on it. Readings are in the sensor frame.
        gtsam::Pose3 offset = gtsam::Pose3();
        /// Estimate a gyroscope bias (a Biases::BiasModel owned by the sensor). Off by default.
        bool estimate_bias = false;
        /// The bias's prior and random walk, used when estimate_bias is set.
        Biases::BiasOptions<gtsam::Vector3> bias = defaultBias();

        /// Zero initial bias, with a 0.01 rad/s prior std and a 1e-4 rad/s/sqrt(s) random walk.
        static Biases::BiasOptions<gtsam::Vector3> defaultBias()
        {
            Biases::BiasOptions<gtsam::Vector3> bias;
            bias.initial = gtsam::Vector3::Zero();
            bias.initial_covariance = 1e-4 * gtsam::Matrix3::Identity();
            bias.random_walk_psd = 1e-8 * gtsam::Matrix3::Identity();
            return bias;
        }
    };

    /**
     * @brief 3DoF gyroscope sensor (the paper's eqs. (30)-(32)): measures the angular
     *        velocity of the robot at a given arclength and time, expressed in the sensor
     *        frame (rotated by GyroOptions::offset from the body frame; the body frame by
     *        default). Measurement values are 3x1 angular velocities [rad/s], covariances 3x3.
     *
     * With GyroOptions::estimate_bias, the sensor owns a Biases::BiasModel<gtsam::Vector3>
     * (one bias node per estimation time, a random walk), interpolated linearly to each
     * measurement's time, and delegates its extendState, onMarginalization, lockedValues,
     * resolveLocks and covarianceKeyGroups to it, so adding the sensor to the Estimator adds
     * its bias states. A measurement before the first bias node, or before the latest
     * marginalization cutoff, is dropped.
     *
     * @tparam ValuesT The measured prior's node-values type (e.g. WNOA::NodeValues<gtsam::Pose3>),
     *         providing a body twist varpi() and an interpolated Velocity() key; its pose must
     *         be 3D.
     */
    template <GyroSensorValuesType ValuesT>
    class GyroSensor : public Types::SensorBase
    {
        using Pose = ValuesT::PoseType;
        using TangentVector = typename ValuesT::TangentVector;
        static constexpr int dim = gtsam::traits<Pose>::dimension;
        using ErrorVector = Eigen::Matrix<double, 3, 1>;

        // The angular velocity is read from varpi's leading 3 components, GTSAM's
        // [rotational, translational] layout -- which only exists for 3D poses.
        static_assert(dim == 6, "GyroSensor requires a 3D pose type (e.g. gtsam::Pose3).");

    public:
        using BiasModelType = Biases::BiasModel<gtsam::Vector3>; ///< The gyroscope bias model.
        /// Weights of each bias node key in a linearly interpolated bias (see Biases::BiasModel::weights).
        using BiasWeights = std::vector<std::pair<gtsam::Key, double>>;

        /**
         * @brief Measurement factor built by GyroSensor::getFactor without a bias. varpi is
         *        the body-frame twist of T_ib (see Priors::WNOA::BinaryTimeFactor::Factor), so the
         *        modelled angular velocity is R_bs^T varpi.head<3>() and the error is
         *        e = w_meas - R_bs^T varpi.head<3>(), referencing only
         *        ValuesT::InterpolatedKeys::Velocity() -- evaluated as the inner factor of a
         *        Priors::InterpolationFactor.
         */
        class Factor : public gtsam::NoiseModelFactorN<TangentVector>
        {
        public:
            using Base = gtsam::NoiseModelFactorN<TangentVector>; ///< gtsam base class.

            /// `measured` is the sensor-frame angular velocity [rad/s]; `mask` is 3 x 1; a 0 entry ignores that component; `R_bs` the sensor's rotation in the body frame.
            Factor(const gtsam::SharedNoiseModel &noise_model, const ErrorVector &measured, const Eigen::MatrixXi &mask,
                   const gtsam::Rot3 &R_bs = gtsam::Rot3())
                : Base(noise_model, ValuesT::InterpolatedKeys::Velocity()), m_measured(measured), m_mask(mask), m_R_sb(R_bs.matrix().transpose()) {}

            /// The masked 3-vector error w_meas - R_bs^T varpi.head<3>(), with H its 3 x dim Jacobian in varpi.
            gtsam::Vector evaluateError(const TangentVector &varpi, gtsam::OptionalMatrixType H = nullptr) const override
            {
                ErrorVector e = m_measured - m_R_sb * varpi.template head<3>();

                for (unsigned int i = 0; i < 3; i++)
                    if (m_mask(i, 0) == 0)
                        e(i, 0) = 0;

                if (H)
                {
                    Eigen::Matrix<double, 3, dim> S = Eigen::Matrix<double, 3, dim>::Zero();
                    S.template leftCols<3>() = -m_R_sb;
                    for (unsigned int i = 0; i < 3; i++)
                        if (m_mask(i, 0) == 0)
                            S.row(i).setZero();
                    *H = S;
                }
                return e;
            }

        private:
            ErrorVector m_measured;
            Eigen::MatrixXi m_mask;
            Eigen::Matrix3d m_R_sb;
        };

        /**
         * @brief Measurement factor built by GyroSensor::getFactor with a bias:
         *        e = w_meas - R_bs^T varpi.head<3>() - sum_i w_i b_i, on
         *        ValuesT::InterpolatedKeys::Velocity() and the bias node keys (passed through
         *        a Priors::InterpolationFactor unchanged), with b interpolated linearly
         *        between the bias nodes by the weights w_i.
         */
        class BiasedFactor : public gtsam::NoiseModelFactor
        {
        public:
            using Base = gtsam::NoiseModelFactor; ///< gtsam base class.
            using Base::error;
            using Base::unwhitenedError;

            /**
             * @param measured The sensor-frame angular velocity [rad/s].
             * @param mask 3 x 1; a 0 entry ignores that component.
             * @param bias_weights The bias node keys (each a gtsam::Vector3) and their interpolation weights.
             * @param R_bs The sensor's rotation in the body frame.
             * @throws std::invalid_argument for no bias weights.
             */
            BiasedFactor(const gtsam::SharedNoiseModel &noise_model, const ErrorVector &measured, const Eigen::MatrixXi &mask,
                         const BiasWeights &bias_weights, const gtsam::Rot3 &R_bs = gtsam::Rot3())
                : Base(noise_model), m_measured(measured), m_mask(mask), m_R_sb(R_bs.matrix().transpose())
            {
                if (bias_weights.empty())
                    throw std::invalid_argument("GyroSensor::BiasedFactor needs at least one bias key.");
                this->keys_.push_back(ValuesT::InterpolatedKeys::Velocity());
                for (const auto &[key, weight] : bias_weights)
                {
                    this->keys_.push_back(key);
                    m_weights.push_back(weight);
                }
            }

            /// The masked error (see the class description), with Jacobians w.r.t. the velocity and every bias key.
            gtsam::Vector unwhitenedError(const gtsam::Values &values, gtsam::OptionalMatrixVecType H = nullptr) const override
            {
                const TangentVector &varpi = values.at<TangentVector>(this->keys_[0]);
                ErrorVector e = m_measured - m_R_sb * varpi.template head<3>();
                for (size_t i = 0; i < m_weights.size(); i++)
                    e -= m_weights[i] * values.at<gtsam::Vector3>(this->keys_[1 + i]);

                Eigen::Matrix3d mask = Eigen::Matrix3d::Identity();
                for (unsigned int i = 0; i < 3; i++)
                    if (m_mask(i, 0) == 0)
                        mask(i, i) = 0.0;
                e = mask * e;

                if (H)
                {
                    Eigen::Matrix<double, 3, GyroSensor::dim> S = Eigen::Matrix<double, 3, GyroSensor::dim>::Zero();
                    S.template leftCols<3>() = -mask * m_R_sb;
                    (*H)[0] = S;
                    for (size_t i = 0; i < m_weights.size(); i++)
                        (*H)[1 + i] = -m_weights[i] * mask;
                }
                return e;
            }

            /// Prints the keys and the measured angular velocity.
            void print(const std::string &s = "", const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override
            {
                std::cout << s << "GyroSensor::BiasedFactor on ";
                for (const auto &k : this->keys())
                    std::cout << keyFormatter(k) << " ";
                std::cout << "measuring " << m_measured.transpose() << std::endl;
            }

            /// Same type, keys, noise model, measurement and weights.
            bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override
            {
                const auto *e = dynamic_cast<const BiasedFactor *>(&expected);
                if (e == nullptr || !Base::equals(*e, tol) || !m_measured.isApprox(e->m_measured, tol) || m_weights.size() != e->m_weights.size())
                    return false;
                for (size_t i = 0; i < m_weights.size(); i++)
                    if (std::abs(m_weights[i] - e->m_weights[i]) > tol)
                        return false;
                return true;
            }

        private:
            ErrorVector m_measured;
            Eigen::MatrixXi m_mask;
            Eigen::Matrix3d m_R_sb;
            std::vector<double> m_weights;
        };

        /// A sensor measuring the state of `stateful` (see GyroOptions).
        explicit GyroSensor(Types::SensorBase::pStatefulType stateful, const GyroOptions &options = GyroOptions())
            : Types::SensorBase(std::move(stateful)), m_options(options)
        {
            if (m_options.estimate_bias)
                m_bias = std::make_shared<BiasModelType>(m_options.bias);
        }

        ~GyroSensor() = default;

        /// The options the sensor was built with.
        const GyroOptions &options() const { return m_options; }
        /// The estimated gyroscope bias (e.g. to query it), or nullptr without GyroOptions::estimate_bias.
        const std::shared_ptr<BiasModelType> &biasModel() const { return m_bias; }

        /// The Factor (or BiasedFactor) for `measurement`, with a Gaussian noise model of its covariance; nullptr if its bias cannot be resolved.
        gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
        {
            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(measurement->covariance);
            const gtsam::Rot3 R_bs = m_options.offset.rotation();
            if (!m_bias)
                return std::make_shared<Factor>(noise_model, ErrorVector(measurement->value), measurement->mask, R_bs);

            const auto bracket = m_bias->bracket(measurement->time);
            if (!bracket || bracket->node0->time < m_cutoff - TOLERANCE)
                return nullptr; // before the first bias node, or its bias nodes were marginalized
            const BiasWeights weights = m_bias->weights(measurement->time);
            return std::make_shared<BiasedFactor>(noise_model, ErrorVector(measurement->value), measurement->mask, weights, R_bs);
        }

        /// Adds the bias node at `time`, if the bias is estimated.
        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            return m_bias ? m_bias->extendState(km, time, current_values, new_values) : Factors();
        }

        /// Forwards to the bias model, if the bias is estimated, and remembers `cutoff`.
        void onMarginalization(double cutoff, const gtsam::Values &values) override
        {
            m_cutoff = std::max(m_cutoff, cutoff);
            if (m_bias)
                m_bias->onMarginalization(cutoff, values);
        }

        /// The bias model's locked values, or none without a bias.
        const gtsam::Values &lockedValues() const override { return m_bias ? m_bias->lockedValues() : Types::SensorBase::lockedValues(); }

        /// Forwards to the bias model, if the bias is estimated.
        void resolveLocks(const gtsam::Values &new_values) override
        {
            if (m_bias)
                m_bias->resolveLocks(new_values);
        }

        /// The bias model's covariance key groups (each bias node alone), or none without a bias.
        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
        {
            return m_bias ? m_bias->covarianceKeyGroups(values) : std::vector<gtsam::KeyVector>();
        }

    private:
        GyroOptions m_options;
        std::shared_ptr<BiasModelType> m_bias;
        double m_cutoff = -std::numeric_limits<double>::infinity(); // the latest marginalization cutoff
    };
} // namespace Spacetime::Sensors
#endif // GyroSensor_HPP
