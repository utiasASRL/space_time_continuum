#ifndef GyroSensor_HPP
#define GyroSensor_HPP

#include <Eigen/Dense>
#include <concepts>
#include <memory>
#include <utility>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

#include <spacetime/Types.hpp>

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

    /**
     * @brief 3DoF gyroscope sensor (the paper's eqs. (30)-(32)): measures the angular
     *        velocity of the robot at a given arclength and time, expressed in the body
     *        frame. Measurement values are 3x1 angular velocities [rad/s], covariances 3x3.
     * @tparam ValuesT The measured prior's node-values type (e.g. WNOA::Values<gtsam::Pose3>),
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
        /**
         * @brief Measurement factor built by GyroSensor::getFactor. varpi is the body-frame
         *        twist of T_ib (see Priors::WNOA::BinaryTimeFactor::Factor), so the modelled angular velocity
         *        is its rotational block and the error is e = w_meas - varpi.head<3>(),
         *        referencing only ValuesT::InterpolatedKeys::Velocity() -- evaluated as the
         *        inner factor of a Priors::InterpolationFactor.
         */
        class Factor : public gtsam::NoiseModelFactorN<TangentVector>
        {
        public:
            using Base = gtsam::NoiseModelFactorN<TangentVector>; ///< gtsam base class.

            /// `measured` is the body-frame angular velocity [rad/s]; `mask` is 3 x 1; a 0 entry ignores that component.
            Factor(const gtsam::SharedNoiseModel &noise_model, const ErrorVector &measured, const Eigen::MatrixXi &mask)
                : Base(noise_model, ValuesT::InterpolatedKeys::Velocity()), m_measured(measured), m_mask(mask) {}

            /// The masked 3-vector error w_meas - varpi.head<3>(), with H its 3 x dim Jacobian in varpi.
            gtsam::Vector evaluateError(const TangentVector &varpi, gtsam::OptionalMatrixType H = nullptr) const override
            {
                ErrorVector e = m_measured - varpi.template head<3>();

                for (unsigned int i = 0; i < 3; i++)
                    if (m_mask(i, 0) == 0)
                        e(i, 0) = 0;

                if (H)
                {
                    Eigen::Matrix<double, 3, dim> S = Eigen::Matrix<double, 3, dim>::Zero();
                    S.template leftCols<3>() = -Eigen::Matrix3d::Identity();
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
        };

        /// A sensor measuring the state of `stateful`.
        explicit GyroSensor(Types::SensorBase::pStatefulType stateful)
            : Types::SensorBase(std::move(stateful)) {}

        ~GyroSensor() = default;

        /// The Factor for `measurement`, with a Gaussian noise model of its covariance.
        gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
        {
            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(measurement->covariance);
            return std::make_shared<Factor>(noise_model, ErrorVector(measurement->value), measurement->mask);
        }
    };
} // namespace Spacetime::Sensors
#endif // GyroSensor_HPP
