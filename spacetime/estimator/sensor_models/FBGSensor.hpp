#ifndef FBGSensor_HPP
#define FBGSensor_HPP

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
        /// Node values with a strain epsilon() and an interpolated Strain() key.
        template <typename ValuesT>
        concept FBGSensorValuesType = Types::ValuesType<ValuesT> && requires(ValuesT n) {
            typename ValuesT::TangentVector;
            typename ValuesT::InterpolatedKeys;
            { n.epsilon() } -> std::convertible_to<typename ValuesT::TangentVector>;
            { ValuesT::InterpolatedKeys::Strain() } -> std::convertible_to<gtsam::Key>;
        };
    };

    /**
     * @brief Fibre Bragg grating (FBG) shape sensor (the RSS paper's Sec. III-C, eqs. (7)-(9)):
     *        measures the body-centric generalized strain epsilon of T_ib at a given arclength
     *        and time. Measurement values are strains in GTSAM's [rotational, translational]
     *        layout (the loader reorders the data's [translational, rotational] rows), with
     *        covariances of the same size.
     */
    template <FBGSensorValuesType ValuesT>
    class FBGSensor : public Types::SensorBase
    {
        using TangentVector = typename ValuesT::TangentVector;
        static constexpr int dim = TangentVector::RowsAtCompileTime;

    public:
        /**
         * @brief Measurement factor built by FBGSensor::getFactor: e = epsilon - epsilon_meas
         *        (eq. (9)), referencing only ValuesT::InterpolatedKeys::Strain() -- evaluated as
         *        the inner factor of a Priors::InterpolationFactor. Masked components have
         *        their error and Jacobian rows zeroed.
         */
        class Factor : public gtsam::NoiseModelFactorN<TangentVector>
        {
        public:
            using Base = gtsam::NoiseModelFactorN<TangentVector>;

            /// `mask` is dim x 1; a 0 entry ignores that component.
            Factor(const gtsam::SharedNoiseModel &noise_model, const TangentVector &measured, const Eigen::MatrixXi &mask)
                : Base(noise_model, ValuesT::InterpolatedKeys::Strain()), m_measured(measured), m_mask(mask) {}

            /// The masked error epsilon - epsilon_meas, with H its Jacobian in epsilon.
            gtsam::Vector evaluateError(const TangentVector &epsilon, gtsam::OptionalMatrixType H = nullptr) const override
            {
                TangentVector e = epsilon - m_measured;
                Eigen::Matrix<double, dim, dim> S = Eigen::Matrix<double, dim, dim>::Identity();
                for (unsigned int i = 0; i < dim; i++)
                    if (m_mask(i, 0) == 0)
                    {
                        e(i) = 0;
                        S.row(i).setZero();
                    }
                if (H)
                    *H = S;
                return e;
            }

        private:
            TangentVector m_measured;
            Eigen::MatrixXi m_mask;
        };

        /// A sensor measuring the state of `stateful`.
        explicit FBGSensor(Types::SensorBase::pStatefulType stateful)
            : Types::SensorBase(std::move(stateful)) {}

        ~FBGSensor() = default;

        /// The Factor for `measurement`, with a Gaussian noise model of its covariance.
        gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
        {
            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(measurement->covariance);
            return std::make_shared<Factor>(noise_model, TangentVector(measurement->value), measurement->mask);
        }
    };
} // namespace Spacetime::Sensors
#endif // FBGSensor_HPP
