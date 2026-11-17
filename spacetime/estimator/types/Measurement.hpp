#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include <Eigen/Core>
#include <concepts>

#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Types
{
    /**
     * @brief A timestamped measurement at an arclength, with its covariance and a validity
     *        mask. The layout of `value` and `covariance` is defined by the model that
     *        consumes it (e.g. a 4x4 pose T_ib for Sensors::PoseSensor, a 3x1 angular
     *        velocity for Sensors::GyroSensor, an input vector for Types::ActuatorBase).
     *        Poses are expressed as T_ib (body frame b in the inertial frame i).
     */
    struct MeasurementBase // Expressed with T_ib frames
    {
        /// Uninitialized arclength and time; empty value, covariance and mask.
        MeasurementBase() = default;
        /// Constructs with every component of `value` marked valid.
        MeasurementBase(double arclength, double time, const Eigen::MatrixXd &value, const Eigen::MatrixXd &covariance)
            : arclength(arclength), time(time), value(value), covariance(covariance) { mask = Eigen::MatrixXi::Ones(value.rows(), value.cols()); }
        /// Constructs with an explicit validity `mask` (same shape as `value`, 1 = valid).
        MeasurementBase(double arclength, double time, const Eigen::MatrixXd &value, const Eigen::MatrixXd &covariance, const Eigen::MatrixXi &mask)
            : arclength(arclength), time(time), value(value), covariance(covariance), mask(mask) {}
        virtual ~MeasurementBase() = default;
            
        Eigen::MatrixXd value;        ///< Measured value; size and meaning depend on the consuming model.
        mutable Eigen::MatrixXi mask; ///< Validity of each component of `value` (1 = valid, 0 = ignored); mutable so a factor may reject outliers at runtime.
        Eigen::MatrixXd covariance;   ///< Measurement covariance; size and layout depend on the consuming model.

        double arclength; ///< Arclength of the measurement along the robot [m].
        double time;      ///< Time of the measurement [s].

        /// Logs the arclength, time and masked value.
        void print() const
        {
            info() << "Measurement\n------------------------------"
                   << "\nArclength: " << arclength
                   << "\nTime: " << time
                   << "\n------------------------------\n";

            for (int row = 0; row < value.rows(); row++)
            {
                for (int col = 0; col < value.cols(); col++)
                {
                    info() << value(row, col) * mask(row, col) << ", ";
                }
                info() << std::endl;
            }
            info() << "\n------------------------------\n";
        }
    };

    /// Types derived from MeasurementBase. @tparam T The candidate measurement type.
    template <typename T>
    concept MeasurementType = std::derived_from<T, MeasurementBase>;
}
#endif // MEASUREMENT_HPP