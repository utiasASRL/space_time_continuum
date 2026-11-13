#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include <Eigen/Core>
#include <concepts>

#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Types
{
    /// A timestamped measurement at an arclength, with its covariance and a validity mask.
    struct MeasurementBase // Expressed with T_ib frames
    {
        MeasurementBase() = default;
        /// Constructs with every component of `value` marked valid.
        MeasurementBase(double arclength, double time, const Eigen::MatrixXd &value, const Eigen::MatrixXd &covariance)
            : arclength(arclength), time(time), value(value), covariance(covariance) { mask = Eigen::MatrixXi::Ones(value.rows(), value.cols()); }
        /// Constructs with an explicit validity `mask` (same shape as `value`, 1 = valid).
        MeasurementBase(double arclength, double time, const Eigen::MatrixXd &value, const Eigen::MatrixXd &covariance, const Eigen::MatrixXi &mask)
            : arclength(arclength), time(time), value(value), covariance(covariance), mask(mask) {}
        virtual ~MeasurementBase() = default;
            
        Eigen::MatrixXd value;        // Measurement value. Size and meaning depends on measurement type
        mutable Eigen::MatrixXi mask; // Mask for valid measurement components. Mutable as some factors may modify it at runtime (i.e. rejected outliers)
        Eigen::MatrixXd covariance;   // Measurement covariance. Size and meaning depends on measurement type

        double arclength; // Arclength of measurement [m]
        double time;      // Timestep of measurement (s)

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

    /// Types derived from MeasurementBase.
    template <typename T>
    concept MeasurementType = std::derived_from<T, MeasurementBase>;
}
#endif // MEASUREMENT_HPP