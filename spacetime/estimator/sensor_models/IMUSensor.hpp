#ifndef IMUSENSOR_HPP
#define IMUSENSOR_HPP

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/sensor_models/AccelerometerSensor.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>
#include <spacetime/estimator/types/Measurement.hpp>
#include <spacetime/estimator/types/SensorBase.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime::Sensors
{
    /// Parameters of an IMUSensor.
    struct IMUOptions
    {
        /// Pose T_bs of the IMU in the robot's body frame at its arclength; it replaces both
        /// gyro.offset and accelerometer.offset.
        gtsam::Pose3 offset = gtsam::Pose3();
        /// The gyroscope's parameters (its offset is ignored). Estimates a bias by default.
        GyroOptions gyro = defaultGyro();
        /// The accelerometer's parameters (its offset is ignored). Estimates a bias by default.
        AccelerometerOptions accelerometer = AccelerometerOptions();

        /// GyroOptions with GyroOptions::estimate_bias set.
        static GyroOptions defaultGyro()
        {
            GyroOptions gyro;
            gyro.estimate_bias = true;
            return gyro;
        }
    };

    /**
     * @class IMUSensor
     * @brief A 6-axis IMU at a fixed arclength of a robot (or end effector): a GyroSensor and
     *        an AccelerometerSensor sharing one sensor frame (IMUOptions::offset), fed from
     *        one measurement queue.
     *
     * Measurement values are 6x1, [angular velocity [rad/s]; specific force [m/s^2]], both in
     * the sensor frame; covariances are 6x6, of which only the two diagonal 3x3 blocks are
     * used; masks are 6x1. Each measurement is split into a gyroscope and an accelerometer
     * measurement (same time and arclength), which the two wrapped sensors then handle as
     * they would on their own (see GyroSensor and AccelerometerSensor). The accelerometer
     * drops a sample with a masked component.
     *
     * Only the IMU's queue is registered with the Estimator: the wrapped sensors' queues are
     * fed by the IMU. It forwards extendState, onMarginalization, lockedValues, resolveLocks
     * and covarianceKeyGroups to both, so adding the IMU adds both bias states.
     *
     * @tparam PriorT The prior extending the measured state (e.g. WNOA::Prior<gtsam::Pose3>,
     *         Joints::EndEffectorPrior<gtsam::Pose3>).
     */
    template <AccelerometerPriorType PriorT>
        requires GyroSensorValuesType<typename PriorT::ValuesType>
    class IMUSensor : public Types::SensorBase
    {
    public:
        using GyroType = GyroSensor<typename PriorT::ValuesType>; ///< The wrapped gyroscope.
        using AccelerometerType = AccelerometerSensor<PriorT>;    ///< The wrapped accelerometer.

        /**
         * @param prior The prior extending the measured state.
         * @param arclength The sensor's arclength [m], within the prior's nodes.
         * @param options See IMUOptions.
         * @throws std::invalid_argument for a null prior or an arclength outside its nodes.
         */
        IMUSensor(std::shared_ptr<PriorT> prior, double arclength, const IMUOptions &options = IMUOptions())
            : Types::SensorBase(checked(prior)), m_arclength(arclength), m_options(withOffset(options))
        {
            m_accelerometer = std::make_shared<AccelerometerType>(prior, arclength, m_options.accelerometer);
            m_gyro = std::make_shared<GyroType>(prior, m_options.gyro);
        }

        /// The sensor's arclength [m].
        double arclength() const { return m_arclength; }
        /// The options the sensor was built with (with both offsets set to IMUOptions::offset).
        const IMUOptions &options() const { return m_options; }
        /// The wrapped gyroscope (e.g. to query its bias model).
        const std::shared_ptr<GyroType> &gyro() const { return m_gyro; }
        /// The wrapped accelerometer (e.g. to query its bias model).
        const std::shared_ptr<AccelerometerType> &accelerometer() const { return m_accelerometer; }

        /// No per-measurement factor: measurements are handed to the wrapped sensors (see getNewFactors).
        gtsam::NoiseModelFactor::shared_ptr getFactor(const pMeasurementType measurement) const override
        {
            (void)measurement;
            return nullptr;
        }

        /// Both wrapped sensors' new variables (their bias nodes).
        Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values) override
        {
            Factors factors = m_gyro->extendState(km, time, current_values, new_values);
            append(factors, m_accelerometer->extendState(km, time, current_values, new_values));
            refreshLockedValues();
            return factors;
        }

        /// Splits the measurements drained up to `time` between the wrapped sensors, and returns both sensors' new factors.
        Factors getNewFactors(double time, gtsam::Values &new_values) override
        {
            for (const pMeasurementType &measurement : drainUpTo(time))
                split(measurement);
            Factors factors = m_gyro->getNewFactors(time, new_values);
            append(factors, m_accelerometer->getNewFactors(time, new_values));
            return factors;
        }

        /// Forwards to both wrapped sensors.
        void onMarginalization(double cutoff, const gtsam::Values &values) override
        {
            m_gyro->onMarginalization(cutoff, values);
            m_accelerometer->onMarginalization(cutoff, values);
        }

        /// Both wrapped sensors' locked values.
        const gtsam::Values &lockedValues() const override { return m_locked; }

        /// Forwards to both wrapped sensors.
        void resolveLocks(const gtsam::Values &new_values) override
        {
            m_gyro->resolveLocks(new_values);
            m_accelerometer->resolveLocks(new_values);
            refreshLockedValues();
        }

        /// Both wrapped sensors' covariance key groups.
        std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const override
        {
            std::vector<gtsam::KeyVector> groups = m_gyro->covarianceKeyGroups(values);
            for (auto &group : m_accelerometer->covarianceKeyGroups(values))
                groups.push_back(std::move(group));
            return groups;
        }

    private:
        static Types::SensorBase::pStatefulType checked(const std::shared_ptr<PriorT> &prior)
        {
            if (!prior)
                throw std::invalid_argument("IMUSensor requires a non-null prior.");
            return prior;
        }

        static IMUOptions withOffset(IMUOptions options)
        {
            options.gyro.offset = options.offset;
            options.accelerometer.offset = options.offset;
            return options;
        }

        static void append(Factors &factors, Factors more)
        {
            for (auto &factor : more)
                factors.push_back(std::move(factor));
        }

        void refreshLockedValues()
        {
            m_locked = gtsam::Values();
            m_locked.insert(m_gyro->lockedValues());
            m_locked.insert(m_accelerometer->lockedValues());
        }

        /// Queues `measurement`'s gyroscope and accelerometer halves on the wrapped sensors; drops (with a warning) a malformed one.
        void split(const pMeasurementType &measurement)
        {
            if (measurement->value.size() != 6 || measurement->covariance.rows() != 6 || measurement->covariance.cols() != 6 ||
                measurement->mask.size() != 6)
            {
                warning() << "IMUSensor: dropping the measurement at (t=" << measurement->time << ", s=" << measurement->arclength
                          << "): expected a 6x1 value and mask and a 6x6 covariance.";
                return;
            }
            const Eigen::VectorXd value = Eigen::Map<const Eigen::VectorXd>(measurement->value.data(), 6);
            const Eigen::VectorXi mask = Eigen::Map<const Eigen::VectorXi>(measurement->mask.data(), 6);
            m_gyro->addMeasurement(std::make_shared<Types::MeasurementBase>(
                measurement->arclength, measurement->time, Eigen::MatrixXd(value.head<3>()),
                Eigen::MatrixXd(measurement->covariance.topLeftCorner<3, 3>()), Eigen::MatrixXi(mask.head<3>())));
            m_accelerometer->addMeasurement(std::make_shared<Types::MeasurementBase>(
                measurement->arclength, measurement->time, Eigen::MatrixXd(value.tail<3>()),
                Eigen::MatrixXd(measurement->covariance.bottomRightCorner<3, 3>()), Eigen::MatrixXi(mask.tail<3>())));
        }

        double m_arclength;
        IMUOptions m_options;
        std::shared_ptr<GyroType> m_gyro;
        std::shared_ptr<AccelerometerType> m_accelerometer;
        gtsam::Values m_locked;
    };
} // namespace Spacetime::Sensors

#endif // IMUSENSOR_HPP
