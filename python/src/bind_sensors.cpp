/**
 * @file bind_sensors.cpp
 * @brief spacetime.sensors: the built-in sensors, for each node values type they support, and
 *        their factors (for Python sensors to reuse).
 */

#include <spacetime/estimator/sensor_models/FBGSensor.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        using Spacetime::Types::SensorBase;
        using pPrior = pStateful;

        template <typename ValuesT>
        using PoseSensor = Spacetime::Sensors::PoseSensor<ValuesT>;
        template <typename ValuesT>
        using FBGSensor = Spacetime::Sensors::FBGSensor<ValuesT>;
        template <typename ValuesT>
        using GyroSensor = Spacetime::Sensors::GyroSensor<ValuesT>;

        /// Binds sensor class SensorT, subclassable in Python, constructed from `CtorArgs...`.
        template <typename SensorT, typename... CtorArgs, typename... Names>
        void bindSensor(py::module_ &m, const char *name, const char *doc, Names... names)
        {
            py::classh<SensorT, PySensor<SensorT>, SensorBase>(m, name, doc).def(py::init<CtorArgs...>(), names...);
        }

        /// Binds a sensor's factor (on its interpolated key), measuring a `Measured`.
        template <typename FactorT, typename Measured>
        void bindFactor(py::module_ &m, const char *name, const char *doc)
        {
            py::class_<FactorT, gtsam::NoiseModelFactor, std::shared_ptr<FactorT>>(m, name, doc)
                .def(py::init<const gtsam::SharedNoiseModel &, const Measured &, const Eigen::MatrixXi &>(), py::arg("noise_model"), py::arg("measured"),
                     py::arg("mask"));
        }

        /// make(tag) with a null pointer tag of the values type of the first of the priors PriorT, Rest... that `prior` is.
        template <typename PriorT, typename... Rest, typename Make>
        py::object dispatchOnPrior(const pPrior &prior, const char *sensor, Make &&make)
        {
            if (std::dynamic_pointer_cast<PriorT>(prior))
                return make(static_cast<typename PriorT::ValuesType *>(nullptr));
            if constexpr (sizeof...(Rest) > 0)
                return dispatchOnPrior<Rest...>(prior, sensor, std::forward<Make>(make));
            else
                throw py::type_error(std::string(sensor) + ": no built-in sensor for this prior (write a Python SensorBase subclass instead).");
        }
    } // namespace

    void bindSensors(py::module_ &m)
    {
        bindSensor<PoseSensor<WnoaValues>, pPrior>(m, "PoseSensorWNOA", "A 6-DoF pose sensor (T_ib) on a WNOA robot (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<PoseSensor<WnovValues>, pPrior>(m, "PoseSensorWNOV", "A 6-DoF pose sensor (T_ib) on a WNOV robot (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<PoseSensor<QuasiStaticValues>, pPrior>(m, "PoseSensorQuasiStatic", "A 6-DoF pose sensor (T_ib) on a quasi-static robot (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<PoseSensor<EndEffectorValues>, pPrior>(m, "PoseSensorEndEffector", "A 6-DoF pose sensor (T_ib) on an end effector (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<FBGSensor<WnoaValues>, pPrior>(m, "FBGSensorWNOA", "A strain sensor (e.g. FBGs) on a WNOA robot.", py::arg("prior_model"));
        bindSensor<FBGSensor<QuasiStaticValues>, pPrior>(m, "FBGSensorQuasiStatic", "A strain sensor (e.g. FBGs) on a quasi-static robot.", py::arg("prior_model"));
        bindSensor<GyroSensor<WnoaValues>, pPrior>(m, "GyroSensorWNOA", "An angular velocity sensor on a WNOA robot.", py::arg("prior_model"));
        bindSensor<GyroSensor<EndEffectorValues>, pPrior>(m, "GyroSensorEndEffector", "An angular velocity sensor on an end effector.", py::arg("prior_model"));

        // The factors, on the interpolated keys (shared by WNOA, WNOV and end effectors for the pose).
        bindFactor<PoseSensor<WnoaValues>::Factor, Pose>(m, "PoseFactor", "The masked error Log(T^-1 T_meas) on the interpolated pose; `measured` is T_ib, `mask` 6 x 1 (0 ignores a component).");
        bindFactor<FBGSensor<WnoaValues>::Factor, Vector6>(m, "StrainFactor", "The masked error epsilon - epsilon_meas on a WNOA interpolated strain ([rot, trans]); `mask` 6 x 1.");
        bindFactor<GyroSensor<WnoaValues>::Factor, Eigen::Vector3d>(m, "GyroFactor",
                                                                    "The masked error w_meas - varpi[:3] on a WNOA interpolated velocity; `measured` is the body-frame angular velocity [rad/s], `mask` 3 x 1.");

        // The factories: the sensor class for the prior's node values.
        m.def(
            "PoseSensor",
            [](const pPrior &prior)
            {
                return dispatchOnPrior<WnoaPrior, WnovPrior, QuasiStaticPrior, EndEffectorPrior>(prior, "PoseSensor", [&](auto *tag)
                                                                               { return py::cast(std::make_shared<PoseSensor<std::remove_pointer_t<decltype(tag)>>>(prior)); });
            },
            py::arg("prior_model"), "A pose sensor on the robot (WNOA, WNOV, quasi-static) or end effector of `prior_model`; TypeError for another prior.");
        m.def(
            "FBGSensor",
            [](const pPrior &prior)
            {
                return dispatchOnPrior<WnoaPrior, QuasiStaticPrior>(prior, "FBGSensor", [&](auto *tag)
                                                  { return py::cast(std::make_shared<FBGSensor<std::remove_pointer_t<decltype(tag)>>>(prior)); });
            },
            py::arg("prior_model"), "A strain sensor on the WNOA or quasi-static robot of `prior_model`; TypeError for another prior.");
        m.def(
            "GyroSensor",
            [](const pPrior &prior)
            {
                return dispatchOnPrior<WnoaPrior, EndEffectorPrior>(prior, "GyroSensor", [&](auto *tag)
                                                                    { return py::cast(std::make_shared<GyroSensor<std::remove_pointer_t<decltype(tag)>>>(prior)); });
            },
            py::arg("prior_model"), "An angular velocity sensor on the WNOA robot or end effector of `prior_model`; TypeError for another prior.");
    }
} // namespace SpacetimePython
