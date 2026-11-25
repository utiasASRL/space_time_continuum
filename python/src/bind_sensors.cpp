/**
 * @file bind_sensors.cpp
 * @brief spacetime.sensors: the built-in sensors, for each node values type they support, and
 *        their factors (for Python sensors to reuse).
 */

#include <spacetime/estimator/bias_models/Bias.hpp>
#include <spacetime/estimator/sensor_models/AccelerometerSensor.hpp>
#include <spacetime/estimator/sensor_models/FBGSensor.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>
#include <spacetime/estimator/sensor_models/IMUSensor.hpp>
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
        template <typename PriorT>
        using AccelerometerSensor = Spacetime::Sensors::AccelerometerSensor<PriorT>;
        template <typename PriorT>
        using IMUSensor = Spacetime::Sensors::IMUSensor<PriorT>;
        using Spacetime::Sensors::AccelerometerOptions;
        using Spacetime::Sensors::GyroOptions;
        using Spacetime::Sensors::IMUOptions;
        using BiasOptions3 = Spacetime::Biases::BiasOptions<gtsam::Vector3>;
        using BiasModel3 = Spacetime::Biases::BiasModel<gtsam::Vector3>;
        using BiasNodeValues3 = Spacetime::Biases::BiasNodeValues<gtsam::Vector3>;

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

        /// make(typed) with `prior` cast to the first of the priors PriorT, Rest... that it is.
        template <typename PriorT, typename... Rest, typename Make>
        py::object dispatchOnPriorType(const pPrior &prior, const char *sensor, Make &&make)
        {
            if (auto typed = std::dynamic_pointer_cast<PriorT>(prior))
                return make(typed);
            if constexpr (sizeof...(Rest) > 0)
                return dispatchOnPriorType<Rest...>(prior, sensor, std::forward<Make>(make));
            else
                throw py::type_error(std::string(sensor) + ": no built-in sensor for this prior (write a Python SensorBase subclass instead).");
        }

        /// Binds a sensor's biasModel() accessor (None without a bias).
        template <typename Class>
        Class &defBiasModel(Class &cls)
        {
            using SensorT = typename Class::type;
            return cls.def("biasModel", [](const SensorT &self)
                           { return self.biasModel(); }, "The estimated bias (a BiasModel3), or None if it is not estimated.");
        }

        /// Binds AccelerometerSensor<PriorT> and IMUSensor<PriorT> as AccelerometerSensor<suffix> and IMUSensor<suffix>.
        template <typename PriorT>
        void bindAccelerometerAndIMU(py::module_ &m, const std::string &suffix, const std::string &on)
        {
            using Accel = AccelerometerSensor<PriorT>;
            using Imu = IMUSensor<PriorT>;
            auto accel = py::classh<Accel, PySensor<Accel>, SensorBase>(
                             m, ("AccelerometerSensor" + suffix).c_str(),
                             ("A 3-axis accelerometer at a fixed arclength of " + on + ": 3x1 specific-force samples [m/s^2] in its frame, "
                                                                                     "preintegrated between consecutive node rows.").c_str())
                             .def(py::init<std::shared_ptr<PriorT>, double, const AccelerometerOptions &>(), py::arg("prior_model"), py::arg("arclength"),
                                  py::arg("options") = AccelerometerOptions())
                             .def("arclength", &Accel::arclength, "The sensor's arclength [m].")
                             .def("options", &Accel::options, "The sensor's AccelerometerOptions.");
            defBiasModel(accel);
            py::classh<Imu, PySensor<Imu>, SensorBase>(
                m, ("IMUSensor" + suffix).c_str(),
                ("A 6-axis IMU at a fixed arclength of " + on + ": a gyroscope and an accelerometer sharing one frame. Measurements are "
                                                               "6x1 [angular velocity; specific force] with a 6x6 covariance (its diagonal 3x3 blocks are used).").c_str())
                .def(py::init<std::shared_ptr<PriorT>, double, const IMUOptions &>(), py::arg("prior_model"), py::arg("arclength"), py::arg("options") = IMUOptions())
                .def("arclength", &Imu::arclength, "The sensor's arclength [m].")
                .def("options", &Imu::options, "The sensor's IMUOptions.")
                .def("gyro", &Imu::gyro, "The wrapped gyroscope sensor (e.g. for its biasModel()).")
                .def("accelerometer", &Imu::accelerometer, "The wrapped accelerometer sensor (e.g. for its biasModel()).");
        }

        void bindBiases(py::module_ &m)
        {
            py::class_<BiasOptions3>(m, "BiasOptions3", "A 3-vector bias's prior and random walk (see BiasModel3).")
                .def(py::init<>())
                .def_readwrite("initial", &BiasOptions3::initial, "Prior mean of the first bias node, and the initial guess until one is estimated.")
                .def_readwrite("initial_covariance", &BiasOptions3::initial_covariance, "3x3 prior covariance of the first bias node.")
                .def_readwrite("random_walk_psd", &BiasOptions3::random_walk_psd, "3x3 power spectral density Q_c of the random walk: nodes dt apart differ by N(0, Q_c dt).");

            py::classh<BiasNodeValues3, Spacetime::Types::NodeValuesBase>(m, "BiasNodeValues3", "A 3-vector bias node's value.")
                .def(py::init<>())
                .def(py::init<const gtsam::Vector3 &>(), py::arg("bias"))
                .def("bias", py::overload_cast<>(&BiasNodeValues3::bias, py::const_), "The bias.");

            py::classh<BiasModel3, StatefulModel>(m, "BiasModel3",
                                                  "A sensor's slowly varying 3-vector bias (e.g. a gyroscope's or an accelerometer's), estimated as a random walk "
                                                  "in time: one bias node per estimation time. query(time, arclength, values) returns its BiasNodeValues3, "
                                                  "interpolated linearly between nodes. Owned by its sensor: do not add it to the estimator.")
                .def("options", &BiasModel3::options, "The model's BiasOptions3.");
        }
    } // namespace

    void bindSensors(py::module_ &m)
    {
        bindBiases(m);

        py::class_<GyroOptions>(m, "GyroOptions", "Parameters of a gyroscope sensor.")
            .def(py::init<>())
            .def_readwrite("offset", &GyroOptions::offset, "T_bs: the gyroscope in the robot's body frame (only its rotation is used); readings are in the sensor frame.")
            .def_readwrite("estimate_bias", &GyroOptions::estimate_bias, "Estimate a gyroscope bias (off by default).")
            .def_readwrite("bias", &GyroOptions::bias, "The bias's BiasOptions3, used with estimate_bias.")
            .def_static("defaultBias", &GyroOptions::defaultBias, "Zero initial bias, 0.01 rad/s prior std, 1e-4 rad/s/sqrt(s) random walk.");

        py::class_<AccelerometerOptions>(m, "AccelerometerOptions", "Parameters of an accelerometer sensor.")
            .def(py::init<>())
            .def_readwrite("offset", &AccelerometerOptions::offset, "T_bs: the accelerometer in the robot's body frame at its arclength.")
            .def_readwrite("use_slerp", &AccelerometerOptions::use_slerp, "Rotate each sample by SLERP between the interval's node rotations, instead of holding the first's.")
            .def_readwrite("gravity", &AccelerometerOptions::gravity, "Gravity in the inertial frame [m/s^2]; the accelerometer reads specific force.")
            .def_readwrite("estimate_bias", &AccelerometerOptions::estimate_bias, "Estimate an accelerometer bias (on by default).")
            .def_readwrite("bias", &AccelerometerOptions::bias, "The bias's BiasOptions3, used with estimate_bias.")
            .def_static("defaultBias", &AccelerometerOptions::defaultBias, "Zero initial bias, 0.1 m/s^2 prior std, 1e-3 m/s^2/sqrt(s) random walk.");

        py::class_<IMUOptions>(m, "IMUOptions", "Parameters of an IMU sensor: a gyroscope and an accelerometer sharing one frame.")
            .def(py::init<>())
            .def_readwrite("offset", &IMUOptions::offset, "T_bs: the IMU in the robot's body frame at its arclength (replaces gyro.offset and accelerometer.offset).")
            .def_readwrite("gyro", &IMUOptions::gyro, "The gyroscope's GyroOptions (estimates a bias by default).")
            .def_readwrite("accelerometer", &IMUOptions::accelerometer, "The accelerometer's AccelerometerOptions (estimates a bias by default).");

        bindSensor<PoseSensor<WnoaValues>, pPrior>(m, "PoseSensorWNOA", "A 6-DoF pose sensor (T_ib) on a WNOA robot (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<PoseSensor<WnovValues>, pPrior>(m, "PoseSensorWNOV", "A 6-DoF pose sensor (T_ib) on a WNOV robot (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<PoseSensor<QuasiStaticValues>, pPrior>(m, "PoseSensorQuasiStatic", "A 6-DoF pose sensor (T_ib) on a quasi-static robot (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<PoseSensor<EndEffectorNodeValues>, pPrior>(m, "PoseSensorEndEffector", "A 6-DoF pose sensor (T_ib) on an end effector (each measurement carries its covariance).", py::arg("prior_model"));
        bindSensor<FBGSensor<WnoaValues>, pPrior>(m, "FBGSensorWNOA", "A strain sensor (e.g. FBGs) on a WNOA robot.", py::arg("prior_model"));
        bindSensor<FBGSensor<QuasiStaticValues>, pPrior>(m, "FBGSensorQuasiStatic", "A strain sensor (e.g. FBGs) on a quasi-static robot.", py::arg("prior_model"));
        {
            auto wnoa = py::classh<GyroSensor<WnoaValues>, PySensor<GyroSensor<WnoaValues>>, SensorBase>(m, "GyroSensorWNOA", "An angular velocity sensor on a WNOA robot.")
                            .def(py::init<pPrior, const GyroOptions &>(), py::arg("prior_model"), py::arg("options") = GyroOptions())
                            .def("options", &GyroSensor<WnoaValues>::options, "The sensor's GyroOptions.");
            defBiasModel(wnoa);
            auto ee = py::classh<GyroSensor<EndEffectorNodeValues>, PySensor<GyroSensor<EndEffectorNodeValues>>, SensorBase>(m, "GyroSensorEndEffector",
                                                                                                                             "An angular velocity sensor on an end effector.")
                          .def(py::init<pPrior, const GyroOptions &>(), py::arg("prior_model"), py::arg("options") = GyroOptions())
                          .def("options", &GyroSensor<EndEffectorNodeValues>::options, "The sensor's GyroOptions.");
            defBiasModel(ee);
        }
        bindAccelerometerAndIMU<WnoaPrior>(m, "WNOA", "a WNOA robot");
        bindAccelerometerAndIMU<EndEffectorPrior>(m, "EndEffector", "an end effector");

        // The factors, on the interpolated keys (shared by WNOA, WNOV and end effectors for the pose).
        bindFactor<PoseSensor<WnoaValues>::Factor, Pose>(m, "PoseFactor", "The masked error Log(T^-1 T_meas) on the interpolated pose; `measured` is T_ib, `mask` 6 x 1 (0 ignores a component).");
        bindFactor<FBGSensor<WnoaValues>::Factor, Vector6>(m, "StrainFactor", "The masked error epsilon - epsilon_meas on a WNOA interpolated strain ([rot, trans]); `mask` 6 x 1.");
        py::class_<GyroSensor<WnoaValues>::Factor, gtsam::NoiseModelFactor, std::shared_ptr<GyroSensor<WnoaValues>::Factor>>(
            m, "GyroFactor", "The masked error w_meas - R_bs^T varpi[:3] on a WNOA interpolated velocity; `measured` is the sensor-frame angular velocity [rad/s], `mask` 3 x 1.")
            .def(py::init<const gtsam::SharedNoiseModel &, const Eigen::Vector3d &, const Eigen::MatrixXi &, const gtsam::Rot3 &>(), py::arg("noise_model"),
                 py::arg("measured"), py::arg("mask"), py::arg("R_bs") = gtsam::Rot3());
        py::class_<GyroSensor<WnoaValues>::BiasedFactor, gtsam::NoiseModelFactor, std::shared_ptr<GyroSensor<WnoaValues>::BiasedFactor>>(
            m, "GyroBiasedFactor",
            "The masked error w_meas - R_bs^T varpi[:3] - sum_i w_i b_i on a WNOA interpolated velocity and the bias keys of `bias_weights`, a list of "
            "(key, weight) pairs (each bias a 3-vector).")
            .def(py::init<const gtsam::SharedNoiseModel &, const Eigen::Vector3d &, const Eigen::MatrixXi &, const GyroSensor<WnoaValues>::BiasWeights &,
                          const gtsam::Rot3 &>(),
                 py::arg("noise_model"), py::arg("measured"), py::arg("mask"), py::arg("bias_weights"), py::arg("R_bs") = gtsam::Rot3());

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
            [](const pPrior &prior, const GyroOptions &options)
            {
                return dispatchOnPrior<WnoaPrior, EndEffectorPrior>(prior, "GyroSensor", [&](auto *tag)
                                                                    { return py::cast(std::make_shared<GyroSensor<std::remove_pointer_t<decltype(tag)>>>(prior, options)); });
            },
            py::arg("prior_model"), py::arg("options") = GyroOptions(),
            "An angular velocity sensor (3x1 measurements [rad/s], sensor frame) on the WNOA robot or end effector of `prior_model`, estimating a bias "
            "if options.estimate_bias; TypeError for another prior.");
        m.def(
            "AccelerometerSensor",
            [](const pPrior &prior, double arclength, const AccelerometerOptions &options)
            {
                return dispatchOnPriorType<WnoaPrior, EndEffectorPrior>(prior, "AccelerometerSensor", [&](auto typed)
                                                                        {
                    using PriorT = typename decltype(typed)::element_type;
                    return py::cast(std::make_shared<AccelerometerSensor<PriorT>>(typed, arclength, options)); });
            },
            py::arg("prior_model"), py::arg("arclength"), py::arg("options") = AccelerometerOptions(),
            "An accelerometer (3x1 specific-force samples [m/s^2], sensor frame) at `arclength` on the WNOA robot or end effector of `prior_model`; "
            "TypeError for another prior, ValueError for an arclength outside the robot.");
        m.def(
            "IMUSensor",
            [](const pPrior &prior, double arclength, const IMUOptions &options)
            {
                return dispatchOnPriorType<WnoaPrior, EndEffectorPrior>(prior, "IMUSensor", [&](auto typed)
                                                                        {
                    using PriorT = typename decltype(typed)::element_type;
                    return py::cast(std::make_shared<IMUSensor<PriorT>>(typed, arclength, options)); });
            },
            py::arg("prior_model"), py::arg("arclength"), py::arg("options") = IMUOptions(),
            "An IMU (6x1 [angular velocity; specific force] measurements, sensor frame) at `arclength` on the WNOA robot or end effector of "
            "`prior_model`, wrapping a gyroscope and an accelerometer (both estimating a bias by default); TypeError for another prior, "
            "ValueError for an arclength outside the robot.");
    }
} // namespace SpacetimePython
