/**
 * @file ConfigLoader.hpp
 * @brief JSON configuration of the estimator, priors, demo sensor data and visualizer
 *        (sections "estimator", "prior", "demo_data", "visualizer").
 * @author Spencer Teetaert
 */

#ifndef CONFIGLOADER_HPP
#define CONFIGLOADER_HPP

#include <utility>
#include <vector>
#include <glob.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <optional>
#include <variant>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <json/json.h>
#include <json/value.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/utilities/WindowHistory.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/prior_models/WNOA.hpp>
#include <spacetime/estimator/prior_models/WNOV.hpp>
#include <spacetime/estimator/prior_models/WNOAQuasiStatic.hpp>
#include <spacetime/estimator/utilities/logger.hpp>
#include <spacetime/estimator/utilities/functions.hpp>
#include <spacetime/estimator/sensor_models/PoseSensor.hpp>
#include <spacetime/estimator/sensor_models/GyroSensor.hpp>
#include <spacetime/estimator/sensor_models/AccelerometerSensor.hpp>
#include <spacetime/estimator/sensor_models/FBGSensor.hpp>
#include <spacetime/visualizer/VisualizerOptions.hpp>

namespace Spacetime::Utilities
{
    /**
     * @brief Reads estimator, prior and demo-data settings from one or more JSON config
     *        files, merged into a single root. Vectors and covariance diagonals in the
     *        config are in GTSAM [rotational, translational] order and 4x4 matrices are
     *        row-major; only the sensor data files (see loadData) use the legacy layout.
     */
    class ConfigLoader
    {
        Json::Value root = Json::Value(Json::objectValue); // Initialize as an empty JSON object

    public:
        /// An empty config.
        ConfigLoader() = default;

        /**
         * @brief Parse a JSON file and merge its top-level sections into the config.
         * @return false if the file cannot be opened or parsed.
         * @throws std::runtime_error if a top-level section is already loaded (see mergeJsonInto).
         * @throws std::invalid_argument if the file's top level is not a JSON object.
         */
        bool loadConfig(std::string filename)
        {
            // info() << "Loading configuration file: " << filename;
            std::cout << "Loading configuration file: " << filename << std::endl;
            std::ifstream config_file(filename, std::ifstream::binary);
            Json::CharReaderBuilder builder;
            JSONCPP_STRING errs;

            Json::Value _root;

            if (!parseFromStream(builder, config_file, &_root, &errs))
            {
                // error() << errs;
                std::cerr << errs << std::endl;
                return false;
            }
            addConfig(_root);
            // root.append(_root); // TODO: This merging logic is wrong
            // error() << "THIS MERGING LOGIC IS BROKEN!!!";
            // return false;
            info() << "Configuration file added: " << filename;
            std::cout << "Configuration file added: " << filename << std::endl;
            return true;
        }

        /// Adds the top-level sections of a config built in memory, as loadConfig does a file's. Throws as mergeJsonInto.
        void addConfig(const Json::Value &config) { mergeJsonInto(root, config); }

        /**
         * @brief Copy each of `source`'s top-level members into `target` (shallow: no
         *        recursive merging of sections).
         * @throws std::invalid_argument if either is not a JSON object.
         * @throws std::runtime_error if a key is already in `target`.
         */
        void mergeJsonInto(Json::Value &target, const Json::Value &source)
        {
            if (!source.isObject() || !target.isObject())
            {
                throw std::invalid_argument("mergeJsonInto: both values must be JSON objects");
            }

            for (const auto &key : source.getMemberNames())
            {
                if (target.isMember(key))
                {
                    throw std::runtime_error("mergeJsonInto: key collision on \"" + key + "\"");
                }
                target[key] = source[key];
            }
        }

        /**
         * @brief Read the "estimator" section: the smoother lag, the sliding-window smoother
         *        ("smoother": "batch" or "incremental"), the Levenberg-Marquardt parameters
         *        ("lm_params"), the iSAM2 parameters ("isam2_params": relinearizeThreshold,
         *        relinearizeSkip), whether to extract posterior covariances
         *        ("compute_covariance") and whether to estimate only at measurement times
         *        ("quasi_static"). Missing entries keep EstimatorOptions' defaults.
         * @throws std::invalid_argument on an unknown smoother or lm_params.orderingType.
         */
        EstimatorOptions loadEstimatorOptions() const
        {
            EstimatorOptions options;
            if (root.isMember("smootherLag") || root.isMember("lm_params"))
                warning() << "Top-level \"smootherLag\"/\"lm_params\" are no longer read; move them into the \"estimator\" section.";

            Json::Value estimator_root;
            if (!safeRead(root, "estimator", estimator_root))
            {
                warning() << "Estimator options (estimator) not found in config. Using default values.";
                return options;
            }
            safeRead(estimator_root, "smootherLag", options.smootherLag);
            safeRead(estimator_root, "compute_covariance", options.compute_covariance);
            safeRead(estimator_root, "quasi_static", options.quasi_static);
            std::string smoother;
            if (safeRead(estimator_root, "smoother", smoother))
            {
                if (smoother == "batch")
                    options.smoother = EstimatorOptions::Smoother::Batch;
                else if (smoother == "incremental")
                    options.smoother = EstimatorOptions::Smoother::Incremental;
                else
                    throw std::invalid_argument("estimator.smoother must be \"batch\" or \"incremental\", got \"" + smoother + "\".");
            }
            Json::Value isam2_root;
            if (safeRead(estimator_root, "isam2_params", isam2_root))
            {
                const double *current = std::get_if<double>(&options.isam2_params.relinearizeThreshold);
                double threshold = current ? *current : 0.0; // only logged if absent
                if (safeRead(isam2_root, "relinearizeThreshold", threshold))
                    options.isam2_params.relinearizeThreshold = threshold;
                safeRead(isam2_root, "relinearizeSkip", options.isam2_params.relinearizeSkip);
            }
            options.lm_params = readLMParams(estimator_root);
            return options;
        }

        /**
         * @brief Read which estimate is exported once a run ends, "estimator.extraction":
         *        "final" (the default) or "back" (see Utilities::Extraction).
         * @throws std::invalid_argument on any other value.
         */
        Extraction loadExtraction() const
        {
            std::string name = "final";
            Json::Value estimator_root;
            if (safeRead(root, "estimator", estimator_root))
                safeRead(estimator_root, "extraction", name);
            return parseExtraction(name);
        }

        /**
         * @brief Read which prior model to use, "prior.type": "wnoa", "wnov" or
         *        "wnoa_quasi_static". Each model's parameters live in its own
         *        "prior.<type>" subsection (see loadWNOAPrior and loadWNOVPrior), except
         *        the quasi-static prior's, which are read from "prior.wnoa" (see
         *        loadWNOAQuasiStaticPrior), so models are swapped by editing the type alone.
         * @return "wnoa" if the section or its type is missing (with a warning).
         * @throws std::invalid_argument on an unknown type.
         */
        std::string loadPriorType() const
        {
            if (root.isMember("WNOA"))
                warning() << "The top-level \"WNOA\" section is no longer read; move it into \"prior.wnoa\" (with \"time_period\" in \"prior\").";

            std::string type = "wnoa";
            Json::Value prior_root;
            if (!safeRead(root, "prior", prior_root) || !safeRead(prior_root, "type", type))
            {
                warning() << "Prior type (prior.type) not found in config. Defaulting to \"wnoa\".";
                return "wnoa";
            }
            if (type != "wnoa" && type != "wnov" && type != "wnoa_quasi_static")
                throw std::invalid_argument("prior.type must be \"wnoa\", \"wnov\" or \"wnoa_quasi_static\", got \"" + type + "\".");
            return type;
        }

        /**
         * @brief Read the WNOA prior from "prior": "time_period", and from "prior.wnoa",
         *        "weights" (diagonals "Q0" (18: [pose, epsilon, varpi]) and "Q1".."Q4" (6 each;
         *        "Q4" defaults to "Q3"), all scaled by an optional "prior_factor") and
         *        "initial_condition" ("T0", a row-major 4x4 T_ib, "epsilon0" and "varpi0").
         *        Vectors are in GTSAM [rotational, translational] order. A missing section
         *        keeps the values passed in.
         * @warning Within a present "weights" / "initial_condition" section, every entry
         *          (but Q4) must be given with the right length: a missing one leaves its
         *          output set from an uninitialized vector, and sizes are not checked
         *          (unlike loadWNOVPrior's loadSizedArray).
         */
        template <typename Pose>
        void loadWNOAPrior(Spacetime::Priors::WNOA::Values<Pose> &base_start_point, double &time_period, Spacetime::Priors::WNOA::Prior<Pose>::Matrix3N &Q0, Spacetime::Priors::WNOA::Prior<Pose>::MatrixN &Q1, Spacetime::Priors::WNOA::Prior<Pose>::MatrixN &Q2, Spacetime::Priors::WNOA::Prior<Pose>::MatrixN &Q3, Spacetime::Priors::WNOA::Prior<Pose>::MatrixN &Q4) const
        {
            Json::Value prior_root;
            if (!safeRead(root, "prior", prior_root))
                return;
            safeRead(prior_root, "time_period", time_period);

            Json::Value wnoa_root;
            if (readPriorSubsection("wnoa", wnoa_root))
            {
                Json::Value weights_root;
                if (safeRead(wnoa_root, "weights", weights_root))
                {
                    // Prior noise
                    double prior_factor = 1.0;
                    safeRead(weights_root, "prior_factor", prior_factor);

                    Eigen::Matrix<double, 6, 1> _Q1, _Q2, _Q3, _Q4;
                    loadArrayData(weights_root, "Q1", _Q1);
                    Q1 = prior_factor * _Q1.asDiagonal();
                    loadArrayData(weights_root, "Q2", _Q2);
                    Q2 = prior_factor * _Q2.asDiagonal();
                    loadArrayData(weights_root, "Q3", _Q3);
                    Q3 = prior_factor * _Q3.asDiagonal();
                    if (!loadArrayData(weights_root, "Q4", _Q4))
                    {
                        info() << "Q4 not found in config. Defaulting to Q3 values.";
                        _Q4 = _Q3;
                    }
                    Q4 = prior_factor * _Q4.asDiagonal();

                    Eigen::Matrix<double, 18, 1> _Q0;
                    loadArrayData(weights_root, "Q0", _Q0);
                    Q0 = prior_factor * _Q0.asDiagonal();
                }

                Json::Value initial_condition_root;
                if (safeRead(wnoa_root, "initial_condition", initial_condition_root))
                {
                    // Load initial condition values
                    Eigen::Matrix<double, 4, 4> T0;
                    Eigen::Matrix<double, 6, 1> epsilon0, varpi0;
                    loadMatrix4Data(initial_condition_root, "T0", T0);
                    loadArrayData(initial_condition_root, "epsilon0", epsilon0);
                    loadArrayData(initial_condition_root, "varpi0", varpi0);
                    base_start_point.pose(Pose(T0));
                    base_start_point.epsilon(epsilon0);
                    base_start_point.varpi(varpi0);
                }

                if (wnoa_root.isMember("time_period"))
                    warning() << "prior.wnoa.time_period is not read; set prior.time_period instead.";
            }
        }

        /**
         * @brief Read the WNOA prior's options from the "prior.wnoa" section: "factor_type"
         *        ("wnoa", "legacy" or "experimental"), "fixed_noise_model", "cache_noise_model" and "remove_unary".
         *        Missing entries keep PriorOptions' defaults.
         * @throws std::invalid_argument on an unknown factor_type.
         */
        Spacetime::Priors::WNOA::PriorOptions loadWNOAPriorOptions() const
        {
            Spacetime::Priors::WNOA::PriorOptions options;
            Json::Value wnoa_root;
            if (!readPriorSubsection("wnoa", wnoa_root))
                return options;

            std::string factor_type;
            if (safeRead(wnoa_root, "factor_type", factor_type))
            {
                if (factor_type == "wnoa")
                    options.factor_type = Spacetime::Priors::WNOA::FactorType::WNOA;
                else if (factor_type == "legacy")
                    options.factor_type = Spacetime::Priors::WNOA::FactorType::Legacy;
                else if (factor_type == "experimental")
                    options.factor_type = Spacetime::Priors::WNOA::FactorType::Experimental;
                else if (factor_type == "paper")
                    throw std::invalid_argument("prior.wnoa.factor_type \"paper\" has been renamed to \"wnoa\" (the old \"wnoa\" factors are now \"experimental\").");
                else
                    throw std::invalid_argument("prior.wnoa.factor_type must be \"wnoa\", \"legacy\" or \"experimental\", got \"" + factor_type + "\".");
            }
            if (wnoa_root.isMember("lock_first_pose"))
                warning() << "prior.wnoa.lock_first_pose is not read; use demo_data.locks instead.";
            safeRead(wnoa_root, "fixed_noise_model", options.fixed_noise_model);
            safeRead(wnoa_root, "cache_noise_model", options.cache_noise_model);
            safeRead(wnoa_root, "remove_unary", options.remove_unary);
            return options;
        }

        /**
         * @brief Read the quasi-static prior: the arclength part of the WNOA prior's section
         *        (see loadWNOAPrior), so that one config gives both priors the same spatial
         *        noise -- "prior.time_period", the pose and strain blocks of
         *        "prior.wnoa.weights.Q0", "Q2", and "T0" and "epsilon0" of
         *        "prior.wnoa.initial_condition". A missing section keeps the values passed in.
         */
        template <typename Pose>
        void loadWNOAQuasiStaticPrior(Spacetime::Priors::WNOAQuasiStatic::Values<Pose> &base_start_point, double &time_period, typename Spacetime::Priors::WNOAQuasiStatic::Prior<Pose>::Matrix2N &Q0, typename Spacetime::Priors::WNOAQuasiStatic::Prior<Pose>::MatrixN &Q2) const
        {
            using WNOAPrior = Spacetime::Priors::WNOA::Prior<Pose>;
            Spacetime::Priors::WNOA::Values<Pose> wnoa_start_point;
            wnoa_start_point.pose(base_start_point.pose());
            wnoa_start_point.epsilon(base_start_point.epsilon());
            typename WNOAPrior::Matrix3N wnoa_Q0 = WNOAPrior::Matrix3N::Identity();
            wnoa_Q0.template topLeftCorner<2 * WNOAPrior::dim, 2 * WNOAPrior::dim>() = Q0;
            typename WNOAPrior::MatrixN unused_Q1, unused_Q3, unused_Q4;
            loadWNOAPrior<Pose>(wnoa_start_point, time_period, wnoa_Q0, unused_Q1, Q2, unused_Q3, unused_Q4);

            base_start_point = Spacetime::Priors::WNOAQuasiStatic::Values<Pose>(wnoa_start_point.pose(), wnoa_start_point.epsilon());
            Q0 = wnoa_Q0.template topLeftCorner<2 * WNOAPrior::dim, 2 * WNOAPrior::dim>();
        }

        /**
         * @brief Read the quasi-static prior's options from the "prior.wnoa" section:
         *        "fixed_noise_model" and "cache_noise_model" (see loadWNOAPriorOptions).
         *        Missing entries keep PriorOptions' defaults.
         */
        Spacetime::Priors::WNOAQuasiStatic::PriorOptions loadWNOAQuasiStaticPriorOptions() const
        {
            Spacetime::Priors::WNOAQuasiStatic::PriorOptions options;
            Json::Value wnoa_root;
            if (readPriorSubsection("wnoa", wnoa_root))
            {
                safeRead(wnoa_root, "fixed_noise_model", options.fixed_noise_model);
                safeRead(wnoa_root, "cache_noise_model", options.cache_noise_model);
            }
            return options;
        }

        /**
         * @brief Read the WNOV prior from "prior": "time_period", and from "prior.wnov",
         *        "weights" ("Q0", "Q1", "Q2", each the diagonal of a covariance / PSD in GTSAM
         *        [rotational, translational] order, all scaled by an optional
         *        "prior_factor") and "initial_condition" ("T0", a row-major 4x4 T_ib, and
         *        "epsilon0", the nominal strain). Missing entries keep the values passed in.
         * @throws std::invalid_argument if an entry has the wrong number of values.
         */
        template <typename Pose>
        void loadWNOVPrior(Pose &T0, typename gtsam::traits<Pose>::TangentVector &epsilon0, double &time_period,
                           typename Spacetime::Priors::WNOV::Prior<Pose>::MatrixN &Q0,
                           typename Spacetime::Priors::WNOV::Prior<Pose>::MatrixN &Q1,
                           typename Spacetime::Priors::WNOV::Prior<Pose>::MatrixN &Q2) const
        {
            using TangentVector = typename gtsam::traits<Pose>::TangentVector;
            Json::Value prior_root;
            if (!safeRead(root, "prior", prior_root))
                return;
            safeRead(prior_root, "time_period", time_period);

            Json::Value wnov_root;
            if (!readPriorSubsection("wnov", wnov_root))
                return;

            Json::Value weights_root;
            if (safeRead(wnov_root, "weights", weights_root))
            {
                double prior_factor = 1.0;
                safeRead(weights_root, "prior_factor", prior_factor);
                TangentVector diagonal;
                if (loadSizedArray(weights_root, "Q0", "prior.wnov.weights", diagonal))
                    Q0 = prior_factor * diagonal.asDiagonal();
                if (loadSizedArray(weights_root, "Q1", "prior.wnov.weights", diagonal))
                    Q1 = prior_factor * diagonal.asDiagonal();
                if (loadSizedArray(weights_root, "Q2", "prior.wnov.weights", diagonal))
                    Q2 = prior_factor * diagonal.asDiagonal();
            }

            Json::Value initial_condition_root;
            if (safeRead(wnov_root, "initial_condition", initial_condition_root))
            {
                Eigen::Matrix<double, 16, 1> T0_row_major;
                if (loadSizedArray(initial_condition_root, "T0", "prior.wnov.initial_condition", T0_row_major))
                    T0 = Pose(Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T0_row_major.data()).eval());
                loadSizedArray(initial_condition_root, "epsilon0", "prior.wnov.initial_condition", epsilon0);
            }
        }

        /**
         * @brief Read the WNOV prior's options from the "prior.wnov" section:
         *        "fixed_noise_model", "cache_noise_model" and "remove_unary". Missing entries keep
         *        PriorOptions' defaults.
         */
        Spacetime::Priors::WNOV::PriorOptions loadWNOVPriorOptions() const
        {
            Spacetime::Priors::WNOV::PriorOptions options;
            Json::Value wnov_root;
            if (readPriorSubsection("wnov", wnov_root))
            {
                safeRead(wnov_root, "fixed_noise_model", options.fixed_noise_model);
                safeRead(wnov_root, "cache_noise_model", options.cache_noise_model);
                safeRead(wnov_root, "remove_unary", options.remove_unary);
            }
            return options;
        }

        /**
         * @brief Apply the locks listed in "demo_data.locks" to `state` (see
         *        Types::ContinuumRobotState::lock). Each entry is
         *        {"type": "pose" | "strain" | "velocity", "s": arclength (-1: tip),
         *         "t": time (optional; omitted: every time), "value": ...}, where the value
         *        of a pose is a row-major 4x4 T_ib (16 numbers) and that of a strain or
         *        velocity is a 6-vector in GTSAM [rotational, translational] order.
         * @throws std::invalid_argument on a malformed entry, or on a variable the node
         *         type does not have (e.g. a strain lock under the pose-only WNOV prior).
         */
        template <typename NodeT>
        void loadLocks(Spacetime::Types::ContinuumRobotState<NodeT> &state) const
        {
            using Pose = typename NodeT::ValuesType::PoseType;
            Json::Value demo_data_root;
            if (!safeRead(root, "demo_data", demo_data_root) || !demo_data_root.isMember("locks"))
                return;
            const Json::Value &locks = demo_data_root["locks"];
            if (!locks.isArray())
                throw std::invalid_argument("demo_data.locks must be an array.");

            for (const Json::Value &entry : locks)
            {
                if (!entry.isMember("type") || !entry.isMember("s") || !entry.isMember("value") || !entry["value"].isArray())
                    throw std::invalid_argument("Each demo_data.locks entry needs \"type\", \"s\" and an array \"value\".");
                const std::string type = entry["type"].asString();
                const double s = entry["s"].asDouble();
                const std::optional<double> t = entry.isMember("t") ? std::optional<double>(entry["t"].asDouble()) : std::nullopt;
                const Json::Value &value = entry["value"];

                if (type == "pose")
                {
                    if (value.size() != 16)
                        throw std::invalid_argument("A pose lock value must be a row-major 4x4 matrix (16 numbers).");
                    Eigen::Matrix4d T;
                    for (unsigned int i = 0; i < 16; i++)
                        T(i / 4, i % 4) = value[i].asDouble();
                    state.lock(type, s, Pose(T), t);
                }
                else
                {
                    using TangentVector = typename gtsam::traits<Pose>::TangentVector;
                    if (value.size() != static_cast<unsigned int>(TangentVector::RowsAtCompileTime))
                        throw std::invalid_argument("A \"" + type + "\" lock value must have " + std::to_string(TangentVector::RowsAtCompileTime) + " numbers.");
                    TangentVector v;
                    for (unsigned int i = 0; i < value.size(); i++)
                        v(i) = value[i].asDouble();
                    state.lock(type, s, v, t);
                }
            }
        }

        /**
         * @brief Read the time window to estimate over, "demo_data.start_time" and
         *        "demo_data.end_time". Missing entries keep the values passed in.
         */
        void loadDemoTimeWindow(double &start_time, double &end_time) const
        {
            Json::Value demo_data_root;
            if (safeRead(root, "demo_data", demo_data_root))
            {
                safeRead(demo_data_root, "start_time", start_time);
                safeRead(demo_data_root, "end_time", end_time);
            }
        }

        /**
         * @brief Read just the robot topology metadata (length, node count) needed
         *        to construct a robot/prior, independently of loading the sensor
         *        measurement files themselves. Sensors are now bound to their prior
         *        model at construction (see Types::SensorBase), so callers must
         *        build the robot and prior from this metadata BEFORE calling
         *        loadDemoData to construct the sensor objects.
         */
        void loadDemoMetadata(double &length, unsigned int &N) const
        {
            Json::Value demo_data_root;
            if (safeRead(root, "demo_data", demo_data_root))
            {
                safeRead(demo_data_root, "robot_length", length);
                safeRead(demo_data_root, "robot_N", N);
            }
        }

        /**
         * @brief Load the demo's sensor data ("demo_data"), building each sensor for the
         *        node values of `prior_model`'s own type (PriorT::ValuesType). Sensors the
         *        node type cannot support -- a gyroscope or accelerometer without a velocity in the
         *        state, e.g. under the WNOV prior -- are skipped with a warning.
         *
         *        Reads "folder_path" (every *.csv in it, see loadData), "start_time" /
         *        "end_time", the "include_aurora" / "include_gyro" / "include_strain" /
         *        "include_accelerometer" flags and "weights": covariance diagonals "R_pose" (alias
         *        "R_aurora"), "R_gyro", "R_strain" and "R_accelerometer" (identity if absent), all
         *        scaled by "measurement_factor". The "accelerometer" section sets every accelerometer's
         *        Sensors::AccelerometerOptions (defaults if absent): "use_slerp", "gravity" [3],
         *        "offset" (T_bs as a 6-vector Log in the data's legacy [translational,
         *        rotational] order) and "bias": {"estimate", "initial" [3], "initial_sigma"
         *        [3], "random_walk_sigma" [3]} (standard deviations, squared into the
         *        covariance and PSD).
         * @throws std::invalid_argument if an "accelerometer" or "R_accelerometer" array has the wrong length.
         * @tparam PriorT The prior the sensors are bound to (e.g. Priors::WNOA::Prior<gtsam::Pose3>).
         * @param sensors Receives the measurement sensors (appended to).
         * @param gt Receives the sensors whose file is flagged as ground truth (appended to).
         * @param prior_model The prior every sensor resolves its state through (see Types::SensorBase).
         *        Build it, and its robot, from loadDemoMetadata first.
         */
        template <typename PriorT>
            requires requires { typename PriorT::ValuesType; }
        void loadDemoData(
            std::vector<std::shared_ptr<Spacetime::Types::SensorBase>> &sensors,
            std::vector<std::shared_ptr<Spacetime::Types::SensorBase>> &gt,
            const std::shared_ptr<PriorT> &prior_model) const
        {
            Json::Value demo_data_root;
            if (safeRead(root, "demo_data", demo_data_root))
            {
                // Sensor selection
                bool include_aurora = false, include_gyro = false, include_strain = false, include_accelerometer = false;
                safeRead(demo_data_root, "include_aurora", include_aurora);
                safeRead(demo_data_root, "include_gyro", include_gyro);
                safeRead(demo_data_root, "include_strain", include_strain);
                safeRead(demo_data_root, "include_accelerometer", include_accelerometer);

                // Load data options
                std::string data_dir;
                safeRead(demo_data_root, "folder_path", data_dir);
                double start_time = 0.0, end_time = 0.0;
                safeRead(demo_data_root, "start_time", start_time);
                safeRead(demo_data_root, "end_time", end_time);

                // Covariances are given in the config as their diagonals, in GTSAM
                // [rotational, translational] order (they are not reordered like the data)
                Eigen::Matrix<double, 6, 6> R_aurora = Eigen::Matrix<double, 6, 6>::Identity();
                Eigen::Matrix<double, 6, 6> R_strain = Eigen::Matrix<double, 6, 6>::Identity();
                Eigen::Matrix<double, 3, 3> R_gyro = Eigen::Matrix<double, 3, 3>::Identity();
                Eigen::Matrix<double, 3, 3> R_accelerometer = Eigen::Matrix<double, 3, 3>::Identity();
                Json::Value weights_root;
                if (safeRead(demo_data_root, "weights", weights_root))
                {
                    double measurement_factor = 1.0;
                    safeRead(weights_root, "measurement_factor", measurement_factor);
                    Eigen::Matrix<double, 6, 1> R_pose_diag = R_aurora.diagonal(), R_strain_diag = R_strain.diagonal();
                    Eigen::Matrix<double, 3, 1> R_gyro_diag = R_gyro.diagonal();
                    if (loadArrayData(weights_root, "R_pose", R_pose_diag) || loadArrayData(weights_root, "R_aurora", R_pose_diag))
                        R_aurora = R_pose_diag.asDiagonal();
                    if (loadArrayData(weights_root, "R_gyro", R_gyro_diag))
                        R_gyro = R_gyro_diag.asDiagonal();
                    if (loadArrayData(weights_root, "R_strain", R_strain_diag))
                        R_strain = R_strain_diag.asDiagonal();
                    Eigen::Vector3d R_accelerometer_diag = R_accelerometer.diagonal();
                    if (loadSizedArray(weights_root, "R_accelerometer", "demo_data.weights", R_accelerometer_diag))
                        R_accelerometer = R_accelerometer_diag.asDiagonal();
                    R_aurora *= measurement_factor;
                    R_gyro *= measurement_factor;
                    R_strain *= measurement_factor;
                    R_accelerometer *= measurement_factor;
                }

                const Spacetime::Sensors::AccelerometerOptions accelerometer_options = readAccelerometerOptions(demo_data_root);

                // Load each sensor file into new sensor objects and add to sensors vector
                if (!loadData(data_dir, start_time, end_time, sensors, gt, prior_model, include_aurora, include_gyro, include_strain, R_aurora, R_gyro, R_strain,
                              include_accelerometer, R_accelerometer, accelerometer_options))
                    warning() << "No demo data loaded from: " << data_dir;
            }
        }

        /**
         * @brief Read the visualizer options from the "visualizer" section: the camera
         *        views ("cameras", a list of {"position", "look_at", "up"}, each a 3-vector
         *        in the world frame; the first is the interactive view, the rest only take
         *        extra screenshots) and the robot views' tube radius ("robot.radius" [m]).
         *        Missing entries keep VisualizerOptions' defaults; the other legacy entries
         *        of the section are not read.
         * @throws std::invalid_argument if "cameras" is not a list of objects, a camera
         *         vector is not 3 numbers, or the radius is not positive.
         */
        Spacetime::Visualization::VisualizerOptions loadVisualizerOptions() const
        {
            Spacetime::Visualization::VisualizerOptions options;
            Json::Value visualizer_root;
            if (!safeRead(root, "visualizer", visualizer_root))
            {
                warning() << "Visualizer options (visualizer) not found in config. Using default values.";
                return options;
            }

            Json::Value cameras_root;
            if (safeRead(visualizer_root, "cameras", cameras_root))
            {
                if (!cameras_root.isArray())
                    throw std::invalid_argument("visualizer.cameras must be a list of cameras.");
                for (Json::ArrayIndex i = 0; i < cameras_root.size(); i++)
                {
                    const std::string section = "visualizer.cameras[" + std::to_string(i) + "]";
                    if (!cameras_root[i].isObject())
                        throw std::invalid_argument(section + " must be an object.");
                    Spacetime::Visualization::VisualizerOptions::CameraOptions camera;
                    loadSizedArray(cameras_root[i], "position", section, camera.position);
                    loadSizedArray(cameras_root[i], "look_at", section, camera.focal_point);
                    loadSizedArray(cameras_root[i], "up", section, camera.view_up);
                    options.cameras.push_back(camera);
                }
            }

            Json::Value robot_root;
            if (safeRead(visualizer_root, "robot", robot_root) && safeRead(robot_root, "radius", options.robot.radius) && !(options.robot.radius > 0.0))
                throw std::invalid_argument("visualizer.robot.radius must be positive.");
            return options;
        }

    private:
        /// The "prior.<type>" subsection, if present.
        bool readPriorSubsection(const std::string &type, Json::Value &out) const
        {
            Json::Value prior_root;
            return safeRead(root, "prior", prior_root) && safeRead(prior_root, type, out);
        }

        /**
         * @brief Read the fixed-size array `section.key` into `out`, if present.
         * @return false if the entry is absent.
         * @throws std::invalid_argument if it is not an array of exactly out.size() numbers
         *         (loadArrayData would read out of bounds on a short array).
         */
        template <typename Derived>
        static bool loadSizedArray(const Json::Value &section_root, const std::string &key, const std::string &section, Eigen::MatrixBase<Derived> &out)
        {
            if (!section_root.isMember(key))
                return false;
            const Json::Value &array = section_root[key];
            if (!array.isArray() || array.size() != static_cast<Json::ArrayIndex>(out.size()))
                throw std::invalid_argument(section + "." + key + " must be an array of " + std::to_string(out.size()) + " numbers.");
            for (Json::ArrayIndex i = 0; i < array.size(); i++)
                out(static_cast<Eigen::Index>(i)) = array[i].asDouble();
            return true;
        }

        /// The "accelerometer" section of `demo_data_root` (see loadDemoData); AccelerometerOptions' defaults if absent.
        static Spacetime::Sensors::AccelerometerOptions readAccelerometerOptions(const Json::Value &demo_data_root)
        {
            Spacetime::Sensors::AccelerometerOptions options;
            Json::Value accelerometer_root;
            if (!safeRead(demo_data_root, "accelerometer", accelerometer_root))
                return options;

            safeRead(accelerometer_root, "use_slerp", options.use_slerp);
            Eigen::Vector3d gravity = options.gravity;
            if (loadSizedArray(accelerometer_root, "gravity", "demo_data.accelerometer", gravity))
                options.gravity = gravity;
            Eigen::Matrix<double, 6, 1> offset;
            if (loadSizedArray(accelerometer_root, "offset", "demo_data.accelerometer", offset))
            {
                // Legacy [translational, rotational] order, as the pose data; GTSAM's is [rotational, translational]
                Eigen::Matrix<double, 6, 1> xi;
                xi << offset.tail<3>(), offset.head<3>();
                options.offset = gtsam::Pose3::Expmap(xi);
            }

            Json::Value bias_root;
            if (safeRead(accelerometer_root, "bias", bias_root))
            {
                safeRead(bias_root, "estimate", options.estimate_bias);
                Eigen::Vector3d initial = options.bias.initial, sigma;
                if (loadSizedArray(bias_root, "initial", "demo_data.accelerometer.bias", initial))
                    options.bias.initial = initial;
                if (loadSizedArray(bias_root, "initial_sigma", "demo_data.accelerometer.bias", sigma))
                    options.bias.initial_covariance = sigma.cwiseProduct(sigma).asDiagonal();
                if (loadSizedArray(bias_root, "random_walk_sigma", "demo_data.accelerometer.bias", sigma))
                    options.bias.random_walk_psd = sigma.cwiseProduct(sigma).asDiagonal();
            }
            return options;
        }

        /**
         * @brief LM parameters from "lm_params" (keys named as the gtsam fields, the ordering
         *        "orderingType" as "COLAMD", "METIS" or "NATURAL"); EstimatorOptions'
         *        defaults if absent.
         * @throws std::invalid_argument on an unknown orderingType.
         */
        static gtsam::LevenbergMarquardtParams readLMParams(const Json::Value &estimator_root)
        {
            gtsam::LevenbergMarquardtParams params = EstimatorOptions::defaultLMParams();

            Json::Value lm_params_root;
            if (safeRead(estimator_root, "lm_params", lm_params_root))
            {
                // LM specific parameters
                safeRead(lm_params_root, "lambdaInitial", params.lambdaInitial);
                safeRead(lm_params_root, "lambdaFactor", params.lambdaFactor);
                safeRead(lm_params_root, "lambdaUpperBound", params.lambdaUpperBound);
                safeRead(lm_params_root, "lambdaLowerBound", params.lambdaLowerBound);
                std::string verbosityLM_str;
                if (safeRead(lm_params_root, "verbosityLM", verbosityLM_str))
                {
                    params.setVerbosityLM(verbosityLM_str);
                }
                safeRead(lm_params_root, "minModelFidelity", params.minModelFidelity);
                safeRead(lm_params_root, "logFile", params.logFile);
                safeRead(lm_params_root, "useFixedLambdaFactor", params.useFixedLambdaFactor);
                safeRead(lm_params_root, "dampingParams.diagonalDamping", params.dampingParams.diagonalDamping);
                safeRead(lm_params_root, "dampingParams.exactHessianDiagonal", params.dampingParams.exactHessianDiagonal);
                safeRead(lm_params_root, "dampingParams.minDiagonal", params.dampingParams.minDiagonal);
                safeRead(lm_params_root, "dampingParams.maxDiagonal", params.dampingParams.maxDiagonal);

                // Generic NonlinearOptimizerParams
                safeRead(lm_params_root, "maxIterations", params.maxIterations);
                safeRead(lm_params_root, "relativeErrorTol", params.relativeErrorTol);
                safeRead(lm_params_root, "absoluteErrorTol", params.absoluteErrorTol);
                safeRead(lm_params_root, "errorTol", params.errorTol);
                std::string verbosity_str;
                if (safeRead(lm_params_root, "verbosity", verbosity_str))
                {
                    params.setVerbosity(verbosity_str);
                }
                std::string ordering;
                if (safeRead(lm_params_root, "orderingType", ordering))
                {
                    if (ordering == "COLAMD")
                        params.orderingType = gtsam::Ordering::COLAMD;
                    else if (ordering == "METIS")
                        params.orderingType = gtsam::Ordering::METIS;
                    else if (ordering == "NATURAL")
                        params.orderingType = gtsam::Ordering::NATURAL;
                    else
                        throw std::invalid_argument("estimator.lm_params.orderingType must be \"COLAMD\", \"METIS\" or \"NATURAL\", got \"" + ordering + "\".");
                }
            }
            else
            {
                warning() << "LM parameters (estimator.lm_params) not found in config. Using default values.";
            }
            return params;
        }

        /**
         * @brief Build one sensor per CSV file in `data_dir` and fill it with the rows
         *        within [start_time, end_time] (end_time <= 0: no upper bound).
         *
         *        File layout: "<label>,<type>" with type "vicon", "aurora", "gyro", "fbgs"
         *        or "accelerometer"; an optional "ground truth:,<1|true>" line; a header line; then rows
         *        "time,arclength,values...". Data rows use the legacy layout: a pose is the
         *        6-vector Log(T_ib) in [translational, rotational] order, and a strain a
         *        6-vector in the same order; both are reordered to GTSAM's here. A gyro row is
         *        a 3-vector angular velocity, an accelerometer row a 3-vector specific force [m/s^2]; an
         *        accelerometer sits at its first row's arclength.
         * @note "vicon" files are always loaded; "aurora" files are always given a
         *       sensor, which is left empty unless include_aurora is set.
         * @param data_dir Directory of the CSV files (`~` is expanded).
         * @param start_time,end_time Rows outside [start_time, end_time] [s] are skipped.
         * @param sensors,sensors_gt Receive the measurement and ground-truth sensors (appended to).
         * @param prior_model The prior every sensor is bound to.
         * @param R_aurora,R_gyro,R_strain,R_accelerometer Measurement covariance given to
         *        every row of that sensor type, in GTSAM [rotational, translational] order.
         * @param accelerometer_options Options of every accelerometer built.
         * @return false if globbing `data_dir` fails, including when it holds no CSV file.
         *         Files without a "<label>,<type>" first line, and rows with the wrong number of
         *         values for their sensor type, are skipped with a warning.
         */
        template <typename PriorT>
        bool loadData(
            std::string data_dir,
            double start_time,
            double end_time,
            std::vector<std::shared_ptr<Spacetime::Types::SensorBase>> &sensors,
            std::vector<std::shared_ptr<Spacetime::Types::SensorBase>> &sensors_gt,
            const std::shared_ptr<PriorT> &prior_model,
            bool include_aurora = false,
            bool include_gyro = false,
            bool include_strain = false,
            const Eigen::Matrix<double, 6, 6> &R_aurora = Eigen::Matrix<double, 6, 6>::Identity(),
            const Eigen::Matrix<double, 3, 3> &R_gyro = Eigen::Matrix<double, 3, 3>::Identity(),
            const Eigen::Matrix<double, 6, 6> &R_strain = Eigen::Matrix<double, 6, 6>::Identity(),
            bool include_accelerometer = false,
            const Eigen::Matrix<double, 3, 3> &R_accelerometer = Eigen::Matrix<double, 3, 3>::Identity(),
            const Spacetime::Sensors::AccelerometerOptions &accelerometer_options = Spacetime::Sensors::AccelerometerOptions()) const
        {
            using ValuesT = typename PriorT::ValuesType;
            std::string glob_path = data_dir + "/*.csv";
            info() << "Loading data from: " << glob_path;

            glob_t glob_result;
            memset(&glob_result, 0, sizeof(glob_result));

            int return_value = glob(glob_path.c_str(), GLOB_TILDE, NULL, &glob_result);

            if (return_value != 0)
            {
                if (return_value == GLOB_NOMATCH)
                    warning() << "No CSV files found in: " << data_dir;
                else
                    error() << "Listing the CSV files in " << data_dir << " failed (glob error " << return_value << ").";
                globfree(&glob_result);
                return false;
            }

            for (size_t i = 0; i < glob_result.gl_pathc; ++i)
            {
                std::string file_path = std::string(glob_result.gl_pathv[i]);

                CSVReader reader(file_path);
                info() << "Opening: " << file_path;
                std::vector<std::string> text;

                // Read sensor type
                reader.readLine(text);
                if (text.size() < 2)
                {
                    warning() << "Missing the \"<label>,<sensor type>\" first line. Skipping file: " << file_path;
                    continue;
                }
                std::string sensor_type = text[1];
                sensor_type.erase(std::remove(sensor_type.begin(), sensor_type.end(), '\r'), sensor_type.end());

                // Optional ground truth flag line ("ground truth:,<0|1>"); without it, this
                // second line is already the column header and the file is not ground truth.
                reader.readLine(text);
                bool is_ground_truth = false;
                if (!text.empty() && text[0].rfind("ground truth:", 0) == 0)
                {
                    std::string ground_truth_flag = text.size() > 1 ? text[1] : std::string();
                    ground_truth_flag.erase(std::remove(ground_truth_flag.begin(), ground_truth_flag.end(), '\r'), ground_truth_flag.end());
                    is_ground_truth = (ground_truth_flag == "true" || ground_truth_flag == "1");
                    reader.burnLine(); // header line
                }
                info() << "Sensor type: " << sensor_type << ", Ground truth: " << is_ground_truth;

                std::shared_ptr<Spacetime::Types::SensorBase> sensor;
                Eigen::MatrixXd R;
                if (sensor_type == std::string("vicon") || sensor_type == std::string("aurora"))
                {
                    sensor = std::make_shared<Spacetime::Sensors::PoseSensor<ValuesT>>(prior_model);
                    R = R_aurora;
                }
                else if (sensor_type == std::string("gyro"))
                {
                    if (!include_gyro)
                    {
                        info() << "Gyro sensors disabled. Skipping file.";
                        continue;
                    }
                    if constexpr (requires { typename Spacetime::Sensors::GyroSensor<ValuesT>; })
                    {
                        sensor = std::make_shared<Spacetime::Sensors::GyroSensor<ValuesT>>(prior_model);
                        R = R_gyro;
                    }
                    else
                    {
                        warning() << "The prior's node has no velocity, so gyro data cannot be used. Skipping file: " << file_path;
                        continue;
                    }
                }
                else if (sensor_type == std::string("fbgs"))
                {
                    if (!include_strain)
                    {
                        info() << "FBG sensors disabled. Skipping file.";
                        continue;
                    }
                    if constexpr (requires { typename Spacetime::Sensors::FBGSensor<ValuesT>; })
                    {
                        sensor = std::make_shared<Spacetime::Sensors::FBGSensor<ValuesT>>(prior_model);
                        R = R_strain;
                    }
                    else
                    {
                        warning() << "The prior's node has no strain, so FBG data cannot be used. Skipping file: " << file_path;
                        continue;
                    }
                }
                else if (sensor_type == std::string("accelerometer"))
                {
                    if (!include_accelerometer)
                    {
                        info() << "Accelerometer sensors disabled. Skipping file.";
                        continue;
                    }
                    if constexpr (!requires { typename Spacetime::Sensors::AccelerometerSensor<PriorT>; })
                    {
                        warning() << "The prior's node has no velocity, so accelerometer data cannot be used. Skipping file: " << file_path;
                        continue;
                    }
                    R = R_accelerometer; // the sensor is built at the first row, at its arclength
                }
                else
                {
                    warning() << "Unknown sensor type: " << sensor_type << ". Skipping file.";
                    continue;
                }

                double timestep = -1.0;
                double arclength = -1.0;
                while (reader.read(timestep))
                {
                    if (timestep < start_time || (end_time > 0 && timestep > end_time))
                    {
                        // Skip any data outside the specified time range
                        reader.burnLine(); // Skip the rest of the line
                        continue;
                    }
                    reader.read(arclength);
                    std::vector<double> values;
                    reader.readLine(values);

                    auto measurement = std::make_shared<Spacetime::Types::MeasurementBase>();
                    measurement->arclength = arclength;
                    measurement->time = timestep; // Already in seconds
                    measurement->covariance = R;  // Set covariance based on sensor type

                    if (sensor_type == std::string("vicon") || (sensor_type == std::string("aurora") && include_aurora))
                    {
                        if (values.size() != 6)
                        {
                            warning() << "Skipping the pose row at t=" << timestep << " in " << file_path << ": expected 6 pose values, got " << values.size();
                            continue;
                        }
                        measurement->mask = Eigen::Matrix<int, 6, 1>::Ones(); // Assuming all components are valid
                        const Eigen::Matrix<double, 6, 1> value_ = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(values.data());
                        Eigen::Matrix<double, 6, 1> value;
                        value << value_.tail<3>(), value_.head<3>();
                        measurement->value = gtsam::traits<gtsam::Pose3>::Expmap(value).matrix();
                    }
                    else if (sensor_type == std::string("gyro") && include_gyro)
                    {
                        if (values.size() != 3)
                        {
                            warning() << "Skipping the gyro row at t=" << timestep << " in " << file_path << ": expected 3 angular velocity values, got " << values.size();
                            continue;
                        }
                        measurement->mask = Eigen::Matrix<int, 3, 1>::Ones(); // Assuming all components are valid
                        const Eigen::Matrix<double, 3, 1> value = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(values.data());
                        measurement->value = value;
                    }
                    else if (sensor_type == std::string("accelerometer"))
                    {
                        if (values.size() != 3)
                        {
                            warning() << "Skipping the accelerometer row at t=" << timestep << " in " << file_path << ": expected 3 acceleration values, got " << values.size();
                            continue;
                        }
                        if constexpr (requires { typename Spacetime::Sensors::AccelerometerSensor<PriorT>; })
                            if (!sensor)
                                sensor = std::make_shared<Spacetime::Sensors::AccelerometerSensor<PriorT>>(prior_model, arclength, accelerometer_options);
                        measurement->mask = Eigen::Matrix<int, 3, 1>::Ones();
                        measurement->value = Eigen::Vector3d(values[0], values[1], values[2]);
                    }
                    else if (sensor_type == std::string("fbgs"))
                    {
                        if (values.size() != 6)
                        {
                            warning() << "Skipping the FBG row at t=" << timestep << " in " << file_path << ": expected 6 strain values, got " << values.size();
                            continue;
                        }
                        // Data strains are [translational, rotational]; GTSAM's are [rotational, translational]
                        measurement->mask = Eigen::Matrix<int, 6, 1>::Ones(); // Assuming all components are valid
                        Eigen::Matrix<double, 6, 1> value;
                        value << values[3], values[4], values[5], values[0], values[1], values[2];
                        measurement->value = value;
                    }
                    else
                    {
                        continue;
                    }
                    sensor->addMeasurement(measurement);
                }

                if (!sensor)
                {
                    warning() << "No " << sensor_type << " rows within the time range. Skipping file: " << file_path;
                    continue;
                }

                if (is_ground_truth)
                {
                    sensors_gt.push_back(sensor);
                }
                else
                {
                    sensors.push_back(sensor);
                }

                info() << "Loaded " << sensor->size() << " measurements for sensor type: " << sensor_type;
            }

            globfree(&glob_result);

            // Enforce measurements are in chronological order
            // std::sort(measurements.begin(), measurements.end(), [](const std::shared_ptr<Spacetime::Factors::MeasurementFactor> &a, const std::shared_ptr<Spacetime::Factors::MeasurementFactor> &b)
            //           { return (a->getMeas().t < b->getMeas().t) || ((a->getMeas().t == b->getMeas().t) && (a->getMeas().s < b->getMeas().s)); });
            // std::sort(ground_truth.begin(), ground_truth.end(), [](const Spacetime::SensorMeasurement &a, const Spacetime::SensorMeasurement &b)
            //           { return (a.t < b.t) || ((a.t == b.t) && (a.s < b.s)); });

            // filterMeasurements(measurements, data_options.start_time, data_options.end_time);
            // filterMeasurements(ground_truth, data_options.start_time, data_options.end_time);

            // info() << "Loaded " << measurements.size() << " measurements between t=" << data_options.start_time << " and t=" << data_options.end_time;
            // info() << "Loaded " << ground_truth.size() << " ground truth measurements between t=" << data_options.start_time << " and t=" << data_options.end_time;

            // warning("Removed measurement validation step. This should probably be added back in.");

            return true;
        }
    };

    // void filterMeasurements(std::vector<std::shared_ptr<Spacetime::Factors::MeasurementFactor>> &measurements, double start_k, double end_k)
    // {
    //     if (measurements.size() == 0)
    //         return;

    //     if (end_k < 0)
    //         end_k = measurements.back()->getMeas().t;

    //     // Returns measurements such that measurement.k \in [start_k, end_k]
    //     auto start_it = std::find_if(measurements.begin(), measurements.end(), [start_k](const auto &m)
    //                                  { return m->getMeas().t >= start_k; });
    //     auto end_it = std::find_if(measurements.begin(), measurements.end(), [end_k](const auto &m)
    //                                { return m->getMeas().t > end_k; });

    //     measurements.erase(end_it, measurements.end());
    //     measurements.erase(measurements.begin(), start_it);
    // }
    // void filterMeasurements(std::vector<Spacetime::SensorMeasurement> &measurements, double start_k, double end_k)
    // {
    //     if (measurements.size() == 0)
    //         return;

    //     if (end_k < 0)
    //         end_k = measurements.back().t;

    //     // Returns measurements such that measurement.k \in [start_k, end_k]
    //     auto start_it = std::find_if(measurements.begin(), measurements.end(), [start_k](const auto &m)
    //                                  { return m.t >= start_k; });
    //     auto end_it = std::find_if(measurements.begin(), measurements.end(), [end_k](const auto &m)
    //                                { return m.t > end_k; });

    //     measurements.erase(end_it, measurements.end());
    //     measurements.erase(measurements.begin(), start_it);
    // }

    // bool loadConfig(std::string filename, Spacetime::RobotTopology &topology, Spacetime::Hyperparameters &params, Spacetime::Options &options, DataLoaderOptions &data_options, Spacetime::VisualizerOptions &vis_options)
    // {
    //     std::ifstream config_file(filename, std::ifstream::binary);
    //     Json::CharReaderBuilder builder;
    //     JSONCPP_STRING errs;

    //     Json::Value root;

    //     if (!parseFromStream(builder, config_file, &root, &errs))
    //     {
    //         error() << errs;
    //         return false;
    //     }
    //     // setLogLevel(root["log"].asInt());
    //     info() << "Configuration file: " << filename;

    //     // Topology
    //     Json::Value topology_root;
    //     if (safeRead(root, "topology", topology_root))
    //     {
    //         // topology.N = root["topology"]["N"].asInt();
    //         safeRead(topology_root, "N", topology.N);
    //         safeRead(topology_root, "K_per_second", topology.K);

    //         // Number of interpolated states between estimation nodes per robot
    //         // M=1 results in no interpolation and the interpolation nodes will be equal to the estimation nodes
    //         // M=2 results in one additional interpolated node between each estimation node etc
    //         safeRead(topology_root, "Ms", topology.Ms);
    //         safeRead(topology_root, "Mt", topology.Mt);

    //         // Load the length and time of the robot
    //         safeRead(topology_root, "length", topology.L);
    //         safeRead(topology_root, "radius", topology.radius);
    //         safeRead(topology_root, "time", topology.T);

    //         // Define if we lock the pose of the robots' ends
    //         safeRead(topology_root, "lock_first_position", topology.lock_first_position);
    //         safeRead(topology_root, "lock_first_pose", topology.lock_first_pose);
    //         safeRead(topology_root, "lock_last_strain", topology.lock_last_strain);

    //         safeRead(topology_root, "fbg_theta_offset", topology.fbg_theta_offset);
    //         loadArrayData(topology_root, "spatial_node_positions", topology.spatial_node_positions);
    //         if (topology.spatial_node_positions.size() > 0)
    //         {
    //             info() << "Loaded spatial node positions. N will be ignored.";
    //             topology.N = topology.spatial_node_positions.size();
    //         }
    //         safeRead(topology_root, "time_nodes_on_measurements", topology.time_nodes_on_measurements);
    //         safeRead(topology_root, "use_1D_estimator", topology.use_1D_estimator);
    //     }

    //     // Measurement noise
    //     Json::Value weights_root;
    //     if (safeRead(root, "weights", weights_root))
    //     {
    //         double measurement_factor = 1.0;
    //         safeRead(weights_root, "measurement_factor", measurement_factor);

    //         Eigen::Matrix<double, 6, 1> R_pose;
    //         loadArrayData(weights_root, "R_pose", R_pose);
    //         data_options.weights.R_pose = measurement_factor * R_pose.asDiagonal();

    //         Eigen::Matrix<double, 3, 1> R_velocity;
    //         loadArrayData(weights_root, "R_velocity", R_velocity);
    //         data_options.weights.R_velocity = measurement_factor * R_velocity.asDiagonal();

    //         Eigen::Matrix<double, 3, 1> R_gyro;
    //         loadArrayData(weights_root, "R_gyro", R_gyro);
    //         data_options.weights.R_gyro = measurement_factor * R_gyro.asDiagonal();

    //         Eigen::Matrix<double, 1, 1> R_lidar;
    //         loadArrayData(weights_root, "R_lidar", R_lidar);
    //         data_options.weights.R_lidar = measurement_factor * R_lidar.asDiagonal();

    //         Eigen::Matrix<double, 6, 1> R_strain;
    //         loadArrayData(weights_root, "R_strain", R_strain);
    //         data_options.weights.R_strain = measurement_factor * R_strain.asDiagonal();

    //         // Prior noise
    //         double prior_factor = 1.0;
    //         safeRead(weights_root, "prior_factor", prior_factor);

    //         Eigen::Matrix<double, 6, 1> Q1, Q2, Q3, Q4;
    //         loadArrayData(weights_root, "Q1", Q1);
    //         params.Q1 = prior_factor * Q1.asDiagonal();
    //         loadArrayData(weights_root, "Q2", Q2);
    //         params.Q2 = prior_factor * Q2.asDiagonal();
    //         loadArrayData(weights_root, "Q3", Q3);
    //         params.Q3 = prior_factor * Q3.asDiagonal();
    //         if (!loadArrayData(weights_root, "Q4", Q4))
    //         {
    //             info() << "Q4 not found in config. Defaulting to Q3 values.";
    //             Q4 = Q3;
    //         }
    //         else
    //             params.Q4 = prior_factor * Q4.asDiagonal();

    //         Eigen::Matrix<double, 18, 1> P0;
    //         loadArrayData(weights_root, "P0", P0);
    //         params.P0 = prior_factor * P0.asDiagonal();
    //     }

    //     Json::Value optimizer_root;
    //     if (safeRead(root, "optimizer", optimizer_root))
    //     {
    //         options.solver = (optimizer_root["line_search"].asBool())
    //                              ? Spacetime::Options::Solver::NewtonLineSearch
    //                              : Spacetime::Options::Solver::Newton;

    //         safeRead(optimizer_root, "max_iterations", options.max_optimization_iterations);
    //         safeRead(optimizer_root, "kirchoff", options.kirchhoff_rods);
    //         safeRead(optimizer_root, "convergence_threshold", options.convergence_threshold);
    //         safeRead(optimizer_root, "LM_damping", options.LM_damping);
    //         safeRead(optimizer_root, "use_autodiff", options.use_autodiff);
    //         safeRead(optimizer_root, "marginalize", options.marginalize);
    //     }

    //     Json::Value options_root;
    //     if (safeRead(root, "options", options_root))
    //     {
    //         std::string estimator_type;
    //         safeRead(options_root, "estimator", estimator_type);
    //         if (estimator_type == "batch")
    //         {
    //             options.estimator_type = Spacetime::Options::EstimatorType::Batch;
    //         }
    //         else if (estimator_type == "filter")
    //         {
    //             options.estimator_type = Spacetime::Options::EstimatorType::Filter;
    //         }
    //         else if (estimator_type == "window")
    //         {
    //             options.estimator_type = Spacetime::Options::EstimatorType::Window;
    //         }
    //         else
    //         {
    //             throw std::runtime_error("Unknown estimator type: " + estimator_type);
    //         }
    //         safeRead(options_root, "interpolate_measurements", options.interpolate_measurements);
    //         safeRead(options_root, "compute_covariances", options.compute_covariances);
    //         safeRead(options_root, "extract_from_front", options.extract_from_front);
    //         safeRead(options_root, "dt", options.estimator_dt);
    //     }

    //     Json::Value icp_root;
    //     if (safeRead(root, "icp_options", icp_root))
    //     {
    //         safeRead(icp_root, "min_neighbors", options.icp_options.MIN_NEIGHBORS);
    //         safeRead(icp_root, "max_neighbors", options.icp_options.MAX_NEIGHBORS);
    //         safeRead(icp_root, "voxel_search_radius", options.icp_options.VOXEL_SEARCH_RADIUS);
    //         safeRead(icp_root, "power_planarity", options.icp_options.POWER_PLANARITY);
    //         safeRead(icp_root, "voxel_size", options.icp_options.VOXEL_SIZE);
    //         safeRead(icp_root, "max_dist_to_plane", options.icp_options.MAX_DIST_TO_PLANE);
    //         safeRead(icp_root, "voxel_lifetime", options.icp_options.VOXEL_LIFETIME);
    //         safeRead(icp_root, "min_distance_points", options.icp_options.MIN_DISTANCE_POINTS);
    //         safeRead(icp_root, "max_num_points_in_voxel", options.icp_options.MAX_NUM_POINTS_IN_VOXEL);
    //         safeRead(icp_root, "point2point", options.icp_options.point2point);
    //         safeRead(icp_root, "rejection_threshold", options.icp_options.rejection_threshold);
    //     }

    //     // Visualizer options
    //     Json::Value vis_root;
    //     if (safeRead(root, "visualizer", vis_root))
    //     {
    //         Json::Value robot_root;
    //         if (safeRead(vis_root, "robot", robot_root))
    //         {
    //             safeRead(robot_root, "show_interpolation", vis_options.robot.show_interpolation);
    //             safeRead(robot_root, "show_covariances", vis_options.robot.show_covariances);
    //             safeRead(robot_root, "show_frames", vis_options.robot.show_frames);
    //         }

    //         Json::Value measurements_root;
    //         if (safeRead(vis_root, "measurements", measurements_root))
    //         {
    //             Json::Value vicon_root;
    //             if (safeRead(measurements_root, "vicon", vicon_root))
    //             {
    //                 safeRead(vicon_root, "show", vis_options.vicon.show);
    //                 safeRead(vicon_root, "count", vis_options.vicon.count);
    //                 safeRead(vicon_root, "size", vis_options.vicon.size);
    //             }

    //             Json::Value lidar_root;
    //             if (safeRead(measurements_root, "lidar", lidar_root))
    //             {
    //                 safeRead(lidar_root, "show", vis_options.lidar.show);
    //                 safeRead(lidar_root, "count", vis_options.lidar.count);
    //                 safeRead(lidar_root, "size", vis_options.lidar.size);
    //             }

    //             Json::Value aurora_root;
    //             if (safeRead(measurements_root, "aurora", aurora_root))
    //             {
    //                 safeRead(aurora_root, "show", vis_options.aurora.show);
    //                 safeRead(aurora_root, "count", vis_options.aurora.count);
    //                 safeRead(aurora_root, "size", vis_options.aurora.size);
    //             }
    //         }

    //         safeRead(vis_root, "show_ground_plane", vis_options.show_ground_plane);
    //         safeRead(vis_root, "show_timeline", vis_options.show_timeline);

    //         Json::Value environment_root;
    //         if (safeRead(vis_root, "environment", environment_root))
    //         {
    //             safeRead(environment_root, "show", vis_options.environment.show);
    //             safeRead(environment_root, "scale", vis_options.environment.scale);
    //             safeRead(environment_root, "opacity", vis_options.environment.opacity);
    //             safeRead(environment_root, "path", vis_options.environment.path);
    //         }

    //         Json::Value camera_root;
    //         if (safeRead(vis_root, "cameras", camera_root))
    //         {
    //             for (unsigned int i = 0; i < camera_root.size(); i++)
    //             {
    //                 Spacetime::VisualizerOptions::CameraOptions cam_options;
    //                 loadArrayData(camera_root[i], "position", cam_options.position);
    //                 loadArrayData(camera_root[i], "look_at", cam_options.focal_point);
    //                 loadArrayData(camera_root[i], "up", cam_options.view_up);
    //                 vis_options.cameras.push_back(cam_options);
    //             }
    //         }
    //     }

    //     return true;
    // }
} // namespace Spacetime::Utilities
#endif // CONFIGLOADER_HPP
