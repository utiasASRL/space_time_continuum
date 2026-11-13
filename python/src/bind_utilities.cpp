/**
 * @file bind_utilities.cpp
 * @brief spacetime.utilities: configuration loading (the JSON configs of the offline demo),
 *        the window history and the ground-truth prediction export.
 */

#include <spacetime/estimator/utilities/ConfigLoader.hpp>
#include <spacetime/estimator/utilities/GroundTruthExport.hpp>
#include <spacetime/estimator/utilities/WindowHistory.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    void bindUtilities(py::module_ &m)
    {
        using Spacetime::Utilities::ConfigLoader;
        using Spacetime::Utilities::Extraction;
        using Spacetime::Utilities::WindowHistory;
        using SensorList = std::vector<std::shared_ptr<Spacetime::Types::SensorBase>>;

        py::enum_<Extraction>(m, "Extraction")
            .value("Final", Extraction::Final, "Every estimate from the final window.")
            .value("Back", Extraction::Back, "Each estimate from the window it last appeared in.");
        m.def("parseExtraction", &Spacetime::Utilities::parseExtraction, py::arg("name"));

        py::class_<ConfigLoader>(m, "ConfigLoader", "Loads estimator, prior, lock and data settings from JSON configuration files.")
            .def(py::init<>())
            .def("loadConfig", &ConfigLoader::loadConfig, py::arg("filename"), "Merges a JSON file into the configuration; False if it cannot be read.")
            .def("loadEstimatorOptions", &ConfigLoader::loadEstimatorOptions)
            .def("loadExtraction", &ConfigLoader::loadExtraction)
#ifdef SPACETIME_PYTHON_VISUALIZER // VisualizerOptions is bound by the visualizer module
            .def("loadVisualizerOptions", &ConfigLoader::loadVisualizerOptions,
                 "visualizer.VisualizerOptions from the \"visualizer\" section: \"cameras\" ({position, look_at, up}) and \"robot.radius\".")
#endif
            .def("loadPriorType", &ConfigLoader::loadPriorType)
            .def(
                "loadWNOAPrior",
                [](const ConfigLoader &self, double time_period)
                {
                    WnoaValues base_start_point;
                    WnoaPrior::Matrix3N Q0 = WnoaPrior::Matrix3N::Identity();
                    WnoaPrior::MatrixN Q1 = WnoaPrior::MatrixN::Identity(), Q2 = Q1, Q3 = Q1, Q4 = Q1;
                    self.loadWNOAPrior(base_start_point, time_period, Q0, Q1, Q2, Q3, Q4);
                    return py::make_tuple(base_start_point, time_period, Q0, Q1, Q2, Q3, Q4);
                },
                py::arg("time_period") = 0.1,
                "(base_start_point, time_period, Q0, Q1, Q2, Q3, Q4): wnoa.Prior's arguments and the node spacing in time (prior.time_period), "
                "defaulting to `time_period` and identities.")
            .def("loadWNOAPriorOptions", &ConfigLoader::loadWNOAPriorOptions)
            .def(
                "loadWNOVPrior",
                [](const ConfigLoader &self, double time_period)
                {
                    Pose T0;
                    Vector6 epsilon0 = Vector6::Zero();
                    WnovPrior::MatrixN Q0 = WnovPrior::MatrixN::Identity(), Q1 = Q0, Q2 = Q0;
                    self.loadWNOVPrior<Pose>(T0, epsilon0, time_period, Q0, Q1, Q2);
                    return py::make_tuple(T0, epsilon0, time_period, Q0, Q1, Q2);
                },
                py::arg("time_period") = 0.1, "(T0, epsilon0, time_period, Q0, Q1, Q2): wnov.Prior's arguments and the node spacing in time.")
            .def("loadWNOVPriorOptions", &ConfigLoader::loadWNOVPriorOptions)
            .def(
                "loadWNOAQuasiStaticPrior",
                [](const ConfigLoader &self, double time_period)
                {
                    QuasiStaticValues base_start_point;
                    QuasiStaticPrior::Matrix2N Q0 = QuasiStaticPrior::Matrix2N::Identity();
                    QuasiStaticPrior::MatrixN Q2 = QuasiStaticPrior::MatrixN::Identity();
                    self.loadWNOAQuasiStaticPrior<Pose>(base_start_point, time_period, Q0, Q2);
                    return py::make_tuple(base_start_point, time_period, Q0, Q2);
                },
                py::arg("time_period") = 0.1,
                "(base_start_point, time_period, Q0, Q2): wnoa_quasi_static.Prior's arguments (from the prior.wnoa section) and the node spacing in time.")
            .def("loadWNOAQuasiStaticPriorOptions", &ConfigLoader::loadWNOAQuasiStaticPriorOptions)
            .def("loadLocks", &ConfigLoader::loadLocks<WnoaNode>, py::arg("state"))
            .def("loadLocks", &ConfigLoader::loadLocks<WnovNode>, py::arg("state"))
            .def("loadLocks", &ConfigLoader::loadLocks<QuasiStaticNode>, py::arg("state"))
            .def(
                "loadDemoTimeWindow",
                [](const ConfigLoader &self, double start_time, double end_time)
                {
                    self.loadDemoTimeWindow(start_time, end_time);
                    return py::make_tuple(start_time, end_time);
                },
                py::arg("start_time") = 0.0, py::arg("end_time") = -1.0, "(start_time, end_time), defaulting to the arguments.")
            .def(
                "loadDemoMetadata",
                [](const ConfigLoader &self)
                {
                    double length = 0.0;
                    unsigned int nodes = 0;
                    self.loadDemoMetadata(length, nodes);
                    return py::make_tuple(length, nodes);
                },
                "(length, number of nodes) of the demo robot.")
            .def(
                "loadDemoData",
                [](const ConfigLoader &self, const pStateful &prior)
                {
                    SensorList sensors, ground_truth;
                    if (auto wnoa = std::dynamic_pointer_cast<WnoaPrior>(prior))
                        self.loadDemoData(sensors, ground_truth, wnoa);
                    else if (auto wnov = std::dynamic_pointer_cast<WnovPrior>(prior))
                        self.loadDemoData(sensors, ground_truth, wnov);
                    else if (auto quasi_static = std::dynamic_pointer_cast<QuasiStaticPrior>(prior))
                        self.loadDemoData(sensors, ground_truth, quasi_static);
                    else
                        throw py::type_error("loadDemoData: the prior must be a wnoa.Prior, a wnov.Prior or a wnoa_quasi_static.Prior.");
                    return py::make_tuple(sensors, ground_truth);
                },
                py::arg("prior"), "(sensors, ground-truth sensors) of the configured demo data, on `prior`.");

        py::class_<WindowHistory>(m, "WindowHistory", "The posterior covariance of every estimator window, by time.")
            .def(py::init<>())
            .def("record", &WindowHistory::record, py::arg("estimator"), "Stores the estimator's current window: call after each estimate.")
            .def("posteriorAt", &WindowHistory::posteriorAt, py::arg("time"), py::return_value_policy::reference_internal);

        const auto exportPredictions = [](const pStateful &prior, const gtsam::Values &values,
                                          const Spacetime::PosteriorCovariance *posterior, const SensorList &ground_truth, const std::string &logs_dir)
        {
            if (auto wnoa = std::dynamic_pointer_cast<WnoaPrior>(prior))
                return Spacetime::Utilities::exportGroundTruthPredictions(*wnoa, values, posterior, ground_truth, logs_dir);
            if (auto wnov = std::dynamic_pointer_cast<WnovPrior>(prior))
                return Spacetime::Utilities::exportGroundTruthPredictions(*wnov, values, posterior, ground_truth, logs_dir);
            if (auto quasi_static = std::dynamic_pointer_cast<QuasiStaticPrior>(prior))
                return Spacetime::Utilities::exportGroundTruthPredictions(*quasi_static, values, posterior, ground_truth, logs_dir);
            throw py::type_error("exportGroundTruthPredictions: the prior must be a wnoa.Prior, a wnov.Prior or a wnoa_quasi_static.Prior.");
        };
        m.def("exportGroundTruthPredictions", exportPredictions, py::arg("prior"), py::arg("values"), py::arg("posterior"), py::arg("ground_truth_sensors"),
              py::arg("logs_dir"),
              "Writes the predictions at the ground-truth samples to logs_dir/error_pred.txt and error_pred_cov.txt; returns how many.");
    }
} // namespace SpacetimePython
