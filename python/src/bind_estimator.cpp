/**
 * @file bind_estimator.cpp
 * @brief spacetime.Estimator and its options, results and posterior covariance, and the graph's own
 *        factor types (LockedFactor, BetweenFactorVector6) that GTSAM's module does not register.
 */

#include <chrono>

#include <pybind11/chrono.h>

#include <gtsam/slam/BetweenFactor.h>

#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/LockedFactor.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/RuntimeStatistics.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>

#include "common.hpp"

namespace SpacetimePython
{
    namespace
    {
        /// Python takes shared_ptr<T>: the const of the library's results is dropped, their bindings expose no mutation.
        template <typename T>
        std::shared_ptr<T> mutableShared(const std::shared_ptr<const T> &pointer)
        {
            return std::const_pointer_cast<T>(pointer);
        }
    } // namespace

    void bindEstimator(py::module_ &m)
    {
        using Spacetime::Estimator;
        using Spacetime::EstimatorOptions;
        using Spacetime::EstimatorSnapshot;
        using Spacetime::PhaseTimings;
        using Spacetime::PosteriorCovariance;
        using Spacetime::RuntimePhase;
        using Spacetime::RuntimeStatistics;

        py::class_<EstimatorOptions> options(m, "EstimatorOptions");
        py::enum_<EstimatorOptions::Smoother>(options, "Smoother", "The fixed-lag smoother of sliding-window mode.")
            .value("Batch", EstimatorOptions::Smoother::Batch)
            .value("Incremental", EstimatorOptions::Smoother::Incremental);
        options.def(py::init<>())
            .def_readwrite("smootherLag", &EstimatorOptions::smootherLag, "The fixed-lag window [s]; Estimator.BATCH_LAG for batch.")
            .def_readwrite("smoother", &EstimatorOptions::smoother, "The sliding-window smoother; ignored in batch mode.")
            .def_readwrite("lm_params", &EstimatorOptions::lm_params, "LM parameters of the batch solve (ordered with METIS by default) and the batch smoother.")
            .def_readwrite("isam2_params", &EstimatorOptions::isam2_params, "iSAM2 parameters of the incremental smoother.")
            .def_readwrite("compute_covariance", &EstimatorOptions::compute_covariance)
            .def_readwrite("quasi_static", &EstimatorOptions::quasi_static,
                           "Estimate only at measurement times: estimate(time) adds a state at each new measurement time up to time, and none without new measurements.");

        py::classh<PosteriorCovariance>(m, "PosteriorCovariance", "The posterior covariance blocks of the estimated variables.")
            .def("has", py::overload_cast<gtsam::Key, gtsam::Key>(&PosteriorCovariance::has, py::const_), py::arg("a"), py::arg("b"))
            .def("has", py::overload_cast<gtsam::Key>(&PosteriorCovariance::has, py::const_), py::arg("key"))
            .def("covariance", py::overload_cast<gtsam::Key, gtsam::Key>(&PosteriorCovariance::covariance, py::const_), py::arg("a"), py::arg("b"))
            .def("covariance", py::overload_cast<gtsam::Key>(&PosteriorCovariance::covariance, py::const_), py::arg("key"))
            .def(
                "joint",
                [](const PosteriorCovariance &self, const gtsam::KeyVector &keys)
                {
                    Eigen::MatrixXd out;
                    const bool ok = self.joint(keys, out);
                    return py::make_tuple(ok, out);
                },
                py::arg("keys"), "(whether every block is known, the joint covariance of `keys`).")
            .def("size", &PosteriorCovariance::size)
            .def("empty", &PosteriorCovariance::empty);

        py::enum_<RuntimePhase>(m, "RuntimePhase", "The timed phases of an estimation step.")
            .value("GraphConstruction", RuntimePhase::GraphConstruction, "Building the new variables and factors.")
            .value("Solve", RuntimePhase::Solve, "The smoother update and estimate.")
            .value("Covariance", RuntimePhase::Covariance, "Posterior covariance extraction.");

        py::class_<PhaseTimings>(m, "PhaseTimings", "One duration [s] per RuntimePhase.")
            .def(py::init<>())
            .def_readwrite("graph_construction", &PhaseTimings::graph_construction)
            .def_readwrite("solve", &PhaseTimings::solve)
            .def_readwrite("covariance", &PhaseTimings::covariance);

        py::class_<RuntimeStatistics>(m, "RuntimeStatistics",
                                      "The last, total and count of each phase's durations. Each phase is counted independently: "
                                      "in batch mode the graph is built once per estimate() but solved once by finalize().")
            .def(py::init<>())
            .def("record", &RuntimeStatistics::record, py::arg("phase"), py::arg("seconds"), "Records one duration of `phase`; raises ValueError if negative.")
            .def("count", &RuntimeStatistics::count, py::arg("phase"))
            .def("last", &RuntimeStatistics::last, "The latest duration of each phase (0 if never recorded).")
            .def("average", &RuntimeStatistics::average, "The mean duration of each phase (0 if never recorded).")
            .def("total", &RuntimeStatistics::total, "The summed duration of each phase.");

        py::classh<EstimatorSnapshot>(m, "EstimatorSnapshot", "The estimator's values (and covariance) after one estimate.")
            .def_readonly("time", &EstimatorSnapshot::time)
            .def_readonly("values", &EstimatorSnapshot::values)
            .def_property_readonly("covariance", [](const EstimatorSnapshot &self) { return mutableShared(self.covariance); })
            .def_readonly("runtime", &EstimatorSnapshot::runtime, "The phase timings up to and including this estimate.")
            .def_readonly("covariance_enabled", &EstimatorSnapshot::covariance_enabled, "Whether the estimator extracts covariances.")
            .def_readonly("batch", &EstimatorSnapshot::batch, "Whether the estimator runs in batch mode.");

        py::class_<Estimator::BatchTimings>(m, "BatchTimings")
            .def_readonly("solve", &Estimator::BatchTimings::solve)
            .def_readonly("covariance", &Estimator::BatchTimings::covariance);

        // The graph's factor types GTSAM's module lacks; without them getFactors() returns them as opaque gtsam.NonlinearFactor.
        using Spacetime::LockedFactor;
        py::class_<LockedFactor, gtsam::NonlinearFactor, std::shared_ptr<LockedFactor>>(
            m, "LockedFactor", "A factor with its locked keys held at constants: keys() are the free ones only.")
            .def("inner", &LockedFactor::inner, "The wrapped factor, on every key.")
            .def("constants", &LockedFactor::constants, py::return_value_policy::copy, "The locked keys' values.");

        using BetweenFactorVector6 = gtsam::BetweenFactor<Vector6>;
        py::class_<BetweenFactorVector6, gtsam::NoiseModelFactor, std::shared_ptr<BetweenFactorVector6>>(
            m, "BetweenFactorVector6", "The 6-vector random walk between two nodes (e.g. WNOA strain in time, velocity in arclength).")
            .def("measured", &BetweenFactorVector6::measured, py::return_value_policy::copy);

        py::class_<Estimator::BatchResult>(m, "BatchResult")
            .def_readonly("iterations", &Estimator::BatchResult::iterations)
            .def_readonly("initial_error", &Estimator::BatchResult::initial_error)
            .def_readonly("final_error", &Estimator::BatchResult::final_error);

        py::class_<Estimator, std::shared_ptr<Estimator>>(
            m, "Estimator", "The fixed-lag (or batch) smoother over every model, prior and sensor added to it.")
            .def(py::init<const EstimatorOptions &>(), py::arg("options"))
            .def(py::init<double, const gtsam::LevenbergMarquardtParams &>(), py::arg("smootherLag") = 0.0,
                 py::arg("params") = EstimatorOptions::defaultLMParams())
            .def_readonly_static("BATCH_LAG", &Estimator::BATCH_LAG)
            .def("addModel", &Estimator::addModel, py::arg("model"))
            .def("addSensor", &Estimator::addSensor, py::arg("sensor"))
            .def("addPrior", &Estimator::addModel, py::arg("prior"))
            .def("addRobot", &Estimator::addRobot, py::arg("robot"), "Adds every model of `robot`, in its order.")
            .def("setCovarianceKeyGroupsFn", &Estimator::setCovarianceKeyGroupsFn, py::arg("fn"))
            .def("getMeasurementQueues", &Estimator::getMeasurementQueues)
            .def("isBatch", &Estimator::isBatch)
            .def("estimate", &Estimator::estimate, py::arg("time"), py::call_guard<py::gil_scoped_release>())
            .def("finalize", &Estimator::finalize, py::call_guard<py::gil_scoped_release>())
            .def("getBatchTimings", &Estimator::getBatchTimings, py::return_value_policy::copy)
            .def("getBatchResult", &Estimator::getBatchResult, py::return_value_policy::copy)
            .def("getRuntimeStatistics", &Estimator::getRuntimeStatistics, py::return_value_policy::copy,
                 "Timings of every phase of every estimate() (and, in batch mode, finalize()) so far (a copy).")
            .def("options", &Estimator::options, py::return_value_policy::copy)
            .def("smoother", py::overload_cast<>(&Estimator::smoother), py::return_value_policy::reference_internal,
                 "The underlying gtsam fixed-lag smoother (in batch mode, it only stores the solution).")
            .def("getFactors", &Estimator::getFactors, py::return_value_policy::copy)
            .def("getLinearizationPoint", &Estimator::getLinearizationPoint, py::return_value_policy::copy)
            .def("timestamps", &Estimator::timestamps, "{key: time} of every variable in the smoother.")
            .def("getCurrentTimestamp", &Estimator::getCurrentTimestamp)
            .def("smootherLag", &Estimator::smootherLag)
            .def("getLastEstimateTime", &Estimator::getLastEstimateTime)
            .def("getCurrentValues", &Estimator::getCurrentValues)
            .def("getCurrentCovariance", [](const Estimator &self) { return mutableShared(self.getCurrentCovariance()); })
            .def("covariance", py::overload_cast<gtsam::Key, gtsam::Key>(&Estimator::covariance, py::const_), py::arg("a"), py::arg("b"))
            .def("covariance", py::overload_cast<gtsam::Key>(&Estimator::covariance, py::const_), py::arg("key"))
            .def("getSnapshot", [](const Estimator &self) { return mutableShared(self.getSnapshot()); })
            .def(
                "waitForSnapshotNewerThan",
                [](const Estimator &self, double time, std::chrono::duration<double> timeout)
                {
                    std::shared_ptr<const EstimatorSnapshot> snapshot;
                    {
                        py::gil_scoped_release release;
                        snapshot = self.waitForSnapshotNewerThan(time, std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
                    }
                    return mutableShared(snapshot);
                },
                py::arg("time"), py::arg("timeout"), "The first snapshot newer than `time`, waiting up to `timeout` [s]; None on timeout.")
            .def("print", &Estimator::print, py::arg("s") = "Estimator State: ");
    }
} // namespace SpacetimePython
