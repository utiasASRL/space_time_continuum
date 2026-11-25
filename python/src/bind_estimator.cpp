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

        py::class_<EstimatorOptions> options(m, "EstimatorOptions", "Runtime options of an Estimator (see utilities.ConfigLoader.loadEstimatorOptions).");
        py::enum_<EstimatorOptions::Smoother>(options, "Smoother", "The fixed-lag smoother of sliding-window mode.")
            .value("Batch", EstimatorOptions::Smoother::Batch, "gtsam.BatchFixedLagSmoother: re-solves the whole window with LM (lm_params) at every update.")
            .value("Incremental", EstimatorOptions::Smoother::Incremental, "gtsam.IncrementalFixedLagSmoother: updates the window with iSAM2 (isam2_params).");
        options.def(py::init<>())
            .def_readwrite("smootherLag", &EstimatorOptions::smootherLag, "The fixed-lag window [s]; Estimator.BATCH_LAG for batch.")
            .def_readwrite("smoother", &EstimatorOptions::smoother, "The sliding-window smoother; ignored in batch mode.")
            .def_readwrite("lm_params", &EstimatorOptions::lm_params, "LM parameters of the batch solve and of the batch smoother. Only the batch solve honors orderingType (METIS by default); "
                           "the batch smoother always orders with constrained COLAMD.")
            .def_readwrite("isam2_params", &EstimatorOptions::isam2_params, "iSAM2 parameters of the incremental smoother.")
            .def_readwrite("compute_covariance", &EstimatorOptions::compute_covariance,
                           "Extract posterior covariances (the models' covarianceKeyGroups) after every solve.")
            .def_readwrite("quasi_static", &EstimatorOptions::quasi_static,
                           "Estimate only at measurement times: estimate(time) adds a state at each new measurement time up to time, and none without new measurements.");

        py::classh<PosteriorCovariance>(m, "PosteriorCovariance", "The posterior covariance blocks of the estimated variables.")
            .def("has", py::overload_cast<gtsam::Key, gtsam::Key>(&PosteriorCovariance::has, py::const_), py::arg("a"), py::arg("b"),
                 "Whether the block Sigma_ab (equivalently Sigma_ba) was extracted.")
            .def("has", py::overload_cast<gtsam::Key>(&PosteriorCovariance::has, py::const_), py::arg("key"), "Whether the marginal Sigma_aa of `key` was extracted.")
            .def("covariance", py::overload_cast<gtsam::Key, gtsam::Key>(&PosteriorCovariance::covariance, py::const_), py::arg("a"), py::arg("b"),
                 "The cross-covariance Sigma_ab = E[da db^T]; raises IndexError if it was not extracted.")
            .def("covariance", py::overload_cast<gtsam::Key>(&PosteriorCovariance::covariance, py::const_), py::arg("key"),
                 "The marginal covariance of `key`; raises IndexError if it was not extracted.")
            .def(
                "joint",
                [](const PosteriorCovariance &self, const gtsam::KeyVector &keys)
                {
                    Eigen::MatrixXd out;
                    const bool ok = self.joint(keys, out);
                    return py::make_tuple(ok, out);
                },
                py::arg("keys"), "(whether every block is known, the joint covariance of `keys` in their order; empty if not).")
            .def("size", &PosteriorCovariance::size, "The number of stored (unordered) blocks.")
            .def("empty", &PosteriorCovariance::empty, "Whether no block is stored.");

        py::enum_<RuntimePhase>(m, "RuntimePhase", "The timed phases of an estimation step.")
            .value("GraphConstruction", RuntimePhase::GraphConstruction, "Building the new variables and factors.")
            .value("Solve", RuntimePhase::Solve, "The smoother update and estimate.")
            .value("Covariance", RuntimePhase::Covariance, "Posterior covariance extraction.");

        py::class_<PhaseTimings>(m, "PhaseTimings", "One duration [s] per RuntimePhase.")
            .def(py::init<>())
            .def_readwrite("graph_construction", &PhaseTimings::graph_construction, "RuntimePhase.GraphConstruction [s].")
            .def_readwrite("solve", &PhaseTimings::solve, "RuntimePhase.Solve [s].")
            .def_readwrite("covariance", &PhaseTimings::covariance, "RuntimePhase.Covariance [s].");

        py::class_<RuntimeStatistics>(m, "RuntimeStatistics",
                                      "The last, total and count of each phase's durations. Each phase is counted independently: "
                                      "in batch mode the graph is built once per estimate() but solved once by finalize().")
            .def(py::init<>())
            .def("record", &RuntimeStatistics::record, py::arg("phase"), py::arg("seconds"), "Records one duration of `phase`; raises ValueError if negative.")
            .def("count", &RuntimeStatistics::count, py::arg("phase"), "The number of durations recorded for `phase`.")
            .def("last", &RuntimeStatistics::last, "The latest duration of each phase (0 if never recorded).")
            .def("average", &RuntimeStatistics::average, "The mean duration of each phase (0 if never recorded).")
            .def("total", &RuntimeStatistics::total, "The summed duration of each phase.");

        py::classh<EstimatorSnapshot>(m, "EstimatorSnapshot", "The estimator's values (and covariance) after one estimate.")
            .def_readonly("time", &EstimatorSnapshot::time, "The estimation time [s] the snapshot was published for.")
            .def_readonly("values", &EstimatorSnapshot::values, "The estimated values, including the locked constants.")
            .def_property_readonly(
                "covariance", [](const EstimatorSnapshot &self) { return mutableShared(self.covariance); },
                "The PosteriorCovariance at this estimate; None unless EstimatorOptions.compute_covariance.")
            .def_readonly("runtime", &EstimatorSnapshot::runtime, "The phase timings up to and including this estimate.")
            .def_readonly("covariance_enabled", &EstimatorSnapshot::covariance_enabled, "Whether the estimator extracts covariances.")
            .def_readonly("batch", &EstimatorSnapshot::batch, "Whether the estimator runs in batch mode.");

        py::class_<Estimator::BatchTimings>(m, "BatchTimings", "Wall-clock timings [s] of the batch solve, filled in by Estimator.finalize().")
            .def_readonly("solve", &Estimator::BatchTimings::solve, "The single Levenberg-Marquardt solve.")
            .def_readonly("covariance", &Estimator::BatchTimings::covariance, "Posterior covariance extraction (0 unless compute_covariance).");

        // The graph's factor types GTSAM's module lacks; without them getFactors() returns them as opaque gtsam.NonlinearFactor.
        using Spacetime::LockedFactor;
        py::class_<LockedFactor, gtsam::NonlinearFactor, std::shared_ptr<LockedFactor>>(
            m, "LockedFactor", "A factor with its locked keys held at constants: keys() are the free ones only.")
            .def("inner", &LockedFactor::inner, "The wrapped factor, on every key.")
            .def("constants", &LockedFactor::constants, py::return_value_policy::copy, "The locked keys' values.");

        using BetweenFactorVector6 = gtsam::BetweenFactor<Vector6>;
        py::class_<BetweenFactorVector6, gtsam::NoiseModelFactor, std::shared_ptr<BetweenFactorVector6>>(
            m, "BetweenFactorVector6", "The 6-vector random walk between two nodes (e.g. WNOA strain in time, velocity in arclength).")
            .def("measured", &BetweenFactorVector6::measured, py::return_value_policy::copy, "The measured difference.");

        py::class_<Estimator::BatchResult>(m, "BatchResult", "How the batch solve went, filled in by Estimator.finalize().")
            .def_readonly("iterations", &Estimator::BatchResult::iterations, "Levenberg-Marquardt iterations.")
            .def_readonly("initial_error", &Estimator::BatchResult::initial_error, "Graph error at the initial guess.")
            .def_readonly("final_error", &Estimator::BatchResult::final_error, "Graph error at the solution.");

        py::class_<Estimator, std::shared_ptr<Estimator>>(
            m, "Estimator", "The fixed-lag (or batch) smoother over every model, prior and sensor added to it.")
            .def(py::init<const EstimatorOptions &>(), py::arg("options"), "Raises ValueError for a negative smootherLag other than BATCH_LAG.")
            .def(py::init<double>(), py::arg("smootherLag") = Estimator::BATCH_LAG,
                 "Default options, keeping `smootherLag` [s] of past states (BATCH_LAG, the default, selects batch mode). "
                 "Raises ValueError for a negative smootherLag other than BATCH_LAG.")
            .def_readonly_static("BATCH_LAG", &Estimator::BATCH_LAG, "The smootherLag selecting batch mode (-1).")
            .def("addModel", &Estimator::addModel, py::arg("model"),
                 "Adds a model; models run in the order added. Raises ValueError for None, or a stateful model whose state "
                 "another added stateful model extends already.")
            .def("addSensor", &Estimator::addSensor, py::arg("sensor"), "Same as addModel.")
            .def("addActuator", &Estimator::addActuator, py::arg("actuator"), "Same as addModel, for an actuator (e.g. actuators.KinematicTendon).")
            .def("addPrior", &Estimator::addModel, py::arg("prior"), "Same as addModel.")
            .def("addRobot", &Estimator::addRobot, py::arg("robot"), "Adds every model of `robot`, in its order; adds none if addModel would raise for any.")
            .def("setCovarianceKeyGroupsFn", &Estimator::setCovarianceKeyGroupsFn, py::arg("fn"),
                 "Overrides which covariance blocks are extracted after each solve: `fn(values)` returns the key groups. "
                 "None restores the default (every model's covarianceKeyGroups).")
            .def("getMeasurementQueues", &Estimator::getMeasurementQueues, "The measurement queues of every model added (sensors and actuators), in order.")
            .def("isBatch", &Estimator::isBatch, "Whether the estimator is in batch mode (smootherLag == BATCH_LAG).")
            .def("estimate", &Estimator::estimate, py::arg("time"), py::call_guard<py::gil_scoped_release>(),
                 "Adds the states and measurements up to `time` [s] and, in sliding-window mode, runs the smoother update "
                 "(batch mode only accumulates; see finalize). With quasi_static, adds a state per distinct queued measurement time instead.")
            .def("finalize", &Estimator::finalize, py::call_guard<py::gil_scoped_release>(),
                 "Batch mode: solves the accumulated problem with one LM optimization, then extracts covariances if enabled. "
                 "Does nothing in sliding-window mode or with nothing new accumulated.")
            .def("getBatchTimings", &Estimator::getBatchTimings, py::return_value_policy::copy,
                 "Timings of the last finalize() (zeros before it, or in sliding-window mode).")
            .def("getBatchResult", &Estimator::getBatchResult, py::return_value_policy::copy,
                 "Iterations and errors of the last finalize() (zeros before it, or in sliding-window mode).")
            .def("getRuntimeStatistics", &Estimator::getRuntimeStatistics, py::return_value_policy::copy,
                 "Timings of every phase of every estimate() (and, in batch mode, finalize()) so far (a copy).")
            .def("options", &Estimator::options, py::return_value_policy::copy, "The options the estimator was built with (a copy).")
            .def("smoother", py::overload_cast<>(&Estimator::smoother), py::return_value_policy::reference_internal,
                 "The underlying gtsam fixed-lag smoother (in batch mode, it only stores the solution).")
            .def("getFactors", &Estimator::getFactors, py::return_value_policy::copy,
                 "The smoother's nonlinear factors (a copy; in batch mode, empty until finalize()).")
            .def("getLinearizationPoint", &Estimator::getLinearizationPoint, py::return_value_policy::copy,
                 "The smoother's linearization point (a copy), without the locked constants.")
            .def("timestamps", &Estimator::timestamps, "{key: time} of every variable in the smoother.")
            .def("getCurrentTimestamp", &Estimator::getCurrentTimestamp, "The latest time in the smoother (-inf before its first update).")
            .def("smootherLag", &Estimator::smootherLag, "The time [s] the smoother keeps past states for (inf in batch mode).")
            .def("getLastEstimateTime", &Estimator::getLastEstimateTime,
                 "The wall-clock duration [s] of the last estimate() call (0 before the first), not an estimation time. "
                 "In batch mode, graph construction only.")
            .def("getCurrentValues", &Estimator::getCurrentValues,
                 "The latest estimate, including the locked constants (a copy); in batch mode before finalize(), the initial guesses.")
            .def(
                "getCurrentCovariance", [](const Estimator &self) { return mutableShared(self.getCurrentCovariance()); },
                "The covariance blocks extracted by the latest solve; None unless compute_covariance (or if extraction failed).")
            .def("covariance", py::overload_cast<gtsam::Key, gtsam::Key>(&Estimator::covariance, py::const_), py::arg("a"), py::arg("b"),
                 "Sigma_ab between two variables in the current window, computed on demand (zero blocks for locked variables). "
                 "Call from the estimating thread. Raises RuntimeError before the first solve, IndexError for a key not in the window.")
            .def("covariance", py::overload_cast<gtsam::Key>(&Estimator::covariance, py::const_), py::arg("key"),
                 "The marginal covariance of `key`; see covariance(a, b).")
            .def(
                "getSnapshot", [](const Estimator &self) { return mutableShared(self.getSnapshot()); },
                "The latest published EstimatorSnapshot, or None before the first (in batch mode, before finalize()). Safe from any thread.")
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
            .def("print", &Estimator::print, py::arg("s") = "Estimator State: ", "Prints `s`, the number of models and the current values to stdout.");
    }
} // namespace SpacetimePython
