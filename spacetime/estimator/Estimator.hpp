/**
 * @file Estimator.hpp
 * @brief State estimation with a fixed-lag smoother (batch or incremental), or a single batch solve.
 * @author Spencer Teetaert
 */

#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/constrained/NonlinearEqualityConstraint.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/BayesTreeCovariance.hpp>
#include <spacetime/estimator/EstimatorSnapshot.hpp>
#include <spacetime/estimator/LockedFactor.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/KeyManager.hpp>
#include <spacetime/estimator/utilities/logger.hpp>

namespace Spacetime
{
    /** @brief Runtime options for Estimator, loaded from the config's "estimator" section (see Utilities::ConfigLoader::loadEstimatorOptions). */
    struct EstimatorOptions
    {
        /// The fixed-lag smoother of sliding-window mode (see smoother).
        enum class Smoother
        {
            Batch,       ///< gtsam::BatchFixedLagSmoother: re-solves the whole window with LM (lm_params) at every update.
            Incremental, ///< gtsam::IncrementalFixedLagSmoother: updates the window with iSAM2 (isam2_params).
        };

        /// Time the fixed-lag smoother keeps past states for estimation. A value of
        /// Estimator::BATCH_LAG (-1) selects batch mode: nothing is marginalized, and the
        /// whole problem is solved once by Estimator::finalize() (offline use only).
        double smootherLag = 0.0;
        /// The sliding-window smoother; ignored in batch mode, which always solves with lm_params.
        Smoother smoother = Smoother::Batch;
        /// Levenberg-Marquardt parameters of the batch solve and of the batch fixed-lag smoother.
        /// Only the batch solve honors orderingType (METIS by default: nested dissection keeps
        /// large problems sparse); the batch fixed-lag smoother always orders with constrained COLAMD.
        gtsam::LevenbergMarquardtParams lm_params = defaultLMParams();
        /// Parameters of the incremental fixed-lag smoother (Smoother::Incremental).
        gtsam::ISAM2Params isam2_params;
        /// Extract posterior covariances after every solve (see Estimator::getCurrentCovariance).
        bool compute_covariance = false;
        /// Quasi-static estimation: Estimator::estimate(time) adds a state only at each new
        /// measurement time up to `time` (every measurement sharing that time joins it), and
        /// none when no measurement arrived, instead of one at `time` itself.
        bool quasi_static = false;

        /// gtsam's defaults, ordered with METIS.
        static gtsam::LevenbergMarquardtParams defaultLMParams()
        {
            gtsam::LevenbergMarquardtParams params;
            params.orderingType = gtsam::Ordering::METIS;
            return params;
        }
    };

    /**
     * @brief State estimation over the models added to it, with a fixed-lag smoother
     *        (sliding-window mode) or a single Levenberg-Marquardt solve (batch mode).
     */
    class Estimator
    {
    public:
        using pSensorType = std::shared_ptr<Types::SensorBase>; ///< Shared pointer to a sensor model.
        using pModelType = Types::pModelType;                   ///< Shared pointer to any model.

        /// smootherLag value selecting batch mode (see EstimatorOptions::smootherLag).
        static constexpr double BATCH_LAG = -1.0;

        /// Wall-clock timings (seconds) of the batch solve, filled in by finalize() (see getRuntimeStatistics()).
        struct BatchTimings
        {
            double solve = 0.0;      ///< The single Levenberg-Marquardt solve.
            double covariance = 0.0; ///< Posterior covariance extraction (0 unless compute_covariance).
        };

        /// How the batch solve went, filled in by finalize().
        struct BatchResult
        {
            std::size_t iterations = 0; ///< Levenberg-Marquardt iterations.
            double initial_error = 0.0; ///< Graph error at the initial guess.
            double final_error = 0.0;   ///< Graph error at the solution.
        };

        /**
         * @param smootherLag Time the smoother keeps past states for estimation, or
         *        BATCH_LAG for batch mode.
         * @param params Levenberg-Marquardt parameters (see EstimatorOptions::lm_params).
         * @throws std::invalid_argument for a negative smootherLag other than BATCH_LAG.
         */
        Estimator(double smootherLag = 0.0, const gtsam::LevenbergMarquardtParams &params = EstimatorOptions::defaultLMParams())
            : Estimator(optionsWith(smootherLag, params)) {}

        /**
         * @brief Construct from runtime options (see EstimatorOptions).
         * @throws std::invalid_argument for a negative smootherLag other than BATCH_LAG.
         */
        explicit Estimator(const EstimatorOptions &options)
            : m_options(options), m_batch(options.smootherLag == BATCH_LAG), m_compute_covariance(options.compute_covariance)
        {
            if (options.smootherLag < 0.0 && !m_batch)
                throw std::invalid_argument("Estimator: smootherLag must be >= 0, or -1 for batch mode.");
            if (m_batch)
            {
                // Only stores the solution of finalize()'s own solve: an error tolerance
                // nothing can miss makes its update return before optimizing.
                gtsam::LevenbergMarquardtParams storage = options.lm_params;
                storage.errorTol = std::numeric_limits<double>::infinity();
                m_smoother = std::make_unique<gtsam::BatchFixedLagSmoother>(std::numeric_limits<double>::infinity(), storage);
            }
            else if (options.smoother == EstimatorOptions::Smoother::Incremental)
            {
                gtsam::ISAM2Params params = options.isam2_params;
                params.findUnusedFactorSlots = true; // marginalization leaves holes in the factor graph
                m_smoother = std::make_unique<gtsam::IncrementalFixedLagSmoother>(options.smootherLag, params);
            }
            else
                m_smoother = std::make_unique<gtsam::BatchFixedLagSmoother>(options.smootherLag, options.lm_params);
        }

        /// The options the estimator was built with.
        const EstimatorOptions &options() const { return m_options; }

        /**
         * @brief The underlying fixed-lag smoother. In batch mode it only stores the
         *        problem after finalize(), and never optimizes on its own.
         */
        gtsam::FixedLagSmoother &smoother() { return *m_smoother; }
        /// @copydoc smoother()
        const gtsam::FixedLagSmoother &smoother() const { return *m_smoother; }

        /// The smoother's nonlinear factors (in batch mode, empty until finalize()).
        const gtsam::NonlinearFactorGraph &getFactors() const
        {
            if (const auto *incremental = dynamic_cast<const gtsam::IncrementalFixedLagSmoother *>(m_smoother.get()))
                return incremental->getFactors();
            return static_cast<const gtsam::BatchFixedLagSmoother &>(*m_smoother).getFactors();
        }

        /// The smoother's linearization point: its variables, without the locked constants.
        const gtsam::Values &getLinearizationPoint() const
        {
            if (const auto *incremental = dynamic_cast<const gtsam::IncrementalFixedLagSmoother *>(m_smoother.get()))
                return incremental->getLinearizationPoint();
            return static_cast<const gtsam::BatchFixedLagSmoother &>(*m_smoother).getLinearizationPoint();
        }

        /// The time of every variable in the smoother.
        const gtsam::FixedLagSmoother::KeyTimestampMap &timestamps() const { return m_smoother->timestamps(); }

        /// The latest time in the smoother (-infinity before its first update).
        double getCurrentTimestamp() const { return m_current_timestamp; }

        /// Time the smoother keeps past states for (infinite in batch mode).
        double smootherLag() const { return m_smoother->smootherLag(); }

        /// Returns the key groups whose joint posterior covariance is extracted after each solve.
        using CovarianceKeyGroupsFn = std::function<std::vector<gtsam::KeyVector>(const gtsam::Values &values)>;

        /**
         * @brief Override which covariance blocks are extracted after each solve. By
         *        default, every model's Types::Model::covarianceKeyGroups is used.
         *        Passing an empty function restores the default.
         */
        void setCovarianceKeyGroupsFn(CovarianceKeyGroupsFn fn) { m_covariance_key_groups_fn = std::move(fn); }

        /**
         * @brief Adds a model to build factors from (see Types::Model). Models run in the
         *        order they were added.
         * @throws std::invalid_argument for a null model, or a stateful model (e.g. a prior,
         *         see Types::StatefulModel) whose state another stateful model added already
         *         extends.
         */
        void addModel(const pModelType &model) { addModels({model}); }

        /// Adds every model of `robot`, in its order (see addModel). Throws, adding none, if addModel would for any.
        void addRobot(const Types::Robot &robot) { addModels(robot.models()); }

        /// Same as addModel.
        void addSensor(pSensorType sensor) { addModel(sensor); }

        /**
         * @brief Same as addModel.
         * @tparam T A Types::StatefulModel type (e.g. a WNOA prior).
         */
        template <Types::StatefulType T>
        void addPrior(std::shared_ptr<T> prior_model) { addModel(prior_model); }

        /// The measurement queues of every model added (sensors and actuators), in the order added.
        const std::vector<Types::pMeasurementQueueType> &getMeasurementQueues() const { return mp_measurement_queues; }

        /// True if the estimator was built with smootherLag == BATCH_LAG.
        bool isBatch() const { return m_batch; }

        /**
         * @brief Adds the states and measurements up to @p time. In sliding-window mode this
         *        also runs the smoother update; in batch mode the new factors are only
         *        accumulated, and the problem is solved by finalize().
         *
         *        With EstimatorOptions::quasi_static, the states are instead added at each
         *        distinct time of the measurements queued up to @p time, oldest first (each
         *        with its own update), and nothing happens if none is queued.
         */
        void estimate(double time)
        {
            if (!m_options.quasi_static)
            {
                estimateAt(time);
                return;
            }

            // A measurement older than the last one estimated (e.g. delivered late) gets no
            // state of its own: it joins the next one's update.
            std::optional<double> last;
            for (std::optional<double> next = earliestQueuedTime(); next && *next <= time && (!last || *next > *last); next = earliestQueuedTime())
            {
                estimateAt(*next);
                last = next;
            }
        }

        /**
         * @brief Batch mode: solves the whole accumulated problem with a single
         *        Levenberg-Marquardt optimization (EstimatorOptions::lm_params, ordering
         *        included), then extracts covariances if enabled. Does nothing in
         *        sliding-window mode or when nothing new has been accumulated.
         */
        void finalize()
        {
            if (!m_batch || m_pending_values.empty())
                return;

            const auto solve_start = std::chrono::steady_clock::now();
            gtsam::LevenbergMarquardtOptimizer optimizer(m_pending_graph, m_pending_values, m_options.lm_params);
            const gtsam::Values solution = optimizer.optimize();
            m_batch_result = {optimizer.iterations(), m_pending_graph.error(m_pending_values), optimizer.error()};
            m_smoother->update(m_pending_graph, solution, m_pending_timestamps); // stores it (see the constructor)
            advanceCurrentTimestamp(m_pending_timestamps);
            m_estimated_values = m_smoother->calculateEstimate();
            m_current_values = m_estimated_values;
            insertLockedValues(m_current_values);
            const auto solve_end = std::chrono::steady_clock::now();

            m_pending_graph = gtsam::NonlinearFactorGraph();
            m_pending_values.clear();
            m_pending_timestamps.clear();

            m_marginals.reset();
            m_current_covariance = m_compute_covariance ? extractCovariance() : nullptr;
            const auto covariance_end = std::chrono::steady_clock::now();

            // Nothing is marginalized in batch mode, so every state's final estimate is this one.
            for (const auto &model : mp_models)
                model->onMarginalization(std::numeric_limits<double>::infinity(), m_current_values);

            m_runtime.record(RuntimePhase::Solve, seconds(solve_start, solve_end));
            if (m_compute_covariance)
                m_runtime.record(RuntimePhase::Covariance, seconds(solve_end, covariance_end));
            info() << "Batch solve: error " << m_batch_result.initial_error << " -> " << m_batch_result.final_error << " in "
                   << m_batch_result.iterations << " iterations (" << m_runtime.last().solve << " s).";
            publishSnapshot(getCurrentTimestamp());
        }

        /// Timings of the last finalize() call (zeros before it, or in sliding-window mode).
        BatchTimings getBatchTimings() const
        {
            if (!m_batch)
                return {};
            const PhaseTimings last = m_runtime.last();
            return {last.solve, last.covariance};
        }

        /// Timings of every phase of every estimate() (and, in batch mode, finalize()) so far.
        const RuntimeStatistics &getRuntimeStatistics() const { return m_runtime; }

        /// Iterations and errors of the last finalize() call (zeros before it, or in sliding-window mode).
        const BatchResult &getBatchResult() const { return m_batch_result; }

        /// Wall-clock time (seconds) of the last estimate() call, including its solve and
        /// covariance extraction (0 before the first call). In batch mode, graph construction only.
        /// With quasi_static, the time of the last per-measurement-time update of that call.
        double getLastEstimateTime() const { return m_last_estimate_seconds; }

        /// The latest estimate, including the locked constants (a copy). In batch mode before
        /// finalize(), the accumulated initial guesses.
        gtsam::Values getCurrentValues() const { return m_current_values; }

        /// Covariance blocks extracted by the latest solve (estimate(), or finalize() in batch mode); nullptr unless compute_covariance is set (or if extraction failed).
        std::shared_ptr<const PosteriorCovariance> getCurrentCovariance() const { return m_current_covariance; }

        /**
         * @brief Posterior cross-covariance Sigma_ab between any two variables still in
         *        the smoother's window, computed on demand from the current solution
         *        (whether or not compute_covariance is set). The factorization is cached
         *        until the next solve. Blocks involving a locked variable are zero.
         *        Call from the estimation thread.
         * @throws std::runtime_error before the first estimate(), or in batch mode before
         *         finalize().
         * @throws std::out_of_range if either key is not in the current window (e.g.
         *         already marginalized out).
         */
        gtsam::Matrix covariance(gtsam::Key a, gtsam::Key b) const
        {
            requireInWindow(a);
            requireInWindow(b);
            if (m_locked_values.exists(a) || m_locked_values.exists(b)) // known exactly
                return gtsam::Matrix::Zero(m_current_values.at(a).dim(), m_current_values.at(b).dim());
            if (a == b)
                return marginals().marginalCovariance(a);
            return marginals().jointMarginalCovariance({a, b}).at(a, b);
        }

        /// Posterior marginal covariance (variance) of a single variable. See covariance(a, b).
        gtsam::Matrix covariance(gtsam::Key a) const { return covariance(a, a); }

        /// The most recently published snapshot, or nullptr before the first estimate() call
        /// (in batch mode, before finalize(): estimate() publishes none). Safe to call from any thread.
        std::shared_ptr<const EstimatorSnapshot> getSnapshot() const
        {
            std::lock_guard<std::mutex> lock(m_snapshot_mutex);
            return m_snapshot;
        }

        /**
         * @brief Blocks until a snapshot stamped later than `time` is published, or `timeout`
         *        passes. Safe to call from any thread; lets a query thread wait for new results
         *        instead of polling getSnapshot().
         * @return That snapshot (the latest one), or nullptr on timeout.
         */
        std::shared_ptr<const EstimatorSnapshot> waitForSnapshotNewerThan(double time, std::chrono::milliseconds timeout) const
        {
            std::unique_lock<std::mutex> lock(m_snapshot_mutex);
            const bool newer = m_snapshot_published.wait_for(lock, timeout, [&]()
                                                             { return m_snapshot && m_snapshot->time > time; });
            return newer ? m_snapshot : nullptr;
        }

        /// Prints `s`, the number of models and the current values to stdout.
        void print(const std::string &s = std::string("Estimator State: "))
        {
            std::cout << s << std::endl;
            std::cout << "Number of models: " << mp_models.size() << std::endl;
            std::cout << "Current Values: " << std::endl;
            m_current_values.print("Current Values:");
        }

    private:
        /// One estimate() step at `time`: every model extends the state and adds its factors
        /// up to `time`, then (in sliding-window mode) the smoother updates.
        void estimateAt(double time)
        {
            const auto estimate_start = std::chrono::steady_clock::now();
            gtsam::NonlinearFactorGraph new_graph;
            gtsam::Values new_values;
            std::vector<std::shared_ptr<gtsam::NonlinearFactor>> new_factors;

            // Phase 1 (see Types::Model): every model adds its new variables (e.g. a prior's
            // nodes, an actuator's inputs).
            for (const auto &model : mp_models)
            {
                auto factors = model->extendState(key_manager, time, m_current_values, new_values);
                new_factors.insert(new_factors.end(), factors.begin(), factors.end());
            }

            // Phase 2: every model's factors on the variables that now exist (each sensor and
            // actuator is bound to its prior model at construction).
            for (const auto &model : mp_models)
            {
                auto factors = model->getNewFactors(time, new_values);
                new_factors.insert(new_factors.end(), factors.begin(), factors.end());
                debug() << "Added " << factors.size() << " factors from a model at time: " << time;
            }

            // Locked variables (see Types::Model::lockedValues) are known constants, not
            // variables: they take their locked value, and the factors touching them are
            // wrapped to evaluate with it.
            auto timestamps = key_manager.drainPending();
            for (const auto &model : mp_models)
                for (const auto &[key, value] : model->lockedValues())
                    if (new_values.exists(key))
                    {
                        new_values.erase(key);
                        m_locked_timestamps[key] = timestamps.count(key) ? timestamps.at(key) : time;
                        timestamps.erase(key);
                        m_locked_values.insert(key, value);
                    }

            // Every model's factors must be pushed together, after all the loops
            // above have finished appending to new_factors (pushing earlier would silently
            // drop whichever factors were appended afterward).
            for (const auto &factor : new_factors)
            {
                if (!LockedFactor::touches(*factor, m_locked_values))
                    new_graph.push_back(factor);
                else if (auto locked = std::make_shared<LockedFactor>(factor, m_locked_values); !locked->keys().empty())
                    new_graph.push_back(locked);
                // else: every key is locked, so the factor is a constant and is dropped
            }

            if (m_batch)
            {
                // Defer the solve: the new states join the initial guess so that later
                // extensions can build on them (see Types::StatefulModel::extendState).
                m_pending_graph.push_back(new_graph);
                m_pending_values.insert(new_values);
                m_pending_timestamps.insert(timestamps.begin(), timestamps.end());
                m_current_values.insert(new_values);
                insertLockedValues(m_current_values);
                const auto construction_end = std::chrono::steady_clock::now();
                m_runtime.record(RuntimePhase::GraphConstruction, seconds(estimate_start, construction_end));
                m_last_estimate_seconds = seconds(estimate_start, construction_end);
                return;
            }

            // The update below marginalizes every state older than the window it leaves;
            // until then m_current_values still holds their final estimate.
            const double window_start = survivingWindowStart(timestamps);
            for (const auto &model : mp_models)
                model->onMarginalization(window_start, m_current_values);
            const auto construction_end = std::chrono::steady_clock::now();

            m_smoother->update(new_graph, new_values, timestamps); // Runs the actual estimation / marginalization step of the fixed-lag smoother
            advanceCurrentTimestamp(timestamps);
            pruneLockedValues();
            m_estimated_values = m_smoother->calculateEstimate();
            m_current_values = m_estimated_values;
            insertLockedValues(m_current_values);
            const auto solve_end = std::chrono::steady_clock::now();
            m_marginals.reset(); // the graph changed; rebuilt lazily by marginals()
            m_current_covariance = m_compute_covariance ? extractCovariance() : nullptr;
            const auto covariance_end = std::chrono::steady_clock::now();

            m_runtime.record(RuntimePhase::GraphConstruction, seconds(estimate_start, construction_end));
            m_runtime.record(RuntimePhase::Solve, seconds(construction_end, solve_end));
            if (m_compute_covariance)
                m_runtime.record(RuntimePhase::Covariance, seconds(solve_end, covariance_end));
            m_last_estimate_seconds = seconds(estimate_start, covariance_end);
            publishSnapshot(time);
        }

        /// The time of the oldest measurement queued in any model's queue, or nullopt if none is.
        std::optional<double> earliestQueuedTime() const
        {
            std::optional<double> earliest;
            for (const auto &queue : mp_measurement_queues)
                if (const std::optional<double> front = queue->frontTime(); front && (!earliest || *front < *earliest))
                    earliest = front;
            return earliest;
        }

        /// See addModel: validates every model of `models` before adding any.
        void addModels(const std::vector<pModelType> &models)
        {
            // The states extended by the stateful models added so far, then by those in `models`.
            std::vector<const Types::State *> extended_states;
            const auto stateOf = [](const pModelType &model) -> const Types::State *
            {
                const auto *stateful = dynamic_cast<const Types::StatefulModel<> *>(model.get());
                return stateful ? stateful->state().get() : nullptr;
            };
            for (const auto &model : mp_models)
                if (const Types::State *state = stateOf(model))
                    extended_states.push_back(state);
            for (const auto &model : models)
            {
                if (!model)
                    throw std::invalid_argument("Estimator::addModel: the model is null.");
                const Types::State *state = stateOf(model);
                if (!state)
                    continue;
                if (std::find(extended_states.begin(), extended_states.end(), state) != extended_states.end())
                    throw std::invalid_argument("Estimator::addModel: another stateful model added already extends this model's state; a state can only be extended by one.");
                extended_states.push_back(state);
            }

            for (const auto &model : models)
            {
                mp_models.push_back(model);
                if (auto queue = std::dynamic_pointer_cast<Types::MeasurementQueue>(model))
                    mp_measurement_queues.push_back(queue);
            }
        }

        /// `options` with `smootherLag` and `params` (see the (smootherLag, params) constructor).
        static EstimatorOptions optionsWith(double smootherLag, const gtsam::LevenbergMarquardtParams &params)
        {
            EstimatorOptions options;
            options.smootherLag = smootherLag;
            options.lm_params = params;
            return options;
        }

        EstimatorOptions m_options;                           // as constructed
        std::unique_ptr<gtsam::FixedLagSmoother> m_smoother;  // see the constructor
        double m_current_timestamp = -std::numeric_limits<double>::infinity(); // newest timestamp given to m_smoother (never marginalized)
        std::vector<pModelType> mp_models;                    // priors, sensors, actuators, ..., in the order added
        std::vector<Types::pMeasurementQueueType> mp_measurement_queues; // the measurement queues among mp_models
        gtsam::Values m_current_values;                       // current estimate plus locked constants (in batch mode, also the accumulated initial guesses); passed to the models
        gtsam::Values m_estimated_values;                     // the smoother's estimate: m_current_values without the locked constants
        gtsam::Values m_locked_values;                        // locked variables (constants), see Types::Model::lockedValues
        std::map<gtsam::Key, double> m_locked_timestamps;     // their node times, for pruneLockedValues()

        KeyManager key_manager;                               // Manager for generating unique keys

        bool m_batch = false;                                 // smootherLag == BATCH_LAG, see finalize()
        gtsam::NonlinearFactorGraph m_pending_graph;          // batch mode: factors awaiting finalize()
        gtsam::Values m_pending_values;                       // batch mode: initial guesses awaiting finalize()
        gtsam::FixedLagSmoother::KeyTimestampMap m_pending_timestamps;
        RuntimeStatistics m_runtime;
        BatchResult m_batch_result;
        double m_last_estimate_seconds = 0.0;

        bool m_compute_covariance = false;
        CovarianceKeyGroupsFn m_covariance_key_groups_fn;
        std::shared_ptr<const PosteriorCovariance> m_current_covariance;
        mutable std::optional<BayesTreeMarginals> m_marginals; // factorization of the current solution, see marginals()

        /**
         * @brief The factorization of the current solution, built on first use.
         * @param linked_groups Key groups whose pairs should all be recoverable from the Bayes
         *        tree (see BayesTreeCovariance::linkingFactors). Only used when the
         *        factorization is built; the covariances do not depend on it.
         */
        const BayesTreeMarginals &marginals(const std::vector<gtsam::KeyVector> &linked_groups = {}) const
        {
            if (!m_marginals)
            {
                // Cholesky cannot factor hard constraints (e.g. gtsam::NonlinearEquality), QR can.
                const gtsam::NonlinearFactorGraph &factors = getFactors();
                const bool constrained = std::any_of(factors.begin(), factors.end(), [](const auto &factor)
                                                     {
                    auto constraint = std::dynamic_pointer_cast<gtsam::NonlinearEqualityConstraint>(factor);
                    return constraint && constraint->isHardConstraint(); });
                gtsam::GaussianFactorGraph linear = *factors.linearize(m_estimated_values);
                linear.push_back(BayesTreeCovariance::linkingFactors(linked_groups, m_estimated_values));
                m_marginals.emplace(linear, m_estimated_values, constrained ? gtsam::Marginals::QR : gtsam::Marginals::CHOLESKY);
            }
            return *m_marginals;
        }

        /// @throws as documented on covariance(a, b), if `key` cannot be queried.
        void requireInWindow(gtsam::Key key) const
        {
            if (m_current_values.empty())
                throw std::runtime_error("Estimator::covariance called before the first estimate().");
            if (!m_pending_values.empty())
                throw std::runtime_error("Estimator::covariance called in batch mode before finalize().");
            if (!m_current_values.exists(key))
                throw std::out_of_range("Estimator::covariance: key is not in the smoother's current window.");
        }

        /// The earliest timestamp of a variable the next smoother update with `new_timestamps`
        /// keeps, by the rule it marginalizes with (timestamp before the newest timestamp
        /// minus the lag): every state before it is marginalized by that update, and none
        /// remains. Infinite if no variable is kept.
        double survivingWindowStart(const gtsam::FixedLagSmoother::KeyTimestampMap &new_timestamps) const
        {
            double newest = getCurrentTimestamp();
            for (const auto &[key, timestamp] : new_timestamps)
                newest = std::max(newest, timestamp);
            const double cutoff = newest - smootherLag();

            double start = std::numeric_limits<double>::infinity();
            for (const auto *map : {&timestamps(), &new_timestamps})
                for (const auto &[key, timestamp] : *map)
                    if (timestamp >= cutoff)
                        start = std::min(start, timestamp);
            return start;
        }

        /// Moves getCurrentTimestamp() to the newest of `new_timestamps`, as the smoother just did.
        void advanceCurrentTimestamp(const gtsam::FixedLagSmoother::KeyTimestampMap &new_timestamps)
        {
            for (const auto &[key, timestamp] : new_timestamps)
                m_current_timestamp = std::max(m_current_timestamp, timestamp);
        }

        /// Drop the locked constants the smoother has moved past, by the rule it uses to
        /// marginalize variables (timestamp before the current time minus the lag).
        void pruneLockedValues()
        {
            const double cutoff = getCurrentTimestamp() - smootherLag();
            for (auto it = m_locked_timestamps.begin(); it != m_locked_timestamps.end();)
            {
                if (it->second < cutoff)
                {
                    m_locked_values.erase(it->first);
                    it = m_locked_timestamps.erase(it);
                }
                else
                    ++it;
            }
        }

        /// Add the locked constants to `values` (those not already in it).
        void insertLockedValues(gtsam::Values &values) const
        {
            for (const auto &[key, value] : m_locked_values)
                if (!values.exists(key))
                    values.insert(key, value);
        }

        /// Duration from `start` to `end`, in seconds.
        static double seconds(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
        {
            return std::chrono::duration<double>(end - start).count();
        }

        /// Publishes m_current_values, m_current_covariance and m_runtime (plus the covariance and batch flags), stamped `time`, for getSnapshot().
        void publishSnapshot(double time)
        {
            // Publish a snapshot for any other thread (e.g. a visualizer) to read (see getSnapshot()).
            // Reuses the calculateEstimate() copy stored in m_current_values rather than deep-copying
            // m_current_values a second time; the lock only guards the pointer swap, so
            // the estimator thread never blocks on anything a visualizer thread is doing.
            auto snapshot = std::make_shared<const EstimatorSnapshot>(EstimatorSnapshot{time, m_current_values, m_current_covariance, m_runtime, m_compute_covariance, m_batch});
            {
                std::lock_guard<std::mutex> lock(m_snapshot_mutex);
                m_snapshot = std::move(snapshot);
            }
            m_snapshot_published.notify_all();
        }

        /// The covariance blocks selected by the key-group policy, or nullptr if the solution cannot be factorized.
        std::shared_ptr<const PosteriorCovariance> extractCovariance() const
        {
            std::vector<gtsam::KeyVector> groups;
            if (m_covariance_key_groups_fn)
                groups = m_covariance_key_groups_fn(m_current_values);
            else
                for (const auto &model : mp_models)
                {
                    auto model_groups = model->covarianceKeyGroups(m_current_values);
                    groups.insert(groups.end(), model_groups.begin(), model_groups.end());
                }

            try
            {
                const BayesTreeMarginals &factorized = marginals(groups);
                return PosteriorCovariance::compute(factorized, groups, m_locked_values, &factorized.bayesTree());
            }
            catch (const std::exception &e)
            {
                warning() << "Estimator: posterior covariance extraction failed (" << e.what() << "); no covariance is published for this estimate.";
                return nullptr;
            }
        }

        mutable std::mutex m_snapshot_mutex;
        mutable std::condition_variable m_snapshot_published; // see waitForSnapshotNewerThan()
        std::shared_ptr<const EstimatorSnapshot> m_snapshot;
    };
}

#endif // ESTIMATOR_HPP