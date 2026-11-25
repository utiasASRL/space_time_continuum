#ifndef MODEL_HPP
#define MODEL_HPP

#include <concepts>
#include <memory>
#include <vector>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <spacetime/estimator/utilities/KeyManager.hpp>

namespace Spacetime::Types
{
    /**
     * @brief Anything the estimator builds factors from: a prior (a Types::StatefulModel,
     *        which also stores the state it extends), a sensor (Types::SensorBase) or an
     *        actuator (Types::ActuatorBase). Each
     *        Estimator::estimate call runs two phases over every model, in the order the
     *        models were added:
     *          1. extendState: add the new variables (with their initial guesses in
     *             `new_values`) and the factors that come with them;
     *          2. getNewFactors: add the factors on variables that now exist, including
     *             those added in phase 1 by any model (e.g. a measurement interpolated
     *             between the nodes a prior just added).
     *        A model's extendState must not depend on another model's extendState of the
     *        same step: anything that needs another model's new variables belongs in
     *        getNewFactors. Every method does nothing by default, so a model overrides only
     *        what it takes part in.
     */
    struct Model
    {
        /// The factors a model returns from a phase.
        using Factors = std::vector<std::shared_ptr<gtsam::NonlinearFactor>>;

        virtual ~Model() = default;

        /**
         * @brief Phase 1: adds this model's new variables up to `time`, writing their
         *        initial guesses into `new_values` and registering each key with `km` (so
         *        the smoother can marginalize it), and returns the factors that come with them.
         * @param current_values The current estimate, holding every variable added so far.
         */
        virtual Factors extendState(KeyManager &km, double time, const gtsam::Values &current_values, gtsam::Values &new_values)
        {
            (void)km, (void)time, (void)current_values, (void)new_values;
            return {};
        }

        /// Phase 2: returns this model's new factors up to `time`, on variables that already exist.
        virtual Factors getNewFactors(double time, gtsam::Values &new_values)
        {
            (void)time, (void)new_values;
            return {};
        }

        /**
         * @brief Called by the estimator just before it marginalizes every state older than
         *        `cutoff`, with `values` still holding them: the last chance to read their
         *        final estimate (e.g. to build a map from the model's data). `cutoff` is
         *        the earliest state the window keeps, so a query before it needs a state
         *        about to be marginalized. In batch mode it is called once, after the
         *        solve, with an infinite cutoff. Does nothing by default.
         */
        virtual void onMarginalization(double cutoff, const gtsam::Values &values) { (void)cutoff, (void)values; }

        /**
         * @brief This model's variables that are held fixed: the estimator treats them as
         *        known constants, at these values, instead of estimating them. None by default.
         */
        virtual const gtsam::Values &lockedValues() const
        {
            static const gtsam::Values none;
            return none;
        }

        /**
         * @brief Called by the Estimator after phase 1 with every new variable's initial
         *        guess, so that variables locked without a value take theirs (see
         *        Types::State::resolveLocks). Does nothing by default.
         */
        virtual void resolveLocks(const gtsam::Values &new_values) { (void)new_values; }

        /**
         * @brief The key groups whose joint posterior covariance the estimator extracts
         *        after each solve (see PosteriorCovariance::compute), restricted to the
         *        variables still present in `values`. A model knows its own variables, so it
         *        decides which blocks are worth extracting. None by default.
         */
        virtual std::vector<gtsam::KeyVector> covarianceKeyGroups(const gtsam::Values &values) const
        {
            (void)values;
            return {};
        }
    };

    /// Types derived from Model. @tparam T The candidate model type.
    template <typename T>
    concept ModelType = std::derived_from<T, Model>;

    /// Shared pointer to a Model.
    using pModelType = std::shared_ptr<Model>;
} // namespace Spacetime::Types

#endif // MODEL_HPP
