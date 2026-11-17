/**
 * @file WindowHistory.hpp
 * @brief Records a sliding-window estimate as it moves, keeping every row at the last
 *        estimate it was part of, so the whole run can be exported once it ends
 *        (back-of-window extraction, as in the legacy SWEstimator).
 * @author Spencer Teetaert
 */

#ifndef WINDOW_HISTORY_HPP
#define WINDOW_HISTORY_HPP

#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtsam/nonlinear/Values.h>

#include <spacetime/Types.hpp>
#include <spacetime/estimator/Estimator.hpp>
#include <spacetime/estimator/PosteriorCovariance.hpp>
#include <spacetime/estimator/utilities/functions.hpp>

namespace Spacetime::Utilities
{
    /**
     * @brief Which estimate is exported once a run ends (config key "estimator.extraction").
     *        - Final: the estimator's last window only (Estimator::getCurrentValues). With
     *          a sliding window, every row already marginalized out is missing.
     *        - Back: every row as last estimated before leaving the window (see
     *          WindowHistory). In batch mode both are the same.
     */
    enum class Extraction
    {
        Final, ///< The last window only ("final").
        Back   ///< Every row as last estimated before leaving the window ("back").
    };

    /// Parse "final" or "back". @throws std::invalid_argument otherwise.
    inline Extraction parseExtraction(const std::string &name)
    {
        if (name == "final")
            return Extraction::Final;
        if (name == "back")
            return Extraction::Back;
        throw std::invalid_argument("Unknown estimator.extraction \"" + name + "\" (expected \"final\" or \"back\").");
    }

    /**
     * @class WindowHistory
     * @brief The estimate of each row at the last estimate() it was part of.
     *
     * After every Estimator::estimate(), record() overwrites the stored values with the
     * current window. A row's stored values are therefore the ones it had when it was
     * the oldest row of the window, just before the smoother marginalized it: the legacy
     * SWF's back-of-window output. Rows still in the window when the run ends keep the
     * final window's values.
     *
     * The window's posterior covariance is kept for each row in the same way, so a query
     * between rows k and k+1 uses the joint covariance of the estimate in which row k was
     * last solved. This is the legacy "store the joint covariances between neighbouring
     * times at lock time" rule. Rows k and k+1 must have been in the window together:
     * in filter mode (smootherLag 0) they never are, so no posterior is available between
     * rows (see GridInterpolator::interpolate).
     */
    class WindowHistory
    {
    public:
        /// Store the estimator's current window. Call after each estimate() (and finalize()).
        void record(const Estimator &estimator)
        {
            m_values.insert_or_assign(estimator.getCurrentValues());

            const std::shared_ptr<const PosteriorCovariance> posterior = estimator.getCurrentCovariance();
            for (const auto &[key, time] : estimator.timestamps())
                m_row_posterior.insert_or_assign(time, posterior);
        }

        /// Every row's values, as last recorded.
        const gtsam::Values &values() const { return m_values; }

        /**
         * @brief The posterior recorded with the latest row at or before `time`, i.e. the
         *        one to query between that row and the next.
         * @return nullptr before the first row, or if that estimate had no covariance.
         */
        const PosteriorCovariance *posteriorAt(double time) const
        {
            auto it = m_row_posterior.upper_bound(time + TOLERANCE);
            if (it == m_row_posterior.begin())
                return nullptr;
            return std::prev(it)->second.get();
        }

        /// Number of row times recorded.
        std::size_t numRows() const { return m_row_posterior.size(); }

    private:
        gtsam::Values m_values; ///< Union of every recorded window, newer records overwriting older ones.
        std::map<double, std::shared_ptr<const PosteriorCovariance>> m_row_posterior; // keyed by row time
    };
} // namespace Spacetime::Utilities

#endif // WINDOW_HISTORY_HPP
