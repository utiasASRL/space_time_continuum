/**
 * @file RuntimeStatistics.hpp
 * @brief Wall-clock timings of the estimator's phases (see Estimator::getRuntimeStatistics).
 */

#ifndef RUNTIMESTATISTICS_HPP
#define RUNTIMESTATISTICS_HPP

#include <array>
#include <cstddef>
#include <stdexcept>

namespace Spacetime
{
    /// The timed phases of an estimation step (see Estimator::estimate / finalize).
    enum class RuntimePhase
    {
        GraphConstruction, ///< Building the new variables and factors.
        Solve,             ///< The smoother update and estimate.
        Covariance,        ///< Posterior covariance extraction.
    };

    /// One duration (seconds) per RuntimePhase.
    struct PhaseTimings
    {
        double graph_construction = 0.0;
        double solve = 0.0;
        double covariance = 0.0;
    };

    /**
     * @brief Accumulates the duration of each RuntimePhase: the last one recorded, their
     *        total and their count. Each phase is counted independently, e.g. in batch
     *        mode the graph is built once per estimate() but solved once by finalize().
     */
    class RuntimeStatistics
    {
    public:
        /**
         * @brief Records one duration (seconds) of `phase`.
         * @throws std::invalid_argument for a negative duration.
         */
        void record(RuntimePhase phase, double seconds)
        {
            if (seconds < 0.0)
                throw std::invalid_argument("RuntimeStatistics::record: the duration must be >= 0.");
            Accumulator &accumulator = m_phases[index(phase)];
            accumulator.last = seconds;
            accumulator.total += seconds;
            accumulator.count++;
        }

        /// Number of durations recorded for `phase`.
        std::size_t count(RuntimePhase phase) const { return m_phases[index(phase)].count; }

        /// The latest duration of each phase (0 if never recorded).
        PhaseTimings last() const
        {
            return collect([](const Accumulator &a)
                           { return a.last; });
        }

        /// The mean duration of each phase (0 if never recorded).
        PhaseTimings average() const
        {
            return collect([](const Accumulator &a)
                           { return a.count ? a.total / static_cast<double>(a.count) : 0.0; });
        }

        /// The summed duration of each phase.
        PhaseTimings total() const
        {
            return collect([](const Accumulator &a)
                           { return a.total; });
        }

    private:
        struct Accumulator
        {
            double last = 0.0;
            double total = 0.0;
            std::size_t count = 0;
        };

        static std::size_t index(RuntimePhase phase) { return static_cast<std::size_t>(phase); }

        template <typename F>
        PhaseTimings collect(F &&f) const
        {
            return {f(m_phases[index(RuntimePhase::GraphConstruction)]),
                    f(m_phases[index(RuntimePhase::Solve)]),
                    f(m_phases[index(RuntimePhase::Covariance)])};
        }

        std::array<Accumulator, 3> m_phases{};
    };
} // namespace Spacetime
#endif // RUNTIMESTATISTICS_HPP
