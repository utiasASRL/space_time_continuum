/**
 * @file FileDataloader.hpp
 * @brief The standalone interface's dataloader: replays measurements already loaded from
 *        files into the models' queues (e.g. by Utilities::ConfigLoader::loadDemoData), as
 *        if they were arriving live. On construction it withholds the queues' contents;
 *        run() then hands each measurement back in time order: as fast as possible on an
 *        offline clock, or once the clock reaches its timestamp in real time.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_FILEDATALOADER_HPP
#define SPACETIME_STANDALONE_FILEDATALOADER_HPP

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <vector>

#include <spacetime/estimator/types/MeasurementQueue.hpp>
#include <spacetime/interface/standalone/Clock.hpp>
#include <spacetime/interface/standalone/ThreadedDataloader.hpp>

namespace Spacetime::Standalone
{
    /// Replays the withheld contents of a set of measurement queues (see the file comment).
    class FileDataloader : public ThreadedDataloader
    {
    public:
        /**
         * @brief Withholds every measurement in `queues`, leaving them empty. Build it before
         *        any measurement observer (e.g. a visualizer view) subscribes, so each
         *        observer sees every measurement exactly once, on delivery.
         * @param queues The models' queues, typically Estimator::getMeasurementQueues(), or a
         *        subset of them (several dataloaders can share the work).
         * @param clock Paces delivery when it is real time; ignored otherwise.
         */
        FileDataloader(const std::vector<Types::pMeasurementQueueType> &queues, std::shared_ptr<PacingClock> clock)
            : mp_clock(std::move(clock))
        {
            for (const auto &queue : queues)
                for (auto &measurement : queue->takeAll())
                    m_pending.push_back({queue, std::move(measurement)});
            // Each queue is already in time order; a stable sort keeps it, and orders ties by queue.
            std::stable_sort(m_pending.begin(), m_pending.end(), [](const Entry &a, const Entry &b)
                             { return a.measurement->time < b.measurement->time; });
        }

        /// Hands any measurement never delivered back to its queue, so no data is lost.
        ~FileDataloader() override { restoreUndelivered(); }

        /**
         * @brief Delivers the withheld measurements in time order, advancing the watermark past
         *        each timestamp once all measurements at it are delivered. Sleeps on the clock
         *        until each timestamp in real time. Calls finish() once everything is delivered;
         *        on a stop request, hands the rest back to their queues instead (without moving
         *        the watermark).
         */
        void run(std::stop_token stop) override
        {
            while (m_next < m_pending.size() && !stop.stop_requested())
            {
                const double time = m_pending[m_next].measurement->time;
                if (mp_clock->isRealTime())
                {
                    mp_clock->sleepUntil(time, stop);
                    if (stop.stop_requested())
                        break;
                }
                // Deliver every measurement stamped `time`, then advance the watermark past it.
                for (; m_next < m_pending.size() && m_pending[m_next].measurement->time == time; ++m_next)
                    m_pending[m_next].queue->addMeasurement(m_pending[m_next].measurement);
                setNextTime(m_next < m_pending.size() ? m_pending[m_next].measurement->time : std::numeric_limits<double>::infinity());
            }

            if (m_next == m_pending.size())
                finish();
            else
                restoreUndelivered(); // as the estimator stops, so it never sees these
        }

        /// The latest withheld timestamp, or -infinity if there is nothing to replay.
        std::optional<double> lastTime() const override
        {
            return m_pending.empty() ? -std::numeric_limits<double>::infinity() : m_pending.back().measurement->time;
        }

    private:
        /// A withheld measurement and the queue it belongs to.
        struct Entry
        {
            Types::pMeasurementQueueType queue;
            Types::MeasurementQueue::pMeasurementType measurement;
        };

        std::shared_ptr<PacingClock> mp_clock;
        std::vector<Entry> m_pending; // in time order
        std::size_t m_next = 0;       // index of the first undelivered entry

        /// Re-adds every undelivered measurement to its queue (no watermark change).
        void restoreUndelivered()
        {
            for (; m_next < m_pending.size(); ++m_next)
                m_pending[m_next].queue->addMeasurement(m_pending[m_next].measurement);
        }
    };
} // namespace Spacetime::Standalone

#endif // SPACETIME_STANDALONE_FILEDATALOADER_HPP
