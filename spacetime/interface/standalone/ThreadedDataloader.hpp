/**
 * @file ThreadedDataloader.hpp
 * @brief A standalone-interface dataloader: an Interface::Dataloader with its own run loop,
 *        which Standalone::EstimatorRunner runs on a dedicated thread, and a blocking wait on
 *        its delivery watermark, for the estimator thread to synchronize with it offline.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_THREADEDDATALOADER_HPP
#define SPACETIME_STANDALONE_THREADEDDATALOADER_HPP

#include <condition_variable>
#include <mutex>
#include <stop_token>

#include <spacetime/interface/Dataloader.hpp>

namespace Spacetime::Standalone
{
    /// A dataloader run on its own thread (see the file comment).
    class ThreadedDataloader : public Interface::Dataloader
    {
    public:
        /**
         * @brief Delivers measurements until the source is exhausted or a stop is requested
         *        on `stop`, then returns. Runs on the dataloader's own thread. A source that
         *        is exhausted should call finish() before returning.
         */
        virtual void run(std::stop_token stop) = 0;

        /**
         * @brief Blocks until deliveredThrough(time), or a stop is requested on `stop`.
         * @return deliveredThrough(time); false only if stopped first.
         */
        bool waitUntilDeliveredThrough(double time, std::stop_token stop) const
        {
            std::unique_lock<std::mutex> lock(m_wait_mutex);
            return m_watermark_moved.wait(lock, stop, [&]()
                                          { return deliveredThrough(time); });
        }

    protected:
        void onWatermarkMoved() override
        {
            // Taking the wait mutex orders this notify after a waiter's predicate check, so no wake-up is lost.
            {
                std::lock_guard<std::mutex> lock(m_wait_mutex);
            }
            m_watermark_moved.notify_all();
        }

    private:
        mutable std::mutex m_wait_mutex;
        mutable std::condition_variable_any m_watermark_moved;
    };
} // namespace Spacetime::Standalone

#endif // SPACETIME_STANDALONE_THREADEDDATALOADER_HPP
