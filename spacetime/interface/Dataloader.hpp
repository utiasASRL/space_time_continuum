/**
 * @file Dataloader.hpp
 * @brief The producer side of the interface layer: a dataloader fills the measurement
 *        queues of the estimator's models (see Types::MeasurementQueue) through
 *        addMeasurement. Each interface provides its own and decides how it runs, e.g.
 *        Standalone::FileDataloader replays recorded files on its own thread; a ROS2
 *        interface would deliver from each subscribed sensor's callback.
 *
 *        Each dataloader also publishes a delivery watermark: the time before which it has
 *        delivered everything it ever will. An interface that estimates offline checks
 *        every dataloader's watermark before each estimate, so results do not depend on
 *        how the producers are scheduled; a real-time one does not.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_DATALOADER_HPP
#define SPACETIME_INTERFACE_DATALOADER_HPP

#include <limits>
#include <mutex>
#include <optional>

namespace Spacetime::Interface
{
    /// A source of measurements (see the file comment). The watermark is safe to read from any thread.
    class Dataloader
    {
    public:
        virtual ~Dataloader() = default;

        /// The latest time this dataloader will deliver, if known (e.g. a recording), or nullopt for a live source.
        virtual std::optional<double> lastTime() const { return std::nullopt; }

        /// True once every measurement stamped at or before `time` has been delivered.
        bool deliveredThrough(double time) const
        {
            std::lock_guard<std::mutex> lock(m_watermark_mutex);
            return time < m_next_time;
        }

    protected:
        /**
         * @brief Declares that everything stamped before `next_time` has been delivered (the
         *        next measurement may be stamped exactly `next_time`). Never moves backwards;
         *        calls onWatermarkMoved() when it moves.
         */
        void setNextTime(double next_time)
        {
            {
                std::lock_guard<std::mutex> lock(m_watermark_mutex);
                if (!(next_time > m_next_time))
                    return;
                m_next_time = next_time;
            }
            onWatermarkMoved();
        }

        /// Declares that nothing more will be delivered, so estimation never has to wait on this dataloader again.
        void finish() { setNextTime(std::numeric_limits<double>::infinity()); }

        /// Called (outside the watermark's lock) whenever the watermark moves forward, e.g. for an interface to wake anything waiting on it.
        virtual void onWatermarkMoved() {}

    private:
        mutable std::mutex m_watermark_mutex;
        double m_next_time = -std::numeric_limits<double>::infinity(); // nothing delivered yet
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_DATALOADER_HPP
