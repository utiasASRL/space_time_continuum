/**
 * @file SnapshotHistory.hpp
 * @brief A thread-safe record of published estimator snapshots, so a query thread or an
 *        offline replay can look up the estimate as of any past time. The fixed-lag
 *        smoother itself only keeps its last smootherLag seconds, and
 *        Estimator::getSnapshot() only the latest snapshot.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_INTERFACE_SNAPSHOTHISTORY_HPP
#define SPACETIME_INTERFACE_SNAPSHOTHISTORY_HPP

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include <spacetime/estimator/EstimatorSnapshot.hpp>

namespace Spacetime::Interface
{
    /// Snapshots in time order, written by one thread (typically the estimator's tick callback) and read by any.
    class SnapshotHistory
    {
    public:
        using pSnapshot = std::shared_ptr<const EstimatorSnapshot>;

        /// Appends `snapshot` (ignored if null). Snapshots must be recorded in non-decreasing time order.
        void record(pSnapshot snapshot)
        {
            if (!snapshot)
                return;
            std::lock_guard<std::mutex> lock(m_mutex);
            m_snapshots.push_back(std::move(snapshot));
        }

        /// The latest snapshot stamped at or before `time`, or nullptr if there is none.
        pSnapshot at(double time) const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto it = std::upper_bound(m_snapshots.begin(), m_snapshots.end(), time, [](double t, const pSnapshot &snapshot)
                                             { return t < snapshot->time; });
            return (it == m_snapshots.begin()) ? nullptr : *(it - 1);
        }

        /// The most recently recorded snapshot, or nullptr if there is none.
        pSnapshot latest() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_snapshots.empty() ? nullptr : m_snapshots.back();
        }

        /// The number of recorded snapshots.
        std::size_t size() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_snapshots.size();
        }

    private:
        mutable std::mutex m_mutex;
        std::vector<pSnapshot> m_snapshots;
    };
} // namespace Spacetime::Interface

#endif // SPACETIME_INTERFACE_SNAPSHOTHISTORY_HPP
