/**
 * @file ThreadPriority.hpp
 * @brief Per-thread scheduling priority, on Linux, macOS and Windows, so the estimator thread
 *        stays the highest-priority spacetime thread: it raises itself where the OS allows
 *        that without privileges (prioritizeEstimatorThread), and every other spacetime
 *        thread (the default visualizer's GUI thread, the ROS2 session's node thread)
 *        lowers itself (lowerCurrentThreadPriority, ScopedLowerThreadPriority).
 *
 *        Each OS has its own mechanism, all per thread:
 *        - Linux: the nice value (setpriority(PRIO_PROCESS, 0, ...), which under NPTL is the
 *          calling thread's). An unprivileged thread can raise its nice value but not lower
 *          it again, and threads inherit their creator's: a thread created by a lowered
 *          thread starts lowered too.
 *        - macOS: the thread's quality-of-service class (pthread_set_qos_class_self_np), which
 *          a thread can set either way. (setpriority would renice the whole process.)
 *        - Windows: the thread priority (SetThreadPriority), which a thread can set either way
 *          within its process's priority class, and which new threads do not inherit.
 * @author Spencer Teetaert
 */

#ifndef SPACETIME_STANDALONE_THREADPRIORITY_HPP
#define SPACETIME_STANDALONE_THREADPRIORITY_HPP

#include <string>

namespace Spacetime::Interface::Standalone
{
    /**
     * @brief The calling thread's priority, comparable between threads of this process on the
     *        same OS: higher gets more CPU. Linux: minus the nice value. macOS: the QoS class's
     *        value (e.g. QOS_CLASS_UTILITY < QOS_CLASS_DEFAULT < QOS_CLASS_USER_INITIATED).
     *        Windows: GetThreadPriority (e.g. THREAD_PRIORITY_LOWEST = -2).
     */
    int currentThreadPriority();

    /**
     * @brief Lowers the calling thread's scheduling priority, and only that thread's. Needs no
     *        privileges. Linux: adds `nice_delta` to its nice value (capped at 19). macOS:
     *        QOS_CLASS_UTILITY. Windows: THREAD_PRIORITY_LOWEST if `nice_delta` >= 10, else
     *        THREAD_PRIORITY_BELOW_NORMAL.
     * @param nice_delta How much to lower it (a nice-value increment); 0 or less changes nothing.
     * @return Whether the priority changed as asked.
     */
    bool lowerCurrentThreadPriority(int nice_delta = 10);

    /**
     * @brief Lowers the calling thread's priority (as lowerCurrentThreadPriority) for its
     *        lifetime, and restores it on destruction where the OS allows: always on macOS and
     *        Windows; on Linux only with the privilege to lower a nice value (CAP_SYS_NICE or
     *        RLIMIT_NICE), so it usually stays lowered there (see restored()). Construct and
     *        destroy it on the same thread.
     */
    class ScopedLowerThreadPriority
    {
    public:
        explicit ScopedLowerThreadPriority(int nice_delta = 10);
        ~ScopedLowerThreadPriority();

        ScopedLowerThreadPriority(const ScopedLowerThreadPriority &) = delete;
        ScopedLowerThreadPriority &operator=(const ScopedLowerThreadPriority &) = delete;

        /// Whether the priority was lowered.
        bool lowered() const { return m_lowered; }
        /// Restores the priority now, if not done yet; returns whether it is back to what it was.
        bool restore();

    private:
        int m_saved = 0;        ///< The OS-specific value to restore.
        bool m_lowered = false; ///< Whether the constructor lowered it.
        bool m_done = false;    ///< Whether restore() has run.
        bool m_restored = false;
    };

    /**
     * @brief Called first thing by every estimator thread: raises it above the default
     *        priority where the OS allows that without privileges. macOS:
     *        QOS_CLASS_USER_INITIATED. Windows: THREAD_PRIORITY_ABOVE_NORMAL. Linux: nothing
     *        (raising needs privileges); it keeps its creator's nice value, and the other
     *        threads lower themselves below it.
     */
    void prioritizeEstimatorThread();

    /**
     * @brief Asks for real-time scheduling of the calling thread. Linux: SCHED_FIFO (needs
     *        CAP_SYS_NICE or an rtprio limit). macOS: QOS_CLASS_USER_INTERACTIVE, the highest
     *        class an unprivileged thread gets. Windows: THREAD_PRIORITY_TIME_CRITICAL.
     * @return An error message, empty on success.
     */
    std::string raiseCurrentThreadToRealTime();

    /**
     * @brief Windows: requests a 1 ms system timer resolution (timeBeginPeriod) for its
     *        lifetime, so the real-time clock's sleeps end on time rather than up to one
     *        ~15.6 ms default tick late. Elsewhere it does nothing (timers are already fine).
     */
    class ScopedTimerResolution
    {
    public:
        ScopedTimerResolution();
        ~ScopedTimerResolution();

        ScopedTimerResolution(const ScopedTimerResolution &) = delete;
        ScopedTimerResolution &operator=(const ScopedTimerResolution &) = delete;

    private:
        [[maybe_unused]] bool m_active = false; ///< Windows: whether timeBeginPeriod succeeded.
    };
} // namespace Spacetime::Interface::Standalone

#endif // SPACETIME_STANDALONE_THREADPRIORITY_HPP
