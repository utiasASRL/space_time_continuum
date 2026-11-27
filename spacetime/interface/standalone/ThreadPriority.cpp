#include "ThreadPriority.hpp"

#include <algorithm>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#include <timeapi.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#else
#include <cerrno>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#endif

namespace Spacetime::Interface::Standalone
{
#if defined(_WIN32)
    // ---- Windows: thread priorities (THREAD_PRIORITY_*), settable either way, not inherited.

    int currentThreadPriority()
    {
        return GetThreadPriority(GetCurrentThread());
    }

    namespace
    {
        int loweredWindowsPriority(int nice_delta)
        {
            return nice_delta >= 10 ? THREAD_PRIORITY_LOWEST : THREAD_PRIORITY_BELOW_NORMAL;
        }
    } // namespace

    bool lowerCurrentThreadPriority(int nice_delta)
    {
        if (nice_delta <= 0)
            return false;
        const int target = loweredWindowsPriority(nice_delta);
        if (currentThreadPriority() <= target)
            return true; // already at least as low
        return SetThreadPriority(GetCurrentThread(), target) != 0;
    }

    ScopedLowerThreadPriority::ScopedLowerThreadPriority(int nice_delta) : m_saved(currentThreadPriority())
    {
        m_lowered = lowerCurrentThreadPriority(nice_delta);
    }

    bool ScopedLowerThreadPriority::restore()
    {
        if (!m_done)
        {
            m_done = true;
            m_restored = !m_lowered || SetThreadPriority(GetCurrentThread(), m_saved) != 0;
        }
        return m_restored;
    }

    void prioritizeEstimatorThread()
    {
        if (currentThreadPriority() < THREAD_PRIORITY_ABOVE_NORMAL)
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }

    std::string raiseCurrentThreadToRealTime()
    {
        if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) != 0)
            return {};
        return "SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL) failed (error " + std::to_string(GetLastError()) + ")";
    }

    ScopedTimerResolution::ScopedTimerResolution() : m_active(timeBeginPeriod(1) == TIMERR_NOERROR) {}

    ScopedTimerResolution::~ScopedTimerResolution()
    {
        if (m_active)
            timeEndPeriod(1);
    }

#elif defined(__APPLE__)
    // ---- macOS: quality-of-service classes, per thread, settable either way.

    namespace
    {
        /// The calling thread's QoS class.
        qos_class_t currentQos()
        {
            qos_class_t qos = QOS_CLASS_UNSPECIFIED;
            int relative_priority = 0;
            pthread_get_qos_class_np(pthread_self(), &qos, &relative_priority);
            return qos;
        }

        /// QOS_CLASS_UNSPECIFIED (no class asked for) is scheduled as QOS_CLASS_DEFAULT.
        int effectiveQos(qos_class_t qos)
        {
            return qos == QOS_CLASS_UNSPECIFIED ? static_cast<int>(QOS_CLASS_DEFAULT) : static_cast<int>(qos);
        }

        bool setQos(qos_class_t qos, int relative_priority = 0)
        {
            return pthread_set_qos_class_self_np(qos, relative_priority) == 0;
        }
    } // namespace

    int currentThreadPriority()
    {
        return effectiveQos(currentQos());
    }

    bool lowerCurrentThreadPriority(int nice_delta)
    {
        if (nice_delta <= 0)
            return false;
        if (currentThreadPriority() <= static_cast<int>(QOS_CLASS_UTILITY))
            return true; // already at least as low
        return setQos(QOS_CLASS_UTILITY);
    }

    ScopedLowerThreadPriority::ScopedLowerThreadPriority(int nice_delta) : m_saved(static_cast<int>(currentQos()))
    {
        m_lowered = lowerCurrentThreadPriority(nice_delta);
    }

    bool ScopedLowerThreadPriority::restore()
    {
        if (!m_done)
        {
            m_done = true;
            const auto saved = static_cast<qos_class_t>(m_saved);
            // A thread cannot go back to "no class": QOS_CLASS_DEFAULT is how that was scheduled.
            m_restored = !m_lowered || setQos(saved == QOS_CLASS_UNSPECIFIED ? QOS_CLASS_DEFAULT : saved);
        }
        return m_restored;
    }

    void prioritizeEstimatorThread()
    {
        if (currentThreadPriority() < static_cast<int>(QOS_CLASS_USER_INITIATED))
            setQos(QOS_CLASS_USER_INITIATED);
    }

    std::string raiseCurrentThreadToRealTime()
    {
        const int result = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        return result == 0 ? std::string() : std::string(std::strerror(result));
    }

    ScopedTimerResolution::ScopedTimerResolution() = default;
    ScopedTimerResolution::~ScopedTimerResolution() = default;

#else
    // ---- Linux (and other POSIX systems): nice values, per thread under NPTL.

    int currentThreadPriority()
    {
        return -getpriority(PRIO_PROCESS, 0);
    }

    bool lowerCurrentThreadPriority(int nice_delta)
    {
        if (nice_delta <= 0)
            return false;
        const int nice = getpriority(PRIO_PROCESS, 0);
        return setpriority(PRIO_PROCESS, 0, std::min(nice + nice_delta, 19)) == 0;
    }

    ScopedLowerThreadPriority::ScopedLowerThreadPriority(int nice_delta) : m_saved(getpriority(PRIO_PROCESS, 0))
    {
        m_lowered = lowerCurrentThreadPriority(nice_delta);
    }

    bool ScopedLowerThreadPriority::restore()
    {
        if (!m_done)
        {
            m_done = true;
            // Lowering a nice value needs CAP_SYS_NICE (or RLIMIT_NICE): unprivileged, this fails
            // (EACCES) and the thread stays lowered.
            m_restored = !m_lowered || setpriority(PRIO_PROCESS, 0, m_saved) == 0;
        }
        return m_restored;
    }

    void prioritizeEstimatorThread()
    {
        // Raising needs privileges: the estimator keeps its creator's nice value, and the other
        // spacetime threads lower themselves below it.
    }

    std::string raiseCurrentThreadToRealTime()
    {
        sched_param param{};
        param.sched_priority = std::max(1, sched_get_priority_min(SCHED_FIFO) + 10);
        const int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        return result == 0 ? std::string() : std::string(std::strerror(result));
    }

    ScopedTimerResolution::ScopedTimerResolution() = default;
    ScopedTimerResolution::~ScopedTimerResolution() = default;
#endif

    ScopedLowerThreadPriority::~ScopedLowerThreadPriority()
    {
        restore();
    }
} // namespace Spacetime::Interface::Standalone
