#ifndef SPACETIME_STANDALONE_THREADPRIORITY_HPP
#define SPACETIME_STANDALONE_THREADPRIORITY_HPP

namespace Spacetime::Standalone
{
    /**
     * @brief Lowers the calling thread's scheduling priority (raises its nice
     *        value) via setpriority(2). Under Linux/NPTL, PRIO_PROCESS with
     *        who=0 resolves to the calling thread's TID, so only this thread is
     *        affected. Needs no privileges, unlike raising a priority.
     *
     *        Called by the lower-priority threads (the visualizer, query threads),
     *        leaving the estimator thread relatively higher priority.
     * @param nice_delta Amount added to the thread's current nice value.
     */
    void lowerCurrentThreadPriority(int nice_delta = 10);
}

#endif // SPACETIME_STANDALONE_THREADPRIORITY_HPP
