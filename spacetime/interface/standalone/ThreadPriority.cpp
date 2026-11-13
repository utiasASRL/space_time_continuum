#include <sys/resource.h>

#include "ThreadPriority.hpp"

namespace Spacetime::Standalone
{
    void lowerCurrentThreadPriority(int nice_delta)
    {
        setpriority(PRIO_PROCESS, 0, getpriority(PRIO_PROCESS, 0) + nice_delta);
    }
} // namespace Spacetime::Standalone
